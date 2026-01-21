import json
import os
import re
import copy
import multiprocessing
import queue
import psycopg2
from tqdm import tqdm

# Database Configuration
DB_CONFIG = {
    "host": "train.darelab.athenarc.gr",
    "port": "5555",
    "database": "fc4eosc",
    "user": "postgres",
    "password": "postgres"
}
DB_SCHEMA = "fc4eosc_subset"

def execute_sql_checks(ground_truth_sql, negative_check_sql, result_queue):
    """
    Executes validation checks in a separate process to handle timeouts.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Set the schema search path
        cursor.execute(f"SET search_path TO {DB_SCHEMA}, public;")

        # CHECK 1: Ground Truth Validity
        # The original SQL must actually return data.
        cursor.execute(ground_truth_sql)
        results = cursor.fetchall()

        if not results:
            result_queue.put(False)
            return

        # CHECK 2: Perturbation Validity
        # The SQL with the perturbed value should return NOTHING.
        cursor.execute(negative_check_sql)
        variation_results = cursor.fetchall()
        
        if not variation_results:
            result_queue.put(True) # Success
        else:
            result_queue.put(False) # Failure: Perturbed value exists in DB

    except (psycopg2.Error, Exception):
        result_queue.put(False)
    finally:
        if conn:
            conn.close()

def generate_benchmark_variations(benchmark_json_path, variations_json_path, output_json_path):
    """
    Generates variations for Postgres benchmark, avoids duplicates, and returns the count.
    """
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    with open(variations_json_path, 'r', encoding='utf-8') as f:
        variations_data = json.load(f)

    output_records = []
    seen_questions = set()  # Set to track unique questions and prevent duplicates

    if not variations_data:
        return 0

    # Identify the dynamic perturbation key (e.g., "abbreviation_acronym")
    sample_record = variations_data[0]
    standard_keys = {'table', 'column', 'original'}
    variation_keys = [k for k in sample_record.keys() if k not in standard_keys]
    
    if not variation_keys:
        print(f"Skipping {variations_json_path}: No variation key found.")
        return 0
    
    variation_type_key = variation_keys[0]
    keep_list = ['human_results_abbreviation_acronym.json','human_results_negated_antonym.json','human_results_typo_space_addition.json']
    for variation_record in tqdm(variations_data, desc=f"Processing {os.path.basename(variations_json_path)}", leave=False):
    
        table_name = variation_record['table']
        column_name = variation_record['column']
        original_value = variation_record['original']
        perturbed_value = variation_record[variation_type_key]

        if str(original_value).lower() == str(perturbed_value).lower():
            continue

        for benchmark_record in benchmark_data:
            
            # 1. Match Table/Column/Value in Benchmark Metadata
            match_found = False
            for value_info in benchmark_record.get('values', []):
                if (value_info['table'] == table_name and 
                    value_info['column'].lower() == column_name.lower() and
                    str(value_info['value']).lower() == str(original_value).lower()):
                    match_found = True
                    break
            
            if not match_found:
                continue

            # 2. Match Value in Question Text
            match_pattern = r'\b' + re.escape(str(original_value)) + r'\b'
            if not re.search(match_pattern, benchmark_record['question'], re.IGNORECASE):
                continue

            original_sql = benchmark_record['SQL']
            original_question = benchmark_record['question']

            # 3. Create Negative Check SQL (Inject Perturbation into SQL)
            negative_sql = re.sub(
                f"'{re.escape(str(original_value))}'",
                f"'{str(perturbed_value)}'",
                original_sql,
                flags=re.IGNORECASE
            )
            if negative_sql == original_sql:
                negative_sql = re.sub(
                    f'"{re.escape(str(original_value))}"',
                    f'"{str(perturbed_value)}"',
                    original_sql,
                    flags=re.IGNORECASE
                )
            if negative_sql == original_sql:
                negative_sql = re.sub(
                    r'=\s*' + match_pattern,
                    f'= {str(perturbed_value)}',
                    original_sql,
                    flags=re.IGNORECASE
                )
            
            if negative_sql == original_sql:
                continue

            # 4. Create New Question
            new_question = re.sub(
                match_pattern, 
                str(perturbed_value), 
                original_question, 
                count=1, 
                flags=re.IGNORECASE
            )

            # --- DEDUPLICATION CHECK ---
            # If we have already generated this exact question in this run, skip it.
            if new_question in seen_questions:
                continue

            # 5. Validate via Multiprocessing (Timeout protection)
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=execute_sql_checks,
                args=(original_sql, negative_sql, result_queue)
            )
            p.start()
            p.join(120) # 120 seconds timeout

            if p.is_alive():
                p.terminate()
                p.join()
                continue

            try:
                checks_passed = result_queue.get_nowait()
            except queue.Empty:
                checks_passed = False

            if checks_passed:
                new_record = copy.deepcopy(benchmark_record)
                new_record['original_question'] = original_question
                new_record['question'] = new_question
                new_record['SQL'] = original_sql 
                new_record['original_SQL'] = original_sql
                
                new_record['changes_information'] = {
                    'original_value': original_value,
                    variation_type_key: perturbed_value
                }
                
                output_records.append(new_record)
                seen_questions.add(new_question) # Mark this question as seen

    # Write output if records exist
    if output_records:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_records, f, indent=4)
            
    return len(output_records)

if __name__ == "__main__":
    # Configuration Paths
    benchmark_json_path = 'assets/scalability_experiments/faircore_benchmark-exact_match.json'
    perturbations_folder_path = 'assets/scalability_experiments/human_annotated_perturbations'
    output_folder_path = 'assets/scalability_experiments/benchmark_variations_postgres/'

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    files = [f for f in os.listdir(perturbations_folder_path) if f.endswith('.json')]
    
    # Statistics container
    generation_report = []
    total_variations = 0

    print(f"Starting generation for {len(files)} perturbation files...\n")

    for file in tqdm(files, desc="Overall Progress"):
        variations_json_path = os.path.join(perturbations_folder_path, file)
        output_json_path = os.path.join(output_folder_path, file)
        
        count = generate_benchmark_variations(
            benchmark_json_path, 
            variations_json_path, 
            output_json_path
        )
        
        generation_report.append((file, count))
        total_variations += count

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print(f"{'GENERATION REPORT':^60}")
    print("="*60)
    print(f"{'Input File':<45} | {'Variations':>10}")
    print("-" * 60)
    
    for filename, count in generation_report:
        print(f"{filename:<45} | {count:>10}")
        
    print("-" * 60)
    print(f"{'TOTAL GENERATED':<45} | {total_variations:>10}")
    print("="*60)
    print(f"Results saved to: {output_folder_path}")