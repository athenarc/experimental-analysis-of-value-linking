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
    "host": "",
    "port": "",
    "database": "",
    "user": "",
    "password": ""
}
DB_SCHEMA = ""

def check_sql_validity(sql, result_queue):
    """
    Executes the SQL. Returns True if it returns > 0 rows.
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"SET search_path TO {DB_SCHEMA}, public;")

        cursor.execute(sql)
        results = cursor.fetchall()

        if results:
            result_queue.put(True)
        else:
            # SQL ran but returned no data (The new entity might not exist in DB)
            result_queue.put(False)

    except Exception as e:
        # SQL Syntax error or DB connection error
        # print(f"\n[DB ERROR] {e}") 
        result_queue.put(False)
    finally:
        if conn: conn.close()

def safe_replace(text, old_value, new_value):
    """
    Case-insensitive replacement of old_value with new_value in text.
    """
    pattern = re.escape(str(old_value))
    new_text = re.sub(pattern, str(new_value), text, flags=re.IGNORECASE)
    return new_text

def generate_benchmark_variations(benchmark_json_path, variations_json_path, output_json_path):
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    with open(variations_json_path, 'r', encoding='utf-8') as f:
        variations_data = json.load(f)

    output_records = []
    seen_questions = set()

    if not variations_data:
        return 0

    # Identify the dynamic perturbation key
    sample_record = variations_data[0]
    standard_keys = {'table', 'column', 'original'}
    variation_keys = [k for k in sample_record.keys() if k not in standard_keys]
    
    if not variation_keys:
        print(f"Skipping {variations_json_path}: No variation key found.")
        return 0
    
    variation_type_key = variation_keys[0]

    for variation_record in tqdm(variations_data, desc=f"Processing {os.path.basename(variations_json_path)}", leave=False):
        table_name = variation_record['table']
        column_name = variation_record['column']
        
        # The value we want to INJECT into the SQL (Clean)
        target_clean_value = variation_record['original']
        # The value we want to INJECT into the Question (Dirty/Perturbed)
        target_perturbed_value = variation_record[variation_type_key]

        if str(target_clean_value).lower() == str(target_perturbed_value).lower():
            continue

        # Find a template in the benchmark that uses this Table/Column
        for benchmark_record in benchmark_data:
            
            template_match = False
            template_value = None
            
            # Check if this benchmark record uses the target table/column
            for value_info in benchmark_record.get('values', []):
                if (value_info['table'].lower() == table_name.lower() and 
                    value_info['column'].lower() == column_name.lower()):
                    
                    template_match = True
                    template_value = value_info['value']
                    break
            
            if not template_match or not template_value:
                continue

            # --- GENERATION LOGIC ---
            
            original_sql = benchmark_record['SQL']
            original_question = benchmark_record['question']

            # 1. Generate New SQL (Ground Truth)
            # Replace the Template Value (e.g., 'Majid Heravi') with the Variation Original (e.g., 'Gokgoz, Ali')
            new_sql = safe_replace(original_sql, template_value, target_clean_value)
            
            if new_sql == original_sql:
                continue

            # 2. Generate New Question (Perturbed)
            # Replace Template with Perturbed Value (e.g., 'Gokgoz, Ali.')
            new_question = safe_replace(original_question, template_value, target_perturbed_value)

            # 3. Generate New Question (Clean) - NEW FIELD
            # Replace Template with Clean Value (e.g., 'Gokgoz, Ali')
            question_clean_value = safe_replace(original_question, template_value, target_clean_value)

            # Deduplication
            if new_question in seen_questions:
                continue

            # 4. Validate via Multiprocessing
            # Check if the NEW SQL (using the clean value) actually returns data
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=check_sql_validity,
                args=(new_sql, result_queue)
            )
            p.start()
            p.join(120) # Timeout

            if p.is_alive():
                p.terminate()
                p.join()
                continue

            try:
                is_valid = result_queue.get_nowait()
            except queue.Empty:
                is_valid = False

            if is_valid:
                new_record = copy.deepcopy(benchmark_record)
                new_record['original_question'] = original_question # Template Question
                new_record['question'] = new_question               # Perturbed Question
                new_record['question_clean_value'] = question_clean_value # Clean Question (New Field)
                new_record['SQL'] = new_sql
                new_record['original_SQL'] = original_sql           # Template SQL
                
                # Update the 'values' metadata to reflect the new reality
                for v in new_record['values']:
                    if v['table'].lower() == table_name.lower() and v['column'].lower() == column_name.lower():
                        v['value'] = target_clean_value
                
                new_record['changes_information'] = {
                    'template_value': template_value,
                    'new_clean_value': target_clean_value,
                    'new_perturbed_value': target_perturbed_value,
                    'perturbation_type': variation_type_key
                }
                
                output_records.append(new_record)
                seen_questions.add(new_question)

    # Write output
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