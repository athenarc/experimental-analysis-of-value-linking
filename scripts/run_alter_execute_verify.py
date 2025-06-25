import json
import os
import sqlite3
import re
import copy
from tqdm import tqdm
import multiprocessing
import queue # For getting results back from the process

def execute_sql_checks(db_path, new_sql, sql_with_variation, result_queue):
    """
    Executes the two validation SQL queries in a separate process.
    This function is designed to be terminated if it runs too long.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # CHECK 1: The SQL with the correct value must return a non-empty result.
        cursor.execute(new_sql)
        results = cursor.fetchall()

        if not results:
            result_queue.put(False)
            return

        # CHECK 2: The SQL with the variation value must return an EMPTY result.
        cursor.execute(sql_with_variation)
        variation_results = cursor.fetchall()
        
        # If the variation yields no results, the checks pass.
        if not variation_results:
            result_queue.put(True)
        else:
            result_queue.put(False)

    except (sqlite3.Error, Exception):
        result_queue.put(False)
    finally:
        if conn:
            conn.close()


def generate_benchmark_variations(benchmark_json_path, variations_json_path, db_folder_path, output_json_path):
    """
    Generates variations of a text-to-SQL benchmark.
    """
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    with open(variations_json_path, 'r', encoding='utf-8') as f:
        variations_data = json.load(f)

    output_records = []

    for variation_record in tqdm(variations_data, desc="Processing Variations"):
        try:
            db_id = variation_record['database_id']
        except KeyError:
            continue

        db_path = os.path.join(db_folder_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            continue

        table_name = variation_record['table']
        column_name = variation_record['column']
        original_value = variation_record['original_value']
        
        variation_key = list(variation_record.keys())[-1]
        variation_value = variation_record[variation_key]

        for benchmark_record in benchmark_data:
            if benchmark_record['db_id'] != db_id:
                continue

            for i, value_info in enumerate(benchmark_record['values']):
                if value_info['table'] == table_name and value_info['column'].lower() == column_name.lower():
                    benchmark_value = value_info['value']
                    
                    match_pattern = r'\b' + re.escape(str(benchmark_value)) + r'\b'
                    if not re.search(match_pattern, benchmark_record['question'], re.IGNORECASE):
                        continue

                    original_sql = benchmark_record['SQL']
                    
                    new_sql = re.sub(
                        f"'{re.escape(str(benchmark_value))}'",
                        f"'{original_value}'",
                        original_sql,
                        flags=re.IGNORECASE
                    )
                    if new_sql == original_sql:
                        new_sql = re.sub(
                            f'"{re.escape(str(benchmark_value))}"',
                            f'"{original_value}"',
                            original_sql,
                            flags=re.IGNORECASE
                        )
                    if new_sql == original_sql:
                        new_sql = re.sub(
                            r'=\s*' + match_pattern,
                            f'= {original_value}',
                            original_sql,
                            flags=re.IGNORECASE
                        )

                    new_question = re.sub(
                        match_pattern, 
                        str(variation_value), 
                        benchmark_record['question'], 
                        count=1, 
                        flags=re.IGNORECASE
                    )
                    sql_with_variation = re.sub(
                        f"'{re.escape(str(original_value))}'",
                        f"'{str(variation_value)}'",
                        new_sql,
                        flags=re.IGNORECASE
                    )
                    if sql_with_variation == new_sql:
                        sql_with_variation = re.sub(
                            f'"{re.escape(str(original_value))}"',
                            f'"{str(variation_value)}"',
                            new_sql,
                            flags=re.IGNORECASE
                        )
                    if sql_with_variation == new_sql:
                        variation_match_pattern = r'\b' + re.escape(str(original_value)) + r'\b'
                        sql_with_variation = re.sub(
                            r'=\s*' + variation_match_pattern,
                            f'= {str(variation_value)}',
                            new_sql,
                            flags=re.IGNORECASE
                        )

                    result_queue = multiprocessing.Queue()
                    p = multiprocessing.Process(
                        target=execute_sql_checks,
                        args=(db_path, new_sql, sql_with_variation, result_queue)
                    )
                    p.start()
                    p.join(120) # Wait for 120 seconds (2 minutes)

                    if p.is_alive():
                        # Process is still running, so it timed out
                        p.terminate() # Terminate the process
                        p.join()      # Wait for termination to complete
                        continue      # Skip to the next iteration

                    try:
                        checks_passed = result_queue.get_nowait()
                    except queue.Empty:
                        checks_passed = False # Queue is empty, assume failure

                    if checks_passed:
                        new_record = copy.deepcopy(benchmark_record)
                        
                        new_record['original_question'] = new_record.pop('question')
                        new_record['question'] = new_question
                        new_record['SQL'] = new_sql
                        new_record['original_SQL'] = original_sql
                        new_record['values'][i]['value'] = str(original_value)
                        
                        normalized_benchmark_val = str(benchmark_value).lower().replace(' ', '_')
                        normalized_original_val = str(original_value).lower().replace(' ', '_')
                        
                        old_val_list_item = f"{table_name}.{value_info['column']}.{normalized_benchmark_val}"
                        new_val_list_item = f"{table_name}.{value_info['column']}.{normalized_original_val}"
                        
                        if old_val_list_item in new_record['values_list']:
                            item_index = new_record['values_list'].index(old_val_list_item)
                            new_record['values_list'][item_index] = new_val_list_item

                        new_record['changes_information'] = {
                            'original_value': original_value,
                            variation_key: variation_value
                        }
                        
                        output_records.append(new_record)
                        break

    print(f"Generated {len(output_records)} variations.")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, indent=4)
        
        
if __name__ == "__main__":
    benchmark_json_path = 'assets/value_linking_valid_values_exact_no_bird_train.json'
    db_folder_path = 'assets/all_databases'
    json_folder_path = 'assets/data_exploration_human'
    output_json_folder = 'assets/all_benchmarks_human/'
    for file in tqdm(os.listdir(json_folder_path), desc="Processing JSON Files"):
        variations_json_path = os.path.join(json_folder_path, file)
        output_json_path = os.path.join(output_json_folder, file)
        generate_benchmark_variations(benchmark_json_path, variations_json_path, db_folder_path, output_json_path)
        print(f"Output written to {output_json_path}")