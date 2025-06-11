import json
import os
import sqlite3
import re
import copy
from tqdm import tqdm

def generate_benchmark_variations(benchmark_json_path, variations_json_path, db_folder_path, output_json_path):
    """
    Generates variations of a text-to-SQL benchmark.

    This function iterates through a set of value variations, finds corresponding
    records in a text-to-SQL benchmark, and creates new benchmark items by
    swapping values in the questions and SQL queries. It verifies that the
    newly generated SQL query returns a non-empty result before adding the
    new record to the output.

    Args:
        benchmark_json_path (str): Path to the input benchmark JSON file.
        variations_json_path (str): Path to the JSON file containing value variations.
        db_folder_path (str): Path to the root folder containing database files.
        output_json_path (str): Path to write the output JSON file.
    """
    with open(benchmark_json_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    with open(variations_json_path, 'r', encoding='utf-8') as f:
        variations_data = json.load(f)

    output_records = []
    db_connections = {}

    try:
        for variation_record in tqdm(variations_data, desc="Processing Variations"):
            try:
                db_id = variation_record['database_id']
            except:
                continue # skip map
            table_name = variation_record['table']
            column_name = variation_record['column']
            original_value = variation_record['original_value']
            
            variation_key = list(variation_record.keys())[-1]
            variation_value = variation_record[variation_key]

            if db_id not in db_connections:
                db_path = os.path.join(db_folder_path, db_id, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    db_connections[db_id] = sqlite3.connect(db_path)
                else:
                    continue 
            
            conn = db_connections[db_id]
            cursor = conn.cursor()

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
                        
                        # Attempt replacement for single-quoted strings (case-insensitive)
                        new_sql = re.sub(
                            f"'{re.escape(str(benchmark_value))}'",
                            f"'{original_value}'",
                            original_sql,
                            flags=re.IGNORECASE
                        )
                        
                        # If no change, attempt for double-quoted strings
                        if new_sql == original_sql:
                            new_sql = re.sub(
                                f'"{re.escape(str(benchmark_value))}"',
                                f'"{original_value}"',
                                original_sql,
                                flags=re.IGNORECASE
                            )

                        # If still no change, attempt for unquoted values
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

                        cursor.execute(new_sql)
                        results = cursor.fetchall()

                        if results:
                            new_record = copy.deepcopy(benchmark_record)
                            
                            new_record['original_question'] = new_record.pop('question')
                            new_record['question'] = new_question
                            new_record['SQL'] = new_sql
                            
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
    finally:
        for db_id, conn in db_connections.items():
            conn.close()

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_records, f, indent=4)
        
        
if __name__ == "__main__":
    benchmark_json_path = 'assets/value_linking_valid_values_exact_no_bird_train.json'
    variations_json_path = 'assets/data_exploration/typo_deletions.json'
    db_folder_path = 'assets/all_databases'
    output_json_path = 'assets/all_benchmarks_human/test.json'

    generate_benchmark_variations(benchmark_json_path, variations_json_path, db_folder_path, output_json_path)
    print(f"Output written to {output_json_path}")
