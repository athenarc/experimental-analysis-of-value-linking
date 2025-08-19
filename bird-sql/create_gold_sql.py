# File: create_gold_sql.py

import json
import os
import argparse

def create_gold_sql_file(input_json_path: str, output_txt_path: str):
    """
    Reads a BIRD-style JSON file and creates a ground truth SQL file.

    The output format for each line is:
    SQL_QUERY<TAB>DB_ID
    """
    print(f"Reading data from: {input_json_path}")
    
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_json_path}'")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{input_json_path}'. Please check the file format.")
        return

    if not isinstance(data, list):
        print("ERROR: Expected the JSON file to contain a list of records.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_txt_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    lines_to_write = []
    records_processed = 0
    records_skipped = 0

    for i, record in enumerate(data):
        if not isinstance(record, dict):
            print(f"WARNING: Item at index {i} is not a dictionary. Skipping.")
            records_skipped += 1
            continue

        sql_query = record.get("SQL")
        db_id = record.get("db_id")

        if sql_query is None or db_id is None:
            print(f"WARNING: Record at index {i} is missing 'SQL' or 'db_id' key. Skipping.")
            records_skipped += 1
            continue

        # Clean the SQL query: remove newlines and trailing whitespace
        # to ensure it fits on a single line.
        cleaned_sql = ' '.join(sql_query.strip().split())
        
        # Format the line as "SQL<TAB>db_id"
        line = f"{cleaned_sql}\t{db_id}"
        lines_to_write.append(line)
        records_processed += 1

    print(f"Writing {records_processed} records to: {output_txt_path}")
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_to_write))
        f.write('\n') # Add a final newline for POSIX compatibility

    print("\n--- Summary ---")
    print(f"Successfully created ground truth file: '{output_txt_path}'")
    print(f"Total records processed: {records_processed}")
    if records_skipped > 0:
        print(f"Total records skipped due to missing data: {records_skipped}")
    print("-----------------")


if __name__ == "__main__":
    # --- Configuration ---
    # Set the path to your input JSON file
    INPUT_JSON = "my_benchmark/test_all.json"
    
    # Set the path for the output ground truth SQL file
    OUTPUT_TXT = "my_benchmark/test_gold_sqls.txt"
    # ---------------------

    parser = argparse.ArgumentParser(description="Generate a ground truth SQL file from a BIRD-style JSON file.")
    parser.add_argument("--input", default=INPUT_JSON, help=f"Path to the input JSON file (default: {INPUT_JSON})")
    parser.add_argument("--output", default=OUTPUT_TXT, help=f"Path for the output .txt file (default: {OUTPUT_TXT})")
    
    args = parser.parse_args()
    
    create_gold_sql_file(args.input, args.output)