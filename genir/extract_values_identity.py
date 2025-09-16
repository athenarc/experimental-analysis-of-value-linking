import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from dateutil.parser import parse as parse_date
import phonenumbers

def get_db_schema(cursor):
    """Retrieves the schema (table and column names) for a given database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        schema[table] = cursor.fetchall()
    return schema

def is_date(string):
    """Checks if a string can be parsed as a date."""
    try:
        parse_date(string, fuzzy=False)
        return True
    except (ValueError, OverflowError, TypeError):
        return False

def is_phone_number(string):
    """Checks if a string is a valid phone number."""
    try:
        if len(string) < 7:
             return False
        parsed_number = phonenumbers.parse(string, None)
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.phonenumberutil.NumberParseException:
        return False

def is_code_like(string):
    """Checks if a string resembles an ID, code, or key."""
    if not any(c.isalpha() for c in string) or not any(c.isdigit() for c in string):
        return False
    # Check for patterns like UUIDs or product keys
    if re.search(r'[A-Z0-9]+-[A-Z0-9-]+', string):
        return True
    # Check if more than half the characters are digits (common in IDs)
    if len(string) > 4 and sum(c.isdigit() for c in string) / len(string) > 0.5:
        return True
    return False

def should_exclude_value(value):
    """
    Determines if a value should be filtered out based on a set of rules.
    Returns True if the value should be excluded, False otherwise.
    """
    # Exclude non-strings or empty/whitespace-only strings
    if not isinstance(value, str) or not value.strip():
        return True
    
    # Exclude dates, phone numbers, and code-like strings
    if is_date(value) or is_phone_number(value) or is_code_like(value):
        return True
    
    # Exclude strings without any alphabetic characters (e.g., "123", "---")
    if not re.search('[a-zA-Z]', value):
        return True
    
    # Exclude very long strings, URLs, or strings with too many words
    if len(value.split()) > 7 or value.startswith('http') or 'www.' in value or len(value) > 128:
        return True
        
    return False

def main(db_root_path, output_path):
    print("Scanning databases and extracting filtered values...")
    db_files = list(Path(db_root_path).rglob("*.sqlite"))

    # This dictionary will store the final results, grouped by database ID
    outputs_by_db = defaultdict(list)
    total_values_extracted = 0

    for db_path in tqdm(db_files, desc="Scanning Databases"):
        db_id = db_path.stem
        try:
            conn = sqlite3.connect(db_path)
            # Handle potential encoding issues in legacy databases
            conn.text_factory = lambda b: b.decode(errors='replace')
            cursor = conn.cursor()
            schema = get_db_schema(cursor)

            for table_name, columns in schema.items():
                for col_info in columns:
                    col_name = col_info[1]
                    col_type = col_info[2].upper()

                    # Only process columns that are likely to contain text
                    if not any(t in col_type for t in ["CHAR", "TEXT", "CLOB"]):
                        continue

                    try:
                        # Fetch unique values to avoid processing duplicates from the same column
                        cursor.execute(f'SELECT DISTINCT `{col_name}` FROM `{table_name}`')
                        for row in cursor.fetchall():
                            value = row[0]
                            # If the value passes the filter, create and store a record
                            if not should_exclude_value(value):
                                record = {
                                    "value": value,
                                    "table": table_name,
                                    "column": col_name,
                                }
                                outputs_by_db[db_id].append(record)
                                total_values_extracted += 1
                    except sqlite3.Error:
                        # Skip columns that cause errors (e.g., virtual tables)
                        continue
            conn.close()
        except sqlite3.Error:
            print(f"Warning: Could not open or read database {db_path}")
            continue

    print(f"\nScan complete. Extracted a total of {total_values_extracted} values from {len(outputs_by_db)} databases.")

    print("Writing results to output files...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for db_id, records in tqdm(outputs_by_db.items(), desc="Writing JSONL files"):
        output_file = output_dir / f"{db_id}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                # Write each record as a new line in JSON format
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete. Output files are located in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and extract text values from .sqlite databases.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the root folder containing .sqlite database files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where output .jsonl files will be saved.")
    
    args = parser.parse_args()
    main(args.db_root_path, args.output_path)