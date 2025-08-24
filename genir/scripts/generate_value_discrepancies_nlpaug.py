import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from dateutil.parser import parse as parse_date
import phonenumbers

def get_db_schema(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        schema[table] = cursor.fetchall()
    return schema

def is_date(string):
    try:
        parse_date(string, fuzzy=False)
        return True
    except (ValueError, OverflowError, TypeError):
        return False

def is_phone_number(string):
    try:
        if len(string) < 7:
             return False
        parsed_number = phonenumbers.parse(string, None)
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.phonenumberutil.NumberParseException:
        return False

def is_code_like(string):
    if not any(c.isalpha() for c in string) or not any(c.isdigit() for c in string):
        return False
    if re.search(r'[A-Z0-9]+-[A-Z0-9-]+', string):
        return True
    if len(string) > 4 and sum(c.isdigit() for c in string) / len(string) > 0.5:
        return True
    return False

def should_exclude_value(value):
    if not isinstance(value, str) or not value.strip():
        return True
    if is_date(value) or is_phone_number(value) or is_code_like(value):
        return True
    if not re.search('[a-zA-Z]', value):
        return True
    if len(value.split()) > 5 or value.startswith('http') or 'www.' in value or len(value) > 128:
        return True
    return False

def generate_formatting_variations(value: str, n: int) -> list:
    variations = set()
    variations.add(value.lower())
    variations.add(value.upper())
    if '-' in value:
        variations.add(value.replace('-', ' '))
        variations.add(value.replace('-', ''))
    if ' ' in value:
        variations.add(value.replace(' ', '-'))
        variations.add(value.replace(' ', ''))
    
    if value in variations:
        variations.remove(value)
        
    return list(variations)[:n]

def main(db_root_path, output_path, num_samples):
    print("Phase 1: Collecting and filtering values from all databases...")
    db_files = list(Path(db_root_path).rglob("*.sqlite"))

    values_to_process = set()
    all_db_values = []
    
    for db_path in tqdm(db_files, desc="Scanning Databases"):
        db_id = db_path.stem
        try:
            conn = sqlite3.connect(db_path)
            conn.text_factory = lambda b: b.decode(errors='replace')
            cursor = conn.cursor()
            schema = get_db_schema(cursor)

            for table_name, columns in schema.items():
                for col_info in columns:
                    col_name = col_info[1]
                    col_type = col_info[2].upper()

                    if not any(t in col_type for t in ["CHAR", "TEXT", "CLOB"]):
                        continue

                    try:
                        cursor.execute(f'SELECT DISTINCT `{col_name}` FROM `{table_name}`')
                        for row in cursor.fetchall():
                            value = row[0]
                            if not should_exclude_value(value):
                                context = (db_id, table_name, col_name)
                                all_db_values.append((value, *context))
                                values_to_process.add(value)
                    except sqlite3.Error:
                        continue
            conn.close()
        except sqlite3.Error:
            print(f"Warning: Could not open or read database {db_path}")
            continue

    unique_values_list = sorted(list(values_to_process))
    print(f"Found {len(all_db_values)} total values, with {len(unique_values_list)} unique, filtered values to process.")

    print(f"\nPhase 2: Defining augmenters and generating variations...")
    
    aug_typo = nac.KeyboardAug(aug_char_p=0.1, aug_word_p=0.2)
    aug_ocr = nac.OcrAug(aug_char_p=0.1, aug_word_p=0.2)
    aug_spelling = naw.SpellingAug()
    aug_split = naw.SplitAug()
    
    aug_swap_word = naw.RandomWordAug(action="swap")
    aug_delete_word = naw.RandomWordAug(action="delete")
    
    aug_synonym = naw.SynonymAug(aug_src='wordnet')
    aug_antonym = naw.AntonymAug()

    aug_contextual = naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased', action="substitute", device='cuda')

    generated_variations_cache = {}

    for value in tqdm(unique_values_list, desc="Augmenting Values"):
        variations = defaultdict(set)

        if len(value) > 3:
            typos = aug_typo.augment(value, n=num_samples)
            ocrs = aug_ocr.augment(value, n=num_samples)
            spellings = aug_spelling.augment(value, n=num_samples)
            variations["typographical_errors"].update(typos)
            variations["typographical_errors"].update(ocrs)
            variations["typographical_errors"].update(spellings)

        if len(value.split()) == 1 and len(value) > 5:
            splits = aug_split.augment(value, n=1)
            variations["structural_variations"].update(splits)

        if len(value.split()) > 1:
            swapped = aug_swap_word.augment(value, n=num_samples)
            deleted = aug_delete_word.augment(value, n=num_samples)
            variations["structural_variations"].update(swapped)
            variations["structural_variations"].update(deleted)
        
        synonyms = aug_synonym.augment(value, n=num_samples)
        antonyms = aug_antonym.augment(value, n=num_samples)
        variations["synonyms_and_paraphrases"].update(synonyms)
        variations["negated_antonyms"].update(antonyms)

        if len(value.split()) > 1:
            contextual_subs = aug_contextual.augment(value, n=num_samples)
            variations["synonyms_and_paraphrases"].update(contextual_subs)
        
        formatting = generate_formatting_variations(value, n=num_samples)
        variations["formatting_variations"].update(formatting)

        final_variations = {}
        for key, value_set in variations.items():
            if value in value_set:
                value_set.remove(value)
            if value_set:
                final_variations[key] = list(value_set)

        generated_variations_cache[value] = final_variations

    print("\nPhase 3: Organizing results and writing to output files...")
    outputs_by_db = defaultdict(list)
    for value, db_id, table_name, col_name in tqdm(all_db_values, desc="Organizing Results"):
        if value in generated_variations_cache and generated_variations_cache[value]:
            record = {
                "value": value,
                "table": table_name,
                "column": col_name,
                "variations": generated_variations_cache[value]
            }
            outputs_by_db[db_id].append(record)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for db_id, records in tqdm(outputs_by_db.items(), desc="Writing JSONL files"):
        output_file = output_dir / f"{db_id}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete. Output files are located in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate variations for database values using advanced nlpaug techniques.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the root folder containing .sqlite database files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where output .jsonl files will be saved.")
    parser.add_argument("-n", "--num_samples", type=int, default=3, help="Number of variation samples to generate per augmentation type.")
    
    args = parser.parse_args()
    main(args.db_root_path, args.output_path, args.num_samples)