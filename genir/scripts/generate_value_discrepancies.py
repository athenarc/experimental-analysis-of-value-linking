import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Generator

from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class ValueVariations(BaseModel):
    typographical_errors: List[str] = Field(
        description="Common typos, including deletions, insertions, substitutions, or transpositions."
    )
    formatting_variations: List[str] = Field(
        description="Variations with added/removed spaces or punctuation changes."
    )
    structural_variations: List[str] = Field(
        description="Variations with word additions, removals, reordering, or singular/plural changes."
    )
    abbreviations_and_clipping: List[str] = Field(
        description="Common abbreviations or clipped versions of the value."
    )
    synonyms_and_paraphrases: List[str] = Field(
        description="Other words or phrases that mean the same thing in this context."
    )
    negated_antonyms: List[str] = Field(
        description="Phrases that explicitly mean 'not the value'."
    )


def get_db_schema(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info(`{table}`)")
        schema[table] = cursor.fetchall()
    return schema


def main(db_root_path, output_path, model_name, num_samples):
    print("Phase 1: Collecting and de-duplicating values from all databases...")
    db_files = list(Path(db_root_path).rglob("*.sqlite"))

    values_to_process = {}
    all_db_values = []
    
    for db_path in tqdm(db_files, desc="Scanning Databases"):
        db_id = db_path.stem
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

                cursor.execute(f'SELECT DISTINCT `{col_name}` FROM `{table_name}`')
                for row in cursor.fetchall():
                    value = row[0]
                    if isinstance(value, str) and re.search('[a-zA-Z]', value) and len(value.split()) <= 4 and not value.startswith('http') and 'www.' not in value and len(value) <= 256:
                        context = (db_id, table_name, col_name)
                        all_db_values.append((value, *context))
                        if value not in values_to_process:
                            values_to_process[value] = context

        conn.close()

    print(f"Found {len(all_db_values)} total values, with {len(values_to_process)} unique values to process.")

    print(f"\nPhase 2: Generating {num_samples} variation samples per value using vLLM...")
    
    # SOLUTION: Reduce gpu_memory_utilization to leave more room for model weights. Start with 0.7 or 0.6.
    # Also consider reducing max_model_len if your prompts are short.
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir="/data/hdd1/vllm_models/",
        gpu_memory_utilization=0.70, # ADJUSTED FROM 0.80
        max_model_len=4096,
        tensor_parallel_size=2
    )
    
    json_schema = ValueVariations.model_json_schema()
    
    guided_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0.1,
        max_tokens=1024, # Reduced as we don't expect huge JSONs
        guided_decoding=guided_params
    )

    system_prompt = "You are an expert data annotator. Your task is to generate realistic variations of a given database value that a user might type in a natural language query. You will be given the value and its context (database, table, and column). Generate a wide range of variations covering typographical errors, formatting changes, structural differences, and semantic alternatives. Respond ONLY with the JSON object matching the provided schema."

    generated_variations_cache = {}
    unique_values_list = list(values_to_process.keys())

    prompts_to_generate = []
    for value in tqdm(unique_values_list, desc="Preparing All Prompts"):
        db_id, table_name, column_name = values_to_process[value]
        user_prompt = f"""
Generate variations for the following database value.

**Context:**
- Database: '{db_id}'
- Table: '{table_name}'
- Column: '{column_name}'

**Value to Generate Variations For:**
'{value}'

Provide the output in a single JSON object.
"""
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
        prompts_to_generate.append(full_prompt)

    outputs = llm.generate(prompts_to_generate, sampling_params)

    for i, request_output in enumerate(tqdm(outputs, desc="Processing Generated Variations")):
        original_value = unique_values_list[i]
        
        merged_variations = defaultdict(list)
        for completion_output in request_output.outputs:
            generated_text = completion_output.text
            try:
                variations_json = json.loads(generated_text)
                for key, value_list in variations_json.items():
                    if isinstance(value_list, list):
                        merged_variations[key].extend(value_list)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for value '{original_value}'. Output: {generated_text}")
        
        final_variations = {}
        for key, value_list in merged_variations.items():
            final_variations[key] = list(set(value_list))

        generated_variations_cache[original_value] = final_variations

    print("\nPhase 3: Organizing results and writing to output files...")
    outputs_by_db = defaultdict(list)
    for value, db_id, table_name, col_name in all_db_values:
        if value in generated_variations_cache:
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
    parser = argparse.ArgumentParser(description="Generate variations for database values using an LLM.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the root folder containing BIRD-style databases.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where output .jsonl files will be saved.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the Hugging Face model to use with vLLM.")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of variation sets to sample per value.")
    
    args = parser.parse_args()
    main(args.db_root_path, args.output_path, args.model_name, args.num_samples)