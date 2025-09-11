import sqlite3
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
import re

def get_db_schema_and_samples(db_path: Path) -> dict | None:
    """
    Connects to a SQLite database and extracts its schema, including table names,
    column names, and two distinct sample values for each column.

    Args:
        db_path: The path to the SQLite database file.

    Returns:
        A dictionary representing the database schema, or None if an error occurs.
    """
    if not db_path.exists():
        print(f"Warning: Database file not found at {db_path}")
        return None
        
    schema = {}
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = con.cursor()

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            schema[table_name] = {}
            # Get column names
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            columns_info = cursor.fetchall()
            
            for col_info in columns_info:
                col_name = col_info[1]
                try:
                    # Get 2 distinct, non-null sample values
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL LIMIT 2;'
                    cursor.execute(query)
                    samples = [str(row[0]) for row in cursor.fetchall()]
                    schema[table_name][col_name] = samples
                except sqlite3.OperationalError as e:
                    print(f"Warning: Could not query column '{col_name}' in table '{table_name}': {e}")
                    schema[table_name][col_name] = []

        con.close()
        return schema
    except Exception as e:
        print(f"Error processing database {db_path}: {e}")
        return None

def format_schema_for_prompt(schema: dict) -> str:
    """
    Formats the extracted schema into a human-readable string for the LLM prompt.
    """
    formatted_string = "Database Schema:\n"
    for table_name, columns in schema.items():
        formatted_string += f"Table '{table_name}':\n"
        for col_name, samples in columns.items():
            # Truncate long sample values before joining
            processed_samples = [
                f'"{s[:50]}...{s[-10:]}"' if len(s) > 100 else f'"{s}"' 
                for s in samples
            ]
            sample_str = ", ".join(processed_samples)
            
            if not sample_str:
                sample_str = " (no values found)"
            else:
                sample_str = f" (e.g., {sample_str})"
            formatted_string += f"  - Column '{col_name}'{sample_str}\n"
    return formatted_string

def construct_prompt(natural_language_query: str, formatted_schema: str) -> str:
    """
    Constructs the final prompt for the LLM using a template.
    """
    prompt = f"""You are an expert entity extractor. Your task is to identify and extract any potential database values mentioned in a natural language query.

Read the database schema and the query provided below. Extract all phrases from the query that could correspond to values in the database columns.

{formatted_schema}
---
Natural Language Query: "{natural_language_query}"
---

Based on the query and the schema, extract the potential values.

Return your answer ONLY as a single, valid JSON list of strings. For example: ["value1", "value2"].
If no values are found, return an empty list: [].

Extracted values:
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Extract value references from natural language queries using vLLM.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to the root folder containing database subdirectories (e.g., 'database/').")
    parser.add_argument("--queries_file_path", type=str, required=True, help="Path to the input JSON file containing queries and their db_id (e.g., 'dev.json').")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output JSON file to save results.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Name or path of the vLLM-compatible model to use.")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use for tensor parallelism.")
    
    args = parser.parse_args()

    # 1. Initialize vLLM
    print(f"Loading model '{args.model}' with vLLM...")
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,download_dir="/data/hdd1/vllm_models/",gpu_memory_utilization=0.7,max_model_len=8192,max_num_seqs=1)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # 2. Load queries
    print(f"Loading queries from '{args.queries_file_path}'...")
    with open(args.queries_file_path, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    # 3. Construct all prompts
    prompts = []
    schema_cache = {}
    db_root = Path(args.db_root_path)

    print("Constructing prompts with database schema information...")
    for item in tqdm(queries_data, desc="Processing queries"):
        db_id = item['db_id']
        query = item['new_question_correct_value'] # Assuming the key is 'question'

        # Get schema from cache or load it
        if db_id not in schema_cache:
            db_path = db_root / db_id / f"{db_id}.sqlite"
            schema = get_db_schema_and_samples(db_path)
            if schema:
                schema_cache[db_id] = format_schema_for_prompt(schema)
            else:
                # Skip if schema can't be loaded
                schema_cache[db_id] = "Error: Could not load schema."
        
        formatted_schema = schema_cache[db_id]
        if "Error" in formatted_schema:
            # Add a placeholder to maintain list alignment
            prompts.append(None)
            continue

        # Create and add the prompt
        prompt = construct_prompt(query, formatted_schema)
        prompts.append(prompt)

    # Filter out any prompts that failed to generate
    valid_prompts = [p for p in prompts if p is not None]
    
    # 4. Run batch generation with vLLM
    print(f"\nGenerating value extractions for {len(valid_prompts)} prompts...")
    outputs = llm.generate(valid_prompts, sampling_params)

    # 5. Parse outputs and save results
    results = []
    output_idx = 0
    for i, item in enumerate(queries_data):
        # Ensure we align original queries with their corresponding model outputs
        if prompts[i] is None:
            result_item = {
                "query": item['new_question_correct_value'],
                "db_id": item['db_id'],
                "extracted_values": ["Error: Database schema could not be loaded."]
            }
        else:
            generated_text = outputs[output_idx].outputs[0].text.strip()
            output_idx += 1
            try:
                # Find the JSON list in the output, robust to surrounding text
                json_match = re.search(r'\[.*\]', generated_text, re.DOTALL)
                if json_match:
                    extracted_values = json.loads(json_match.group(0))
                else:
                    extracted_values = ["Error: Model did not produce a valid list format."]
            except json.JSONDecodeError:
                extracted_values = ["Error: Model output was not valid JSON.", f"Raw output: {generated_text}"]
            
            result_item = {
                "query": item['new_question_correct_value'],
                "db_id": item['db_id'],
                "extracted_values": extracted_values
            }
        results.append(result_item)

    print(f"\nSaving results to '{args.output_file_path}'...")
    with open(args.output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Extraction complete.")

if __name__ == "__main__":
    main()