import json
import os
import sqlite3
import argparse
import re
import time
import wandb
import backoff
from tqdm import tqdm
from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError

# ==========================================
# CONFIGURATION
# ==========================================

OPENROUTER_API_KEY = ""     #INSERT YOUR OPENROUTER API KEY
MODEL_ID = "openai/gpt-5.2" 
SITE_URL = "https://wandb.ai"
APP_NAME = "SQL-Eval-Script"

WANDB_ENTITY = ""
WANDB_PROJECT = "value_linking_prop_llms"

# Pricing (Per 1M tokens)
PRICE_INPUT_UNCACHED = 1.75
PRICE_INPUT_CACHED = 0.175
PRICE_OUTPUT = 14.00

if not OPENROUTER_API_KEY:
    # Fallback to env var if not hardcoded
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": SITE_URL,
        "X-Title": APP_NAME,
    }
)

# ==========================================
# SCHEMA GENERATION
# ==========================================

def nice_look_table(column_names: list, values: list):
    rows = []
    if not values:
        return ""
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    return header + '\n' + rows

def generate_schema_prompt(db_path, num_rows=3):
    full_schema_prompt_list = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except:
        try:
            conn = sqlite3.connect(db_path)
        except:
            return ""
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}

    for table in tables:
        table_name = table[0]
        if table_name == 'sqlite_sequence':
            continue
        
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table_name))
        res = cursor.fetchone()
        if res:
            create_prompt = res[0]
        else:
            continue
        
        if num_rows:
            safe_table = f"`{table_name}`" if table_name in ['order', 'by', 'group'] else table_name
            try:
                cursor.execute(f"SELECT * FROM {safe_table} LIMIT {num_rows}")
                column_names = [description[0] for description in cursor.description]
                values = cursor.fetchall()
                rows_prompt = nice_look_table(column_names=column_names, values=values)
                verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {safe_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
                schemas[table_name] = f"{create_prompt} \n {verbose_prompt}"
            except:
                schemas[table_name] = create_prompt
        else:
            schemas[table_name] = create_prompt

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    conn.close()
    return "\n\n".join(full_schema_prompt_list)

# ==========================================
# API INTERACTION (FIXED FOR COST)
# ==========================================

# Added APITimeoutError to retry logic
@backoff.on_exception(backoff.expo, (APIConnectionError, RateLimitError, APITimeoutError), max_tries=5)
def invoke_llm_optimized(schema_context, question, evidence):
    
    system_message = f"""You are an expert in SQLite. Given the database schema and example rows below, write a SQL query to answer the question.

DATABASE SCHEMA:
{schema_context}"""

    user_message = f"""QUESTION: {question}

{f'EVIDENCE: {evidence}' if evidence else ''}

INSTRUCTIONS:
1. Output ONLY the SQL query. 
2. NO explanation. NO reasoning. NO markdown.
3. Start with SELECT.
4. End with ;"""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=300,
            stop=[";", "```"],
            timeout=40  # <--- ADDED TIMEOUT to prevent hanging
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            
        return content, prompt_tokens, completion_tokens, cached_tokens

    except Exception as e:
        print(f"API Logic Error: {e}", flush=True)
        raise e # Re-raise to trigger backoff

def clean_sql(text):
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    match = re.search(r'SELECT.*', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0)
    return text

# ==========================================
# EXECUTION & EVALUATION
# ==========================================

def execute_sql(sql, db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return set(result)
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_pair(pred_sql, gold_sql, db_path):
    gold_res = execute_sql(gold_sql, db_path)
    pred_res = execute_sql(pred_sql, db_path)
    
    if isinstance(gold_res, str):
        return 0, "Gold Error", gold_res, pred_res
    if isinstance(pred_res, str):
        return 0, "Execution Error", gold_res, pred_res
    
    is_correct = 1 if gold_res == pred_res else 0
    return is_correct, "Success", gold_res, pred_res

# ==========================================
# CHECKPOINTING UTILS
# ==========================================

def load_processed_ids(run_name):
    """Load IDs that have already been processed from a local JSONL file."""
    processed = set()
    filename = f"results_{run_name}.jsonl"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We use a combination of DB_ID and Question as a unique key
                    # or just the index if available. Let's use the index 'i' we log.
                    if 'index' in data:
                        processed.add(data['index'])
                except:
                    pass
    return processed

def save_checkpoint(run_name, data_dict):
    """Append a single result to the JSONL file."""
    filename = f"results_{run_name}.jsonl"
    with open(filename, 'a') as f:
        f.write(json.dumps(data_dict) + "\n")

# ==========================================
# MAIN PIPELINE
# ==========================================

def process_dataset(file_path, run_name, db_root, limit, global_schema_cache):
    print(f"\n{'#'*60}", flush=True)
    print(f"STARTING RUN: {run_name}", flush=True)
    print(f"FILE: {file_path}", flush=True)
    print(f"{'#'*60}\n", flush=True)

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found. Skipping.", flush=True)
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Sort data
    data.sort(key=lambda x: x['db_id'])

    if limit:
        data = data[:limit]
        print(f"Limiting to first {limit} examples.", flush=True)

    # LOAD CHECKPOINT
    processed_indices = load_processed_ids(run_name)
    print(f"Found {len(processed_indices)} already processed items. Resuming...", flush=True)

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        reinit=True,
        resume="allow", # Allow resuming wandb runs
        id=f"{run_name}_chkpt", # Static ID to resume the same chart
        config={
            "model_id": MODEL_ID,
            "dataset_file": file_path,
            "provider": "OpenRouter",
        }
    )

    wandb_table = wandb.Table(columns=[
        "ID", "DB_ID", "Question", "Gold SQL", "Pred SQL", "Correct", 
        "Latency", "Total Input Tokens", "Cached Tokens", "New Input Tokens", 
        "Output Tokens", "Cost ($)", "Error"
    ])

    metrics = {
        "correct": 0,
        "total": 0,
        "total_input": 0,
        "total_cached": 0,
        "total_output": 0,
        "total_cost": 0.0
    }

    current_db_id = None
    current_schema_context = ""

    try:
        for i, item in tqdm(enumerate(data), total=len(data), desc=run_name):
            
            # SKIP IF ALREADY DONE
            if i in processed_indices:
                continue

            db_id = item['db_id']
            question = item['question']
            evidence = item['evidence']
            gold_sql = item['SQL']
            
            db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
            
            # Schema Caching Logic
            if db_id != current_db_id:
                if db_id not in global_schema_cache:
                    # Clear cache if it gets too big (prevent OOM)
                    if len(global_schema_cache) > 50: 
                        global_schema_cache.clear()
                    global_schema_cache[db_id] = generate_schema_prompt(db_path, num_rows=3)
                current_schema_context = global_schema_cache[db_id]
                current_db_id = db_id

            start_time = time.time()
            
            try:
                raw_response, prompt_tokens, completion_tokens, cached_tokens = invoke_llm_optimized(
                    current_schema_context, question, evidence
                )
            except Exception as e:
                print(f"Failed to get response for {i}: {e}", flush=True)
                continue # Skip this item if API fails completely
            
            end_time = time.time()
            latency = end_time - start_time
            
            pred_sql = clean_sql(raw_response)

            is_correct, status, gold_res, pred_res = evaluate_pair(pred_sql, gold_sql, db_path)
            
            uncached_tokens = prompt_tokens - cached_tokens
            cost_input = (uncached_tokens * PRICE_INPUT_UNCACHED / 1_000_000) + \
                         (cached_tokens * PRICE_INPUT_CACHED / 1_000_000)
            cost_output = (completion_tokens * PRICE_OUTPUT / 1_000_000)
            item_cost = cost_input + cost_output

            metrics["correct"] += is_correct
            metrics["total"] += 1
            metrics["total_input"] += prompt_tokens
            metrics["total_cached"] += cached_tokens
            metrics["total_output"] += completion_tokens
            metrics["total_cost"] += item_cost

            error_msg = str(pred_res) if isinstance(pred_res, str) and not pred_res.startswith("{") else ""
            
            # Log to WandB
            wandb_table.add_data(
                i, db_id, question, gold_sql, pred_sql, is_correct, 
                latency, prompt_tokens, cached_tokens, uncached_tokens, 
                completion_tokens, item_cost, error_msg
            )

            wandb.log({
                "acc": metrics["correct"] / metrics["total"],
                "cost_cumulative": metrics["total_cost"],
                "cache_hit_rate": metrics["total_cached"] / metrics["total_input"] if metrics["total_input"] > 0 else 0,
                "current_latency": latency
            })

            # SAVE CHECKPOINT TO DISK
            save_checkpoint(run_name, {
                "index": i,
                "db_id": db_id,
                "correct": is_correct,
                "cost": item_cost
            })

    except KeyboardInterrupt:
        print(f"\nRun {run_name} interrupted. Saving...", flush=True)
    except Exception as e:
        print(f"\nCRITICAL ERROR in {run_name}: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        if metrics["total"] > 0:
            acc = metrics["correct"] / metrics["total"]
            print(f"\n--- RESULTS: {run_name} ---", flush=True)
            print(f"Accuracy: {acc:.2%}", flush=True)
            print(f"Cost: ${metrics['total_cost']:.4f}", flush=True)
            print("---------------------------", flush=True)
        
        wandb.log({
            "evaluation_results": wandb_table,
            "final_accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
            "final_cost": metrics["total_cost"]
        })
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_root', type=str, default='bird-sql/my_benchmark/test_databases')
    parser.add_argument('--limit', type=int, default=None, help="Limit queries per dataset")
    args = parser.parse_args()

    global_schema_cache = {}

    datasets = [
        {
            "path": "assets/dev_p1.json",
            "name": "gpt-5.2-prec-1"
        },
        {
            "path": "assets/dev_p05.json",
            "name": "gpt-5.2-prec-05"
        },
        {
            "path": "assets/dev_p001.json",
            "name": "gpt-5.2-prec-001"
        }
    ]

    for ds in datasets:
        process_dataset(
            file_path=ds["path"],
            run_name=ds["name"],
            db_root=args.db_root,
            limit=args.limit,
            global_schema_cache=global_schema_cache
        )

if __name__ == "__main__":
    main()