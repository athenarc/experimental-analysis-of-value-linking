import json
import os
import sqlite3
import argparse
import re
import time
import boto3
import backoff
import wandb
from tqdm import tqdm
from botocore.exceptions import ClientError


AWS_ACCESS_KEY_ID = ""  #INSERT YOUR AWS ACCESS KEY ID
AWS_SECRET_ACCESS_KEY = "" #INSERT YOUR AWS SECRET ACCESS KEY
REGION_NAME = "" 


WANDB_ENTITY = ""
WANDB_PROJECT = "value_linking_prop_llms"

bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def nice_look_table(column_names: list, values: list):
    rows = []
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    return header + '\n' + rows

def generate_schema_prompt(db_path, num_rows=3):
    """
    Extracts CREATE TABLE statements and 3 example rows (Value Linking).
    """
    full_schema_prompt_list = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except:
        conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}

    for table in tables:
        table_name = table[0]
        if table_name == 'sqlite_sequence':
            continue
        
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table_name))
        create_prompt = cursor.fetchone()[0]
        
        # Get 3 example rows for value linking
        if num_rows:
            safe_table = f"`{table_name}`" if table_name in ['order', 'by', 'group'] else table_name
            cursor.execute(f"SELECT * FROM {safe_table} LIMIT {num_rows}")
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = f"/* \n {num_rows} example rows: \n SELECT * FROM {safe_table} LIMIT {num_rows}; \n {rows_prompt} \n */"
            schemas[table_name] = f"{create_prompt} \n {verbose_prompt}"
        else:
            schemas[table_name] = create_prompt

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    return "\n\n".join(full_schema_prompt_list)

def generate_full_prompt_with_caching(db_path, question, evidence):
    """
    Generate prompt with cache control.
    The schema (static part) is marked for caching.
    """
    schema_context = generate_schema_prompt(db_path, num_rows=3)
    
    # Split into cacheable (schema) and non-cacheable (question) parts
    return schema_context, question, evidence

# ==========================================
# BEDROCK INTERACTION WITH CACHING
# ==========================================

def bedrock_backoff_handler(e):
    if isinstance(e, ClientError):
        error_code = e.response['Error']['Code']
        if error_code in ['ThrottlingException', 'TooManyRequestsException', 'ServiceUnavailable']:
            return True
    return False

@backoff.on_exception(backoff.expo, ClientError, giveup=lambda e: not bedrock_backoff_handler(e), max_tries=8)
def invoke_bedrock_with_caching(model_id, schema_context, question, evidence):
    """
    Claude-specific inference with custom SQL constraints and prompt caching.
    """
    
    # Your specific instructions integrated into the system prompt
    system_content = [
        {
            "type": "text",
            "text": (
                "You are an expert in SQLite. Given the database schema and example rows below, "
                "write a SQL query to answer the question. Be careful: if the question includes "
                "the word 'publication' or 'paper', you MUST add a constraint: result.type='publication'. "
                "Use only exact match with '=' in WHERE clauses; do not use LIKE or '%'."
            )
        },
        {
            "type": "text",
            "text": f"SCHEMA:\n{schema_context}",
            "cache_control": {"type": "ephemeral"}  # Caches the schema for cost savings
        }
    ]
    
    user_message = f"""QUESTION:
{question}

{f'EVIDENCE: {evidence}' if evidence else ''}

INSTRUCTIONS:
Output ONLY the SQL query.
Do not output markdown formatting like ```sql.
Start the query with SELECT."""

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": system_content,
        "messages": [{"role": "user", "content": user_message}]
    }
    
    response = bedrock_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    response_body = json.loads(response.get('body').read())
    
    # Extract text and usage
    pred_sql = response_body.get('content')[0].get('text')
    usage = response_body.get('usage', {})
    
    return pred_sql, usage

def clean_sql(text):
    # Remove markdown code blocks
    text = text.strip()
    if "```sql" in text:
        text = text.split("```sql")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    # Remove any conversational prefix
    match = re.search(r'SELECT.*', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0)
    return text

# ==========================================
# EXECUTION & EVALUATION
# ==========================================

def execute_sql(sql, db_path):
    """Executes SQL and returns the result set (set of tuples)"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return set(result)
    except Exception as e:
        return f"ERROR: {str(e)}"

def evaluate_pair(pred_sql, gold_sql, db_path):
    """Compares execution results of Prediction vs Gold"""
    gold_res = execute_sql(gold_sql, db_path)
    pred_res = execute_sql(pred_sql, db_path)
    
    if isinstance(gold_res, str):
        return 0, "Gold Error", gold_res, pred_res
    
    if isinstance(pred_res, str):
        return 0, "Execution Error", gold_res, pred_res
    
    is_correct = 1 if gold_res == pred_res else 0
    return is_correct, "Success", gold_res, pred_res
''
# ==========================================
# MAIN PIPELINE
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./assets/scalability_experiments/openaire_perturbed_subsampled.json')
    parser.add_argument('--db_root', type=str, default='./CHESS/data/openaire_og/dev_databases')
    parser.add_argument('--model_id', type=str, default="eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of examples for testing")
    args = parser.parse_args()
    
    # 1. Initialize WandB
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="claude-sonnet-4.5-openaire-subsampled-og",
        config={
            "model_id": args.model_id,
            "dataset": "BIRD-Dev",
            "use_knowledge": True,
            "prompt_caching": True
        }
    )

    # 2. Load Data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} examples.")

    # Table for WandB logging
    wandb_table = wandb.Table(columns=[
        "ID", "Question", "Evidence", "Gold SQL", "Pred SQL", "Correct", 
        "Latency (s)", "Cache Hit", "Non-Cached Tokens", "Cache Write Tokens", 
        "Cache Read Tokens", "Total Input Tokens", "Output Tokens", "Error Message"
    ])

    correct_count = 0
    total_count = 0
    
    # Track cache statistics
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # 3. Main Loop
    print(f"Starting evaluation on {len(data)} items...")

    for item in tqdm(data):
        db_id = item['db_id']
        question = item['question_clean_value'] #CHANGE THIS FOR PERTURBED
        evidence = item['evidence']
        gold_sql = item['SQL']
        
        db_path = os.path.join(args.db_root, db_id, f"{db_id}.sqlite")
        
        # A. Generate Prompt with Caching
        schema_context, question_text, evidence_text = generate_full_prompt_with_caching(
            db_path, question, evidence
        )
        
        # B. Inference with Timing
        start_time = time.time()
        raw_response, usage = invoke_bedrock_with_caching(
            args.model_id, schema_context, question_text, evidence_text
        )
        pred_sql = clean_sql(raw_response)
        
        # Claude-specific usage keys (snake_case)
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        
        # Caching metrics
        cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)
        cache_read_tokens = usage.get('cache_read_input_tokens', 0)
        
        # Total input for this call
        total_input_for_call = input_tokens + cache_creation_tokens + cache_read_tokens
        
        # Accumulate totals for the final report
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cache_creation_tokens += cache_creation_tokens
        total_cache_read_tokens += cache_read_tokens
        
        cache_hit = cache_read_tokens > 0

        


            
        end_time = time.time()
        latency = end_time - start_time
        
        # C. Evaluation
        is_correct, status, gold_res, pred_res = evaluate_pair(pred_sql, gold_sql, db_path)
        
        # D. Logging
        correct_count += is_correct
        total_count += 1
        
        # Log to WandB Table
        error_msg = pred_res if isinstance(pred_res, str) else ""
        wandb_table.add_data(
            total_count, 
            question, 
            evidence, 
            gold_sql, 
            pred_sql, 
            is_correct, 
            latency,
            cache_hit,
            input_tokens,              # Non-cached input tokens
            cache_creation_tokens,     # Cache write tokens
            cache_read_tokens,         # Cache read tokens
            total_input_for_call,      # Total input (all sources)
            output_tokens,
            error_msg
        )
        
        # Log running metrics
        wandb.log({
            "execution_accuracy": correct_count / total_count,
            "current_correct": is_correct,
            "latency": latency,
            "cache_hit": cache_hit,
            "non_cached_input_tokens": input_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            "total_input_tokens": total_input_for_call,
            "output_tokens": output_tokens
        })

    # 4. Finalize
    accuracy = correct_count / total_count
    cache_hit_rate = total_cache_read_tokens / (total_input_tokens + total_cache_creation_tokens + total_cache_read_tokens) if (total_input_tokens + total_cache_creation_tokens) > 0 else 0
    
    # Calculate costs
    cost_non_cached = total_input_tokens * 0.003 / 1000
    cost_cache_write = total_cache_creation_tokens * 0.00375 / 1000
    cost_cache_read = total_cache_read_tokens * 0.0003 / 1000
    cost_output = total_output_tokens * 0.015 / 1000
    total_cost = cost_non_cached + cost_cache_write + cost_cache_read + cost_output
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Execution Accuracy: {accuracy:.2%}")
    print(f"Total Queries: {total_count}")
    print(f"\nTOKEN USAGE:")
    print(f"  Non-cached input tokens:  {total_input_tokens:>10,} → ${cost_non_cached:>8.4f}")
    print(f"  Cache write tokens:       {total_cache_creation_tokens:>10,} → ${cost_cache_write:>8.4f}")
    print(f"  Cache read tokens:        {total_cache_read_tokens:>10,} → ${cost_cache_read:>8.4f}")
    print(f"  Output tokens:            {total_output_tokens:>10,} → ${cost_output:>8.4f}")
    print(f"  {'-'*70}")
    print(f"  TOTAL COST:                                    ${total_cost:>8.2f}")
    print(f"\nCACHE STATISTICS:")
    print(f"  Cache hit rate: {cache_hit_rate:.1%}")
    print(f"  Savings from caching: ~{(cost_cache_read / (cost_cache_write + cost_cache_read) * 100) if (cost_cache_write + cost_cache_read) > 0 else 0:.1f}%")
    print(f"{'='*70}\n")

    wandb.log({
        "final_accuracy": accuracy,
        "cache_hit_rate": cache_hit_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cache_creation_tokens": total_cache_creation_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_cost_usd": total_cost,
        "results_table": wandb_table
    })
    wandb.finish()

if __name__ == "__main__":
    main()