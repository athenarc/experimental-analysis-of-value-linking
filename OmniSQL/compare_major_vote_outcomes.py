import json
import os
import sqlite3
import multiprocessing as mp
import sys
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import random

# --- Hard-coded file paths ---
# Prediction files
FILE_PATH_1 = "/data/hdd1/users/akouk/darelab_clean/DarelabDB/development/experimental_analysis_of_value_linking/OmniSQL/train_and_evaluate/results/predictions_precision_new_prompt_1.json"
FILE_PATH_0_5 = "/data/hdd1/users/akouk/darelab_clean/DarelabDB/development/experimental_analysis_of_value_linking/OmniSQL/train_and_evaluate/results/predictions_precision_new_prompt_0_5.json"

# Original data file (needed for db_id)
GOLD_DATA_FILE = "/data/hdd1/users/akouk/darelab_clean/DarelabDB/development/experimental_analysis_of_value_linking/OmniSQL/train_and_evaluate/data/value_linking/dev.json"

# Path to the directory containing all the SQLite databases
DB_PATH = "/data/hdd1/users/akouk/darelab_clean/DarelabDB/development/experimental_analysis_of_value_linking/OmniSQL/train_and_evaluate/data/value_linking/databases"

# Output file
OUTPUT_PATH = "/data/hdd1/users/akouk/darelab_clean/DarelabDB/test.json"

# --- Helper functions for SQL execution (adapted from evaluate_bird.py) ---

def execute_sql(sql, db_file):
    """Executes a single SQL query against a given database file."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")
        cursor.execute(sql)
        execution_res = cursor.fetchall()
        # Use frozenset to make the list of tuples hashable for dictionary keys
        execution_res = frozenset(execution_res)
        conn.rollback()
        conn.close()
        return sql, execution_res, 1  # sql, result, valid_flag
    except Exception:
        conn.rollback()
        conn.close()
        return sql, None, 0

def execute_sql_wrapper(args):
    """A wrapper for func_timeout to handle timeouts and exceptions."""
    sql, db_file, timeout = args
    try:
        return func_timeout(timeout, execute_sql, args=(sql, db_file))
    except FunctionTimedOut:
        return sql, "TIMEOUT", 0
    except Exception:
        return sql, "EXECUTION_ERROR", 0

def get_major_vote_outcome(sqls_to_execute, db_file, num_cpus=10, timeout=10):
    """
    Executes a list of SQL queries in parallel, performs major voting, and
    returns the winning SQL query AND its execution result.
    """
    pool = mp.Pool(processes=num_cpus)
    args_list = [(sql, db_file, timeout) for sql in sqls_to_execute]
    
    execution_results = pool.map(execute_sql_wrapper, args_list)
    pool.close()
    pool.join()

    valid_results = [res for res in execution_results if res[2] == 1]

    if not valid_results:
        return "ALL_QUERIES_FAILED", None

    major_voting_dict = {}
    for sql, result, _ in valid_results:
        if result in major_voting_dict:
            major_voting_dict[result]['votes'] += 1
        else:
            major_voting_dict[result] = {'votes': 1, 'sql': sql}
    
    if not major_voting_dict:
        return "ALL_QUERIES_FAILED", None

    # Find the execution result (the key) with the most votes
    major_vote_exec_result = max(major_voting_dict.keys(), key=lambda k: major_voting_dict[k]['votes'])
    winning_sql = major_voting_dict[major_vote_exec_result]['sql']
    
    # *** KEY CHANGE: Return both the winning SQL and its execution result ***
    return winning_sql, major_vote_exec_result


def compare_major_voting_outcomes():
    """
    Main function to load data, run execution-based major voting for each file,
    and save the differences based on execution results.
    """
    print("--- Starting Execution-Based Prediction Comparison ---")

    try:
        print(f"Loading file 1: {FILE_PATH_1}")
        with open(FILE_PATH_1, 'r', encoding='utf-8') as f: data_1 = json.load(f)
        
        print(f"Loading file 2: {FILE_PATH_0_5}")
        with open(FILE_PATH_0_5, 'r', encoding='utf-8') as f: data_0_5 = json.load(f)

        print(f"Loading gold data file for db_ids: {GOLD_DATA_FILE}")
        with open(GOLD_DATA_FILE, 'r', encoding='utf-8') as f: gold_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Please check paths. Details: {e}")
        return

    if not (len(data_1) == len(data_0_5) == len(gold_data)):
        print("Error: Files have different numbers of records. Cannot compare.")
        return

    differences = []
    print("\nProcessing and comparing major vote outcomes for each question...")
    
    for i in tqdm(range(len(gold_data)), desc="Comparing Questions"):
        db_id = gold_data[i]['db_id']
        db_file = os.path.join(DB_PATH, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_file):
            print(f"Warning: DB file not found for db_id '{db_id}'. Skipping question {i}.")
            continue

        sqls_1 = data_1[i].get("pred_sqls", [])
        sqls_0_5 = data_0_5[i].get("pred_sqls", [])

        # Get the winning SQL and its execution result for each file
        major_vote_sql_1, exec_res_1 = get_major_vote_outcome(sqls_1, db_file)
        major_vote_sql_0_5, exec_res_0_5 = get_major_vote_outcome(sqls_0_5, db_file)

        # *** KEY CHANGE: Compare the execution results, not the SQL text ***
        if exec_res_1 != exec_res_0_5:
            question = gold_data[i].get("question", "Question not found")
            if gold_data[i].get("evidence", "").strip():
                 question = gold_data[i]["evidence"] + "\n" + question

            differences.append({
                "question_index": i,
                "question": question,
                "db_id": db_id,
                "major_vote_sql_precision_1.0": major_vote_sql_1,
                "major_vote_sql_precision_0.5": major_vote_sql_0_5,
                # Optionally convert frozenset to list for JSON serialization if needed
                "execution_result_precision_1.0": list(exec_res_1) if exec_res_1 else None,
                "execution_result_precision_0.5": list(exec_res_0_5) if exec_res_0_5 else None,
            })

    print(f"\nFound {len(differences)} questions with different major voting outcomes.")

    try:
        print(f"Saving differences to: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(differences, f, indent=2, ensure_ascii=False)
        print("--- Comparison complete. Output saved successfully. ---")
    except IOError as e:
        print(f"Error: Could not write to the output file. Details: {e}")

if __name__ == "__main__":
    # Set start method for multiprocessing to avoid issues on some systems
    if sys.platform.startswith('darwin'): # macOS
        mp.set_start_method("spawn", force=True)
    else: # Linux
        mp.set_start_method("fork", force=True)
    compare_major_voting_outcomes()