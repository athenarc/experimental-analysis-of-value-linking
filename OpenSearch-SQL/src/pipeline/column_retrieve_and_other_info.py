import logging,re,json,random
from typing import Any, Dict, List, Tuple
from pathlib import Path
from pipeline.utils import node_decorator,get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.model import model_chose
from llm.db_conclusion import find_foreign_keys_MYSQL_like
from llm.prompts import *
from runner.extract import DES_new
from database_process.make_emb import load_emb
from runner.column_retrieve import ColumnRetriever
from runner.column_update import ColumnUpdater

def _get_oracle_values(task: Any, runtime_args: Any) -> Tuple[List, float]:
    """Helper to load and prepare oracle values if the path is provided."""
    oracle_values_path = runtime_args.oracle_values_path
    precision = max(0.01, min(1.0, runtime_args.oracle_precision))

    logging.info(f"Oracle mode enabled. Loading from: {oracle_values_path} with precision {precision}")
    
    with open(oracle_values_path, 'r') as f:
        oracle_data = json.load(f)
    
    matched_record = next((record for record in oracle_data if record.get("question") == task.raw_question), None)

    if not matched_record:
        logging.error(f"Could not find a matching question in the oracle file for: {task.raw_question}")
        return None, precision

    correct_value = matched_record.get("correct_value")
    random_values = matched_record.get("random_values", [])
    
    if not correct_value:
        logging.warning("Matching question found, but 'correct_value' is missing.")
        return None, precision

    values_for_prompt = [(None, correct_value)]

    if precision < 1.0:
        num_false_positives = int(round((1.0 / precision) - 1.0))
        if num_false_positives > 0:
            distractors_to_add = random.sample(random_values, min(num_false_positives, len(random_values)))
            for val_str in distractors_to_add:
                try:
                    table_col, value = val_str.rsplit('.', 1)
                    values_for_prompt.append((table_col, value))
                except ValueError:
                    logging.warning(f"Could not parse oracle random_value: {val_str}")

    random.shuffle(values_for_prompt)
    logging.info(f"Prepared {len(values_for_prompt)} oracle values for the prompt.")
    return values_for_prompt, precision

@node_decorator(check_schema_status=False)
def column_retrieve_and_other_info(state: Dict[str, Any]) -> Dict[str, Any]:
    task = state["keys"]["task"]
    execution_history = state["keys"]["execution_history"]
    paths = state["keys"]["db_manager"]
    paths = state["keys"]["db_manager"]
    
    config,node_name=PipelineManager().get_model_para()
    runtime_args = PipelineManager().get_runtime_args()
    
    # --- THIS LOGIC IS NOW INTEGRATED, NOT A FULL BYPASS ---
    
    # --- Schema and Column Analysis (ALWAYS RUNS) ---
    bert_model = PipelineManager().get_bert_model()
    tables_info_dir=paths.db_tables
    chat_model = model_chose(node_name,config["engine"])

    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"]
    origin_col = get_last_node_result(execution_history, "extract_query_noun")["col"]
    values = get_last_node_result(execution_history, "extract_query_noun")["values"]
    db=task.db_id

    db_col = {x: all_db_col[x][0] for x in all_db_col }
    db_keys_col=all_db_col.keys()

    col_retrieve = ColumnRetriever(bert_model,tables_info_dir).get_col_retrieve(task.question, db,db_keys_col)
    foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(tables_info_dir, db)      
    cols=ColumnUpdater(db_col).col_pre_update(origin_col,col_retrieve,foreign_set)

    # --- Value Grounding (Conditional: Oracle vs. Dense Search) ---
    L_values = []
    oracle_precision = None
    
    if runtime_args.oracle_values_path:
        L_values, oracle_precision = _get_oracle_values(task, runtime_args)
        if L_values is None: L_values = [] # Handle case where oracle lookup fails
        # Even with oracle values, we still need to determine the relevant columns for the schema
        cols_select = cols 
    else:
        # Original dense search logic
        emb_dir=paths.emb_dir
        DB_emb, col_values = load_emb(db, emb_dir)
        des = DES_new(bert_model, DB_emb, col_values)   
        cols_select, L_values = des.get_key_col_des(cols,
                                        values,
                                        debug=False,
                                        topk=config['top_k'],
                                        shold=0.65)

    # --- Final Formatting (ALWAYS RUNS) ---
    column=ColumnUpdater(db_col).col_suffix(cols_select)
    
    q_order = []
    try:
        q_order=query_order(task.raw_question,chat_model,db_check_prompts().select_prompt,temperature=config['temperature'])
    except Exception as e:
        logging.warning(f"Failed to get query order: {e}")

    response = {
        "L_values": L_values,
        "oracle_precision": oracle_precision,
        "column": column,
        "foreign_keys": foreign_keys,
        "foreign_set": foreign_set,
        "q_order": q_order
    }

    return response

# Helper functions query_order and json_ext remain the same
def query_order(question, chat_model, select_prompt,temperature):
    select_prompt = select_prompt.format(question=question)
    ans = chat_model.get_ans(select_prompt, temperature=temperature)
    ans = re.sub("```json|```", "", ans)
    select_json = json.loads(ans)
    res, judge = json_ext(select_json)
    return res

def json_ext(jsonf):
    ans = []
    judge = False
    for x in jsonf:
        if x["Type"] == "QIC":
            Q = x["Extract"]["Q"].lower()
            if Q in ["how many", "how much", "which","how often"]:
                for item in x["Extract"]["I"]:
                    ans.append(x["Extract"]["Q"] + " " + item)
            elif Q in ["when", "who", "where"]:
                ans.append(x["Extract"]["Q"])
            else:
                ans.extend(x["Extract"]["I"])
        elif x["Type"] == "JC":
            ans.append(x["Extract"]["J"])
            judge = True
    return ans, judge