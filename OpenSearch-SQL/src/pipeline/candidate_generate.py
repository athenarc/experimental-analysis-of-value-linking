import logging
from typing import Any, Dict, List
from pipeline.utils import node_decorator,get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from pipeline.utils import make_newprompt
from llm.model import model_chose
from llm.db_conclusion import *
import json
from llm.prompts import *
from runner.check_and_correct import get_sql, get_sql_batch

def candidate_generate_core(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Core logic for generating candidate SQL, callable directly."""
    task = state["keys"]["task"]
    execution_history = state["keys"]["execution_history"]
    paths = state["keys"]["db_manager"]
    
    fewshot_path=paths.db_fewshot_path

    with open(fewshot_path) as f:
        df_fewshot = json.load(f)

    prev_node_result = get_last_node_result(execution_history, "column_retrieve_and_other_info")
    L_values = prev_node_result["L_values"]
    oracle_precision = prev_node_result.get("oracle_precision")
    
    column = prev_node_result["column"]
    foreign_keys = prev_node_result["foreign_keys"]
    q_order = prev_node_result["q_order"]
    
    db=task.db_id
    
    if oracle_precision is not None:
        formatted_values = [f"{tc or 'unknown'}: '{v}'" for tc, v in L_values]
        values_str = "\n".join(formatted_values)
        if oracle_precision == 1.0:
            key_col_des = f"#Important Oracle Value:\n...use this exact value...\n{values_str}"
        else:
            key_col_des = f"#Important Oracle Values:\n...use the correct one...\n{values_str}"
    else:
        values = [f"{x[0]}: '{x[1]}'" for x in L_values]
        key_col_des = "#Values in Database:\n" + '\n'.join(values)

    new_db_info = f"Database Management System: SQLite\n#Database name: {db} \n{column}\n\n#Forigen keys:\n{foreign_keys}\n"
    
    question=task.question
    fewshot=df_fewshot["questions"][task.question_id]['prompt']
    
    new_prompt = make_newprompt(db_check_prompts().new_prompt, fewshot,
                            key_col_des, new_db_info, question,
                            task.evidence,q_order)

    # This function now just returns the prompt and other necessary info
    return {
        "prompt": new_prompt,
        "rewrite_question": question
    }

# +++ THE LANGGRAPH NODE IS NOW A WRAPPER +++
@node_decorator(check_schema_status=False)
def candidate_generate(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    config,node_name=PipelineManager().get_model_para()
    chat_model = model_chose(node_name,config["engine"])

    # 1. Get the prompt from the core logic
    core_result = candidate_generate_core(task, execution_history, config)
    prompt = core_result["prompt"]

    # 2. Call the single-prompt LLM function
    single = config['single'].lower() == 'true'
    return_question=config['return_question']== 'true' 
    SQL,_ = get_sql(chat_model, prompt, config['temperature'], return_question=return_question,n=config['n'],single=single)
    
    # 3. Format the response
    response = {
        "rewrite_question": core_result["rewrite_question"],
        "SQL": SQL
    }
    return response


def rewrite_question(question):
    if question.find(" / ")!=-1:
        question+=". For division operations, use CAST xxx AS REAL to ensure precise decimal results"
    return question
