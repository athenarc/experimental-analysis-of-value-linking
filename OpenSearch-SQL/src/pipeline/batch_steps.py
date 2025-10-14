import json
import os
import sqlite3
from tqdm import tqdm
from typing import List, Dict, Any
import pandas as pd

from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from llm.db_conclusion import db_agent_string, find_foreign_keys_MYSQL_like
from llm.prompts import db_check_prompts
from runner.column_update import ColumnUpdater
from runner.check_and_correct import soft_check, sql_raw_parse, retable
from pipeline.utils import make_newprompt
from sentence_transformers import SentenceTransformer
from runner.logger import Logger


def batch_generate_db_schema(tasks: List[Task], args: Any, pipeline_setup: Dict, vllm_model: Any, result_dir: str) -> Dict[str, Dict]:
    config = pipeline_setup["generate_db_schema"]
    bert_model = SentenceTransformer(config["bert_model"], device=config["device"])
    
    unique_db_ids = sorted(list(set(task.db_id for task in tasks)))
    print(f"Found {len(unique_db_ids)} unique databases to process.")

    db_schemas = {}
    
    db_info_agent = db_agent_string(vllm_model)
    prompts = []
    db_info_map = {}

    for db_id in tqdm(unique_db_ids, desc="Preparing DB Schema Prompts"):
        paths = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=db_id)
        db_info, db_col = db_info_agent.get_db_des(paths.db_path, paths.db_directory_path, bert_model)
        foreign_keys, _ = find_foreign_keys_MYSQL_like(paths.db_tables, db_id)
        
        all_info_str = f"Database Management System: SQLite\n#Database name: {db_id}\n{db_info}\n#Forigen keys:\n{foreign_keys}\n"
        prompt = db_info_agent.db_conclusion(all_info_str)
        
        prompts.append(prompt)
        db_info_map[db_id] = {"db_info_str": all_info_str, "db_col": db_col}

    print(f"Sending {len(prompts)} prompts to VLLM for DB summarization...")
    summaries = vllm_model.batch_generate(prompts, temperature=0.0)

    for i, db_id in enumerate(unique_db_ids):
        summary = summaries[i][0]
        full_schema = f"{db_info_map[db_id]['db_info_str']}\n{summary}\n"
        db_schemas[db_id] = {
            "db_list": full_schema,
            "db_col_dic": db_info_map[db_id]['db_col']
        }
        # Log the conversation for this specific DB schema generation
        logger = Logger(db_id=db_id, question_id="schema_generation", result_directory=result_dir)
        logger.log_conversation(prompts[i], "Human", "generate_db_schema")
        logger.log_conversation(summary, "AI", "generate_db_schema")
        
    print("DB Schema generation complete.")
    return db_schemas


def batch_candidate_generate(tasks: List[Task], db_schemas: Dict, args: Any, pipeline_setup: Dict, vllm_model: Any, result_dir: str) -> List[Dict]:
    config = pipeline_setup["candidate_generate"]
    prompts_template = db_check_prompts()
    
    paths = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=tasks[0].db_id)
    with open(paths.db_fewshot_path) as f:
        df_fewshot = json.load(f)

    prompts = []
    candidate_results = []

    for task in tqdm(tasks, desc="Preparing Candidate Generation Prompts"):
        db_id = task.db_id
        schema_info = db_schemas[db_id]
        
        db_col = {x: schema_info["db_col_dic"][x][0] for x in schema_info["db_col_dic"]}
        column_formatted = ColumnUpdater(db_col).col_suffix(set(db_col.keys()))
        foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(paths.db_tables, db_id)
        
        new_db_info = f"Database Management System: SQLite\n#Database name: {db_id} \n{column_formatted}\n\n#Forigen keys:\n{foreign_keys}\n"
        
        fewshot = df_fewshot["questions"][task.question_id]['prompt']
        
        prompt = make_newprompt(
            prompts_template.new_prompt, fewshot, "", new_db_info, task.question, task.evidence, ""
        )
        prompts.append(prompt)
        
        candidate_results.append({
            "task": task,
            "prompt": prompt,
            "schema_info": schema_info,
            "simplified_info": {
                "column": column_formatted,
                "foreign_keys": foreign_keys,
                "foreign_set": foreign_set,
                "L_values": [],
                "q_order": ""
            },
            "db_info_for_correction": new_db_info,
            "db_col_for_correction": db_col,
        })

    print(f"Sending {len(prompts)} prompts to VLLM for candidate generation...")
    outputs = vllm_model.batch_generate(
        prompts,
        temperature=config['temperature'],
        n=config['n']
    )

    for i, result_list in enumerate(outputs):
        task = candidate_results[i]["task"]
        prompt = candidate_results[i]["prompt"]
        sqls = [sql_raw_parse(text, return_question=False)[0] for text in result_list]
        candidate_results[i]["SQLs"] = sqls
        candidate_results[i]["rewrite_question"] = task.question
        
        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=result_dir)
        logger.log_conversation(prompt, "Human", "candidate_generate")
        logger.log_conversation(result_list, "AI", "candidate_generate")


    print("Candidate generation complete.")
    return candidate_results


def batch_align_correct(tasks: List[Task], candidate_results: List[Dict], db_schemas: Dict, args: Any, pipeline_setup: Dict, vllm_model: Any, result_dir: str) -> List[Dict]:
    config = pipeline_setup["align_correct"]
    prompts_template = db_check_prompts()
    bert_model = SentenceTransformer(config["bert_model"], device=config["device"])

    paths = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=tasks[0].db_id)
    with open(paths.db_fewshot2_path) as f:
        correct_dic = json.load(f)

    final_results = []
    d_check = soft_check(bert_model, vllm_model, prompts_template.soft_prompt, correct_dic, prompts_template.correct_prompt, prompts_template.vote_prompt)

    for i, res in enumerate(tqdm(candidate_results, desc="Aligning SQLs (local)")):
        task = res['task']
        db_sqlite_path = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=task.db_id).db_path
        
        aligned_sqls = []
        for sql in res["SQLs"]:
            sql_retable = retable(sql)
            sql, _ = d_check.double_check_style_align(sql, task.question, res['db_col_for_correction'].keys(), sql_retable)
            sql, _ = d_check.double_check_function_align(sql, task.question, db_sqlite_path)
            sql, _ = d_check.double_check_agent_align(sql_retable, [], [], sql, task.question, res['db_info_for_correction'], res['db_col_for_correction'].keys(), "")
            aligned_sqls.append(sql)
        res["aligned_SQLs"] = aligned_sqls

    correction_prompts = []
    prompt_indices = []
    
    for i, res in enumerate(tqdm(candidate_results, desc="Executing and Preparing Correction")):
        task = res['task']
        db_sqlite_path = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=task.db_id).db_path
        
        sql_to_check = res["aligned_SQLs"][0]
        
        try:
            with sqlite3.connect(db_sqlite_path) as conn:
                pd.read_sql_query(sql_to_check, conn)
            res["corrected_sql"] = sql_to_check
        except Exception as e:
            e_s = str(e).split("':")[-1]
            result_info = f"{sql_to_check}\nError: {e_s}"
            
            cor_prompt = prompts_template.correct_prompt.format(
                fewshot="",
                db_info=res['db_info_for_correction'],
                key_col_des="",
                q=task.question,
                hint=task.evidence,
                result_info=result_info,
                advice=""
            )
            correction_prompts.append(cor_prompt)
            prompt_indices.append(i)
            res["correction_prompt"] = cor_prompt

    if correction_prompts:
        print(f"Sending {len(correction_prompts)} prompts to VLLM for correction...")
        corrected_outputs = vllm_model.batch_generate(correction_prompts, temperature=0.2)
        
        for i, output in enumerate(corrected_outputs):
            task_index = prompt_indices[i]
            task = candidate_results[task_index]["task"]
            corrected_sql = sql_raw_parse(output[0], return_question=False)[0]
            candidate_results[task_index]["corrected_sql"] = corrected_sql
            
            logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=result_dir)
            logger.log_conversation(correction_prompts[i], "Human", "align_correct")
            logger.log_conversation(output[0], "AI", "align_correct")


    for res in tqdm(candidate_results, desc="Aggregating Final Results"):
        final_sql = res.get("corrected_sql", res["aligned_SQLs"][0])
        # Carry over all previous information and add new info
        final_result_item = res.copy()
        final_result_item["final_sql"] = final_sql
        final_results.append(final_result_item)
        
    print("Alignment and correction complete.")
    return final_results


def batch_evaluation(tasks: List[Task], final_results: List[Dict], result_dir: str, args: Any) -> List[Dict]:
    stats_manager = StatisticsManager(result_dir)
    
    results_to_save = {
        "final_voted": {},
        "raw_candidate_1": {}
    }
    
    evaluation_details = []

    for i, task in enumerate(tqdm(tasks, desc="Evaluating")):
        ground_truth_sql = task.SQL
        db_manager = DatabaseManager(db_mode=args.data_mode, db_root_path=args.db_root_path, db_id=task.db_id)
        
        task_eval_details = {}

        # Evaluate final corrected SQL
        predicted_sql = final_results[i]["final_sql"]
        response = db_manager.compare_sqls(
            predicted_sql=predicted_sql, ground_truth_sql=ground_truth_sql, meta_time_out=180
        )
        stats_manager.update_stats(task.db_id, str(task.question_id), "final_voted", response)
        results_to_save["final_voted"][task.question_id] = predicted_sql
        task_eval_details["vote"] = {**response, "SQL": predicted_sql} # Using 'vote' to match original node name

        # Evaluate the first raw candidate
        raw_candidate_sql = final_results[i]["SQLs"][0]
        response_raw = db_manager.compare_sqls(
            predicted_sql=raw_candidate_sql, ground_truth_sql=ground_truth_sql, meta_time_out=180
        )
        stats_manager.update_stats(task.db_id, str(task.question_id), "raw_candidate_1", response_raw)
        results_to_save["raw_candidate_1"][task.question_id] = raw_candidate_sql
        task_eval_details["candidate_generate"] = {**response_raw, "SQL": raw_candidate_sql}
        
        evaluation_details.append(task_eval_details)

    stats_manager.dump_statistics_to_file()
    print(f"Evaluation complete. Statistics saved to {stats_manager.statistics_file_path}")

    for key, sqls in results_to_save.items():
        with open(os.path.join(result_dir, f"-{key}.json"), 'w') as f:
            json.dump(sqls, f, indent=4, ensure_ascii=False)
    print("SQL files for different stages saved.")
    return evaluation_details

def save_per_query_histories(final_results: List[Dict], evaluation_results: List[Dict], result_dir: str):
    print("Assembling and saving per-query execution history JSONs...")
    for i, res in enumerate(tqdm(final_results, desc="Saving JSON histories")):
        task = res["task"]
        eval_res = evaluation_results[i]
        
        execution_history = []

        # 1. generate_db_schema
        execution_history.append({
            "node_type": "generate_db_schema",
            "status": "success",
            **res["schema_info"]
        })

        # 2. simplified_info_gathering
        execution_history.append({
            "node_type": "simplified_info_gathering",
            "status": "success",
            **res["simplified_info"]
        })

        # 3. candidate_generate
        execution_history.append({
            "node_type": "candidate_generate",
            "status": "success",
            "rewrite_question": res["rewrite_question"],
            "SQL": res["SQLs"]
        })

        # 4. align_correct
        execution_history.append({
            "node_type": "align_correct",
            "status": "success",
            "vote": [{
                "sql_history": {"style_align": s, "function_align": s, "agent_align": s},
                "sql": res.get("corrected_sql", s),
                "count": 1
            } for s in res["aligned_SQLs"]],
            "none_case": False
        })

        # 5. vote
        execution_history.append({
            "node_type": "vote",
            "status": "success",
            "SQL": res["final_sql"],
            "nonecase": False
        })

        # 6. evaluation
        eval_node_result = {
            "node_type": "evaluation",
            "status": "success",
            "candidate_generate": {
                "exec_res": eval_res["candidate_generate"]["exec_res"],
                "exec_err": eval_res["candidate_generate"]["exec_err"],
                "PREDICTED_SQL": eval_res["candidate_generate"]["SQL"],
                "GOLD_SQL": task.SQL
            },
            "vote": {
                "exec_res": eval_res["vote"]["exec_res"],
                "exec_err": eval_res["vote"]["exec_err"],
                "PREDICTED_SQL": eval_res["vote"]["SQL"],
                "GOLD_SQL": task.SQL
            }
        }
        execution_history.append(eval_node_result)

        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=result_dir)
        logger.dump_history_to_file(execution_history)