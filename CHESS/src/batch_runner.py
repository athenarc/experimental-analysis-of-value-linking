# src/batch_runner.py

import json
from tqdm import tqdm
from typing import List, Dict, Any
import time

from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from workflow.system_state import SystemState
from workflow.sql_meta_info import SQLMetaInfo
from llm.vllm_manager import VLLMManager
from llm.models import get_llm_chain
from llm.prompts import get_prompt
from llm.parsers import get_parser
from runner.logger import Logger

# Import tool classes and the constant
from workflow.agents.information_retriever.tool_kit.retrieve_entity import RetrieveEntity
# REMOVED: from workflow.agents.information_retriever.tool_kit.retrieve_context import RetrieveContext
from workflow.agents.candidate_generator.tool_kit.generate_candidate import GenerateCandidate
from workflow.agents.unit_tester.tool_kit.generate_unit_test import GenerateUnitTest, HARD_CODES_TEST_CASES
from workflow.agents.unit_tester.tool_kit.evaluate import Evaluate
from workflow.agents.evaluation import ExecutionAccuracy

def run_and_log_tool(tool, state):
    """Runs a tool, measures execution time, and logs the result to the state's history."""
    start_time = time.time()
    try:
        tool._run(state)
        status = "success"
    except Exception as e:
        status = "error"
        print(f"ERROR: (Task: {state.task.db_id}, {state.task.question_id}) Tool '{tool.tool_name}'\n{type(e)}: {e}\n")

    execution_time = round(time.time() - start_time, 1)
    
    updates = tool._get_updates(state)
    
    run_log = {
        "tool_name": tool.tool_name,
        **updates,
        "status": status,
        "execution_time": execution_time
    }
    state.execution_history.append(run_log)

def run_batch_pipeline(tasks: List[Task], config: Dict[str, Any], result_directory: str, data_mode: str):
    """
    Runs the entire pipeline in a staged, cross-query batch mode.
    """
    stats_manager = StatisticsManager(result_directory)

    # --- STAGE 1: Information Retrieval (CPU Bound) ---
    print("--- Stage 1: Information Retrieval ---")
    system_states = []
    retrieve_entity_tool = RetrieveEntity()
    # REMOVED: retrieve_context_tool = RetrieveContext(...)

    for task in tqdm(tasks, desc="Preparing initial states"):
        Logger(db_id=task.db_id, question_id=str(task.question_id), result_directory=result_directory)
        DatabaseManager(db_mode=data_mode, db_id=task.db_id)
        state = SystemState(task=task, tentative_schema=DatabaseManager().get_db_schema(), execution_history=[])
        
        # Only run the entity tool. The schema generator will handle descriptions implicitly.
        run_and_log_tool(retrieve_entity_tool, state)
        
        system_states.append(state)

    # --- STAGE 2: Candidate Generation (GPU Bound) ---
    print("\n--- Stage 2: Candidate Generation ---")
    cg_config = config["team_agents"]["candidate_generator"]["tools"]["generate_candidate"]
    gen_config = cg_config["generator_configs"][0]
    cg_tool = GenerateCandidate(**cg_config)
    cg_prompt = get_prompt(template_name=gen_config["template_name"])
    cg_parser = get_parser(parser_name=gen_config["parser_name"])
    sampling_count = gen_config["sampling_count"]

    all_cg_prompts = []
    for state in system_states:
        # Set DB context for the schema generator to find the right CSVs
        DatabaseManager(db_mode=data_mode, db_id=state.task.db_id)
        request_kwargs = {
            "DATABASE_SCHEMA": state.get_schema_string(schema_type="complete"),
            "QUESTION": state.task.question,
            "HINT": state.task.evidence,
        }
        formatted_prompt = cg_prompt.invoke(request_kwargs).to_string()
        all_cg_prompts.extend([formatted_prompt] * sampling_count)

    start_time_cg = time.time()
    sampling_params_cg = {"temperature": gen_config["engine_config"]["temperature"], "max_tokens": 2048}
    cg_outputs = VLLMManager.generate(all_cg_prompts, sampling_params_cg)
    cg_execution_time = round(time.time() - start_time_cg, 1)

    output_idx = 0
    for state in tqdm(system_states, desc="Updating states with candidates"):
        task_specific_cg_tool = GenerateCandidate(**cg_config)
        
        for gen_conf in task_specific_cg_tool.generator_configs:
            task_specific_cg_tool.generators_queries[gen_conf.template_name] = []

        for _ in range(sampling_count):
            if output_idx < len(cg_outputs):
                try:
                    parsed_output = cg_parser.parse(cg_outputs[output_idx])
                    sql_meta_info = SQLMetaInfo(**parsed_output)
                    task_specific_cg_tool.generators_queries[gen_config["template_name"]].append(sql_meta_info)
                except Exception as e:
                    print(f"Warning: Failed to parse candidate for task {state.task.question_id}. Error: {e}")
            output_idx += 1
        
        state.SQL_meta_infos[cg_tool.tool_name] = task_specific_cg_tool.generators_queries[gen_config["template_name"]]
        
        updates = task_specific_cg_tool._get_updates(state)
        state.execution_history.append({
            "tool_name": cg_tool.tool_name, **updates, "status": "success", "execution_time": cg_execution_time / len(system_states)
        })

    # --- Stages 3 & 4 & 5 remain the same as the last correct version ---
    # ...
    # (The rest of the file is identical to the previous version)
    # ...
    print("\n--- Stage 3: Unit Test Generation (Batched) ---")
    ut_gen_config = config["team_agents"]["unit_tester"]["tools"]["generate_unit_test"]
    ut_gen_tool = GenerateUnitTest(**ut_gen_config)
    ut_gen_prompt = get_prompt(template_name=ut_gen_tool.template_name)
    ut_gen_parser = get_parser(parser_name=ut_gen_tool.parser_name)
    
    ut_gen_prompts = []
    states_needing_tests_indices = [] 
    
    for i, state in enumerate(system_states):
        target_SQLs = state.SQL_meta_infos.get(cg_tool.tool_name, [])
        if not target_SQLs: continue
        DatabaseManager(db_mode=data_mode, db_id=state.task.db_id)
        clusters = ut_gen_tool.execution_based_clustering(target_SQLs)
        if len(clusters) > 1:
            states_needing_tests_indices.append(i)
            
            formatted_candidates = ""
            index = 0
            for key, candidate_queries in clusters.items():
                formatted_candidates += f"Cluster #{index+1}: \n"
                for candidate_query in candidate_queries:
                    formatted_candidates += f"Query: {candidate_query.SQL}\n"
                formatted_candidates += "########\n"
                formatted_candidates += f"Execution result: {ut_gen_tool._format_sql_query_result(candidate_queries[-1])}\n"
                formatted_candidates += "=====================\n"
                index += 1

            request_kwargs = {
                "HINT": state.task.evidence,
                "QUESTION": state.task.question,
                "DATABASE_SCHEMA": state.get_database_schema_for_queries([sql.SQL for sql in target_SQLs]),
                "CANDIDATE_QUERIES": formatted_candidates,
                "UNIT_TEST_CAP": ut_gen_tool.unit_test_count
            }
            formatted_prompt = ut_gen_prompt.invoke(request_kwargs).to_string()
            ut_gen_prompts.extend([formatted_prompt] * ut_gen_tool.sampling_count)

    if ut_gen_prompts:
        start_time_ut_gen = time.time()
        sampling_params_ut_gen = {"temperature": ut_gen_tool.engine_config["temperature"], "max_tokens": 1024}
        ut_gen_outputs = VLLMManager.generate(ut_gen_prompts, sampling_params_ut_gen)
        ut_gen_execution_time = round(time.time() - start_time_ut_gen, 1)
        
        output_idx = 0
        for state_idx in tqdm(states_needing_tests_indices, desc="Updating states with unit tests"):
            state = system_states[state_idx]
            state.unit_tests["unit_test_generation"] = []
            for _ in range(ut_gen_tool.sampling_count):
                if output_idx < len(ut_gen_outputs):
                    try:
                        parsed = ut_gen_parser.parse(ut_gen_outputs[output_idx])
                        if parsed and 'unit_tests' in parsed:
                            state.unit_tests["unit_test_generation"].extend(parsed['unit_tests'])
                    except Exception as e:
                        print(f"Warning: Failed to parse unit tests for task {state.task.question_id}. Error: {e}")
                output_idx += 1
            state.unit_tests["unit_test_generation"].extend(HARD_CODES_TEST_CASES)
            
            updates = ut_gen_tool._get_updates(state)
            state.execution_history.append({
                "tool_name": ut_gen_tool.tool_name, **updates, "status": "success", "execution_time": ut_gen_execution_time / len(states_needing_tests_indices)
            })

    for i, state in enumerate(system_states):
        if i not in states_needing_tests_indices:
            state.unit_tests["unit_test_generation"] = []
            updates = ut_gen_tool._get_updates(state)
            state.execution_history.append({
                "tool_name": ut_gen_tool.tool_name, **updates, "status": "skipped", "execution_time": 0.0
            })

    print("\n--- Stage 4: Evaluation (Batched) ---")
    ut_eval_config = config["team_agents"]["unit_tester"]["tools"]["evaluate"]
    ut_eval_tool = Evaluate(**ut_eval_config)
    ut_eval_prompt = get_prompt(template_name=ut_eval_tool.template_name)
    ut_eval_parser = get_parser(parser_name=ut_eval_tool.parser_name)

    all_eval_prompts = []
    states_for_evaluation_indices = []
    prompt_counts_per_state = []

    for i, state in enumerate(system_states):
        target_SQLs = state.SQL_meta_infos.get(cg_tool.tool_name, [])
        unit_tests = state.unit_tests.get("unit_test_generation", [])
        
        if len(target_SQLs) > 1 and unit_tests:
            states_for_evaluation_indices.append(i)
            prompt_counts_per_state.append(len(unit_tests))
            DatabaseManager(db_mode=data_mode, db_id=state.task.db_id)
            
            formatted_candidates = ""
            for idx, candidate in enumerate(target_SQLs):
                formatted_candidates += f"Candidate Response #{idx+1}: Query: {candidate.SQL}\n, Execution Result: {ut_eval_tool._format_sql_query_result(candidate)}\n"
            
            database_schema = state.get_database_schema_for_queries([sql.SQL for sql in target_SQLs])
            
            for unit_test in unit_tests:
                request_kwargs = {
                    "DATABASE_SCHEMA": database_schema,
                    "QUESTION": state.task.question,
                    "HINT": state.task.evidence,
                    "CANDIDATE_RESPONSES": formatted_candidates,
                    "UNIT_TEST": unit_test
                }
                formatted_prompt = ut_eval_prompt.invoke(request_kwargs).to_string()
                all_eval_prompts.append(formatted_prompt)

    if all_eval_prompts:
        start_time_eval = time.time()
        sampling_params_eval = {"temperature": ut_eval_tool.engine_config["temperature"], "max_tokens": 512}
        eval_outputs = VLLMManager.generate(all_eval_prompts, sampling_params_eval)
        eval_execution_time = round(time.time() - start_time_eval, 1)
        
        output_idx = 0
        for i, state_idx in enumerate(tqdm(states_for_evaluation_indices, desc="Updating states with evaluations")):
            state = system_states[state_idx]
            target_SQLs = state.SQL_meta_infos.get(cg_tool.tool_name, [])
            num_prompts_for_this_state = prompt_counts_per_state[i]
            
            comparison_matrix = []
            for _ in range(num_prompts_for_this_state):
                if output_idx < len(eval_outputs):
                    try:
                        parsed = ut_eval_parser.parse(eval_outputs[output_idx])
                        if parsed and "scores" in parsed and len(parsed["scores"]) == len(target_SQLs):
                           comparison_matrix.append(parsed["scores"])
                    except Exception as e:
                        print(f"Warning: Failed to parse evaluation for task {state.task.question_id}. Error: {e}")
                output_idx += 1
            
            key_to_evaluate = list(state.SQL_meta_infos.keys())[-1]
            if key_to_evaluate.startswith(ut_eval_tool.tool_name):
                id_val = int(key_to_evaluate.split('_')[-1])
                sql_id = f"{ut_eval_tool.tool_name}_{id_val + 1}"
            else:
                sql_id = f"{ut_eval_tool.tool_name}_1"
            ut_eval_tool.SQL_id = sql_id

            if comparison_matrix:
                scores = [sum(col) for col in zip(*comparison_matrix)]
                clusters = ut_eval_tool.execution_based_clustering(target_SQLs)
                best_candidate = ut_eval_tool.pick_the_best_candidate(scores, target_SQLs, clusters)
                state.SQL_meta_infos[sql_id] = [best_candidate]
            else:
                state.SQL_meta_infos[sql_id] = [target_SQLs[0]]
            
            updates = ut_eval_tool._get_updates(state)
            state.execution_history.append({
                "tool_name": ut_eval_tool.tool_name, **updates, "status": "success", "execution_time": eval_execution_time / len(states_for_evaluation_indices)
            })
    
    for i, state in enumerate(system_states):
        if i not in states_for_evaluation_indices:
            sql_id_key = ut_eval_tool.tool_name + "_1"
            target_SQLs = state.SQL_meta_infos.get(cg_tool.tool_name, [])
            if target_SQLs:
                state.SQL_meta_infos[sql_id_key] = [target_SQLs[0]]
            else:
                state.SQL_meta_infos[sql_id_key] = [SQLMetaInfo(SQL="--NO CANDIDATE GENERATED--")]
            
            ut_eval_tool.SQL_id = sql_id_key
            updates = ut_eval_tool._get_updates(state)
            state.execution_history.append({
                "tool_name": ut_eval_tool.tool_name, **updates, "status": "skipped", "execution_time": 0.0
            })

    print("\n--- Stage 5: Final Evaluation ---")
    eval_tool = ExecutionAccuracy()
    for state in tqdm(system_states, desc="Running final evaluation"):
        Logger(db_id=state.task.db_id, question_id=str(state.task.question_id), result_directory=result_directory)
        DatabaseManager(db_mode=data_mode, db_id=state.task.db_id)
        
        run_and_log_tool(eval_tool, state)
        
        evaluation_results = state.execution_history[-1]
        final_sql_result = evaluation_results.get("final_SQL")

        if final_sql_result:
            stats_manager.update_stats(state.task.db_id, str(state.task.question_id), "final_SQL", final_sql_result)
        else:
            error_result = {"exec_res": "error", "exec_err": "No final SQL selected"}
            stats_manager.update_stats(state.task.db_id, str(state.task.question_id), "final_SQL", error_result)
        
        Logger().dump_history_to_file(state.execution_history)

    stats_manager.dump_statistics_to_file()
    print("\nBatch processing complete.")