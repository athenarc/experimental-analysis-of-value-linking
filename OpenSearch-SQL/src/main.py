import argparse
import json
from datetime import datetime
from typing import Any, Dict, List
import os
from tqdm import tqdm

from runner.task import Task
from llm.model import model_chose
from pipeline.batch_steps import (
    batch_generate_db_schema,
    batch_candidate_generate,
    batch_align_correct,
    batch_evaluation,
    save_per_query_histories
)

def load_dataset(data_path: str, start: int, end: int) -> List[Task]:
    """
    Loads the dataset from the specified path and creates Task objects.
    """
    with open(data_path, 'r') as file:
        dataset_json = json.load(file)
    
    tasks = []
    for i, data in enumerate(dataset_json):
        if i < start:
            continue
        if i >= end:
            break
        if "question_id" not in data:
            data = {"question_id": i, **data}
        task = Task(data)
        tasks.append(task)
    print(f"Loaded {len(tasks)} tasks from index {start} to {end}.")
    return tasks

def get_result_directory(args: Any) -> str:
    from pathlib import Path
    RESULT_ROOT_PATH = "results"
    data_mode = args.data_mode
    pipeline_nodes = args.pipeline_nodes
    dataset_name = Path(args.db_root_path).stem
    run_folder_name = str(args.run_start_time)
    run_folder_path = Path(RESULT_ROOT_PATH) / data_mode / pipeline_nodes / dataset_name / run_folder_name
    
    run_folder_path.mkdir(parents=True, exist_ok=True)
    
    arg_file_path = run_folder_path / "-args.json"
    with arg_file_path.open('w') as file:
        json.dump(vars(args), file, indent=4)
    
    log_folder_path = run_folder_path / "logs"
    log_folder_path.mkdir(exist_ok=True)
    
    return str(run_folder_path)

def main(args):
    """
    Main function to run the batch processing pipeline.
    """
    pipeline_setup = json.loads(args.pipeline_setup)
    db_json_path = os.path.join(args.db_root_path, 'data_preprocess', f'{args.data_mode}.json')
    
    tasks = load_dataset(db_json_path, args.start, args.end)
    
    result_dir = get_result_directory(args)

    engine_name = pipeline_setup["generate_db_schema"]["engine"]
    print(f"Initializing VLLM for model: {engine_name}")
    vllm_model = model_chose("batch_processing", engine_name)
    print("VLLM model initialized.")

    print("\n--- Step 1: Generating DB Schemas ---")
    db_schemas = batch_generate_db_schema(
        tasks=tasks, args=args, pipeline_setup=pipeline_setup, vllm_model=vllm_model, result_dir=result_dir
    )

    print("\n--- Step 2: Generating Candidate SQLs ---")
    candidate_results = batch_candidate_generate(
        tasks=tasks, db_schemas=db_schemas, args=args, pipeline_setup=pipeline_setup, vllm_model=vllm_model, result_dir=result_dir
    )

    print("\n--- Step 3: Aligning and Correcting SQLs ---")
    final_results = batch_align_correct(
        tasks=tasks, candidate_results=candidate_results, db_schemas=db_schemas, args=args, pipeline_setup=pipeline_setup, vllm_model=vllm_model, result_dir=result_dir
    )

    print("\n--- Step 4: Evaluating Results ---")
    evaluation_results = batch_evaluation(
        tasks=tasks, final_results=final_results, result_dir=result_dir, args=args
    )
    
    print("\n--- Step 5: Saving Per-Query JSON Histories ---")
    save_per_query_histories(
        final_results=final_results, evaluation_results=evaluation_results, result_dir=result_dir
    )

    print("\nPipeline finished successfully.")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    args_parser.add_argument('--db_root_path', type=str, required=True, help="Path to the data file.")
    args_parser.add_argument('--pipeline_nodes', type=str, required=True, help="Pipeline nodes configuration.")
    args_parser.add_argument('--pipeline_setup', type=str, required=True, help="Pipeline setup in JSON format.")
    args_parser.add_argument('--use_checkpoint', action='store_true', help="Flag to use checkpointing.")
    args_parser.add_argument('--checkpoint_nodes', type=str, required=False, help="Checkpoint nodes configuration.")
    args_parser.add_argument('--checkpoint_dir', type=str, required=False, help="Directory for checkpoints.")
    args_parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    args_parser.add_argument('--start', type=int, default=0, help="Start point")
    args_parser.add_argument('--end', type=int, default=1, help="End point")
    args = args_parser.parse_args()
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.use_checkpoint:
        print("Warning: Checkpointing is not supported in batch mode and will be ignored.")
    
    main(args)