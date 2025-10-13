# src/main.py

import argparse
import yaml
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from runner.run_manager import RunManager
from runner.task import Task # Import Task
from llm.vllm_manager import VLLMManager # Import VLLMManager

# --- NEW IMPORT ---
from batch_runner import run_batch_pipeline

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration.")
    parser.add_argument('--data_mode', type=str, required=True, help="Mode of the data to be processed.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for agentic mode.")
    parser.add_argument('--log_level', type=str, default='warning', help="Logging level.")
    
    # --- MODIFIED/NEW ARGUMENTS ---
    parser.add_argument('--runner_mode', type=str, default='agentic', choices=['agentic', 'batch'], 
                        help="Execution mode: 'agentic' (one-by-one) or 'batch' (cross-query batching).")
    parser.add_argument('--vllm_model_path', type=str, default=None, 
                        help="Path to the model for vLLM (required for batch mode).")
    
    # Deprecate old flags, they are implied by runner_mode
    # parser.add_argument('--use_vllm_batch', action='store_true', ...)
    
    args = parser.parse_args()

    args.run_start_time = datetime.now().isoformat()
    with open(args.config, 'r') as file:
        args.config=yaml.safe_load(file)
    
    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return [Task(**data) for data in dataset]

def get_result_directory(args: argparse.Namespace) -> str:
    """
    Creates and returns the result directory path.
    (This is refactored from RunManager to be shared)
    """
    data_mode = args.data_mode
    setting_name = args.config["setting_name"]
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    run_folder_name = str(args.run_start_time)
    run_folder_path = os.path.join("results", data_mode, setting_name, dataset_name, run_folder_name)
    
    os.makedirs(run_folder_path, exist_ok=True)
    
    arg_file_path = os.path.join(run_folder_path, "-args.json")
    # Convert args to dict for JSON serialization
    args_dict = vars(args)
    args_dict['config'] = args.config # Ensure config is included
    with open(arg_file_path, 'w') as file:
        json.dump(args_dict, file, indent=4)
    
    return run_folder_path

def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    dataset = load_dataset(args.data_path)
    result_directory = get_result_directory(args)

    if args.runner_mode == 'batch':
        if not args.vllm_model_path:
            raise ValueError("--vllm_model_path is required for batch mode.")
        
        print("Initializing vLLM for batch mode...")
        VLLMManager.initialize_model(model_path=args.vllm_model_path)
        
        run_batch_pipeline(dataset, args.config, result_directory, args.data_mode)

    elif args.runner_mode == 'agentic':
        # The old logic remains here
        run_manager = RunManager(args)
        # We need to re-initialize tasks using the dict version of the dataset
        with open(args.data_path, 'r') as file:
            dict_dataset = json.load(file)
        run_manager.initialize_tasks(dict_dataset)
        run_manager.run_tasks()
        run_manager.generate_sql_files()

if __name__ == '__main__':
    main()