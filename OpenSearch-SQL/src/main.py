# src/main.py

import argparse
import json
from datetime import datetime
from typing import Any, Dict, List
import argparse
from runner.run_manager import RunManager
import os
import vllm
from llm.model import VLLM_req

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    return dataset

def main(args):
    """
    Main function to run the pipeline with the specified configuration.
    """
    ##
    db_json=os.path.join(args.db_root_path,'data_preprocess',f'{args.data_mode}.json')
    
    # +++ Add logic to initialize VLLM model if in offline mode +++
    if args.offline_vllm_batch:
        if not args.vllm_model_path:
            raise ValueError("Please provide --vllm_model_path for offline batch mode.")
        print(f"Loading VLLM model for offline generation: {args.vllm_model_path}...")
        # Initialize the VLLM instance
        vllm_instance = vllm.LLM(model=args.vllm_model_path, trust_remote_code=True, tensor_parallel_size=2,gpu_memory_utilization=0.8,download_dir="/data/hdd1/vllm_models/",max_model_len=30000)
        # Make the instance accessible to the VLLM_req class
        VLLM_req.set_offline_vllm_instance(vllm_instance)
        print("VLLM model loaded successfully.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    dataset = load_dataset(db_json)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(args.start,args.end,dataset)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

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
    args_parser.add_argument('--offline_vllm_batch', action='store_true', help="Enable offline VLLM batch generation instead of sending API requests.")
    args_parser.add_argument('--vllm_model_path', type=str, help="Path to the VLLM model for offline generation (e.g., 'Qwen/Qwen2.5-Coder-32B-Instruct').")
    args = args_parser.parse_args()
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.use_checkpoint:
        print('Using checkpoint')
        if not args.checkpoint_nodes:
            raise ValueError('Please provide the checkpoint nodes to use checkpoint')
        if not args.checkpoint_dir:
            raise ValueError('Please provide the checkpoint path to use checkpoint')
    
    main(args)