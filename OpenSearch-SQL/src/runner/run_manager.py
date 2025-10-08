import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from runner.logger import Logger
from runner.task import Task
from runner.database_manager import DatabaseManager
from runner.statistics_manager import StatisticsManager
from pipeline.pipeline_manager import PipelineManager

# Import the node functions to be called directly by the orchestrator
from pipeline.generate_db_schema import generate_db_schema
from pipeline.extract_col_value import extract_col_value
from pipeline.extract_query_noun import extract_query_noun
from pipeline.column_retrieve_and_other_info import column_retrieve_and_other_info
from pipeline.candidate_generate import candidate_generate_core
from pipeline.align_correct import align_correct
from pipeline.vote import vote
from pipeline.evaluation import evaluation
from llm.model import model_chose
from runner.check_and_correct import get_sql_batch

class RunManager:
    RESULT_ROOT_PATH = "results"

    def __init__(self, args: Any):
        self.args = args
        self.result_directory = self.get_result_directory()
        self.statistics_manager = StatisticsManager(self.result_directory)
        self.tasks: List[Task] = []
        self.total_number_of_tasks = 0
        self.processed_tasks = 0
        # Initialize shared models here, once.
        pipeline_manager = PipelineManager(json.loads(self.args.pipeline_setup), self.args)
        pipeline_manager.initialize_shared_models()

    def get_result_directory(self) -> str:
        # ... (unchanged)
        data_mode = self.args.data_mode
        pipeline_nodes = self.args.pipeline_nodes
        dataset_name = Path(self.args.db_root_path).stem
        run_folder_name = str(self.args.run_start_time)
        run_folder_path = Path(self.RESULT_ROOT_PATH) / data_mode / pipeline_nodes / dataset_name / run_folder_name
        
        run_folder_path.mkdir(parents=True, exist_ok=True)
        
        arg_file_path = run_folder_path / "-args.json"
        with arg_file_path.open('w') as file:
            json.dump(vars(self.args), file, indent=4)
        
        log_folder_path = run_folder_path / "logs"
        log_folder_path.mkdir(exist_ok=True)
        
        return str(run_folder_path)

    def initialize_tasks(self, start: int, end: int, dataset: List[Dict[str, Any]]):
        # ... (unchanged)
        for i, data in enumerate(dataset):
            if i < start:
                continue
            if i >= end:
                break
            if "question_id" not in data:
                data = {"question_id": i, **data}
            task = Task(data)
            self.tasks.append(task)
        self.total_number_of_tasks = len(self.tasks)
        print(f"Total number of tasks: {self.total_number_of_tasks}")

    def run_tasks(self):
        """Entry point for running tasks, which now calls the batched orchestrator."""
        self.run_tasks_batched()

    def run_tasks_batched(self):
        """
        Orchestrates the pipeline execution for a batch of tasks,
        with batched LLM calls.
        """
        batch_states = []
        for task in self.tasks:
            Logger(db_id=task.db_id, question_id=task.question_id, result_directory=self.result_directory)
            PipelineManager() 
            
            execution_history = self.load_checkpoint(task.db_id, task.question_id)
            
            # +++ START MODIFICATION: Pass raw args instead of the manager object +++
            db_manager = DatabaseManager(
                db_mode=self.args.data_mode, 
                db_root_path=self.args.db_root_path, 
                db_id=task.db_id
            )
            batch_states.append({
                "keys": {
                    "task": task, 
                    "execution_history": execution_history,
                    "db_manager": db_manager,  # Keep this for nodes that use it
                    # Add raw paths for nodes that need them directly
                    "db_root_path": self.args.db_root_path,
                    "db_mode": self.args.data_mode,
                    "db_id": task.db_id
                }
            })
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        pre_batch_nodes = [
            generate_db_schema,
            extract_col_value,
            extract_query_noun,
            column_retrieve_and_other_info,
        ]
        
        post_batch_nodes = [
            align_correct,
            vote,
            evaluation
        ]


        for node_func in pre_batch_nodes:
            # +++ NO LONGER NEED TO CREATE DB_MANAGER IN THIS LOOP +++
            for state in tqdm(batch_states, desc=f"Executing Node: {node_func.__name__.upper()}", unit="task"):
                node_func(state)
                
        config, node_name = PipelineManager().get_model_para(node_name="candidate_generate")
        prompts_for_batch = []
        for state in tqdm(batch_states, desc="Preparing prompts for batch", unit="task"):
            # db_manager is already in the state, no need to create it again
            core_result = candidate_generate_core(state, config)
            prompts_for_batch.append(core_result["prompt"])
            state["keys"]["rewrite_question"] = core_result["rewrite_question"]

        print(f"Sending batch of {len(prompts_for_batch)} prompts to LLM...")
        chat_model = model_chose(node_name, config["engine"])
        batched_sqls = get_sql_batch(chat_model, prompts_for_batch, temp=config['temperature'], n=config['n'])
        print("...Batch received from LLM.")

        for i, state in enumerate(tqdm(batch_states, desc="Processing LLM results", unit="task")):
            result = {
                "node_type": "candidate_generate",
                "status": "success",
                "rewrite_question": state["keys"]["rewrite_question"],
                "SQL": batched_sqls[i]
            }
            state["keys"]["execution_history"].append(result)
            Logger(db_id=state["keys"]["task"].db_id, question_id=state["keys"]["task"].question_id).dump_history_to_file(state["keys"]["execution_history"])

        for node_func in post_batch_nodes:
            # +++ NO LONGER NEED TO CREATE DB_MANAGER IN THIS LOOP +++
            for state in tqdm(batch_states, desc=f"Executing Node: {node_func.__name__.upper()}", unit="task"):
                node_func(state)

        for state in batch_states:
            self.task_done((
                state,
                state["keys"]["task"].db_id,
                state["keys"]["task"].question_id
            ))

    def task_done(self, log: Tuple[Any, str, int]):
        # ... (unchanged)
        state, db_id, question_id = log
        if state is None:
            return
        
        evaluation_result = state["keys"]['execution_history'][-1]
        if evaluation_result.get("node_type") == "evaluation":
            for evaluation_for, result in evaluation_result.items():
                if evaluation_for in ['node_type', 'status']:
                    continue
                self.statistics_manager.update_stats(db_id, question_id, evaluation_for, result)
            self.statistics_manager.dump_statistics_to_file()
        
        self.processed_tasks += 1
        self.plot_progress()

    def plot_progress(self, bar_length: int = 100):
        """
        Plots the overall progress of task processing.
        """
        processed_ratio = self.processed_tasks / self.total_number_of_tasks
        # +++ ADD THIS LINE BACK +++
        progress_length = int(processed_ratio * bar_length)
        # ++++++++++++++++++++++++++
        # Use a carriage return to overwrite the line, ensuring it works well with tqdm
        print(f"Overall Progress: [{'=' * progress_length}>{' ' * (bar_length - progress_length)}] {self.processed_tasks}/{self.total_number_of_tasks}", end='\r')
        if self.processed_tasks == self.total_number_of_tasks:
            print() # Print a final newline when complete
            
    def load_checkpoint(self, db_id: str, question_id: int) -> List[Dict[str, Any]]:
        # ... (unchanged)
        execution_history = []
        if self.args.use_checkpoint:
            checkpoint_file = Path(self.args.checkpoint_dir) / f"{question_id}_{db_id}.json"
            if checkpoint_file.exists():
                with checkpoint_file.open('r') as file:
                    checkpoint = json.load(file)
                    for step in checkpoint:
                        node_type = step["node_type"]
                        if node_type in self.args.checkpoint_nodes:
                            execution_history.append(step)
            else:
                Logger().log(f"Checkpoint file not found: {checkpoint_file}", "warning")
        return execution_history

    def generate_sql_files(self):
        # ... (unchanged)
        sqls = {}
        
        for file in os.listdir(self.result_directory):
            if file.endswith(".json") and "_" in file:
                try:
                    _index = file.find("_")
                    question_id = int(file[:_index])
                    db_id = file[_index + 1:-5]
                    with open(os.path.join(self.result_directory, file), 'r') as f:
                        exec_history = json.load(f)
                        for step in exec_history:
                            if "SQL" in step:
                                node_type = step["node_type"]
                                if node_type not in sqls:
                                    sqls[node_type] = {}
                                sqls[node_type][question_id] = step["SQL"]
                except (ValueError, IndexError):
                    continue

        for key, value in sqls.items():
            with open(os.path.join(self.result_directory, f"-{key}.json"), 'w') as f:
                json.dump(value, f, indent=4, ensure_ascii=False)