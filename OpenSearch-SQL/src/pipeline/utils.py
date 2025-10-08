from functools import wraps
from typing import Dict, List, Any, Callable
from runner.logger import Logger
from runner.database_manager import DatabaseManager

def node_decorator(check_schema_status: bool = False) -> Callable:
    """
    A decorator to add logging and error handling to pipeline node functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            node_name = func.__name__
            task = state["keys"]["task"]
            execution_history = state["keys"]["execution_history"]

            # Set logger context for the current task
            Logger(db_id=task.db_id, question_id=task.question_id)
            
            # Check if this node has already been run (from a checkpoint)
            if any(x["node_type"] == node_name for x in execution_history):
                Logger().log(f"---SKIPPING {node_name.upper()} (found in checkpoint)---")
                return state

            Logger().log(f"---{node_name.upper()}---")
            result = {"node_type": node_name}

            try:
                # Call the node function with the whole state
                output = func(state)
                result.update(output)
                result["status"] = "success"
            except Exception as e:
                Logger().log(f"Node '{node_name}': {task.db_id}_{task.question_id}\n{type(e)}: {e}\n", "error")
                result.update({
                    "status": "error",
                    "error": f"{type(e)}: <{e}>",
                })
            
            execution_history.append(result)
            Logger().dump_history_to_file(execution_history)
            
            return state
        return wrapper
    return decorator

def get_last_node_result(execution_history: List[Dict[str, Any]], node_type: str) -> Dict[str, Any]:
    """
    Retrieves the last result for a specific node type from the execution history.
    """
    for node in reversed(execution_history):
        if node["node_type"] == node_type:
            return node
    return None
            
def make_newprompt(new_prompt,
                   fewshot,
                   key_col_des,
                   db_info,
                   question,
                   hint="",q_order=""):
    n_prompt = new_prompt.format(fewshot=fewshot,
                                 db_info=db_info,
                                 question=question,
                                 hint=hint,
                                 key_col_des=key_col_des,q_order=q_order)

    return n_prompt