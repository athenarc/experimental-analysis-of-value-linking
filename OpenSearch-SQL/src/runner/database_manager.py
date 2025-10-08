import os
import pickle
from pathlib import Path

from typing import Callable, Dict, List, Any
from runner.execution import compare_sqls


class DatabaseManager:
    """
    Manages database operations including schema generation, querying, and managing column profiles.
    This is a regular class to ensure instance-per-task state management.
    """
    
    def __init__(self, db_mode: str, db_root_path: str, db_id: str):
        """
        Initializes the DatabaseManager instance for a specific task.
        """
        self.db_mode = db_mode
        self.db_root_path = db_root_path
        self.db_id = db_id
        self._set_paths()


    def _set_paths(self):
        """Sets the paths for the database files and directories."""
        # Ensure the root path is absolute to be thread-safe
        root_path = Path(self.db_root_path).resolve()
        
        # Corrected path construction
        self.db_path = root_path / self.db_mode / self.db_id / f"{self.db_id}.sqlite"
        self.db_directory_path = root_path / self.db_mode / self.db_id

        # Set the rest of the paths
        self.db_json = root_path / "data_preprocess" / f"{self.db_mode}.json"
        self.db_tables = root_path / "data_preprocess" / "tables.json"
        self.db_fewshot_path = root_path / "fewshot" / "questions.json"
        self.db_fewshot2_path = root_path / "correct_fewshot2.json"
        self.emb_dir = root_path / "emb"

    @staticmethod
    def with_db_path(func: Callable):
        """
        Decorator to inject db_path as the first argument to the function.
        """
        def wrapper(self, *args, **kwargs):
            return func(self.db_path, *args, **kwargs)
        return wrapper

    @classmethod
    def add_methods_to_class(cls, funcs: List[Callable]):
        """
        Adds methods to the class with db_path automatically provided.

        Args:
            funcs (List[Callable]): List of functions to be added as methods.
        """
        for func in funcs:
            method = cls.with_db_path(func)
            setattr(cls, func.__name__, method)

# List of functions to be added to the class
functions_to_add = [
    compare_sqls
]

# Adding methods to the class
DatabaseManager.add_methods_to_class(functions_to_add)