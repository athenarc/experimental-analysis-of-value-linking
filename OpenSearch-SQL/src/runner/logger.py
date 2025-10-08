import logging
import json
from threading import Lock
from pathlib import Path
from typing import Any, List, Dict, Union

class Logger:
    _instance = None
    _lock = Lock()

    def __new__(cls, db_id: str = None, question_id: str = None, result_directory: str = None):
        """
        Ensures a singleton instance of Logger.
        The first call MUST provide a result_directory.
        Subsequent calls can update the db_id and question_id context.
        """
        with cls._lock:
            # This block handles the very first initialization of the singleton
            if cls._instance is None:
                if result_directory is None:
                    raise ValueError("Logger must be initialized with a 'result_directory' on its first call.")
                cls._instance = super(Logger, cls).__new__(cls)
                cls._instance._init(db_id, question_id, result_directory)
            
            # This block handles subsequent calls, treating them as context updates
            # It updates the context without re-initializing the result_directory
            elif db_id is not None and question_id is not None:
                cls._instance.db_id = db_id
                cls._instance.question_id = question_id
                # If a new result_directory is provided, update it. Otherwise, keep the old one.
                if result_directory is not None:
                    cls._instance.result_directory = Path(result_directory)

            return cls._instance

    def _init(self, db_id: str, question_id: str, result_directory: str):
        """
        Initializes the Logger instance with the provided parameters.
        This is now only called once by the __new__ method.
        """
        self.db_id = db_id
        self.question_id = question_id
        self.result_directory = Path(result_directory)

    def _set_log_level(self, log_level: str):
        """
        Sets the logging level.
        """
        log_level_attr = getattr(logging, log_level.upper(), None)
        if log_level_attr is None:
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(level=log_level_attr, format='%(levelname)s: %(message)s')

    def log(self, message: str, log_level: str = "info", **kwargs):
    # +++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Logs a message at the specified log level.
        """
        log_method = getattr(logging, log_level, None)
        if log_method is None:
            raise ValueError(f"Invalid log level: {log_level}")
        # +++ PASS THE KWARGS THROUGH TO THE REAL LOGGER +++
        log_method(message, **kwargs)

    def log_conversation(self, text: Union[str, List[Any], Dict[str, Any], bool], _from: str, step: str):
        """
        Logs a conversation text to a file.
        """
        log_file_path = self.result_directory / "logs" / f"{self.question_id}_{self.db_id}.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with log_file_path.open("a") as file:
            file.write(f"############################## {_from} at step {step} ##############################\n\n")
            if isinstance(text, str):
                file.write(text)
            elif isinstance(text, (list, dict)):
                formatted_text = json.dumps(text, indent=4)
                file.write(formatted_text)
            elif isinstance(text, bool):
                file.write(str(text))
            file.write("\n\n")

    def dump_history_to_file(self, execution_history: List[Dict[str, Any]]):
        """
        Dumps the execution history to a JSON file.
        """
        execution_history_tmp=make_serial(execution_history)

        file_path = self.result_directory / f"{self.question_id}_{self.db_id}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as file:
            json.dump(execution_history_tmp, file, indent=4,ensure_ascii=False)

def make_serial(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, list):
        return [make_serial(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_serial(item) for item in obj]
    elif isinstance(obj, set):
        return [make_serial(item) for item in obj]
    elif isinstance(obj, dict):
        return {make_serial(key): make_serial(value) for key, value in obj.items()}
    else:
        try:
            return str(obj)
        except Exception as e:
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable") from e