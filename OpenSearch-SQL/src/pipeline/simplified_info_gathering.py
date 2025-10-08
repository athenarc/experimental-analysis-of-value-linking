# src/pipeline/simplified_info_gathering.py

import logging, re, json
from typing import Any, Dict, List
from pathlib import Path
from pipeline.utils import node_decorator, get_last_node_result
from pipeline.pipeline_manager import PipelineManager
from runner.database_manager import DatabaseManager
from llm.db_conclusion import find_foreign_keys_MYSQL_like
from runner.column_update import ColumnUpdater

@node_decorator(check_schema_status=False)
def simplified_info_gathering(task: Any, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A simplified node that gathers basic schema info without performing value retrieval.
    It assumes oracle values are already in the question.
    """
    config, node_name = PipelineManager().get_model_para()
    paths = DatabaseManager()
    tables_info_dir = paths.db_tables
    db = task.db_id

    # Get the full database schema from the previous step
    all_db_col = get_last_node_result(execution_history, "generate_db_schema")["db_col_dic"]
    db_col = {x: all_db_col[x][0] for x in all_db_col}

    # Get all columns as a flat set for the 'column' output
    # This provides all possible columns to the next step.
    all_columns_formatted = ColumnUpdater(db_col).col_suffix(set(db_col.keys()))

    # Get foreign key information
    foreign_keys, foreign_set = find_foreign_keys_MYSQL_like(tables_info_dir, db)

    # Since we are not retrieving values, these will be empty.
    # The candidate_generate node will rely solely on the info in the question.
    L_values = []
    q_order = "" # Bypassing the LLM call for query ordering

    response = {
        "L_values": L_values,
        "column": all_columns_formatted, # Provide the full schema
        "foreign_keys": foreign_keys,
        "foreign_set": foreign_set,
        "q_order": q_order
    }

    return response