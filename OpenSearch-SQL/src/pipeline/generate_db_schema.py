import logging
from typing import Any, Dict, List
from pathlib import Path
from pipeline.utils import node_decorator
from pipeline.pipeline_manager import PipelineManager
from llm.model import model_chose
from llm.db_conclusion import *
import json
import os # <-- Add this import

@node_decorator(check_schema_status=False)
def generate_db_schema(state: Dict[str, Any]) -> Dict[str, Any]:
    task = state["keys"]["task"]
    # +++ START MODIFICATION: Use raw paths from state +++
    db_root_path = state["keys"]["db_root_path"]
    db_mode = state["keys"]["db_mode"]
    db_id = state["keys"]["db_id"]

    # Construct paths directly
    root_path = Path(db_root_path).resolve()
    db_json_dir = root_path / "data_preprocess" / f"{db_mode}.json"
    tables_info_dir = root_path / "data_preprocess" / "tables.json"
    sqllite_dir = root_path / db_mode / db_id / f"{db_id}.sqlite"
    db_dir = root_path / db_mode / db_id
    ext_file = root_path / "db_schema.json"
    # +++ END MODIFICATION +++

    config,node_name=PipelineManager().get_model_para()
    # Get the shared model instance, don't load it here
    bert_model = PipelineManager().get_bert_model()

    chat_model = model_chose(node_name,config["engine"])

    if os.path.exists(ext_file):
        with open(ext_file, 'r') as f:
            data = json.load(f)
    else:
        data ={}

    # Get database info agent
    DB_info_agent = db_agent_string(chat_model)
    
    # Check if the database has already been processed
    db = task.db_id
    existing_entry = data.get(db)

    if existing_entry:
        all_info,db_col = existing_entry
    else:
        # Pass the correctly constructed paths to the agent
        all_info, db_col = DB_info_agent.get_allinfo(db_json_dir, db,sqllite_dir,db_dir,tables_info_dir, bert_model)
        data[db]=[all_info,db_col]
        with open(ext_file, 'w') as f:
            json.dump(data, f, indent=4,ensure_ascii=False)
    
    response = {
        "db_list": all_info,
        "db_col_dic": db_col
    }
    return response