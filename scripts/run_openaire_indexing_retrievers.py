import os
import time
import pandas as pd
import wandb

# --- Core Framework Imports ---
from nlp_retrieval.searcher import Searcher

# --- Method-Specific Imports ---
from retrievers.CHESS.chess_retriever import ChessMinHashLshRetriever
from retrievers.OmniSQL.omnisql_retriever import OmniSQLRetriever
from retrievers.OpenSearch.opensearch_retriever import OpenSearchDenseValueRetriever
from retrievers.ValueNet.valuenet_retriever import ValueNetRetriever
from retrievers.BRIDGE.bridge_retriever import BridgeRetriever
from nlp_retrieval.loaders.openaire_loader import OpenAireLoader

# --- Configuration ---
BASE_PATH = "assets/retrievers_openaire"
INDEXES_ROOT = os.path.join(BASE_PATH, "indexes")

# PostgreSQL Connection Configuration
DB_CONFIG = { #INSERT YOUR POSTGRESQL CONNECTION DETAILS
    "host": "", 
    "port": "",
    "dbname": "",
    "user": "",
    "password": ""
}
SCHEMA_NAME = ""

# Weights & Biases Configuration
WANDB_ENTITY = ""
WANDB_PROJECT = "value_linking_openaire"
WANDB_GROUP = "indexing"

# Define the different indexing methods
SEARCHER_METHODS = [
    {"name": "CHESS", "retriever_class": ChessMinHashLshRetriever, "index_subdir": "chess"},
    {"name": "OmniSQL", "retriever_class": OmniSQLRetriever, "index_subdir": "omnisql"},
    {"name": "OpenSearch", "retriever_class": OpenSearchDenseValueRetriever, "index_subdir": "opensearch"},
    {"name": "ValueNet", "retriever_class": ValueNetRetriever, "index_subdir": "valuenet"},
    {"name": "BRIDGE", "retriever_class": BridgeRetriever, "index_subdir": "bridge"},
]

def main():
    os.makedirs(INDEXES_ROOT, exist_ok=True)
    db_id = DB_CONFIG["dbname"]

    for method in SEARCHER_METHODS:
        method_name = method["name"]
        print(f"\n{'='*25} Starting Method: {method_name} {'='*25}")

        wandb.init(
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY, 
            group=WANDB_GROUP,
            name=f"{method_name}-Indexing-Report",
            config={"method": method_name, "schema": SCHEMA_NAME},
            reinit=True
        )

        retriever = method["retriever_class"]()
        
        # Pass the SCHEMA_NAME here!
        loader = OpenAireLoader(db_config=DB_CONFIG, schema=SCHEMA_NAME, max_values=-1)

        searcher = Searcher(retrievers=[retriever])
        index_path = os.path.join(INDEXES_ROOT, method["index_subdir"], db_id)

        start_time = time.time()
        searcher.index(loader=loader, output_path=index_path)
        duration = time.time() - start_time
        
        # Log aggregated results
        wandb.summary["total_indexing_time_seconds"] = duration
        wandb.summary["total_items_indexed"] = loader.total_yielded
        
        wandb.finish()

    print(f"\n{'='*20} All indexing complete. {'='*20}")
    
if __name__ == "__main__":
    main()