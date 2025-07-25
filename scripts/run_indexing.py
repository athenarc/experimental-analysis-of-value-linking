import os
import time

import pandas as pd
import wandb

# --- Core Framework Imports ---
from darelabdb.nlp_retrieval.searcher import Searcher

# --- Method-Specific Imports ---
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_db_loader import ChessDBLoader
from development.experimental_analysis_of_value_linking.retrievers.CHESS.chess_retriever import ChessMinHashLshRetriever
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_loader import OmniSQLLoader
from development.experimental_analysis_of_value_linking.retrievers.OmniSQL.omnisql_retriever import OmniSQLRetriever
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_loader import OpenSearchValueLoader
from development.experimental_analysis_of_value_linking.retrievers.OpenSearch.opensearch_retriever import OpenSearchDenseValueRetriever

# --- Configuration ---
BASE_PATH = "development/experimental_analysis_of_value_linking/assets/retrievers"
DATABASES_ROOT = os.path.join(BASE_PATH, "databases")
INDEXES_ROOT = os.path.join(BASE_PATH, "indexes")

# Weights & Biases Configuration
WANDB_ENTITY = "darelab"
WANDB_PROJECT = "value_linking"
WANDB_GROUP = "indexing"

# Define the different indexing methods to be run
SEARCHER_METHODS = [
    {
        "name": "CHESS",
        "loader_class": ChessDBLoader,
        "retriever_class": ChessMinHashLshRetriever,
        "index_subdir": "chess",
    },
    {
        "name": "OmniSQL",
        "loader_class": OmniSQLLoader,
        "retriever_class": OmniSQLRetriever,
        "index_subdir": "omnisql",
    },
    {
        "name": "OpenSearch",
        "loader_class": OpenSearchValueLoader,
        "retriever_class": OpenSearchDenseValueRetriever,
        "index_subdir": "opensearch",
    },
]
SEARCHER_METHODS = [
    {
        "name": "CHESS",
        "loader_class": ChessDBLoader,
        "retriever_class": ChessMinHashLshRetriever,
        "index_subdir": "chess",
    }
]
def main():
    """
    Main function to orchestrate the indexing of all databases for all methods,
    logging one W&B run per method with a detailed performance table.
    This version is optimized to avoid re-initializing heavy models in a loop.
    """
    os.makedirs(INDEXES_ROOT, exist_ok=True)

    db_ids = [d for d in os.listdir(DATABASES_ROOT) if os.path.isdir(os.path.join(DATABASES_ROOT, d))]

    if not db_ids:
        print(f"No database directories found in {DATABASES_ROOT}. Exiting.")
        return

    print(f"Found {len(db_ids)} databases to index: {db_ids}")

    # --- Outer loop iterates over each method ---
    for method in SEARCHER_METHODS:
        method_name = method["name"]
        print(f"\n{'='*25} Starting Method: {method_name} {'='*25}")

        # Initialize one W&B run for the entire method
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, group=WANDB_GROUP,
            name=f"{method_name}-Indexing-Report",
            config={"method": method_name},
            reinit=True
        )

        table_data = []
        total_indexing_time = 0
        total_items_indexed = 0
        print(f"Initializing retriever for {method_name}...")
        retriever = method["retriever_class"]()
        print("Retriever initialized.")

        # --- Inner loop iterates over each database for the current method ---
        for db_id in db_ids:
            print(f"\n--- Indexing Database: {db_id} for Method: {method_name} ---")
            db_dir_path = os.path.join(DATABASES_ROOT, db_id)
            db_file_path = os.path.join(db_dir_path, f"{db_id}.sqlite")

            # Instantiate the correct loader for the method (loaders are db-specific)
            if method_name == "CHESS":
                loader = method["loader_class"](db_directory_path=db_dir_path)
            elif method_name == "OmniSQL":
                loader = method["loader_class"](db_file_path=db_file_path)
            else: # OpenSearch
                loader = method["loader_class"](db_path=db_file_path, db_id=db_id)


            items_to_index = loader.load()
            num_items = len(items_to_index)
            total_items_indexed += num_items
            print(f"Loaded {num_items} items from {db_id}.")

            # The Searcher itself is lightweight, so creating it in the loop is fine.
            # We pass the already-initialized retriever to it.
            searcher = Searcher(retrievers=[retriever])

            # Define the output path for the index
            index_path = os.path.join(INDEXES_ROOT, method["index_subdir"], db_id)

            # Time the indexing process
            start_time = time.time()
            searcher.index(loader=loader, output_path=index_path)
            duration = time.time() - start_time
            total_indexing_time += duration

            # Append data for the W&B table
            table_data.append([db_id, duration, num_items])
            print(f"Indexing for {db_id} took {duration:.2f} seconds.")

        # --- After processing all databases, log the aggregated results for the method ---
        if table_data:
            summary_df = pd.DataFrame(table_data, columns=["Database ID", "Indexing Time (s)", "Items Indexed"])
            print(f"\n--- Summary for {method_name} ---")
            print(summary_df.to_markdown(index=False))

            columns = ["Database ID", "Indexing Time (s)", "Items Indexed"]
            performance_table = wandb.Table(columns=columns, data=table_data)
            wandb.log({"indexing_times_per_db": performance_table})

            wandb.summary["total_indexing_time_seconds"] = total_indexing_time
            wandb.summary["total_items_indexed"] = total_items_indexed
            wandb.summary["average_time_per_db_seconds"] = total_indexing_time / len(db_ids) if db_ids else 0

        wandb.finish()
        print(f"Finished W&B run for {method_name}.")

    print(f"\n{'='*20} All indexing complete. {'='*20}")

if __name__ == "__main__":
    main()