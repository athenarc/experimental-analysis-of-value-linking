import os
import sys
import time
import shutil
import pandas as pd
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
LOCAL_TMP_DIR = os.path.join(os.path.dirname(__file__), '../tmp')
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = LOCAL_TMP_DIR
print(f"Redirected temporary files to: {LOCAL_TMP_DIR}")

# --- Imports ---
from nlp_retrieval.searcher import Searcher
from nlp_retrieval.retrievers.bm25_retriever import PyseriniRetriever
from nlp_retrieval.retrievers.minhash_lsh_retriever import MinHashLshRetriever
from nlp_retrieval.retrievers.dense_retriever import FaissRetriever
from nlp_retrieval.loaders.openaire_loader import OpenAireLoader

# --- Configuration ---
DB_CONFIG = {
    "host": "",
    "port": "",
    "database": "",
    "user": "",
    "password": ""
}

OUTPUT_ROOT = "assets/scalability_experiments"
WANDB_ENTITY = "darelab"
WANDB_PROJECT = "value_linking_scalability"

SLICES = [
    ("10k", 10000),
    ("100k", 100000),
    ("1M", 1000000),
    ("10M", 10000000),
    ("50M", 50000000),
]

METHODS = [
    #{
    #    "name": "BM25",
    #    "retriever_cls": PyseriniRetriever,
    #    "params": {} 
    #},
    {
        "name": "MinHashLSH",
        "retriever_cls": MinHashLshRetriever,
        "params": {"num_perm": 100}
    },
    #{
    #    "name": "Dense_BGE_M3",
    #    "retriever_cls": FaissRetriever,
    #    "params": {
    #        "model_name_or_path": "BAAI/bge-m3", 
    #        "batch_size": 4096,
    #        "embedding_backend": "tei",
    #        "tei_pod_addresses": [
    #            "http://localhost:8080", 
    #            "http://localhost:8081",
    #            "http://localhost:8082",
    #            "http://localhost:8083"
    #        ]
    #    }
    #}
]

def get_dir_size(path):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for method in METHODS:
        method_name = method["name"]
        table_data = []

        print(f"Initializing {method_name}...")
        retriever = method["retriever_cls"](**method["params"])

        for slice_name, max_values in SLICES:
            wandb.init(
                project=WANDB_PROJECT, 
                entity=WANDB_ENTITY, 
                name=f"{method_name}-{slice_name}", # Unique name per slice
                group=method_name,                   # Group them by method
                config={**method, "slice": slice_name, "max_values": max_values},
                reinit=True
            )

            index_path = os.path.join(OUTPUT_ROOT, method_name, slice_name)
            if os.path.exists(index_path) and len(os.listdir(index_path)) > 0:
                print(f"Skipping {slice_name}, already exists.")
            
            print(f"\n--- Processing Slice: {slice_name} (Max Values: {max_values}) ---")
            
            loader = OpenAireLoader(DB_CONFIG, max_values=max_values)
            
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            os.makedirs(index_path, exist_ok=True)

            searcher = Searcher(retrievers=[retriever])
            
            start_time = time.time()
            searcher.index(loader, index_path)
            duration = time.time() - start_time
            
            size_bytes = get_dir_size(index_path)
            size_gb = size_bytes / (1024**3)

            print(f"Done. Time: {duration:.2f}s | Size: {size_gb:.2f} GB")
            
            table_data.append([slice_name, max_values, duration, size_gb])
            
            wandb.log({
                "slice": slice_name,
                "max_values": max_values,
                "indexing_time_s": duration,
                "index_size_gb": size_gb
            })
            wandb.finish()
            
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f"{method_name}-Summary", group=method_name, reinit=True)
        columns = ["Slice", "Fraction", "Time (s)", "Size (GB)"]
        wandb.log({"scalability_metrics": wandb.Table(columns=columns, data=table_data)})
        wandb.finish()

if __name__ == "__main__":
    main()