import os
import time
from pathlib import Path
from typing import List, Tuple, Type

import wandb
from tqdm import tqdm

from utils.sqlite_db import DatabaseSqlite
from value_index.value_index_abc import ValueIndexABC
from value_index.value_index_bm25_pyserini import BM25Index
from value_index.value_index_minhashlsh import MinHashForestIndex
from value_index.value_index_faiss_flat import FaissFlatIndex

DATABASES_ROOT_PATH = Path("assets/retrievers/databases/")
INDEXES_CACHE_PATH = Path("assets/retrievers/indexes/")

WANDB_ENTITY = "darelab"
WANDB_PROJECT = "value_linking"
WANDB_GROUP = "indexing"


def find_database_paths(root_path: Path) -> List[Path]:
    db_paths = []
    if not root_path.is_dir():
        print(f"Error: Database root path not found at '{root_path}'")
        return []
        
    for db_id_dir in root_path.iterdir():
        if db_id_dir.is_dir():
            db_file = db_id_dir / f"{db_id_dir.name}.sqlite"
            if db_file.is_file():
                db_paths.append(db_file)
    
    db_paths.sort()
    return db_paths


def run_indexing_for_one_method(
    indexer_name: str,
    IndexerClass: Type[ValueIndexABC],
    db_paths: List[Path]
):
    print(f"\n{'='*20} Starting Indexing for: {indexer_name} {'='*20}")

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=WANDB_GROUP,
        name=f"run-indexing-{indexer_name}",
        job_type="indexing",
        reinit=True,
    )
    
    wandb.config.update({
        "indexer": indexer_name,
        "database_count": len(db_paths),
        "cache_path": str(INDEXES_CACHE_PATH)
    })

    indexer_instance = IndexerClass()
    db_indexing_times = []
    total_start_time = time.time()

    db_times_table = wandb.Table(columns=["database_id", "indexing_time_sec"])

    for db_path in tqdm(db_paths, desc=f"Indexing with {indexer_name}"):
        db_id = db_path.parent.name
        db_start_time = time.time()

        db = DatabaseSqlite(str(db_path))

        specific_output_path = INDEXES_CACHE_PATH / indexer_name / db_id
        os.makedirs(specific_output_path, exist_ok=True)

        indexer_instance.create_index(database=db, output_path=str(specific_output_path))

        db_end_time = time.time()
        db_duration = db_end_time - db_start_time
        db_indexing_times.append(db_duration)

        db_times_table.add_data(db_id, db_duration)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    avg_time_per_db = sum(db_indexing_times) / len(db_indexing_times)

    print(f"\n{'---'*10}\nFinished indexing for: {indexer_name}")
    print(f"Successfully indexed {len(db_paths)}/{len(db_paths)} databases.")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Average time per database: {avg_time_per_db:.2f} seconds")
    print(f"{'---'*10}")

    wandb.log({
        "total_indexing_time_sec": total_duration,
        "avg_time_per_db_sec": avg_time_per_db,
        "databases_indexed_successfully": len(db_paths),
        "databases_failed": 0,
        "per_database_times": db_times_table  # Log the table here
    })

    run.finish()


def main():
    db_paths = find_database_paths(DATABASES_ROOT_PATH)
    if not db_paths:
        print("No databases found. Exiting.")
        return
    print(f"Found {len(db_paths)} databases to index in '{DATABASES_ROOT_PATH}'.")

    indexers_to_run: List[Tuple[str, Type[ValueIndexABC]]] = [
        ("BM25", BM25Index),
        ("MinHashLSH", MinHashForestIndex),
        #("FaissFlat", FaissFlatIndex),
    ]

    os.makedirs(INDEXES_CACHE_PATH, exist_ok=True)
    print(f"Using index cache path: {INDEXES_CACHE_PATH}")

    for indexer_name, IndexerClass in indexers_to_run:
        run_indexing_for_one_method(indexer_name, IndexerClass, db_paths)
        
    print("\nAll indexing tasks are complete.")


if __name__ == "__main__":
    main()