import os
import numpy as np
import pickle
from utils.sqlite_db import DatabaseSqlite
from typing import List
import pandas as pd
import gensim.downloader as api
from value_index.value_index_abc import (
    ValueIndexABC,
)
from filtering.filtering_abc import FilterABC
import faiss
from pathlib import Path
INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class DartSQLIndex(ValueIndexABC):
    def __init__(self):
        self.glove = self._load_glove()
        self.loaded_indexes = {}  # {index_path: (faiss_index, row_mappings)}
        self.device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"

    def _load_glove(self):
        """Load GloVe embeddings using Gensim"""
        print("Loading GloVe embeddings...")
        return api.load("glove-wiki-gigaword-300")

    def _get_row_embedding(self, row_dict):
        """Convert row dictionary to averaged GloVe embedding exactly as in paper"""
        tokens = []
        for col, value in row_dict.items():
            # Include both column names and cell values as per paper
            tokens += str(col).lower().split() + str(value).lower().split()

        vectors = [self.glove[token] for token in tokens if token in self.glove]
        embedding = np.mean(vectors, axis=0) if vectors else np.zeros(300)
        return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity

    def create_index(
        self, database: DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """Create index following the paper's exact methodology"""
        dart_folder = os.path.join(output_path, "DARTSQL")
        os.makedirs(dart_folder, exist_ok=True)

        # Create FAISS index with inner product (cosine similarity)
        dimension = 300
        index = faiss.IndexFlatIP(dimension)
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        row_mappings = []
        schema = database.get_tables_and_columns()

        for table in schema["tables"]:
            if table.startswith("sqlite_autoindex") or table == "sqlite_sequence" :
                continue

            try:
                rows = database.execute(f'SELECT * FROM "{table}"', limit=-1).to_dict(
                    "records"
                )
                for row in rows:
                    # Convert row to paper's JSON dictionary format
                    row_dict = {k: str(v) for k, v in row.items() if not pd.isna(v)}

                    # Generate all table.column.value strings for this row
                    row_values = [
                        f"{table}.{col}.{value}"
                        for col, value in row_dict.items()
                        if not isinstance(value, bytes)  # Skip binary data
                    ]

                    # Store mapping and embedding
                    if row_values:
                        embedding = self._get_row_embedding(row_dict)
                        index.add(np.array([embedding.astype("float32")]))
                        row_mappings.append(row_values)

            except Exception as e:
                print(f"Error indexing {table}: {str(e)}")

        # Save index and mappings
        index_path = os.path.join(dart_folder, "dartsql.index")
        mappings_path = os.path.join(dart_folder, "mappings.pkl")

        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.device == "cuda" else index,
            index_path,
        )
        with open(mappings_path, "wb") as f:
            pickle.dump(row_mappings, f)

    def query_index(
        self,
        keywords: List[str],
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: DatabaseSqlite = None,
    ):
        """Query implementation following paper's cosine similarity approach"""
        dart_folder = os.path.join(index_path, "DARTSQL")
        index_file = os.path.join(dart_folder, "dartsql.index")
        mappings_file = os.path.join(dart_folder, "mappings.pkl")

        # Load index and mappings
        if dart_folder not in self.loaded_indexes:
            if not os.path.exists(index_file) or not os.path.exists(mappings_file):
                return []

            index = faiss.read_index(index_file)
            if self.device == "cuda":
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            with open(mappings_file, "rb") as f:
                row_mappings = pickle.load(f)

            self.loaded_indexes[dart_folder] = (index, row_mappings)

        index, row_mappings = self.loaded_indexes[dart_folder]
        query_text = keywords[0]  # Full natural language question

        # Compute query embedding
        query_tokens = query_text.lower().split()
        vectors = [self.glove[token] for token in query_tokens if token in self.glove]
        if not vectors:
            return []

        query_embed = np.mean(vectors, axis=0)
        query_embed = query_embed / np.linalg.norm(query_embed)  # Normalize
        query_embed = query_embed.astype("float32")

        # Search with cosine similarity
        distances, indices = index.search(np.array([query_embed]), top_k)

        # Aggregate results from all matching rows
        results = []
        for i in indices[0]:
            if i < len(row_mappings):
                results.extend(row_mappings[i])

        return list(set(results))

