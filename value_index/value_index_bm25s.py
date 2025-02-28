from darelabdb.nlp_value_linking.value_index.value_index_abc import ValueIndexABC
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
import bm25s
from bm25s.tokenization import Tokenizer
from pathlib import Path
import os
import pickle
from typing import List

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class BM25sIndex(ValueIndexABC):
    def __init__(self, per_value=False, delimiter="."):
        self.per_value = per_value
        self.delimiter = delimiter
        self.bm25_retrievers = {}  # Cache for loaded BM25 retrievers
        self.metadata_cache = {}  # Cache for loaded metadata
        self.tokenizer = Tokenizer()  # Default tokenizer

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        # Prepare paths and directories
        index_dir = os.path.join(output_path, "bm25s_index")
        os.makedirs(index_dir, exist_ok=True)

        # Get database schema and prepare data
        schema = database.get_tables_and_columns()
        tables = [t for t in schema["tables"] if t != "sqlite_sequence"]

        corpus = []
        metadata = []

        # Process each table and column
        for table in tables:
            columns = [
                col.split(".")[1]
                for col in schema["columns"]
                if col.startswith(f"{table}.")
            ]
            for col in columns:
                # Get distinct values from database
                result = database.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL;',
                    limit=-1,
                )
                values = result[col].tolist()

                for val in values:
                    val_str = str(val).strip()
                    if not val_str:
                        continue

                    if self.per_value:
                        # Store value in corpus and maintain metadata
                        corpus.append(val_str)
                        metadata.append((table, col, val_str))
                    else:
                        # Create combined corpus entry
                        corpus_entry = (
                            f"{table}{self.delimiter}{col}{self.delimiter}{val_str}"
                        )
                        corpus.append(corpus_entry)

        # Tokenize corpus
        corpus_tokens = bm25s.tokenize(corpus)

        # Create and index BM25 model
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # Save index and metadata
        index_path = os.path.join(index_dir, "index")
        if self.per_value:
            # Save metadata separately
            with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            retriever.save(index_path)
        else:
            # Save with corpus for direct document retrieval
            retriever.save(index_path, corpus=corpus)

        print(f"BM25s index created at {index_dir}")

    def query_index(
        self,
        keywords: List[str],
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        index_dir = os.path.join(index_path, "bm25s_index")
        results = []

        if not os.path.exists(index_dir):
            print(f"BM25s index not found at {index_dir}")
            return results

        # Load BM25 retriever
        if index_dir not in self.bm25_retrievers:
            index_file = os.path.join(index_dir, "index")
            load_corpus = not self.per_value
            retriever = bm25s.BM25.load(index_file, load_corpus=load_corpus)
            self.bm25_retrievers[index_dir] = retriever
        retriever = self.bm25_retrievers[index_dir]

        # Load metadata if needed
        if self.per_value and index_dir not in self.metadata_cache:
            metadata_file = os.path.join(index_dir, "metadata.pkl")
            with open(metadata_file, "rb") as f:
                self.metadata_cache[index_dir] = pickle.load(f)

        # Process each query keyword
        for keyword in keywords:
            query_tokens = bm25s.tokenize(keyword)

            if self.per_value:
                # Retrieve document IDs and use metadata
                if not query_tokens:
                    continue
                doc_ids, _ = retriever.retrieve(
                    query_tokens,
                    k=top_k,
                    show_progress=False,
                    n_threads=-1,
                    sorted=False,
                )
                metadata = self.metadata_cache.get(index_dir, [])

                for doc_id in doc_ids[0]:  # Assuming single query
                    if doc_id < len(metadata):
                        table, col, val = metadata[doc_id]
                        result_str = (
                            f"{table}{self.delimiter}{col}{self.delimiter}{val}"
                        )
                        if filter_instance:
                            filter_instance.add_pair(keyword, (val, result_str))
                        else:
                            results.append(result_str)
            else:
                if not query_tokens:
                    continue
                docs, _ = retriever.retrieve(
                    query_tokens, k=top_k, return_as="documents"
                )
                for doc in docs[0]:  # Assuming single query
                    parts = doc.split(self.delimiter)
                    if len(parts) >= 3:
                        val = self.delimiter.join(parts[2:])
                        result_str = (
                            f"{parts[0]}{self.delimiter}{parts[1]}{self.delimiter}{val}"
                        )
                        if filter_instance:
                            filter_instance.add_pair(keyword, (val, result_str))
                        else:
                            results.append(result_str)

        # Return filtered or direct results
        return list(set(filter_instance.filter() if filter_instance else results))
