from value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from utils.sqlite_db import DatabaseSqlite
from filtering.filtering_abc import FilterABC
from datasketch import MinHash, MinHashLSHForest
import pickle
from pathlib import Path
import os
import json

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class MinHashForestIndex(ValueIndexABC, FormattedValuesMixin):
    """MinHash LSH Forest implementation for approximating Jacard similarity."""

    def __init__(
        self,
        minhash_signature_size=128,
        per_value=False,
        skip_non_text=True,
        delimeter=".",
    ):
        """
        Initialize MinHash LSH Forest.

        Args:
            minhash_signature_size: Size of MinHash signatures
            per_value: Index values without table/column context
            skip_non_text: Skip non-text columns
            delimeter: Separator for table.column.value formatting
        """
        self.minhash_signature_size = minhash_signature_size
        self.per_value = per_value
        self.skip_non_text = skip_non_text
        self.delimeter = delimeter
        self.min_hash_indexes = {}

    def create_index(
        self, database: DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create MinHash LSH Forest index from database values.

        Args:
            database: Database connection object
            output_path: Output directory for index files
        """
        minhash_folder = os.path.join(output_path, "MinHashLSH")
        os.makedirs(minhash_folder, exist_ok=True)
        print(f"Creating MinHashLSH index in {minhash_folder}")
        schema = database.get_tables_and_columns()  # get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]
        forest = MinHashLSHForest(
            num_perm=self.minhash_signature_size
        )  # we use forest instead of simple LSH to retrieve the top-k nearest neighbors instead of a threshold based retrieval
        minhash_objects = {}
        for table in tables:
            formatted_values = self.get_formatted_values(
                database, table, skip_non_text_bruteforce=True
            )  # get all unique values in a structured format with their table and column name
            for value in formatted_values:
                table_name = value["table"]
                column_name = value["column"]
                cell_value = value["value"]
                # if cell_value is integer, skip
                if isinstance(cell_value, int):
                    continue
                if isinstance(cell_value, bytes):  # <--- FIX ADDED HERE
                    continue
                if isinstance(cell_value, float):
                    continue
                if self.per_value:
                    tokens = cell_value.split()  # Tokenize value into words
                else:
                    formatted_value = f"{table_name}{self.delimeter}{column_name}{self.delimeter}{cell_value}"
                    tokens = formatted_value.split()
                minhashvalue = MinHash(
                    num_perm=self.minhash_signature_size
                )  # initialize the minhash object
                for token in tokens:
                    minhashvalue.update(
                        token.encode("utf-8")
                    )  # update the minhash object with the token for all tokens in the value
                entry_id = json.dumps(
                    {  # construct a structured id for the minhash object
                        "table": table_name,
                        "column": column_name,
                        "value": cell_value,
                    },
                    separators=(",", ":"),
                )
                minhash_objects[entry_id] = (
                    minhashvalue  # store the minhash object in a dictionary with the structured id as key
                )
                forest.add(
                    entry_id, minhashvalue
                )  # add the minhash object to the forest

        forest.index()
        lsh_file_path = os.path.join(minhash_folder, "lsh_index.pkl")
        with open(lsh_file_path, "wb") as f:
            pickle.dump((forest, minhash_objects), f)
        print(f"MinHashLSH index created in {minhash_folder}")

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: DatabaseSqlite = None,
    ):
        """
        Query MinHash Forest for similar values.

        Args:
            keywords: List of search terms
            index_path: Path containing MinHash index
            top_k: Number of results per keyword
            filter_instance: Optional filter for results

        """
        lsh_path = os.path.join(index_path, "MinHashLSH", "lsh_index.pkl")
        if not os.path.exists(lsh_path):
            print(f"MinHashLSH index not found in: {index_path}. Skipping.")
            return []
        if lsh_path not in self.min_hash_indexes:
            with open(lsh_path, "rb") as f:
                lsh, minhash_objects = pickle.load(f)  # Unpack the tuple
                self.min_hash_indexes[lsh_path] = (lsh, minhash_objects)
        else:
            lsh, minhash_objects = self.min_hash_indexes[lsh_path]

        results = []
        for keyword in keywords:
            query_minhash = MinHash(num_perm=self.minhash_signature_size)
            for token in keyword.split():
                query_minhash.update(token.encode("utf-8"))
            nearest_neighbors = lsh.query(query_minhash, top_k)  # Query the forest
            result_data = []
            if nearest_neighbors:
                for neighbor in nearest_neighbors:
                    neighbor_data = json.loads(neighbor)
                    table_name = neighbor_data["table"]
                    column_name = neighbor_data["column"]
                    cell_value = neighbor_data["value"]
                    to_append = f"{table_name}.{column_name}.{cell_value}"
                    if filter_instance != None:
                        filter_instance.add_pair(keyword, (cell_value, to_append))
                    else:
                        result_data.append(to_append)
                if filter_instance == None:
                    results.extend(result_data)

        if filter_instance != None:
            return list(set(filter_instance.filter()))
        else:
            return list(set(results))
