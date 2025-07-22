# chess_db_loader.py

import hashlib
import os
import sqlite3
from typing import List

from darelabdb.nlp_retrieval.core.models import SearchableItem
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader
from tqdm import tqdm


class ChessDBLoader(BaseLoader):
    """
    Loads data by extracting unique values from a CHESS-style SQLite database.
    """

    def __init__(self, db_directory_path: str):
        """
        Initializes the loader with the path to the database directory.

        Args:
            db_directory_path: The path to the directory containing the
        """
        db_id = os.path.basename(db_directory_path)
        self.db_path = os.path.join(db_directory_path, f"{db_id}.sqlite")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"SQLite database not found at expected path: {self.db_path}"
            )

    def _execute_sql(self, query: str) -> List:
        """A simple utility to execute a SQL query and fetch all results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.text_factory = lambda b: b.decode(errors='ignore')
            cursor = conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()


    def load(self) -> List[SearchableItem]:
        """
        Loads unique values from the database and converts them to SearchableItem objects.
        """
        items: List[SearchableItem] = []
        
        # 1. Get all table names
        tables = self._execute_sql(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        table_names = [
            table[0] for table in tables if table[0] != "sqlite_sequence"
        ]

        pbar_desc = f"Loading unique values from {os.path.basename(self.db_path)}"
        for table_name in tqdm(table_names, desc=pbar_desc):
            columns_info = self._execute_sql(f'PRAGMA table_info("{table_name}")')
                
            # Filter for text columns, excluding common ID/key columns
            text_columns = [
                info[1] for info in columns_info
                if "TEXT" in info[2].upper()
            ]

            # 3. For each text column, get distinct values
            for col_name in text_columns:
                # Skip columns that are likely identifiers or noisy
                if any(
                    keyword in col_name.lower()
                    for keyword in ["_id", " id", "url", "email", "phone", "date"]
                ) or col_name.lower().endswith("id"):
                    continue

                distinct_values = self._execute_sql(
                    f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL;'
                )

                for row in distinct_values:
                    value = str(row[0])
                    if not value.strip():
                        continue
                    
                    # Create a stable, unique ID for the item
                    value_hash = hashlib.md5(value.encode()).hexdigest()
                    item_id = f"{table_name}.{col_name}.{value_hash}"

                    items.append(
                        SearchableItem(
                            item_id=item_id,
                            content=value,
                            metadata={"table": table_name, "column": col_name, "value": value},
                        )
                    )
        return items