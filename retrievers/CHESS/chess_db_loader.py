import hashlib
import os
import sqlite3
from typing import Any, List

from darelabdb.nlp_retrieval.core.models import SearchableItem
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader
from tqdm import tqdm


def _is_number(s: Any) -> bool:
    """Helper function to check if a value is numeric."""
    if isinstance(s, (int, float)):
        return True
    if isinstance(s, str):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


class ChessDBLoader(BaseLoader):
    """
    Loads data by extracting unique, non-numeric string values from a
    CHESS-style SQLite database, excluding common identifier columns.
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
        Loads unique, non-numeric string values from the database and converts
        them to SearchableItem objects.
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
                
            # Get all column names, no longer filtering for TEXT type
            all_columns = [info[1] for info in columns_info]

            # For each column, get distinct values
            for col_name in all_columns:
                # Keep the original column name filtering logic
                if any(
                    keyword in col_name.lower()
                    for keyword in ["_id", " id", "url", "email", "phone", "date"]
                ) or col_name.lower().endswith("id"):
                    continue

                distinct_values = self._execute_sql(
                    f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL;'
                )

                for row in distinct_values:
                    value = row[0]

                    # NEW: Check if the value is a non-numeric string, like in OmniSQLLoader
                    if not isinstance(value, str) or _is_number(value):
                        continue

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