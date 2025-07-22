import os
import sqlite3
from typing import Any, List

from func_timeout import func_set_timeout, FunctionTimedOut
from tqdm import tqdm

from darelabdb.nlp_retrieval.core.models import SearchableItem
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader


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


class OmniSQLLoader(BaseLoader):
    """
    Loads data from a single SQLite database file, serializing each string
    value from its cells into a `SearchableItem`.
    """

    def __init__(self, db_file_path: str):
        """
        Initializes the loader.

        Args:
            db_file_path: The path to the SQLite database file.
        """
        if not os.path.exists(db_file_path):
            raise FileNotFoundError(f"Database file not found at: {db_file_path}")
        self.db_file_path = db_file_path

    @staticmethod
    @func_set_timeout(3600)
    def _execute_sql(cursor, sql: str) -> List[tuple]:
        """Executes SQL with a long timeout."""
        cursor.execute(sql)
        return cursor.fetchall()

    def _get_cursor(self) -> sqlite3.Cursor:
        """Gets a cursor for the configured SQLite database."""
        try:
            connection = sqlite3.connect(self.db_file_path, check_same_thread=False)
            connection.text_factory = lambda b: b.decode(errors="ignore")
            return connection.cursor()
        except Exception as e:
            print(f"Error connecting to database: {self.db_file_path}")
            raise e

    def load(self) -> List[SearchableItem]:
        """
        Reads the SQLite database, extracts distinct string values, and converts
        them into a list of `SearchableItem` objects.
        """
        print(f"Loading data from database: {self.db_file_path}")
        cursor = self._get_cursor()

        results = self._execute_sql(
            cursor, "SELECT name FROM sqlite_master WHERE type='table';"
        )

        table_names = [result[0] for result in results]

        all_items: List[SearchableItem] = []
        for table_name in tqdm(table_names, desc="Extracting from tables"):
            if table_name == "sqlite_sequence":
                continue
            results = self._execute_sql(
                cursor, f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}')"
            )


            column_names_in_one_table = [result[0] for result in results]

            for column_name in column_names_in_one_table:
                sql_query = f'SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL;'
                distinct_values = self._execute_sql(cursor, sql_query)

                column_contents = [
                    res[0]
                    for res in distinct_values
                    if isinstance(res[0], str) and not _is_number(res[0])
                ]

                for c_id, content_val in enumerate(column_contents):
                    # remove empty and extremely-long contents
                    if 0 < len(content_val) <= 40:
                        item_id = f"{table_name}-**-{column_name}-**-{c_id}"
                        metadata = {
                            "table": table_name,
                            "column": column_name,
                            "value": content_val
                        }
                        all_items.append(
                            SearchableItem(
                                item_id=item_id,
                                content=content_val,
                                metadata=metadata,
                            )
                        )


        cursor.connection.close()
        print(f"Loaded {len(all_items)} searchable items from the database.")
        return all_items