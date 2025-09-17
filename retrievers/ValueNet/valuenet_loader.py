from typing import List

from darelabdb.nlp_retrieval.core.models import SearchableItem
from darelabdb.nlp_retrieval.loaders.database_loader import (
    DatabaseLoader,
    SerializationStrategy,
)
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite


class ValueNetLoader(BaseLoader):
    """
    Loads all unique, non-null cell values from a database.
    """

    def __init__(self, db_path: str):
        """
        Initializes the loader for a specific SQLite database.

        Args:
            db_path: The full path to the SQLite database file.
        """
        db_connection = DatabaseSqlite(db_path)
        self.internal_loader = DatabaseLoader(
            db=db_connection, strategy=SerializationStrategy.VALUE_LEVEL
        )

    def load(self) -> List[SearchableItem]:
        return self.internal_loader.load()