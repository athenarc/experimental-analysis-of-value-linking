from typing import List
from darelabdb.nlp_retrieval.core.models import SearchableItem
from darelabdb.nlp_retrieval.loaders.loader_abc import BaseLoader
from darelabdb.nlp_retrieval.loaders.database_loader import DatabaseLoader, SerializationStrategy
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite

class OpenSearchValueLoader(BaseLoader):
    """
    A wrapper loader for the OpenSearch pipeline that extracts all unique,
    non-numeric string values from a specified database.
    """
    def __init__(self, db_path: str, db_id: str):
        """
        Initializes the loader for a specific database.

        Args:
            db_path (str): The full path to the SQLite database file.
        """
        db_connection = DatabaseSqlite(db_path)
        self.internal_loader = DatabaseLoader(
            db=db_connection,
            strategy=SerializationStrategy.VALUE_LEVEL
        )
    def load(self) -> List[SearchableItem]:
        """
        Executes the value extraction process by calling the wrapped loader.

        Returns:
            A list of `SearchableItem` objects, where each item's content
            is a unique value from the database.
        """
        return self.internal_loader.load()