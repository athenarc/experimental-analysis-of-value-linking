from typing import List, Union

from nlp_retrieval.core.models import SearchableItem
from nlp_retrieval.loaders.database_loader import (
    DatabaseLoader,
    SerializationStrategy,
)
from nlp_retrieval.loaders.loader_abc import BaseLoader
from nlp_retrieval.sqlite_db import DatabaseSqlite


class BridgeLoader(BaseLoader):
    """
    A specific data loader for the BRIDGE value retrieval system.

    This class acts as a wrapper around the generic `DatabaseLoader`,
    pre-configured with the `VALUE_LEVEL` serialization strategy.
    This strategy is essential for creating the `SearchableItem`s that the
    `BridgeValueRetriever` expects, where each item represents a unique
    database cell value along with its schema context.
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