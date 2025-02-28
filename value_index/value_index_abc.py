from abc import ABC, abstractmethod
import os
import time
from pathlib import Path
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from typing import List
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
from darelabdb.nlp_value_linking.cvr_extractor.cvr_extractor_abc import CVRExtractorABC


INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class ValueIndexABC(ABC):
    """Abstract base class defining the interface for value indexing strategies."""

    @abstractmethod
    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create an index from database values.

        Args:
            database: Database connection object
            output_path: Path to store the created index files
        """
        pass

    @abstractmethod
    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        """
        Query the index for similar values.

        Args:
            keywords: Search terms to query
            index_path: Path containing index files
            top_k: Number of top results to return
            filter_instance: Optional filter to apply to results
            database: Database connection object (optional)

        Returns:
            List of matching values in 'table.column.value' format
        """
        pass


class FormattedValuesMixin:
    """Mixin class providing methods for extracting structured values from databases."""

    def get_formatted_values(
        self, database: Database | DatabaseSqlite, table, skip_non_text_bruteforce=False
    ):
        """
        Extract and format values from a database table.

        Args:
            database: Database connection object
            table: Name of table to process
            skip_non_text_bruteforce: Whether to skip non-text columns

        Returns:
            List of dictionaries containing table, column, and value information
        """
        formatted_values = []
        skip_non_text = self.skip_non_text or skip_non_text_bruteforce

        try:
            if (
                hasattr(database, "config")
                and hasattr(database.config, "type")
                and database.config.type == "postgresql"
            ):
                query = f"""
                    SELECT column_name AS name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                    AND table_schema = '{database.specific_schema or 'public'}';
                """
                columns_info = database.execute(query, limit=-1)
                columns = columns_info.to_dict("records")
            else:
                columns_info = database.execute(
                    f'PRAGMA table_info("{table}");', limit=-1
                )
                columns = columns_info.to_dict("records")

            # Standardize column type key for consistency
            for col in columns:
                col["column_type"] = col.get("data_type", col.get("type", "")).lower()

            # Filter out non-text columns if required
            if skip_non_text:
                columns = [
                    col
                    for col in columns
                    if col["column_type"].lower() not in {"integer", "date", "real"}
                ]

            for col in columns:
                col_name = col["name"]
                print(f"Processing column {col_name} in table {table}")
                # Use proper quoting for PostgreSQL
                if (
                    hasattr(database, "config")
                    and hasattr(database.config, "type")
                    and database.config.type == "postgresql"
                ):
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL;'
                else:
                    query = f"SELECT DISTINCT `{col_name}` FROM `{table}` WHERE `{col_name}` IS NOT NULL;"

                try:
                    values_df = database.execute(query, limit=-1)
                    values = values_df[col_name].dropna().tolist()
                    for value in values:
                        formatted_values.append(
                            {
                                "table": table,
                                "column": col_name,
                                "value": value,
                            }
                        )
                except Exception as e:
                    print(f"Error processing column {col_name} in table {table}: {e}")
        except Exception as e:
            print(f"Error processing table {table}: {e}")

        return formatted_values


class ValueLinker:
    """Orchestrates multiple indexing strategies for comprehensive value matching."""

    def __init__(
        self,
        index_creators: List[ValueIndexABC],
        keyword_extractor: CVRExtractorABC = None,
    ):
        """
        Initialize ValueLinker with multiple indexing strategies.

        Args:
            index_creators: List of ValueIndexABC implementations
            keyword_extractor: Keyword extraction component
        """
        self.index_creators = index_creators
        self.keyword_extractor = keyword_extractor
        # set timers for the indexes for the index creation and for the query
        self.creation_timers = {}
        self.query_timers = {}
        for index_creator in self.index_creators:
            self.creation_timers[index_creator.__class__.__name__] = 0
            self.query_timers[index_creator.__class__.__name__] = 0

    def create_indexes(
        self, database: Database | DatabaseSqlite, output_path: str = INDEXES_CACHE_PATH
    ):
        """
        Create all configured indexes.

        Args:
            database: Database connection object
            output_path: Base directory for storing indexes
        """
        for index_creator in self.index_creators:
            start = time.time()
            index_creator.create_index(database, output_path)
            end = time.time()
            self.creation_timers[index_creator.__class__.__name__] += end - start

    def query_indexes(
        self,
        input_text: str,
        index_path: str = INDEXES_CACHE_PATH,
        top_k: int = 5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ) -> List[str]:
        """
        Query all indexes and aggregate results.

        Args:
            input_text: Natural language query text
            index_path: Base directory containing indexes
            top_k: Results per index to retrieve
            filter_instance: Filter to apply to aggregated results
            database: Database connection object (optional)

        Returns:
            List of matched values in 'table.column.value' format
        """
        # Extract keywords from the input text
        if self.keyword_extractor is not None:
            keywords = self.keyword_extractor.extract_keywords(input_text)

        # Aggregate results from all indexes
        all_results = []
        for index_creator in self.index_creators:
            start = time.time()

            if filter_instance.__class__.__name__ != "LLMFilter":
                results = index_creator.query_index(
                    keywords=keywords,
                    index_path=index_path,
                    top_k=top_k,
                    filter_instance=filter_instance,
                    database=database,
                )
            else:
                results = index_creator.query_index(
                    keywords=keywords,
                    index_path=index_path,
                    top_k=top_k,
                    filter_instance=None,
                    database=database,
                )
                for res in results:
                    filter_instance.add_pair(input_text, (res, res))
                results = filter_instance.filter()
            end = time.time()
            self.query_timers[index_creator.__class__.__name__] += end - start
            all_results.extend(results)

        results = list(set(all_results))
        # remove results with length more than 100
        results = [result for result in results if len(result) < 100]
        # remove results with length less than 3
        results = [result for result in results if len(result) > 3]
        return results

    def print_timers(self):
        """Print timing statistics for index creation and query operations."""
        print("Index creation times:")
        total_index_creation_time = 0
        total_query_time = 0
        for index_creator, time in self.creation_timers.items():
            print(f"{index_creator}: {time} seconds")
            total_index_creation_time += time
        print("Query times:")
        for index_creator, time in self.query_timers.items():
            print(f"{index_creator}: {time} seconds")
            total_query_time += time
        print(f"Total index creation time: {total_index_creation_time} seconds")
        print(f"Total query time: {total_query_time} seconds")
