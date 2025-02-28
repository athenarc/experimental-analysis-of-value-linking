from darelabdb.nlp_value_linking.value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
import os
from pathlib import Path
import json
from rapidfuzz.distance import JaroWinkler

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class PicklistIndex(ValueIndexABC, FormattedValuesMixin):
    """Index for categorical values using Jaro Winkler distance."""

    def __init__(self, per_value=False, categorical_size=100, threshold=0.3):
        """
        Initialize picklist indexer.

        Args:
            per_value: Not used (maintained for interface consistency)
            categorical_size: Max unique values considered categorical
            threshold: Jaro-Winkler similarity threshold for fuzzy matching
        """
        self.per_value = per_value
        self.categorical_size = categorical_size
        self.picklist_indexes = {}
        self.filter_threshold = threshold

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create picklist of categorical values.

        Args:
            database: Database connection object
            output_path: Output directory for picklist file
        """
        picklist_json_path = os.path.join(output_path, "picklist.json")
        os.makedirs(output_path, exist_ok=True)

        # Retrieve all tables in the database
        schema = database.get_tables_and_columns()  # get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]
        picklist_values = []

        for table in tables:
            columns = [
                col.split(".")[1]
                for col in schema["columns"]
                if col.startswith(f"{table}.")
            ]
            for column in columns:
                # Count the unique values in the column
                query = f"SELECT `{column}` AS value, COUNT(*) AS count FROM `{table}` GROUP BY `{column}`;"
                results = database.execute(query, limit=-1)
                value_counts = results.set_index("value")["count"].to_dict()

                if len(value_counts) > self.categorical_size:
                    # Skip columns with more than `categorical_size` unique values
                    continue

                for cell_value, count in value_counts.items():
                    if (
                        cell_value is None
                        or str(cell_value).isdigit()
                        or isinstance(cell_value, bytes)
                    ):
                        # skip None, numeric and binary values
                        continue

                    formatted_value = str(cell_value).strip()
                    # storing formatted value here is overkill, however we do it in case we want to change the format of the value in the future
                    picklist_values.append(
                        {
                            "table": table,
                            "column": column,
                            "value": cell_value,
                            "formatted_value": formatted_value,
                        }
                    )

        # Save the picklist to a JSON file
        with open(picklist_json_path, "w") as f:
            json.dump(picklist_values, f, indent=2)
        print(f"Picklist created in {picklist_json_path}")

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        """
        Query picklist using fuzzy string matching.

        Args:
            keywords: List of search terms
            index_path: Path containing picklist file
            top_k: Number of results to return
            filter_instance: Not used (maintained for interface consistency)

        """
        picklist_path = os.path.join(index_path, "picklist.json")
        if not os.path.exists(picklist_path):
            print(f"Picklist index not found in: {index_path}. Skipping.")
            return []
        if picklist_path not in self.picklist_indexes:
            with open(picklist_path, "r") as f:
                picklist = json.load(f)
                self.picklist_indexes[picklist_path] = picklist
        else:
            picklist = self.picklist_indexes[picklist_path]
        results = []
        for keyword in keywords:
            for value in picklist:
                table_name = value["table"]
                column_name = value["column"]
                cell_value = value["value"]
                reconstructed = f"{table_name}.{column_name}.{cell_value}"
                formatted_value = value["formatted_value"]
                if formatted_value is None:
                    continue
                jaro_score = JaroWinkler.similarity(
                    str(keyword).lower(), str(formatted_value).lower()
                )
                if jaro_score > self.filter_threshold:
                    results.append((reconstructed, jaro_score))

        results = sorted(results, key=lambda x: x[1], reverse=True)
        top_values = [result[0] for result in results[:top_k]]
        return top_values
