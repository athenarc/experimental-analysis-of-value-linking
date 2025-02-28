from darelabdb.nlp_value_linking.value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
import os
from pathlib import Path
from ragatouille import RAGPretrainedModel

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class ColbertIndex(ValueIndexABC, FormattedValuesMixin):
    def __init__(
        self,
        model="jinaai/jina-colbert-v2",
        delimeter=".",
        per_value=False,
        skip_non_text=True,
    ):
        """
        Initialize ColBERT indexer.

        Args:
            model: Pretrained ColBERT model name/path
            delimeter: Separator for table.column.value formatting
            per_value: Index values without table/column context
            skip_non_text: Skip non-text columns
        """
        self.model = model
        self.per_value = per_value
        self.skip_non_text = skip_non_text
        self.delimeter = delimeter
        self.colbert_indexes = {}

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create ColBERT index from database values.

        Args:
            database: Database connection object
            output_path: Output directory for index files
        """
        RAG = RAGPretrainedModel.from_pretrained(self.model)
        colbert_folder = os.path.join(output_path, "Colbert")
        os.makedirs(colbert_folder, exist_ok=True)
        print(f"Creating Colbert index in {colbert_folder}")

        schema = database.get_tables_and_columns()  # Get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]
        docs = []
        docs_ids = []

        for table in tables:
            try:
                formatted_values = self.get_formatted_values(
                    database, table, skip_non_text_bruteforce=True
                )
                for value in formatted_values:
                    table_name = value["table"]
                    column_name = value["column"]
                    cell_value = value["value"]

                    # Ensure cell_value is converted to a string
                    cell_value_str = str(cell_value)

                    if self.per_value:
                        formatted_value = cell_value_str
                    else:
                        formatted_value = f"table : {table_name} column : {column_name} value : {cell_value_str}"

                    docs.append(formatted_value)
                    docs_ids.append(f"{table_name}.{column_name}.{cell_value_str}")
            except Exception as e:
                print(f"Error processing table {table}: {e}")

        # Ensure all elements in docs are strings before indexing
        docs = [str(doc) for doc in docs]

        index_path = os.path.join(colbert_folder, "ColbertRAG")
        RAG.index(
            index_name=index_path,
            collection=docs,
            document_ids=docs_ids,
            use_faiss=True,
        )
        print(f"Colbert index created in {colbert_folder}")

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        """
        Query ColBERT index using semantic search.

        Args:
            keywords: List of search terms
            index_path: Path containing ColBERT index
            top_k: Number of results per query
            filter_instance: Optional filter for results

        Returns:
            List of matching values in 'table.column.value' format
        """
        index_path = os.path.join(index_path, "Colbert", "ColbertRAG")
        if index_path not in self.colbert_indexes:
            self.colbert_indexes[index_path] = RAGPretrainedModel.from_index(index_path)
        RAG = self.colbert_indexes[index_path]
        results = []
        for keyword in keywords:
            res = RAG.search(keyword, k=top_k)
            result_data = []
            for x in res:
                if filter_instance != None:
                    if self.per_value:
                        value = x["content"]
                    else:
                        content = x["content"]
                        value = content.split("value : ")[1].strip()
                    filter_instance.add_pair(keyword, (value, x["document_id"]))
                else:
                    result_data.append(x["document_id"])
            if filter_instance == None:
                results.extend(result_data)
        if filter_instance != None:
            return list(set(filter_instance.filter()))
        else:
            return list(set(results))
