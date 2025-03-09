from value_index.value_index_abc import ValueIndexABC
from utils.sqlite_db import DatabaseSqlite
from filtering.filtering_abc import FilterABC
import os
import shutil
from pathlib import Path
from pyserini.search.lucene import LuceneSearcher
import json

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class BM25Index(ValueIndexABC):
    """BM25 indexing implementation using Pyserini/Lucene."""

    def __init__(self, per_value=True, delimeter="."):
        """
        Initialize BM25 indexer.

        Args:
            per_value: If True, index only values without table/column context
            delimeter: Separator for table.column.value formatting
        """
        self.per_value = per_value
        self.delimeter = delimeter
        self.bm25_indexes = {}

    def create_index(
        self, database:  DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create BM25 index from database values. Generates JSON documents and builds
        Lucene index using Pyserini.

        Args:
            database: Database connection object
            output_path: Output directory for index files
        """
        temp_json_dir = os.path.join(output_path, "temp_db_index")
        index_path = os.path.join(output_path, "bm25_index")
        #if index path exists, skip the creation
        if os.path.exists(index_path):
            print(f"BM25 index already exists at {index_path}. Skipping.")
            return
        os.makedirs(temp_json_dir, exist_ok=True)
        print(f"Creating BM25 index at {index_path}")
        schema = database.get_tables_and_columns()  # get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]

        all_column_contents = []
        for table_name in tables:
            column_names_in_one_table = [
                col.split(".")[1]
                for col in schema["columns"]
                if col.startswith(f"{table_name}.")
            ]
            for column_name in column_names_in_one_table:
                column_contents = database.execute(
                    f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL;',
                    limit=-1,
                )  # get all unique values in the column
                # column_contents = [str(row[0]).strip() for row in column_contents]  #extract the values and remove any leading or trailing whitespaces
                column_contents = column_contents[column_name].tolist()
                column_contents = [str(row).strip() for row in column_contents]
                for c_id, column_content in enumerate(
                    column_contents
                ):  # iterate over the values and create a json object for each value
                    if len(column_content) != 0:
                        if (
                            self.per_value
                        ):  # this means that we only store the value without the table and column name as the indexed content
                            all_column_contents.append(
                                {
                                    "id": f"{table_name}-**-{column_name}-**-{c_id}".lower(),  # create a unique id for the value from which we can retrieve the table and column name
                                    "contents": column_content,
                                }
                            )
                        else:  # otherwise we store the value with the table and column name
                            connten_to_append = f"{table_name}{self.delimeter}{column_name}{self.delimeter}{column_content}"  # use the delimeter to separate the table, column and value
                            all_column_contents.append(
                                {
                                    "id": f"{table_name}-**-{column_name}-**-{column_content}".lower(),  # in this case we also keep as id the content so that we can retrieve it later
                                    "contents": connten_to_append,
                                }
                            )
        json_file_path = os.path.join(temp_json_dir, "contents.json")
        with open(json_file_path, "w") as f:
            json.dump(all_column_contents, f, indent=2, ensure_ascii=True)
        cmd = (
            f"python -m pyserini.index.lucene --collection JsonCollection --input {temp_json_dir} "
            f"--index {index_path} --generator DefaultLuceneDocumentGenerator --threads 16 "
            f"--storePositions --storeDocvectors --storeRaw"
        )
        try:
            result = os.system(cmd)
            if result != 0:
                print(f"Error during BM25 index creation: {result}")
        except Exception as e:
            print(f"Error during BM25 index creation: {e}")
        finally:
            shutil.rmtree(temp_json_dir)

        print(f"BM25 index created in {output_path}")

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: DatabaseSqlite = None,
    ):
        """
        Query BM25 index using keyword search.

        Args:
            keywords: List of search terms
            index_path: Path containing BM25 index
            top_k: Number of results per keyword
            filter_instance: Optional filter for results
        """
        index_path = os.path.join(index_path, "bm25_index")
        results = []
        if not os.path.exists(index_path):
            print(f"BM25 index not found for in {index_path}. Skipping.")
            return results
        if index_path not in self.bm25_indexes:
            searcher = LuceneSearcher(index_path)
            self.bm25_indexes[index_path] = searcher
        else:
            searcher = self.bm25_indexes[index_path]
        for keyword in keywords:
            hits = searcher.search(keyword, k=top_k)
            result_data = []
            for hit in hits:
                # Following CodeS
                to_append = ""
                value = ""
                tc_name = ""
                if self.per_value:
                    matched_result = json.loads(searcher.doc(hit.docid).raw())
                    tc_name = ".".join(matched_result["id"].split("-**-")[:2])
                    table_name, column_name = tc_name.split(".")
                    # append to result data a string with "table.colmn.contents"
                    to_append = (
                        f"{table_name}.{column_name}.{matched_result['contents']}"
                    )
                    value = matched_result["contents"]
                else:
                    matched_result = json.loads(searcher.doc(hit.docid).raw())
                    tc_name = matched_result["id"].split("-**-")
                    table_name, column_name = tc_name[:2]
                    value = "-**-".join(
                        tc_name[2:]
                    )  # Reconstruct the value part if it includes '-**-'
                    to_append = f"{table_name}.{column_name}.{value}"
                if filter_instance != None:
                    filter_instance.add_pair(keyword, (value, to_append))
                else:
                    result_data.append(to_append)
            if filter_instance == None:
                results.extend(result_data)
        if filter_instance != None:
            return list(set(filter_instance.filter()))
        else:
            return list(set(results))
