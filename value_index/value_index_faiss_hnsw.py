from darelabdb.nlp_value_linking.value_index.value_index_abc import ValueIndexABC,FormattedValuesMixin
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import torch
import os
import json

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"

class FaissHNSWFaissIndex(ValueIndexABC, FormattedValuesMixin):
    def __init__(
        self,
        model_used="BAAI/bge-large-en-v1.5",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=50,
        delimeter=".",
        per_value=False,
        skip_non_text=True,
    ):
        """
        Initialize HNSW indexer.

        Args:
            model_used: Transformer model for embeddings
            hnsw_m: HNSW graph parameter
            hnsw_ef_construction: Construction time search scope
            hnsw_ef_search: Query time search scope
            delimeter: Separator for table.column.value formatting
            per_value: Index values without table/column context
            skip_non_text: Skip non-text columns
        """
        self.model_used = model_used
        self.delimeter = delimeter
        self.per_value = per_value
        self.skip_non_text = skip_non_text
        self.model_used = model_used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_used)
        self.model = AutoModel.from_pretrained(self.model_used).to(self.device).half()
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.faiss_hnsw_indexes = {}
        self.faiss_hnsw_mappings = {}

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create HNSW FAISS index from database embeddings.

        Args:
            database: Database connection object
            output_path: Output directory for index files
        """
        hnsw_faiss_folder = os.path.join(output_path, "HNSWFaiss")
        os.makedirs(hnsw_faiss_folder, exist_ok=True)
        print(f"Creating HNSW FAISS index in {hnsw_faiss_folder}")
        dimension = self.model.config.hidden_size
        index = faiss.IndexHNSWFlat(dimension, self.hnsw_m)
        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search

        schema = database.get_tables_and_columns()  # get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]

        embeddings = []
        mappings = []
        seen = set()

        for table in tables:
            try:
                formatted_values = self.get_formatted_values(
                    database, table, skip_non_text_bruteforce=True
                )
                batch_texts = []
                for value in formatted_values:
                    table_name = value["table"]
                    column_name = value["column"]
                    cell_value = value["value"]
                    entry_id = {
                        "table": table_name,
                        "column": column_name,
                        "value": cell_value,
                        "formatted_value": f"{table_name}{self.delimeter}{column_name}{self.delimeter}{cell_value}",
                    }
                    entry_id_tuple = tuple(sorted(entry_id.items()))
                    if entry_id_tuple not in seen:
                        seen.add(entry_id_tuple)
                        if self.per_value:
                            batch_texts.append(cell_value)
                        else:
                            batch_texts.append(
                                f"{table_name}{self.delimeter}{column_name}{self.delimeter}{cell_value}"
                            )
                        mappings.append(entry_id)

                        if len(batch_texts) >= 512:
                            inputs = self.tokenizer(
                                batch_texts,
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                            ).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                                batch_embeddings = (
                                    outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                                )
                                embeddings.append(batch_embeddings)
                            batch_texts = []
                if batch_texts:
                    inputs = self.tokenizer(
                        batch_texts, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_embeddings = (
                            outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        )
                        embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error processing table {table}: {e}")

        embeddings = np.vstack(embeddings).astype("float32")
        index.add(embeddings)

        index_path = os.path.join(hnsw_faiss_folder, f"hnsw.index")
        faiss.write_index(index, index_path)
        mappings_path = os.path.join(hnsw_faiss_folder, f"mappings.json")
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=4)
        print(f"HNSW FAISS index created in {hnsw_faiss_folder}", flush=True)

    def encode_text(self, text, batch_size=256):
        # function that encodes the text using the model from the configuration
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def is_number(self, s: str) -> bool:
        try:
            float(s.replace(",", ""))
            return True
        except:
            return False

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        """
        Query HNSW index using vector similarity.

        Args:
            keywords: List of search terms
            index_path: Path containing HNSW index
            top_k: Number of results per query
            filter_instance: Optional filter for results

        """

        original_index_path = index_path
        index_path = os.path.join(index_path, "HNSWFaiss", f"hnsw.index")
        if not os.path.exists(index_path):
            print(f"HNSW FAISS index not found for in: {index_path}. Skipping.")
            return []
        if index_path not in self.faiss_hnsw_indexes:
            index = faiss.read_index(index_path)
            self.faiss_hnsw_indexes[index_path] = index
        else:
            index = self.faiss_hnsw_indexes[index_path]
        mappings_path = os.path.join(original_index_path, "HNSWFaiss", f"mappings.json")
        if mappings_path not in self.faiss_hnsw_mappings:
            with open(mappings_path, "r") as f:
                value_mappings = json.load(f)  # Load JSON mappings
            self.faiss_hnsw_mappings[mappings_path] = value_mappings
        else:
            value_mappings = self.faiss_hnsw_mappings[mappings_path]
        results = []
        for keyword in keywords:
            query_embedding = self.encode_text([keyword]).astype("float32")
            distances, indices = index.search(query_embedding, top_k)
            result_data = []
            for idx, i in enumerate(indices[0]):
                mapping_entry = value_mappings[i]
                table_name = mapping_entry["table"]
                column_name = mapping_entry["column"]
                cell_value = mapping_entry["value"]
                reconstructed = f"{table_name}.{column_name}.{cell_value}"
                if filter_instance != None:
                    filter_instance.add_pair(keyword, (cell_value, reconstructed))
                else:
                    result_data.append(reconstructed)
            if filter_instance == None:
                results.extend(result_data)
        if filter_instance != None:
            return list(set(filter_instance.filter()))
        else:
            return list(set(results))
