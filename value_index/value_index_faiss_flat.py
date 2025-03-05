from value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from utils.sqlite_db import DatabaseSqlite
from filtering.filtering_abc import FilterABC
from pathlib import Path
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


class FaissFlatIndex(ValueIndexABC, FormattedValuesMixin):
    def __init__(
        self,
        model_used="BAAI/bge-large-en-v1.5",
        delimeter=".",
        per_value=True,
        skip_non_text=False,
    ):
        """
        Initialize FAISS indexer.

        Args:
            model_used: Transformer model for embeddings
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
        self.faiss_flat_indexes = {}
        self.faiss_flat_mappings = {}

    def create_index(
        self, database: DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """
        Create FAISS flat index from database embeddings.

        Args:
            database: Database connection object
            output_path: Output directory for index files
        """
        flat_faiss_folder = os.path.join(output_path, "FlatFaiss")
        os.makedirs(flat_faiss_folder, exist_ok=True)
        print(f"Creating Flat FAISS index in {flat_faiss_folder}")
        dimension = (
            self.model.config.hidden_size
        )  # get the dimension of the embeddings by the size of the hidden layer of the model
        gpu_resources = faiss.StandardGpuResources()  # use the standard gpu resources
        device_id = 0 if torch.cuda.is_available() else -1
        if device_id != -1:
            # if gpu is available, move the index to the gpu
            index = faiss.index_cpu_to_gpu(
                gpu_resources, device_id, faiss.IndexFlatL2(dimension)
            )  # move the index to the gpu if available
        else:
            index = faiss.IndexFlatL2(dimension)
        schema = database.get_tables_and_columns()  # get the schema of the database
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]

        embeddings = []
        mappings = []
        seen = set()  # Track seen values across all tables

        for table in tables:
            try:
                # Use original delimiter for embeddings
                formatted_values = self.get_formatted_values(
                    database, table, skip_non_text_bruteforce=True
                )  # get all unique values in a structured format with their table and column name
                batch_texts = []
                for value in formatted_values:
                    table_name = value["table"]
                    column_name = value["column"]
                    cell_value = value["value"]
                    if not isinstance(cell_value, str):
                        if self.skip_non_text:
                            continue
                        else:
                            try:
                                cell_value = str(cell_value)
                            except:
                                continue
                    entry_id = {
                        "table": table_name,
                        "column": column_name,
                        "value": cell_value,
                        "formatted_value": f"{table_name}{self.delimeter}{column_name}{self.delimeter}{cell_value}",
                    }
                    entry_id_tuple = tuple(
                        sorted(entry_id.items())
                    )  # sort the id so it is hashable and can be used as a key to avoid duplicates
                    if entry_id_tuple not in seen:
                        seen.add(entry_id_tuple)
                        if self.per_value:
                            batch_texts.append(
                                cell_value
                            )  # if per_value is True, then we only store the value without the table and column name
                        else:
                            batch_texts.append(
                                f"{table_name}{self.delimeter}{column_name}{self.delimeter}{cell_value}"
                            )  # otherwise we store the value with the table and column name
                        mappings.append(
                            entry_id
                        )  # append the structured id to the mappings list

                        if (
                            len(batch_texts) >= 512
                        ):  # if the batch size is larger than 512, then we process the batch
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
        if len(embeddings) == 0:
            print("No embeddings found, skipping FAISS index creation")
            return
        # Combine embeddings and create the FAISS index
        embeddings = np.vstack(embeddings).astype("float32")
        index.add(embeddings)

        index_path = os.path.join(flat_faiss_folder, f"faiss.index")

        if device_id != -1:
            faiss.write_index(faiss.index_gpu_to_cpu(index), index_path)
        else:
            faiss.write_index(index, index_path)
        import base64

        def decode_bytes(obj):
            if isinstance(obj, dict):
                return {key: decode_bytes(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [decode_bytes(item) for item in obj]
            elif isinstance(obj, bytes):
                try:
                    return obj.decode("utf-8")  # Try UTF-8 decoding
                except UnicodeDecodeError:
                    return base64.b64encode(obj).decode("utf-8")  # Fallback to Base64
            else:
                return obj

        decoded_mappings = decode_bytes(mappings)

        # Write mappings to a file
        mappings_path = os.path.join(flat_faiss_folder, f"mappings.json")
        with open(mappings_path, "w") as f:
            json.dump(decoded_mappings, f, indent=4)

        print(f"Flat FAISS index created in {flat_faiss_folder}", flush=True)

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
        database: DatabaseSqlite = None,
    ):
        """
        Query FAISS index using vector similarity.

        Args:
            keywords: List of search terms
            index_path: Path containing FAISS index
            top_k: Number of results per query
            filter_instance: Optional filter for results

        """
        original_index_path = index_path
        index_path = os.path.join(index_path, "FlatFaiss", f"faiss.index")
        if not os.path.exists(index_path):
            print(f"Flat FAISS index not found for in: {index_path}. Skipping.")
            return []
        if index_path not in self.faiss_flat_indexes:
            index = faiss.read_index(index_path)
            if self.device.type == "cuda":
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            self.faiss_flat_indexes[index_path] = index
        else:
            index = self.faiss_flat_indexes[index_path]
        mappings_path = os.path.join(original_index_path, "FlatFaiss", f"mappings.json")
        if mappings_path not in self.faiss_flat_mappings:
            with open(mappings_path, "r") as f:
                value_mappings = json.load(f)  # Load JSON mappings
            self.faiss_flat_mappings[mappings_path] = (
                value_mappings  # Cache the mappings
            )
        else:
            value_mappings = self.faiss_flat_mappings[mappings_path]
        results = []
        for keyword in keywords:
            query_embedding = self.encode_text([keyword]).astype("float32")
            distances, indices = index.search(query_embedding, top_k)
            # add the results that only pass the faiss distance threshold
            result_data = []
            for idx, i in enumerate(indices[0]):
                mapping_entry = value_mappings[i]
                table_name = mapping_entry["table"]
                column_name = mapping_entry["column"]
                cell_value = mapping_entry["value"]
                reconstructed = f"{table_name}.{column_name}.{cell_value}"
                # if cell value is a number then skip
                if self.is_number(cell_value):
                    continue
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
