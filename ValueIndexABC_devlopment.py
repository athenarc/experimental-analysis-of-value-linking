from abc import ABC, abstractmethod
import os
from datasketch import MinHash, MinHashLSH, MinHashLSHForest
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import pickle
import time
from mo_future import string_types
from typing import Union, List, Tuple
from darelabdb.utils_database_connector.core import Database
from darelabdb.utils_database_connector.sqlite_db import DatabaseSqlite
import difflib
from typing import List, Optional, Tuple, Dict, Any
from rapidfuzz import fuzz
from tqdm import tqdm
import pandas as pd
import gensim.downloader as api
from darelabdb.nlp_value_linking.value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from darelabdb.nlp_value_linking.filtering.filtering_abc import FilterABC


INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"


    class CHESSIndex(ValueIndexABC, FormattedValuesMixin):
        def __init__(
            self,
            minhash_threshold=0.01,
            minhash_signature_size=100,
            ngrams=4,
            model_used="BAAI/bge-large-en-v1.5",
            edit_distance_threshold=0.3,
            embedding_similarity_threshold=0.6,
        ):
            self.minhash_threshold = minhash_threshold
            self.minhash_signature_size = minhash_signature_size
            self.min_hash_indexes = {}
            self.ngrams = ngrams
            self.model_used = model_used
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_used)
            self.model = AutoModel.from_pretrained(self.model_used).to(self.device).half()
            self.edit_distance_threshold = edit_distance_threshold
            self.embedding_similarity_threshold = embedding_similarity_threshold
            self.minhashes = {}

        def get_unique_values(
            self, database: Database | DatabaseSqlite
        ) -> Dict[str, Dict[str, List[str]]]:
            """
            Retrieves unique text values from the database excluding primary keys.

            Args:
                db_path (str): The path to the SQLite database file.

            Returns:
                Dict[str, Dict[str, List[str]]]: A dictionary containing unique values for each table and column.
            """
            schema = database.get_tables_and_columns()  # get the schema of the database
            table_names = [
                table for table in schema["tables"] if table != "sqlite_sequence"
            ]
            primary_keys = []

            for table_name in table_names:
                if (
                    hasattr(database, "config")
                    and hasattr(database.config, "type")
                    and database.config.type == "postgresql"
                ):
                    # Fetch column information
                    column_query = f"""
                        SELECT column_name AS name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                        AND table_schema = '{database.specific_schema or 'public'}';
                    """
                    columns = database.execute(column_query, limit=-1)

                    # Fetch primary key information
                    pk_query = f"""
                        SELECT a.attname AS column_name
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = '{table_name}'::regclass
                        AND i.indisprimary;
                    """
                    pk_columns = database.execute(pk_query, limit=-1)
                    pk_column_names = [
                        row["column_name"] for _, row in pk_columns.iterrows()
                    ]
                else:
                    # For SQLite, PRAGMA returns the primary key directly
                    columns = database.execute(
                        f'PRAGMA table_info("{table_name}");', limit=-1
                    )
                    pk_column_names = [
                        row["name"] for _, row in columns.iterrows() if row["pk"] > 0
                    ]

                # Check columns against primary keys
                for _, row in columns.iterrows():
                    column_name = row["name"]
                    if column_name in pk_column_names:
                        if column_name.lower() not in [c.lower() for c in primary_keys]:
                            primary_keys.append(column_name)

            unique_values: Dict[str, Dict[str, List[str]]] = {}
            for table_name in table_names:
                if table_name == "sqlite_sequence":
                    continue
                if (
                    hasattr(database, "config")
                    and hasattr(database.config, "type")
                    and database.config.type == "postgresql"
                ):
                    query = f"""
                        SELECT column_name AS name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}'
                        AND table_schema = '{database.specific_schema or 'public'}';
                    """
                    columns = [
                        row["name"]
                        for _, row in database.execute(query, limit=-1).iterrows()
                        if (
                            "text" in row["data_type"]
                            or "character varying" in row["data_type"]
                        )
                        and row["name"].lower() not in [c.lower() for c in primary_keys]
                    ]
                else:
                    columns = [
                        row["name"]
                        for _, row in database.execute(
                            f'PRAGMA table_info("{table_name}");', limit=-1
                        ).iterrows()
                        if "TEXT" in row["type"]
                        and row["name"].lower() not in [c.lower() for c in primary_keys]
                    ]
                table_values: Dict[str, List[str]] = {}
                print(f"Processing columns {columns} in table {table_name}", flush=True)
                for column in columns:
                    if any(
                        keyword in column.lower()
                        for keyword in [
                            "_id",
                            " id",
                            "url",
                            "email",
                            "web",
                            "time",
                            "phone",
                            "date",
                            "address",
                        ]
                    ) or column.endswith("Id"):
                        continue

                    try:
                        if (
                            hasattr(database, "config")
                            and hasattr(database.config, "type")
                            and database.config.type == "postgresql"
                        ):
                            result_df = database.execute(
                                f"""
                                SELECT SUM(LENGTH(unique_values::text)) AS sum_of_lengths, COUNT(unique_values) AS count_distinct
                                FROM (
                                    SELECT DISTINCT "{column}" AS unique_values
                                    FROM "{table_name}"
                                    WHERE "{column}" IS NOT NULL
                                ) AS subquery
                                LIMIT 1
                            """
                            )
                        else:
                            result_df = database.execute(
                                f"""
                                SELECT SUM(LENGTH(unique_values)) AS sum_of_lengths, COUNT(unique_values) AS count_distinct
                                FROM (
                                    SELECT DISTINCT `{column}` AS unique_values
                                    FROM `{table_name}`
                                    WHERE `{column}` IS NOT NULL
                                ) AS subquery
                            """,
                                limit=1,
                            )

                        # Ensure result_df is a DataFrame and parse numeric values
                        if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                            sum_of_lengths = result_df.loc[0, "sum_of_lengths"]
                            count_distinct = result_df.loc[0, "count_distinct"]

                            # Convert to numeric types
                            sum_of_lengths = (
                                int(sum_of_lengths) if pd.notnull(sum_of_lengths) else None
                            )
                            count_distinct = (
                                int(count_distinct) if pd.notnull(count_distinct) else 0
                            )
                        else:
                            sum_of_lengths, count_distinct = None, 0

                    except Exception as e:
                        print(f"Error fetching sum of lengths and distinct count: {e}")
                        sum_of_lengths, count_distinct = None, 0

                    if sum_of_lengths is None or count_distinct == 0:
                        continue

                    average_length = sum_of_lengths / count_distinct

                    average_length = sum_of_lengths / count_distinct

                    if (
                        ("name" in column.lower() and sum_of_lengths < 5000000)
                        or (sum_of_lengths < 2000000 and average_length < 25)
                        or count_distinct < 100
                    ):
                        try:
                            if (
                                hasattr(database, "config")
                                and hasattr(database.config, "type")
                                and database.config.type == "postgresql"
                            ):
                                query = f'SELECT DISTINCT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL;'
                            else:
                                query = f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL;"
                            values_df = database.execute(query, limit=-1)
                            values = values_df[column].dropna().tolist()
                        except:
                            values = []
                        table_values[column] = values

                unique_values[table_name] = table_values
            return unique_values

        def create_minhash(self, signature_size: int, string: str, n_gram: int) -> MinHash:
            """
            Creates a MinHash object for a given string.

            Args:
                signature_size (int): The size of the MinHash signature.
                string (str): The input string to create the MinHash for.
                n_gram (int): The n-gram size for the MinHash.

            Returns:
                MinHash: The MinHash object for the input string.
            """
            m = MinHash(num_perm=signature_size)
            for d in [string[i : i + n_gram] for i in range(len(string) - n_gram + 1)]:
                m.update(d.encode("utf8"))
            return m

        def make_lsh(
            self,
            unique_values: Dict[str, Dict[str, List[str]]],
            signature_size: int,
            n_gram: int,
            threshold: float,
            verbose: bool = True,
        ) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
            """
            Creates a MinHash LSH from unique values.

            Args:
                unique_values (Dict[str, Dict[str, List[str]]]): The dictionary of unique values.
                signature_size (int): The size of the MinHash signature.
                n_gram (int): The n-gram size for the MinHash.
                threshold (float): The threshold for the MinHash LSH.
                verbose (bool): Whether to display progress information.

            Returns:
                Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]: The MinHash LSH object and the dictionary of MinHashes.
            """
            lsh = MinHashLSH(threshold=threshold, num_perm=signature_size)
            minhashes: Dict[str, Tuple[MinHash, str, str, str]] = {}
            total_unique_values = sum(
                len(column_values)
                for table_values in unique_values.values()
                for column_values in table_values.values()
            )

            progress_bar = (
                tqdm(total=total_unique_values, desc="Creating LSH") if verbose else None
            )

            for table_name, table_values in unique_values.items():
                for column_name, column_values in table_values.items():
                    if column_name.lower() == "doctype":
                        print("=" * 20)
                        print("Doctype found")
                        print("=" * 20)

                    for id, value in enumerate(column_values):
                        minhashvalue = self.create_minhash(signature_size, value, n_gram)
                        minhash_key = f"{table_name}.{column_name}.{id}"
                        minhashes[minhash_key] = (
                            minhashvalue,
                            table_name,
                            column_name,
                            value,
                        )
                        lsh.insert(minhash_key, minhashvalue)

                        if verbose:
                            progress_bar.update(1)

            if verbose:
                progress_bar.close()

            return lsh, minhashes

        def create_index(
            self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
        ):

            chess_folder = os.path.join(output_path, "CHESS")
            os.makedirs(chess_folder, exist_ok=True)
            print(f"Creating CHESS index in {chess_folder}")
            unique_values = self.get_unique_values(database)
            print(f"Lenfgth of unique values: {len(unique_values)}")
            with open(os.path.join(chess_folder, "unique_values.pkl"), "wb") as file:
                pickle.dump(unique_values, file)
            lsh, minhashes = self.make_lsh(
                unique_values,
                self.minhash_signature_size,
                self.ngrams,
                self.minhash_threshold,
            )
            lsh_path = os.path.join(chess_folder, "lsh.pkl")
            minhashes_path = os.path.join(chess_folder, "minhashes.pkl")

            with open(lsh_path, "wb") as file:
                pickle.dump(lsh, file)
            with open(minhashes_path, "wb") as file:
                pickle.dump(minhashes, file)

        def _extract_paranthesis(self, string: str) -> List[str]:
            """
            Extracts strings within parentheses from a given string.

            Args:
                string (str): The string to extract from.

            Returns:
                List[str]: A list of strings within parentheses.
            """
            paranthesis_matches = []
            open_paranthesis = []
            for i, char in enumerate(string):
                if char == "(":
                    open_paranthesis.append(i)
                elif char == ")" and open_paranthesis:
                    start = open_paranthesis.pop()
                    found_string = string[start : i + 1]
                    if found_string:
                        paranthesis_matches.append(found_string)
            return paranthesis_matches

        def _column_value(self, string: str) -> Tuple[Optional[str], Optional[str]]:
            """
            Splits a string into column and value parts if it contains '='.

            Args:
                string (str): The string to split.

            Returns:
                Tuple[Optional[str], Optional[str]]: The column and value parts.
            """
            if "=" in string:
                left_equal = string.find("=")
                first_part = string[:left_equal].strip()
                second_part = (
                    string[left_equal + 1 :].strip()
                    if len(string) > left_equal + 1
                    else None
                )
                return first_part, second_part
            return None, None

        def _get_similar_column_names(
            self, keywords: List[str], question: str
        ) -> List[Tuple[str, str]]:
            """
            Finds column names similar to given keywords based on question and hint.

            Args:
                keywords (List[str]): The list of keywords.
                question (str): The question string.
                hint (str): The hint string.

            Returns:
                List[Tuple[str, str]]: A list of tuples containing table and column names.
            """
            potential_column_names = []
            for keyword in keywords:
                keyword = keyword.strip()
                potential_column_names.append(keyword)

                column, value = self._column_value(keyword)
                if column:
                    potential_column_names.append(column)

                potential_column_names.extend(self._extract_paranthesis(keyword))

                if " " in keyword:
                    potential_column_names.extend(part.strip() for part in keyword.split())

            schema = self.database.get_tables_and_columns()

            # Prepare the list of strings to embed
            column_strings = [
                f"`{col.split('.')[0]}`.`{col.split('.')[1]}`" for col in schema["columns"]
            ]
            question_hint_string = f"{question}"

            to_embed_strings = column_strings + [question_hint_string]

            # Tokenize and process strings to embeddings using the custom model
            inputs = self.tokenizer(
                to_embed_strings, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**inputs)

            # Pooling the embeddings (mean pooling of the last hidden states)
            token_embeddings = (
                model_output.last_hidden_state
            )  # Shape: (batch_size, seq_len, hidden_size)
            attention_mask = inputs["attention_mask"]
            embeddings = self._mean_pooling(token_embeddings, attention_mask)

            # Normalize the embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

            # Separate embeddings
            column_embeddings = embeddings[:-1]  # All except the last one
            question_hint_embedding = embeddings[-1]  # The last one

            # Compute similarities
            similar_column_names = []
            for i, column_embedding in enumerate(column_embeddings):
                table, column = column_strings[i].split(".")[0].strip("`"), column_strings[
                    i
                ].split(".")[1].strip("`")
                for potential_column_name in potential_column_names:
                    if self._does_keyword_match_column(potential_column_name, column):
                        similarity_score = np.dot(column_embedding, question_hint_embedding)
                        similar_column_names.append((table, column, similarity_score))

            similar_column_names.sort(key=lambda x: x[2], reverse=True)
            table_column_pairs = list(
                set([(table, column) for table, column, _ in similar_column_names])
            )
            return table_column_pairs

        def _does_keyword_match_column(
            self, keyword: str, column_name: str, threshold: float = 0.9
        ) -> bool:
            """
            Checks if a keyword matches a column name based on similarity.

            Args:
                keyword (str): The keyword to match.
                column_name (str): The column name to match against.
                threshold (float, optional): The similarity threshold. Defaults to 0.9.

            Returns:
                bool: True if the keyword matches the column name, False otherwise.
            """
            keyword = keyword.lower().replace(" ", "").replace("_", "").rstrip("s")
            column_name = column_name.lower().replace(" ", "").replace("_", "").rstrip("s")
            similarity = difflib.SequenceMatcher(None, column_name, keyword).ratio()
            return similarity >= threshold

        def _mean_pooling(self, token_embeddings, attention_mask):
            """
            Apply mean pooling to the token embeddings.
            Args:
                token_embeddings: The embeddings from the model (batch_size, seq_len, hidden_size).
                attention_mask: The attention mask (batch_size, seq_len).
            Returns:
                A tensor of mean-pooled embeddings (batch_size, hidden_size).
            """
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        def get_similar_columns(
            self, keywords: List[str], question: str
        ) -> Dict[str, List[str]]:
            """
            Finds columns similar to given keywords based on question and hint.

            Args:
                keywords (List[str]): The list of keywords.
                question (str): The question string.
                hint (str): The hint string.

            Returns:
                Dict[str, List[str]]: A dictionary mapping table names to lists of similar column names.
            """
            selected_columns = {}
            similar_columns = self._get_similar_column_names(
                keywords=keywords, question=question
            )
            for table_name, column_name in similar_columns:
                if table_name not in selected_columns:
                    selected_columns[table_name] = []
                if column_name not in selected_columns[table_name]:
                    selected_columns[table_name].append(column_name)
            return selected_columns

        def _get_to_search_values(self, keywords: List[str]) -> List[str]:
            """
            Extracts values to search from the keywords.

            Args:
                keywords (List[str]): The list of keywords.

            Returns:
                List[str]: A list of values to search.
            """

            def get_substring_packet(keyword: str, substring: str) -> Dict[str, str]:
                return {"keyword": keyword, "substring": substring}

            to_search_values = []
            for keyword in keywords:
                keyword = keyword.strip()
                to_search_values.append(get_substring_packet(keyword, keyword))
                if " " in keyword:
                    for i in range(len(keyword)):
                        if keyword[i] == " ":
                            first_part = keyword[:i]
                            second_part = keyword[i + 1 :]
                            to_search_values.append(
                                get_substring_packet(keyword, first_part)
                            )
                            to_search_values.append(
                                get_substring_packet(keyword, second_part)
                            )
                hint_column, hint_value = self._column_value(keyword)
                if hint_value:
                    to_search_values.append(get_substring_packet(keyword, hint_value))
            to_search_values.sort(
                key=lambda x: (x["keyword"], len(x["substring"]), x["substring"]),
                reverse=True,
            )
            return to_search_values

        def _jaccard_similarity(self, m1: MinHash, m2: MinHash) -> float:
            """
            Computes the Jaccard similarity between two MinHash objects.

            Args:
                m1 (MinHash): The first MinHash object.
                m2 (MinHash): The second MinHash object.

            Returns:
                float: The Jaccard similarity between the two MinHash objects.
            """
            return m1.jaccard(m2)

        def query_lsh(
            self,
            lsh: MinHashLSH,
            minhashes: Dict[str, Tuple[MinHash, str, str, str]],
            keyword: str,
            signature_size: int = 100,
            n_gram: int = 3,
            top_n: int = 10,
        ) -> Dict[str, Dict[str, List[str]]]:
            """
            Queries the LSH for similar values to the given keyword and returns the top results.

            Args:
                lsh (MinHashLSH): The LSH object.
                minhashes (Dict[str, Tuple[MinHash, str, str, str]]): The dictionary of MinHashes.
                keyword (str): The keyword to search for.
                signature_size (int, optional): The size of the MinHash signature.
                n_gram (int, optional): The n-gram size for the MinHash.
                top_n (int, optional): The number of top results to return.

            Returns:
                Dict[str, Dict[str, List[str]]]: A dictionary containing the top similar values.
            """
            query_minhash = self.create_minhash(signature_size, keyword, n_gram)
            results = lsh.query(query_minhash)
            similarities = [
                (result, self._jaccard_similarity(query_minhash, minhashes[result][0]))
                for result in results
            ]
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

            similar_values_trimmed: Dict[str, Dict[str, List[str]]] = {}
            for result, similarity in similarities:
                table_name, column_name, value = minhashes[result][1:]
                if table_name not in similar_values_trimmed:
                    similar_values_trimmed[table_name] = {}
                if column_name not in similar_values_trimmed[table_name]:
                    similar_values_trimmed[table_name][column_name] = []
                similar_values_trimmed[table_name][column_name].append(value)

            return similar_values_trimmed

        def _get_similar_entities_via_LSH(
            self, substring_packets: List[Dict[str, str]]
        ) -> List[Dict[str, Any]]:
            similar_entities_via_LSH = []
            for packet in substring_packets:
                keyword = packet["keyword"]
                substring = packet["substring"]
                unique_similar_values = self.query_lsh(
                    lsh=self.lsh,
                    minhashes=self.minhashes,
                    keyword=substring,
                    signature_size=self.minhash_signature_size,
                    n_gram=self.ngrams,
                    top_n=10,
                )
                for table_name, column_values in unique_similar_values.items():
                    for column_name, values in column_values.items():
                        for value in values:
                            similar_entities_via_LSH.append(
                                {
                                    "keyword": keyword,
                                    "substring": substring,
                                    "table_name": table_name,
                                    "column_name": column_name,
                                    "similar_value": value,
                                }
                            )
            return similar_entities_via_LSH

        def _get_similar_entities(self, keywords: List[str]) -> List[str]:
            """
            Retrieves similar entities from the database based on keywords.

            Args:
                keywords (List[str]): The list of keywords.

            Returns:
                Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
            """
            to_seartch_values = self._get_to_search_values(keywords)
            similar_entities_via_LSH = self._get_similar_entities_via_LSH(to_seartch_values)
            similar_entities_via_edit_distance = (
                self._get_similar_entities_via_edit_distance(similar_entities_via_LSH)
            )
            similar_entities_via_embedding = self._get_similar_entities_via_embedding(
                similar_entities_via_edit_distance
            )

            selected_values = {}
            for entity in similar_entities_via_embedding:
                table_name = entity["table_name"]
                column_name = entity["column_name"]
                if table_name not in selected_values:
                    selected_values[table_name] = {}
                if column_name not in selected_values[table_name]:
                    selected_values[table_name][column_name] = []
                selected_values[table_name][column_name].append(entity)

            to_return = []
            for table_name, column_values in selected_values.items():
                for column_name, values in column_values.items():
                    max_edit_distance_similarity = max(
                        entity["edit_distance_similarity"] for entity in values
                    )
                    values = [
                        entity
                        for entity in values
                        if entity["edit_distance_similarity"]
                        >= 0.9 * max_edit_distance_similarity
                    ]
                    max_embedding_similarity = max(
                        entity["embedding_similarity"] for entity in values
                    )
                    selected_values[table_name][column_name] = [
                        entity["similar_value"]
                        for entity in values
                        if entity["embedding_similarity"] >= 0.9 * max_embedding_similarity
                    ]
                    # add to_returh all the strings in format 'table_name.column_name.value'
                    to_return.extend(
                        [
                            f"{table_name}.{column_name}.{value}"
                            for value in selected_values[table_name][column_name]
                        ]
                    )
            return to_return

        def _get_similar_entities_via_edit_distance(
            self, similar_entities_via_LSH: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            similar_entities_via_edit_distance_similarity = []
            for entity_packet in similar_entities_via_LSH:
                edit_distance_similarity = difflib.SequenceMatcher(
                    None,
                    entity_packet["substring"].lower(),
                    entity_packet["similar_value"].lower(),
                ).ratio()
                if edit_distance_similarity >= self.edit_distance_threshold:
                    entity_packet["edit_distance_similarity"] = edit_distance_similarity
                    similar_entities_via_edit_distance_similarity.append(entity_packet)
            return similar_entities_via_edit_distance_similarity

        def _get_similar_entities_via_embedding(
            self, similar_entities_via_edit_distance: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            similar_values_dict = {}
            to_embed_strings = []

            # Organize entities by keyword and substring
            for entity_packet in similar_entities_via_edit_distance:
                keyword = entity_packet["keyword"]
                substring = entity_packet["substring"]
                similar_value = entity_packet["similar_value"]

                if substring and similar_value:  # Ensure valid strings
                    if keyword not in similar_values_dict:
                        similar_values_dict[keyword] = {}
                    if substring not in similar_values_dict[keyword]:
                        similar_values_dict[keyword][substring] = []
                        to_embed_strings.append(substring)  # Add substring for embedding
                    similar_values_dict[keyword][substring].append(entity_packet)
                    to_embed_strings.append(
                        similar_value
                    )  # Add similar value for embedding

            # Validate to_embed_strings
            if not to_embed_strings:
                print("to_embed_strings is empty.", flush=True)
                return []

            # Calculate embeddings using the custom model
            inputs = self.tokenizer(
                to_embed_strings, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**inputs)

            # Pooling the embeddings (mean pooling of the last hidden states)
            token_embeddings = model_output.last_hidden_state
            attention_mask = inputs["attention_mask"]
            all_embeddings = self._mean_pooling(token_embeddings, attention_mask)

            # Normalize embeddings for similarity calculation
            all_embeddings = (
                torch.nn.functional.normalize(all_embeddings, p=2, dim=1).cpu().numpy()
            )

            # Match embeddings to entities and calculate similarity scores
            similar_entities_via_embedding_similarity = []
            index = 0
            for keyword, substring_dict in similar_values_dict.items():
                for substring, entity_packets in substring_dict.items():
                    substring_embedding = all_embeddings[index]
                    index += 1
                    similar_values_embeddings = all_embeddings[
                        index : index + len(entity_packets)
                    ]
                    index += len(entity_packets)

                    # Calculate similarity scores
                    similarities = np.dot(similar_values_embeddings, substring_embedding)
                    for i, entity_packet in enumerate(entity_packets):
                        if similarities[i] >= self.embedding_similarity_threshold:
                            entity_packet["embedding_similarity"] = similarities[i]
                            similar_entities_via_embedding_similarity.append(entity_packet)

            return similar_entities_via_embedding_similarity

        def query_index(
            self,
            keywords: str,
            index_path=INDEXES_CACHE_PATH,
            top_k=5,
            filter_instance: FilterABC = None,
            database: Database | DatabaseSqlite = None,
        ):
            if database is None:
                raise ValueError(
                    "The `database` parameter must be provided and cannot be None."
                )
            self.database = database

            question = keywords[0]
            keywords = keywords[1:]
            self.similar_columns = self.get_similar_columns(
                keywords=keywords, question=question
            )

            lsh_path = os.path.join(index_path, "CHESS", "lsh.pkl")
            if not os.path.exists(lsh_path):
                print(f"MinHashLSH index not found in: {index_path}. Skipping.")
                return []
            if lsh_path not in self.min_hash_indexes:
                with open(lsh_path, "rb") as f:
                    self.lsh = pickle.load(f)
                    self.min_hash_indexes[lsh_path] = self.lsh
            else:
                self.lsh = self.min_hash_indexes[lsh_path]

            minhash_path = os.path.join(index_path, "CHESS", "minhashes.pkl")
            if not os.path.exists(minhash_path):
                print(f"MinHash objects not found in: {index_path}. Skipping.")
                return []
            if minhash_path not in self.min_hash_indexes:
                with open(minhash_path, "rb") as f:
                    self.minhashes = pickle.load(f)
                    self.min_hash_indexes[minhash_path] = self.minhashes
            else:
                self.minhashes = self.min_hash_indexes[minhash_path]

            results = self._get_similar_entities(keywords)
            return results


class DartSQLIndex(ValueIndexABC):
    def __init__(self):
        self.glove = self._load_glove()
        self.loaded_indexes = {}  # {index_path: (faiss_index, row_mappings)}
        self.device = "cuda" if faiss.get_num_gpus() > 0 else "cpu"

    def _load_glove(self):
        """Load GloVe embeddings using Gensim"""
        print("Loading GloVe embeddings...")
        return api.load("glove-wiki-gigaword-300")

    def _get_row_embedding(self, row_dict):
        """Convert row dictionary to averaged GloVe embedding exactly as in paper"""
        tokens = []
        for col, value in row_dict.items():
            # Include both column names and cell values as per paper
            tokens += str(col).lower().split() + str(value).lower().split()

        vectors = [self.glove[token] for token in tokens if token in self.glove]
        embedding = np.mean(vectors, axis=0) if vectors else np.zeros(300)
        return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        """Create index following the paper's exact methodology"""
        dart_folder = os.path.join(output_path, "DARTSQL")
        os.makedirs(dart_folder, exist_ok=True)

        # Create FAISS index with inner product (cosine similarity)
        dimension = 300
        index = faiss.IndexFlatIP(dimension)
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        row_mappings = []
        schema = database.get_tables_and_columns()

        for table in schema["tables"]:
            if table == "sqlite_sequence":
                continue

            try:
                rows = database.execute(f'SELECT * FROM "{table}"', limit=-1).to_dict(
                    "records"
                )
                for row in rows:
                    # Convert row to paper's JSON dictionary format
                    row_dict = {k: str(v) for k, v in row.items() if not pd.isna(v)}

                    # Generate all table.column.value strings for this row
                    row_values = [
                        f"{table}.{col}.{value}"
                        for col, value in row_dict.items()
                        if not isinstance(value, bytes)  # Skip binary data
                    ]

                    # Store mapping and embedding
                    if row_values:
                        embedding = self._get_row_embedding(row_dict)
                        index.add(np.array([embedding.astype("float32")]))
                        row_mappings.append(row_values)

            except Exception as e:
                print(f"Error indexing {table}: {str(e)}")

        # Save index and mappings
        index_path = os.path.join(dart_folder, "dartsql.index")
        mappings_path = os.path.join(dart_folder, "mappings.pkl")

        faiss.write_index(
            faiss.index_gpu_to_cpu(index) if self.device == "cuda" else index,
            index_path,
        )
        with open(mappings_path, "wb") as f:
            pickle.dump(row_mappings, f)

    def query_index(
        self,
        keywords: List[str],
        index_path=INDEXES_CACHE_PATH,
        top_k=5,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        """Query implementation following paper's cosine similarity approach"""
        dart_folder = os.path.join(index_path, "DARTSQL")
        index_file = os.path.join(dart_folder, "dartsql.index")
        mappings_file = os.path.join(dart_folder, "mappings.pkl")

        # Load index and mappings
        if dart_folder not in self.loaded_indexes:
            if not os.path.exists(index_file) or not os.path.exists(mappings_file):
                return []

            index = faiss.read_index(index_file)
            if self.device == "cuda":
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

            with open(mappings_file, "rb") as f:
                row_mappings = pickle.load(f)

            self.loaded_indexes[dart_folder] = (index, row_mappings)

        index, row_mappings = self.loaded_indexes[dart_folder]
        query_text = keywords[0]  # Full natural language question

        # Compute query embedding
        query_tokens = query_text.lower().split()
        vectors = [self.glove[token] for token in query_tokens if token in self.glove]
        if not vectors:
            return []

        query_embed = np.mean(vectors, axis=0)
        query_embed = query_embed / np.linalg.norm(query_embed)  # Normalize
        query_embed = query_embed.astype("float32")

        # Search with cosine similarity
        distances, indices = index.search(np.array([query_embed]), top_k)

        # Aggregate results from all matching rows
        results = []
        for i in indices[0]:
            if i < len(row_mappings):
                results.extend(row_mappings[i])

        return list(set(results))


class Match(object):
    def __init__(self, start: int, size: int) -> None:
        self.start = start
        self.size = size


class BRIDGEIndex(ValueIndexABC):
    def __init__(self, top_k=1, threshold=0.85):
        self.picklist_indexes = dict()
        self.match_threshold = threshold
        self.stopwords = {
            "who",
            "ourselves",
            "down",
            "only",
            "were",
            "him",
            "at",
            "weren't",
            "has",
            "few",
            "it's",
            "m",
            "again",
            "d",
            "haven",
            "been",
            "other",
            "we",
            "an",
            "own",
            "doing",
            "ma",
            "hers",
            "all",
            "haven't",
            "in",
            "but",
            "shouldn't",
            "does",
            "out",
            "aren",
            "you",
            "you'd",
            "himself",
            "isn't",
            "most",
            "y",
            "below",
            "is",
            "wasn't",
            "hasn",
            "them",
            "wouldn",
            "against",
            "this",
            "about",
            "there",
            "don",
            "that'll",
            "a",
            "being",
            "with",
            "your",
            "theirs",
            "its",
            "any",
            "why",
            "now",
            "during",
            "weren",
            "if",
            "should",
            "those",
            "be",
            "they",
            "o",
            "t",
            "of",
            "or",
            "me",
            "i",
            "some",
            "her",
            "do",
            "will",
            "yours",
            "for",
            "mightn",
            "nor",
            "needn",
            "the",
            "until",
            "couldn't",
            "he",
            "which",
            "yourself",
            "to",
            "needn't",
            "you're",
            "because",
            "their",
            "where",
            "it",
            "didn't",
            "ve",
            "whom",
            "should've",
            "can",
            "shan't",
            "on",
            "had",
            "have",
            "myself",
            "am",
            "don't",
            "under",
            "was",
            "won't",
            "these",
            "so",
            "as",
            "after",
            "above",
            "each",
            "ours",
            "hadn",
            "having",
            "wasn",
            "s",
            "doesn",
            "hadn't",
            "than",
            "by",
            "that",
            "both",
            "herself",
            "his",
            "wouldn't",
            "into",
            "doesn't",
            "before",
            "my",
            "won",
            "more",
            "are",
            "through",
            "same",
            "how",
            "what",
            "over",
            "ll",
            "yourselves",
            "up",
            "mustn",
            "mustn't",
            "she's",
            "re",
            "such",
            "didn",
            "you'll",
            "shan",
            "when",
            "you've",
            "themselves",
            "mightn't",
            "she",
            "from",
            "isn",
            "ain",
            "between",
            "once",
            "here",
            "shouldn",
            "our",
            "and",
            "not",
            "too",
            "very",
            "further",
            "while",
            "off",
            "couldn",
            "hasn't",
            "itself",
            "then",
            "did",
            "just",
            "aren't",
        }
        self.commonwords = {"no", "yes", "many"}
        self.m_theta = threshold
        self.s_theta = threshold

    def is_number(self, s: str) -> bool:
        try:
            float(s.replace(",", ""))
            return True
        except:
            return False

    def is_stopword(self, s: str) -> bool:
        return s.strip() in self.stopwords

    def is_commonword(self, s: str) -> bool:
        return s.strip() in self.commonwords

    def is_common_db_term(self, s: str) -> bool:
        return s.strip() in ["id"]

    def is_span_separator(self, c: str) -> bool:
        return c in "'\"()`,.?! "

    def split_to_chars(self, s: str) -> List[str]:
        return [c.lower() for c in s.strip()]

    def prefix_match(self, s1: str, s2: str) -> bool:
        i, j = 0, 0
        for i in range(len(s1)):
            if not self.is_span_separator(s1[i]):
                break
        for j in range(len(s2)):
            if not self.is_span_separator(s2[j]):
                break
        if i < len(s1) and j < len(s2):
            return s1[i] == s2[j]
        elif i >= len(s1) and j >= len(s2):
            return True
        else:
            return False

    def get_effective_match_source(self, s: str, start: int, end: int) -> Match:
        _start = -1

        for i in range(start, start - 2, -1):
            if i < 0:
                _start = i + 1
                break
            if self.is_span_separator(s[i]):
                _start = i
                break

        if _start < 0:
            return None

        _end = -1
        for i in range(end - 1, end + 3):
            if i >= len(s):
                _end = i - 1
                break
            if self.is_span_separator(s[i]):
                _end = i
                break

        if _end < 0:
            return None

        while _start < len(s) and self.is_span_separator(s[_start]):
            _start += 1
        while _end >= 0 and self.is_span_separator(s[_end]):
            _end -= 1

        return Match(_start, _end - _start + 1)

    def get_matched_entries(
        self, s: str, field_values: List[str]
    ) -> Optional[List[Tuple[str, Tuple[str, str, float, float, int]]]]:
        if not field_values:
            return None

        if isinstance(s, str):
            n_grams = self.split_to_chars(s)
        else:
            n_grams = s

        matched = dict()
        for field_value in field_values:
            if not isinstance(field_value, str):
                continue
            fv_tokens = self.split_to_chars(field_value)
            sm = difflib.SequenceMatcher(None, n_grams, fv_tokens)
            match = sm.find_longest_match(0, len(n_grams), 0, len(fv_tokens))
            if match.size > 0:
                source_match = self.get_effective_match_source(
                    n_grams, match.a, match.a + match.size
                )
                if source_match:
                    match_str = field_value[match.b : match.b + match.size]
                    source_match_str = s[
                        source_match.start : source_match.start + source_match.size
                    ]
                    c_match_str = match_str.lower().strip()
                    c_source_match_str = source_match_str.lower().strip()
                    c_field_value = field_value.lower().strip()
                    if c_match_str and not self.is_common_db_term(c_match_str):
                        if (
                            self.is_stopword(c_match_str)
                            or self.is_stopword(c_source_match_str)
                            or self.is_stopword(c_field_value)
                        ):
                            continue
                        if c_source_match_str.endswith(c_match_str + "'s"):
                            match_score = 1.0
                        else:
                            if self.prefix_match(c_field_value, c_source_match_str):
                                match_score = (
                                    fuzz.ratio(c_field_value, c_source_match_str) / 100
                                )
                            else:
                                match_score = 0
                        if (
                            self.is_commonword(c_match_str)
                            or self.is_commonword(c_source_match_str)
                            or self.is_commonword(c_field_value)
                        ) and match_score < 1:
                            continue
                        s_match_score = match_score
                        if (
                            match_score >= self.m_theta
                            and s_match_score >= self.s_theta
                        ):
                            if (
                                field_value.isupper()
                                and match_score * s_match_score < 1
                            ):
                                continue
                            matched[match_str] = (
                                field_value,
                                source_match_str,
                                match_score,
                                s_match_score,
                                match.size,
                            )

        if not matched:
            return None
        else:
            return sorted(
                matched.items(),
                key=lambda x: (1e16 * x[1][2] + 1e8 * x[1][3] + x[1][4]),
                reverse=True,
            )

    def create_index(
        self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH
    ):
        pass

    def query_index(
        self,
        keywords: str,
        index_path=INDEXES_CACHE_PATH,
        top_k=1,
        filter_instance: FilterABC = None,
        database: Database | DatabaseSqlite = None,
    ):
        nl_query = keywords[0]
        matched_values = []
        schema = database.get_tables_and_columns()
        tables = [table for table in schema["tables"] if table != "sqlite_sequence"]

        for table in tables:
            # Get columns for current table
            columns_info = database.execute(f'PRAGMA table_info("{table}");', limit=-1)
            columns = columns_info.to_dict("records")

            for column in columns:
                # Check cache first
                col_name = column["name"]
                cache_key = (table, col_name)

                # Build picklist if not cached
                if cache_key not in self.picklist_indexes:
                    # Get value counts from database
                    print(f"Creating picklist for {table}.{col_name}")
                    try:
                        query = f"SELECT DISTINCT `{col_name}` FROM `{table}` WHERE `{col_name}` IS NOT NULL;"

                        values_df = database.execute(query, limit=-1)
                        values = values_df[col_name].dropna().tolist()

                        self.picklist_indexes[cache_key] = values
                    except Exception as e:
                        print(f"Error processing {table}.{col_name}: {e}")
                        continue

                # Get matches for current picklist
                picklist = self.picklist_indexes.get(cache_key, [])
                if not picklist:
                    continue
                matches = self.get_matched_entries(nl_query, picklist)
                if matches:
                    added = 0
                    for match in matches:
                        if added >= top_k:
                            break

                        value = match[0]  # Adjust based on match structure
                        matched_values.append(f"{table}.{col_name}.{value}")
                        added += 1

        return list(set(matched_values))
