import os
from datasketch import MinHash, MinHashLSH
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
from typing import List, Tuple
from utils.sqlite_db import DatabaseSqlite
import difflib
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
import pandas as pd
from value_index.value_index_abc import (
    ValueIndexABC,
    FormattedValuesMixin,
)
from filtering.filtering_abc import FilterABC
from pathlib import Path
import json
import sqlite3
from typing import Any, Union, List, Dict
import threading
import random

INDEXES_CACHE_PATH = str(Path.home()) + "/.cache/darelabdb/db_value_indexes/"
from darelabdb.utils_database_connector.core import Database
def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all", timeout: int = 60) -> Any:
    class QueryThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None
            self.exception = None

        def run(self):
            try:
                with sqlite3.connect(db_path, timeout=60) as conn:
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    if fetch == "all":
                        self.result = cursor.fetchall()
                    elif fetch == "one":
                        self.result = cursor.fetchone()
                    elif fetch == "random":
                        samples = cursor.fetchmany(10)
                        self.result = random.choice(samples) if samples else []
                    elif isinstance(fetch, int):
                        self.result = cursor.fetchmany(fetch)
                    else:
                        raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
            except Exception as e:
                self.exception = e
    query_thread = QueryThread()
    query_thread.start()
    query_thread.join(timeout)
    if query_thread.is_alive():
        raise TimeoutError(f"SQL query execution exceeded the timeout of {timeout} seconds.")
    if query_thread.exception:
        # logging.error(f"Error in execute_sql: {query_thread.exception}\nSQL: {sql}")
        raise query_thread.exception
    return query_thread.result


class CHESSIndex(ValueIndexABC,FormattedValuesMixin):
    def __init__(self, minhash_threshold=0.01,minhash_signature_size=128,ngrams = 4,model_used="BAAI/bge-large-en-v1.5",edit_distance_threshold=0.3,embedding_similarity_threshold=0.6):
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
        
    def _get_unique_values(db_path: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieves unique text values from the database excluding primary keys.

        Args:
            db_path (str): The path to the SQLite database file.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary containing unique values for each table and column.
        """
        table_names = [table[0] for table in execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';", fetch="all")]
        primary_keys = []

        for table_name in table_names:
            columns = execute_sql(db_path, f"PRAGMA table_info('{table_name}')", fetch="all")
            for column in columns:
                if column[5] > 0:  # Check if it's a primary key
                    column_name = column[1]
                    if column_name.lower() not in [c.lower() for c in primary_keys]:
                        primary_keys.append(column_name)
        
        unique_values: Dict[str, Dict[str, List[str]]] = {}
        for table_name in table_names:
            if table_name == "sqlite_sequence":
                continue
            columns = [col[1] for col in execute_sql(db_path, f"PRAGMA table_info('{table_name}')", fetch="all") if ("TEXT" in col[2] and col[1].lower() not in [c.lower() for c in primary_keys])]
            table_values: Dict[str, List[str]] = {}
            
            for column in columns:
                if any(keyword in column.lower() for keyword in ["_id", " id", "url", "email", "web", "time", "phone", "date", "address"]) or column.endswith("Id"):
                    continue

                try:
                    result = execute_sql(db_path, f"""
                        SELECT SUM(LENGTH(unique_values)), COUNT(unique_values)
                        FROM (
                            SELECT DISTINCT `{column}` AS unique_values
                            FROM `{table_name}`
                            WHERE `{column}` IS NOT NULL
                        ) AS subquery
                    """, fetch="one", timeout = 480)
                except:
                    result = 0, 0

                sum_of_lengths, count_distinct = result
                if sum_of_lengths is None or count_distinct == 0:
                    continue

                average_length = sum_of_lengths / count_distinct
                
                if ("name" in column.lower() and sum_of_lengths < 5000000) or (sum_of_lengths < 2000000 and average_length < 25) or count_distinct < 100:
                    try:
                        values = [str(value[0]) for value in execute_sql(db_path, f"SELECT DISTINCT `{column}` FROM `{table_name}` WHERE `{column}` IS NOT NULL", fetch="all", timeout = 480)]
                    except:
                        values = []
                    table_values[column] = values
            
            unique_values[table_name] = table_values

        return unique_values

    
    def create_minhash(self,signature_size: int, string: str, n_gram: int) -> MinHash:
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
        for d in [string[i:i + n_gram] for i in range(len(string) - n_gram + 1)]:
            m.update(d.encode('utf8'))
        return m

    def make_lsh(self,unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float, verbose: bool = True) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
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
        total_unique_values = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        
        progress_bar = tqdm(total=total_unique_values, desc="Creating LSH") if verbose else None
        
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                if column_name.lower() == "doctype":
                    print("="*20)
                    print("Doctype found")
                    print("="*20)
                
                for id, value in enumerate(column_values):
                    minhashvalue = self.create_minhash(signature_size, value, n_gram)
                    minhash_key = f"{table_name}.{column_name}.{id}"
                    minhashes[minhash_key] = (minhashvalue, table_name, column_name, value)
                    lsh.insert(minhash_key, minhashvalue)
                    
                    if verbose:
                        progress_bar.update(1)
        
        if verbose:
            progress_bar.close()
    
        return lsh, minhashes
    def create_index(self, database: Database | DatabaseSqlite, output_path=INDEXES_CACHE_PATH):


        chess_folder = os.path.join(output_path, "CHESS")
        os.makedirs(chess_folder, exist_ok=True)
        print(f"Creating CHESS index in {chess_folder}")
        unique_values = self.get_unique_values(database)
        print(f"Lenfgth of unique values: {len(unique_values)}")
        with open(os.path.join(chess_folder, "unique_values.pkl"), "wb") as file:
            pickle.dump(unique_values, file)
        lsh, minhashes = self.make_lsh(unique_values, self.minhash_signature_size, self.ngrams, self.minhash_threshold)
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
                found_string = string[start:i + 1]
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
            second_part = string[left_equal + 1:].strip() if len(string) > left_equal + 1 else None
            return first_part, second_part
        return None, None
    def _get_similar_column_names(self, keywords: List[str], question: str) -> List[Tuple[str, str]]:
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
        column_strings = [f"`{col.split('.')[0]}`.`{col.split('.')[1]}`" for col in schema["columns"]]
        question_hint_string = f"{question}"

        to_embed_strings = column_strings + [question_hint_string]

        # Tokenize and process strings to embeddings using the custom model
        inputs = self.tokenizer(
            to_embed_strings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**inputs)

        # Pooling the embeddings (mean pooling of the last hidden states)
        token_embeddings = model_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
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
            table, column = column_strings[i].split('.')[0].strip('`'), column_strings[i].split('.')[1].strip('`')
            for potential_column_name in potential_column_names:
                if self._does_keyword_match_column(potential_column_name, column):
                    similarity_score = np.dot(column_embedding, question_hint_embedding)
                    similar_column_names.append((table, column, similarity_score))

        similar_column_names.sort(key=lambda x: x[2], reverse=True)
        table_column_pairs = list(set([(table, column) for table, column, _ in similar_column_names]))
        return table_column_pairs
    
    def _does_keyword_match_column(self, keyword: str, column_name: str, threshold: float = 0.9) -> bool:
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
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_similar_columns(self, keywords: List[str], question: str) -> Dict[str, List[str]]:
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
        similar_columns = self._get_similar_column_names(keywords=keywords, question=question)
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
                        second_part = keyword[i+1:]
                        to_search_values.append(get_substring_packet(keyword, first_part))
                        to_search_values.append(get_substring_packet(keyword, second_part))
            hint_column, hint_value = self._column_value(keyword)
            if hint_value:
                to_search_values.append(get_substring_packet(keyword, hint_value))
        to_search_values.sort(key=lambda x: (x["keyword"], len(x["substring"]), x["substring"]), reverse=True)
        return to_search_values

    def _jaccard_similarity(self,m1: MinHash, m2: MinHash) -> float:
        """
        Computes the Jaccard similarity between two MinHash objects.

        Args:
            m1 (MinHash): The first MinHash object.
            m2 (MinHash): The second MinHash object.

        Returns:
            float: The Jaccard similarity between the two MinHash objects.
        """
        return m1.jaccard(m2)

    def query_lsh(self,lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str, 
                signature_size: int = 100, n_gram: int = 3, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
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
        similarities = [(result, self._jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]
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

    def _get_similar_entities_via_LSH(self, substring_packets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        similar_entities_via_LSH = []
        for packet in substring_packets:
            keyword = packet["keyword"]
            substring = packet["substring"]
            unique_similar_values = self.query_lsh(lsh=self.lsh, minhashes= self.minhashes, keyword=substring, signature_size=self.minhash_signature_size,n_gram=self.ngrams, top_n=10)
            for table_name, column_values in unique_similar_values.items():
                for column_name, values in column_values.items():
                    for value in values:
                        similar_entities_via_LSH.append({"keyword": keyword, 
                                                "substring": substring,
                                                "table_name": table_name,
                                                "column_name": column_name,
                                                "similar_value": value})
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
        similar_entities_via_edit_distance = self._get_similar_entities_via_edit_distance(similar_entities_via_LSH)
        similar_entities_via_embedding = self._get_similar_entities_via_embedding(similar_entities_via_edit_distance)
        
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
                max_edit_distance_similarity = max(entity["edit_distance_similarity"] for entity in values)
                values = [entity for entity in values if entity["edit_distance_similarity"] >= 0.9*max_edit_distance_similarity]
                max_embedding_similarity = max(entity["embedding_similarity"] for entity in values)
                selected_values[table_name][column_name] = [entity['similar_value'] for entity in values if entity["embedding_similarity"] >= 0.9*max_embedding_similarity]
                #add to_returh all the strings in format 'table_name.column_name.value'
                to_return.extend([f"{table_name}.{column_name}.{value}" for value in selected_values[table_name][column_name]])
        return to_return
    def _get_similar_entities_via_edit_distance(self, similar_entities_via_LSH: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        similar_entities_via_edit_distance_similarity = []
        for entity_packet in similar_entities_via_LSH:
            edit_distance_similarity = difflib.SequenceMatcher(None, entity_packet["substring"].lower(), entity_packet["similar_value"].lower()).ratio()
            if edit_distance_similarity >= self.edit_distance_threshold:
                entity_packet["edit_distance_similarity"] = edit_distance_similarity
                similar_entities_via_edit_distance_similarity.append(entity_packet)
        return similar_entities_via_edit_distance_similarity
    
    def _get_similar_entities_via_embedding(self, similar_entities_via_edit_distance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                to_embed_strings.append(similar_value)  # Add similar value for embedding

        # Validate to_embed_strings
        if not to_embed_strings:
            return []

        # Calculate embeddings using the custom model
        inputs = self.tokenizer(
            to_embed_strings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**inputs)

        # Pooling the embeddings (mean pooling of the last hidden states)
        token_embeddings = model_output.last_hidden_state
        attention_mask = inputs["attention_mask"]
        all_embeddings = self._mean_pooling(token_embeddings, attention_mask)

        # Normalize embeddings for similarity calculation
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1).cpu().numpy()

        # Match embeddings to entities and calculate similarity scores
        similar_entities_via_embedding_similarity = []
        index = 0
        for keyword, substring_dict in similar_values_dict.items():
            for substring, entity_packets in substring_dict.items():
                substring_embedding = all_embeddings[index]
                index += 1
                similar_values_embeddings = all_embeddings[index:index + len(entity_packets)]
                index += len(entity_packets)

                # Calculate similarity scores
                similarities = np.dot(similar_values_embeddings, substring_embedding)
                for i, entity_packet in enumerate(entity_packets):
                    if similarities[i] >= self.embedding_similarity_threshold:
                        entity_packet["embedding_similarity"] = similarities[i]
                        similar_entities_via_embedding_similarity.append(entity_packet)

        return similar_entities_via_embedding_similarity


    
    def query_index(self,keywords : str, index_path=INDEXES_CACHE_PATH, top_k = 5 , filter_instance: FilterABC = None , database: Database | DatabaseSqlite = None):
        if database is None:
            raise ValueError("The `database` parameter must be provided and cannot be None.")
        self.database = database

        question = keywords[0]
        keywords = keywords[1:]
        self.similar_columns = self.get_similar_columns(keywords=keywords, question=question)
        
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
