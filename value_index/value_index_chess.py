import os
from datasketch import MinHash, MinHashLSH
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
from typing import List, Tuple
import difflib
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from pathlib import Path
import sqlite3
from typing import Any, Union, List, Dict
import threading
import random
from threading import Lock
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

def get_table_all_columns(db_path: str, table_name: str) -> List[str]:
    """
    Retrieves all column names for a given table.
    
    Args:
        db_path (str): The path to the database file.
        table_name (str): The name of the table.
        
    Returns:
        List[str]: A list of column names.
    """
    try:
        table_info_rows = execute_sql(db_path, f"PRAGMA table_info(`{table_name}`);")
        return [row[1].replace('\"', '').replace('`', '') for row in table_info_rows]
    except Exception as e:
        raise e
def _jaccard_similarity(m1: MinHash, m2: MinHash) -> float:
    """
    Computes the Jaccard similarity between two MinHash objects.

    Args:
        m1 (MinHash): The first MinHash object.
        m2 (MinHash): The second MinHash object.

    Returns:
        float: The Jaccard similarity between the two MinHash objects.
    """
    return m1.jaccard(m2)
def get_db_all_tables(db_path: str) -> List[str]:
    """
    Retrieves all table names from the database.
    
    Args:
        db_path (str): The path to the database file.
        
    Returns:
        List[str]: A list of table names.
    """
    try:
        raw_table_names = execute_sql(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0].replace('\"', '').replace('`', '') for table in raw_table_names if table[0] != "sqlite_sequence"]
    except Exception as e:
        raise e
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

def _create_minhash(signature_size: int, string: str, n_gram: int) -> MinHash:
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

def make_lsh(unique_values: Dict[str, Dict[str, List[str]]], signature_size: int, n_gram: int, threshold: float, verbose: bool = True) -> Tuple[MinHashLSH, Dict[str, Tuple[MinHash, str, str, str]]]:
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
    try:
        total_unique_values = sum(len(column_values) for table_values in unique_values.values() for column_values in table_values.values())
        
        progress_bar = tqdm(total=total_unique_values, desc="Creating LSH") if verbose else None
        
        for table_name, table_values in unique_values.items():
            for column_name, column_values in table_values.items():
                if column_name.lower() == "doctype":
                    print("="*20)
                    print("Doctype found")
                    print("="*20)
                
                for id, value in enumerate(column_values):
                    minhash = _create_minhash(signature_size, value, n_gram)
                    minhash_key = f"{table_name}_{column_name}_{id}"
                    minhashes[minhash_key] = (minhash, table_name, column_name, value)
                    lsh.insert(minhash_key, minhash)
                    
                    if verbose:
                        progress_bar.update(1)
        
        if verbose:
            progress_bar.close()
    except Exception as e:
        print(f"Error creating LSH: {e}")
    return lsh, minhashes
def make_db_lsh(db_directory_path: str, **kwargs: Any) -> None:
    """
    Creates a MinHash LSH for the database and saves the results.

    Args:
        db_directory_path (str): The path to the database directory.
        **kwargs (Any): Additional arguments for the LSH creation.
    """
    db_id = Path(db_directory_path).name
    # if db_id starts with ., skip
    if db_id.startswith("."):
        return
    preprocessed_path = Path(db_directory_path) / "preprocessed"
    #if exists, skip
    if (preprocessed_path / f"{db_id}_lsh.pkl").exists():
        return
    preprocessed_path.mkdir(exist_ok=True)
    
    unique_values = _get_unique_values(str(Path(db_directory_path) / f"{db_id}.sqlite"))
    
    with open(preprocessed_path / f"{db_id}_unique_values.pkl", "wb") as file:
        pickle.dump(unique_values, file)
    
    lsh, minhashes = make_lsh(unique_values, **kwargs)
    
    with open(preprocessed_path / f"{db_id}_lsh.pkl", "wb") as file:
        pickle.dump(lsh, file)
    with open(preprocessed_path / f"{db_id}_minhashes.pkl", "wb") as file:
        pickle.dump(minhashes, file)
class DatabaseManager:
    """
    A singleton class to manage database operations including schema generation, 
    querying LSH and vector databases, and managing column profiles.
    """
    _instance = None
    _lock = Lock()
    
    def get_db_schema(self,db_path: str) -> Dict[str, List[str]]:
        """
        Retrieves the schema of the database.
        
        Args:
            db_path (str): The path to the database file.
            
        Returns:
            Dict[str, List[str]]: A dictionary mapping table names to lists of column names.
        """
        try:
            table_names = get_db_all_tables(db_path)
            return {table_name: get_table_all_columns(db_path, table_name) for table_name in table_names}
        except Exception as e:
            raise e
    def query_lsh(self,lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keyword: str, 
              signature_size: int = 100, n_gram: int = 4, top_n: int = 10) -> Dict[str, Dict[str, List[str]]]:
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
        query_minhash = _create_minhash(signature_size, keyword, n_gram)
        results = lsh.query(query_minhash)
        similarities = [(result, _jaccard_similarity(query_minhash, minhashes[result][0])) for result in results]
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
         
class CHESSIndex():
    def __init__(self, minhash_threshold=0.01,minhash_signature_size=100,ngrams = 4,model_used="BAAI/bge-large-en-v1.5",edit_distance_threshold=0.3,embedding_similarity_threshold=0.6):
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

    def _get_similar_entities(self,lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], keywords: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieves similar entities from the database based on keywords.

        Args:
            keywords (List[str]): The list of keywords.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
        """
        to_seartch_values = self._get_to_search_values(keywords)
        similar_entities_via_LSH = self._get_similar_entities_via_LSH(lsh,minhashes,to_seartch_values)
        similar_entities_via_embedding = self._get_similar_entities_via_embedding(similar_entities_via_LSH)
        similar_entities_via_edit_distance = self._get_similar_entities_via_edit_distance(similar_entities_via_embedding)
        
        
        selected_values = {}
        for entity in similar_entities_via_edit_distance:
            table_name = entity["table_name"]
            column_name = entity["column_name"]
            if table_name not in selected_values:
                selected_values[table_name] = {}
            if column_name not in selected_values[table_name]:
                selected_values[table_name][column_name] = []
            selected_values[table_name][column_name].append(entity)
        for table_name, column_values in selected_values.items():
            for column_name, values in column_values.items():
                max_edit_distance_similarity = max(entity["edit_distance_similarity"] for entity in values)
                values = [entity for entity in values if entity["edit_distance_similarity"] >= 0.9*max_edit_distance_similarity]
                max_embedding_similarity = max(entity["embedding_similarity"] for entity in values)
                selected_values[table_name][column_name] = [entity['similar_value'] for entity in values if entity["embedding_similarity"] >= 0.9*max_embedding_similarity]
                    
        return selected_values

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
    
    def _get_similar_entities_via_LSH(self,lsh: MinHashLSH, minhashes: Dict[str, Tuple[MinHash, str, str, str]], substring_packets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        similar_entities_via_LSH = []
        for packet in substring_packets:
            keyword = packet["keyword"]
            substring = packet["substring"]
            unique_similar_values = DatabaseManager().query_lsh(lsh,minhashes,keyword=substring, signature_size=100, top_n=10)
            for table_name, column_values in unique_similar_values.items():
                for column_name, values in column_values.items():
                    for value in values:
                        similar_entities_via_LSH.append({"keyword": keyword, 
                                                "substring": substring,
                                                "table_name": table_name,
                                                "column_name": column_name,
                                                "similar_value": value})
        return similar_entities_via_LSH
    
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
        for entity_packet in similar_entities_via_edit_distance:
            keyword = entity_packet["keyword"]
            substring = entity_packet["substring"]
            similar_value = entity_packet["similar_value"]
            if keyword not in similar_values_dict:
                similar_values_dict[keyword] = {}
            if substring not in similar_values_dict[keyword]:
                similar_values_dict[keyword][substring] = []
                to_embed_strings.append(substring)
            similar_values_dict[keyword][substring].append(entity_packet)
            to_embed_strings.append(similar_value)
        
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
        
        similar_entities_via_embedding_similarity = []
        index = 0
        for keyword, substring_dict in similar_values_dict.items():
            for substring, entity_packets in substring_dict.items():
                substring_embedding = all_embeddings[index]
                index += 1
                similar_values_embeddings = all_embeddings[index:index+len(entity_packets)]
                index += len(entity_packets)
                similarities = np.dot(similar_values_embeddings, substring_embedding)
                for i, entity_packet in enumerate(entity_packets):
                    if similarities[i] >= self.embedding_similarity_threshold:
                        entity_packet["embedding_similarity"] = similarities[i]
                        similar_entities_via_embedding_similarity.append(entity_packet)
        return similar_entities_via_embedding_similarity
    
    
    def create_index(self, database_str : str):
        print(f"Creating index for database: {database_str}")
        make_db_lsh(database_str, signature_size=self.minhash_signature_size, n_gram=self.ngrams, threshold=self.minhash_threshold)
                    
    
    def query_index(self,keywords : str, index_path : str):

        db_id = os.path.basename(os.path.normpath(index_path))
        lsh_path = os.path.join(index_path, 'preprocessed', f'{db_id}_lsh.pkl')
        if not os.path.exists(lsh_path):
            print(f"MinHashLSH index not found in: {index_path}. Skipping.")
            return []
        if lsh_path not in self.min_hash_indexes:
            with open(lsh_path, "rb") as f:
                lsh = pickle.load(f)
                self.min_hash_indexes[lsh_path] = lsh
        else:
            lsh = self.min_hash_indexes[lsh_path]
        
        minhash_path = os.path.join(index_path, 'preprocessed', f'{db_id}_minhashes.pkl')
        if not os.path.exists(minhash_path):
            print(f"MinHash objects not found in: {index_path}. Skipping.")
            return []
        if minhash_path not in self.min_hash_indexes:
            with open(minhash_path, "rb") as f:
                minhashes = pickle.load(f)
                self.min_hash_indexes[minhash_path] = minhashes
        else:
            minhashes = self.min_hash_indexes[minhash_path]
        
        results = self._get_similar_entities(keywords=keywords,lsh=lsh,minhashes=minhashes)
        return results
