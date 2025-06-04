import json
import time
import copy
import random
import string
import re
import sqlite3
import os
from tqdm import tqdm
from collections import defaultdict
import itertools
LLM_MODEL_NAME_DEFAULT = "google/gemma-3-27b-it" 
LLM_CACHE_DIR_DEFAULT = "/data/hdd1/vllm_models/"
LLM_TENSOR_PARALLEL_SIZE_DEFAULT = 2
LLM_QUANTIZATION_DEFAULT = "fp8" 
LLM_GPU_MEM_UTIL_DEFAULT = 0.80
MAX_MODEL_LEN_DEFAULT = 8192
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
NO_SYNONYM_TOKEN = "[NO_SYNONYM]"
from spellchecker import SpellChecker

class DataExplorer:
    
    @staticmethod
    def _is_eligible_english_value(value: str, spell_checker) -> bool: # Changed parameter name
        if not isinstance(value, str) or not value.strip():
            return False
        words = value.split()
        if not words: 
            return False
        
        # pyspellchecker's known() method takes a list of words
        # and returns a set of words from that list that are known.
        # We'll check each word individually for clarity, though batching might be slightly more performant.
        for word in words:
            if not word.isalpha(): # Still check if it's purely alphabetical first
                return False
            # spell_checker.known([word]) returns a set. If the word is known, the set will contain the word.
            if not spell_checker.known([word.lower()]): # Check lowercase version for consistency
                return False
        return True

    @staticmethod
    def _build_synonym_prompt_messages(value_to_find_synonym_for: str, database_name: str, table_name: str, column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert linguist and database specialist. Your task is to identify a "hard synonym" for a given database value.

        A "hard synonym" is a word or phrase that means exactly the same thing as the original value in most contexts. Crucially, if the original value in a natural language question (NLQ) was replaced by this synonym, the meaning of the NLQ would not change, and an SQL query filtering on this value would remain semantically correct (assuming the synonym itself is not in the database and the SQL could be adapted to use it).
        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the synonym is appropriate for the context of the database.
        
        Requirements:
        - Provide synonyms that are common and direct
        - Avoid overly niche, archaic, or context-dependent synonyms
        - If no suitable hard synonym exists, return exactly: [NO_SYNONYM]
        - If the value is a proper noun (specific name, city, brand) without a common direct synonym, return: [NO_SYNONYM]
        - If any synonym would alter the precise meaning required for database querying, return: [NO_SYNONYM]
        - Do not return the original value as a synonym
        - Only provide one synonym if a suitable one is found
        - If the value is too technical or not a layman's term, return: [NO_SYNONYM]
        - Please minimize the false positives: if one synonym may mean something different given a different context, it is better to return [NO_SYNONYM] than to risk a false positive.
        - Do not include abbreviations, acronyms or shortened forms as synonyms.
        - If a value is abbreviation, acronym or shortened form, return [NO_SYNONYM].
        
        CRITICAL: Your response must contain ONLY the synonym word/phrase OR the token [NO_SYNONYM]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: card_games, Table: cards, Column: subtypes
        Value: Equipment
        United Gear

        Database: concert_singer, Table: singer, Column: country
        Value: Netherlands
        Holland

        Database: student_club, Table: zip_code, Column: type
        Value: Unique
        Distinct

        Database: codebase_community, Table: users, Column: displayname
        Value: crash
        accident

        Database: location_db, Table: states, Column: state_name
        Value: California
        [NO_SYNONYM]

        Database: users_db, Table: profiles, Column: last_name
        Value: Smith
        [NO_SYNONYM]

        Database: student_transcripts_tracking, Table: departments, Column: department_name
        Value: history
        [NO_SYNONYM]
        
        Database: california_schools, Table: schools, Column: city
        Value: Challenge
        [NO_SYNONYM]

        Remember: Output ONLY the synonym or [NO_SYNONYM] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value_to_find_synonym_for,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
      
    @staticmethod
    def find_hard_synonyms_with_vllm(
            input_json_path: str, 
            sqlite_folders_path: str, 
            output_json_path: str,
            model_name: str = LLM_MODEL_NAME_DEFAULT,
            cache_dir: str = LLM_CACHE_DIR_DEFAULT,
            tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
            quantization: str = LLM_QUANTIZATION_DEFAULT,
            gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT,
            max_model_len: int = MAX_MODEL_LEN_DEFAULT
        ):
        start_time = time.time()

        sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0,       
            max_tokens=2048 
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len 
        )
        spell = SpellChecker() 
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        columns_to_process = set()
        for record in input_data:
            db_id = record['db_id'] 
            for value_entry in record.get('values', []):
                table_name = value_entry['table']
                column_name = value_entry['column']
                columns_to_process.add((db_id, table_name, column_name))
        
        prompts_for_vllm_generation = []
        batch_processing_info = [] 
        unique_values_for_llm_processing = set()

        for db_id, table_name, column_name in columns_to_process:
            db_path = os.path.join(sqlite_folders_path, db_id, f"{db_id}.sqlite")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if not table_name or not column_name: 
                conn.close()
                continue      
            
            cursor.execute(f'SELECT DISTINCT "{column_name}" FROM "{table_name}"')
            
            all_distinct_values_from_db = [row[0] for row in cursor.fetchall()]
            conn.close()

            eligible_english_values = [
                val for val in all_distinct_values_from_db 
                if BenchmarkVariator._is_eligible_english_value(str(val), spell) # Pass spell object
            ]

            sampled_values_for_processing = []
            if len(eligible_english_values) > 10000:
                sampled_values_for_processing = random.sample(eligible_english_values, 10000)
            else:
                sampled_values_for_processing = eligible_english_values
            
            for original_value_from_db in sampled_values_for_processing:
                processing_key = (db_id, table_name, column_name, original_value_from_db)
                if processing_key not in unique_values_for_llm_processing:
                    messages = BenchmarkVariator._build_synonym_prompt_messages(original_value_from_db, db_id, table_name, column_name)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True 
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append({
                        'db_id': db_id, 
                        'table': table_name, 
                        'column': column_name, 
                        'original_value': original_value_from_db
                    })
                    unique_values_for_llm_processing.add(processing_key)
            conn.close()

        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        output_records = []
        for i, raw_response in enumerate(llm_raw_responses):
            info = batch_processing_info[i]
            parsed_synonym = BenchmarkVariator.parse_qwen3_output(raw_response)

            original_db_value = info['original_value']
            # The LLM was prompted with str(original_db_value).
            # Compare against this string form for the case-insensitivity check.
            original_value_as_str_for_llm = str(original_db_value)

            if parsed_synonym and \
               parsed_synonym != NO_SYNONYM_TOKEN and \
               parsed_synonym.lower() != original_value_as_str_for_llm.lower():
                
                output_records.append({
                    "database_id": info['db_id'],
                    "table": info['table'],
                    "column": info['column'],
                    "original_value": original_db_value, # Store the actual original value (could be None)
                    "synonym": parsed_synonym.strip() 
                })
                
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_records, f, indent=2)
        
        del llm 
        
        end_time = time.time()
        print(f"Synonym generation completed in {end_time - start_time:.2f} seconds.")
        print(f"Output file: {output_json_path}")
        print(f"Total synonym records generated: {len(output_records)}")
        
        if prompts_for_vllm_generation and llm_raw_responses:
             print(f"Example LLM input value: {batch_processing_info[0]['original_value']}")
             print(f"Example LLM raw output: {llm_raw_responses[0]}")
             if output_records:
                 print(f"Example processed record: {output_records[0]}")
             elif parsed_synonym: # if no output records, but we have a parsed synonym for the first item
                 print(f"Example LLM parsed output (not included in final): {parsed_synonym}")
                 
                 
                 
