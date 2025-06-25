import json
import time
import random
import sqlite3
import os
from tqdm import tqdm
from collections import defaultdict
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from spellchecker import SpellChecker
import string 

LLM_MODEL_NAME_DEFAULT = "gaunernst/gemma-3-27b-it-int4-awq"
LLM_CACHE_DIR_DEFAULT = "/data/hdd1/vllm_models/"
LLM_TENSOR_PARALLEL_SIZE_DEFAULT = 2
LLM_QUANTIZATION_DEFAULT = None
LLM_GPU_MEM_UTIL_DEFAULT = 0.80
MAX_MODEL_LEN_DEFAULT = 2048
NOT_VALID_TOKEN = "[NOT_VALID]"
class DataExplorer:
    
    @staticmethod
    def pass_check(original_value: str, altered_value: str) -> bool:
        # check if not valid token is in the altered value
        if altered_value is None or not isinstance(altered_value, str):
            return False
        if NOT_VALID_TOKEN in altered_value:
            return False
        return True
    @staticmethod
    def _is_eligible_english_value(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        return True
    
    @staticmethod
    def _is_eligible_english_value_with_punctuation(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char in string.digits for char in value):
            return False
        if not any(char in string.punctuation for char in value):
            return False
        return True
    
    @staticmethod
    def _is_eligible_english_value_over(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if len(value) < 6:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_without_space(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if ' ' in value:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_over_without_space(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if len(value) < 6:
            return False
        if ' ' in value:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_over_with_space(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if len(value) < 6:
            return False
        if value.count(' ') == 0:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_over_with_space_more_than_one(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if len(value) < 6:
            return False
        if ' ' not in value:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_over_with_exactly_one_space(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if len(value) < 6:
            return False
        if value.count(' ') != 1:
            return False
        return True

    @staticmethod
    
    def _is_eligible_english_value_maximum_one_space(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if value.count(' ') > 1:
            return False
        return True

    @staticmethod
    def _is_eligible_english_value_maximum_three_spaces(value: str, spell_checker) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if any(char.isdigit() for char in value):
            return False
        if value.count(' ') > 1:
            return False
        return True
    
        return True

    @staticmethod
    def _build_synonym_prompt_messages(value_to_find_synonym_for: str,
                                       database_name: str,
                                       table_name: str,
                                       column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert linguist and database specialist. Your task is to identify a \"hard synonym\" for a given database value.

        A \"hard synonym\" is a word or phrase that means exactly the same thing as the original value in most contexts. Crucially, if the original value in a natural language question (NLQ) was replaced by this synonym, the meaning of the NLQ would not change, and an SQL query filtering on this value would remain semantically correct (assuming the synonym itself is not in the database and the SQL could be adapted to use it).
        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the synonym is appropriate for the context of the database.
        
        Requirements:
        - Provide synonyms that are common and direct
        - Avoid overly niche, archaic, or context-dependent synonyms
        - If no suitable hard synonym exists, return exactly: [NOT_VALID]
        - If the value is a proper noun (specific name, city, brand) without a common direct synonym, return: [NOT_VALID]
        - If any synonym would alter the precise meaning required for database querying, return: [NOT_VALID]
        - Do not return the original value as a synonym
        - Only provide one synonym if a suitable one is found
        - If the value is too technical or not a layman's term, return: [NOT_VALID]
        - Please minimize the false positives: if one synonym may mean something different given a different context, it is better to return [NOT_VALID] than to risk a false positive.
        - Do not include abbreviations, acronyms or shortened forms as synonyms.
        - If a value is abbreviation, acronym or shortened form, return [NOT_VALID].
        
        CRITICAL: Your response must contain ONLY the synonym word/phrase OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

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
        [NOT_VALID]

        Database: users_db, Table: profiles, Column: last_name
        Value: Smith
        [NOT_VALID]

        Database: student_transcripts_tracking, Table: departments, Column: department_name
        Value: history
        [NOT_VALID]
        
        Database: california_schools, Table: schools, Column: city
        Value: Challenge
        [NOT_VALID]

        Remember: Output ONLY the synonym or [NOT_VALID] with no quotes, punctuation, or additional text.

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
    def parse_qwen3_output(raw_output_text: str) -> str:
        text_after_full_block_removal = re.sub(
            r"<think>.*?</think>",
            "",
            raw_output_text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        if re.search(r"<think>|</think>", text_after_full_block_removal, re.IGNORECASE):
            return ""
        if not text_after_full_block_removal:
            return ""
        return text_after_full_block_removal


    @staticmethod
    def _run_llm_pipeline(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str,
            filter_fn,
            build_prompt_fn,
            generated_field_name: str,
            post_filtering_fn=None,
            model_name: str = LLM_MODEL_NAME_DEFAULT,
            cache_dir: str = LLM_CACHE_DIR_DEFAULT,
            tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
            quantization: str = LLM_QUANTIZATION_DEFAULT,
            gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT,
            max_model_len: int = MAX_MODEL_LEN_DEFAULT,
                    ):
        start_time = time.time()

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            
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
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
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

        prompts_for_llm = []
        batch_info = []
        unique_values = set()

        for db_id, table_name, column_name in columns_to_process:
            db_path = os.path.join(sqlite_folders_path, db_id, f"{db_id}.sqlite")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if not table_name or not column_name:
                conn.close()
                continue

            cursor.execute(f'SELECT DISTINCT "{column_name}" FROM "{table_name}"')
            all_values = [row[0] for row in cursor.fetchall()]
            conn.close()

            eligible_values = [val for val in all_values if filter_fn(str(val), spell)]
            if len(eligible_values) > 10000:
                sampled_values = random.sample(eligible_values, 10000)
            else:
                sampled_values = eligible_values

            for original_value in sampled_values:
                key = (db_id, table_name, column_name, original_value)
                if key not in unique_values:
                    messages = build_prompt_fn(original_value, db_id, table_name, column_name)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    prompts_for_llm.append(prompt_text)
                    batch_info.append({
                        'db_id': db_id,
                        'table': table_name,
                        'column': column_name,
                        'original_value': original_value
                    })
                    unique_values.add(key)
        print(f"Sample of one prompt: {prompts_for_llm[0] if prompts_for_llm else 'No prompts generated'}")
        llm_responses = []
        if prompts_for_llm:
            outputs = llm.generate(prompts_for_llm, sampling_params)
            for out in outputs:
                llm_responses.append(out.outputs[0].text)

        output_records = []
        for i, raw_resp in enumerate(llm_responses):
            info = batch_info[i]
            parsed = DataExplorer.parse_qwen3_output(raw_resp)
            orig = info['original_value']
            if parsed and parsed != NOT_VALID_TOKEN and parsed.lower() != str(orig).lower():
                if post_filtering_fn:
                    if not post_filtering_fn(parsed, orig):
                        continue
                output_records.append({
                    "database_id": info['db_id'],
                    "table": info['table'],
                    "column": info['column'],
                    "original_value": orig,
                    generated_field_name: parsed.strip()
                })

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_records, f, indent=2)

        del llm
        end_time = time.time()
        print(f"LLM pipeline completed in {end_time - start_time:.2f} seconds.")
        print(f"Output file: {output_json_path}")
        print(f"Total records generated: {len(output_records)}")

    @staticmethod
    def find_hard_synonyms_with_vllm(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value,
            DataExplorer._build_synonym_prompt_messages,
            "synonym",
            
        )

    @staticmethod
    def _build_typo_substitution_prompt_messages(value_to_generate_typo_for: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic substitution-based typo for a given database value.

        A substitution-based typo involves replacing one character with another character that would commonly be mistyped. Focus on values that are proper nouns, entities, technical terms, or non-dictionary words that a spellchecker would not automatically correct.
        
        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        Requirements:
        - Generate only substitution typos (replace one character with another)
        - Focus on common typing mistakes: adjacent keys, similar looking letters, or frequent finger slips
        - Prioritize values that are NOT common dictionary words (entities, proper nouns, technical terms, codes, etc.)
        - The typo should be realistic - something a human would actually mistype
        - Common substitution patterns: o↔0, i↔l, e↔a, n↔m, u↔y, adjacent keyboard keys
        - If the value is a common dictionary word that spellcheck would easily catch, return: [NOT_VALID]
        - If the value is too short (1-2 characters) to create a meaningful typo, return: [NOT_VALID]
        - If no realistic substitution typo can be made, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        - Minimize false positives: the typo should be genuinely likely to occur in real typing
        - Avoid typos that would create other valid words unless they're clearly contextually wrong
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: california_schools, Table: frpm, Column: county name
        Value: Alameda
        Alaneda

        Database: concert_singer, Table: singer, Column: country
        Value: Netherlands
        [NOT_VALID]

        Database: card_games, Table: sets, Column: name
        Value: 90210
        90210

        Database: codebase_community, Table: users, Column: username
        Value: Coldsnap
        Coldsnep

        Database: users_db, Table: profiles, Column: first_name
        Value: John
        [NOT_VALID]

        Database: card_games, Table: sets, Column: code
        Value: PKHC
        PKGC
        
        Database: retail_db, Table: products, Column: category
        Value: electronics
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value_to_generate_typo_for,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def check_valid_substitution(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(original_value) != len(altered_value):
            return False
        differences = 0
        for i in range(len(original_value)):
            if original_value[i] != altered_value[i]:
                differences += 1
        return differences == 1
    
    @staticmethod
    def find_typo_substitution(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over,
            DataExplorer._build_typo_substitution_prompt_messages,
            "typo_substitution",
            post_filtering_fn=DataExplorer.check_valid_substitution,
            
        )
        
    @staticmethod
    def _build_typo_insertion_prompt_messages(value_to_generate_typo_for: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic insertion-based typo for a given database value.

        An insertion-based typo involves adding an extra character in a location where it commonly might be mistyped. Focus on values that are proper nouns, entities, technical terms, or non-dictionary words that a spellchecker would not automatically correct.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        Requirements:
        - Generate only insertion typos (add one character)
        - Focus on common typing mistakes: adjacent keys, similar looking letters, or frequent finger slips
        - Focus on values that are NOT common dictionary words (entities, proper nouns, technical terms, codes, etc.)
        - The typo should be realistic - something a human would actually mistype
        - If the value is a common dictionary word that spellcheck would easily catch, return: [NOT_VALID]
        - If no realistic insertion typo can be made, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        - Minimize false positives: the typo should be genuinely likely to occur in real typing
        - Avoid typos that would create other valid words unless they're clearly contextually wrong
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: california_schools, Table: frpm, Column: county name
        Value: Alameda
        Alameada

        Database: concert_singer, Table: singer, Column: country
        Value: Netherlands
        [NOT_VALID]

        Database: codebase_community, Table: users, Column: username
        Value: Coldsnap
        Coldsnapp

        Database: users_db, Table: profiles, Column: first_name
        Value: John
        [NOT_VALID]

        Database: card_games, Table: sets, Column: code
        Value: PKHC
        PKAHC

        Database: retail_db, Table: products, Column: category
        Value: electronics
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value_to_generate_typo_for,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def check_valid_insertion(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(altered_value) != len(original_value) + 1:
            return False
        differences = 0
        i = j = 0
        while i < len(original_value) and j < len(altered_value):
            if original_value[i] != altered_value[j]:
                differences += 1
                if differences > 1:
                    return False
                j += 1
            else:
                i += 1
                j += 1
        return differences == 1
    
    @staticmethod
    def find_typo_insertion(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over,
            DataExplorer._build_typo_insertion_prompt_messages,
            "typo_insertion",
            post_filtering_fn=DataExplorer.check_valid_insertion,

        )

    @staticmethod
    def _build_typo_deletion_prompt_messages(value: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic deletion-based typo for a given database value.

        A deletion-based typo involves removing a character from a location where it commonly might be mistyped. Focus on values that are proper nouns, entities, technical terms, or non-dictionary words that a spellchecker would not automatically correct.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        Requirements:
        - Generate only deletion typos (remove one character)
        - Focus on common typing mistakes: adjacent keys, similar looking letters, or frequent finger slips
        - Focus on values that are NOT common dictionary words (entities, proper nouns, technical terms, codes, etc.)
        - The typo should be realistic - something a human would actually mistype
        - If the value is a common dictionary word that spellcheck would easily catch, return: [NOT_VALID]
        - If no realistic deletion typo can be made, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        - Minimize false positives: the typo should be genuinely likely to occur in real typing
        - Avoid typos that would create other valid words unless they're clearly contextually wrong
        - Avoid words that are in dictionaries, as a spellchecker would catch them
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: california_schools, Table: frpm, Column: county name
        Value: Alameda
        Alamed

        Database: concert_singer, Table: singer, Column: country
        Value: Netherlands
        [NOT_VALID]

        Database: codebase_community, Table: users, Column: username
        Value: Coldsnap
        Coldsap

        Database: users_db, Table: profiles, Column: first_name
        Value: John
        [NOT_VALID]

        Database: card_games, Table: sets, Column: code
        Value: PKHC
        PKC

        Database: retail_db, Table: products, Column: category
        Value: electronics
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def check_valid_deletion(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(original_value) != len(altered_value) + 1:
            return False
        differences = 0
        i = j = 0
        while i < len(original_value) and j < len(altered_value):
            if original_value[i] != altered_value[j]:
                differences += 1
                if differences > 1:
                    return False
                i += 1
            else:
                i += 1
                j += 1
        return differences == 1
        
    @staticmethod
    def find_typo_deletion(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over,
            DataExplorer._build_typo_deletion_prompt_messages,
            "typo_deletion",
            post_filtering_fn=DataExplorer.check_valid_deletion,

        )
        
    @staticmethod
    def _build_typo_transposition_prompt_messages(value: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic transposition-based typo for a given database value.

        A transposition-based typo involves swapping two adjacent characters in a string. Focus on values that are proper nouns, entities, technical terms, or non-dictionary words that a spellchecker would not automatically correct.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        Requirements:
        - Generate only transposition typos (swap two adjacent characters)
        - Focus on common typing mistakes: adjacent keys, similar looking letters, or frequent finger slips
        - Focus on values that are NOT common dictionary words (entities, proper nouns, technical terms, codes, etc.)
        - The typo should be realistic - something a human would actually mistype
        - If the value is a common dictionary word that spellcheck would easily catch, return: [NOT_VALID]
        - If no realistic transposition typo can be made, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        - Minimize false positives: the typo should be genuinely likely to occur in real typing
        - Avoid typos that would create other valid words unless they're clearly contextually wrong
        - Avoid words that are in dictionaries, as a spellchecker would catch them
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: california_schools, Table: frpm, Column: county name
        Value: Alameda
        Almaeda

        Database: concert_singer, Table: singer, Column: country
        Value: Netherlands
        [NOT_VALID]

        Database: codebase_community, Table: users, Column: username
        Value: Coldsnap
        Colsdanp

        Database: users_db, Table: profiles, Column: first_name
        Value: John
        [NOT_VALID]

        Database: card_games, Table: sets, Column: code
        Value: PKHC
        PCKH

        Database: retail_db, Table: products, Column: category
        Value: electronics
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def check_valid_transposition(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(original_value) != len(altered_value):
            return False
        differences = []
        for i in range(len(original_value)):
            if original_value[i] != altered_value[i]:
                differences.append(i)
        if len(differences) != 2:
            return False
        i, j = differences
        return (j == i + 1 and 
                original_value[i] == altered_value[j] and 
                original_value[j] == altered_value[i])
        
    @staticmethod
    def find_typo_transposition(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over,
            DataExplorer._build_typo_transposition_prompt_messages,
            "typo_transposition",
            post_filtering_fn=DataExplorer.check_valid_transposition,

        )
    @staticmethod
    def _build_typo_space_addition_prompt_messages(value: str,
                                database_name: str,
                                table_name: str,
                                column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic space addition typo for a given database value.

        A space addition typo involves adding a single space to split one word into two meaningful parts. This should create a realistic scenario where someone might accidentally hit the spacebar while typing a compound word or technical term.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        The value may already have a space, in that case you should add a space in a different location to create a new typo if possible or return [NOT_VALID] if no new typo can be created.
        
        Requirements:
        - Generate only space addition typos (add one space to create two words)
        - The split should create two meaningful word parts, in case this is a name or entity, since they can be whatever, the splits should be easier to syntesize
        - Focus on compound words, technical terms, or words with clear morphological boundaries
        - The typo should be realistic - something a human would actually mistype by accidentally hitting spacebar
        - If the value cannot be realistically split with a space, return: [NOT_VALID]
        - If the value is too short or simple to warrant a space addition, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        
        
        """
        
        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: economics_db, Table: terms, Column: concept
        Value: macroeconomics
        macro economics

        Database: tech_companies, Table: companies, Column: name
        Value: Prachatice
        Prachat ice

        Database: users_db, Table: profiles, Column: first_name
        Value: Chrales
        Chr ales
        
        Database: products_db, Table: owners, Column: surname
        Value: abdelaziz
        abdel aziz

        Database: simple_db, Table: data, Column: word
        Value: cat
        [NOT_VALID]
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.
        
        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
   
    @staticmethod
    def check_valid_space_addition(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(altered_value) != len(original_value) + 1:
            return False
        if altered_value.count(' ') != original_value.count(' ') + 1:
            return False
        return altered_value.replace(' ', '') == original_value.replace(' ', '')
    
    @staticmethod
    def find_typo_space_addition(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_without_space,
            DataExplorer._build_typo_space_addition_prompt_messages,
            "typo_space_addition",
            post_filtering_fn=DataExplorer.pass_check,

        )

    @staticmethod
    def _build_typo_space_removal_prompt_messages(value: str,
                                database_name: str,
                                table_name: str,
                                column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing errors and database value analysis. Your task is to generate a realistic space removal typo for a given database value.

        A space removal typo involves removing a single space between two words to merge them into one continuous string. This should create a realistic scenario where someone might forget to hit the spacebar or accidentally skip it while typing separate words.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the typo is realistic for the context of the database.
        
        If there are multiple spaces, remove only one space that would create the most realistic typo.
        
        Requirements:
        - Generate only space removal typos (remove one space to merge two words)
        - The merged result should look like a plausible single word or compound term
        - Focus on removing spaces between words that could reasonably be typed as one word
        - Avoid removing spaces from common word pairs that would be obviously wrong (like "the cat" -> "thecat")
        - The typo should be realistic - something a human would actually create by missing the spacebar
        - If the value has no spaces, return: [NOT_VALID]
        - If removing any space would create an obviously invalid result, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        - Focus on values like entities, names or technical terms
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: economics_db, Table: terms, Column: concept
        Value: macro economics
        macroeconomics

        Database: tech_companies, Table: companies, Column: name
        Value: Prachat ice
        Prachatice

        Database: simple_db, Table: data, Column: phrase
        Value: the cat
        [NOT_VALID]

        Database: countries, Table: greece, Column: city
        Value: south anw marko poulos
        south anw markopoulos
        
        Database: tech_db, Table: software, Column: name
        Value: data base
        database
        
        Database: products_db, Table: items, Column: category
        Value: home decor
        [NOT_VALID]
        
        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.
        IMPORTANT: Remove ONLY ONE space, not more.
        
        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]

    @staticmethod
    def check_valid_space_removal(original_value: str, altered_value: str) -> bool:
        if original_value == altered_value:
            return False
        if len(altered_value) != len(original_value) - 1:
            return False
        if altered_value.count(' ') != original_value.count(' ') - 1:
            return False
        return altered_value.replace(' ', '') == original_value.replace(' ', '')
    
    @staticmethod
    def find_typo_space_removal(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over_with_exactly_one_space,
            DataExplorer._build_typo_space_removal_prompt_messages,
            "typo_space_removal",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    
    @staticmethod
    def _build_word_to_symbol_prompt_messages(value: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common typing shortcuts and database value analysis. Your task is to generate a realistic word-to-symbol change for a given database value.

        A word-to-symbol change involves replacing one word in the value with its commonly used symbolic equivalent. This creates a realistic scenario where someone might use a shortcut symbol instead of typing out the full word.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the replacement is realistic for the context of the database.
        
        Requirements:
        - Replace exactly one word with its symbolic equivalent
        - Only use commonly recognized word-to-symbol mappings
        - The replacement should be realistic - something a human would actually type as a shortcut
        - Maintain the original spacing and capitalization of surrounding words
        - If no word in the value has a commonly used symbolic equivalent, return: [NOT_VALID]
        - If the value is too simple or has no replaceable words, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        
        Common word-to-symbol mappings include:
        - and → &
        - at → @
        - percent/percentage → %
        - dollar/dollars → $
        - number → #
        - plus → +
        - minus → -
        - equals → =
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: business_db, Table: companies, Column: name
        Value: Johnson and Associates
        Johnson & Associates

        Database: contact_db, Table: emails, Column: address
        Value: support at example dot com
        support @ example dot com

        Database: finance_db, Table: rates, Column: description
        Value: interest rate percent
        interest rate %

        Database: products_db, Table: items, Column: price_info
        Value: five dollar item
        five $ item
        
        Database: documents_db, Table: files, Column: reference
        Value: document number 123
        document # 123

        Database: simple_db, Table: data, Column: word
        Value: hello
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    
    @staticmethod
    def find_word_to_symbol_change(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over_with_space_more_than_one,
            DataExplorer._build_word_to_symbol_prompt_messages,
            "word_to_symbol_change",
            post_filtering_fn=DataExplorer.pass_check,

        )
        
    @staticmethod
    def _build_synonym_prompt_messages(value: str,
                            database_name: str,
                            table_name: str,
                            column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in language variations and database value analysis. Your task is to generate a realistic synonym for a given database value.

        A synonym involves replacing one word in the value with a commonly used synonym. This creates a realistic scenario where someone might use a different word with similar meaning, either due to personal preference, regional variations, or simple word choice differences.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the synonym replacement is realistic for the context of the database.
        
        Requirements:
        - Replace exactly one word with a commonly recognized synonym
        - The synonym should be natural and contextually appropriate
        - Focus on synonyms that are genuinely interchangeable in the given context
        - The replacement should be realistic - something a human would naturally use
        - If no word in the value has a suitable synonym, return: [NOT_VALID]
        - If the value is too simple or technical to warrant synonym replacement, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one synonym variant

        CRITICAL: Your response must contain ONLY the synonym variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text.
        IMPORTANT: We should avoid false positives, the synonym should be really of high relevance to the original value and used in common language"""
        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: business_db, Table: companies, Column: description
        Value: large
        big 

        Database: retail_db, Table: customers, Column: feedback
        Value: excellent
        great 

        Database: finance_db, Table: transactions, Column: description
        Value: customer
        client 
        

        Database: simple_db, Table: data, Column: code
        Value: XYZ123
        [NOT_VALID]

        Database: movies, Table: authors, Column: name
        Value: John
        [NOT_VALID]
        
        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def find_synonym_change(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_without_space,
            DataExplorer._build_synonym_prompt_messages,
            "synonym",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    
    @staticmethod
    def _build_abbreviation_acronym_prompt_messages(value: str,
                                        database_name: str,
                                        table_name: str,
                                        column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in common abbreviations, acronyms, and database value analysis. Your task is to generate a realistic abbreviation or acronym for a given database value.

        An abbreviation/acronym involves replacing a word or phrase in the value with its commonly recognized abbreviated form or acronym. This creates a realistic scenario where someone might use a shorter form instead of typing out the full name or phrase.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the abbreviation/acronym replacement is realistic for the context of the database.
        
        Requirements:
        - Replace one value with its widely recognized abbreviation or acronym
        - Only use abbreviations/acronyms that are commonly known and used
        - The replacement should be realistic - something a human would naturally use as a shortcut
        - Abbreviations should be well-established, not made-up shortenings
        - If no word/phrase in the value has a recognized abbreviation/acronym, return: [NOT_VALID]
        - If the value is too simple or already abbreviated, return: [NOT_VALID]
        - Only provide one variant
        - You should minimize false positives: the abbreviation/acronym should be genuinely likely to occur in real typing
    
        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: locations_db, Table: cities, Column: name
        Value: Los Angeles California
        LA California

        Database: organizations_db, Table: agencies, Column: name
        Value: National Aeronautics and Space Administration
        NASA

        Database: contacts_db, Table: people, Column: title
        Value: Doctor Smith
        Dr Smith
        
        Database: simple_db, Table: data, Column: word
        Value: hello
        [NOT_VALID]

        Database: tech_db, Table: software, Column: type
        Value: database application
        DB application
        
        Database: zoo, Table: animals, Column: species
        Value: Elephant
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def find_abbreviation_acronym_change(
            input_json_path: str,
            sqlite_folders_path: str,
            output_json_path: str
        ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value,
            DataExplorer._build_abbreviation_acronym_prompt_messages,
            "abbreviation_acronym",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    @staticmethod
    def _build_clipping_prompt_messages(value: str,
                                        database_name: str,
                                        table_name: str,
                                        column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in linguistics, common word shortenings (clippings), and database value analysis. Your task is to generate a realistic shortened form for a given database value.

        A shortened form, or "clipping," involves shortening a longer word by dropping the end part, while keeping the beginning. This is a common linguistic process used for informal or efficient communication (e.g., 'administrator' -> 'admin', 'information' -> 'info'). This is DIFFERENT from an acronym (NASA) or an initialism (LA).

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure that the shortened form is realistic for the context of the database.
        
        Requirements:
        - Replace one word with its commonly recognized shortened form (clipping).
        - Only use shortened forms that are commonly known and used (e.g., 'admin', 'prof', 'doc', 'info').
        - The replacement should be realistic - something a human would naturally use as a shortcut.
        - Shortened forms should be well-established, not just arbitrarily truncated words (e.g., 'administrator' -> 'admin' is good, but 'administrator' -> 'administ' is bad).
        - If no word in the value has a common clipped form, return: [NOT_VALID]
        - If the value is too simple or already a shortened form, return: [NOT_VALID]
        - Only provide one variant.
        - You should minimize false positives: the shortened form should be genuinely likely to occur in real typing.

        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: user_accounts, Table: users, Column: role
        Value: system administrator
        system admin

        Database: university_db, Table: faculty, Column: title
        Value: Professor Plum
        Prof Plum

        Database: project_management, Table: tasks, Column: details
        Value: See document for more information
        See document for more info

        Database: files_db, Table: records, Column: type
        Value: Scanned Document
        Scanned Doc

        Database: simple_db, Table: data, Column: word
        Value: hello
        [NOT_VALID]

        Database: zoo, Table: animals, Column: species
        Value: Giraffe
        [NOT_VALID]
        
        Database: locations_db, Table: cities, Column: name
        Value: Los Angeles California
        [NOT_VALID]

        Database: user_accounts, Table: users, Column: role
        Value: admin
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def find_clipping_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_maximum_one_space,
            DataExplorer._build_clipping_prompt_messages,
            "clipping",
            post_filtering_fn=DataExplorer.pass_check,

        )

    @staticmethod
    def _build_paraphrasing_prompt_messages(value: str,
                                            database_name: str,
                                            table_name: str,
                                            column_name: str) -> list:
        """
        Builds a prompt to generate a paraphrased version of a database value.
        """
        SYSTEM_PROMPT = """You are an expert in natural language understanding, semantics, and database value analysis. Your task is to paraphrase a given database value.

        Paraphrasing means rephrasing the value to express the same meaning using different words or sentence structure. This is DIFFERENT from simply replacing a word with a synonym or using an abbreviation. The goal is to create a variant that a different human might have typed to convey the exact same information.

        The values are actual cell values from a database, so along with the values, the database name, table and column will be given. Make sure the paraphrase is realistic for the context of the database.
        
        Requirements:
        - Rephrase the value, potentially by changing word order, adding/removing function words (like 'a', 'the', 'for'), or using different phrasing.
        - The core meaning must be strictly preserved.
        - The paraphrase should be a natural and common way of expressing the same information.
        - Avoid simple, single-word synonym swaps (e.g., 'Car' -> 'Automobile' is not a good paraphrase). The change should be more structural.
        - If the value is too simple (e.g., a single word, a proper name) or cannot be naturally paraphrased, return: [NOT_VALID]
        - Only provide one variant.
        - You should minimize false positives: the paraphrase should be genuinely plausible as a human-entered alternative.

        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: full_name
        Value: Smith, John
        John Smith

        Database: reports_db, Table: quarterly_reports, Column: title
        Value: Report for the Third Quarter
        Third Quarter Report

        Database: inventory_db, Table: products, Column: status
        Value: Item is currently out of stock
        Currently out of stock

        Database: ecommerce_db, Table: orders, Column: payment_terms
        Value: Payment required upon delivery
        Payment due on delivery

        Database: tasks_db, Table: todos, Column: status
        Value: Complete
        [NOT_VALID]

        Database: locations_db, Table: cities, Column: name
        Value: New York
        [NOT_VALID]

        Database: planning_db, Table: milestones, Column: deadline
        Value: To be determined
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def find_paraphrases_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_maximum_three_spaces,
            DataExplorer._build_paraphrasing_prompt_messages,
            "paraphrasing",
            post_filtering_fn=DataExplorer.pass_check,

        )

    @staticmethod
    def _build_negated_antonym_prompt_messages(value: str,
                                            database_name: str,
                                            table_name: str,
                                            column_name: str) -> list:
        """
        Builds a prompt to generate a negated antonym for a database value.
        """
        SYSTEM_PROMPT = """You are an expert in semantics, logic, and linguistic transformations. Your task is to generate a specific type of paraphrase for a given database value: a negated antonym.

        This transformation involves replacing a value with the negation of its opposite (its antonym). The transformation follows the pattern: `[value]` -> `not [antonym of value]`. This creates a phrase that is semantically equivalent to the original value. For example, the value 'active' would be transformed into 'not inactive'.

        This is DIFFERENT from a simple antonym ('active' -> 'inactive') or a simple negation ('active' -> 'not active'). The meaning must be preserved.

        The values are actual cell values from a database, so the database name, table, and column will be given. The transformation should only be applied if it results in a natural-sounding phrase. This is most common for adjectives and status words.
        
        Requirements:
        - Replace the value with 'not' followed by its direct, common antonym.
        - The resulting phrase must be semantically equivalent to the original value.
        - The antonym used must be a common and natural opposite (e.g., the antonym of 'possible' is 'impossible').
        - If the value has no clear, common antonym, or if the transformation would be awkward, return: [NOT_VALID]
        - If the value is a complex phrase, a proper name, or is already in a negative form, return: [NOT_VALID]
        - Only provide one variant.

        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: user_accounts, Table: users, Column: status
        Value: active
        not inactive

        Database: forms_db, Table: fields, Column: validation
        Value: required
        not optional

        Database: compliance_db, Table: actions, Column: status
        Value: legal
        not illegal

        Database: security_db, Table: permissions, Column: access_level
        Value: allowed
        not forbidden

        Database: contacts_db, Table: people, Column: name
        Value: John Smith
        [NOT_VALID]

        Database: inventory_db, Table: products, Column: color
        Value: blue
        [NOT_VALID]

        Database: reports_db, Table: quarterly_reports, Column: title
        Value: Report for the Third Quarter
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    
    @staticmethod
    def find_negated_antonyms_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over_without_space,
            DataExplorer._build_negated_antonym_prompt_messages,
            "negated_antonym",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    @staticmethod
    def _build_word_removal_prompt_messages(value: str,
                                            database_name: str,
                                            table_name: str,
                                            column_name: str) -> list:
        """
        Builds a prompt to generate a variant of a value by removing one non-essential word.
        """
        SYSTEM_PROMPT = """You are an expert in semantics, linguistics, and data conciseness. Your task is to identify and remove a single, non-essential word from a given database value.

        The goal is to create a more concise version of the value that preserves the original's core meaning and identity. This often involves removing words that are optional or redundant, such as middle initials, titles, or certain qualifiers.

        The values are actual cell values from a database. The database name, table, and column are provided for context. The resulting value must be a natural and common way of representing the same information.
        
        What to look for and remove:
        - Middle initials (e.g., 'John F. Kennedy' -> 'John Kennedy')
        - Titles or honorifics (e.g., 'Dr. Jane Smith' -> 'Jane Smith')
        - Redundant words (e.g., 'final conclusion' -> 'conclusion')
        - Non-essential adverbs or adjectives (e.g., 'currently unavailable' -> 'unavailable')

        What NOT to remove:
        - Words that are part of a proper name (e.g., 'New York', 'Los Angeles').
        - Words that specify a key attribute (e.g., 'Senior' in 'Senior Developer').
        - Any word whose removal would significantly change the meaning.

        Requirements:
        - You must remove exactly one word.
        - The core meaning must be strictly preserved.
        - If no single word can be removed without changing the meaning, return: [NOT_VALID]
        - Only provide one variant.

        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: full_name
        Value: John D Smith
        John Smith

        Database: user_accounts, Table: users, Column: full_name
        Value: Mister Adam Jones
        Adam Jones

        Database: inventory_db, Table: products, Column: status
        Value: Currently unavailable
        Unavailable

        Database: reports_db, Table: documents, Column: type
        Value: Final Conclusion
        Conclusion

        Database: roles_db, Table: job_titles, Column: title
        Value: Senior Developer
        [NOT_VALID]

        Database: locations_db, Table: cities, Column: name
        Value: New York
        [NOT_VALID]

        Database: reports_db, Table: quarterly_reports, Column: title
        Value: Quarterly Report
        [NOT_VALID]

        Database: contacts_db, Table: people, Column: full_name
        Value: Jane Doe
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
        
    @staticmethod
    def find_word_removal_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over_with_space,
            DataExplorer._build_word_removal_prompt_messages,
            "word_removal",
            post_filtering_fn=DataExplorer.pass_check,

        )
        
    @staticmethod
    def _build_word_addition_prompt_messages(value: str,
                                            database_name: str,
                                            table_name: str,
                                            column_name: str) -> list:
        """
        Builds a prompt to generate a variant of a value by adding one contextually appropriate word.
        """
        SYSTEM_PROMPT = """You are an expert in natural language generation, semantics, and common phrasing. Your task is to add a single, contextually appropriate word to a given database value.

        The goal is to create a slightly more descriptive or formal version of the value that a human might naturally use, without changing its core meaning. This often involves adding common but technically optional words like titles, qualifiers, or articles.

        The values are actual cell values from a database. The database name, table, and column are provided for context. The resulting value must be a natural and common way of representing the same information.
        
        What to look for and add:
        - Titles or honorifics (e.g., 'John Smith' -> 'Mr. John Smith').
        - Common descriptive nouns or adjectives (e.g., 'Report' -> 'Status Report').
        - Articles or determiners where appropriate (e.g., 'User Guide' -> 'The User Guide').

        What NOT to add:
        - Words that significantly change the meaning (e.g., adding 'Senior' to 'Developer' changes the role's seniority).
        - Words that make the phrase sound unnatural or awkward.
        - Any word if the value is already specific, formal, or verbose.

        Requirements:
        - You must add exactly one word.
        - The core meaning must be strictly preserved.
        - If no single word can be naturally added, return: [NOT_VALID]
        - Only provide one variant.

        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: full_name
        Value: Jane Doe
        Ms. Jane Doe

        Database: tasks_db, Table: todos, Column: status
        Value: Complete
        Task Complete

        Database: documentation_db, Table: guides, Column: title
        Value: User Manual
        The User Manual

        Database: roles_db, Table: job_titles, Column: title
        Value: Senior Developer
        [NOT_VALID]

        Database: locations_db, Table: cities, Column: name
        Value: Los Angeles
        [NOT_VALID]

        Database: contacts_db, Table: people, Column: full_name
        Value: Mr. John Smith
        [NOT_VALID]

        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def find_word_addition_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_maximum_three_spaces,
            DataExplorer._build_word_addition_prompt_messages,
            "word_addition",
            post_filtering_fn=DataExplorer.pass_check,

        )
        
    @staticmethod
    def _build_typo_punctuation_removal_prompt_messages(value: str,
                                        database_name: str,
                                        table_name: str,
                                        column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in punctuation usage and database value analysis. Your task is to generate a realistic punctuation removal for a given database value.

        A punctuation removal typo involves removing one piece of punctuation from the value where the meaning remains clear and unchanged. This creates a realistic scenario where someone might omit punctuation for brevity or due to casual typing habits.

        The values are actual cell values from a database containing only text (no digits) and are maximum 3 words long. Along with the values, the database name, table and column will be given. Make sure that the punctuation removal is realistic for the context of the database.
        
        Requirements:
        - Remove exactly one punctuation mark that doesn't change the semantic meaning
        - The removal should be realistic - something a human might skip while typing quickly
        - Focus on optional punctuation that is commonly omitted in informal contexts
        - Maintain all other formatting, spacing, and capitalization
        - Only remove punctuation where the meaning remains completely clear
        - Avoid removing punctuation that would create ambiguity or grammatical errors
        - If no punctuation can be safely removed without changing meaning, return: [NOT_VALID]
        - If the value has no punctuation, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one variant
        
        CRITICAL: Your response must contain ONLY the variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: title
        Value: Dr. Peter
        Dr Peter

        Database: business_db, Table: companies, Column: suffix
        Value: Smith Inc.
        Smith Inc

        Database: names_db, Table: people, Column: full_name
        Value: Smith, Jr
        Smith Jr

        Database: adjectives_db, Table: descriptions, Column: quality
        Value: well-known brand
        well known brand
        
        Database: contractions_db, Table: phrases, Column: text
        Value: it's working
        its working

        Database: simple_db, Table: data, Column: word
        Value: hello
        [NOT_VALID]

        Database: expressions_db, Table: phrases, Column: saying
        Value: don't worry
        dont worry
        
        Database: titles_db, Table: people, Column: name
        Value: Prof. Johnson
        Prof Johnson
        
        Database: compound_db, Table: words, Column: term
        Value: twenty-one
        twenty one
        
        Database: text_db, Table: messages, Column: content
        Value: we're ready
        were ready
        
        Remember: Output ONLY the variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def find_punct_removal_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_with_punctuation,
            DataExplorer._build_typo_punctuation_removal_prompt_messages,
            "punct_removal",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    
    @staticmethod
    def _build_typo_punctuation_change_prompt_messages(value: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in punctuation variations and database value analysis. Your task is to generate a realistic punctuation change typo for a given database value.

        A punctuation change typo involves replacing one piece of punctuation with a different punctuation mark. This creates a realistic scenario where someone might use an alternative punctuation mark due to typing habits, keyboard layout differences, or stylistic preferences.

        The values are actual cell values from a database containing only text (no digits) and are maximum 3 words long. Along with the values, the database name, table and column will be given. Make sure that the punctuation change is realistic for the context of the database.
        
        Requirements:
        - Replace exactly one punctuation mark with a different punctuation mark
        - The change should be realistic - something a human might naturally substitute
        - Focus on common punctuation substitutions that maintain readability
        - Maintain all other formatting, spacing, and capitalization
        - The meaning should remain reasonably clear with the new punctuation
        - Only make changes that are contextually appropriate
        - If no punctuation can be realistically changed, return: [NOT_VALID]
        - If the value has no punctuation, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one typo variant
        
        CRITICAL: Your response must contain ONLY the typo variant OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: title
        Value: Dr. Smith
        Dr, Smith

        Database: expressions_db, Table: phrases, Column: text
        Value: don't go
        don"t go

        Database: compound_db, Table: words, Column: term
        Value: well-known
        well_known

        Database: names_db, Table: people, Column: suffix
        Value: Smith, Jr
        Smith. Jr
        
        Database: simple_db, Table: data, Column: word
        Value: hello
        [NOT_VALID]

        Database: contractions_db, Table: phrases, Column: text
        Value: it's working
        it"s working
        
        Database: movies, Table: actors, Column: name
        Value: John Doe
        [NOT_VALID]

        Remember: Output ONLY the typo variant or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def find_punct_change_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_with_punctuation,
            DataExplorer._build_typo_punctuation_change_prompt_messages,
            "punct_change",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    @staticmethod
    def _build_typo_word_order_variation_prompt_messages(value: str,
                                        database_name: str,
                                        table_name: str,
                                        column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in data entry variations and database value analysis. Your task is to generate a realistic word order variation for a given database value.

        A word order variation involves swapping the order of two words while maintaining the same semantic meaning. This creates a realistic scenario where someone might reverse the word order due to different data entry conventions, form field expectations, cultural naming practices, or alternative formatting standards.

        The values are actual cell values from a database containing exactly two words. Along with the values, the database name, table and column will be given. Make sure that the word order change is realistic for the context of the database.
        
        Requirements:
        - Swap the order of the two words (first word becomes second, second becomes first)
        - The reversal should be realistic and contextually appropriate
        - Maintain all original capitalization, punctuation, and formatting of each word
        - The semantic meaning should remain essentially the same
        - Focus on cases where word order reversal is natural (names, titles, locations)
        - Only swap if the reversal makes contextual sense
        - If word order reversal would be unnatural or incorrect, return: [NOT_VALID]
        - If the value doesn't have exactly two words, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one variation
        
        Common word order variations include:
        - Names: John Smith → Smith John, Mary Johnson → Johnson Mary
        - Titles: Doctor Smith → Smith Doctor, Professor Lee → Lee Professor
        - Descriptive pairs: Blue Sky → Sky Blue, Big House → House Big
        
        CRITICAL: Your response must contain ONLY the variation OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: contacts_db, Table: people, Column: full_name
        Value: John Smith
        Smith John

        Database: academic_db, Table: faculty, Column: title_name
        Value: Professor Lee
        Lee Professor

        Database: colors_db, Table: descriptions, Column: color_object
        Value: Blue Sky
        Sky Blue

        Database: articles_db, Table: words, Column: phrase
        Value: the cat
        [NOT_VALID]

        Database: medical_db, Table: staff, Column: title_name
        Value: Doctor Wilson
        Wilson Doctor
        
        Database: properties_db, Table: descriptions, Column: size_type
        Value: Los Angeles
        [NOT_VALID]
        
        Remember: Output ONLY the variation or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def find_word_order_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_over_with_exactly_one_space,
            DataExplorer._build_typo_word_order_variation_prompt_messages,
            "word_order_change",
            post_filtering_fn=DataExplorer.pass_check,

        )
    
    @staticmethod
    def _build_singular_plural_variation_prompt_messages(value: str,
                                    database_name: str,
                                    table_name: str,
                                    column_name: str) -> list:
        SYSTEM_PROMPT = """You are an expert in English grammar variations and database value analysis. Your task is to generate a realistic singular/plural variation for a given database value.

        A singular/plural variation involves changing one word in the value between its singular and plural form while maintaining essentially the same semantic meaning in a database context. This creates a realistic scenario where someone might use either form due to different perspectives on categorization, data entry habits, or grammatical preferences.

        The values are actual cell values from a database containing up to two words. Along with the values, the database name, table and column will be given. Make sure that the singular/plural change is realistic for the context of the database.
        
        Requirements:
        - Change exactly one word from singular to plural OR plural to singular
        - The change should maintain the same essential meaning in a database context
        - Focus on nouns that can naturally exist in both forms in database contexts
        - Maintain all original capitalization, punctuation, and formatting
        - The semantic meaning should remain essentially equivalent for database purposes
        - Only make changes where both forms would be contextually valid
        - If no word can be realistically changed between singular/plural, return: [NOT_VALID]
        - If the change would significantly alter the meaning, return: [NOT_VALID]
        - Do not return the original value
        - Only provide one variation
        
        Common singular/plural variations include:
        - Categories: kid → kids, product → products, item → items
        - Objects: book → books, car → cars, house → houses
        - Groups: team → teams, company → companies, user → users
        - Descriptive terms: red car → red cars, big house → big houses
        - Technical terms: file → files, record → records, database → databases
        
        CRITICAL: Your response must contain ONLY the variation OR the token [NOT_VALID]. Do not include quotes, explanations, or any other text."""

        USER_PROMPT = """Here are examples of the expected input and output format:

        Database: products_db, Table: categories, Column: type
        Value: kid
        kids

        Database: inventory_db, Table: items, Column: category
        Value: books
        book

        Database: retail_db, Table: products, Column: description
        Value: red car
        red cars

        Database: grammar_db, Table: words, Column: example
        Value: running
        [NOT_VALID]

        Database: movies, Table: authors, Column: name
        Value: John Doe
        [NOT_VALID]
        
        Remember: Output ONLY the variation or [NOT_VALID] with no quotes, punctuation, or additional text.

        Database: {database_name}, Table: {table_name}, Column: {column_name}
        Value: {VALUE_PLACEHOLDER}"""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(
                VALUE_PLACEHOLDER=value,
                database_name=database_name,
                table_name=table_name,
                column_name=column_name
            )}
        ]
    
    @staticmethod
    def find_singular_plural_change(
        input_json_path: str,
        sqlite_folders_path: str,
        output_json_path: str
    ):
        DataExplorer._run_llm_pipeline(
            input_json_path,
            sqlite_folders_path,
            output_json_path,
            DataExplorer._is_eligible_english_value_maximum_one_space,
            DataExplorer._build_singular_plural_variation_prompt_messages,
            "singular_plural_change",
            post_filtering_fn=DataExplorer.pass_check,

        )
        
    
        
if __name__ == "__main__":
    output_dir = "assets/data_exploration"
    sqlite_folders_path = "CHESS/data/value_linking/dev_databases"
    input_json_path = "assets/value_linking_valid_values_exact_no_bird_train.json"

    output_filename = "word_order_change.json"
    output_json_path = os.path.join(output_dir, output_filename)
    DataExplorer.find_word_order_change(
        input_json_path=input_json_path,
        sqlite_folders_path=sqlite_folders_path,
        output_json_path=output_json_path
    )