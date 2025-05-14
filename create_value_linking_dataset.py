import json
import random
import string
import re
import nltk
from typing import List, Dict, Tuple, Optional
import sqlglot
import sqlglot.expressions as exp
import shutil
from pathlib import Path
import copy
from sentence_transformers import SentenceTransformer, util
import torch
import os
from tqdm import tqdm
nltk.download("wordnet", quiet=True)


class ValueLinkingDatasetProcessor:
    """Processes dataset for value linking tasks including formatting, typos, synonyms, and predictions."""
    def __init__(self, schema_data: List[Dict]):
        # Initialize schema mapping from schema data
        self.schema_mapping = self._build_schema_mapping(schema_data)
        
    def format_value_strings(self, input_path, output_path):
        """Formats values into 'table.column.value' strings and saves them.

        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to save formatted JSON
        """
        with open(input_path, "r") as file:
            data = json.load(file)

        results = []
        for record in data:
            value_strings = [
                f"{v['table']}.{v['column']}.{v['value']}".lower()
                for v in record["values"]
            ]
            results.append(value_strings)

        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=4)

    

    def generate_predictions_with_precision(
        self, pred_path, gt_path, precision, output_path
    ):
        """Generates predictions calibrated to target precision.

        Args:
            pred_path (str): Path to predicted JSON file
            gt_path (str): Path to ground truth JSON file
            precision (float): Target precision between 0-1
            output_path (str): Path to save calibrated predictions
        """
        if not 0 <= precision <= 1:
            raise ValueError("Precision must be between 0 and 1")

        with open(pred_path) as pred_file, open(gt_path) as gt_file:
            pred_data = json.load(pred_file)
            gt_data = json.load(gt_file)

        if len(pred_data) != len(gt_data):
            raise ValueError("Input files must have same number of records")

        results = []
        for preds, truths in zip(pred_data, gt_data):
            combined = set(truths)
            if precision < 1:
                required = int(len(truths) / precision)
                available = set(preds) - combined
                combined.update(
                    random.sample(
                        list(available), min(required - len(combined), len(available))
                    )
                )
            results.append(list(combined))

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

    @staticmethod
    def _build_schema_mapping(schema_data: List[Dict]) -> Dict[str, Dict]:
        # Build a mapping of database IDs to their schema details
        schema_mapping = {}
        for schema in schema_data:
            schema_mapping[schema["db_id"]] = {
                "schema_items": [
                    {
                        "table_name": schema["table_names"][col[0]],
                        "column_names": [
                            schema["column_names"][idx][1].lower()
                            for idx in range(len(schema["column_names"]))
                            if schema["column_names"][idx][0] == col[0]
                        ],
                    }
                    for col in schema["column_names"]
                    if col[0] != -1
                ]
            }
        return schema_mapping

    def extract_tables_columns_and_values(
        self, sql_query: str, db_id: str, dialect: str = "sqlite"
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        # Retrieve schema for the given database ID
        schema = self.schema_mapping.get(db_id, None)
        if not schema:
            return [], [], []

        def get_subquery_tables_columns_and_values(expression, cte_aliases):
            # Extract table names from the query, excluding CTE aliases
            tables = [
                t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.name.lower() not in cte_aliases
            ]

            # Map table aliases to their original table names
            table_aliases = {
                t.alias.lower(): t.name.lower()
                for t in expression.find_all(exp.Table)
                if t.alias != ""
            }

            columns = []
            values = []
            # Extract columns from the query
            for c in expression.find_all(exp.Column):
                column_name = c.name.lower()
                table_name_or_alias = c.table.lower()

                if table_name_or_alias == "":
                    # Disambiguate columns when table name is not provided
                    if len(tables) == 1:
                        table_name = tables[0]
                    else:
                        table_name = ""
                        for table in schema["schema_items"]:
                            if (
                                column_name in table["column_names"]
                                and table["table_name"] in tables
                            ):
                                table_name = table["table_name"]
                                break
                        if table_name == "":
                            continue
                elif table_name_or_alias in table_aliases:
                    table_name = table_aliases[table_name_or_alias]
                elif table_name_or_alias in tables:
                    table_name = table_name_or_alias
                else:
                    continue

                columns.append(f"{table_name}.{column_name}")

            # Extract values from conditions in the query
            for condition in expression.find_all(exp.Condition):
                if isinstance(
                    condition,
                    (exp.EQ, exp.NEQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Like, exp.In),
                ):
                    operator_map = {
                        "eq": "=",
                        "neq": "!=",
                        "gt": ">",
                        "lt": "<",
                        "gte": ">=",
                        "lte": "<=",
                        "like": "LIKE",
                        "in": "IN",
                    }
                    operator = operator_map.get(
                        condition.__class__.__name__.lower(),
                        condition.__class__.__name__.lower(),
                    )

                    if isinstance(condition, exp.In):
                        left = condition.this
                        right = condition.expressions

                        if isinstance(left, exp.Column):
                            column_name = left.name.lower()
                            table_name = left.table.lower()

                            if table_name == "" and len(tables) == 1:
                                table_name = tables[0]
                            elif table_name in table_aliases:
                                table_name = table_aliases[table_name]

                            for literal in right:
                                if isinstance(literal, exp.Literal):
                                    values.append({
                                        "table": table_name,
                                        "column": column_name,
                                        "value": str(literal).strip("'\""),
                                    })
                    else:
                        left = condition.left
                        right = condition.right

                        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                            column_name = left.name.lower()
                            table_name = left.table.lower()

                            if table_name == "" and len(tables) == 1:
                                table_name = tables[0]
                            elif table_name in table_aliases:
                                table_name = table_aliases[table_name]

                            values.append({
                                "table": table_name,
                                "column": column_name,
                                "value": str(right).strip("'\""),
                                "condition": operator,
                            })

            return tables, columns, values

        # Parse the SQL query
        try:
            expression = sqlglot.parse_one(sql_query, read=dialect)
        except:
            return [], [], []
        # Collect CTE aliases to distinguish them from actual tables
        cte_aliases = [cte.alias for cte in expression.find_all(exp.CTE)]

        # Collect sub-queries and process them in reverse order
        sub_queries = list(expression.find_all((exp.Subquery, exp.CTE), bfs=False))
        sub_queries.reverse()
        sub_queries.append(expression)

        tables = []
        columns = []
        values = []

        for sub_query in sub_queries:
            sub_tables, sub_columns, sub_values = get_subquery_tables_columns_and_values(
                sub_query, cte_aliases
            )
            #sub_query.pop()
            tables.extend(sub_tables)
            columns.extend(sub_columns)
            values.extend(sub_values)

        return list(set(tables)), list(set(columns)), values
    @staticmethod
    def _has_english_char(s):
        """Checks if a string contains at least one English alphabet character."""
        # Fast check assuming ASCII or compatible encoding
        s_str = str(s) # Ensure it's a string first
        for char in s_str:
            # Check if character is in the ranges 'a'-'z' or 'A'-'Z'
            if ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                return True
        return False
    @staticmethod
    def filter_json_file(input_json_path, output_json_path):
        filtered_records = []
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
            
        for record in data:
            values_list = record["values"] # Direct access - unsafe
            if values_list: 
                contains_valid_value = False
                for value_item in values_list:
                    if ValueLinkingDatasetProcessor._has_english_char(value_item["value"]): # Direct access - unsafe
                        contains_valid_value = True
                        break 
                if contains_valid_value:
                    filtered_records.append(record)
                    
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_records, outfile, indent=None) # No indentation for speed/compactness

    @staticmethod
    def filter_json_by_question_values(input_json_path, output_json_path):
        """
        Filters records from an input JSON file based on whether all 'value' fields
        in the 'values' list exist case-insensitively within the 'question' field.
        Writes the output with standard JSON indentation.

        Args:
            input_json_path (str): Path to the input JSON file.
            output_json_path (str): Path to the output JSON file.
        """
        filtered_records = []
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        for record in data:
            # Make case-insensitive comparison robust, handle potential non-string questions
            try:
                question_lower = record["question"].lower()
            except AttributeError: 
                # If question is not a string (e.g., null or number), it can't contain values
                continue # Skip this record

            values_list = record.get("values", []) # Use .get for safety, default to empty list

            all_values_present = True # Assume true until proven otherwise

            if not values_list:
                # Empty list means all (zero) values are present. Keep it.
                pass
            else:
                for value_entry in values_list:
                    # Handle potential non-existence or non-string values safely
                    value_to_check_raw = value_entry.get("value")
                    if value_to_check_raw is None:
                        # If a value entry is missing the 'value' key, treat as not present? 
                        # Or skip the check? Let's assume it means it doesn't match.
                        all_values_present = False
                        break

                    value_to_check = str(value_to_check_raw).lower()
                    if value_to_check not in question_lower:
                        all_values_present = False
                        break # No need to check further values for this record

            if all_values_present:
                filtered_records.append(record)

        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            # Use indent=4 for standard pretty-printing
            json.dump(filtered_records, outfile, indent=4)
        print("Final number of records:", len(filtered_records))


    @staticmethod
    def delete_char(s):
        """Deletes a random character from the string s."""
        if not s:
            return s, -1, '' # No change if empty
        idx = random.randrange(len(s))
        original_char = s[idx]
        new_s = s[:idx] + s[idx+1:]
        return new_s, idx, original_char
    @staticmethod
    def insert_char(s):
        """Inserts a random lowercase letter into the string s."""
        idx = random.randrange(len(s) + 1)
        # Insert a random lowercase ascii letter
        char_to_insert = random.choice(string.ascii_lowercase) 
        new_s = s[:idx] + char_to_insert + s[idx:]
        return new_s, idx, char_to_insert
    @staticmethod
    def substitute_char(s):
        """Substitutes a random character in s with a random lowercase letter."""
        if not s:
            return s, -1, '', '' # No change if empty
        idx = random.randrange(len(s))
        original_char = s[idx]
        # Ensure the new char is different from the original
        possible_chars = list(string.ascii_lowercase)
        if original_char.lower() in possible_chars:
            possible_chars.remove(original_char.lower())
            # Handle edge case if original_char was the only possible char (unlikely)
            if not possible_chars: possible_chars = list(string.ascii_lowercase)
                
        new_char = random.choice(possible_chars)
        # Preserve case if original was uppercase
        if s[idx].isupper():
            new_char = new_char.upper()
            
        new_s = s[:idx] + new_char + s[idx+1:]
        return new_s, idx, original_char, new_char
    @staticmethod
    def transpose_chars(s):
        """Swaps two adjacent characters in the string s."""
        if len(s) < 2:
            return s, -1 # No change if too short
        idx = random.randrange(len(s) - 1) # Index of the first char in the pair to swap
        new_s = s[:idx] + s[idx+1] + s[idx] + s[idx+2:]
        return new_s, idx
    
    # --- Main Function ---
    @staticmethod
    def introduce_typos_in_question(input_json_path, output_json_path):
        """
        Reads records from input_json_path, introduces a single typo into the question
        for each value found within it (case-insensitive match, first occurrence modified),
        tracks the changes, and writes to output_json_path.
        """
        processed_records = []
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        for record in data:
            original_question = record["question"]
            modified_question = original_question # Start with the original
            values_list = record.get("values", [])
            typo_details = []
            
            # To avoid modifying the same value string multiple times if it appears
            # multiple times in the values list or the question itself.
            # We modify only the *first* occurrence found for each unique value string.
            
            # Find potential modification sites (start_index, value_string, value_len)
            # Store sites to modify later, working backward to avoid index shifts.
            potential_sites = []
            processed_value_strings_lower = set() # Track unique values from the list

            if isinstance(original_question, str): # Only process if question is a string
                question_lower = original_question.lower()
                for value_entry in values_list:
                    value_str_raw = value_entry.get("value")
                    if value_str_raw is None: continue # Skip if value is missing
                    
                    value_str = str(value_str_raw)
                    value_str_lower = value_str.lower()
                    
                    # Only find the first match for this specific value string (case-insensitive)
                    if value_str and value_str_lower not in processed_value_strings_lower:
                        start_index = question_lower.find(value_str_lower)
                        if start_index != -1:
                            potential_sites.append({
                                "start": start_index,
                                "original_value": value_str, # Keep original casing from 'value' field
                                "len": len(value_str)
                            })
                            processed_value_strings_lower.add(value_str_lower)

            # Sort sites by start index in descending order
            potential_sites.sort(key=lambda x: x["start"], reverse=True)

            # Apply modifications from the end of the string backwards
            for site in potential_sites:
                start = site["start"]
                original_value = site["original_value"]
                val_len = site["len"]
                
                # Extract the exact segment from the possibly already modified question
                current_segment = modified_question[start : start + val_len]

                # Choose a random typo type
                typo_type = random.choice(["delete", "insert", "substitute", "transpose"])
                
                modified_segment = current_segment # Default if typo fails
                typo_info = {"original_value": original_value, "original_segment": current_segment, "type": typo_type, "index_in_question": start}

                if typo_type == "delete" and len(current_segment) > 0:
                    modified_segment, idx, orig_char = ValueLinkingDatasetProcessor.delete_char(current_segment)
                    typo_info.update({"index_in_value": idx, "deleted_char": orig_char})
                elif typo_type == "insert":
                    modified_segment, idx, inserted_char = ValueLinkingDatasetProcessor.insert_char(current_segment)
                    typo_info.update({"index_in_value": idx, "inserted_char": inserted_char})
                elif typo_type == "substitute" and len(current_segment) > 0:
                    modified_segment, idx, orig_char, new_char = ValueLinkingDatasetProcessor.substitute_char(current_segment)
                    typo_info.update({"index_in_value": idx, "original_char": orig_char, "new_char": new_char})
                elif typo_type == "transpose" and len(current_segment) > 1:
                    modified_segment, idx = ValueLinkingDatasetProcessor.transpose_chars(current_segment)
                    typo_info.update({"index_in_value": idx, "swapped_pair": current_segment[idx:idx+2]})
                else:
                    # If typo couldn't be applied (e.g., delete from empty, transpose len<2)
                    # Or if the segment somehow changed length unexpectedly before processing (shouldn't happen with reverse processing)
                    typo_info["type"] = "none" # Mark that no typo was actually made for this value

                # Reconstruct the modified question string
                modified_question = modified_question[:start] + modified_segment + modified_question[start + val_len:]
                
                # Only add detail if a typo was actually made
                if typo_info["type"] != "none":
                    typo_info["modified_segment"] = modified_segment
                    typo_details.append(typo_info)


            # Add the modified question and typo details to the record
            record["modified_question"] = modified_question
            record["typo_details"] = typo_details # Add even if empty
            processed_records.append(record)

        # Write the processed records to the output file
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(processed_records, outfile, indent=4)
        print("Final number of records:", len(processed_records))



    @staticmethod
    def build_chat_messages(question, keyword):
        # Construct the message list for the chat template
        # System prompt sets the overall instruction
        # Few-shot examples provide context
        # Final user message contains the actual task
        messages = [
            {"role": "system", "content": "You are an AI assistant. Given a query and a keyword within it, replace the keyword with a suitable synonym if one exists and makes sense in context. If no suitable synonym exists or the keyword is a proper noun (like a name or place) that shouldn't be changed, return the original query exactly. Only return the resulting query string, without any introductory text."},
            {"role": "user", "content": 'Query: "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"\nKeyword: "Alameda"'},
            {"role": "assistant", "content": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"},
            {"role": "user", "content": 'Query: "Which active district has the highest average score in Reading?"\nKeyword: "Active"'},
            {"role": "assistant", "content": "Which current district has the highest average score in Reading?"},
            {"role": "user", "content": f'Query: "{question}"\nKeyword: "{keyword}"'}
        ]
        return messages
    @staticmethod
    def parse_qwen3_output(raw_output_text):
        """
        Parses the raw output from Qwen3 (potentially with thinking tags)
        to extract the final content.
        """
        think_end_tag = "</think>"
        # Find the *last* occurrence of the closing tag
        last_tag_index = raw_output_text.rfind(think_end_tag)

        if last_tag_index != -1:
            # If tag found, content starts after the tag
            content_start_index = last_tag_index + len(think_end_tag)
            final_content = raw_output_text[content_start_index:]
        else:
            # If no tag found, assume the whole output is the content
            final_content = raw_output_text

        # Strip leading/trailing whitespace and newlines
        return final_content.strip()

    @staticmethod
    def generate_synonyms_with_vllm_parsed(input_json_path, output_json_path):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer 
        model_name = "Qwen/Qwen3-32B" # More likely to work generally

        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.95,
            top_k=20,
            max_tokens=4096 # Adjust based on expected output length
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,cache_dir="assets/cache")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=2, # Adjust based on your GPU setup
            enable_prefix_caching=True,
            trust_remote_code=True,
            download_dir="assets/cache",
            quantization="fp8",
            gpu_memory_utilization=0.85
        )

        prompts_as_text = []
        mapping_info = []

        with open(input_json_path, 'r', encoding='utf-8') as infile:
            original_data = json.load(infile)

        for record_idx, record in enumerate(original_data):
            original_question = record.get("question")
            values_list = record.get("values", [])

            if not isinstance(original_question, str) or not values_list:
                continue

            question_lower = original_question.lower()
            processed_values_for_record = set()

            for value_idx, value_entry in enumerate(values_list):
                keyword_raw = value_entry.get("value")
                if keyword_raw is None: continue
                keyword = str(keyword_raw)
                keyword_lower = keyword.lower()

                if (keyword_lower not in processed_values_for_record and
                    ValueLinkingDatasetProcessor._has_english_char(keyword) and
                    keyword_lower in question_lower):

                    # Build the chat messages list
                    messages = ValueLinkingDatasetProcessor.build_chat_messages(original_question, keyword)

                    # Apply the chat template - This enables thinking by default
                    try:
                        # enable_thinking=True is the default, no need to explicitly pass usually
                        prompt_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                            # enable_thinking=True # Explicitly set if default changes or for clarity
                        )
                        prompts_as_text.append(prompt_text)
                        mapping_info.append((record_idx, value_idx, original_question, keyword))
                        processed_values_for_record.add(keyword_lower)
                    except Exception as e:
                        pass # Skip this prompt if template fails


        # Perform batch generation
        modified_records = []
        if prompts_as_text:
            outputs = llm.generate(prompts_as_text, sampling_params)

            # Process results
            for i, output in enumerate(outputs):
                record_idx, value_idx, original_question, keyword = mapping_info[i]
                raw_response_text = output.outputs[0].text

                # Manually parse the output to remove <think> blocks
                parsed_response = ValueLinkingDatasetProcessor.parse_qwen3_output(raw_response_text)
                #replace multiple spaces with a single space
                parsed_response = re.sub(r'\s+', ' ', parsed_response).strip()
                #same for original question
                original_question = re.sub(r'\s+', ' ', original_question).strip()
                # Check if modification occurred and it's not just a case change
                is_modified = parsed_response != original_question
                # Ensure case-insensitivity check uses the *parsed* response
                is_just_case_change = is_modified and (parsed_response.lower() == original_question.lower())

                if is_modified and not is_just_case_change:
                    # Create a copy of the original record
                    # Find the original record in the list
                    original_record_copy = next((copy.deepcopy(r) for idx, r in enumerate(original_data) if idx == record_idx), None)

                    if original_record_copy:
                        original_record_copy["original_question"] = original_question
                        original_record_copy["modified_question_by_synonym"] = parsed_response
                        modified_records.append(original_record_copy)


        # Write the successfully modified records
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(modified_records, outfile, indent=4)

    @staticmethod
    def prepare_data_chess(input_json_path: str, output_json_path: str):
        """
        Reads a JSON file with a specific input record format, transforms each record
        to a new format, and writes the results to a new JSON file.

        Args:
            input_json_path: Path to the input JSON file.
            output_json_path: Path where the output JSON file will be saved.
        """
        with open(input_json_path, 'r', encoding='utf-8') as infile:
            input_data = json.load(infile)

        output_data = []
        question_id_counter = 1  # Initialize the self-incrementing ID

        for record in input_data:
            transformed_record = {
                "question_id": question_id_counter,
                "db_id": record.get("db_id", None),  # Safely get db_id
                "question": record.get("question", ""),
                "evidence": record.get("evidence", ""),
                "SQL": record.get("SQL", ""),
                "difficulty": "simple"
            }
            output_data.append(transformed_record)
            question_id_counter += 1

        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(output_data, outfile, indent=4)

    @staticmethod
    def copy_databases(json_path: str, output_folder: str):
        source_map = {
            "bird_dev": Path("assets/dev_20240627/dev_databases"),
            "bird_train": Path("assets/train/train_databases"),
            "spider_dev": Path("assets/spider_data/database"),
            "spider_test": Path("assets/spider_data/test_database")
        }

        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        output_dir_path = Path(output_folder)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        processed_db_ids = set()

        for record in records:
            db_id = record["db_id"]
            source_key = record["source"]

            if db_id in processed_db_ids:
                continue

            target_db_specific_path = output_dir_path / db_id

            if target_db_specific_path.exists():
                processed_db_ids.add(db_id)
                continue
            
            source_base_dir = source_map[source_key]
            source_db_specific_path = source_base_dir / db_id
            
            shutil.copytree(source_db_specific_path, target_db_specific_path)
            processed_db_ids.add(db_id)

    @staticmethod
    def generate_prompts_for_eval_open_search(original_questions_path: str,
                                few_shot_source_path: str,
                                output_path: str,
                                embedding_model_name: str = 'BAAI/bge-m3'):
        """
        Generates a new questions.json-like file for an evaluation set.

        For each question in the original_questions_path:
        - If an identical question (based on 'raw_question') exists in few_shot_source_path,
        its 'prompt' and other relevant fields are used.
        - Otherwise, the 'prompt' from the most semantically similar question
        (based on 'raw_question' embeddings) in few_shot_source_path is used.

        Args:
            original_questions_path (str): Path to the JSON file containing evaluation questions
                                        (format: list of dicts, each with 'raw_question', 'question', etc.).
            few_shot_source_path (str): Path to the existing questions.json file
                                        (format: dict with a "questions" key, which is a list of dicts).
            output_path (str): Path to save the newly generated JSON file.
            embedding_model_name (str): Name of the sentence transformer model from Hugging Face.
        """
        #create the dir of the output path if it does not exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Load data
        with open(original_questions_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        with open(few_shot_source_path, 'r', encoding='utf-8') as f:
            few_shot_source_data = json.load(f)

        # Prepare few-shot source for lookup and similarity
        # We'll primarily use the 'raw_question' for matching and similarity
        source_questions_list = few_shot_source_data.get("questions", [])
        if not source_questions_list:
            # Handle case where "questions" key is missing or empty
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"args": {}, "costs": {}, "questions": [], "extract": []}, f, indent=4)
            return


        source_raw_questions_map = {item['raw_question']: item for item in source_questions_list}
        source_raw_question_texts = [item['raw_question'] for item in source_questions_list]

        # Load sentence transformer model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(embedding_model_name, device=device)

        # Pre-compute embeddings for all raw_questions in the few-shot source
        source_embeddings = model.encode(source_raw_question_texts, convert_to_tensor=True, show_progress_bar=False)

        output_questions_list = []
        # The "extract" part of few_shot_source_data might also need similar handling if used.
        # For simplicity, this example focuses on the "questions" part.
        # We can create a parallel "extract" list if needed.
        output_extract_list = []
        source_extract_list = few_shot_source_data.get("extract", [])
        source_extract_raw_questions_map = {item['raw_question']: item for item in source_extract_list}


        for eval_item in tqdm(eval_data):
            eval_raw_question = eval_item['raw_question']
            matched_source_item_q = None
            matched_source_item_e = None # For the 'extract' part

            # Check for exact match in the "questions" part
            if eval_raw_question in source_raw_questions_map:
                matched_source_item_q = source_raw_questions_map[eval_raw_question]
            else:
                # If no exact match, find the most similar one
                eval_embedding = model.encode(eval_raw_question, convert_to_tensor=True, show_progress_bar=False)
                cosine_scores = util.cos_sim(eval_embedding, source_embeddings)[0]
                best_match_idx = torch.argmax(cosine_scores).item()
                matched_source_item_q = source_questions_list[best_match_idx]

            # Create the new item for the output "questions" list
            new_output_item_q = {
                "question": eval_item["question"],
                "evidence": eval_item["evidence"],
                "raw_question": eval_item["raw_question"],
                "prompt": matched_source_item_q["prompt"], # Use prompt from matched/similar
                "n_examples": matched_source_item_q.get("n_examples", 0), # Preserve n_examples
                "db_id": eval_item["db_id"]
            }
            output_questions_list.append(new_output_item_q)

            # Handle the "extract" part similarly
            # Check for exact match in the "extract" part (if it exists and has raw_question)
            if source_extract_list: # only if source_extract_list is not empty
                if eval_raw_question in source_extract_raw_questions_map:
                    matched_source_item_e = source_extract_raw_questions_map[eval_raw_question]
                else:
                    # Find the most similar one in the "extract" list's raw_questions
                    # Note: This assumes 'extract' list items also have 'raw_question' and 'prompt'
                    # Re-using source_embeddings and source_raw_question_texts for similarity,
                    # assuming 'extract' list is parallel or has similar raw questions.
                    # A more robust way would be to embed 'extract' raw_questions separately if they differ.
                    if source_raw_question_texts: # Check if there are source questions to compare against
                        eval_embedding = model.encode(eval_raw_question, convert_to_tensor=True, show_progress_bar=False)
                        cosine_scores = util.cos_sim(eval_embedding, source_embeddings)[0] # Re-use source_embeddings
                        best_match_idx = torch.argmax(cosine_scores).item()
                        # Find the corresponding item in source_extract_list.
                        # This assumes that the source_extract_list has items whose raw_question matches
                        # source_questions_list[best_match_idx]['raw_question']
                        # A safer way is to build a dedicated map or list for source_extract_list raw_questions.

                        # Simplified: find by raw_question from the best match in the "questions" list
                        best_match_raw_q_for_extract = source_questions_list[best_match_idx]['raw_question']
                        if best_match_raw_q_for_extract in source_extract_raw_questions_map:
                            matched_source_item_e = source_extract_raw_questions_map[best_match_raw_q_for_extract]
                        elif source_extract_list: # Fallback to the first item if no good match
                            matched_source_item_e = source_extract_list[0]


                if matched_source_item_e:
                    new_output_item_e = {
                        "question": eval_item["question"],
                        "evidence": eval_item["evidence"],
                        "raw_question": eval_item["raw_question"],
                        "prompt": matched_source_item_e["prompt"],
                        "n_examples": matched_source_item_e.get("n_examples", 0),
                        "db_id": eval_item["db_id"]
                    }
                    output_extract_list.append(new_output_item_e)
                elif source_extract_list : # If no match for extract but source_extract_list exists, maybe add a default or skip
                    # Fallback: if no specific match for extract, but extract list exists, use the first one
                    default_extract_item = source_extract_list[0]
                    new_output_item_e = {
                        "question": eval_item["question"],
                        "evidence": eval_item["evidence"],
                        "raw_question": eval_item["raw_question"],
                        "prompt": default_extract_item["prompt"],
                        "n_examples": default_extract_item.get("n_examples", 0),
                        "db_id": eval_item["db_id"]
                    }
                    output_extract_list.append(new_output_item_e)


        # Construct the final output structure similar to the input few_shot_source_data
        final_output_data = {
            "args": few_shot_source_data.get("args", {}), # Preserve original args
            "costs": few_shot_source_data.get("costs", {}), # Preserve original costs
            "questions": output_questions_list,
            "extract": output_extract_list # Add the populated extract list
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output_data, f, indent=4)