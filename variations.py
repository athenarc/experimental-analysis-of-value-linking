import json
import time
import copy
import random
import string
import re
LLM_MODEL_NAME_DEFAULT = "Qwen/Qwen3-32B" 
LLM_CACHE_DIR_DEFAULT = "/data/hdd1/vllm_models/"
LLM_TENSOR_PARALLEL_SIZE_DEFAULT = 2 
LLM_QUANTIZATION_DEFAULT = "fp8" 
LLM_GPU_MEM_UTIL_DEFAULT = 0.80

class BenchmarkVariator:

    @staticmethod
    def _is_eligible_text_value(value_text: str) -> bool:
        if not isinstance(value_text, str):
            return False
        return any(c.isalpha() for c in value_text)

    @staticmethod
    def substitute_char_typo(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []
        alphabet = string.ascii_lowercase

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue

                altered_value_text = None
                if len(original_value_from_meta) > 0:
                    idx = random.randrange(len(original_value_from_meta))
                    original_char_at_idx = original_value_from_meta[idx]
                    
                    new_char = original_char_at_idx
                    if len(alphabet) == 1 and alphabet[0] == original_char_at_idx:
                        pass 
                    else:
                        # Ensure the new character is different
                        while new_char == original_char_at_idx:
                            new_char = random.choice(alphabet)
                    
                    temp_altered_value_list = list(original_value_from_meta)
                    temp_altered_value_list[idx] = new_char
                    altered_value_text = "".join(temp_altered_value_list)

                if altered_value_text is None or original_value_from_meta == altered_value_text:
                    continue

                # Case-insensitive search for original_value_from_meta in current_q_being_modified
                match = re.search(re.escape(original_value_from_meta), current_q_being_modified, re.IGNORECASE)
                
                if match:
                    matched_segment_in_question = match.group(0)
                    
                    start_index = match.start()
                    end_index = match.end()
                    
                    # Construct next_q_state by replacing the actual matched segment
                    next_q_state = current_q_being_modified[:start_index] + \
                                   altered_value_text + \
                                   current_q_being_modified[end_index:]
                    
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": matched_segment_in_question, # Store the actual segment that was replaced
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "substitute_char_typo"
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")

    @staticmethod
    def insert_char_typo(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []
        alphabet = string.ascii_lowercase

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                altered_value_text = None
                # _is_eligible_text_value ensures original_value_from_meta is not empty
                idx = random.randrange(len(original_value_from_meta) + 1)
                char_to_insert = random.choice(alphabet)
                altered_value_text = original_value_from_meta[:idx] + char_to_insert + original_value_from_meta[idx:]

                if altered_value_text is None or original_value_from_meta == altered_value_text:
                    continue

                # Case-insensitive search for original_value_from_meta in current_q_being_modified
                match = re.search(re.escape(original_value_from_meta), current_q_being_modified, re.IGNORECASE)
                
                if match:
                    matched_segment_in_question = match.group(0)
                    
                    start_index = match.start()
                    end_index = match.end()
                    
                    # Construct next_q_state by replacing the actual matched segment
                    next_q_state = current_q_being_modified[:start_index] + \
                                   altered_value_text + \
                                   current_q_being_modified[end_index:]
                                   
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": matched_segment_in_question, # Store the actual segment that was replaced
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "insert_char_typo"
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")

    @staticmethod
    def delete_char_typo(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue

                altered_value_text = None
                if len(original_value_from_meta) >= 1: # Original condition was >=1, _is_eligible ensures >0
                    idx = random.randrange(len(original_value_from_meta))
                    altered_value_text = original_value_from_meta[:idx] + original_value_from_meta[idx+1:]
                
                if altered_value_text is None or original_value_from_meta == altered_value_text:
                    continue

                # Case-insensitive search for original_value_from_meta in current_q_being_modified
                match = re.search(re.escape(original_value_from_meta), current_q_being_modified, re.IGNORECASE)
                
                if match:
                    matched_segment_in_question = match.group(0)
                    
                    start_index = match.start()
                    end_index = match.end()
                    
                    # Construct next_q_state by replacing the actual matched segment
                    next_q_state = current_q_being_modified[:start_index] + \
                                   altered_value_text + \
                                   current_q_being_modified[end_index:]
                                   
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": matched_segment_in_question, # Store the actual segment that was replaced
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "delete_char_typo"
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")

    @staticmethod
    def transpose_char_typo(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue

                altered_value_text = None
                if len(original_value_from_meta) >= 2:
                    possible_indices = [
                        i for i in range(len(original_value_from_meta) - 1) 
                        if original_value_from_meta[i] != original_value_from_meta[i+1]
                    ]
                    if possible_indices:
                        idx = random.choice(possible_indices)
                        val_list = list(original_value_from_meta)
                        val_list[idx], val_list[idx+1] = val_list[idx+1], val_list[idx]
                        altered_value_text = "".join(val_list)
                
                if altered_value_text is None or original_value_from_meta == altered_value_text:
                    continue

                # Case-insensitive search for original_value_from_meta in current_q_being_modified
                match = re.search(re.escape(original_value_from_meta), current_q_being_modified, re.IGNORECASE)
                
                if match:
                    matched_segment_in_question = match.group(0)
                    
                    start_index = match.start()
                    end_index = match.end()
                    
                    # Construct next_q_state by replacing the actual matched segment
                    next_q_state = current_q_being_modified[:start_index] + \
                                   altered_value_text + \
                                   current_q_being_modified[end_index:]
                                   
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": matched_segment_in_question, # Store the actual segment that was replaced
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "transpose_char_typo"
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")

    @staticmethod
    def add_middle_space(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                altered_value_text = None
                if len(original_value_from_meta) >= 1: 
                    middle_index = len(original_value_from_meta) // 2
                    if middle_index == 0 and len(original_value_from_meta) == 1: # "A" -> "A "
                         altered_value_text = original_value_from_meta + " "
                    elif middle_index > 0 :
                        altered_value_text = original_value_from_meta[:middle_index] + " " + original_value_from_meta[middle_index:]
                    else: 
                        altered_value_text = original_value_from_meta


                if altered_value_text is None or original_value_from_meta == altered_value_text:
                    continue

                if original_value_from_meta in current_q_being_modified:
                    next_q_state = current_q_being_modified.replace(original_value_from_meta, altered_value_text, 1)
                    
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": original_value_from_meta,
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "middle_space_addition" 
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    
    @staticmethod
    def _apply_selective_space_removal(value_str: str) -> str:
        if ' ' not in value_str:
            return value_str

        space_indices = [i for i, char in enumerate(value_str) if char == ' ']
        num_spaces = len(space_indices)

        if num_spaces == 0: # Should be caught by ' ' not in value_str check earlier
            return value_str
        
        if num_spaces == 1:
            idx_to_remove = space_indices[0]
            return value_str[:idx_to_remove] + value_str[idx_to_remove+1:]
        
        # num_spaces > 1: keep the "middle" space character
        # The middle space is determined by its order of appearance.
        middle_space_occurrence_index = (num_spaces - 1) // 2
        char_idx_of_space_to_keep = space_indices[middle_space_occurrence_index]
        
        new_str_chars = []
        for i, char in enumerate(value_str):
            if char == ' ':
                if i == char_idx_of_space_to_keep:
                    new_str_chars.append(' ') # Keep this specific space
                # Else (it's a space but not the one to keep), do nothing (remove it)
            else:
                new_str_chars.append(char) # Keep non-space characters
            
        return "".join(new_str_chars)

    @staticmethod
    def remove_spaces_selectively(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                # Apply the selective space removal logic
                altered_value_text = BenchmarkVariator._apply_selective_space_removal(original_value_from_meta)

                if original_value_from_meta == altered_value_text:
                    continue # No change was made (e.g., no spaces, or logic resulted in same string)

                if original_value_from_meta in current_q_being_modified:
                    next_q_state = current_q_being_modified.replace(original_value_from_meta, altered_value_text, 1)
                    
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": original_value_from_meta,
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "selective_space_removal" 
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
         
    @staticmethod
    def remove_punctuation_from_values(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []
        
        # Create a translation table that maps each punctuation character to None (for removal)
        punctuation_remover = str.maketrans('', '', string.punctuation)

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                if not any(char in string.punctuation for char in original_value_from_meta):
                    continue
                
                altered_value_text = original_value_from_meta.translate(punctuation_remover)
                if original_value_from_meta == altered_value_text or not altered_value_text:
                    continue

                if original_value_from_meta in current_q_being_modified:
                    next_q_state = current_q_being_modified.replace(original_value_from_meta, altered_value_text, 1)
                    
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": original_value_from_meta,
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "punctuation_removal" 
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
    @staticmethod
    def _alter_first_punctuation(value_str: str) -> str:
        if not any(char in string.punctuation for char in value_str):
            return value_str

        value_list = list(value_str)
        altered = False
        for i, char_in_value in enumerate(value_list):
            if char_in_value in string.punctuation:
                if char_in_value == '.':
                    value_list[i] = '-'
                elif char_in_value == '-':
                    value_list[i] = '.'
                else:
                    value_list[i] = random.choice(['.', '-'])
                altered = True
                break # Alter only the first encountered punctuation
        
        if altered:
            return "".join(value_list)
        return value_str # Should not be reached if punctuation was present and logic is correct

    @staticmethod
    def alter_punctuation_in_values(input_json_path: str, output_json_path: str):
        start_time = time.time()
        all_new_records = []

        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for original_record in data:
            current_q_being_modified = original_record['question']
            modifications_for_this_record = []
            made_at_least_one_successful_change = False

            for value_entry in original_record['values']:
                original_value_from_meta = str(value_entry['value'])

                if not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                # Check if the original value actually contains any punctuation
                if not any(char in string.punctuation for char in original_value_from_meta):
                    continue
                
                altered_value_text = BenchmarkVariator._alter_first_punctuation(original_value_from_meta)

                if original_value_from_meta == altered_value_text:
                    continue

                if original_value_from_meta in current_q_being_modified:
                    next_q_state = current_q_being_modified.replace(original_value_from_meta, altered_value_text, 1)
                    
                    if next_q_state != current_q_being_modified:
                        modifications_for_this_record.append({
                            "value_source_table": value_entry['table'],
                            "value_source_column": value_entry['column'],
                            "original_value_segment": original_value_from_meta,
                            "altered_value_segment": altered_value_text,
                            "alteration_type": "punctuation_alteration" 
                        })
                        current_q_being_modified = next_q_state
                        made_at_least_one_successful_change = True
            
            if made_at_least_one_successful_change:
                new_record_item = copy.deepcopy(original_record)
                new_record_item['question'] = current_q_being_modified
                new_record_item['changes_information'] = {
                    "original_nlq": original_record['question'],
                    "modifications": modifications_for_this_record
                }
                all_new_records.append(new_record_item)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
        
    @staticmethod
    def _build_synonym_prompt_messages(question: str, keyword: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding.
Your task is to provide a suitable synonym for a given keyword, considering its context in the provided question.
If no appropriate synonym exists, or if the keyword is a proper noun that should not be changed (like a specific city name, person's name, or a technical ID), you should output the special token: [NO_CHANGE].
If a synonym is found, output only the synonym. Do not output any other text, explanations, or the original keyword if no change is made. Just the synonym or [NO_CHANGE].

Here are some examples:

Question: "What is the price of the most expensive car?"
Keyword: "expensive"
Output: costly

Question: "Find all schools in Alameda county."
Keyword: "Alameda"
Output: [NO_CHANGE]

Question: "Show me flights from USA to UK."
Keyword: "USA"
Output: United States

Question: "How many students are named Michael?"
Keyword: "Michael"
Output: Mike

Question: "What is the status of order #AB123?"
Keyword: "AB123"
Output: [NO_CHANGE]

Now, process the following:

Question: "{question}"
Keyword: "{keyword}"
Output:"""
        return [{"role": "user", "content": prompt_template}]

    @staticmethod
    def parse_qwen3_output(raw_output_text: str) -> str:
        """
        Robustly parses Qwen3 output to remove thinking blocks and extract the intended response.
        Handles complete, incomplete, and malformed <think> tags.
        """
        # Make tag matching case-insensitive for robustness
        # 1. Remove all complete <think>...</think> blocks
        text_after_full_block_removal = re.sub(
            r"<think>.*?</think>",
            "",
            raw_output_text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # 2. If any <think> or </think> tag remnants are still present,
        #    it indicates an incomplete or malformed thinking block. Discard.
        if re.search(r"<think>|</think>", text_after_full_block_removal, re.IGNORECASE):
            return ""

        # 3. If the result is empty after processing, return empty.
        if not text_after_full_block_removal:
            return ""

        return text_after_full_block_removal

    @staticmethod
    def substitute_synonyms_with_vllm(input_json_path: str, output_json_path: str,
                                      model_name: str = LLM_MODEL_NAME_DEFAULT,
                                      cache_dir: str = LLM_CACHE_DIR_DEFAULT,
                                      tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
                                      quantization: str = LLM_QUANTIZATION_DEFAULT,
                                      gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT
                                     ):
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            print("vLLM or Transformers is not installed. Please install them to use this feature.")
            print(f"Processing completed in {time.time() - start_time:.2f} seconds.\n"
                  f"Output file: {output_json_path} (not created due to missing dependencies)\n"
                  f"Total new records generated: 0")
            return

        sampling_params = SamplingParams(
            temperature=0.0, # For deterministic output
            top_p=1.0,       # Consider all tokens (since temp is 0, this has less effect)
            max_tokens=4096 
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path, # Recommended by vLLM docs
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_value_from_meta = str(value_entry.get('value', ''))

                if not original_value_from_meta or \
                   not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                if original_value_from_meta in original_question_text:
                    messages = BenchmarkVariator._build_synonym_prompt_messages(original_question_text, original_value_from_meta)
                    try:
                        # Explicitly enable thinking, as per vLLM docs for Qwen3 behavior control
                        prompt_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True # Qwen3 models think by default; this makes it explicit.
                        )
                        prompts_for_vllm_generation.append(prompt_text)
                        batch_processing_info.append(
                            (record_idx, value_entry_idx, original_value_from_meta)
                        )
                    except Exception:
                        pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        print(f"First 5 inputs to LLM: {prompts_for_vllm_generation[:5]}")
        print(f"First 5 outputs from LLM: {llm_raw_responses[:5]}")
        llm_synonyms_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, keyword_sent_to_llm = batch_processing_info[i]
            
            parsed_synonym = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not parsed_synonym or \
               parsed_synonym == "[NO_CHANGE]" or \
               parsed_synonym.lower() == keyword_sent_to_llm.lower():
                continue 
            
            llm_synonyms_map[(record_idx, value_entry_idx)] = parsed_synonym

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_value_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                synonym_for_this_entry = llm_synonyms_map.get((record_idx, value_entry_idx))
                
                if not synonym_for_this_entry:
                    continue

                if original_value_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_value_str_for_this_entry, synonym_for_this_entry, 1)
                    
                    if temp_question_state != current_question_state:
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_value_str_for_this_entry,
                            "altered_value_segment": synonym_for_this_entry,
                            "alteration_type": "synonym_substitution"
                        })
                        change_occurred_in_record = True
            
            if change_occurred_in_record:
                new_record = copy.deepcopy(original_record_instance)
                new_record['question'] = current_question_state
                new_record['changes_information'] = {
                    "original_nlq": original_record_instance['question'],
                    "modifications": modifications_applied_to_this_record
                }
                all_new_records.append(new_record)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    
    @staticmethod
    def _build_abbreviation_prompt_messages(question: str, keyword: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and common abbreviations.
Your task is to provide a common abbreviation or acronym for a given keyword, considering its context in the provided question.
If a common, widely understood abbreviation exists for the keyword, output ONLY the abbreviation.
Make sure that the abbreviation is appropriate for the context and may indeed be used in the question.
If no common abbreviation exists, or if abbreviating the keyword would be inappropriate or ambiguous in this context (e.g., many proper nouns that are not typically abbreviated), you should output the special token: [NO_CHANGE].
Do not output any other text, explanations, or the original keyword if no change is made. Just the abbreviation or [NO_CHANGE].

Here are some examples:

Question: "What is the population of the United States of America?"
Keyword: "United States of America"
Output: USA

Question: "Find schools located in California."
Keyword: "California"
Output: CA

Question: "The patient was seen by Doctor Smith."
Keyword: "Doctor"
Output: Dr.

Question: "The package was delivered to 123 Main Street."
Keyword: "Street"
Output: St.

Question: "List all counties, including Alameda."
Keyword: "Alameda"
Output: [NO_CHANGE]

Question: "What is the total number of students?"
Keyword: "Number"
Output: No.

Question: "Convert 5 kilograms to pounds."
Keyword: "kilograms"
Output: kg

Question: "The lecture was given by Professor Jones."
Keyword: "Professor"
Output: Prof.

Question: "What is the maximum speed limit?"
Keyword: "maximum"
Output: max

Now, process the following:

Question: "{question}"
Keyword: "{keyword}"
Output:"""
        return [{"role": "user", "content": prompt_template}]
    
    
    @staticmethod
    def substitute_abbreviations_with_vllm(input_json_path: str, output_json_path: str,
                                           model_name: str = LLM_MODEL_NAME_DEFAULT,
                                           cache_dir: str = LLM_CACHE_DIR_DEFAULT,
                                           tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
                                           quantization: str = LLM_QUANTIZATION_DEFAULT,
                                           gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT
                                          ):
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            print("vLLM or Transformers is not installed. Please install them to use this feature.")
            print(f"Processing completed in {time.time() - start_time:.2f} seconds.\n"
                  f"Output file: {output_json_path} (not created due to missing dependencies)\n"
                  f"Total new records generated: 0")
            return

        sampling_params = SamplingParams(
            temperature=0.0, 
            top_p=1.0,       
            max_tokens=4096    
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
            max_model_len=8192
        )

        prompts_for_vllm_generation = []
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_value_from_meta = str(value_entry.get('value', ''))

                if not original_value_from_meta or \
                   not BenchmarkVariator._is_eligible_text_value(original_value_from_meta):
                    continue
                
                if original_value_from_meta in original_question_text:
                    # Use the new prompt builder for abbreviations
                    messages = BenchmarkVariator._build_abbreviation_prompt_messages(original_question_text, original_value_from_meta)
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True 
                        )
                        prompts_for_vllm_generation.append(prompt_text)
                        batch_processing_info.append(
                            (record_idx, value_entry_idx, original_value_from_meta)
                        )
                    except Exception:
                        pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        llm_abbreviations_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, keyword_sent_to_llm = batch_processing_info[i]
            
            parsed_abbreviation = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not parsed_abbreviation or \
               parsed_abbreviation == "[NO_CHANGE]" or \
               parsed_abbreviation.lower() == keyword_sent_to_llm.lower(): # No actual change
                continue 
            
            # Ensure the abbreviation is not longer than the original (common sense for abbreviations)
            if len(parsed_abbreviation) >= len(keyword_sent_to_llm):
                continue

            llm_abbreviations_map[(record_idx, value_entry_idx)] = parsed_abbreviation
        print(f"First 5 outputs to LLM: {llm_raw_responses[:5]}")
        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_value_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                abbreviation_for_this_entry = llm_abbreviations_map.get((record_idx, value_entry_idx))
                
                if not abbreviation_for_this_entry:
                    continue

                if original_value_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_value_str_for_this_entry, abbreviation_for_this_entry, 1)
                    
                    if temp_question_state != current_question_state:
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_value_str_for_this_entry,
                            "altered_value_segment": abbreviation_for_this_entry,
                            "alteration_type": "abbreviation_substitution" # Changed alteration type
                        })
                        change_occurred_in_record = True
            
            if change_occurred_in_record:
                new_record = copy.deepcopy(original_record_instance)
                new_record['question'] = current_question_state
                new_record['changes_information'] = {
                    "original_nlq": original_record_instance['question'],
                    "modifications": modifications_applied_to_this_record
                }
                all_new_records.append(new_record)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
    @staticmethod
    def _build_full_question_paraphrase_prompt_messages(original_question: str, keyword_to_paraphrase: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and query paraphrasing, specifically for questions interacting with databases.
Your task is to rephrase the 'Original Question'. The rephrasing should focus on how the concept represented by the 'Keyword' (which is a direct database value) is expressed.
The goal is to create a 'New Paraphrased Question' that is semantically equivalent but uses different phrasing or sentence structure. The original 'Keyword' itself might not appear verbatim in the new question, but its meaning and role should be clearly conveyed through the paraphrase.
Output ONLY the complete 'New Paraphrased Question'.
If the 'Keyword's concept cannot be meaningfully rephrased to create a natural-sounding and distinct new question (e.g., it's a very specific ID, a proper noun like a city name that's already concise, or a simple number without clear semantic context for rephrasing its role), output the special token: [NO_CHANGE].

Here are some examples where 'Keyword' is a database value and the question is paraphrased based on its concept:

Original Question: "Show products with status 'unavailable'."
Keyword: "unavailable"
New Paraphrased Question: Show products that are currently out of stock.

Original Question: "List all employees with employment_type 'Full-Time'."
Keyword: "Full-Time"
New Paraphrased Question: Display all staff members who are permanently employed.

Original Question: "What are the tickets for event_priority 'High'?"
Keyword: "High"
New Paraphrased Question: Can you show me the tickets that are marked as urgent priority?

Original Question: "Get details for order_id 'XYZ123'."
Keyword: "XYZ123"
New Paraphrased Question: [NO_CHANGE]

Original Question: "Find accounts with account_status 'CLOSED'."
Keyword: "CLOSED"
New Paraphrased Question: Find accounts that are no longer active.

Original Question: "Show me orders with shipping_method 'Express'."
Keyword: "Express"
New Paraphrased Question: Display orders that are designated for expedited shipping.

Original Question: "Filter by payment_status 'Paid'."
Keyword: "Paid"
New Paraphrased Question: Show me records where the payment has been completed.

Now, process the following:

Original Question: "{original_question}"
Keyword: "{keyword_to_paraphrase}"
New Paraphrased Question:"""
        return [{"role": "user", "content": prompt_template}]
    
    @staticmethod
    def substitute_full_question_paraphrases_with_vllm(input_json_path: str, output_json_path: str,
                                                       model_name: str = LLM_MODEL_NAME_DEFAULT,
                                                       cache_dir: str = LLM_CACHE_DIR_DEFAULT,
                                                       tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
                                                       quantization: str = LLM_QUANTIZATION_DEFAULT,
                                                       gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT
                                                      ):
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            print("vLLM or Transformers is not installed. Please install them to use this feature.")
            print(f"Processing completed in {time.time() - start_time:.2f} seconds.\n"
                  f"Output file: {output_json_path} (not created due to missing dependencies)\n"
                  f"Total new records generated: 0")
            return

        sampling_params = SamplingParams(
            temperature=0.3, 
            top_p=0.95,       
            max_tokens=256    
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name, tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size, trust_remote_code=True,
            download_dir=cache_dir, quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text: continue

            for value_entry in record.get('values', []):
                keyword_to_paraphrase = str(value_entry.get('value', ''))

                if not keyword_to_paraphrase or \
                   not BenchmarkVariator._is_eligible_text_value(keyword_to_paraphrase):
                    continue
                
                if keyword_to_paraphrase not in original_question_text:
                    continue

                messages = BenchmarkVariator._build_full_question_paraphrase_prompt_messages(
                    original_question_text, keyword_to_paraphrase
                )
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append(
                        (record_idx, original_question_text, keyword_to_paraphrase, copy.deepcopy(value_entry))
                    )
                except Exception:
                    pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        successful_full_paraphrases_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, original_q_text, keyword_that_triggered, val_entry_data = batch_processing_info[i]
            
            if record_idx in successful_full_paraphrases_map:
                continue

            parsed_new_full_question = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not parsed_new_full_question or \
               parsed_new_full_question == "[NO_CHANGE]" or \
               parsed_new_full_question.lower() == original_q_text.lower(): 
                continue 
            
            # Additional check: ensure the original keyword is NOT in the new question,
            # if we want to enforce that the paraphrase truly rephrased its concept away.
            # This might be too strict, as sometimes the keyword might naturally reappear.
            # For now, let's comment it out and rely on the prompt and semantic equivalence.
            # if keyword_that_triggered.lower() in parsed_new_full_question.lower():
            #     continue

            successful_full_paraphrases_map[record_idx] = {
                "new_question": parsed_new_full_question,
                "trigger_keyword": keyword_that_triggered,
                "trigger_value_entry": val_entry_data
            }

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            if record_idx in successful_full_paraphrases_map:
                paraphrase_info = successful_full_paraphrases_map[record_idx]
                new_full_question = paraphrase_info["new_question"]
                trigger_keyword = paraphrase_info["trigger_keyword"]
                trigger_value_entry = paraphrase_info["trigger_value_entry"]

                new_record = copy.deepcopy(original_record_instance)
                new_record['question'] = new_full_question 
                
                new_record['changes_information'] = {
                    "original_nlq": original_record_instance['question'],
                    "modifications": [
                        {
                            "value_source_table": trigger_value_entry['table'],
                            "value_source_column": trigger_value_entry['column'],
                            "original_value_segment": trigger_keyword, 
                            "altered_value_segment": new_full_question, 
                            "alteration_type": "full_question_paraphrase" 
                        }
                    ]
                }
                all_new_records.append(new_record)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
        
        
if __name__ == "__main__":
    input_path = "assets/value_linking_valid_values_exact_no_bird_train.json"
    output_folder = "assets/all_benchmarks"
    
    
    """
    output_filename = "typo_substitution.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.substitute_char_typo(input_path, output_path)
    output_filename = "typo_insertion.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.insert_char_typo(input_path, output_path)
    output_filename = "typo_deletion.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.delete_char_typo(input_path, output_path)
    output_filename = "typo_transposition.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.transpose_char_typo(input_path, output_path)"""

    output_filename = "abbreviations_awq.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.substitute_abbreviations_with_vllm(input_path, output_path)

