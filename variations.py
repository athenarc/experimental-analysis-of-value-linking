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
        del llm
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
        del llm
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
            temperature=0, 
            top_p=0.95,       
            max_tokens=4096    
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
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    
    @staticmethod
    def _build_antonym_rewrite_prompt_messages(question: str, keyword: str) -> list:
        # This prompt asks the LLM to rewrite the question.
        # It's a more complex task.
        prompt_template = f"""You are an expert in natural language understanding and rewriting sentences.
Your task is to rewrite the given question by replacing the specified 'Keyword' with its antonym, while ensuring the overall meaning of the question remains the same. This involves adding or removing negations or rephrasing parts of the sentence.
Output only the rewritten question.
If it's not possible to naturally rewrite the question this way, or if the keyword is a proper noun or ID that shouldn't be changed, output the special token: [NO_CHANGE].

Here are some examples:

Original Question: "Show me all active users."
Keyword: "active"
Rewritten Question: Show me all users who are not inactive.

Original Question: "Find all apple products."
Keyword: "apple"
Rewritten Question: [NO_CHANGE]

Original Question: "List products that are available."
Keyword: "available"
Rewritten Question: List products that are not unavailable.

Original Question: "Find employees who are permanent staff."
Keyword: "permanent"
Rewritten Question: Find employees who are not temporary staff.

Original Question: "Find employees with id x_123"
Keyword: "x_123"
Rewritten Question: [NO_CHANGE]

Original Question: "Filter for enabled features."
Keyword: "enabled"
Rewritten Question: Filter for features that are not disabled.

Original Question: "Are there any valid entries?"
Keyword: "valid"
Rewritten Question: Are there any entries that are not invalid?

Original Question: "Search for completed tasks."
Keyword: "completed"
Rewritten Question: Search for tasks that are not incomplete.

Original Question: "Show me results for Alameda county."
Keyword: "Alameda"
Rewritten Question: [NO_CHANGE]

Now, process the following:

Original Question: "{question}"
Keyword: "{keyword}"
Rewritten Question:"""
        return [{"role": "user", "content": prompt_template}]
    
    @staticmethod
    def rewrite_with_antonym_negation_vllm(input_json_path: str, output_json_path: str,
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
            print("vLLM or Transformers not installed.")
            return

        # For question rewriting, we might need more tokens.
        # Temperature is kept low for more deterministic rewrites.
        sampling_params = SamplingParams(
            temperature=0, # Slightly higher for more creative but still constrained rewrites
            top_p=0.95,
            max_tokens=4096 # Allow for full question rewrite
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        # (record_idx, original_question_text, keyword_for_this_prompt, value_entry_for_logging)
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # Store which (record_idx, value_str_lower) have been processed to avoid redundant LLM calls
        # if the same value appears multiple times in a record's 'values' list but we only want to try rewriting for it once.
        processed_keywords_for_record = {}


        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue
            
            processed_keywords_for_record[record_idx] = set()

            # We iterate through values to find keywords, but the LLM rewrites the whole question.
            # We only want to attempt one successful rewrite per original question.
            # However, we might try multiple keywords from that question if the first ones fail.
            for value_entry in record.get('values', []):
                keyword_to_try = str(value_entry.get('value', ''))

                if not keyword_to_try or \
                   not BenchmarkVariator._is_eligible_text_value(keyword_to_try) or \
                   keyword_to_try.lower() in processed_keywords_for_record[record_idx]:
                    continue
                
                # Check if keyword is in question (case-insensitive for robustness, but use original case for prompt)
                # Find actual casing of keyword in question
                match = re.search(re.escape(keyword_to_try), original_question_text, re.IGNORECASE)
                if not match:
                    continue
                
                actual_keyword_in_question = match.group(0) # Use the cased version from the question

                messages = BenchmarkVariator._build_antonym_rewrite_prompt_messages(original_question_text, actual_keyword_in_question)
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append(
                        (record_idx, original_question_text, actual_keyword_in_question, value_entry)
                    )
                    processed_keywords_for_record[record_idx].add(keyword_to_try.lower())
                except Exception:
                    pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        # Store successful rewrites: record_idx -> (rewritten_question, keyword_used, original_value_entry)
        successful_rewrites = {} 

        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, original_q_text, keyword_used, value_entry_info = batch_processing_info[i]
            
            # If this record_idx already has a successful rewrite, skip further processing for it.
            if record_idx in successful_rewrites:
                continue

            rewritten_q = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not rewritten_q or \
               rewritten_q == "[NO_CHANGE]" or \
               rewritten_q.lower() == original_q_text.lower(): # No actual change
                continue
            
            # Basic check: ensure the original keyword is NOT in the rewritten question
            # and that some form of its antonym (or related words) might be. This is heuristic.
            if keyword_used.lower() in rewritten_q.lower():
                # This might happen if LLM fails to replace.
                # A more sophisticated check would be to see if an antonym of keyword_used is present.
                # For now, if original keyword is still there, assume it failed to rewrite properly.
                continue

            successful_rewrites[record_idx] = (rewritten_q, keyword_used, value_entry_info)

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            if record_idx in successful_rewrites:
                rewritten_question, keyword_that_triggered_rewrite, val_entry = successful_rewrites[record_idx]
                
                new_record = copy.deepcopy(original_record_instance)
                new_record['question'] = rewritten_question
                new_record['changes_information'] = {
                    "original_nlq": original_record_instance['question'],
                    "modifications": [
                        {
                            "value_source_table": val_entry['table'],
                            "value_source_column": val_entry['column'],
                            "original_keyword_for_rewrite": keyword_that_triggered_rewrite,
                            "original_value_segment_in_db": val_entry['value'], # The actual DB value
                            "alteration_type": "antonym_negation_rewrite"
                            # Note: altered_value_segment is now the whole question.
                        }
                    ]
                }
                all_new_records.append(new_record)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
    @staticmethod
    def _build_word_removal_prompt_messages(question: str, multi_word_phrase: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and concise phrasing.
Your task is to shorten the given 'Multi-Word Phrase' (which appears in the 'Original Question') by removing one or more words.
The shortened phrase MUST still clearly and unambiguously refer to the exact same entity or concept as the original phrase within the context of the question.
Prioritize removing common, less informative words (e.g., articles like 'the', 'a'; prepositions like 'of', 'in', 'for'; generic nouns like 'city of', 'department of', 'state of') if the remaining words form a strong and unique identifier.
Do not remove core identifying words if it leads to ambiguity or changes the meaning.
If no words can be safely removed without losing clarity or introducing ambiguity, output the special token: [NO_CHANGE].
Output only the shortened phrase or [NO_CHANGE].

Here are some examples:

Original Question: "Find attractions in the city of London."
Multi-Word Phrase: "city of London"
Shortened Phrase: London

Original Question: "Details for employee Mr. Johnathan Smitherson."
Multi-Word Phrase: "Mr. Johnathan Smitherson"
Shortened Phrase: Johnathan Smitherson

Original Question: "Report for the Department of Health and Human Services."
Multi-Word Phrase: "Department of Health and Human Services"
Shortened Phrase: Health and Human Services

Original Question: "Information about Fantastic Four."
Multi-Word Phrase: "Fantastic Four"
Shortened Phrase: [NO_CHANGE]

Original Question: "Show me the table of contents."
Multi-Word Phrase: "table of contents"
Shortened Phrase: contents

Original Question: "List all items in the 'Category of General Goods'."
Multi-Word Phrase: "Category of General Goods"
Shortened Phrase: General Goods

Original Question: "Who is the current president of the United States of America?"
Multi-Word Phrase: "United States of America"
Shortened Phrase: United States

Original Question: "Find the birth of Mike James."
Multi-Word Phrase: "Mike James"
Shortened Phrase: [NO_CHANGE]

Now, process the following:

Original Question: "{question}"
Multi-Word Phrase: "{multi_word_phrase}"
Shortened Phrase:"""
        return [{"role": "user", "content": prompt_template}]

    @staticmethod
    def shorten_phrases_by_word_removal_vllm(input_json_path: str, output_json_path: str,
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
            print("vLLM or Transformers not installed.")
            return

        sampling_params = SamplingParams(
            temperature=0.0, # For deterministic shortening
            top_p=1.0,
            max_tokens=4096 # Max length of the shortened phrase
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        # (record_idx, value_entry_idx, original_multi_word_phrase_from_meta)
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_phrase_from_meta = str(value_entry.get('value', ''))

                # Eligibility for this specific transformation:
                # 1. General text eligibility
                # 2. Must have more than 2 words
                # 3. Must be present in the question (case-sensitive for replacement later)
                if not BenchmarkVariator._is_eligible_text_value(original_phrase_from_meta) or \
                   len(original_phrase_from_meta.split()) <= 2 or \
                   original_phrase_from_meta not in original_question_text:
                    continue
                
                # The phrase from meta is what we try to shorten.
                # It's guaranteed to be in original_question_text at this point.
                messages = BenchmarkVariator._build_word_removal_prompt_messages(original_question_text, original_phrase_from_meta)
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append(
                        (record_idx, value_entry_idx, original_phrase_from_meta)
                    )
                except Exception:
                    pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        # Store shortened phrases: (record_idx, value_entry_idx) -> "shortened_phrase_text"
        llm_shortened_phrases_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, original_phrase = batch_processing_info[i]
            
            shortened_phrase = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not shortened_phrase or \
               shortened_phrase == "[NO_CHANGE]" or \
               shortened_phrase.lower() == original_phrase.lower() or \
               len(shortened_phrase) >= len(original_phrase): # Must be shorter
                continue 
            
            llm_shortened_phrases_map[(record_idx, value_entry_idx)] = shortened_phrase

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            # Iterate through values of *this* record in their original order to apply changes
            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_phrase_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                shortened_version = llm_shortened_phrases_map.get((record_idx, value_entry_idx))
                
                if not shortened_version:
                    continue 

                # Attempt to replace in the *current state* of the question
                # This uses the original_phrase_str_for_this_entry which was confirmed to be in the question
                if original_phrase_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_phrase_str_for_this_entry, shortened_version, 1)
                    
                    if temp_question_state != current_question_state: 
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_phrase_str_for_this_entry,
                            "altered_value_segment": shortened_version,
                            "alteration_type": "word_removal_shortening"
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
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    @staticmethod
    def _build_word_addition_prompt_messages(question: str, keyword: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and descriptive phrasing.
Your task is to enrich the given 'Keyword' (which appears in the 'Original Question') by adding a short, relevant, descriptive word or phrase before or after it.
The added words should be contextually appropriate and provide common, often redundant, information that a robust system might ignore.
The core meaning of the keyword and its reference should not change.
Do not make the keyword itself part of a fundamentally different, longer, official name if that's not the intent. The goal is to add descriptive fluff.
If no natural and safe enrichment can be added, or if the keyword is part of a very specific identifier that shouldn't be altered, output the special token: [NO_CHANGE].
Output only the enriched phrase (original keyword + added words) or [NO_CHANGE].

Here are some examples:

Original Question: "Find attractions in London."
Keyword: "London"
Enriched Phrase: the city of London

Original Question: "Details for employee John Smith."
Keyword: "John Smith"
Enriched Phrase: Mr. John Smith

Original Question: "Report for Health and Human Services."
Keyword: "Health and Human Services"
Enriched Phrase: the Department of Health and Human Services

Original Question: "Information about New York."
Keyword: "New York"
Enriched Phrase: New York City

Original Question: "Show me the contents."
Keyword: "contents"
Enriched Phrase: table of contents

Original Question: "List all items in General Goods."
Keyword: "General Goods"
Enriched Phrase: the Category of General Goods

Original Question: "Who is the current president of the United States?"
Keyword: "United States"
Enriched Phrase: United States of America

Original Question: "Find the main branch of Bank of America."
Keyword: "Bank of America"
Enriched Phrase: [NO_CHANGE]

Original Question: "Hotels in Paris."
Keyword: "Paris"
Enriched Phrase: the beautiful city of Paris

Original Question: "Flights to LAX."
Keyword: "LAX"
Enriched Phrase: LAX airport

Now, process the following:

Original Question: "{question}"
Keyword: "{keyword}"
Enriched Phrase:"""
        return [{"role": "user", "content": prompt_template}]

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
    def enrich_phrases_by_word_addition_vllm(input_json_path: str, output_json_path: str,
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
            print("vLLM or Transformers not installed.")
            return

        sampling_params = SamplingParams(
            temperature=0, # Slightly more creative for descriptive additions
            top_p=0.95,
            max_tokens=4096 # Max length of the enriched phrase
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        # (record_idx, value_entry_idx, original_keyword_from_meta)
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_keyword_from_meta = str(value_entry.get('value', ''))

                # Eligibility:
                # 1. General text eligibility
                # 2. Must be present in the question (case-sensitive for replacement later)
                if not BenchmarkVariator._is_eligible_text_value(original_keyword_from_meta) or \
                   original_keyword_from_meta not in original_question_text:
                    continue
                
                messages = BenchmarkVariator._build_word_addition_prompt_messages(original_question_text, original_keyword_from_meta)
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append(
                        (record_idx, value_entry_idx, original_keyword_from_meta)
                    )
                except Exception:
                    pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        # Store enriched phrases: (record_idx, value_entry_idx) -> "enriched_phrase_text"
        llm_enriched_phrases_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, original_keyword = batch_processing_info[i]
            
            enriched_phrase = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not enriched_phrase or \
               enriched_phrase == "[NO_CHANGE]" or \
               enriched_phrase.lower() == original_keyword.lower() or \
               len(enriched_phrase) <= len(original_keyword) or \
               original_keyword.lower() not in enriched_phrase.lower(): # Original keyword must be part of the enriched phrase
                continue 
            
            llm_enriched_phrases_map[(record_idx, value_entry_idx)] = enriched_phrase

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_keyword_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                enriched_version = llm_enriched_phrases_map.get((record_idx, value_entry_idx))
                
                if not enriched_version:
                    continue 

                if original_keyword_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_keyword_str_for_this_entry, enriched_version, 1)
                    
                    if temp_question_state != current_question_state: 
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_keyword_str_for_this_entry,
                            "altered_value_segment": enriched_version,
                            "alteration_type": "word_addition_enrichment"
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
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    
    @staticmethod
    def _build_word_reorder_prompt_messages(question: str, multi_word_phrase: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and syntax.
Your task is to reorder the words in the given 'Multi-Word Phrase' (which appears in the 'Original Question') to form a new arrangement.
The reordered phrase MUST still clearly refer to the exact same entity or concept and sound natural. Prioritize rearrangements that maintain or improve naturalness. Avoid nonsensical or grammatically incorrect orderings.
The reordered phrase must use all the original words exactly once.
If no meaningful or natural reordering is possible (e.g., the phrase is already in its most natural order, or any change would make it awkward or change meaning), output the special token: [NO_CHANGE].
Output only the reordered phrase or [NO_CHANGE].

Here are some examples:

Original Question: "Show me the big red car."
Multi-Word Phrase: "big red car"
Reordered Phrase: red big car

Original Question: "Information about the United States of America."
Multi-Word Phrase: "United States of America"
Reordered Phrase: America of the United States

Original Question: "Details for the first national bank."
Multi-Word Phrase: "first national bank"
Reordered Phrase: national first bank

Original Question: "Report for the Department of Health."
Multi-Word Phrase: "Department of Health"
Reordered Phrase: Health Department

Original Question: "Find the New York City Marathon."
Multi-Word Phrase: "New York City Marathon"
Reordered Phrase: Marathon of New York City

Original Question: "A tale of two cities."
Multi-Word Phrase: "tale of two cities"
Reordered Phrase: [NO_CHANGE]

Original Question: "The quick brown fox."
Multi-Word Phrase: "quick brown fox"
Reordered Phrase: brown quick fox

Now, process the following:

Original Question: "{question}"
Multi-Word Phrase: "{multi_word_phrase}"
Reordered Phrase:"""
        return [{"role": "user", "content": prompt_template}]

    @staticmethod
    def reorder_words_in_phrases_vllm_hybrid(input_json_path: str, output_json_path: str,
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
            print("vLLM or Transformers not installed.")
            return

        sampling_params = SamplingParams(
            temperature=0, # Low temperature for more constrained, natural reordering
            top_p=0.95,
            max_tokens=4096 # Max length of the reordered phrase
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        # (record_idx, value_entry_idx, original_phrase_from_meta, num_words)
        batch_processing_info = []
        
        # Store direct reorders for 2-word phrases: (record_idx, value_entry_idx) -> "reordered_phrase"
        direct_reorders_map = {}

        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_phrase_from_meta = str(value_entry.get('value', ''))
                words = original_phrase_from_meta.split()
                num_words = len(words)

                # Eligibility for this specific transformation:
                # 1. General text eligibility
                # 2. Must have more than 1 word
                # 3. Must be present in the question (case-sensitive for replacement later)
                if not BenchmarkVariator._is_eligible_text_value(original_phrase_from_meta) or \
                   num_words <= 1 or \
                   original_phrase_from_meta not in original_question_text:
                    continue
                
                if num_words == 2:
                    # Direct swap for 2-word phrases
                    if words[0].lower() != words[1].lower(): # Avoid swapping if words are identical (case-insensitive)
                        reordered_phrase = f"{words[1]} {words[0]}"
                        direct_reorders_map[(record_idx, value_entry_idx)] = reordered_phrase
                    # If words are same, it's effectively [NO_CHANGE] for 2-word case
                elif num_words > 2:
                    # Prepare for LLM for phrases with 3+ words
                    messages = BenchmarkVariator._build_word_reorder_prompt_messages(original_question_text, original_phrase_from_meta)
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                        )
                        prompts_for_vllm_generation.append(prompt_text)
                        batch_processing_info.append(
                            (record_idx, value_entry_idx, original_phrase_from_meta, num_words)
                        )
                    except Exception:
                        pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        # Store LLM-based reorders: (record_idx, value_entry_idx) -> "reordered_phrase_text"
        llm_reorders_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, original_phrase, num_original_words = batch_processing_info[i]
            
            reordered_phrase_llm = BenchmarkVariator.parse_qwen3_output(raw_response)
            reordered_words_llm = reordered_phrase_llm.split()

            # Basic validation for LLM output:
            # 1. Not empty, not [NO_CHANGE], different from original
            # 2. Must contain the same number of words as original
            # 3. All original words must be present in the reordered phrase (case-insensitive check for robustness)
            original_words_set_lower = {w.lower() for w in original_phrase.split()}
            reordered_words_set_lower_llm = {w.lower() for w in reordered_words_llm}

            if not reordered_phrase_llm or \
               reordered_phrase_llm == "[NO_CHANGE]" or \
               reordered_phrase_llm.lower() == original_phrase.lower() or \
               len(reordered_words_llm) != num_original_words or \
               original_words_set_lower != reordered_words_set_lower_llm:
                continue 
            
            llm_reorders_map[(record_idx, value_entry_idx)] = reordered_phrase_llm

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_phrase_str_for_this_entry = str(value_entry_data.get('value', ''))
                reordered_version = None

                if (record_idx, value_entry_idx) in direct_reorders_map:
                    reordered_version = direct_reorders_map[(record_idx, value_entry_idx)]
                elif (record_idx, value_entry_idx) in llm_reorders_map:
                    reordered_version = llm_reorders_map[(record_idx, value_entry_idx)]
                
                if not reordered_version:
                    continue 

                if original_phrase_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_phrase_str_for_this_entry, reordered_version, 1)
                    
                    if temp_question_state != current_question_state: 
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_phrase_str_for_this_entry,
                            "altered_value_segment": reordered_version,
                            "alteration_type": "word_reorder"
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
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        
    @staticmethod
    def _build_singular_plural_toggle_prompt_messages(question: str, keyword: str) -> list:
        prompt_template = f"""You are an expert in English morphology.
Your task is to convert the given 'Keyword' (which appears in the 'Original Question') between its singular and plural form.
- If the keyword is singular, convert it to its plural form.
- If the keyword is plural, convert it to its singular form.
Consider irregular forms (e.g., child/children, mouse/mice, man/men).
If the keyword is a proper noun (like a name or specific place), a mass noun that doesn't typically pluralize (e.g., 'information', 'water'), an acronym, or if changing its form would be unnatural or incorrect in this context, output the special token: [NO_CHANGE].
Output only the converted word or [NO_CHANGE].

Here are some examples:

Original Question: "List all products."
Keyword: "products"
Converted Word: product

Original Question: "Find the child seat."
Keyword: "child"
Converted Word: children

Original Question: "How many men work here?"
Keyword: "men"
Converted Word: man

Original Question: "Information about Alameda county."
Keyword: "Alameda"
Converted Word: [NO_CHANGE]

Original Question: "Show me the data."
Keyword: "data"
Converted Word: datum

Original Question: "What are the criteria?"
Keyword: "criteria"
Converted Word: criterion

Original Question: "Any news on this?"
Keyword: "news"
Converted Word: [NO_CHANGE]

Original Question: "Details for the geese."
Keyword: "geese"
Converted Word: goose

Now, process the following:

Original Question: "{question}"
Keyword: "{keyword}"
Converted Word:"""
        return [{"role": "user", "content": prompt_template}]
    
    @staticmethod
    def toggle_singular_plural_vllm(input_json_path: str, output_json_path: str,
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
            print("vLLM or Transformers not installed.")
            return

        sampling_params = SamplingParams(
            temperature=0.0, # For deterministic conversion
            top_p=1.0,
            max_tokens=4096 # Singular/plural forms are usually short
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_name,
            tokenizer=tokenizer.name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            quantization=quantization if quantization else None,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        prompts_for_vllm_generation = []
        # (record_idx, value_entry_idx, original_keyword_from_meta)
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            for value_entry_idx, value_entry in enumerate(record.get('values', [])):
                original_keyword_from_meta = str(value_entry.get('value', ''))

                # Eligibility for this specific transformation:
                # 1. Must be a single word (checked by _is_eligible_text_value refinement)
                # 2. Must be present in the question (case-sensitive for replacement)
                if not BenchmarkVariator._is_eligible_text_value(original_keyword_from_meta) or \
                   original_keyword_from_meta not in original_question_text:
                    continue
                
                messages = BenchmarkVariator._build_singular_plural_toggle_prompt_messages(original_question_text, original_keyword_from_meta)
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    prompts_for_vllm_generation.append(prompt_text)
                    batch_processing_info.append(
                        (record_idx, value_entry_idx, original_keyword_from_meta)
                    )
                except Exception:
                    pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        # Store converted words: (record_idx, value_entry_idx) -> "converted_word_text"
        llm_converted_words_map = {}
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, original_keyword = batch_processing_info[i]
            
            # Assuming parse_qwen3_output is available in the class
            converted_word = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not converted_word or \
               converted_word == "[NO_CHANGE]" or \
               converted_word.lower() == original_keyword.lower():
                continue 
            
            # Ensure the converted word is also a single token for consistency
            if len(converted_word.split()) > 1:
                continue

            llm_converted_words_map[(record_idx, value_entry_idx)] = converted_word

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_keyword_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                converted_version = llm_converted_words_map.get((record_idx, value_entry_idx))
                
                if not converted_version:
                    continue 

                if original_keyword_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(original_keyword_str_for_this_entry, converted_version, 1)
                    
                    if temp_question_state != current_question_state: 
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_keyword_str_for_this_entry,
                            "altered_value_segment": converted_version,
                            "alteration_type": "singular_plural_toggle"
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
        del llm
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
    
    @staticmethod
    def _build_entity_clarification_prompt_messages(question: str, keyword: str) -> list:
        prompt_template = f"""You are an expert in natural language understanding and entity recognition.
Your task is to clarify or augment a given keyword from a question by adding a common entity designator (e.g., "Inc.", "City", "County", "State") or making its type more explicit, if doing so adds necessary clarity or specificity that might be missing.
If the keyword is already specific, universally understood in context, or if no clarification is applicable or helpful, output the special token: [NO_CHANGE].
Output only the clarified/augmented entity or [NO_CHANGE]. Do not output explanations.

Here are some examples:

Question: "Show me employees of Google."
Keyword: "Google"
Output: Google Inc.

Question: "In Los Angeles how many schools have more than 500 free meals?"
Keyword: "Los Angeles"
Output: Los Angeles city

Question: "What are the services offered by Amazon?"
Keyword: "Amazon"
Output: Amazon Inc.

Question: "What's the name of customer with id x_123"
Keyword: "x_123"
Output: [NO_CHANGE]

Question: "How many people live in Texas?"
Keyword: "Texas"
Output: Texas State


ing:

Question: "{question}"
Keyword: "{keyword}"
Output:"""
        return [{"role": "user", "content": prompt_template}]

    @staticmethod
    def augment_entities_with_vllm(input_json_path: str, output_json_path: str,
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
            max_tokens=70 # Slightly more tokens for potentially longer clarifications like "County" or "State"
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
        )

        prompts_for_vllm_generation = []
        batch_processing_info = [] # (record_idx, value_entry_idx, original_value_from_meta)
        
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
                    messages = BenchmarkVariator._build_entity_clarification_prompt_messages(
                        original_question_text, original_value_from_meta
                    )
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
        
        llm_clarifications_map = {} # (record_idx, value_entry_idx) -> "clarified_entity_text"
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, value_entry_idx, keyword_sent_to_llm = batch_processing_info[i]
            
            parsed_clarification = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not parsed_clarification or \
               parsed_clarification == "[NO_CHANGE]" or \
               parsed_clarification.lower() == keyword_sent_to_llm.lower(): # No actual change
                continue 
            
            llm_clarifications_map[(record_idx, value_entry_idx)] = parsed_clarification

        all_new_records = []
        for record_idx, original_record_instance in enumerate(original_data):
            current_question_state = original_record_instance['question']
            modifications_applied_to_this_record = []
            change_occurred_in_record = False

            for value_entry_idx, value_entry_data in enumerate(original_record_instance.get('values', [])):
                original_value_str_for_this_entry = str(value_entry_data.get('value', ''))
                
                clarification_for_this_entry = llm_clarifications_map.get((record_idx, value_entry_idx))
                
                if not clarification_for_this_entry:
                    continue

                if original_value_str_for_this_entry in current_question_state:
                    temp_question_state = current_question_state.replace(
                        original_value_str_for_this_entry, clarification_for_this_entry, 1
                    )
                    
                    if temp_question_state != current_question_state:
                        current_question_state = temp_question_state
                        modifications_applied_to_this_record.append({
                            "value_source_table": value_entry_data['table'],
                            "value_source_column": value_entry_data['column'],
                            "original_value_segment": original_value_str_for_this_entry,
                            "altered_value_segment": clarification_for_this_entry,
                            "alteration_type": "entity_clarification" # Changed type
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
    def _build_number_to_word_prompt_messages(question: str) -> list:
        prompt_template = f"""You are an expert in natural language processing.
Your task is to rewrite the given question by converting standalone numerical digits into their word representations (e.g., "5" to "five", "250" to "two hundred fifty").
Only convert numbers that represent quantities, counts, or ordinal numbers where the word form is natural.
Do NOT convert numbers that are part of identifiers, codes, version numbers, or alphanumeric strings (e.g., "x_123", "version 2.0", "room 101B").
If no such convertible numbers are present, or if all numbers are part of identifiers/codes, output the special token: [NO_CHANGE].
If changes are made, output the entire rewritten question.

Here are some examples:

Original Question: "Give me 5 star hotels."
Rewritten Question: Give me five star hotels.

Original Question: "How many schools have more than 250 students?"
Rewritten Question: How many schools have more than two hundred fifty students?

Original Question: "Show me customer with id x_123."
Rewritten Question: [NO_CHANGE]

Original Question: "List the top 3 products."
Rewritten Question: List the top three products.

Original Question: "What is the price of item #A-456?"
Rewritten Question: [NO_CHANGE]

Original Question: "Find orders placed after 2023-01-15."
Rewritten Question: [NO_CHANGE]

Original Question: "Are there any rooms available on the 2nd floor?"
Rewritten Question: Are there any rooms available on the second floor?

Original Question: "The event is on May 1st."
Rewritten Question: The event is on May first.

Original Question: "We need 10 apples and 2 bananas."
Rewritten Question: We need ten apples and two bananas.

Now, process the following:

Original Question: "{question}"
Rewritten Question:"""
        return [{"role": "user", "content": prompt_template}]

    @staticmethod
    def convert_numbers_to_words_in_nlq_with_vllm(input_json_path: str, output_json_path: str,
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
            max_tokens=512 # Allow ample space for the rewritten question
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
        )

        prompts_for_vllm_generation = []
        # Stores info to map LLM outputs back: (record_idx, original_question_text)
        batch_processing_info = []
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record_idx, record in enumerate(original_data):
            original_question_text = record.get('question', "")
            if not original_question_text:
                continue

            # Heuristic: Check if there's at least one digit that's likely standalone or part of a simple number
            # This is a pre-filter to avoid sending questions with no numbers to the LLM.
            # It looks for digits that are not immediately preceded or followed by an alphanumeric character (common in IDs).
            if not re.search(r'(?<![a-zA-Z0-9_])\d+(?![a-zA-Z0-9_])', original_question_text) and \
               not re.search(r'\b\d+\b', original_question_text): # Simpler check for whole numbers
                continue


            messages = BenchmarkVariator._build_number_to_word_prompt_messages(original_question_text)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True 
                )
                prompts_for_vllm_generation.append(prompt_text)
                batch_processing_info.append(
                    (record_idx, original_question_text)
                )
            except Exception:
                pass
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        all_new_records = []
        for i, raw_response in enumerate(llm_raw_responses):
            record_idx, original_nlq_for_this_prompt = batch_processing_info[i]
            
            rewritten_nlq = BenchmarkVariator.parse_qwen3_output(raw_response)

            if not rewritten_nlq or \
               rewritten_nlq == "[NO_CHANGE]" or \
               rewritten_nlq.strip().lower() == original_nlq_for_this_prompt.strip().lower(): # No actual change
                continue 
            
            original_record_instance = original_data[record_idx]
            new_record = copy.deepcopy(original_record_instance)
            new_record['question'] = rewritten_nlq.strip() # Ensure clean output
            
            # For this transformation, the "modifications" list might be harder to populate
            # We'll store the original and new NLQ.
            new_record['changes_information'] = {
                "original_nlq": original_nlq_for_this_prompt,
                "altered_nlq_by_number_conversion": rewritten_nlq.strip(),
                "alteration_type": "number_to_word_conversion"
            }
            all_new_records.append(new_record)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_new_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.\n"
              f"Output file: {output_json_path}\n"
              f"Total new records generated: {len(all_new_records)}")
        

    @staticmethod
    def _format_changes_made(modifications: list) -> str:
        changes_made_string_parts = []
        if not modifications:
            return "No specific value modifications listed."
        for mod in modifications:
            part = (
                f"- Original Value Segment: \"{mod.get('original_value_segment', 'N/A')}\"\n"
                f"  Altered Value Segment: \"{mod.get('altered_value_segment', 'N/A')}\"\n"
                f"  Alteration Type: \"{mod.get('alteration_type', 'N/A')}\"\n"
            )
            changes_made_string_parts.append(part)
        return "\n".join(changes_made_string_parts)

    @staticmethod
    def _build_verification_prompt_messages(original_nlq: str, altered_nlq: str, changes_made_list: list) -> list:
        VERIFICATION_PROMPT_SYSTEM_MESSAGE = """
You are an expert linguistic and database query assistant.
Your task is to determine if an altered Natural Language Question (NLQ) retains the semantic meaning of the original NLQ, specifically concerning values that would be used in a SQL query's WHERE clause.
The core question is: **Could a reasonably sophisticated value retrieval component in a text-to-SQL system still identify the *original, correct* database value(s) based on the *altered* NLQ?**

Consider the following:
- The alteration might be a typo, an abbreviation, a synonym, punctuation change, minor or whatever else that a user that had no knowledge of the underlying data might make.
- If the alteration makes the value unrecognizable, too ambiguous, or fundamentally changes what is being asked for, then the meaning is not preserved for retrieval.
- The goal is to simulate if a user, despite making such a mistake in their query, could still be understood by a system designed to be robust to common errors.

You will be given:
1.  `Original NLQ`: The initial, correct natural language question.
2.  `Altered NLQ`: The modified natural language question.
3.  `Changes Made`: A list describing each modification from an original value segment to an altered value segment, including the type of alteration.

Respond with only 'YES' or 'NO'.
"""

        VERIFICATION_PROMPT_USER_TEMPLATE = """
--- EXAMPLES ---

**Example 1:**
Original NLQ: "What type of bond is there between the atoms TR004_8 and TR004_20?"
Altered NLQ: "What type of bond is there between the atoms TR0048 and TR00420?"
Changes Made:
- Original Value Segment: "TR004_20"
  Altered Value Segment: "TR00420"
  Alteration Type: "punctuation_removal"
- Original Value Segment: "TR004_8"
  Altered Value Segment: "TR0048"
  Alteration Type: "punctuation_removal"
Expected Output: YES
Reasoning: Removing an underscore is a common variation. A good retriever could easily map "TR0048" back to "TR004_8" (or vice-versa if the database stores it without underscores and the original NLQ had it). The core identifiers are intact.

**Example 2:**
Original NLQ: "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
Altered NLQ: "What is the highest eligible free rate for K-12 students in the schools in Alaemda County?"
Changes Made:
- Original Value Segment: "Alameda"
  Altered Value Segment: "Alaemda"
  Alteration Type: "transpose_char_typo"
Expected Output: YES
Reasoning: "Alaemda" is a clear typographical error for "Alameda". A fuzzy matching or spell-checking component in a value retriever should handle this.

**Example 3:**
Original NLQ: "What is the total amount of money spent for food?"
Altered NLQ: "What is the total amount of money spent for aood?"
Changes Made:
- Original Value Segment: "food"
  Altered Value Segment: "aood"
  Alteration Type: "substitute_char_typo"
Expected Output: NO
Reasoning: While "aood" is a single character substitution from "food", it is also a single character substitution from other plausible words like "hood", "wood", or "good". A value retriever might not be able to confidently determine that "food" was the original intended value without further context or a very sophisticated disambiguation mechanism. The alteration introduces significant ambiguity, potentially changing the query's target.


--- TASK ---

Original NLQ: "{original_nlq}"
Altered NLQ: "{altered_nlq}"
Changes Made:
{changes_made_string}

Expected Output:
"""

        changes_made_string = BenchmarkVariator._format_changes_made(changes_made_list)
        
        user_content = VERIFICATION_PROMPT_USER_TEMPLATE.format(
            original_nlq=original_nlq,
            altered_nlq=altered_nlq,
            changes_made_string=changes_made_string
        )
        
        messages = [
            {"role": "system", "content": VERIFICATION_PROMPT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content}
        ]
        return messages

    @staticmethod
    def _parse_qwen_output_for_verification(raw_output_text: str) -> str:
        cleaned_text = re.sub(
            r"<think>.*?</think>",
            "",
            raw_output_text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        if re.search(r"<think>|</think>", cleaned_text, re.IGNORECASE):
            return "" 

        cleaned_text = re.sub(r"^(<\|im_start\|>assistant\s*)+", "", cleaned_text, flags=re.IGNORECASE).strip()
        cleaned_text = re.sub(r"(<\|im_end\|>\s*)+$", "", cleaned_text, flags=re.IGNORECASE).strip()
        
        if not cleaned_text:
            return ""

        match = re.search(r"^\s*(YES|NO)\b", cleaned_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return ""

    @staticmethod
    def verify_nlq_alterations_with_vllm(input_json_path: str, output_json_path: str,
                                         model_name: str = LLM_MODEL_NAME_DEFAULT,
                                         cache_dir: str = LLM_CACHE_DIR_DEFAULT,
                                         tensor_parallel_size: int = LLM_TENSOR_PARALLEL_SIZE_DEFAULT,
                                         quantization: str = LLM_QUANTIZATION_DEFAULT,
                                         gpu_memory_utilization: float = LLM_GPU_MEM_UTIL_DEFAULT
                                        ):
        start_time = time.time()
        
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
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
            max_model_len=8192
        )

        prompts_for_vllm_generation = []
        batch_processing_info = [] 
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        for record in original_data:
            altered_nlq = record.get('question')
            changes_info = record.get('changes_information')

            if not altered_nlq or not changes_info:
                continue

            original_nlq_from_info = changes_info.get('original_nlq')
            modifications = changes_info.get('modifications')

            if not original_nlq_from_info or modifications is None: # Ensure modifications list exists, even if empty
                continue
            
            messages = BenchmarkVariator._build_verification_prompt_messages(original_nlq_from_info, altered_nlq, modifications)
            
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True, 
                enable_thinking=True 
            )
            prompts_for_vllm_generation.append(prompt_text)
            batch_processing_info.append(record)
        
        llm_raw_responses = []
        if prompts_for_vllm_generation:
            vllm_outputs = llm.generate(prompts_for_vllm_generation, sampling_params)
            for output in vllm_outputs:
                llm_raw_responses.append(output.outputs[0].text)
        
        verified_records = []
        for i, raw_response in enumerate(llm_raw_responses):
            original_record_instance = batch_processing_info[i]
            llm_decision = BenchmarkVariator._parse_qwen_output_for_verification(raw_response)
            
            if llm_decision == "YES":
                verified_records.append(original_record_instance)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(verified_records, f, indent=2)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds.")
        print(f"Input records: {len(original_data)}")
        print(f"Prompts sent to LLM: {len(prompts_for_vllm_generation)}")
        print(f"Output file: {output_json_path}")
        print(f"Total records verified as YES: {len(verified_records)}")
        
    
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
    BenchmarkVariator.transpose_char_typo(input_path, output_path)

    output_filename = "synonyms.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.substitute_abbreviations_with_vllm(input_path, output_path)
    
    output_filename = "abbreviations.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.substitute_abbreviations_with_vllm(input_path, output_path)
    
    output_filename = "paraphrases.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.substitute_full_question_paraphrases_with_vllm(input_path, output_path)
    
    output_filename = "antonym_negation.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.rewrite_with_antonym_negation_vllm(input_path, output_path)
    
    output_filename = "word_removal.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.shorten_phrases_by_word_removal_vllm(input_path, output_path)
    
    output_filename = "word_addition.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.enrich_phrases_by_word_addition_vllm(input_path, output_path)
    
    output_filename = "word_reorder.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.reorder_words_in_phrases_vllm_hybrid(input_path, output_path)
    
    output_filename = "singular_plural_toggle.json"
    output_path = f"{output_folder}/{output_filename}"
    BenchmarkVariator.toggle_singular_plural_vllm(input_path, output_path)
    """
    
    input_filename = "assets/all_benchmarks/all_benchmarks_combined.json"
    output_filename = "assets/all_benchmarks/all_benchmarks_combined_verified.json"

    BenchmarkVariator.verify_nlq_alterations_with_vllm(input_filename, output_filename)