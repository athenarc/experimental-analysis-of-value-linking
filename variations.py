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
LLM_GPU_MEM_UTIL_DEFAULT = 0.85


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