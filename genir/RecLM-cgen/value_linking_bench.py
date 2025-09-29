import json
import re
import glob
import argparse
import math
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from typing import Callable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from train_utils.processor import Trie_link, FastPrefixConstrainedLogitsProcessor


def find_best_checkpoint(model_root_path: Path, db_id: str) -> Path | None:
    """Finds the latest epoch checkpoint file for a given db_id."""
    search_pattern = str(model_root_path / f"*-{db_id}-*")
    potential_dirs = glob.glob(search_pattern)
    if not potential_dirs:
        print(f"Warning: No model directory found for db_id '{db_id}' using pattern '{search_pattern}'")
        return None

    model_dir = Path(potential_dirs[0])
    checkpoints = list(model_dir.glob("Epoch*.pth"))
    if not checkpoints:
        print(f"Warning: No checkpoints found in '{model_dir}' for db_id '{db_id}'")
        return None

    latest_epoch = -1
    best_checkpoint = None
    for ckpt in checkpoints:
        match = re.search(r"Epoch(\d+)", ckpt.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                best_checkpoint = ckpt
    
    return best_checkpoint

def main(model_root_path, data_root_path, eval_file_path, extracted_values_path, output_file_path, device):
    model_root_path = Path(model_root_path)
    data_root_path = Path(data_root_path)
    
    print(f"Loading pre-extracted value references from '{extracted_values_path}'...")
    with open(extracted_values_path, 'r', encoding='utf-8') as f:
        extracted_data = json.load(f)
    
    extracted_values_map = {item['query']: item['extracted_values'] for item in extracted_data}
    print(f"Loaded {len(extracted_values_map)} query-to-value mappings.")


    print("\nLoading and preparing evaluation data...")
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    eval_data.sort(key=itemgetter('db_id'))
    grouped_data = {k: list(v) for k, v in groupby(eval_data, key=itemgetter('db_id'))}

    all_results = []
    correct_predictions = 0
    total_predictions = 0
    
    NUM_BEAMS = 10

    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    system_prompt = "You are a precise database value linking tool. Your task is to take a user's mention and return the single, correct canonical database value that it maps to, wrapped in <SOI> and <EOI> tokens."

    for db_id, records in tqdm(grouped_data.items(), desc="Processing Databases"):
        if db_id == 'card_games':
            print("Skipping db_id: 'card_games'")
            continue

        best_checkpoint_path = find_best_checkpoint(model_root_path, db_id)
        if not best_checkpoint_path:
            continue
            
        print(f"\nLoading model for db_id: {db_id} from checkpoint: {best_checkpoint_path}")
        
        base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir="/data/hdd1/vllm_models/"
        )
        tokenizer.add_special_tokens({'additional_special_tokens': ['<SOI>', '<EOI>']})
        tokenizer.soi_token = "<SOI>"
        tokenizer.eoi_token = "<EOI>"
        tokenizer.soi_token_id = tokenizer.convert_tokens_to_ids("<SOI>")
        tokenizer.eoi_token_id = tokenizer.convert_tokens_to_ids("<EOI>")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir="/data/hdd1/vllm_models/"
        )
        base_model.resize_token_embeddings(len(tokenizer))

        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, peft_config)
        state_dict = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.merge_and_unload()
        model.eval()

        values_path = data_root_path / db_id / "values.jsonl"
        processor = None
        if not values_path.exists():
            print(f"Warning: values.jsonl not found for {db_id}. Skipping constrained generation.")
        else:
            with open(values_path, 'r', encoding='utf-8') as f:
                canonical_values_dict = json.load(f)
            
            canonical_values = list(canonical_values_dict.keys())
            item_ids = tokenizer(canonical_values, add_special_tokens=False).input_ids
            
            item_prefix_tree = Trie_link(item_ids, tokenizer)
            processor = FastPrefixConstrainedLogitsProcessor(item_prefix_tree.constrain_search_list, num_beams=NUM_BEAMS)

        for record in tqdm(records, desc=f"Evaluating '{db_id}' Questions", leave=False):
            question = record['question']
            #new_question_correct_value
            ground_truth_values = {str(v['value']).lower() for v in record['values'] if re.search('[a-zA-Z]', str(v['value']))}

            candidate_phrases = extracted_values_map.get(question, [])
            candidate_phrases = [p for p in candidate_phrases if not p.startswith("Error:")]

            predicted_values = set()
            debug_records = []
            
            if candidate_phrases:
                full_prompts = []
                for phrase in candidate_phrases:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Link the following mention to a correct database value: '{phrase}'"}
                    ]
                    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    full_prompts.append(prompt_string)
                
                tokenizer.padding_side = 'left'
                inputs = tokenizer(
                    full_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)
                
                batch_size = inputs['input_ids'].shape[0]
                soi_tokens = torch.full((batch_size, 1), tokenizer.soi_token_id, dtype=torch.long).to(device)
                inputs['input_ids'] = torch.cat([inputs['input_ids'], soi_tokens], dim=1)
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(soi_tokens)], dim=1)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        logits_processor=[processor] if processor else None,
                        num_beams=NUM_BEAMS,
                        num_return_sequences=NUM_BEAMS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.eoi_token_id]
                    )
                
                input_lengths = inputs['input_ids'].shape[1]
                decoded_outputs = tokenizer.batch_decode(outputs[:, input_lengths:], skip_special_tokens=False)
                
                for i in range(len(candidate_phrases)):
                    start_index = i * NUM_BEAMS
                    end_index = (i + 1) * NUM_BEAMS
                    prompt_beams = decoded_outputs[start_index:end_index]

                    prompt_predictions = set()
                    for beam_output in prompt_beams:
                        match = re.search(r"(.*?)<EOI>", beam_output)
                        raw_output_for_log = f"<SOI>{beam_output}"
                        prediction = match.group(1).strip() if match else ""
                        
                        if prediction:
                            prompt_predictions.add(prediction)

                    for pred in prompt_predictions:
                        predicted_values.add(pred.lower())

                    debug_records.append({
                        "candidate_phrase": candidate_phrases[i],
                        "raw_model_output_top_beam": f"<SOI>{prompt_beams[0]}",
                        "parsed_predictions_from_beams": sorted(list(prompt_predictions))
                    })

            is_correct = bool(ground_truth_values.intersection(predicted_values))
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            all_results.append({
                "question": question,
                "db_id": db_id,
                "ground_truth": sorted(list(ground_truth_values)),
                "predicted": sorted(list(predicted_values)),
                "is_correct": is_correct,
                "debug_info": debug_records
            })
        
        del model, base_model, tokenizer
        torch.cuda.empty_cache()

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nPredictions saved to {output_file_path}")

    print("\n--- Evaluation Complete ---")
    print(f"Total Questions Evaluated: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Question-Level Recall (Hit Rate): {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark fine-tuned Llama-3 Value Linking models with constrained generation.")
    parser.add_argument("--model_root_path", type=str, required=True, help="Path to the root folder where training outputs are saved (e.g., 'snap/ValueLinking/').")
    parser.add_argument("--data_root_path", type=str, required=True, help="Path to the root folder of preprocessed datasets (e.g., 'data/dataset/').")
    parser.add_argument("--eval_file_path", type=str, required=True, help="Path to the input JSON evaluation file (e.g., 'dev.json').")
    parser.add_argument("--extracted_values_path", type=str, required=True, help="Path to the JSON file with pre-extracted value references from the previous script.")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output JSON file to save predictions.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()
    main(args.model_root_path, args.data_root_path, args.eval_file_path, args.extracted_values_path, args.output_file_path, args.device)