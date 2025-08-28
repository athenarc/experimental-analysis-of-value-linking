import json
import re
from pathlib import Path
import argparse
import marisa_trie
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from flashtext import KeywordProcessor

class PrefixAllowedTokens:
    def __init__(self, tokenizer, trie):
        self.tokenizer = tokenizer
        self.trie = trie

    def __call__(self, batch_id, sent):
        sent_str = self.tokenizer.decode(sent, skip_special_tokens=True)
        
        # marisa-trie.keys() is very fast for prefix search
        next_values = self.trie.keys(sent_str)
        
        allowed_tokens = set()
        for val in next_values:
            # Get the part of the value that comes after the current prefix
            next_part = val[len(sent_str):]
            if next_part:
                # Tokenize just the next part to find potential next tokens
                next_tokens = self.tokenizer(next_part, add_special_tokens=False)['input_ids']
                if next_tokens:
                    allowed_tokens.add(next_tokens[0])
        
        # Always allow the end-of-sequence token
        allowed_tokens.add(self.tokenizer.eos_token_id)
        
        return list(allowed_tokens)

def main(model_root_path, data_gen_path, eval_file_path, device):
    model_root_path = Path(model_root_path)
    data_gen_path = Path(data_gen_path)
    
    print("Phase 1: Building keyword processor from all generated data...")
    keyword_processor = KeywordProcessor(case_sensitive=False)
    all_jsonl_files = list(data_gen_path.glob("*.jsonl"))
    for jsonl_file in tqdm(all_jsonl_files, desc="Loading variations"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                canonical_value = record['value']
                keyword_processor.add_keyword(canonical_value, canonical_value)
                for var_list in record['variations'].values():
                    if var_list:
                        for variation in var_list:
                            keyword_processor.add_keyword(variation, canonical_value)

    print(f"Keyword processor built with {len(keyword_processor)} total keywords.")

    print("\nPhase 2: Running evaluation...")
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    model_cache = {}
    correct_predictions = 0
    total_predictions = 0

    for record in tqdm(eval_data, desc="Evaluating Questions"):
        db_id = record['db_id']
        question = record['new_question_correct_value']

        # Filter ground truth values according to rules
        ground_truth_values = set()
        for v in record['values']:
            value_str = str(v['value'])
            if re.search('[a-zA-Z]', value_str):
                ground_truth_values.add(value_str.lower())

        # Load model, tokenizer, and trie from cache or disk
        if db_id not in model_cache:
            print(f"Loading model for db_id: {db_id}")
            model_path = model_root_path / db_id / "final_model"
            trie_path = model_root_path / db_id / "constraint.marisa"

            if not model_path.exists() or not trie_path.exists():
                print(f"Warning: Model or trie not found for {db_id}. Skipping.")
                continue

            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
            tokenizer = T5TokenizerFast.from_pretrained(model_path)
            trie = marisa_trie.Trie()
            trie.load(trie_path)
            model_cache[db_id] = (model, tokenizer, trie)
        
        model, tokenizer, trie = model_cache[db_id]

        # Stage 1: Candidate Span Identification
        found_keywords = keyword_processor.extract_keywords(question, span_info=True)
        
        prompts = []
        for keyword, start, end in found_keywords:
            prompt = f"{question[:start]}<v>{question[start:end]}</v>{question[end:]}"
            prompts.append(prompt)
        
        predicted_values = set()
        if prompts:
            prefix_fn = PrefixAllowedTokens(tokenizer, trie)
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                prefix_allowed_tokens_fn=prefix_fn,
                num_beams=5
            )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for output in decoded_outputs:
                if output:
                    predicted_values.add(output.lower())

        # Stage 2: Compare and score
        total_predictions += 1
        if predicted_values == ground_truth_values:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print("\n--- Evaluation Complete ---")
    print(f"Total Questions Evaluated: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned canonicalizer models.")
    parser.add_argument("--model_root_path", type=str, required=True, help="Path to the root folder where final models are saved (e.g., data/fine_tuned_flan_new).")
    parser.add_argument("--data_gen_path", type=str, required=True, help="Path to the folder containing the original augmented .jsonl files (e.g., data/value_discrepancies).")
    parser.add_argument("--eval_file_path", type=str, required=True, help="Path to the input JSON evaluation file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()
    main(args.model_root_path, args.data_gen_path, args.eval_file_path, args.device)