import json
import re
from pathlib import Path
import argparse
import pickle
from tqdm.auto import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from keybert import KeyBERT

class PrefixAllowedTokens:
    def __init__(self, tokenizer, token_trie):
        self.tokenizer = tokenizer
        self.token_trie = token_trie

    def __call__(self, batch_id, sent_token_ids):
        sent_token_ids = sent_token_ids.tolist()
        
        if sent_token_ids and sent_token_ids[0] == self.tokenizer.pad_token_id:
            sent_token_ids = sent_token_ids[1:]

        node = self.token_trie
        for token_id in sent_token_ids:
            if token_id in node:
                node = node[token_id]
            else:
                return [self.tokenizer.eos_token_id]
        
        allowed_token_ids = list(node.keys())
        allowed_token_ids.append(self.tokenizer.eos_token_id)
        
        return allowed_token_ids

def main(model_root_path, eval_file_path, output_file_path, device):
    model_root_path = Path(model_root_path)
    
    print("Initializing KeyBERT for semantic candidate extraction...")
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')

    print("\nPhase 2: Running evaluation with detailed debug logging...")
    with open(eval_file_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    model_cache = {}
    all_results = []
    correct_predictions = 0
    total_predictions = 0

    for record in tqdm(eval_data, desc="Evaluating Questions"):
        db_id = record['db_id']
        question = record['new_question_correct_value']

        ground_truth_values = set()
        for v in record['values']:
            value_str = str(v['value'])
            if re.search('[a-zA-Z]', value_str):
                ground_truth_values.add(value_str.lower())

        if db_id not in model_cache:
            print(f"Loading model for db_id: {db_id}")
            model_path = model_root_path / db_id / "final_model"
            trie_path = model_root_path / db_id / "token_trie.pkl"
            if not model_path.exists() or not trie_path.exists():
                continue
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
            tokenizer = T5TokenizerFast.from_pretrained(model_path)
            with open(trie_path, 'rb') as f:
                trie = pickle.load(f)
            model_cache[db_id] = (model, tokenizer, trie)
        
        model, tokenizer, trie = model_cache[db_id]
        
        keybert_candidates = kw_model.extract_keywords(
            question, 
            keyphrase_ngram_range=(1, 5), 
            stop_words=None, 
            top_n=10
        )
        
        prompts_to_process = []
        processed_spans = set()
        for candidate_phrase, _ in keybert_candidates:
            for match in re.finditer(re.escape(candidate_phrase), question, re.IGNORECASE):
                start, end = match.span()
                if (start, end) not in processed_spans:
                    original_span = question[start:end]
                    prompt = f"{question[:start]}<v>{original_span}</v>{question[end:]}"
                    prompts_to_process.append({
                        "original_span": original_span,
                        "prompt": prompt
                    })
                    processed_spans.add((start, end))

        predicted_values = set()
        debug_records = []
        
        if prompts_to_process:
            prompts = [p["prompt"] for p in prompts_to_process]
            prefix_fn = PrefixAllowedTokens(tokenizer, trie)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                prefix_allowed_tokens_fn=prefix_fn,
                num_beams=5
            )
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for i, decoded_output in enumerate(decoded_outputs):
                original_span = prompts_to_process[i]["original_span"]
                prompt = prompts_to_process[i]["prompt"]
                
                debug_records.append({
                    "keybert_span": original_span,
                    "prompt_generated": prompt,
                    "model_prediction": decoded_output
                })

                if decoded_output:
                    predicted_values.add(decoded_output.lower())

        is_correct = (predicted_values == ground_truth_values)
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

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    print("\n--- Saving prediction results ---")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {output_file_path}")

    print("\n--- Evaluation Complete ---")
    print(f"Total Questions Evaluated: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Exact Match Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned canonicalizer models using KeyBERT for span detection.")
    parser.add_argument("--model_root_path", type=str, required=True, help="Path to the root folder where final models are saved.")
    parser.add_argument("--eval_file_path", type=str, required=True, help="Path to the input JSON evaluation file.")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to the output JSON file to save predictions.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    
    args = parser.parse_args()
    main(args.model_root_path, args.eval_file_path, args.output_file_path, args.device)