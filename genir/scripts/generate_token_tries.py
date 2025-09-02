import json
import pickle
from pathlib import Path
import argparse

from tqdm.auto import tqdm
from transformers import T5TokenizerFast

def main(data_gen_path, model_output_path, base_model):
    data_gen_path = Path(data_gen_path)
    model_output_path = Path(model_output_path)
    
    print(f"Loading tokenizer for base model: {base_model}")
    tokenizer = T5TokenizerFast.from_pretrained(base_model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<v>', '</v>']})

    jsonl_files = sorted(list(data_gen_path.glob("*.jsonl")))
    
    for jsonl_file in tqdm(jsonl_files, desc="Processing Databases"):
        db_id = jsonl_file.stem
        
        # Determine the output directory for this database's trie
        db_output_dir = model_output_path / db_id
        db_output_dir.mkdir(parents=True, exist_ok=True)
        
        trie_path = db_output_dir / "token_trie.pkl"
        
        # --- Collect all unique, filtered canonical values ---
        unique_canonical_values = set()
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                value = record['value']
                # Apply the same filtering logic as the training script for consistency
                if len(value) <= 100:
                    unique_canonical_values.add(value)
        
        if not unique_canonical_values:
            print(f"Warning: No values found for {db_id} after filtering. Skipping trie creation.")
            continue
            
        # --- Build the token-level trie ---
        token_trie = {}
        for value in tqdm(unique_canonical_values, desc=f"[{db_id}] Building token trie", leave=False):
            # Tokenize without special tokens, as these are target sequences
            token_ids = tokenizer(value, add_special_tokens=False)['input_ids']
            node = token_trie
            for token_id in token_ids:
                node = node.setdefault(token_id, {})
        
        # --- Save the trie to a pickle file ---
        with open(trie_path, 'wb') as f:
            pickle.dump(token_trie, f)
            
        print(f"Successfully generated and saved token trie for {db_id} to {trie_path}")

    print("\nAll token tries have been generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate token-level tries from augmented .jsonl files.")
    parser.add_argument("--data_gen_path", type=str, required=True, help="Path to the folder containing the augmented .jsonl files (e.g., data/value_discrepancies).")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to the root folder where final models are saved and tries will be stored (e.g., data/fine_tuned_flan_new).")
    parser.add_argument("--base_model", type=str, default="google/flan-t5-base", help="Name of the base model used for training, to ensure consistent tokenization.")
    
    args = parser.parse_args()
    main(args.data_gen_path, args.model_output_path, args.base_model)
    