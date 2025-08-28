import json
import random
from pathlib import Path
import argparse

import marisa_trie
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

QUERY_TEMPLATES = [
    "List all {table} where the {column} is <v>{variation}</v>.",
    "Show me the {table} with a {column} of <v>{variation}</v>.",
    "Find {table} for which {column} equals <v>{variation}</v>.",
    "Which {table} have {column} as <v>{variation}</v>?",
    "I need the {table} from <v>{variation}</v>.",
    "What are the {table} in which the {column} is <v>{variation}</v>?",
    "Provide {table} that {column} is <v>{variation}</v>.",
]

def main(data_path, output_path, base_model, epochs, batch_size, learning_rate):
    data_path = Path(data_path)
    output_path = Path(output_path)
    jsonl_files = sorted(list(data_path.glob("*.jsonl")))

    db_pbar = tqdm(jsonl_files, desc="Processing Databases")
    for jsonl_file in db_pbar:
        db_id = jsonl_file.stem
        db_pbar.set_description(f"Processing DB: {db_id}")
        
        db_output_dir = output_path / db_id
        final_model_path = db_output_dir / "final_model"

        if final_model_path.exists():
            print(f"Skipping {db_id}: Final model already exists.")
            continue

        db_output_dir.mkdir(parents=True, exist_ok=True)

        raw_dataset = load_dataset('json', data_files=str(jsonl_file), split='train')

        all_examples = []
        unique_canonical_values = set()

        for record in tqdm(raw_dataset, desc=f"[{db_id}] Filtering and generating examples"):
            canonical_value = record['value']
            
            if len(canonical_value) <= 100:
                table = record['table']
                column = record['column']
                unique_canonical_values.add(canonical_value)
                
                all_variations = set()
                for var_list in record['variations'].values():
                    if var_list:
                        all_variations.update(var_list)

                for variation in all_variations:
                    template = random.choice(QUERY_TEMPLATES)
                    input_text = template.format(table=table, column=column, variation=variation)
                    all_examples.append({
                        "input_text": input_text,
                        "target_text": canonical_value
                    })
        
        if not all_examples:
            print(f"Skipping {db_id}: No values found after filtering. Nothing to train.")
            continue
            
        train_dataset = Dataset.from_list(all_examples)

        trie = marisa_trie.Trie(list(unique_canonical_values))
        
        trie_path = db_output_dir / "constraint.marisa"
        trie.save(trie_path)

        tokenizer = T5TokenizerFast.from_pretrained(base_model)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<v>', '</v>']})

        model = T5ForConditionalGeneration.from_pretrained(base_model)
        model.resize_token_embeddings(len(tokenizer))

        def preprocess_function(examples):
            model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
            labels = tokenizer(text_target=examples["target_text"], max_length=128, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

        training_args = TrainingArguments(
            output_dir=str(db_output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            save_strategy="epoch",
            logging_strategy="epoch",
            group_by_length=True,
            remove_unused_columns=False,
            report_to="none",
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"\nFinished training for {db_id}. Model saved to {final_model_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a canonicalizer model for each database.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the folder containing the augmented .jsonl files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where final models and tries will be saved.")
    parser.add_argument("--base_model", type=str, default="google/flan-t5-base", help="Name of the base model from Hugging Face Hub.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size PER DEVICE (micro-batch).")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    
    args = parser.parse_args()
    main(args.data_path, args.output_path, args.base_model, args.epochs, args.batch_size, args.learning_rate)