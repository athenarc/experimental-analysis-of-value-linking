import os
import json
import argparse
from tqdm import tqdm

def process_db_file(source_file_path, output_base_dir):
    """
    Processes a single database .jsonl file to create the required
    `values.jsonl` and `training_pairs.jsonl` files.

    Args:
        source_file_path (str): The full path to the source .jsonl file.
        output_base_dir (str): The base directory where processed data will be saved.
    """
    # 1. Determine database name and create output directory
    db_name = os.path.splitext(os.path.basename(source_file_path))[0]
    output_dir = os.path.join(output_base_dir, db_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing database: '{db_name}'")
    print(f"Output will be saved to: {output_dir}")

    values_output_path = os.path.join(output_dir, 'values.jsonl')
    pairs_output_path = os.path.join(output_dir, 'training_pairs.jsonl')

    # 2. Initialize data structures
    values_dict = {}
    training_pairs_list = []

    # 3. Read and process the input file
    with open(source_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"  Parsing '{db_name}.jsonl'"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON line in {source_file_path}: {line.strip()}")
            continue

        canonical_value = data['value']
        table = data['table']
        column = data['column']

        # Populate values_dict for the prefix tree
        if canonical_value not in values_dict:
            values_dict[canonical_value] = {"table": table, "column": column}

        # Populate training_pairs_list for SFT
        # a. Add the identity mapping
        training_pairs_list.append({
            "variation": canonical_value,
            "canonical_value": canonical_value
        })

        # b. Add all other variations
        variations_dict = data.get('variations', {})
        if variations_dict:
            for variation_type, variation_list in variations_dict.items():
                for variation in variation_list:
                    training_pairs_list.append({
                        "variation": variation,
                        "canonical_value": canonical_value
                    })

    # 4. Write the output files
    # a. Write values.jsonl (a single JSON object)
    with open(values_output_path, 'w', encoding='utf-8') as f:
        json.dump(values_dict, f, indent=2, ensure_ascii=False)

    # b. Write training_pairs.jsonl (one JSON object per line)
    with open(pairs_output_path, 'w', encoding='utf-8') as f:
        for pair in training_pairs_list:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # 5. Provide feedback
    print(f"  Successfully created '{values_output_path}'")
    print(f"    - Total unique canonical values: {len(values_dict)}")
    print(f"  Successfully created '{pairs_output_path}'")
    print(f"    - Total training pairs generated: {len(training_pairs_list)}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess value discrepancy data for fine-tuning a value-linking model.")
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help="Directory containing the source .jsonl files (e.g., ./data/value_discrepancies)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/dataset/',
        help="Base directory to save the processed datasets (e.g., ./RecLM-cgen/data/dataset/)."
    )
    args = parser.parse_args()

    # Ensure the script is run from the correct root directory
    if not os.path.isdir(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' not found.")
        print("Please ensure you are running this script from the root of the 'RecLM-cgen' project directory.")
        return

    source_files = [f for f in os.listdir(args.source_dir) if f.endswith('.jsonl')]

    if not source_files:
        print(f"Error: No .jsonl files found in '{args.source_dir}'.")
        return

    print(f"Found {len(source_files)} database file(s) to process: {', '.join(source_files)}")

    for file_name in source_files:
        full_path = os.path.join(args.source_dir, file_name)
        process_db_file(full_path, args.output_dir)

    print("\nPreprocessing complete for all databases!")


if __name__ == '__main__':
    main()