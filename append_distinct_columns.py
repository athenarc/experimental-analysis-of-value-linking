import json
from pathlib import Path

def add_metadata_records_to_json(directory: str):
    """
    Processes JSON files to add new records for each unique table-column pair.

    The new records are added to the beginning of the existing list of records,
    resulting in a single, flat list.

    Args:
        directory (str): The path to the directory containing the JSON files.
    """
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"Error: Directory not found at '{directory}'")
        return

    json_files = list(dir_path.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in '{directory}'")
        return

    print(f"Found {len(json_files)} JSON files to process...")

    for file_path in json_files:
        print(f"Processing '{file_path.name}'...")
        try:
            # Read the existing JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content:
                    print(f"  - Skipping empty file: {file_path.name}")
                    continue
                original_data = json.loads(content)

            if not isinstance(original_data, list):
                print(f"  - Skipping non-list JSON file: {file_path.name}")
                continue

            # Extract unique table-column pairs
            unique_pairs = set()
            for record in original_data:
                # This check prevents trying to add metadata from already-added metadata
                if isinstance(record, dict) and 'table' in record and 'column' in record and 'database_id' in record:
                    table_name = record.get('table')
                    column_name = record.get('column')
                    unique_pairs.add((table_name, column_name))

            # If no valid pairs were found, there's nothing to do
            if not unique_pairs:
                 print(f"  - No valid data records found to generate metadata in '{file_path.name}'. Skipping.")
                 continue

            # Format the pairs into a list of new records
            sorted_pairs = sorted(list(unique_pairs))
            metadata_records = [
                {"table": table, "column": column} for table, column in sorted_pairs
            ]

            # Create the new flat list by concatenating the two lists
            # [new_record_1, new_record_2] + [original_record_1, original_record_2]
            new_data = metadata_records + original_data

            # Write the updated flat list back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2)
            
            print(f"  - Successfully added {len(metadata_records)} metadata records to '{file_path.name}'")

        except json.JSONDecodeError:
            print(f"  - Error: Could not decode JSON from '{file_path.name}'. Skipping.")
        except Exception as e:
            print(f"  - An unexpected error occurred with '{file_path.name}': {e}")

    print("\nProcessing complete.")


# --- Main execution block ---
if __name__ == "__main__":
    TARGET_DIRECTORY = "assets/data_exploration"
    add_metadata_records_to_json(TARGET_DIRECTORY)