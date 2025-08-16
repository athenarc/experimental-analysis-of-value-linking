import os
import shutil

# --- Configuration ---
# Set this to False to actually perform the file operations.
# Set to True to just print what the script would do without changing anything.
DRY_RUN = False

# The directory containing your files. '.' means the current directory
# where the script is running.
TARGET_DIRECTORY = '.' 
# --- End of Configuration ---


def process_files():
    """
    Finds 'copy' files, replaces the content of the original with the copy's content,
    and then removes the copy file.
    """
    print("--- Starting File Processing ---")
    if DRY_RUN:
        print("!!! DRY RUN is ACTIVE. No files will be modified or deleted. !!!")
    
    # Get a list of all files in the directory
    try:
        all_files = os.listdir(TARGET_DIRECTORY)
    except FileNotFoundError:
        print(f"Error: The directory '{TARGET_DIRECTORY}' was not found.")
        return

    processed_count = 0
    # Iterate over every file to find the copies
    for filename in all_files:
        # 1. Find files that are copies and are in the specified number range
        if filename.endswith(' copy.json'):
            try:
                # Check if the file name starts with a number in the desired range
                prefix_str = filename.split('_')[0]
                prefix_num = int(prefix_str)
                if not (1518 <= prefix_num <= 1782):
                    continue # Skip if not in range
            except (ValueError, IndexError):
                # The filename doesn't start with "number_" so we ignore it
                continue

            # 2. Construct the original filename from the copy's name
            # This assumes '...namecopy.json' -> '...name.json'
            original_name = filename.replace(' copy.json', '.json')
            
            # Create full paths for the files
            copy_filepath = os.path.join(TARGET_DIRECTORY, filename)
            original_filepath = os.path.join(TARGET_DIRECTORY, original_name)

            # 3. Check if the original file actually exists
            if os.path.exists(original_filepath):
                print(f"\nFound match:")
                print(f"  - Copy:     {filename}")
                print(f"  - Original: {original_name}")

                if DRY_RUN:
                    print("  - [DRY RUN] Would replace content of original with copy.")
                    print("  - [DRY RUN] Would delete the copy file.")
                else:
                    try:
                        # 4. Replace content by copying the 'copy' over the 'original'
                        shutil.copy2(copy_filepath, original_filepath)
                        print(f"  - SUCCESS: Replaced content of '{original_name}'.")
                        
                        # 5. Remove the copy file
                        os.remove(copy_filepath)
                        print(f"  - SUCCESS: Removed '{filename}'.")
                        
                    except Exception as e:
                        print(f"  - ERROR: Could not process '{filename}'. Reason: {e}")
                
                processed_count += 1
            else:
                print(f"\nWarning: Found copy '{filename}' but its original '{original_name}' does not exist. Skipping.")

    print("\n--- Processing Complete ---")
    if processed_count == 0:
        print("No matching files were found to process.")
    else:
        action = "would be" if DRY_RUN else "were"
        print(f"A total of {processed_count} file pairs {action} processed.")


if __name__ == "__main__":
    process_files()