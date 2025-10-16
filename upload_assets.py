# upload_script.py
from huggingface_hub import HfApi, create_repo
import os

# --- Configuration ---
# Your Hugging Face username and the desired repository name
REPO_ID = "ValueLinking/value_linking_assets" 

# The exact list of folder paths you want to upload
# These paths are relative to where this script is run.
PATHS_TO_UPLOAD = [
    "assets/retrievers",
    "bird-sql/few_shots",
    "bird-sql/models",
    "bird-sql/my_benchmark",
    "csc_sql/value_linking",
    "OmniSQL/train_and_evaluate/data/value_linking",
    "OpenSearch-SQL/value_linking",
    "CHESS/data/value_linking"
]

# --- Main Script ---
api = HfApi()

# 1. Create the repository on the Hub (will not fail if it already exists)
create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
print(f"Repository '{REPO_ID}' is ready.")

# 2. Upload each specified folder
for local_path in PATHS_TO_UPLOAD:
    if not os.path.exists(local_path):
        print(f"⚠️  Warning: Local path '{local_path}' not found. Skipping.")
        continue

    # The `path_in_repo` argument ensures the structure is preserved.
    # We set it to be the same as the local path.
    destination_path = local_path
    
    print(f"Uploading '{local_path}' to '{destination_path}' in the repo...")
    api.upload_folder(
        folder_path=local_path,
        path_in_repo=destination_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"✅ Successfully uploaded '{local_path}'.")

print("\n🚀 All specified folders have been uploaded!")