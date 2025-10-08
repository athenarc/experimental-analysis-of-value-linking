#!/bin/bash

# This script finds all preprocessed database directories in ./data/dataset/
# and launches a separate fine-tuning job for each one using train_valuelinking.sh.
# It will run them SEQUENTIALLY, waiting for one to finish before starting the next.
# It will skip any database that already has a completed training checkpoint (Epoch25*.pth).
# Usage: ./scripts/train_all_dbs.sh [gpu_ids]
# Example: ./scripts/train_all_dbs.sh 0,1

# --- Configuration ---
if [ -z "$1" ]; then
  echo "Error: No GPU IDs provided."
  echo "Usage: ./scripts/train_all_dbs.sh [gpu_ids]"
  exit 1
fi

GPU_IDS=$1
DATASET_DIR="./data/dataset"
OUTPUT_DIR_BASE="snap/ValueLinkingIdentity" # Base directory for checking results

# Find all subdirectories in the dataset directory. Each is assumed to be a database.
DATABASES=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

if [ -z "$DATABASES" ]; then
  echo "Error: No database subdirectories found in '$DATASET_DIR'."
  echo "Please run the preprocessing script first."
  exit 1
fi

echo "--- Found Databases to Train Sequentially ---"
echo "$DATABASES"
echo "--------------------------------"
echo "GPU IDs to be used: $GPU_IDS"
echo ""

# --- Loop and Launch Training ---
for db_name in $DATABASES; do
  
  # --- Check for existing completed training ---
  checkpoint_found=false
  # Use a glob to find any potential output directories for this db_name
  # This handles the variable timestamp and hyperparameters in the folder name
  potential_dirs=(${OUTPUT_DIR_BASE}/*-${db_name}-*/)
  
  # The glob returns the pattern itself if no match is found, so we check if the first result is a directory
  if [ -d "${potential_dirs[0]}" ]; then
    for dir in "${potential_dirs[@]}"; do
      # Check if a file matching the pattern Epoch25*.pth exists in the directory
      if ls "${dir}/Epoch25"*.pth 1> /dev/null 2>&1; then
        echo ">>> SKIPPING: Found completed training for '$db_name' in '$dir' <<<"
        checkpoint_found=true
        break # Exit the inner loop once a checkpoint is found
      fi
    done
  fi
  
  # If a checkpoint was found, continue to the next database
  if [ "$checkpoint_found" = true ]; then
    continue
  fi
  # --- End of check ---

  echo ">>> Launching training for database: $db_name <<<"

  # Check if the training script exists and is executable
  if [ ! -x "./scripts/train_valuelinking.sh" ]; then
    echo "Error: ./scripts/train_valuelinking.sh is not found or not executable."
    echo "Please ensure the script exists and run 'chmod +x ./scripts/train_valuelinking.sh'"
    exit 1
  fi

  # ================================================================= #
  # === CORRECTED LINE: REMOVED the inner `nohup` and ensured NO `&` === #
  # This makes the script wait for the training to finish before continuing the loop.
  ./scripts/train_valuelinking.sh "$db_name" "$GPU_IDS"
  # ================================================================= #

  echo "Training for '$db_name' has COMPLETED."
  echo "--------------------------------------------------------"
  # The sleep is no longer necessary as we are running sequentially
  # sleep 5 
done

echo "All training jobs have been launched and completed."
echo "You can find their individual log files in their respective output directories inside snap/ValueLinkingIdentity/."