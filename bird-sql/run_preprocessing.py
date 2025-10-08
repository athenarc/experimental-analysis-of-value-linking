# File: run_preprocessing.py

import sys
import os
from pathlib import Path

# --- Start of Fix ---
# This adds the project's root directory to Python's path.
# It allows the script to find the 'src' module.
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
# --- End of Fix ---

from src.prep_data import preprocess_data

# Set the path to your benchmark directory
benchmark_dir = Path("my_benchmark_precision_05")

print(f"Starting preprocessing for directory: {benchmark_dir}")
preprocess_data(benchmark_dir)
print("Preprocessing complete!")