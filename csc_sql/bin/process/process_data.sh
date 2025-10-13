#!/bin/bash
set -v
set -e

DATASET_NAME="bird"
DATASET_BASE_DIR="./value_linking_prec_001"

# for BIRD dev
DATASET_MODE="dev"
DATAFILE_PATH="${DATASET_BASE_DIR}/dev.json"
DATASET_PATH="${DATASET_BASE_DIR}/databases"
TABLES_PATH="${DATASET_BASE_DIR}/tables.json"
SAVE_INDEX_PATH="./value_linking/db_contents_index"
PROMPT_OUTPUT_PATH="${DATASET_BASE_DIR}/dev_prompts.json"


python -m cscsql.service.process.process_dataset \
--input_data_file $DATAFILE_PATH \
--output_data_file $PROMPT_OUTPUT_PATH \
--db_path $DATASET_PATH \
--tables $TABLES_PATH \
--source $DATASET_NAME \
--mode $DATASET_MODE \
--value_limit_num 2 \
--db_content_index_path $SAVE_INDEX_PATH
