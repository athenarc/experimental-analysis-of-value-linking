#!/bin/bash
set -v
set -e

begin_time=$(date "+%Y_%m_%d_%H_%M_%S")
echo "run train text2sql begin time = ${begin_time}"

export TOKENIZERS_PARALLELISM=true
#export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True

DATASET_NAME="bird"
DATASET_BASE_DIR="./value_linking"

######################################################################################
## Evaluation file path
##
######################################################################################
# for BIRD dev
DATASET_MODE="dev"
DATAFILE_PATH="${DATASET_BASE_DIR}/dev.json"
GOLD_PATH="${DATASET_BASE_DIR}/gold.sql"
DATASET_PATH="${DATASET_BASE_DIR}/databases"


## for BIRD test
#DATASET_MODE="test"
#DATAFILE_PATH="${DATASET_BASE_DIR}/bird/test/test.json"
#GOLD_PATH="${DATASET_BASE_DIR}/bird/test/test.sql"
#DATASET_PATH="${DATASET_BASE_DIR}/bird/test/test_databases"


######################################################################################

# Predicted SQL file path
PREDICTED_SQL_PATH="${DATASET_BASE_DIR}/outputs/20250817_183037/sampling_think_sql_merge_pred_major_voting_sqls.sql"

python3 src/cscsql/service/eval/evaluation.py \
--data_mode $DATASET_MODE \
--db_root_path $DATASET_PATH \
--ground_truth_path $GOLD_PATH \
--diff_json_path $DATAFILE_PATH \
--predicted_sql_path ${PREDICTED_SQL_PATH} \
--num_cpus 16 \
--meta_time_out 30.0
