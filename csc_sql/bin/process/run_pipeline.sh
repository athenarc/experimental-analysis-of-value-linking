#!/bin/bash
set -v
set -e

begin_time=$(date "+%Y_%m_%d_%H_%M_%S")
echo "run train text2sql begin time = ${begin_time}"

export TOKENIZERS_PARALLELISM=true
#export VLLM_USE_V1=0
export VLLM_USE_MODELSCOPE=True

DATASET_NAME="bird"
DATASET_BASE_DIR="./value_linking_prec_001"
DATASET_MODE="dev"
DATAFILE_PATH="${DATASET_BASE_DIR}/dev_prompts.json"
# GOLD文件路径，当无GOLD文件时，设置为none
GOLD_PATH="none"
DATASET_PATH="${DATASET_BASE_DIR}/dev_databases"
TABLES_PATH="${DATASET_BASE_DIR}/tables.json"
PROMPT_OUTPUT_PATH="${DATASET_BASE_DIR}/dev_prompts.json"
ORIGINAL_DATA_JSON="${DATASET_BASE_DIR}/dev.json"


## Output file path
OUTPUT_DIR="${DATASET_BASE_DIR}/outputs"

## cuda device
CUDA_VISIBLE_DEVICES="0,1"
TENSOR_PARALLEL_SIZE=2
#CUDA_VISIBLE_DEVICES="2,3"
#TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.93

#EVAL_STEP="pipeline"
EVAL_STEP="sql_pipeline"
#EVAL_STEP="table_link"
#EVAL_STEP="sql_generate"
#EVAL_STEP="sql_merge"

LINK_TABLES='none'
SELECTION_VOTE='none'

EVAL_MODE="major_voting"
N_TABLE_LINK=8
N_SQL_GENERATE=16
N_SQL_MERGE=8

MODEL_TABLE_LINK="none"
# 7B size SQL merge model path
MODEL_SQL_MERGE="/data/hdd1/vllm_models/cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct"
#MODEL_SQL_MERGE="cycloneboy/CscSQL-Merge-Qwen2.5-Coder-3B-Instruct"

# 7B SQL generation model
MODEL_SQL_GENERATE="/data/hdd1/vllm_models/XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
#MODEL_SQL_GENERATE="cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct"
# The number of SQL generated. For a 7B model, the default value is 64; for a 32B model, the default value is 64

# 32B SQL generation model
#MODEL_SQL_GENERATE="XGenerationLab/XiYanSQL-QwenCoder-32B-2412"

LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

train_log="${LOG_DIR}/run_train_${begin_time}.log"
echo "train log file ${train_log}"

nohup   python -m cscsql.model.pipeline_infer \
--model_table_link  $MODEL_TABLE_LINK \
--model_sql_generate  $MODEL_SQL_GENERATE \
--model_sql_merge $MODEL_SQL_MERGE \
--source $DATASET_NAME \
--input_file  $PROMPT_OUTPUT_PATH \
--gold_file  $GOLD_PATH \
--db_path $DATASET_PATH \
--output_dir  $OUTPUT_DIR \
--visible_devices $CUDA_VISIBLE_DEVICES \
--tensor_parallel_size  $TENSOR_PARALLEL_SIZE \
--gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
--eval_step $EVAL_STEP \
--eval_mode $EVAL_MODE \
--link_tables $LINK_TABLES \
--selection_vote $SELECTION_VOTE \
--n_table_link $N_TABLE_LINK \
--temperature_table_link 0.8 \
--n_sql_generate $N_SQL_GENERATE \
--temperature_sql_generate 0.8 \
--n_sql_merge $N_SQL_MERGE \
--original_data_json $ORIGINAL_DATA_JSON \
--temperature_sql_merge 0.8 \
 >> ${train_log} 2>&1 &

echo "tail -f ${train_log}" > ./bin/temp.log
echo "run tflog to tail -f log"
echo "running"

echo "tail -f ${train_log}"
