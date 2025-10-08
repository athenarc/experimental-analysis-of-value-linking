#!/bin/bash

# This script fine-tunes a model for the value linking task on a specific database.
# It incorporates best practices and hyperparameters from the original RecLM-cgen script.
# Usage: ./scripts/train_valuelinking.sh [db_name] [gpu_ids]
# Example: ./scripts/train_valuelinking.sh student_db 0,1

# --- Configuration ---
if [ -z "$1" ]; then
  echo "Error: No database name provided."
  echo "Usage: ./scripts/train_valuelinking.sh [db_name] [gpu_ids]"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Error: No GPU IDs provided."
  echo "Usage: ./scripts/train_valuelinking.sh [db_name] [gpu_ids]"
  exit 1
fi

DB_NAME=$1
GPU_IDS=$2
NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

BACKBONE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
CHAT_TEMPLATE="llama-3"
OUTPUT_DIR_BASE="snap/ValueLinkingIdentity"

# --- Training Parameters (inspired by original script) ---
# Small per-device batch size with large accumulation to save VRAM and simulate a large batch
BATCH_SIZE_PER_GPU=8
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
EPOCHS=25
LORA_R=16
LORA_ALPHA=32

# Calculate total batch size
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

# --- Construct Output Path ---
TIMESTAMP=$(date "+%m%d")
OUTPUT_PATH="${OUTPUT_DIR_BASE}/${TIMESTAMP}-${DB_NAME}-bs${TOTAL_BATCH_SIZE}-lr${LEARNING_RATE}/"
mkdir -p "$OUTPUT_PATH"
echo "Output will be saved to: $OUTPUT_PATH"

echo "--- Starting Value Linking Fine-Tuning ---"
echo "Database: $DB_NAME"
echo "Base Model: $BACKBONE_MODEL"
echo "GPU IDs: $GPU_IDS"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Size per GPU: $BATCH_SIZE_PER_GPU"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Total Effective Batch Size: $TOTAL_BATCH_SIZE"
echo "-------------------------------------------"

# --- Accelerate Launch Command ---
# Using nohup to run in the background and redirecting output to a log file
accelerate launch \
  --config_file accelerate.yaml \
  --num_processes $NUM_GPUS \
  --gpu_ids $GPU_IDS \
  main.py \
  --train_stage ValueLinking_SFT \
  --data_path data/dataset_identity/${DB_NAME}/ \
  --output ${OUTPUT_PATH} \
  --backbone ${BACKBONE_MODEL} \
  --chat_template ${CHAT_TEMPLATE} \
  \
  --SFT_train_tasks ValueLinking \
  \
  --use_control_symbol \
  --use_scope_mask \
  --scope_mask_type 3 \
  \
  --epoch ${EPOCHS} \
  --batch_size ${BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --lr ${LEARNING_RATE} \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  \
  --SFT_actor_lora_r ${LORA_R} \
  --SFT_actor_lora_a ${LORA_ALPHA} \
  --lora_dropout 0.05 \
  \
  --seed 42 \
  --clip_grad_norm 1.0 \
  --FA2 \
  --max_token_length 512 \
  --no_value_linking_curriculum \
  --gen_max_length 256 > "${OUTPUT_PATH}output.log" 2>&1

echo "Training started in the background. PID: $!"
echo "You can monitor the progress with: tail -f ${OUTPUT_PATH}output.log"