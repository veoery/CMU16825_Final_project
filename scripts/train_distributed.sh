#!/bin/bash

# Distributed Training Script for CAD-MLLM with DeepSpeed ZeRO-3
# Usage: bash scripts/train_distributed.sh

# NCCL environment variables (for troubleshooting)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
# Uncomment if you have network issues:
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# Number of GPUs
NUM_GPUS=4

# DeepSpeed configuration
DEEPSPEED_CONFIG="configs/deepspeed/ds_zero3_config.json"

# Training arguments
LLM_MODEL="Qwen/Qwen3-8B"
OUTPUT_DIR="./CAD-MLLM-distributed-ckpts"
BATCH_SIZE=4  # Per GPU batch size
GRAD_ACCUM=4
MAX_SEQ_LEN=512

# Curriculum settings
STAGE1_EPOCHS=3
STAGE2_EPOCHS=3
STAGE3_EPOCHS=5

# Learning rates
STAGE1_LR=2e-4
STAGE2_LR=2e-4
STAGE3_LR=1e-4

# LoRA settings
LORA_R=32
LORA_ALPHA=64

# Data paths
OMNICAD_TXT="./data/Omni-CAD-subset-complete/txt"
OMNICAD_JSON="./data/Omni-CAD-subset-complete/json"
OMNICAD_IMG="./data/Omni-CAD-subset-complete/img"
OMNICAD_PC="./data/Omni-CAD-subset-complete/pointcloud"

# W&B settings (optional)
WANDB_PROJECT="CAD-MLLM-Curriculum-Distributed"
WANDB_RUN_NAME="4gpu-zero3-"

# Launch distributed training
deepspeed --num_gpus=$NUM_GPUS scripts/train_curriculum.py \
    --deepspeed \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --llm_model_name $LLM_MODEL \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --attn_implementation flash_attention_2 \
    --use_gradient_checkpointing \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_seq_length $MAX_SEQ_LEN \
    --stage1_epochs $STAGE1_EPOCHS \
    --stage2_epochs $STAGE2_EPOCHS \
    --stage3_epochs $STAGE3_EPOCHS \
    --stage1_lr $STAGE1_LR \
    --stage2_lr $STAGE2_LR \
    --stage3_lr $STAGE3_LR \
    --omnicad_txt_path $OMNICAD_TXT \
    --omnicad_json_root $OMNICAD_JSON \
    --omnicad_img_root $OMNICAD_IMG \
    --omnicad_pc_root $OMNICAD_PC \
    --logging_steps 10 \
    --save_steps 500 \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
    --seed 42

# Alternative: Use with dummy data for testing
# Add --create_dummy_data --num_dummy_samples 1000

# Alternative: Resume from checkpoint
# Add --resume_from_ckpt outputs_curriculum_distributed/stage2_text_pc_model --start_from_stage 3

# Alternative: Use ZeRO-2 instead (faster, less memory efficient)
# Change DEEPSPEED_CONFIG to "configs/deepspeed/ds_zero2_config.json"

# Alternative: Use CPU offloading (for even larger models)
# Change DEEPSPEED_CONFIG to "configs/deepspeed/ds_zero3_offload.json"
