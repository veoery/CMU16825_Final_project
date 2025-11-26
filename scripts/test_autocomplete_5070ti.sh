#!/bin/bash
# Test script for autocompletion training on RTX 5070 Ti
# Run this in WSL with conda environment activated: conda activate p3d_5070ti

echo "============================="
echo "Testing Autocompletion on 5070 Ti"
echo "============================="

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Configuration for 5070 Ti (16GB VRAM)
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_SEQ_LEN=16384
MAX_SAMPLES=1000  # Test with 1000 samples first

# Data paths (use WSL paths, not Windows paths)
DATA_ROOT="./data/Omni-CAD-subset"
TXT_PATH="$DATA_ROOT/txt"
TRUNCATED_JSON="$DATA_ROOT/json_truncated"
FULL_JSON="$DATA_ROOT/json"
IMG_ROOT="$DATA_ROOT/img"
PC_ROOT="$DATA_ROOT/pointcloud"

echo ""
echo "Data paths:"
echo "  Text: $TXT_PATH"
echo "  Truncated JSON: $TRUNCATED_JSON"
echo "  Full JSON: $FULL_JSON"
echo "  Images: $IMG_ROOT"
echo "  Point clouds: $PC_ROOT"
echo ""

# Check if truncated JSON directory exists
if [ ! -d "$TRUNCATED_JSON" ]; then
    echo "ERROR: Truncated JSON directory not found: $TRUNCATED_JSON"
    exit 1
fi

# Run training
python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --truncated_json_root "$TRUNCATED_JSON" \
    --omnicad_txt_path "$TXT_PATH" \
    --omnicad_json_root "$FULL_JSON" \
    --omnicad_img_root "$IMG_ROOT" \
    --omnicad_pc_root "$PC_ROOT" \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 3 \
    --stage1_epochs 0 \
    --stage2_epochs 0 \
    --stage3_epochs 1 \
    --stage3_lr 2e-4 \
    --max_seq_length $MAX_SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_train_samples $MAX_SAMPLES \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16 \
    --use_wandb \
    --wandb_project "CAD-MLLM-Autocomplete-Test" \
    --output_dir "./outputs_test"

echo ""
echo "============================="
echo "Test completed!"
echo "============================="
