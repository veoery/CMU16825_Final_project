#!/bin/bash
# Extreme conservative test for 5070 Ti - seq_len=1024
# Run this in WSL with conda environment activated: conda activate p3d_5070ti
# Use this only if seq_len=2048 still OOMs

echo "============================="
echo "Testing Autocompletion on 5070 Ti (seq_len=1024 - MINIMAL)"
echo "============================="

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Extreme conservative configuration for 5070 Ti (16GB VRAM)
BATCH_SIZE=1
GRAD_ACCUM=64
MAX_SEQ_LEN=1024  # Extremely conservative
LORA_R=8  # Minimal LoRA rank
MAX_SAMPLES=50  # Test with just 50 samples

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
echo "Memory settings:"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Max sequence length: $MAX_SEQ_LEN (EXTREMELY CONSERVATIVE)"
echo "  LoRA rank: $LORA_R"
echo "  Max samples: $MAX_SAMPLES"
echo ""

# Check if truncated JSON directory exists
if [ ! -d "$TRUNCATED_JSON" ]; then
    echo "ERROR: Truncated JSON directory not found: $TRUNCATED_JSON"
    exit 1
fi

# Clear CUDA cache before starting
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

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
    --lora_r $LORA_R \
    --lora_alpha 16 \
    --warmup_steps 5 \
    --logging_steps 2 \
    --save_steps 25 \
    --device cuda \
    --dtype bfloat16 \
    --use_gradient_checkpointing \
    --use_wandb \
    --wandb_project "CAD-MLLM-Autocomplete-Test" \
    --output_dir "./outputs_test_1024"

echo ""
echo "============================="
echo "Test completed!"
echo "============================="
