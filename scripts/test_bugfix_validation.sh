#!/bin/bash
# Quick validation test for bug fixes on 5070 Ti
# Tests: batching consistency, image normalization, no crashes

python scripts/train_curriculum.py \
  --use_autocomplete_dataset \
  --use_gradient_checkpointing \
  --llm_model_name "Qwen/Qwen3-8B" \
  --truncated_json_root "./data/Omni-CAD-subset/json_truncated" \
  --omnicad_txt_path "./data/Omni-CAD-subset/txt" \
  --omnicad_json_root "./data/Omni-CAD-subset/json" \
  --omnicad_img_root "./data/Omni-CAD-subset/img" \
  --omnicad_pc_root "./data/Omni-CAD-subset/pointcloud" \
  --max_train_samples 50 \
  --max_seq_length 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --stage3_epochs 1 \
  --stage3_lr 2e-5 \
  --lora_r 8 \
  --lora_alpha 16 \
  --warmup_steps 5 \
  --logging_steps 2 \
  --save_steps 1000 \
  --start_from_stage 3 \
  --stage1_epochs 0 \
  --stage2_epochs 0 \
  --device cuda \
  --dtype bfloat16

echo ""
echo "✓ Test completed! Check for:"
echo "  1. No shape mismatch errors"
echo "  2. Loss decreases over steps"
echo "  3. Memory stays under 16GB"
echo "  4. All modality combinations processed"
