#!/bin/bash
# ============================================================================
# STAGE 1: TEXT-ONLY TRAINING (Foundation Stage)
# This teaches the model to generate CAD JSON from text descriptions FIRST
# ============================================================================

python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --use_wandb \
    --use_gradient_checkpointing \
    --output_dir /content/gdrive/MyDrive/CAD-MLLM-checkpoints \
    --truncated_json_root /content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/json_truncated \
    --omnicad_txt_path /content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/txt \
    --omnicad_json_root /content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/json \
    --omnicad_img_root /content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/img \
    --omnicad_pc_root /content/gdrive/.shortcut-targets-by-id/1hjHd8hSpbvh2rFPApUzjUViYVV9vdK5H/Omni-CAD-subset-complete/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 1 \
    --stage1_epochs 5 \
    --stage2_epochs 0 \
    --stage3_epochs 0 \
    --stage1_lr 2e-5 \
    --max_seq_length 4000 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_train_samples 5000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --warmup_steps 50 \
    --logging_steps 5 \
    --save_steps 100 \
    --keep_last_n_checkpoints 3 \
    --device cuda \
    --dtype bfloat16 \
    --wandb_project "CAD-MLLM-Stage1-TextOnly"

# After Stage 1 completes, use the checkpoint-best from stage1_text/ for Stage 2 or 3!
