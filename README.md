# CAD-MLLM Reproduction

This is a reproduction of the paper "CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM" ([arXiv:2411.04954](https://arxiv.org/abs/2411.04954)).


## Current Implementation Status

- [x] Project structure and environment setup
- [x] Text-only input pipeline
- [x] Image encoder integration (DINOv2)
- [x] Point cloud encoder integration (Michelangelo)
- [x] Training pipeline
  - [x] LoRA for LLM
  - [x] wandb monitoring
  - [x] Curriculum-based progressive training
  - [x] Multimodal data sampling
  - [x] Train with Text only
  - [x] Train with Text + Point Cloud
  - [x] Train with Text + Point Cloud + Image
  - [x] Projector training with separate learning rates
  - [ ] Stage 1+2
  - [ ] Stage 1+2+3

- [x] Evaluation metrics

## Setup

### Prerequisites
- Python >= 3.10
- `uv` package manager

### Installation

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -e .

# Install Michelangelo
uv add --editable ./Michelangelo --no-build-isolation
```

Put the Michelangelo config file here:
```configs/michelangelo_point_encoder_cfg.yaml```

Put the Michelangelo checkpoint here:
```checkpoints/michelangelo_point_encoder_state_dict.pt```

Download Michelangelo checkpoint from:
https://drive.google.com/file/d/1wzfa4EoijmyfTpLPfma9r03wvQ_mbjFD/view?usp=drive_link

**If using Google Colab:**
```
# Check scripts/curriculum_training.ipynb
!uv venv
!source .venv/bin/activate && uv pip install -e .
!source .venv/bin/activate && uv pip install -e ./Michelangelo --no-build-isolation
```

### Dateset

Follow the original repo for the dataset: 
https://github.com/CAD-MLLM/CAD-MLLM?tab=readme-ov-file#data



## Usage

### Inference

```bash
python scripts/inference.py --prompt "Generate a CAD model of a simple cube." --device mps --image_path "data/Omni-CAD/img/cube.jpeg"  --dtype bfloat16
```
```
python scripts/inference.py --prompt "Generate a CAD model of a simple cube." --device cuda --llm_model_name "Qwen/Qwen3-0.6B" --pc_path data/Omni-CAD/pcd/00000071_00005.npz --dtype bfloat16
```

Check the scripts/inference.py and config.py for more details.



### Training

#### Curriculum Training

The recommended training approach uses a curriculum-based strategy that progressively introduces modalities:

**Stage 1: Text Only** → **Stage 2: Text + Point Cloud** → **Stage 3: Text + Point Cloud + Image**

Each stage randomly combines available modalities during training for robust multimodal learning.

#### Test Training with Dummy Data
```bash
# Train from scratch
!python scripts/train_curriculum.py \
    --start_from_stage 1 \
    --create_dummy_data \
    --stage1_epochs 2 \
    --stage2_epochs 2 \
    --stage3_epochs 3 \
    --device cuda

# Resume from Stage 1 checkpoint and continue with Stages 2 & 3
!python scripts/train_curriculum.py \
    --resume_from_ckpt /path/to/ckpt/stage1_text_model \
    --start_from_stage 2 \
    --create_dummy_data \
    --stage2_epochs 2 \
    --stage3_epochs 3 \
    --device cuda

# Resume from Stage 3 checkpoint and continue with Stages 3
!python scripts/train_curriculum.py \
    --resume_from_ckpt /path/to/ckpt/stage3_all_model \
    --start_from_stage 3 \
    --create_dummy_data \
    --stage2_epochs 2 \
    --stage3_epochs 3 \
    --device cuda

# Train with stage 1, 2 and 3 (full curriculum training)
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset-complete/txt \
    --omnicad_json_root ./data/Omni-CAD-subset-complete/json \
    --omnicad_img_root ./data/Omni-CAD-subset-complete/img \
    --omnicad_pc_root ./data/Omni-CAD-subset-complete/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 1 \
    --stage1_epochs 1 \
    --stage2_epochs 1 \
    --stage3_epochs 1 \
    --stage1_lr 2e-4 \
    --stage2_lr 2e-4 \
    --stage3_lr 2e-4 \
    --max_seq_length 32768 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \    
    --device cuda \
    --dtype bfloat16
```

#### Training with Omni-CAD Dataset (what we need for now)
```bash
# Resume from Stage 1 checkpoint and continue with Stages 2
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset-complete/txt \
    --omnicad_json_root ./data/Omni-CAD-subset-complete/json \
    --omnicad_img_root ./data/Omni-CAD-subset-complete/img \
    --omnicad_pc_root ./data/Omni-CAD-subset-complete/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --resume_from_ckpt /path/to/ckpt/stage1_text_model \
    --start_from_stage 2 \
    --stage2_epochs 1 \
    --stage3_epochs 0 \
    --stage2_lr 2e-4 \
    --stage3_lr 2e-4 \
    --max_seq_length 32768 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \    
    --device cuda \
    --dtype bfloat16

# Resume from Stage 2 checkpoint and only run Stage 3
!python scripts/train_curriculum.py \
    --use_wandb \
    --omnicad_txt_path ./data/Omni-CAD-subset-complete/txt \
    --omnicad_json_root ./data/Omni-CAD-subset-complete/json \
    --omnicad_img_root ./data/Omni-CAD-subset-complete/img \
    --omnicad_pc_root ./data/Omni-CAD-subset-complete/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --resume_from_ckpt /path/to/ckpt/stage2_text_pc_model \
    --start_from_stage 3 \
    --stage3_epochs 1 \
    --stage3_lr 2e-4 \
    --max_seq_length 32768 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 100 \    
    --device cuda \
    --dtype bfloat16
```

**Arguments:**
- `--resume_from_ckpt`: Path to checkpoint to resume from (can be any checkpoint - stage model or epoch checkpoint). Works independently of `--start_from_stage`.
- `--start_from_stage`: Stage number to start from (1=Stage 1, 2=Stage 2, 3=Stage 3). Can start from any stage with or without a checkpoint.


#### Monitoring Training with Weights & Biases

The training script supports Weights & Biases (wandb) for comprehensive experiment tracking.

##### First-time Setup

```bash
# Install wandb (if not already installed)
uv add wandb

# Login to wandb
wandb login
```

##### Wandb Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_wandb` | False | Enable wandb logging |
| `--wandb_project` | "CAD-MLLM" | Project name in wandb |
| `--wandb_run_name` | Auto-generated | Custom run name (see below) |
| `--wandb_entity` | None | Team/organization name |

**Auto-generated Run Name Format:**

When `--wandb_run_name` is not specified, the run name is automatically constructed from key hyperparameters:

```
<model>-<lora|full>-<lr>-<bs>-<ep>
```

Examples:
- `Qwen3-4B-lora-r8-lr2e-05-bs8-ep10` (LoRA training)
- `Qwen3-4B-full-lr1e-04-bs16-ep20` (Full fine-tuning)
- `Qwen3-0.6B-lora-r16-lr5e-05-bs4-ep5` (Small model with LoRA)



## Project Structure

```
CAD-MLLM/
├── assets/
├── data/                   # follow the original repo for the dataset
├── cad_mllm/
│   ├── __init__.py
│   ├── config.py           # Configuration classes
│   ├── model.py            # Main CAD-MLLM model
│   ├── encoders/           # Modality encoders
│   │   ├── __init__.py
│   │   ├── text_encoder.py
│   │   ├── image_encoder.py
│   │   └── pointcloud_encoder.py
│   ├── projectors/         # Projection layers
│   │   ├── __init__.py
│   │   └── mlp_projector.py
├── scripts/
│   └── inference.py       # Inference script
├── pyproject.toml
└── README.md
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@misc{xu2024CADMLLM,
    title={CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM},
    author={Jingwei Xu and Chenyu Wang and Zibo Zhao and Wen Liu and Yi Ma and Shenghua Gao},
    year={2024},
    eprint={2411.04954},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

This is a research reproduction for educational purposes.
