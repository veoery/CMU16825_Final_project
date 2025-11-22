# CAD-MLLM Reproduction

This is a reproduction of the paper "CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM" ([arXiv:2411.04954](https://arxiv.org/abs/2411.04954)).


## Current Implementation Status

- [x] Project structure and environment setup
- [x] Text-only input pipeline
- [x] Image encoder integration
- [x] Point cloud encoder integration
- [ ] Training pipeline
  - [x] LoRA for LLM (text only for now)
  - [x] wandb monitor
  - [ ] Train with Text + pc
  - [ ] Train with Text + pc + img
  
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


### Dateset

Follow the original repo for the dataset: 
https://github.com/CAD-MLLM/CAD-MLLM?tab=readme-ov-file#data



## Usage

### Text-to-CAD Generation

```bash
python scripts/inference.py --prompt "Generate a CAD model of a simple cube." --device mps --image_path "data/Omni-CAD/img/cube.jpeg"  --dtype bfloat16
```

Check the scripts/inference.py and config.py for more details.



### Training

#### Basic Training with Dummy Data

```bash
python scripts/train.py \
    --create_dummy_data \
    --num_dummy_samples 100 \
    --num_epochs 1 \
    --device mps \
    --lora_r 4 \
    --llm_model_name "Qwen/Qwen3-0.6B"
```

#### Training with Omni-CAD Dataset

```bash
# Single file
python scripts/train.py \
    --omnicad_txt_path data/Omni-CAD/txt/0000.json \
    --omnicad_json_root data/Omni-CAD/json \
    --num_epochs 3 \
    --batch_size 2 \
    --device cuda

# Multiple files (directory)
python scripts/train.py \
    --omnicad_txt_path data/Omni-CAD/txt/ \
    --omnicad_json_root data/Omni-CAD/json \
    --num_epochs 3 \
    --batch_size 4 \
    --device cuda
```

#### Training with Custom Train/Val Split

```bash
python scripts/train.py \
    --train_data_path data/Omni-CAD/txt/train/ \
    --val_data_path data/Omni-CAD/txt/val/ \
    --omnicad_json_root data/Omni-CAD/json \
    --num_epochs 10 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --device cuda
```

#### Monitoring Training with Weights & Biases

The training script supports Weights & Biases (wandb) for comprehensive experiment tracking.

##### First-time Setup

```bash
# Install wandb (if not already installed)
uv add wandb

# Login to wandb
wandb login
```

##### Basic Usage

```bash
# Enable wandb logging with auto-generated run name
# Run name will be automatically generated as: <model>-<lora-r#>-<lr>-<bs>-<ep>
# Example: "Qwen3-4B-lora-r8-lr2e-05-bs8-ep10"
python scripts/train.py \
    --use_wandb \
    --num_epochs 10 \
    --batch_size 4
```

##### Custom Project and Run Names

```bash
# Override auto-generated run name with custom name
python scripts/train.py \
    --use_wandb \
    --wandb_project "my-cad-experiments" \
    --wandb_run_name "baseline-lora-r8" \
    --num_epochs 10 \
    --lora_r 8
```

##### Complete Example with Wandb

```bash
python scripts/train.py \
    --use_wandb \
    --wandb_project "CAD-MLLM-Experiments" \
    --wandb_run_name "lora-r16-lr2e5-bs8" \
    --train_data_path data/Omni-CAD/txt/train/ \
    --val_data_path data/Omni-CAD/txt/val/ \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --lora_r 16 \
    --lora_alpha 32 \
    --device cuda
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

This makes it easy to identify and compare runs based on their hyperparameters!

##### What Gets Logged

**Training Metrics (per step):**

- Training loss
- Learning rate
- Gradient norm
- Current epoch and step

**Epoch Metrics:**
- Average training loss per epoch

**Hyperparameters:**
- Model configuration (LLM name, LoRA settings)
- Training configuration (batch size, learning rate, etc.)
- Dataset information (paths, number of samples)

**Model Information:**
- Parameter count and gradients
- Weight histograms
- System metrics (GPU/CPU usage, memory)





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
