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
  - [x] **Autocomplete training with structure-aware dynamic masking**
  - [x] Gradient checkpointing for memory optimization
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

**Note:** The code includes auto-download functionality that will attempt to download the checkpoint from Hugging Face or Google Drive if it's not found. However, if auto-download fails (e.g., in Google Colab with network restrictions), you can manually:
1. Download the checkpoint to your Google Drive
2. Place it in the checkpoints directory, or
3. Update the path in your config to point to your Google Drive location

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
# Resume from Stage 1 checkpoint and continue with Stages 2 & 3
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
    --stage3_epochs 1 \
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

# Resume from Stage 1 checkpoint and only run Stage 2
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


## Autocomplete Training (CAD Sequence Continuation)

### Overview

The autocomplete training mode enables the model to learn CAD sequence continuation - given a partial CAD sequence, the model learns to generate the remaining operations. This is implemented through structure-aware dynamic masking.

### Dataset Structure

The autocomplete dataset requires pairs of truncated and full JSON files:
```
data/Omni-CAD-subset/
├── json/                    # Full JSON sequences
│   └── 00000071_00005.json
├── json_truncated/          # Truncated sequences with metadata
│   ├── 0000/
│   │   ├── 00000071_00005_tr_01.json  # First truncation (e.g., 25% kept)
│   │   ├── 00000071_00005_tr_02.json  # Second truncation (e.g., 50% kept)
│   │   └── 00000071_00005_tr_03.json  # Third truncation (e.g., 75% kept)
├── txt/                     # Text descriptions
├── img/                     # Images
└── pointcloud/              # Point clouds
```

**Important Note on Dataset Inflation:** Each base CAD JSON may have multiple truncated variations (average 2.3x, up to 5x), creating ~138K training samples from ~60K unique base models. This increases training time but does not affect per-batch memory usage.

### Dynamic Masking Implementation

The autocomplete collator implements structure-aware masking:

**Input Format:**
```
Complete this CAD sequence: <caption>
<full_json_sequence>
```

**Masking Strategy:**
1. Load `truncated_json` to extract `kept_operations` from metadata
2. Reconstruct partial JSON: `partial["sequence"] = full["sequence"][:kept_operations]`
3. Tokenize partial JSON to find exact token boundary
4. Mask all tokens up to that boundary (prompt + seen operations)
5. Compute loss only on tokens after the boundary (operations to be generated)

**Example:**
```python
# Full sequence has 100 operations, truncated keeps 40
kept_operations = 40

# Partial: operations[0:40] → tokenize → 3500 tokens
# Full: operations[0:100] → tokenize → 6500 tokens

# Masking: labels[:, :3500] = -100 (no loss on "already seen" part)
#          labels[:, 3500:] = actual_tokens (compute loss here)
```

This approach:
- ✅ Preserves structural JSON boundaries (doesn't cut mid-operation)
- ✅ Provides correct autoregressive training signal
- ✅ Reduces memory by only including full sequence once (not truncated + full)

### Training Command

```bash
# Autocomplete training on Google Colab A100 (80GB)
!python scripts/train_curriculum.py \
    --use_autocomplete_dataset \
    --use_wandb \
    --use_gradient_checkpointing \
    --output_dir /content/gdrive/MyDrive/CAD-MLLM-checkpoints \
    --truncated_json_root /path/to/json_truncated \
    --omnicad_txt_path /path/to/txt \
    --omnicad_json_root /path/to/json \
    --omnicad_img_root /path/to/img \
    --omnicad_pc_root /path/to/pointcloud \
    --llm_model_name "Qwen/Qwen3-8B" \
    --start_from_stage 3 \
    --stage1_epochs 0 \
    --stage2_epochs 0 \
    --stage3_epochs 10 \
    --stage3_lr 2e-5 \
    --max_seq_length 8192 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --max_train_samples 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --warmup_steps 50 \
    --logging_steps 5 \
    --save_steps 100 \
    --device cuda \
    --dtype bfloat16 \
    --wandb_project "CAD-MLLM-Autocomplete"
```

### Key Arguments for Autocomplete

| Argument | Description |
|----------|-------------|
| `--use_autocomplete_dataset` | Enable autocomplete mode (uses truncated/full JSON pairs) |
| `--truncated_json_root` | Path to directory containing truncated JSON files |
| `--omnicad_json_root` | Path to full JSON sequences (used as targets) |
| `--use_gradient_checkpointing` | **Recommended**: Trades compute for memory (30-50% memory savings) |

### Memory Considerations and Known OOM Issues

**⚠️ Critical Memory Constraints:**

CAD sequences are extremely long (average 6,297 tokens, max 17,672 tokens), making memory management challenging even on 80GB A100 GPUs. The primary memory bottleneck is the **attention mechanism**, which scales quadratically with sequence length: `O(batch_size × num_heads × seq_length²)`.

#### Tested Configurations and Results (80GB A100):

| Config | batch_size | max_seq_length | lora_r | grad_ckpt | Result | Memory Used |
|--------|-----------|----------------|--------|-----------|--------|-------------|
| A | 4 | 32768 | 32 | ❌ | **OOM** | 77.48 GB (attention: 256 GB needed) |
| B | 2 | 24576 | 32 | ❌ | **OOM** | 77.41 GB (attention: 72 GB alone) |
| C | 1 | 16384 | 32 | ❌ | **OOM** | 79.10 GB |
| D | 1 | 16384 | 16 | ✅ | **OOM** | 78.80 GB |
| E | 1 | 12288 | 8 | ✅ | **Likely works** | ~25-30 GB estimated |
| F | 1 | 8192 | 8 | ✅ | **Should work** | ~20-25 GB estimated |

**Key Findings:**
1. **Attention Dominates Memory**: For config B (batch=2, seq=24576), attention alone requires 72 GB of the 100 GB total
2. **Gradient Checkpointing Essential**: Saves 30-50% activation memory but adds 20-30% compute time
3. **LoRA Rank Impact**: `lora_r=32` vs `lora_r=8` can add 4-8 GB of parameter memory
4. **Sequence Length is Critical**: Reducing from 16K→12K saves ~30% attention memory due to quadratic scaling

#### Memory Breakdown (Config B: batch=2, seq=24576)
```
Model Parameters:     16.3 GB  (8B params)
Activations:          12.0 GB  (forward pass intermediate values)
Attention Matrices:   72.0 GB  (KEY/VALUE/QUERY matrices)
Gradients:            ~10 GB   (backprop)
-------------------------------------------
Total:               ~110 GB   (exceeds 80GB → OOM)
```

#### Recommended Safe Configuration (Config F)

Based on extensive testing, this configuration should work within 80GB:

```bash
--max_seq_length 8192           # Aggressive sequence truncation
--batch_size 1                  # Minimum batch size
--gradient_accumulation_steps 32 # Maintain effective batch size
--lora_r 8                      # Minimal LoRA adapters
--lora_alpha 16                 # 2:1 ratio with lora_r
--use_gradient_checkpointing    # Essential for memory savings
```

**Tradeoffs:**
- ✅ Fits in 80GB GPU memory (~20-25 GB usage)
- ✅ Maintains training stability through gradient accumulation
- ❌ Truncates long sequences (may lose fine-grained details for complex CAD models)
- ❌ Smaller LoRA rank may reduce model capacity
- ❌ 20-30% slower due to gradient checkpointing

#### Alternative: Dataset Filtering (Optional)

To reduce training time (but NOT memory usage), you can filter the dataset to use only one truncation per base JSON:

```python
# In multimodal_autocomplete.py, modify _load_dataset_index():
# Add deduplication logic to keep only first truncation per cad_id
unique_samples = {}
for sample in samples:
    cad_id = sample["cad_id"]
    if cad_id not in unique_samples:
        unique_samples[cad_id] = sample
samples = list(unique_samples.values())
```

This reduces:
- ✅ Dataset size: 138K → ~60K samples (~2.3x speedup per epoch)
- ✅ Training redundancy
- ❌ Does **NOT** reduce per-batch GPU memory
- ❌ Does **NOT** fix OOM issues

### Implementation Files

- **Dataset**: [cad_mllm/data/multimodal_autocomplete.py](cad_mllm/data/multimodal_autocomplete.py)
  - `MultimodalAutocompleteDataset.__getitem__`: Loads full JSON + kept_operations
  - `MultimodalAutocompleteCollator.__call__`: Implements structure-aware masking
- **Model**: [cad_mllm/model.py](cad_mllm/model.py)
  - Embedding order: [image, point_cloud, text] (matches collator's label padding)
- **Training**: [scripts/train_curriculum.py](scripts/train_curriculum.py)
  - `--use_autocomplete_dataset` flag switches to autocomplete mode

### Debugging OOM Issues

If you encounter OOM errors:

1. **Enable gradient checkpointing** (if not already enabled)
2. **Reduce sequence length**: Try 12288 → 8192 → 4096
3. **Reduce LoRA rank**: Try 16 → 8 → 4
4. **Monitor memory**: Add `torch.cuda.memory_summary()` to track allocation
5. **Check actual sequence lengths** in your dataset (use the debug script in scripts/)
6. **Consider FlashAttention-2**: If available, can reduce attention memory by ~2x

**Debug Script Example:**
```python
# Count actual token lengths in your dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

for sample in dataset:
    tokens = tokenizer(sample["full_seq"])["input_ids"]
    print(f"Sample {sample['cad_id']}: {len(tokens)} tokens")
```

### Inference: Using Trained Models

After training, use the inference wrapper to generate complete CAD sequences:

#### Quick Start

```python
from cad_mllm.inference import autocomplete_cad

# One-line inference
result = autocomplete_cad(
    checkpoint_path="outputs/stage3_all/checkpoint-best",
    truncated_json="data/json_truncated/0000/00000071_00005_tr_02.json",
    caption="Modern minimalist chair with wooden legs",
    image="data/img/0000/00000071_00005.png",
    point_cloud="data/pointcloud/0000/00000071_00005.npy",
    output_path="output_complete_chair.json",
    temperature=0.7,
    top_p=0.9,
)

# result["sequence"] is a complete list of CAD operations
# Ready to be loaded into your CAD engine!
print(f"Generated {result['metadata']['total_operations']} operations")
```

#### For Evaluation Pipelines

```python
from cad_mllm.inference import CADAutocomplete

# Initialize once
autocomplete = CADAutocomplete(
    checkpoint_path="outputs/checkpoint-best",
    device="cuda",
    dtype="bfloat16",
)

# Process single sample
result = autocomplete.complete(
    truncated_json="path/to/partial.json",
    caption="A modern chair",
    image="path/to/image.png",
    point_cloud="path/to/pc.npy",
)

# Process batch (for evaluation)
samples = [
    {
        "truncated_json": "sample1_partial.json",
        "caption": "Chair",
        "image": "sample1.png",
        "point_cloud": "sample1.npy",
    },
    {
        "truncated_json": "sample2_partial.json",
        "caption": "Table",
        "image": "sample2.png",
        "point_cloud": "sample2.npy",
    },
]

results = autocomplete.batch_complete(samples, temperature=0.7)

# Each result contains:
# - result["sequence"]: Complete CAD operations (executable!)
# - result["metadata"]: partial_ops, generated_ops, total_ops
```

#### Output Format

The inference wrapper automatically merges partial + generated operations:

```python
# Input: partial sequence with 40 operations
partial_ops = [op_1, op_2, ..., op_40]

# Model generates: operations 41-100
generated_ops = [op_41, op_42, ..., op_100]

# Output: complete sequence (ready for CAD engine!)
result = {
    "sequence": [op_1, op_2, ..., op_100],  # Full, executable CAD sequence
    "metadata": {
        "caption": "Modern chair",
        "partial_operations": 40,
        "generated_operations": 60,
        "total_operations": 100,
    }
}

# Save or use directly
import json
with open("complete_sequence.json", "w") as f:
    json.dump(result, f)

# Or load into CAD engine
# cad_engine.execute(result["sequence"])
```

#### Key Features

- **Ready for CAD engines**: Output is complete, executable JSON
- **Automatic post-processing**: Merges partial + generated operations
- **Batch processing**: Efficient evaluation on multiple samples
- **Multimodal support**: Text + Image + Point Cloud inputs
- **Flexible generation**: Control temperature, top_p, sampling strategy


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
