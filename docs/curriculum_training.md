# Curriculum-Based Progressive Training for CAD-MLLM

## Overview

This document explains the curriculum-based training strategy implemented for CAD-MLLM, which progressively introduces modalities to improve multimodal learning.

## Motivation

Training multimodal models directly on all modalities simultaneously can lead to:
- **Optimization difficulties**: Different modalities have different learning dynamics
- **Modality imbalance**: The model may over-rely on certain modalities
- **Slower convergence**: Conflicting gradients from different modalities

The curriculum approach addresses these issues by:
1. Building a strong text-based foundation first
2. Gradually introducing additional modalities
3. Using random modality combinations to prevent overfitting

## Training Stages

### Stage 1: Text Only (Foundation)
**Goal:** Learn CAD generation from text descriptions

- **Modalities:** Text only
- **Trainable Components:**
  - LLM with LoRA adapters
  - Text encoder (LLM embeddings)
- **Frozen Components:** N/A
- **Sampling Strategy:** 100% text-only samples
- **Typical Duration:** 3-5 epochs

**Learning Objectives:**
- Understand CAD command syntax
- Map natural language to parametric CAD operations
- Build strong text-to-CAD baseline

### Stage 2: Text + Point Cloud (Geometric Understanding)
**Goal:** Incorporate 3D geometric understanding

- **Modalities:** Text + Point Cloud
- **Trainable Components:**
  - LLM with LoRA adapters
  - Point cloud projector (higher LR)
- **Frozen Components:**
  - Point cloud encoder (Michelangelo)
- **Sampling Strategy:**
  - 30% text-only
  - 70% text + point cloud
- **Typical Duration:** 3-5 epochs

**Learning Objectives:**
- Align point cloud features with LLM space
- Learn to condition CAD generation on 3D geometry
- Maintain text-only generation capability

### Stage 3: All Modalities (Complete Multimodal)
**Goal:** Full multimodal CAD generation

- **Modalities:** Text + Point Cloud + Image
- **Trainable Components:**
  - LLM with LoRA adapters
  - Point cloud projector (higher LR)
  - Image projector (higher LR)
- **Frozen Components:**
  - Point cloud encoder (Michelangelo)
  - Image encoder (DINOv2)
- **Sampling Strategy:**
  - 20% text-only
  - 30% text + point cloud
  - 20% text + image
  - 30% text + point cloud + image
- **Typical Duration:** 5-10 epochs
- **Learning Rate:** Typically reduced (1e-5 vs 2e-5)

**Learning Objectives:**
- Integrate visual information
- Handle any combination of modalities
- Robust multimodal fusion

## Key Design Decisions

### 1. Frozen Encoders
**Why:** Encoders (DINOv2, Michelangelo) are pre-trained on large-scale data and provide strong feature representations. Keeping them frozen:
- Reduces trainable parameters (memory efficient)
- Prevents catastrophic forgetting
- Leverages transfer learning

### 2. Higher Projector Learning Rate
**Why:** Projectors (5x LLM LR) are randomly initialized and need to learn alignment from scratch:
- Faster adaptation to LLM feature space
- Compensates for smaller parameter count
- Prevents projectors from becoming bottlenecks

### 3. Random Modality Sampling
**Why:** Each batch contains different modality combinations:
- Prevents overfitting to specific combinations
- Ensures model works with any input configuration
- Simulates real-world usage patterns

### 4. Progressive Learning Rates
**Why:** Lower LR in final stage (1e-5 vs 2e-5):
- Fine-tuning on top of learned representations
- Prevents disrupting earlier learning
- Stabilizes training

## Implementation Details

### Parameter Groups

```python
optimizer_params = [
    {
        'params': llm_params,
        'lr': stage_lr,  # e.g., 2e-5
    },
    {
        'params': projector_params,
        'lr': stage_lr * 5.0,  # e.g., 1e-4
    },
]
```

### Modality Sampling

```python
# Stage 3 example
stage = CurriculumStage(
    name="stage3_all",
    modalities=["text", "point_cloud", "image"],
    modality_sample_probs={
        "text": 0.2,
        "text+point_cloud": 0.3,
        "text+image": 0.2,
        "text+point_cloud+image": 0.3,
    },
)
```

During each `__getitem__` call, the dataset randomly samples a modality combination based on these probabilities.

### Model State Management

The model's `set_trainable_params()` method controls which components are trained:

```python
model.set_trainable_params(
    train_llm=True,          # LoRA adapters
    train_projectors=True,   # MLP projectors
    train_encoders=False,    # Keep frozen
)
```

## Usage

### Basic Usage

```bash
python scripts/train_curriculum.py \
    --omnicad_txt_path data/Omni-CAD/txt/ \
    --omnicad_json_root data/Omni-CAD/json \
    --stage1_epochs 5 \
    --stage2_epochs 5 \
    --stage3_epochs 10 \
    --device cuda
```

### Advanced Configuration

```bash
python scripts/train_curriculum.py \
    --omnicad_txt_path data/Omni-CAD/txt/ \
    --omnicad_json_root data/Omni-CAD/json \
    --omnicad_img_root data/Omni-CAD/img \
    --omnicad_pc_root data/Omni-CAD/pcd \
    --llm_model_name "Qwen/Qwen2.5-7B" \
    --lora_r 16 \
    --lora_alpha 32 \
    --stage1_epochs 5 \
    --stage2_epochs 5 \
    --stage3_epochs 10 \
    --stage1_lr 2e-5 \
    --stage2_lr 2e-5 \
    --stage3_lr 1e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --projector_lr_multiplier 5.0 \
    --use_wandb \
    --wandb_project "CAD-MLLM-Curriculum" \
    --device cuda
```

## Expected Training Behavior

### Stage 1: Text Only
- **Loss:** Should decrease steadily
- **Convergence:** Relatively fast (text is the primary signal)
- **Memory:** Lower usage (no image/PC encoders loaded)

### Stage 2: Text + Point Cloud
- **Loss:** May spike initially as projector is initialized
- **Convergence:** Gradual improvement as projector learns alignment
- **Memory:** Increased (PC encoder + projector loaded)

### Stage 3: All Modalities
- **Loss:** May spike again with image addition
- **Convergence:** Slower but more stable
- **Memory:** Highest (all encoders + projectors loaded)

## Monitoring with Weights & Biases

The training script logs:
- Per-step loss and learning rate
- Per-stage metrics
- Gradient norms
- Active modalities per stage

Example wandb tags:
- `train/loss`: Current training loss
- `train/stage`: Current curriculum stage
- `epoch/stage1_text_loss`: Average loss for Stage 1
- `epoch/stage2_text_pc_loss`: Average loss for Stage 2
- `epoch/stage3_all_loss`: Average loss for Stage 3

## Troubleshooting

### Issue: Loss doesn't decrease in Stage 2/3

**Possible Causes:**
- Projector LR too low → Increase `--projector_lr_multiplier`
- Stage LR too high → Reduce `--stage2_lr` or `--stage3_lr`
- Insufficient epochs → Increase stage epochs

### Issue: Out of memory

**Solutions:**
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Use smaller model (e.g., Qwen3-0.6B instead of Qwen2.5-7B)
- Use `--dtype float16` instead of `bfloat16`

### Issue: Model forgets text-only generation

**Cause:** Insufficient text-only sampling in later stages

**Solution:**
- Increase text-only probability in stage modality sampling
- Add more Stage 1 epochs to build stronger foundation

## Future Improvements

1. **Dynamic Curriculum:** Automatically adjust stage duration based on validation loss
2. **Modality Dropout:** Randomly drop entire modalities during training
3. **Adaptive LR:** Per-projector learning rates based on modality complexity
4. **Curriculum Scheduling:** Gradually shift sampling probabilities within each stage

## References

- Original Paper: [CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM](https://arxiv.org/abs/2411.04954)
- LLaVA: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- Curriculum Learning: [On the Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626)
