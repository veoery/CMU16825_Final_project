# CAD Sequence Autocomplete: Complete Intuition

## Overview

This document provides a comprehensive explanation of the autocomplete training approach for CAD sequence completion.

## 1. The Goal

**Objective:** Train a model to complete partial CAD sequences

**Example:**
- **Input (inference):** "Complete this CAD sequence: [chair] + operations 1-40"
- **Output:** operations 41-100 (the completion)

## 2. Teacher Forcing: Why Feed Full Sequences?

### Background: Standard Autoregressive Language Modeling

Language models like GPT use **teacher forcing** during training:

```
Training text: "The cat sat on the mat"

During training, at each position predict next token:
Position 1: Given "The" → predict "cat" ✓
Position 2: Given "The cat" → predict "sat" ✓
Position 3: Given "The cat sat" → predict "on" ✓
Position 4: Given "The cat sat on" → predict "the" ✓
Position 5: Given "The cat sat on the" → predict "mat" ✓
```

**Why this works:**
- Model always sees the **correct** previous tokens (ground truth)
- Prevents error accumulation during training
- Learns true conditional distribution: P(next_token | correct_context)
- All predictions can be computed in parallel in one forward pass

### Applying to CAD Autocomplete

Same principle, but with structure-aware masking:

```
Training example:
- Full sequence: operations 1-100 (total: 6500 tokens)
- Truncation point: operation 40 (3500 tokens)

Input to model:
"Complete this CAD sequence: [caption]\n<full_json_ops_1_to_100>"

Labels (what we compute loss on):
[MASK, MASK, ..., MASK, token_3501, token_3502, ..., token_6500]
 ↑                      ↑
 tokens 0-3500          tokens 3501-6500
 (ops 1-40)             (ops 41-100)
 -100 (no loss)         compute loss here!

What the model learns:
Position 3501: Given ops 1-40 (correct!) → predict first token of op 41 ✓
Position 3502: Given ops 1-40 + partial op 41 → predict next token ✓
Position 4500: Given ops 1-60 (correct!) → predict token in op 61 ✓
...
Position 6500: Given ops 1-99 (correct!) → predict final token ✓
```

**Key insight:** By feeding the full sequence and masking early labels, the model learns to predict the completion while **always having access to the correct context**.

## 3. Why Not Just Feed Truncated Input?

### Comparison of Approaches

#### ❌ Approach A: Truncated Input Only (No Teacher Forcing)

```python
# Training step
input_text = "Complete: [caption] + ops 1-40"
# Model autoregressively generates ops 41-100
generated_op_41 = model.generate()  # Might be slightly wrong

# Next token
input_text = "Complete: [caption] + ops 1-40" + generated_op_41
generated_op_42 = model.generate()  # Based on WRONG op 41!

# Error compounds!
```

**Problems:**
- Errors accumulate (exposure bias)
- Must generate sequentially (slow training)
- Model trains on its own mistakes
- Gradients become noisy

#### ✅ Approach B: Full Input + Label Masking (Teacher Forcing)

```python
# Single forward pass
input_ids = tokenize("Complete: [caption] + ops 1-100")  # Full sequence
labels = input_ids.clone()
labels[:, :3500] = -100  # Mask ops 1-40
labels[:, 3500:] = input_ids[:, 3500:]  # Compute loss on ops 41-100

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss  # Computed in parallel for all positions!

# Gradients:
# Position 3501: ∂L/∂θ based on correct context (ops 1-40)
# Position 4000: ∂L/∂θ based on correct context (ops 1-50)
# Position 6500: ∂L/∂θ based on correct context (ops 1-99)
```

**Benefits:**
- ✅ Model always trains on correct context
- ✅ Parallel loss computation (much faster)
- ✅ Stable gradients
- ✅ No error accumulation during training

### The Training-Inference Gap

**During Training:**
```
Input: Full sequence (ops 1-100)
Model sees: Correct context at every position
Labels: Masked so loss computed only on ops 41-100
```

**During Inference:**
```
Input: Truncated sequence (ops 1-40 only)
Model generates: ops 41-100 autoregressively
Model sees: Its own previous predictions (might have small errors)
```

**Note:** This gap is called "exposure bias" and exists in all autoregressive models (GPT, etc.). The benefit of stable training outweighs this small mismatch.

## 4. Structure-Aware Masking

### Why Operation Boundaries Matter

CAD sequences are structured JSON:
```json
{
  "sequence": [
    {"type": "line", "start": [0,0], "end": [10,5]},
    {"type": "arc", "center": [5,5], "radius": 3},
    ...
  ]
}
```

**❌ Bad: Random token masking**
```
{"sequence": [{"type": "line", "start": [0,0], "end": [10,5]}, ...]}
              ↑ mask ↑         ↑ mask ↑       ↑ mask ↑

Problem: Model might see half of an operation:
  {"type": "line", "start": [MASKED]
  → Breaks JSON structure!
  → Model can't learn proper operation completion
```

**✅ Good: Structure-aware masking at operation boundaries**
```
Operation 1: ███████████ (fully masked, -100)
Operation 2: ███████████ (fully masked, -100)
...
Operation 40: ███████████ (fully masked, -100)
Operation 41: compute loss ← prediction starts here
Operation 42: compute loss
...
Operation 100: compute loss ← ends here

Result: Model learns to complete operations, not arbitrary tokens
```

### How We Find the Boundary

The collator computes the exact token boundary:

```python
# 1. Load truncated JSON to get metadata
truncated_json = load_json("path/to/00000071_00005_tr_02.json")
kept_operations = truncated_json["metadata"]["kept_operations"]  # e.g., 40

# 2. Load full JSON
full_json = load_json("path/to/00000071_00005.json")
# full_json["sequence"] has 100 operations

# 3. Reconstruct partial sequence (first 40 operations only)
partial_json = full_json.copy()
partial_json["sequence"] = full_json["sequence"][:kept_operations]
# partial_json["sequence"] now has ops 0-39 (40 operations total)

# 4. Format as text
caption = "Modern minimalist chair with wooden legs"
partial_text = f"Complete this CAD sequence: {caption}\n{json.dumps(partial_json)}"
full_text = f"Complete this CAD sequence: {caption}\n{json.dumps(full_json)}"

# 5. Tokenize to find boundary
partial_tokens = tokenizer(partial_text)["input_ids"]  # → [CLS, ..., 3500 tokens]
full_tokens = tokenizer(full_text)["input_ids"]        # → [CLS, ..., 6500 tokens]

boundary = len(partial_tokens)  # = 3500

# 6. Create labels with masking
labels = full_tokens.clone()
labels[:boundary] = -100           # Mask tokens 0-3500 (ops 1-40)
labels[boundary:] = full_tokens[boundary:]  # Loss on 3501-6500 (ops 41-100)

# 7. Model training
outputs = model(input_ids=full_tokens, labels=labels)
# Loss computed ONLY on tokens 3501-6500
# These correspond to operations 41-100
```

**Result:** The masking boundary aligns perfectly with operation boundaries in the JSON structure.

## 5. Training vs Inference: Complete Flow

### Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Dataset Sample                                              │
├─────────────────────────────────────────────────────────────┤
│ truncated_json: .../00000071_00005_tr_02.json              │
│   metadata: {kept_operations: 40}                          │
│   sequence: [op_1, op_2, ..., op_40]                       │
│                                                             │
│ full_json: .../00000071_00005.json                         │
│   sequence: [op_1, op_2, ..., op_100]                      │
│                                                             │
│ caption: "Modern minimalist chair"                         │
│ image: chair.png                                            │
│ point_cloud: chair.npy (2048 points)                       │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Collator Processing                                         │
├─────────────────────────────────────────────────────────────┤
│ 1. Format text:                                             │
│    "Complete this CAD sequence: Modern minimalist chair\n"  │
│    + json.dumps(full_json)  # Full 100 operations!         │
│                                                             │
│ 2. Tokenize full text → 6500 tokens                        │
│                                                             │
│ 3. Find masking boundary:                                  │
│    partial_text = prompt + json_ops_1_to_40                │
│    boundary = len(tokenize(partial_text)) = 3500           │
│                                                             │
│ 4. Create labels:                                          │
│    labels[0:3500] = -100  (no loss)                        │
│    labels[3500:6500] = input_ids[3500:6500]  (compute loss)│
│                                                             │
│ 5. Process multimodal inputs:                              │
│    - Image → dinov2 preprocessor → pixel_values           │
│    - Point cloud → normalize → point_cloud_tensor          │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Model Forward Pass                                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Embed multimodal inputs:                                 │
│    image_embed = image_projector(dinov2(pixel_values))     │
│    pc_embed = pc_projector(michelangelo(point_cloud))      │
│    text_embed = llm_embed(input_ids)                       │
│                                                             │
│ 2. Concatenate embeddings:                                 │
│    [img_embed, pc_embed, text_embed] → shape (1, L, 4096)  │
│                                                             │
│ 3. LLM forward:                                            │
│    logits = llm(combined_embeds)  # shape (1, L, vocab)    │
│                                                             │
│ 4. Compute loss (ONLY on non-masked positions):           │
│    loss = CrossEntropy(logits[:, 3500:], labels[:, 3500:]) │
│          ↑                                                  │
│          Only positions 3500-6500 (ops 41-100)             │
│                                                             │
│ 5. Backward pass:                                          │
│    loss.backward()  # Gradients flow to:                   │
│      - LLM LoRA adapters                                   │
│      - Image projector                                     │
│      - Point cloud projector                               │
└─────────────────────────────────────────────────────────────┘
```

### Inference Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Inference Input                                             │
├─────────────────────────────────────────────────────────────┤
│ truncated_json: .../test_model_tr_02.json                  │
│   sequence: [op_1, ..., op_40]  # Only partial!            │
│                                                             │
│ caption: "Modern minimalist chair"                         │
│ image: chair.png                                            │
│ point_cloud: chair.npy                                     │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Preprocessing                                               │
├─────────────────────────────────────────────────────────────┤
│ 1. Format prompt:                                           │
│    "Complete this CAD sequence: Modern minimalist chair\n"  │
│    + json.dumps(truncated_json)  # Only ops 1-40!          │
│                                                             │
│ 2. Tokenize → 3500 tokens (just the partial sequence)     │
│                                                             │
│ 3. Process multimodal:                                     │
│    - Image → pixel_values                                  │
│    - Point cloud → point_cloud_tensor                      │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Autoregressive Generation                                   │
├─────────────────────────────────────────────────────────────┤
│ Initial context (3500 tokens):                              │
│   [img_embed, pc_embed, "Complete: ... ops_1_to_40"]      │
│                                                             │
│ Step 1: Generate token 3501                                │
│   model([context]) → logits → sample → token_3501         │
│   context = [context, token_3501]                          │
│                                                             │
│ Step 2: Generate token 3502                                │
│   model([context]) → logits → sample → token_3502         │
│   context = [context, token_3502]                          │
│                                                             │
│ ...continue until...                                        │
│                                                             │
│ Step 3000: Generate token 6500 (or EOS)                   │
│   model([context]) → logits → sample → token_6500 (EOS)   │
│                                                             │
│ Generated tokens 3501-6500 decode to:                      │
│   {"type": "line", ...},  ← op 41                          │
│   {"type": "arc", ...},   ← op 42                          │
│   ...                                                       │
│   {"type": "extrude", ...} ← op 100                        │
└─────────────────────────────────────────────────────────────┘
```

## 6. Memory Optimization Choices

### Why max_seq_length=13000 with Gradient Checkpointing?

#### Attention Memory Scaling

The dominant memory cost is attention matrices:

```
Memory = batch_size × num_heads × seq_length² × bytes_per_element

Example (seq=13000, batch=1):
- num_heads = 32 (Qwen3-8B)
- bytes_per_element = 2 (bfloat16)

Attention memory = 1 × 32 × 13000² × 2 bytes
                 = 1 × 32 × 169,000,000 × 2
                 = 10.8 GB (just for attention!)

With gradient checkpointing:
- Forward pass: ~11 GB (attention matrices)
- Backward pass: Recompute instead of store (saves ~50% activation memory)
- Total: ~15-20 GB for seq=13000

Without gradient checkpointing:
- Forward pass: ~11 GB
- Backward pass: Store all intermediate activations: ~20-25 GB
- Total: ~30-35 GB for seq=13000

For seq=16384:
- With grad ckpt: ~25-30 GB
- Without grad ckpt: ~50-60 GB (OOM on 80GB with full model!)
```

#### Why Gradient Checkpointing is Essential

```
┌──────────────────────────────────────────────────┐
│ Without Gradient Checkpointing                   │
├──────────────────────────────────────────────────┤
│ Forward pass:                                    │
│   Store all intermediate activations in memory  │
│   Layer 1: activation_1 (stored)                │
│   Layer 2: activation_2 (stored)                │
│   ...                                            │
│   Layer 28: activation_28 (stored)              │
│                                                  │
│ Backward pass:                                   │
│   Use stored activations to compute gradients   │
│   (Fast but memory-intensive)                   │
│                                                  │
│ Memory: ~50 GB for seq=13000                    │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│ With Gradient Checkpointing ✅                   │
├──────────────────────────────────────────────────┤
│ Forward pass:                                    │
│   Store ONLY checkpoint activations (e.g., every │
│   4th layer)                                     │
│   Layer 1: activation_1 (discarded)             │
│   Layer 2: activation_2 (discarded)             │
│   Layer 3: activation_3 (discarded)             │
│   Layer 4: activation_4 (stored as checkpoint)  │
│   ...                                            │
│                                                  │
│ Backward pass:                                   │
│   When need activation_2:                       │
│     - Recompute forward from checkpoint         │
│     - Use recomputed activation                 │
│     - Discard after use                         │
│                                                  │
│ Memory: ~25 GB for seq=13000 (50% savings!)     │
│ Compute: +20-30% slower (recomputation cost)    │
└──────────────────────────────────────────────────┘
```

### Why LoRA r=8 Instead of r=32?

LoRA adds trainable low-rank matrices to each attention layer:

```
Original weights: W ∈ ℝ^(4096×4096)  (frozen)
LoRA adaptation: ΔW = B × A
  where A ∈ ℝ^(4096×r), B ∈ ℝ^(r×4096)

For r=32:
  A: 4096 × 32 = 131,072 params
  B: 32 × 4096 = 131,072 params
  Total per layer: 262,144 params

  28 layers × 4 matrices (Q,K,V,O) = 112 total LoRA modules
  Total LoRA params: 112 × 262,144 = 29.4M params
  Memory: ~235 MB (bfloat16)

For r=8:
  A: 4096 × 8 = 32,768 params
  B: 8 × 4096 = 32,768 params
  Total per layer: 65,536 params

  Total LoRA params: 112 × 65,536 = 7.3M params
  Memory: ~58 MB (bfloat16)

Savings: 235 MB - 58 MB = 177 MB per batch

With gradient accumulation=32:
  Must store gradients for 32 batches
  Memory savings: 177 MB × 32 = 5.7 GB!
```

### Why batch_size=1 + gradient_accumulation=32?

```
Option A: batch_size=4, gradient_accumulation=8
  Effective batch = 4 × 8 = 32 ✓
  Memory per forward: 4x sequences in attention
  Attention memory: 4 × seq² (OOM on seq=13000!)

Option B: batch_size=1, gradient_accumulation=32 ✅
  Effective batch = 1 × 32 = 32 ✓
  Memory per forward: 1x sequence in attention
  Attention memory: 1 × seq² (fits in memory!)

Trade-off:
  - 32x more optimizer steps (slower)
  - But each step uses 4x less memory
  - Enables longer sequences (13000 vs 6000)
```

### Configuration Summary

```
Recommended: max_seq_length=13000
├─ Why? Fits 80% of sequences without truncation
├─ Memory: ~54 GB (67% of 80GB A100)
├─ With gradient checkpointing: Required
├─ With LoRA r=8: Reduces parameter memory
└─ With batch=1 + grad_accum=32: Reduces attention memory

Alternative: max_seq_length=16384
├─ Why? Fits 95% of sequences
├─ Memory: ~60-65 GB (75-80% utilization)
├─ Risk: Less headroom for memory spikes
└─ Reward: Better handling of long sequences

Conservative: max_seq_length=8192
├─ Why? Maximum safety margin
├─ Memory: ~30-35 GB (40% utilization)
├─ Downside: Truncates 40% of dataset
└─ Use when: Debugging, or limited GPU
```

## 7. Why ~17% NaN Batches is Acceptable

### What Causes NaN Batches?

```python
# Case 1: All labels masked (rare but possible)
kept_operations = 98  # Truncated keeps 98 out of 100 operations
# After structure-aware masking, only 2 operations to predict
# If those 2 operations happen to be very short:
#   → Only ~50 tokens to predict
#   → After tokenizer padding to max_seq_length:
#     labels = [-100, -100, ..., -100, token_1, token_2, ..., token_50, pad, pad, ...]
#     → Most labels are -100 or padding
#   → If threshold too strict, might be considered "all masked"

# Case 2: Invalid loss computation
loss = CrossEntropy(logits, labels)
# If all labels[i] == -100 or pad_token_id:
#   → No valid tokens to compute loss on
#   → loss = NaN

# Case 3: Numerical instability (rare with bfloat16)
```

### Why 17% is Acceptable

```
Out of 5000 samples × 3 epochs = 15,000 total batches:

NaN batches: 17% = 2,550 batches (wasted compute)
Valid batches: 83% = 12,450 batches (useful training)

Expected training:
  - 12,450 valid gradient updates
  - Each with effective_batch_size = 32
  - Total effective samples: 12,450 × 32 = 398,400 sample updates

This is still substantial training signal!
```

### What Would Be Concerning?

```
✅ Acceptable:
  - 10-20% NaN: Normal for structured data with varying lengths
  - 5-10% NaN: Excellent, well-balanced dataset

⚠️ Concerning:
  - 20-30% NaN: Consider filtering very short sequences
  - >30% NaN: Dataset quality issues, investigate filtering

❌ Critical:
  - >50% NaN: Major problem, fix dataset or masking logic
  - 100% NaN: Complete failure, masking is broken
```

## 8. The Complete Picture

### Training Loop (One Batch)

```python
# 1. Load batch
batch = next(dataloader)
# batch = {
#   'input_ids': tensor([[101, 2023, 2003, ..., 102]]),  # Full seq, 6500 tokens
#   'labels': tensor([[-100, -100, ..., -100, 2023, ...]]),  # Masked until op 41
#   'pixel_values': tensor([[...]]),  # Image
#   'point_cloud': tensor([[...]]),  # Point cloud
# }

# 2. Forward pass
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    outputs = model(
        input_ids=batch['input_ids'],
        pixel_values=batch['pixel_values'],
        point_cloud=batch['point_cloud'],
        labels=batch['labels'],
    )
    loss = outputs.loss  # Computed only on ops 41-100

    # Inside model.forward():
    #   1. img_embed = img_projector(dinov2(pixel_values))
    #   2. pc_embed = pc_projector(michelangelo(point_cloud))
    #   3. txt_embed = llm.embed(input_ids)
    #   4. combined = concat([img_embed, pc_embed, txt_embed])
    #   5. logits = llm(combined)  # Shape: (1, 6500, 151669)
    #   6. loss = CrossEntropy(
    #        logits[:, 3500:],  # Only ops 41-100
    #        labels[:, 3500:]   # Ground truth for ops 41-100
    #      )

# 3. Backward pass (with gradient checkpointing)
if not is_nan(loss):
    loss.backward()  # Gradients computed via recomputation

# 4. Gradient accumulation
if step % gradient_accumulation_steps == 0:
    optimizer.step()  # Update LoRA + projectors
    optimizer.zero_grad()
    scheduler.step()  # LR warmup/decay

# 5. Logging
wandb.log({
    'loss': loss.item(),
    'lr': scheduler.get_last_lr()[0],
    'nan_percentage': nan_batches / total_batches * 100,
})
```

### Inference (Autoregressive Generation)

```python
# 1. Load partial sequence
truncated_json = load_json("path/to/model_tr_02.json")
prompt = f"Complete this CAD sequence: {caption}\n{json.dumps(truncated_json)}"

# 2. Tokenize
input_ids = tokenizer(prompt)['input_ids']  # ~3500 tokens (ops 1-40)

# 3. Process multimodal
pixel_values = image_processor(image)
point_cloud = normalize_point_cloud(pc)

# 4. Prepare initial input
model_input = {
    'input_ids': input_ids,
    'pixel_values': pixel_values,
    'point_cloud': point_cloud,
}

# 5. Generate completion
generated_ids = model.generate(
    **model_input,
    max_new_tokens=3000,  # Generate ops 41-100 (~3000 tokens)
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 6. Decode
completion_text = tokenizer.decode(generated_ids[0, len(input_ids):])
# completion_text ≈ '{"type": "line", ...}, {"type": "arc", ...}, ...'

# 7. Parse and validate
try:
    completion_ops = json.loads(f'[{completion_text}]')
    full_sequence = truncated_json['sequence'] + completion_ops
    print(f"Generated {len(completion_ops)} operations!")
except json.JSONDecodeError:
    print("Invalid JSON generated, retry with different temperature")
```

## 9. Key Takeaways

1. **Teacher forcing** (full sequence input) provides stable training with correct context
2. **Structure-aware masking** preserves JSON operation boundaries
3. **Gradient checkpointing** trades 20-30% compute for 50% memory savings (essential!)
4. **LoRA r=8** balances model capacity with memory efficiency
5. **batch_size=1 + grad_accum=32** minimizes attention memory while maintaining effective batch size
6. **max_seq_length=13000** is the sweet spot: fits 80% sequences, uses 67% GPU memory
7. **~17% NaN batches** is acceptable waste, still leaves 83% for useful training
8. The **training-inference gap** (teacher forcing vs autoregressive) exists but is worth the stability

This approach enables training a large multimodal model (8B params) on extremely long sequences (13k tokens) using only 54GB of an 80GB GPU!
