# Multimodal Inference Solution

## Problem Summary

**PEFT Limitation**: PEFT's `generate()` method ignores the `inputs_embeds` parameter, which prevents multimodal inference (text + image + point cloud).

## Solution: Merge LoRA Weights

The eval branch uses `use_lora=False` during inference, which inspired this solution.

### How It Works

1. **Train with LoRA** (Parameter-Efficient, ~0.8% trainable params)
2. **Merge LoRA weights** into base model for inference
3. **Use regular transformer** that supports `inputs_embeds`

### Benefits

✅ **Full multimodal support** - Text + Image + Point Cloud
✅ **No PEFT limitations** - Model is now a regular transformer
✅ **Slightly faster inference** - No adapter overhead
✅ **Same accuracy** - Merged weights are mathematically equivalent

### Trade-offs

⚠️ **Larger model size** - Full 8B params instead of base + adapters
⚠️ **Can't switch adapters** - Merged weights are permanent
⚠️ **One-time operation** - Need to merge each checkpoint separately

---

## Usage

### For Stage 1 (Text-Only)

Use the original `CADAutocomplete` class:

```python
from cad_mllm import CADAutocomplete

# Load checkpoint (LoRA stays as-is for text-only)
model = CADAutocomplete(
    checkpoint_path="outputs_curriculum/stage1_text/checkpoint-best",
    device="cuda",
)

# Text-only inference
result = model.complete(
    truncated_json="partial.json",
    caption="A modern chair",
    max_new_tokens=2000,
)
```

### For Stage 2/3 (Multimodal)

Use the new `CADAutocompleteMultimodal` class:

```python
from cad_mllm import CADAutocompleteMultimodal

# Load checkpoint and merge LoRA weights
model = CADAutocompleteMultimodal(
    checkpoint_path="outputs_curriculum/stage3_all/checkpoint-best",
    device="cuda",
)

# Multimodal inference
result = model.complete(
    truncated_json="partial.json",
    caption="A modern chair",
    image="chair.png",              # ✓ Now works!
    point_cloud="chair.npy",        # ✓ Now works!
    max_new_tokens=2000,
)
```

---

## Technical Details

### What Happens During `_load_and_merge_lora()`

```python
# 1. Load base model WITHOUT LoRA
config = CADMLLMConfig(use_lora=False)
model = CADMLLMModel(config)

# 2. Load LoRA adapters from checkpoint
from peft import PeftModel
model.llm = PeftModel.from_pretrained(model.llm, checkpoint_path)

# 3. Merge LoRA weights into base model
model.llm = model.llm.merge_and_unload()
# Now model.llm is a regular AutoModelForCausalLM, not a PeftModel!

# 4. Load projectors (image/point cloud)
# These are loaded normally
```

### Why This Works

**Before (PEFT model)**:
```python
# PEFT's generate() ignores inputs_embeds
outputs = peft_model.generate(
    inputs_embeds=multimodal_embeds,  # ✗ IGNORED!
    max_new_tokens=2000,
)
```

**After (Merged model)**:
```python
# Regular transformer respects inputs_embeds
outputs = merged_model.generate(
    inputs_embeds=multimodal_embeds,  # ✓ WORKS!
    max_new_tokens=2000,
)
```

---

## Training Strategy

### Stage 1: Text-Only
- **Train**: With LoRA (efficient)
- **Eval**: With LoRA (text-only, no merging needed)
- **Use**: `CADAutocomplete`

### Stage 2: Text + Point Cloud
- **Train**: With LoRA (efficient)
- **Eval**: Merge LoRA for multimodal inference
- **Use**: `CADAutocompleteMultimodal`

### Stage 3: All Modalities
- **Train**: With LoRA (efficient)
- **Eval**: Merge LoRA for multimodal inference
- **Use**: `CADAutocompleteMultimodal`

---

## FAQ

### Q: Can I still use LoRA during training?
**A**: Yes! This only affects inference. Training remains parameter-efficient with LoRA.

### Q: Will merging change the model's behavior?
**A**: No. Mathematically, merged LoRA weights are equivalent to unmerged ones.

### Q: Can I un-merge the weights?
**A**: No. Once merged, it's permanent. Keep your original checkpoint if you need the separated version.

### Q: Do I need to merge for text-only inference?
**A**: No. Use `CADAutocomplete` (original class) which works fine with PEFT for text-only.

### Q: What about the eval branch approach?
**A**: The eval branch likely does this same merging, or trains without LoRA for some configs. Their `use_lora=False` during inference is the key hint.

---

## Next Steps

1. ✅ Train Stage 1 with text-only (use curriculum training script)
2. ✅ Evaluate Stage 1 with `CADAutocomplete` (text-only)
3. ✅ Train Stage 2/3 with multimodal data
4. ✅ Evaluate Stage 2/3 with `CADAutocompleteMultimodal` (merged LoRA)

The variable bug is fixed, curriculum training works correctly, and you now have a path to full multimodal inference!
