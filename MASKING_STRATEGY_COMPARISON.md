# Masking Strategy: autocomplete_2 vs Current (autocomplete)

## Overview

The autocomplete branch has **significantly improved** masking strategy compared to autocomplete_2, with better entity handling and robustness.

---

## 🔴 OLD: autocomplete_2 Branch

### Masking Approach
**Simple Prompt-Based Masking**

```python
# Basic approach: Mask just the prompt
for i, prompt_ids in enumerate(prompt_encodings["input_ids"]):
    prompt_len = len(prompt_ids)
    labels[i, :prompt_len] = -100  # Mask entire prompt

# Mask padding
labels[labels == self.tokenizer.pad_token_id] = -100
```

### Problems
1. ❌ **No entity awareness** - Doesn't filter entities in truncated JSON
2. ❌ **Simple truncation** - Just counts operations, ignores entity dependencies
3. ❌ **Potential mismatch** - Truncated JSON might reference undefined entities
4. ❌ **Fixed truncation** - Uses `truncation=True` which can cut off mid-operation

### Input Format
```python
text = f"Complete this CAD sequence: {caption}\n{full_seq}"
# Masks just prompt portion, no entity filtering
```

---

## ✅ NEW: Current autocomplete Branch

### Masking Approach
**Entity-Aware Smart Masking**

```python
# Lines 379-440 in current multimodal_autocomplete.py

# 1. Build partial JSON with ONLY entities used in kept operations
for op in full_json["sequence"][:kept_operations]:
    if "entity" in op:
        used_entities.add(op["entity"])

partial_json["entities"] = {
    k: v for k, v in full_json.get("entities", {}).items()
    if k in used_entities  # ✅ Only include referenced entities!
}

# 2. Create partial text representation
partial_json_str = json.dumps(partial_json, separators=(',', ':'))
partial_text = prompt + partial_json_str

# 3. Tokenize WITHOUT truncation to get true length
partial_tokens = self.tokenizer(
    partial_text,
    add_special_tokens=False,
    truncation=False  # ✅ Don't cut off!
)["input_ids"]

# 4. Get actual sequence length (respects max_length)
actual_seq_len = (encodings["attention_mask"][i] == 1).sum().item()

# 5. Mask up to minimum of partial vs actual
mask_until = min(len(partial_tokens), actual_seq_len)
labels[i, :mask_until] = -100
```

### Improvements
1. ✅ **Entity filtering** - Only includes entities referenced in kept operations
2. ✅ **Robust tokenization** - Uses `truncation=False` to get true partial length
3. ✅ **Actual length awareness** - Respects the tokenized sequence's real length
4. ✅ **Smart masking boundary** - Takes minimum of partial vs actual to avoid over-masking
5. ✅ **Metadata preservation** - Copies other fields from full JSON
6. ✅ **Graceful fallback** - If JSON parsing fails, masks just the prompt

### Input Format
```python
# Same format, but with improved masking logic
text = f"Complete this CAD sequence: {sample['input_text']}\n{sample['full_seq']}"
# But masks with entity-aware partial JSON reconstruction
```

---

## 📊 Comparison Table

| Feature | autocomplete_2 | Current (autocomplete) |
|---------|----------------|------------------------|
| Entity filtering | ❌ No | ✅ Yes - only used entities |
| Truncation handling | ❌ Fixed truncation | ✅ No truncation for masking |
| Masking boundary | ❌ Prompt length only | ✅ Smart partial + actual length |
| Entity validation | ❌ Can reference undefined | ✅ Only defined entities |
| Error handling | ⚠️ Basic | ✅ Comprehensive fallback |
| Metadata preservation | ❌ No | ✅ Yes |
| Token length calculation | ❌ Simple | ✅ Actual attention mask aware |

---

## 🎯 Key Technical Differences

### 1. Entity Filtering (NEW in autocomplete)

**Problem in autocomplete_2:**
```json
{
  "sequence": [{"index": 0, "entity": "Sketch1"}, {"index": 1, "entity": "Extrude1"}],
  "entities": {
    "Sketch1": {...},
    "Extrude1": {...},
    "Extrude2": {...}  // ❌ Referenced but not in kept operations!
  }
}
```

**Fixed in autocomplete:**
```json
{
  "sequence": [{"index": 0, "entity": "Sketch1"}],
  "entities": {
    "Sketch1": {...}  // ✅ Only entity actually used in sequence[0]
  }
}
```

### 2. Tokenization Without Truncation (NEW)

**autocomplete_2:**
```python
partial_tokens = self.tokenizer(
    partial_text,
    max_length=self.max_seq_length,
    truncation=True  # ❌ Can cut off mid-operation
)
```

**autocomplete (current):**
```python
partial_tokens = self.tokenizer(
    partial_text,
    truncation=False  # ✅ Get true length, don't cut off
)
mask_until = min(len(partial_tokens), actual_seq_len)  # ✅ Use minimum
```

### 3. Actual Sequence Length Awareness (NEW)

**autocomplete_2:**
```python
mask_until = len(partial_tokens)  # ❌ Might exceed actual sequence
```

**autocomplete (current):**
```python
actual_seq_len = (encodings["attention_mask"][i] == 1).sum().item()
mask_until = min(len(partial_tokens), actual_seq_len)  # ✅ Never over-mask
```

---

## 🔬 Why These Changes Matter

### Example Scenario

**Input**: Truncated JSON with 3 operations, full JSON has 10 operations

**autocomplete_2 behavior:**
1. Tokenizes prompt: "Complete this CAD sequence: {caption}\n"
2. Masks just prompt tokens
3. Model learns to predict: `operation[3]` → `operation[10]`
4. But partial JSON might have entities for operations 0-9 (wrong!)

**autocomplete behavior:**
1. Builds partial JSON with ONLY operations 0-2 and their entities
2. Tokenizes partial text to get exact boundary
3. Masks prompt + operations 0-2
4. Model learns to predict: `operation[3]` → `operation[10]`
5. Partial JSON is valid and self-contained (correct!)

### Training Impact

**Better masking = Better training:**
- ✅ Model sees valid, self-contained partial sequences
- ✅ No confusion from undefined entity references
- ✅ Clearer task: complete from a valid starting point
- ✅ More robust to different sequence lengths

---

## 🚀 Recommendation

**Use the CURRENT autocomplete branch (not autocomplete_2)**

The current branch has:
1. ✅ Fixed the variable name bug
2. ✅ Improved masking strategy
3. ✅ Entity-aware truncation
4. ✅ Better error handling
5. ✅ Multimodal inference solution (LoRA merging)
6. ✅ Comprehensive documentation

---

## 📝 Other Notable Improvements in Current Branch

### 1. Checkpoint Management
- Timestamped checkpoint names: `checkpoint-step100-loss2.4567-20250129_143520`
- Keep last N checkpoints (cleanup old ones)
- Backward-compatible optimizer/scheduler restoration

### 2. Dataset Robustness
- Skip corrupted files gracefully
- Validate file existence before loading
- Filter outliers (>32KB files)

### 3. Inference
- New `CADAutocompleteMultimodal` class
- LoRA weight merging for multimodal inference
- Improved prompt format matching training

### 4. Documentation
- `MULTIMODAL_INFERENCE_SOLUTION.md`
- `TRAINING_IMPROVEMENTS.md`
- This comparison document

---

## Summary

**autocomplete_2 → autocomplete represents ~17 commits of improvements:**
- 🎯 Core fix: Entity-aware masking
- 🎯 Core fix: Variable name mismatch
- 🎯 Enhancement: Checkpoint management
- 🎯 Enhancement: Multimodal inference
- 🎯 Enhancement: Documentation

**You're already on the better branch!** ✅
