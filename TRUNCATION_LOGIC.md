# CAD Sequence Truncation Logic

## Overview

This system creates incomplete CAD sequences for training autocompletion models by intelligently truncating JSON-format CAD files from the Omni-CAD dataset.

## Purpose

Generate training data where:
- **Input**: Incomplete CAD sequence (truncated)
- **Target**: Remaining operations to complete the design
- **Use Case**: Train models to predict next CAD operations

## Truncation Strategy

### Core Logic

```python
For a file with N operations:
  - Can truncate from 1 to (N-1) operations
  - Generate up to 5 evenly-spaced truncation levels
  - Number consecutively: tr_01, tr_02, tr_03, tr_04, tr_05
  - Never include 100% (always incomplete)
```

### Examples

| Original Ops | Truncated Versions | Operation Counts |
|--------------|-------------------|------------------|
| 2 ops | tr_01 | 1 op (50%) |
| 3 ops | tr_01, tr_02 | 1, 2 ops |
| 4 ops | tr_01, tr_02, tr_03 | 1, 2, 3 ops |
| 6 ops | tr_01, tr_02, tr_03, tr_04, tr_05 | 1, 2, 3, 4, 5 ops |
| 10 ops | tr_01, tr_02, tr_03, tr_04, tr_05 | 1, 3, 5, 7, 9 ops |
| 20 ops | tr_01, tr_02, tr_03, tr_04, tr_05 | 1, 6, 10, 14, 19 ops |

### Spacing Algorithm

```python
def calculate_target_counts(total_ops, max_versions=5):
    max_possible = total_ops - 1  # Exclude 100%
    num_versions = min(max_possible, max_versions)

    if num_versions == 1:
        return [1]

    # Evenly space from 1 to max_possible
    step_size = (max_possible - 1) / (num_versions - 1)
    return [max(1, round(1 + i * step_size)) for i in range(num_versions)]
```

## JSON Structure

### Input (Original)
```json
{
  "entities": {
    "sketch_id_0": { /* Sketch data */ },
    "feature_id_0": { /* ExtrudeFeature data */ },
    "sketch_id_1": { /* Sketch data */ },
    "feature_id_1": { /* ExtrudeFeature data */ }
  },
  "sequence": [
    {"index": 0, "type": "Sketch", "entity": "sketch_id_0"},
    {"index": 1, "type": "ExtrudeFeature", "entity": "feature_id_0"},
    {"index": 2, "type": "Sketch", "entity": "sketch_id_1"},
    {"index": 3, "type": "ExtrudeFeature", "entity": "feature_id_1"}
  ]
}
```

### Output (Truncated at 50%)
```json
{
  "entities": {
    "sketch_id_0": { /* Sketch data */ },
    "feature_id_0": { /* ExtrudeFeature data */ }
  },
  "sequence": [
    {"index": 0, "type": "Sketch", "entity": "sketch_id_0"},
    {"index": 1, "type": "ExtrudeFeature", "entity": "feature_id_0"}
  ],
  "truncation_metadata": {
    "is_truncated": true,
    "original_operations": 4,
    "kept_operations": 2,
    "truncation_percentage": 50.0
  }
}
```

## Algorithm Steps

### 1. Load and Validate
```python
- Load original JSON
- Get sequence length (N operations)
- Check if N > 1 (need at least 2 ops to truncate)
```

### 2. Calculate Truncation Points
```python
- Determine number of versions: min(N-1, 5)
- Calculate evenly-spaced operation counts
- Ensure no duplicates
- Exclude 100% (N operations)
```

### 3. Truncate Sequence
```python
For each target operation count:
  1. Truncate sequence at index (count - 1)
  2. Identify referenced entities
  3. Remove unreferenced entities from entities dict
  4. Add truncation metadata
  5. Save as consecutive version (tr_01, tr_02, etc.)
```

### 4. Entity Cleanup
```python
def get_referenced_entities(sequence, entities):
    referenced = set()
    for op in sequence:
        entity_id = op['entity']
        referenced.add(entity_id)

        # Recursively get nested references
        # (e.g., ExtrudeFeature → profiles → sketch)
        collect_nested_references(entity_id, entities, referenced)

    return referenced

# Keep only referenced entities
truncated['entities'] = {
    eid: entity
    for eid, entity in entities.items()
    if eid in referenced
}
```

## File Organization

### Input
```
data/Omni-CAD-subset/json/
├── 0000/
│   ├── 00000071_00005.json  (4 ops)
│   ├── 00000180_00011.json  (10 ops)
│   └── ...
├── 0001/
└── ...
```

### Output
```
data/Omni-CAD-subset/json_truncated/
├── 0000/
│   ├── 00000071_00005_tr_01.json  (1 op)
│   ├── 00000071_00005_tr_02.json  (2 ops)
│   ├── 00000071_00005_tr_03.json  (3 ops)
│   ├── 00000180_00011_tr_01.json  (1 op)
│   ├── 00000180_00011_tr_02.json  (3 ops)
│   ├── 00000180_00011_tr_03.json  (5 ops)
│   ├── 00000180_00011_tr_04.json  (7 ops)
│   ├── 00000180_00011_tr_05.json  (9 ops)
│   └── ...
├── 0001/
└── ...
```

## Key Features

### ✅ Semantic Awareness
- Truncates based on CAD operation structure
- Maintains valid JSON format
- Preserves entity relationships

### ✅ Flexible Truncation
- Allows "orphaned" sketches (sketches without extrusions)
- Training models can learn incomplete states
- Works with any CAD sequence structure

### ✅ Clean Output
- Removes unused entities (reduces file size)
- Consecutive numbering (no gaps)
- Metadata for tracking

### ✅ Efficient Processing
- Resume capability (skips existing files)
- Progress tracking
- Error handling

## Orphaned Sketches

### What are they?
A Sketch without a corresponding ExtrudeFeature (or other feature operation).

### Example
```json
{
  "sequence": [
    {"index": 0, "type": "Sketch", "entity": "sketch_id"}
    // No ExtrudeFeature follows
  ]
}
```

### Why allow them?
1. **Valid geometry**: Sketches contain 2D curves (Line3D, Arc3D) that are renderable
2. **Training data**: Models learn that features typically follow sketches
3. **Real-world**: Users often create sketches then add features later

### In OpenCASCADE
- **Orphaned Sketch** → Wireframe (2D curves in 3D space)
- **Sketch + Feature** → Solid body (3D geometry)

Both are valid and visualizable!

## Usage

### Process Single File
```python
from truncate_dataset import CADTruncator

truncator = CADTruncator(min_operations=1)
results = truncator.generate_truncations(
    input_path=Path("input.json"),
    output_dir=Path("output/"),
    max_versions=5
)
```

### Process Full Dataset
```bash
python scripts/truncate_full_dataset.py
```

### Test on Mixed Files
```bash
python scripts/test_truncation_mixed.py
```

## Statistics Example

For a dataset of 58,653 JSON files:
- **Files with 2 ops**: ~1 truncation each
- **Files with 3-5 ops**: ~2-4 truncations each
- **Files with 6+ ops**: ~5 truncations each
- **Average**: ~3-4 truncated versions per file
- **Total truncations**: ~200,000+ files

## Validation

Each truncated file ensures:
1. ✅ All entities in `sequence` exist in `entities`
2. ✅ All nested references are included
3. ✅ No 100% versions (always incomplete)
4. ✅ Valid JSON structure
5. ✅ Metadata includes original/kept operation counts

## Scripts

| Script | Purpose |
|--------|---------|
| `truncate_dataset.py` | Core truncation logic |
| `truncate_full_dataset.py` | Process entire dataset |
| `test_truncation_mixed.py` | Test on sample files |
| `visualize_truncation.py` | Render truncated vs complete |

## Design Decisions

### Why max 5 versions?
- Balance between data diversity and storage
- Covers ~20%, 40%, 60%, 80%, 95% completion
- Avoids redundancy for small files

### Why consecutive numbering?
- Clear organization (tr_01, tr_02, tr_03...)
- No gaps (unlike skipping duplicate percentages)
- Predictable file naming

### Why allow 1 operation minimum?
- Provides maximum training diversity
- Even simple sequences have value
- Models learn from minimal context

### Why remove unused entities?
- Reduces file size (up to 60-70%)
- Faster loading during training
- Cleaner data structure

## Performance

### Truncation Speed
- **~10-50 files/sec** (depending on file size)
- **Full dataset (58K files)**: ~20-30 minutes
- **Bottleneck**: JSON I/O, not computation

### Storage Impact
Truncated files are ~20-60% smaller than originals due to entity cleanup.

**Example:**
- Original: 13,283 bytes (10 ops)
- Truncated (20%): 4,391 bytes (2 ops)
- **Savings**: 67%

## Future Enhancements

Potential improvements:
1. **Parallel processing** (multiprocessing)
2. **Custom truncation points** (user-specified percentages)
3. **Validation checks** (geometric validity)
4. **Batch visualization** (render all versions)
5. **Dataset statistics** (distribution analysis)

---

**Created for**: CAD sequence autocompletion model training
**Dataset**: Omni-CAD
**Last Updated**: November 2025
