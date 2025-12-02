---
license: apache-2.0
tags:
- cad
- autocomplete
- 3d
- multimodal
- qwen
library_name: transformers
pipeline_tag: text-generation
---

# CAD-MLLM Autocomplete Stage 3 (8K Context)

Fine-tuned Qwen3-8B model for CAD sequence autocompletion with 8000 token context.

## Model Info

- **Task**: CAD sequence autocompletion (partial → full sequence)
- **Context**: 8000 tokens (70% data coverage)
- **Base**: Qwen3-8B + LoRA (r=8, alpha=16)
- **Training**: 2000 samples, 10 epochs, LR=1.2e-4
- **Final Loss**: ~0.25

## Installation

```bash
# Clone the repository (autocomplete_2 branch)
git clone -b autocomplete_2 https://github.com/veoery/CMU16825_Final_project.git
cd CMU16825_Final_project

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Text-Only Inference

```python
from cad_mllm import CADAutocomplete

model = CADAutocomplete(checkpoint_path="YOUR_USERNAME/autocomplete-stage3-8000", device="cuda")

result = model.complete(
    truncated_json="path/to/truncated.json",
    caption="A mechanical part with mounting holes",
    max_new_tokens=2000,
    temperature=0.1
)

print(f"Generated {result['metadata']['generated_operations']} operations")
print(result['full_json'])
```

### Multimodal Inference

⚠️ **Important**: Use `CADAutocompleteMultimodal` for multimodal (merges LoRA weights)

```python
from cad_mllm import CADAutocompleteMultimodal

model = CADAutocompleteMultimodal(checkpoint_path="YOUR_USERNAME/autocomplete-stage3-8000", device="cuda")

result = model.complete(
    truncated_json="path/to/truncated.json",
    caption="A modern chair design",
    image_path="path/to/view.png",  # Optional
    pointcloud_path="path/to/pc.ply",  # Optional
    max_new_tokens=2000,
    temperature=0.1
)
```

## Input/Output Format

### Model Behavior

**What the model does**:
1. Takes a **truncated CAD sequence** (partial operations + their entities)
2. Generates a **complete CAD sequence** (all operations + all entities)
3. Output is the FULL JSON from scratch, not just appending to the truncated part

**Training**: The model was trained to see the truncated sequence as context and generate the complete sequence.

**Input** (Truncated JSON):
```json
{
  "sequence": [
    {"index": 0, "op": "NewSketchOnFace", "entity": "Sketch1"},
    {"index": 1, "op": "ExtrudeAdd", "entity": "Extrude1"}
  ],
  "entities": {
    "Sketch1": {"type": "Sketch", "plane": "XY", "curves": [...]},
    "Extrude1": {"type": "Extrude", "profile": "Sketch1", "distance": 10.0}
  }
}
```

**Output**: The model generates a **complete CAD JSON** (full sequence with all operations + entities).

**Example raw output**:
```
Here is the complete CAD sequence: {"sequence": [{"index": 0, ...}, {"index": 1, ...}, ...], "entities": {"Sketch1": {...}, ...}}
```

**After parsing** (extracts just the JSON):
```json
{
  "sequence": [
    {"index": 0, "op": "NewSketchOnFace", "entity": "Sketch1"},
    {"index": 1, "op": "ExtrudeAdd", "entity": "Extrude1"},
    ...
  ],
  "entities": {
    "Sketch1": {...},
    "Extrude1": {...},
    ...
  }
}
```

Use the parsing function below to extract the JSON.

## Parsing & Validation

```python
import json
import re

def parse_autocomplete_output(truncated_json, generated_text):
    """
    Parse model output and validate.

    Note: The model generates the COMPLETE JSON (not just continuation),
    but may include extra text. This function extracts just the JSON block.
    """
    # Extract JSON from generated text
    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if not json_match:
        return None, "No JSON found"

    try:
        generated_json = json.loads(json_match.group())

        # Validate structure
        if "sequence" not in generated_json or "entities" not in generated_json:
            return None, "Missing required fields"

        # Validate entity references
        used_entities = {op.get("entity") for op in generated_json["sequence"] if "entity" in op}
        for entity in used_entities:
            if entity not in generated_json["entities"]:
                return None, f"Entity {entity} referenced but not defined"

        return generated_json, None

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"

# Usage
full_json, error = parse_autocomplete_output(truncated_json, result['generated_text'])
if error:
    print(f"Parse failed: {error}")
else:
    print(f"✓ Parsed {len(full_json['sequence'])} operations")
```

## Validation

```python
def validate_cad_sequence(cad_json):
    """Validate CAD sequence structure."""
    issues = []

    # Check required fields
    if "sequence" not in cad_json:
        issues.append("Missing 'sequence'")
    if "entities" not in cad_json:
        issues.append("Missing 'entities'")

    if issues:
        return False, issues

    # Check sequence ordering
    for i, op in enumerate(cad_json["sequence"]):
        if op.get("index") != i:
            issues.append(f"Op {i} has wrong index")

    # Check entity references
    used_entities = {op.get("entity") for op in cad_json["sequence"] if "entity" in op}
    defined_entities = set(cad_json["entities"].keys())

    missing = used_entities - defined_entities
    if missing:
        issues.append(f"Undefined entities: {missing}")

    unused = defined_entities - used_entities
    if unused:
        issues.append(f"Unused entities: {unused}")

    # Check operation types
    valid_ops = {
        "NewSketchOnFace", "ExtrudeAdd", "ExtrudeCut",
        "FilletEdge", "ChamferEdge", "RevoluteAdd",
        "RevoluteCut", "PatternLinear", "PatternCircular", "Mirror"
    }
    for i, op in enumerate(cad_json["sequence"]):
        if op.get("op") not in valid_ops:
            issues.append(f"Unknown op type at {i}: {op.get('op')}")

    return len(issues) == 0, issues

# Usage
is_valid, issues = validate_cad_sequence(full_json)
if not is_valid:
    for issue in issues:
        print(f"  - {issue}")
```

## Evaluation Metrics

### 1. Operation Count Accuracy

```python
def eval_operation_count(gen_json, gt_json):
    gen_ops = len(gen_json["sequence"])
    gt_ops = len(gt_json["sequence"])
    return {
        "generated": gen_ops,
        "expected": gt_ops,
        "accuracy": gen_ops / gt_ops if gt_ops > 0 else 0
    }
```

### 2. Entity Consistency

```python
def eval_entity_consistency(gen_json):
    used = {op.get("entity") for op in gen_json["sequence"] if "entity" in op}
    defined = set(gen_json["entities"].keys())

    all_defined = used.issubset(defined)
    no_unused = used == defined

    return {
        "all_entities_defined": all_defined,
        "no_unused_entities": no_unused,
        "score": 1.0 if (all_defined and no_unused) else 0.0
    }
```

### 3. Continuation Accuracy

```python
def eval_continuation(gen_json, gt_json, truncated_ops):
    """Measure accuracy of continuation after truncation point."""
    gen_cont = gen_json["sequence"][truncated_ops:]
    gt_cont = gt_json["sequence"][truncated_ops:]

    matches = sum(
        1 for i in range(min(len(gen_cont), len(gt_cont)))
        if gen_cont[i]["op"] == gt_cont[i]["op"]
    )

    return {
        "matched_ops": matches,
        "total_expected": len(gt_cont),
        "accuracy": matches / len(gt_cont) if gt_cont else 0
    }
```

### Complete Evaluation Script

```python
from cad_mllm import CADAutocomplete
from tqdm import tqdm

def run_evaluation(checkpoint_path, test_samples, num_samples=100):
    model = CADAutocomplete(checkpoint_path=checkpoint_path, device="cuda")

    results = {
        "valid_json": 0,
        "total": 0,
        "op_counts": [],
        "entity_scores": [],
        "cont_accuracies": [],
        "errors": []
    }

    for sample in tqdm(test_samples[:num_samples]):
        try:
            result = model.complete(
                truncated_json=sample["truncated_path"],
                caption=sample["caption"],
                max_new_tokens=2000,
                temperature=0.1
            )

            full_json, error = parse_autocomplete_output(
                sample["truncated_json"],
                result["generated_text"]
            )

            if error:
                results["errors"].append({"sample": sample["id"], "error": error})
                continue

            is_valid, _ = validate_cad_sequence(full_json)
            if is_valid:
                results["valid_json"] += 1

            results["op_counts"].append(
                eval_operation_count(full_json, sample["gt"])
            )
            results["entity_scores"].append(
                eval_entity_consistency(full_json)
            )
            results["cont_accuracies"].append(
                eval_continuation(full_json, sample["gt"], sample["truncated_ops"])
            )

            results["total"] += 1

        except Exception as e:
            results["errors"].append({"sample": sample.get("id"), "error": str(e)})

    # Aggregate metrics
    results["valid_json_rate"] = results["valid_json"] / results["total"]
    results["avg_cont_acc"] = sum(
        m["accuracy"] for m in results["cont_accuracies"]
    ) / len(results["cont_accuracies"]) if results["cont_accuracies"] else 0

    print(f"Valid JSON Rate: {results['valid_json_rate']:.2%}")
    print(f"Avg Continuation Acc: {results['avg_cont_acc']:.2%}")

    return results
```

## Troubleshooting

### Model generates text instead of JSON
**Solution**: Lower temperature
```python
result = model.complete(..., temperature=0.05, top_p=0.9)
```

### Incomplete JSON output
**Solution**: Increase max_new_tokens
```python
result = model.complete(..., max_new_tokens=4000)
```

### Multimodal PEFT error
**Solution**: Use `CADAutocompleteMultimodal` (merges LoRA)
```python
from cad_mllm import CADAutocompleteMultimodal
model = CADAutocompleteMultimodal(...)
```

### Entity reference errors
**Solution**: Post-process to filter invalid entities
```python
def fix_entities(cad_json):
    defined = set(cad_json["entities"].keys())
    cad_json["sequence"] = [
        op for op in cad_json["sequence"]
        if "entity" not in op or op["entity"] in defined
    ]
    return cad_json
```

## Training Details

- **Dataset**: 2000 CAD sequences (autocomplete_2 masking strategy)
- **Median length**: 4,464 tokens (70% fit in 8K context)
- **LR scaling**: 1.2e-4 (from base 2e-4 via sqrt scaling for 2x context)
- **Batch**: 2 × 8 grad_accum = 16 effective batch size
- **Warmup**: 200 steps
- **Loss curve**: 0.6 → 0.25 over 2-3 epochs

## Comparison: 4K vs 8K

| Metric | 4K Version | 8K (This) |
|--------|-----------|-----------|
| Context | 4096 | 8000 |
| Coverage | 46% | 70% |
| LR | 2e-4 | 1.2e-4 |
| Loss | ~0.22 | ~0.25 |
| Speed | Faster | Slower |
| Use Case | General | Long sequences |

## Citation

```bibtex
@misc{cad-mllm-autocomplete-8k,
  title={CAD-MLLM Autocomplete Model (8K Context)},
  year={2025},
  url={https://huggingface.co/YOUR_USERNAME/autocomplete-stage3-8000}
}
```

## License

Apache 2.0

## Code Repository

This model was trained using the code from:
- **Repository**: [veoery/CMU16825_Final_project](https://github.com/veoery/CMU16825_Final_project)
- **Branch**: `autocomplete_2`
- **Key Features**: Entity-aware masking strategy, curriculum learning, multimodal inference support

## Acknowledgments

- Base model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- Inspired by: [autocomplete-stage3-4096](https://huggingface.co/omnicad-lab-L3d/autocomplete-stage3-4096)
- Training framework: PyTorch, HuggingFace Transformers, PEFT
