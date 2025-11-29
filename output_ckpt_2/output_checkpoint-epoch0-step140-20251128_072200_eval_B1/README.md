*Report generated: 2025-11-29*
*Version: B1*

## Overview

| Metric | Count |
|--------|-------|
| Total Text prompts | 50 |
| Total JSON files generated & Json validated | 50 |
| STEP files generated | 44 |
| OMNICAD JSON validation errors | 3 |
| STEP files missing | 6 (incl 3 validation errors) |

### Topology Evaluation (44 STEP files)
- DangEL (Boundary edges): 0.0 (closed)
- SIR (Self-intersections): 1.0
- FluxEE (Enclosure error): 0.0 (enclosed)
- Result: All metrics pass

### Quick Summary
Generated 44 screenshots of STEP files. Limitations suspected:
- **Token Constraint**: looks like the primary shape is limited. But also might becuz of we are only generating text_prompts that ground truth json limited to 2048 tokens per text prompt, and has limited cad sequence generation. Sample text prompt below 2048, `"0090/00903454_00003": "Generate a CAD model with a cylindrical shape featuring a hollow center and a consistent diameter throughout. The model exhibits a smooth, uniform surface with no visible textures or patterns."`, tokens ard 1972, nothing much can do, constrains CAD sequence complexity

- **Shape Generation**: "cube, cylinde" etc can be identify but looks like the size not really can understand, but noted that OmniCAD doesnt incl the precise unit length in both the CAD seq or text_caption
- text only is lack of refrecen, although the basic shape correct but lack of precise guide to composite them hence the composite shape gone (hence no need to do chamfer distance)

- **Shape Generation**
- Topology well see (Topology Evaluation), basic primitives (cube, cylinder) recognized correctly. 
- However the composition and the params of primitives not gurantee. Might becuz dimensional accuracy limited—OmniCAD dataset does not encode precise unit lengths.
- Text-only prompts provide insufficient reference for complex composition & hence composite shapes lack precise spatial relationships

### Evaluation Impact
- Topology metrics (DangEL, SIR, FluxEE) reliably assess geometry
- Chamfer distance not applicable to composite shapes due to generation limitations
- Dimensional evaluation limited by model's ability to infer sizes from text

### CAD Sequence:
```
| Metric         | Score | Meaning                       |
|----------------|-------|-------------------------------|
| Entity Count   | 0.482 | ⚠️ Generates too few features |
| Sequence Order | 0.915 | ✅ Gets operation order right  |
| Type Mix       | 0.676 | ✅ Good proportions            |
```


## Known Issues

### JSON Files with Errors (3 files)

**`00900284_00001_repaired.json`**
- Missing `'sequence'` key
- Entities: 6 (Sketch, ExtrudeFeature)

**`00900920_00001_repaired.json`**
- ExtrudeFeature missing `'extent_type'` field (Extrude 1, Extrude 2)
- Entities: 4 (Sketch, ExtrudeFeature)

**`00902447_00001_repaired.json`**
- Missing `'sequence'` key
- Entities: Incomplete

---

## Evaluation Pipeline

### 1. Text Prompt to JSON
- Input: 50 text prompts
- Output: 50 valid JSON files

### 2. JSON to STEP Export
- Command: `python scripts/export2step_progress.py --src output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1 --form json -o output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/step`
- Result: 50 JSON → 44 STEP (6 errors)

### 3. STEP File Visualization
- Command: `python visualize_step_files.py --input_dir ../output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/step --resolution 1024 768`
- Result: 44 screenshots generated

### 4. Topology Evaluation
- Command: `python run_topology_eval.py --input_dir ../output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/step`
- Results:
  - DangEL (Boundary Edge Length): 0.0
  - SIR (Self-Intersection Ratio): 1.0
  - FluxEE (Enclosure Error): 0.0
- Output: `topology_results/topology_summary.json`

### 5. JSON Structure Validation
- Command: `python validate_json_structure.py --generated_dir ... --gt_dir ... --output_dir ...`
- Results:
  - Total files: 48
  - Valid: 48
  - Warnings: 0
  - Errors: 0
  - Ground truth matches: 16/48

### 6. Sequence Metrics Evaluation
- Command: `python eval_sequence_simple.py --generated_dir ... --gt_dir ... --output_dir ...`
```

================================================================================
📈 EVALUATION SUMMARY
================================================================================

Total files evaluated: 48
✅ Successful:         48
⚠️  No GT:             0
❌ Errors:             0

🎯 ACCURACY METRICS:

  Entity Count Accuracy (lower = more entities in generated):
    Mean:   0.482
    Median: 0.667
    Min:    -1.000
    Max:    1.000

  Entity Type Sequence Accuracy (0-1, 1 = perfect match):
    Mean:   0.915
    Median: 1.000
    Min:    0.250
    Max:    1.000

  Type Distribution Similarity (0-1, 1 = identical distribution):
    Mean:   0.676
    Median: 0.600
    Min:    0.333
    Max:    1.000

================================================================================

📁 Results saved:
  - ../output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/sequence_eval_results/sequence_eval_results.json
  - ../output_ckpt_2/output_checkpoint-epoch0-step140-20251128_072200_eval_B1/sequence_eval_results/sequence_eval_results.csv
```
```
📊 What These Numbers Mean

Entity Count Accuracy: 0.482 (mean)

- Measures how many entities your model generates vs. ground truth
- 0.482 = Your model generates on average 52% fewer entities than ground truth
- Range: -1.0 to 1.0
- 1.0 = Perfect match (same entity count)
- 0.0 = Off by 50%
- -1.0 = Way too many/few entities

Finding: Models tend to be simpler than ground truth—generating fewer features/sketches.

---
Entity Type Sequence Accuracy: 0.915 (mean) ⭐ Best result

- Checks if the ORDER of entity types matches (e.g., Sketch→Extrude→Sketch)
- 0.915 = Very good! 91.5% sequence correctness
- Range: 0 to 1 (1 = perfect)
- 1.0 = Exact sequence match
- 0.5 = Half the sequence matches
- 0.25 = Minimum (worst case in your data)

Finding: Model correctly learns what sequence of operations to perform, even if it uses fewer of them.

---
Type Distribution Similarity: 0.676 (mean)

- Checks if the mix of types is similar (e.g., 60% Sketches, 40% Extrudes)
- 0.676 = Good match on the mix of entity types
- Range: 0 to 1 (1 = identical distribution)

Finding: Entity type proportions are reasonably accurate.

---
🎯 Overall Assessment

| Metric         | Score | Meaning                       |
|----------------|-------|-------------------------------|
| Entity Count   | 0.482 | ⚠️ Generates too few features |
| Sequence Order | 0.915 | ✅ Gets operation order right  |
| Type Mix       | 0.676 | ✅ Good proportions            |

Bottom Line:
- ✅ Model learns correct CAD operation sequences
- ❌ But generates simpler models with fewer features
- → Aligns with your token constraint & dimensional limitation observations
```