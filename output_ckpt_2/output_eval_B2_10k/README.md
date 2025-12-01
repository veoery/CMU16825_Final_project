*Report generated: 2025-11-30*
*Version: CMU16825_Final_project/output_ckpt_2/output_eval_B2_10k*

### 1. Text Prompt to JSON
- on modal notebook `https://modal.com/notebooks/huiyenc/main/nb-pTP4TjPyhjuDzQTMxVV9Yo`

### 2. JSON to STEP Export
- `python scripts/export2step_progress.py --src output_ckpt_2/output_eval_B2_10k --form json -o output_ckpt_2/output_eval_B2_10k/step`
```
Total files:          10
Successfully exported: 8
Skipped (existing):    0
Errors:               2
```
### 3. STEP File Visualization
- `python visualize_step_files.py --input_dir ../output_ckpt_2/output_eval_B2_10k/step --resolution 1024 768`
- 📁 Images saved to: `/root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_2/output_eval_B2_10k/imgs`

### 4. Topology Evaluation
- `python run_topology_eval.py --input_dir ../output_ckpt_2/output_eval_B2_10k/step`
```
📁 Results saved to: /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_2/output_eval_B2_10k/topology_results/topology_results.json

📈 SUMMARY STATISTICS
================================================================================
Total Files:    8
Successful:     8 ✅
Failed:         0 ❌

Metrics (Successful files only):

  DangEL (Boundary Edge Length - lower is better):
    Mean: 0.000000
    Min:  0.000000
    Max:  0.000000

  SIR (Self-Intersection Ratio - lower is better, 0-1):
    Mean: 1.000000
    Min:  1.000000
    Max:  1.000000

  FluxEE (Flux Enclosure Error - lower is better, ~0 = closed):
    Mean: 0.000000
    Min:  0.000000
    Max:  0.000000

📄 Summary saved to: /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_2/output_eval_B2_10k/topology_results/topology_summary.json
```

### 5. JSON Structure Validation
- `python validate_json_structure.py --generated_dir ../output_ckpt_2/output_eval_B2_10k --gt_dir ../data/gt/test/json/0090 --output_dir output_ckpt_2/output_eval_B2_10k`

### 6. Sequence Metrics Evaluation
- `python eval_sequence_simple.py --generated_dir ../output_ckpt_2/output_eval_B2_10k --gt_dir ../data/gt/test/json --output_dir output_ckpt_2/output_eval_B2_10k`

python CMU16825_Final_project/evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img/json --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img/json


```
Total files evaluated: 14
✅ Successful:         14
⚠️  No GT:             0
❌ Errors:             0

🎯 ACCURACY METRICS:

  Entity Count Accuracy (lower = more entities in generated):
    Mean:   0.299
    Median: 0.422
    Min:    -2.000
    Max:    1.000

  Entity Type Sequence Accuracy (0-1, 1 = perfect match):
    Mean:   0.547
    Median: 0.486
    Min:    0.100
    Max:    1.000

  Type Distribution Similarity (0-1, 1 = identical distribution):
    Mean:   0.480
    Median: 0.400
    Min:    0.222
    Max:    1.000

================================================================================

📁 Results saved:
  - output_ckpt_2/output_eval_B2_10k/sequence_eval_results.json
  - output_ckpt_2/output_eval_B2_10k/sequence_eval_results.csv
```

## Overview

| Metric | Count |
|--------|-------|
| Total Text prompts | 14 |
| Total JSON files generated & Json validated | 14 |
| STEP files generated | 8 |
| OMNICAD JSON validation errors |  |
| STEP files missing | 2 |