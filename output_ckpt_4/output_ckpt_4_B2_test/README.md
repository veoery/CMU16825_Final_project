*Report generated: 2025-11-30*
*Version: CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B1*
*model: stage3-epoch0-step100-20251128_220651*

### 1. Text Prompt to JSON
`modal run batch_modal_inference.py     --num-samples 2 --folder 0090     --max-new-tokens 1024 --min-tokens 512`

### 2. JSON to STEP Export
- `python scripts/export2step_progress.py --src output_ckpt_2/output_eval_B2_10k --form json -o output_ckpt_2/output_eval_B2_10k/step`

### 3. STEP File Visualization
- `python visualize_step_files.py --input_dir ../output_ckpt_2/output_eval_B2_10k/step --resolution 1024 768`

### 4. Topology Evaluation
- `python run_topology_eval.py --input_dir ../output_ckpt_2/output_eval_B2_10k/step`

### 5. JSON Structure Validation
- `python validate_json_structure.py --generated_dir ../output_ckpt_2/output_eval_B2_10k --gt_dir ../data/gt/test/json/0090 --output_dir output_ckpt_2/output_eval_B2_10k`

### 6. Sequence Metrics Evaluation
- `python eval_sequence_simple.py --generated_dir ../output_ckpt_2/output_eval_B2_10k --gt_dir ../data/gt/test/json --output_dir output_ckpt_2/output_eval_B2_10k`


## Overview

| Metric | Count |
|--------|-------|
| Total Text prompts |  |
| Total JSON files generated & Json validated |  |
| STEP files generated |  |
| OMNICAD JSON validation errors |  |
| STEP files missing |  |

```
======================================================================
BATCH EVALUATION PIPELINE
======================================================================
Timestamp: 2025-11-30 18:55:44
Output versions: 5
======================================================================

[1/5] Processing: output_ckpt_4_B2_no_pc
----------------------------------------------------------------------
  JSON files: 2

  → JSON to STEP Export
    Command: python scripts/export2step_progress.py --src /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc --form json -o /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc/step
    ✓ Success

  → STEP File Visualization
    Command: python evaluation/visualize_step_files.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc/step --resolution 1024 768
    ✓ Success

  → Topology Evaluation
    Command: python evaluation/run_topology_eval.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc/step
    ✓ Success

  → JSON Structure Validation
    Command: python evaluation/validate_json_structure.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc
    ✓ Success

  → Sequence Metrics Evaluation
    Command: python evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc
    ✓ Success
  Result: 5/5 steps passed

[2/5] Processing: output_ckpt_4_T3_3img
----------------------------------------------------------------------
  JSON files: 1

  → JSON to STEP Export
    Command: python scripts/export2step_progress.py --src /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img --form json -o /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img/step
    ✓ Success

  → STEP File Visualization
    Command: python evaluation/visualize_step_files.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img/step --resolution 1024 768
    ✓ Success

  → Topology Evaluation
    Command: python evaluation/run_topology_eval.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img/step
    ✓ Success

  → JSON Structure Validation
    Command: python evaluation/validate_json_structure.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img
    ✓ Success

  → Sequence Metrics Evaluation
    Command: python evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T3_3img
    ✓ Success
  Result: 5/5 steps passed

[3/5] Processing: output_ckpt_4_T4_pc_txt
----------------------------------------------------------------------
  JSON files: 0

  → JSON to STEP Export
    Command: python scripts/export2step_progress.py --src /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt --form json -o /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt/step
    ✓ Success

  → STEP File Visualization
    Command: python evaluation/visualize_step_files.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt/step --resolution 1024 768
    ✓ Success

  → Topology Evaluation
    Command: python evaluation/run_topology_eval.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt/step
    ✓ Success

  → JSON Structure Validation
    Command: python evaluation/validate_json_structure.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt
    ✓ Success

  → Sequence Metrics Evaluation
    Command: python evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T4_pc_txt
    ✓ Success
  Result: 5/5 steps passed

[4/5] Processing: output_ckpt_4_T5_txt_only
----------------------------------------------------------------------
  JSON files: 3

  → JSON to STEP Export
    Command: python scripts/export2step_progress.py --src /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only --form json -o /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only/step
    ✓ Success

  → STEP File Visualization
    Command: python evaluation/visualize_step_files.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only/step --resolution 1024 768
    ✓ Success

  → Topology Evaluation
    Command: python evaluation/run_topology_eval.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only/step
    ✓ Success

  → JSON Structure Validation
    Command: python evaluation/validate_json_structure.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only
    ✓ Success

  → Sequence Metrics Evaluation
    Command: python evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only
    ✓ Success
  Result: 5/5 steps passed

[5/5] Processing: output_ckpt_4_T6_txt_img
----------------------------------------------------------------------
  JSON files: 1

  → JSON to STEP Export
    Command: python scripts/export2step_progress.py --src /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img --form json -o /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img/step
    ✓ Success

  → STEP File Visualization
    Command: python evaluation/visualize_step_files.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img/step --resolution 1024 768
    ✓ Success

  → Topology Evaluation
    Command: python evaluation/run_topology_eval.py --input_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img/step
    ✓ Success

  → JSON Structure Validation
    Command: python evaluation/validate_json_structure.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img
    ✓ Success

  → Sequence Metrics Evaluation
    Command: python evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T6_txt_img
    ✓ Success
  Result: 5/5 steps passed

======================================================================
EVALUATION SUMMARY
======================================================================

output_ckpt_4_B2_no_pc
  Status: ✓ Completed
  JSON files: 2
  Steps passed: 5/5

output_ckpt_4_T3_3img
  Status: ✓ Completed
  JSON files: 1
  Steps passed: 5/5

output_ckpt_4_T4_pc_txt
  Status: ✓ Completed
  JSON files: 0
  Steps passed: 5/5

output_ckpt_4_T5_txt_only
  Status: ✓ Completed
  JSON files: 3
  Steps passed: 5/5

output_ckpt_4_T6_txt_img
  Status: ✓ Completed
  JSON files: 1
  Steps passed: 5/5

======================================================================
Note: Check individual output directories for detailed results
======================================================================
```