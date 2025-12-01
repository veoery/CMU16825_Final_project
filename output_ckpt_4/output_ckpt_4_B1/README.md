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