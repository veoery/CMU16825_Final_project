
1. Test 1: CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B1
- text + img + pc

2. Test 2: CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_no_pc
- only text + 8 imgs
- ground truth bewlow 10204 but looks like ⚠️ JSON truncated, attempting auto-repair...
⚠️ Repair validation failed -> overflow genrated

### 3. Test 3: txt + 3 imgs + pc
- `modal run batch_modal_inference.py     --num-samples 2 --folder 0090     --max-new-tokens 10240 --min-tokens 1024`, max_tokens=4096
```
✓ [1] 00900119_00001: success
      JSON: /mnt/data/output_ckpt_4_T2_3img/json/00900119_00001_repaired_20251130_2155.json
      Raw:  /mnt/data/output_ckpt_4_T2_3img/raw/00900119_00001_repaired_20251130_2155.txt

⚠ [2] 00900120_00002: warning
      Raw:  /mnt/data/output_ckpt_4_T2_3img/raw/00900120_00002_repaired_20251130_2200.txt
```

### 4. Test 4: pc + text
```
⚠ [1] 00900119_00001: warning
      Raw:  /mnt/data/output_ckpt_4_T4_pc_txt/raw/00900119_00001_repaired_20251130_2211.txt

⚠ [2] 00900120_00002: warning
      Raw:  /mnt/data/output_ckpt_4_T4_pc_txt/raw/00900120_00002_repaired_20251130_2216.txt

⚠ [3] 00900124_00002: warning
      Raw:  /mnt/data/output_ckpt_4_T4_pc_txt/raw/00900124_00002_repaired_20251130_2221.txt
```

### 5. Test 5: Text only
```
✓ [1] 00900119_00001: success
      JSON: /mnt/data/output_ckpt_4_T5_txt_only/json/00900119_00001_repaired_20251130_2238.json
      Raw:  /mnt/data/output_ckpt_4_T5_txt_only/raw/00900119_00001_repaired_20251130_2238.txt

✓ [2] 00900120_00002: success
      JSON: /mnt/data/output_ckpt_4_T5_txt_only/json/00900120_00002_repaired_20251130_2239.json
      Raw:  /mnt/data/output_ckpt_4_T5_txt_only/raw/00900120_00002_repaired_20251130_2239.txt

✓ [3] 00900124_00002: success
      JSON: /mnt/data/output_ckpt_4_T5_txt_only/json/00900124_00002_repaired_20251130_2240.json
      Raw:  /mnt/data/output_ckpt_4_T5_txt_only/raw/00900124_00002_repaired_20251130_2240.txt
```

### 6. T6_txt_img
```
⚠ [1] 00900119_00001: warning
      Raw:  /mnt/data/output_ckpt_4_T6_txt_img/raw/00900119_00001_repaired_20251130_2305.txt

⚠ [2] 00900120_00002: warning
      Raw:  /mnt/data/output_ckpt_4_T6_txt_img/raw/00900120_00002_repaired_20251130_2310.txt

✓ [3] 00900124_00002: success
      JSON: /mnt/data/output_ckpt_4_T6_txt_img/json/00900124_00002_repaired_20251130_2315.json
      Raw:  /mnt/data/output_ckpt_4_T6_txt_img/raw/00900124_00002_repaired_20251130_2315.txt
```

<!-- python CMU16825_Final_project/evaluation/eval_sequence_simple.py --generated_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only/json --gt_dir /root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090 --output_dir /root/cmu/16825_l43d/CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only/json --pattern *repaired*.json -->