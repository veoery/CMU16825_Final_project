1. Basic Pipeline (Repair → Diagnose → Export)
python eval_process.py \
    --src output_ckpt_2/tmp_test/test_1_cube_3d_v2_repaired.json \
    --repair-output output_ckpt_2/tmp_test_output/test_1_cube_3d_v2_repaired_fix.json \
    --step-output output_ckpt_2/tmp_test_output/test_1_cube_3d_v2_repaired_fix_step

2. Full Pipeline with All Evaluations
python eval_process.py \
    --src gen_cad_all/v5_cylinder \
    --repair-output gen_cad_all/v5_cylinder_fixed_3 \
    --step-output gen_cad_all/v5_cylinder_step_3 \
    --evaluate \
    --eval-output gen_cad_all/v5_cylinder_eval_3 \
    --eval-seq \
    --eval-ae-acc \
    --h5-dir /path/to/reference/h5 \
    --eval-cd

3. Export Only (Skip repair/diagnose)
python eval_process.py \
    --src output_ckpt_2/tmp_test/test_1_cube_3d_v2_repaired.json \
    --repair-output output_ckpt_2/tmp_test_output \
    --step-output output_ckpt_2/tmp_test_output \
    --skip-repair \
    --skip-diagnose

python scripts/export2step_progress.py --src output_ckpt_2/tmp_test --form json -o output_ckpt_2/tmp_test/step

4. eval wihtout ground truth
python evaluate_with_groundtruth.py \
         --generated gen_cad_all/v6/v6_raw/generated_cad_00003816_00001_v6.json \
         --groundtruth data/reference_cad/00003816_00001.json \
         --output eval_result/eval_results.json

