#!/usr/bin/env python3
"""
Complete evaluation pipeline: Repair → Diagnose → Export → Evaluate

Runs the full CAD generation evaluation workflow in sequence:
1. Repair generated JSON files
2. Diagnose issues in repaired files
3. Export to STEP format
4. (Optional) Evaluate metrics using evaluation suite

Usage:
    # Basic pipeline (repair + diagnose + export)
    python eval_process.py \
        --src gen_cad_all/v5_cylinder \
        --repair-output gen_cad_all/v5_cylinder_fixed \
        --step-output gen_cad_all/v5_cylinder_step

    # Full pipeline with evaluation
    python eval_process.py \
        --src gen_cad_all/v5_cylinder \
        --repair-output gen_cad_all/v5_cylinder_fixed \
        --step-output gen_cad_all/v5_cylinder_step \
        --evaluate \
        --eval-output gen_cad_all/v5_cylinder_eval
"""

import subprocess
import argparse
import sys
import os
from pathlib import Path
import json
import glob


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 80)
    print(f"▶ {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete CAD evaluation pipeline: Repair → Diagnose → Export → Evaluate"
    )
    parser.add_argument("--src", type=str, required=True, help="Source directory with generated JSON files")
    parser.add_argument("--repair-output", type=str, required=True, help="Output directory for repaired files")
    parser.add_argument("--step-output", type=str, required=True, help="Output directory for STEP files")
    parser.add_argument("--skip-repair", action="store_true", help="Skip repair step (use existing repaired files)")
    parser.add_argument("--skip-diagnose", action="store_true", help="Skip diagnose step")
    parser.add_argument("--skip-export", action="store_true", help="Skip export step")
    parser.add_argument("--form", type=str, default="json", help="Input format for export (default: json)")

    # Evaluation arguments
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation metrics")
    parser.add_argument("--eval-output", type=str, help="Output directory for evaluation results")
    parser.add_argument("--eval-seq", action="store_true", help="Evaluate sequence accuracy (requires eval_seq.py)")
    parser.add_argument("--eval-ae-acc", action="store_true", help="Evaluate parameter accuracy (requires H5 files)")
    parser.add_argument("--eval-cd", action="store_true", help="Evaluate Chamfer distance")
    parser.add_argument("--h5-dir", type=str, help="Directory with H5 evaluation data")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔄 CAD EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Source:        {args.src}")
    print(f"Repair output: {args.repair_output}")
    print(f"STEP output:   {args.step_output}")
    print()

    results = {
        "repair": None,
        "diagnose": None,
        "export": None,
        "evaluate": None
    }

    # Validate evaluation arguments
    if args.evaluate and not args.eval_output:
        print("⚠️  Warning: --evaluate flag requires --eval-output. Skipping evaluation.")
        args.evaluate = False

    # Step 1: Repair
    if not args.skip_repair:
        cmd = [
            "python", "repair_cad_for_export.py",
            "--src", args.src,
            "-o", args.repair_output
        ]
        results["repair"] = run_command(cmd, "Step 1: Repair CAD JSON")
    else:
        print("\n⏭ Skipping repair step")
        results["repair"] = True

    if not results["repair"]:
        print("\n❌ Pipeline stopped: Repair failed")
        return 1

    # Step 2: Diagnose
    if not args.skip_diagnose:
        cmd = [
            "python", "diagnose_cad_issues.py",
            "--src", args.repair_output
        ]
        results["diagnose"] = run_command(cmd, "Step 2: Diagnose repaired files")
    else:
        print("\n⏭ Skipping diagnose step")
        results["diagnose"] = True

    # Step 3: Export to STEP
    if not args.skip_export:
        cmd = [
            "python", "scripts/export2step_progress.py",
            "--src", args.repair_output,
            "--form", args.form,
            "-o", args.step_output
        ]
        results["export"] = run_command(cmd, "Step 3: Export to STEP format")
    else:
        print("\n⏭ Skipping export step")
        results["export"] = True

    # Step 4: Evaluate (optional)
    if args.evaluate:
        # Convert JSON to H5 format for evaluation
        cmd_convert = [
            "python", "convert_json_for_evaluation.py",
            "--src", args.repair_output,
            "--output", args.eval_output
        ]
        if run_command(cmd_convert, "Step 4a: Convert JSON to H5 format"):
            # Run sequence evaluation
            if args.eval_seq:
                cmd_seq = [
                    "python", "evaluation/eval_seq.py",
                    "--src", args.eval_output,
                    "--output", os.path.join(args.eval_output, "seq_results")
                ]
                run_command(cmd_seq, "Step 4b: Evaluate sequence accuracy")

            # Run parameter accuracy evaluation
            if args.eval_ae_acc and args.h5_dir:
                cmd_acc = [
                    "python", "evaluation/eval_ae_acc.py",
                    "--src", args.eval_output,
                    "--ref", args.h5_dir,
                    "--output", os.path.join(args.eval_output, "accuracy_results")
                ]
                run_command(cmd_acc, "Step 4c: Evaluate parameter accuracy")

            # Run Chamfer distance evaluation
            if args.eval_cd:
                cmd_cd = [
                    "python", "evaluation/eval_ae_cd.py",
                    "--src", args.step_output,
                    "--output", os.path.join(args.eval_output, "cd_results")
                ]
                run_command(cmd_cd, "Step 4d: Evaluate Chamfer distance")

            results["evaluate"] = True
        else:
            results["evaluate"] = False
    else:
        print("\n⏭ Skipping evaluation step")
        results["evaluate"] = True

    # Final summary
    print("\n" + "=" * 80)
    print("📋 PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Repair:    {'✅ PASS' if results['repair'] else '❌ FAIL'}")
    print(f"Diagnose:  {'✅ PASS' if results['diagnose'] else '⏭ SKIPPED' if args.skip_diagnose else '❌ FAIL'}")
    print(f"Export:    {'✅ PASS' if results['export'] else '⏭ SKIPPED' if args.skip_export else '❌ FAIL'}")
    print(f"Evaluate:  {'✅ PASS' if results['evaluate'] else '⏭ SKIPPED' if not args.evaluate else '❌ FAIL'}")
    print("=" * 80)

    # Check if all steps passed
    if all(results.values()):
        print("\n✅ Pipeline completed successfully!")
        print(f"\nOutput locations:")
        print(f"  Repaired files:  {args.repair_output}")
        print(f"  STEP files:      {args.step_output}")
        if args.evaluate:
            print(f"  Evaluation data: {args.eval_output}")
        return 0
    else:
        print("\n❌ Pipeline completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
