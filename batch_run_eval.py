#!/usr/bin/env python3
"""
Batch Evaluation Pipeline for Multiple Output Versions

Runs the full evaluation pipeline (steps 1-7) for each output version:
1. Text Prompt to JSON (already done - outputs exist)
2. JSON to STEP Export
3. STEP File Visualization
4. Topology Evaluation
5. JSON Structure Validation
6. Sequence Metrics Evaluation (JSON-based)
7. CAD Sequence Evaluation (Pickle-based, if available)

Usage:
    python batch_run_eval.py
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Configuration
OUTPUT_VERSIONS = [
    "output_ckpt_2/output_eval_B2_10k",
    "output_ckpt_2/output_eval_B1_2048",
    "output_ckpt_4/output_ckpt_4_B1",
    "output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc",
    "output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only"
    "output_ckpt_4/output_ckpt_4_B3_txt_img",
    "output_ckpt_4/output_ckpt_4_B4_pc_txt_3img",
    "output_ckpt_4/output_ckpt_4_B4_txt_img_fix",
    "output_ckpt_4/output_ckpt_4_B4_pc_txt_3img_fix",
    "output_ckpt_4/output_ckpt_4_B4_txt_img_fix"
]

# Add a function, append all the result of output version to a csv file to do visualizae. do plot matlib graph too.
# output_all_csv = "/root/cmu/16825_l43d/CMU16825_Final_project/output_all.csv"
# columns = OUTPUT_VERSIONS, len(raw_text files), len(json to step exports), no. of png visualization saved, number of success topology (each metrics DangEL, SIR, FluxEE), json validation, seq metrics (entity count acc, type seq acc, type dist sim),best seq metrics file id

BASE_OUTPUT_DIR = "/root/cmu/16825_l43d/CMU16825_Final_project"
SCRIPTS_DIR = "/root/cmu/16825_l43d/CMU16825_Final_project/scripts"
GT_JSON_DIR = "/root/cmu/16825_l43d/CMU16825_Final_project/data/gt/test/json/0090"

# Results tracking
results = {}


def run_command(cmd, description, cwd=None):
    """Run a shell command and return success status."""
    print(f"\n  → {description}")
    print(f"    Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"    ✓ Success")
            return True
        else:
            print(f"    ✗ Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    ✗ Timeout (10 minutes)")
        return False
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        return False


def step2_json_to_step(output_version):
    """Step 2: Export JSON to STEP files."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    step_output_dir = os.path.join(src_dir, "step")

    cmd = [
        "python", "scripts/export2step_progress.py",
        "--src", src_dir,
        "--form", "json",
        "-o", step_output_dir
    ]

    cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
    success = run_command(cmd, f"JSON to STEP Export", cwd=cwd)

    # Post-process: flatten STEP files from step/json/ to step/
    if success:
        json_step_dir = os.path.join(step_output_dir, "json")
        if os.path.exists(json_step_dir):
            import shutil
            step_files = [f for f in os.listdir(json_step_dir) if f.endswith('.step')]
            for step_file in step_files:
                src_path = os.path.join(json_step_dir, step_file)
                dst_path = os.path.join(step_output_dir, step_file)
                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
            print(f"    ✓ Flattened {len(step_files)} STEP files from step/json/ to step/")

    return success


def step3_visualize_step(output_version):
    """Step 3: Visualize STEP files."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    step_dir = os.path.join(src_dir, "step")

    if not os.path.exists(step_dir):
        print(f"  → STEP Visualization: Skipped (no step/ directory)")
        return None

    cmd = [
        "python", "evaluation/visualize_step_files.py",
        "--input_dir", step_dir,
        "--resolution", "1024", "768"
    ]

    cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
    return run_command(cmd, f"STEP File Visualization", cwd=cwd)


def step4_topology_eval(output_version):
    """Step 4: Topology evaluation."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    step_dir = os.path.join(src_dir, "step")

    if not os.path.exists(step_dir):
        print(f"  → Topology Evaluation: Skipped (no step/ directory)")
        return None

    cmd = [
        "python", "evaluation/run_topology_eval.py",
        "--input_dir", step_dir
    ]

    cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
    return run_command(cmd, f"Topology Evaluation", cwd=cwd)


def step5_json_validation(output_version):
    """Step 5: JSON structure validation."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)

    cmd = [
        "python", "evaluation/validate_json_structure.py",
        "--generated_dir", src_dir,
        "--gt_dir", GT_JSON_DIR,
        "--output_dir", src_dir,
        "--pattern", "*repaired*.json"
    ]

    cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
    return run_command(cmd, f"JSON Structure Validation", cwd=cwd)


def step6_sequence_metrics(output_version):
    """Step 6: Sequence metrics evaluation (JSON-based)."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    src_dir = os.path.join(src_dir,"json")

    cmd = [
        "python", "evaluation/eval_sequence_simple.py",
        "--generated_dir", src_dir,
        "--gt_dir", GT_JSON_DIR,
        "--output_dir", src_dir,
        "--pattern", "*repaired*.json"
    ]

    cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
    return run_command(cmd, f"Sequence Metrics Evaluation", cwd=cwd)


def find_pickle_files(output_version):
    """Find pickle files in output directory."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)

    # Common locations for pickle files
    potential_paths = [
        os.path.join(src_dir, "predictions.pkl"),
        os.path.join(src_dir, "results.pkl"),
        os.path.join(src_dir, "predictions"),
        os.path.join(src_dir, "pkl"),
    ]

    found_files = []
    for path in potential_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                found_files.append(path)
            elif os.path.isdir(path):
                # Search in directory
                pkl_files = [
                    os.path.join(path, f) for f in os.listdir(path)
                    if f.endswith('.pkl')
                ]
                found_files.extend(pkl_files)

    return found_files


def step7_cad_sequence_eval(output_version, use_validate_only=True):
    """
    Step 7: CAD sequence evaluation (Pickle-based).

    Args:
        output_version: Output version name
        use_validate_only: If True, only validate data format (fast, no CadSeqProc needed)
                          If False, perform full evaluation (slow, requires CadSeqProc)
    """
    # Find pickle files
    pkl_files = find_pickle_files(output_version)

    if not pkl_files:
        print(f"  → CAD Sequence Evaluation: Skipped (no pickle files found)")
        return None

    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    output_dir = os.path.join(src_dir, "cad_sequence_eval")

    results = []
    for pkl_file in pkl_files:
        filename = os.path.basename(pkl_file)
        cmd = [
            "python", "evaluation/eval_seq.py",
            "--input_path", pkl_file,
            "--output_dir", output_dir
        ]

        # Add --validate_only flag for data validation mode
        if use_validate_only:
            cmd.append("--validate_only")

        cwd = "/root/cmu/16825_l43d/CMU16825_Final_project"
        mode_desc = "validation" if use_validate_only else "full evaluation"
        success = run_command(cmd, f"CAD Sequence Evaluation ({mode_desc}): {filename}", cwd=cwd)
        results.append(success)

    # Return True if all succeeded, False if any failed, None if skipped
    return all(results) if results else None


def load_json_safe(filepath):
    """Safely load JSON file, return empty dict if not found."""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def format_metrics(output_version):
    """Extract and format metrics from evaluation outputs."""
    src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
    metrics = {}

    # Raw text files (inference outputs)
    raw_dir = os.path.join(src_dir, "raw")
    raw_files = len([f for f in os.listdir(raw_dir) if f.endswith('.txt')]) if os.path.exists(raw_dir) else 0
    metrics['raw_files'] = raw_files

    # Step 2: Count STEP files
    step_dir = os.path.join(src_dir, "step")
    step_files = len([f for f in os.listdir(step_dir) if f.endswith('.step')]) if os.path.exists(step_dir) else 0
    metrics['step_files'] = step_files

    # Step 3: Count PNG visualizations
    # Images are saved to imgs/ subdirectory, not step/
    imgs_dir = os.path.join(src_dir, "imgs")
    png_files = len([f for f in os.listdir(imgs_dir) if f.endswith('.png')]) if os.path.exists(imgs_dir) else 0
    metrics['png_visualizations'] = png_files

    # Step 4: Topology metrics
    topo_summary = load_json_safe(os.path.join(src_dir, "topology_results", "topology_summary.json"))
    if topo_summary:
        metrics['topology'] = {
            'total': topo_summary.get('total_files', 0),
            'successful': topo_summary.get('successful', 0),
            'failed': topo_summary.get('failed', 0),
            'metrics': topo_summary.get('metrics', {})
        }

    # Step 5: JSON validation
    json_val_summary = load_json_safe(os.path.join(src_dir, "json_validation_summary.json"))
    if json_val_summary:
        metrics['json_validation'] = {
            'total': json_val_summary.get('total', 0),
            'valid': json_val_summary.get('valid', 0),
            'warnings': json_val_summary.get('warnings', 0),
            'errors': json_val_summary.get('errors', 0)
        }

    # Step 6: Sequence metrics (JSON-based)
    # Results are saved in json/ subdirectory since that's where --output_dir points
    json_subdir = os.path.join(src_dir, "json")
    seq_csv = os.path.join(json_subdir, "sequence_eval_results.csv")
    seq_json = load_json_safe(os.path.join(json_subdir, "sequence_eval_results.json"))
    metrics['sequence'] = {
        'has_csv': os.path.exists(seq_csv) and os.path.getsize(seq_csv) > 2,
        'has_json': isinstance(seq_json, list) and len(seq_json) > 0,
        'results': seq_json if isinstance(seq_json, list) else []
    }

    # Step 7: CAD Sequence metrics (Pickle-based)
    cad_seq_dir = os.path.join(src_dir, "cad_sequence_eval")
    cad_seq_results = {}

    if os.path.exists(cad_seq_dir):
        # Look for level-based evaluation results (level_1, level_2, etc.)
        for level in range(1, 5):
            level_dir = os.path.join(cad_seq_dir, f"level_{level}")
            if os.path.exists(level_dir):
                mean_report = load_json_safe(os.path.join(level_dir, f"mean_report_level_{level}.json"))
                if mean_report:
                    cad_seq_results[f"level_{level}"] = mean_report

    metrics['cad_sequence'] = {
        'has_results': len(cad_seq_results) > 0,
        'levels': cad_seq_results
    }

    return metrics


def print_detailed_results(metrics):
    """Print detailed evaluation results with metrics."""
    raw_files = metrics.get('raw_files', 0)
    print(f"\n  📊 EVALUATION METRICS: ({raw_files} raw text files generated)")

    # Step 2
    step_files = metrics.get('step_files', 0)
    print(f"    [Step 2] JSON to STEP: {step_files} files exported")

    # Step 3
    png_count = metrics.get('png_visualizations', 0)
    if png_count > 0:
        print(f"    [Step 3] Visualization: {png_count} PNGs saved")
    else:
        print(f"    [Step 3] Visualization: (no output)")

    # Step 4: Topology
    topo = metrics.get('topology', {})
    if topo:
        total = topo.get('total', 0)
        successful = topo.get('successful', 0)
        metrics_data = topo.get('metrics', {})
        print(f"    [Step 4] Topology: {successful}/{total} successful")
        if metrics_data:
            danglel = metrics_data.get('DangEL', {}).get('mean', 'N/A')
            sir = metrics_data.get('SIR', {}).get('mean', 'N/A')
            fluxee = metrics_data.get('FluxEE', {}).get('mean', 'N/A')
            if isinstance(danglel, (int, float)):
                danglel = f"{danglel:.2e}"
            if isinstance(sir, (int, float)):
                sir = f"{sir:.4f}"
            if isinstance(fluxee, (int, float)):
                fluxee = f"{fluxee:.2e}"
            print(f"              • DangEL (boundary edge): {danglel}")
            print(f"              • SIR (self-intersection): {sir}")
            print(f"              • FluxEE (enclosure error): {fluxee}")

    # Step 5: JSON validation
    json_val = metrics.get('json_validation', {})
    if json_val:
        total = json_val.get('total', 0)
        valid = json_val.get('valid', 0)
        errors = json_val.get('errors', 0)
        warnings = json_val.get('warnings', 0)
        if total > 0:
            print(f"    [Step 5] JSON Validation: {valid}/{total} valid")
            if errors > 0:
                print(f"              • Errors: {errors}")
            if warnings > 0:
                print(f"              • Warnings: {warnings}")
        else:
            print(f"    [Step 5] JSON Validation: (no GT data to compare)")

    # Step 6: Sequence metrics (JSON-based)
    seq = metrics.get('sequence', {})
    if seq.get('has_csv') or seq.get('has_json'):
        results = seq.get('results', [])
        if results:
            # Calculate average metrics
            successful = [r for r in results if r.get('status') == 'success']
            if successful:
                entity_count_accs = [r['metrics']['entity_count_acc'] for r in successful]
                type_seq_accs = [r['metrics']['entity_type_sequence_acc'] for r in successful]
                type_dist_sims = [r['metrics']['type_distribution_sim'] for r in successful]

                print(f"    [Step 6] Sequence Metrics (JSON): {len(successful)} files evaluated")
                print(f"              • Entity Count Accuracy: {sum(entity_count_accs)/len(entity_count_accs):.3f}")
                print(f"              • Type Sequence Accuracy: {sum(type_seq_accs)/len(type_seq_accs):.3f}")
                print(f"              • Type Distribution Similarity: {sum(type_dist_sims)/len(type_dist_sims):.3f}")
            else:
                print(f"    [Step 6] Sequence Metrics (JSON): completed ({len(results)} files)")
        else:
            print(f"    [Step 6] Sequence Metrics (JSON): evaluation completed")
    else:
        print(f"    [Step 6] Sequence Metrics (JSON): (no results)")

    # Step 7: CAD Sequence metrics (Pickle-based)
    cad_seq = metrics.get('cad_sequence', {})
    if cad_seq.get('has_results'):
        levels = cad_seq.get('levels', {})
        print(f"    [Step 7] CAD Sequence Metrics (Pickle): {len(levels)} levels evaluated")
        for level_name, level_data in levels.items():
            if isinstance(level_data, dict):
                # Display Line metrics
                line_metrics = level_data.get('line', {})
                if line_metrics:
                    print(f"              {level_name} - Line:")
                    print(f"                • Recall: {line_metrics.get('recall', 'N/A'):.2f}%")
                    print(f"                • Precision: {line_metrics.get('precision', 'N/A'):.2f}%")
                    print(f"                • F1: {line_metrics.get('f1', 'N/A'):.2f}%")

                # Display Extrusion metrics
                ext_metrics = level_data.get('extrusion', {})
                if ext_metrics:
                    print(f"              {level_name} - Extrusion:")
                    print(f"                • Recall: {ext_metrics.get('recall', 'N/A'):.2f}%")
                    print(f"                • Precision: {ext_metrics.get('precision', 'N/A'):.2f}%")
                    print(f"                • F1: {ext_metrics.get('f1', 'N/A'):.2f}%")

                # Display Chamfer Distance
                cd_metrics = level_data.get('cd', {})
                if cd_metrics:
                    print(f"              {level_name} - Chamfer Distance:")
                    print(f"                • Mean: {cd_metrics.get('mean', 'N/A'):.4f}")
                    print(f"                • Median: {cd_metrics.get('median', 'N/A'):.4f}")
    else:
        print(f"    [Step 7] CAD Sequence Metrics (Pickle): (no results)")


def save_results_summary(results, output_csv):
    """Save evaluation results summary to CSV and generate plots."""
    import csv
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        print("  Warning: matplotlib not available, skipping plots")

    # Prepare data for CSV
    rows = []
    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        if result.get("status") == "completed":
            metrics = result.get("metrics", {})

            # Raw files
            raw_files = metrics.get("raw_files", 0)

            # Step 2: STEP files
            step_files = metrics.get("step_files", 0)

            # Step 3: PNG visualizations
            png_count = metrics.get("png_visualizations", 0)

            # Step 4: Topology metrics
            topo = metrics.get("topology", {})
            topo_total = topo.get("total", 0)
            topo_success = topo.get("successful", 0)
            topo_metrics = topo.get("metrics", {})
            danglel = topo_metrics.get("DangEL", {}).get("mean", "N/A")
            sir = topo_metrics.get("SIR", {}).get("mean", "N/A")
            fluxee = topo_metrics.get("FluxEE", {}).get("mean", "N/A")

            # Step 5: JSON validation
            json_val = metrics.get("json_validation", {})
            json_total = json_val.get("total", 0)
            json_valid = json_val.get("valid", 0)

            # Step 6: Sequence metrics
            seq = metrics.get("sequence", {})
            seq_files = len(seq.get("results", []))
            seq_successful = len([r for r in seq.get("results", []) if r.get("status") == "success"])

            entity_count_acc = 0.0
            type_seq_acc = 0.0
            type_dist_sim = 0.0

            if seq_successful > 0:
                successful = [r for r in seq.get("results", []) if r.get("status") == "success"]
                entity_count_accs = [r["metrics"]["entity_count_acc"] for r in successful]
                type_seq_accs = [r["metrics"]["entity_type_sequence_acc"] for r in successful]
                type_dist_sims = [r["metrics"]["type_distribution_sim"] for r in successful]

                entity_count_acc = sum(entity_count_accs) / len(entity_count_accs) if entity_count_accs else 0.0
                type_seq_acc = sum(type_seq_accs) / len(type_seq_accs) if type_seq_accs else 0.0
                type_dist_sim = sum(type_dist_sims) / len(type_dist_sims) if type_dist_sims else 0.0

            # Step 7: CAD Sequence metrics
            cad_seq = metrics.get("cad_sequence", {})
            cad_has_results = cad_seq.get("has_results", False)

            row = {
                "output_version": output_version,
                "raw_files": raw_files,
                "step_files": step_files,
                "png_visualizations": png_count,
                "topology_success": f"{topo_success}/{topo_total}" if topo_total > 0 else "N/A",
                "danglel": danglel,
                "sir": sir,
                "fluxee": fluxee,
                "json_validation": f"{json_valid}/{json_total}" if json_total > 0 else "N/A",
                "seq_files_evaluated": seq_successful,
                "entity_count_acc": f"{entity_count_acc:.4f}",
                "type_seq_acc": f"{type_seq_acc:.4f}",
                "type_dist_sim": f"{type_dist_sim:.4f}",
                "cad_seq_available": "Yes" if cad_has_results else "No"
            }
            rows.append(row)

    # Write CSV
    if rows:
        keys = rows[0].keys()
        csv_path = output_csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✓ Results saved to: {csv_path}")

        # Generate plots if matplotlib available
        if has_matplotlib and len(rows) > 1:
            generate_comparison_plots(rows, output_csv.replace('.csv', ''))

    return rows


def rank_best_cad_sequence_files(results, output_csv_rank):
    """
    Rank the best files by CAD sequence accuracy.

    For each output_version with sequence evaluation results,
    extract individual file metrics and rank by the three metrics:
    - entity_count_acc
    - type_seq_acc
    - type_dist_sim

    Save top min(len(files), 10) results to CSV.
    """
    import csv

    all_files = []

    # Collect all sequence evaluation results
    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        if result.get("status") == "completed":
            metrics = result.get("metrics", {})
            seq = metrics.get("sequence", {})
            seq_results = seq.get("results", [])

            for file_result in seq_results:
                if file_result.get("status") == "success":
                    file_metrics = file_result.get("metrics", {})

                    # Calculate average score (weighted average of three metrics)
                    entity_count_acc = file_metrics.get("entity_count_acc", 0.0)
                    type_seq_acc = file_metrics.get("entity_type_sequence_acc", 0.0)
                    type_dist_sim = file_metrics.get("type_distribution_sim", 0.0)

                    # Weighted average: all three metrics equally important
                    avg_score = (entity_count_acc + type_seq_acc + type_dist_sim) / 3.0

                    all_files.append({
                        "output_version": output_version,
                        "file": file_result.get("file", "unknown"),
                        "entity_count_acc": f"{entity_count_acc:.4f}",
                        "type_seq_acc": f"{type_seq_acc:.4f}",
                        "type_dist_sim": f"{type_dist_sim:.4f}",
                        "avg_score": avg_score,
                        "avg_score_formatted": f"{avg_score:.4f}"
                    })

    if not all_files:
        print("  ✗ No sequence evaluation results found to rank")
        return

    # Sort by average score (descending)
    all_files.sort(key=lambda x: x["avg_score"], reverse=True)

    # Take top min(len(files), 10)
    num_top = min(len(all_files), 10)
    top_files = all_files[:num_top]

    # Write to CSV
    if top_files:
        keys = ["rank", "output_version", "file", "entity_count_acc", "type_seq_acc", "type_dist_sim", "avg_score_formatted"]

        with open(output_csv_rank, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

            for rank, file_data in enumerate(top_files, 1):
                row = {
                    "rank": rank,
                    "output_version": file_data["output_version"],
                    "file": file_data["file"],
                    "entity_count_acc": file_data["entity_count_acc"],
                    "type_seq_acc": file_data["type_seq_acc"],
                    "type_dist_sim": file_data["type_dist_sim"],
                    "avg_score_formatted": file_data["avg_score_formatted"]
                }
                writer.writerow(row)

        print(f"✓ Top {num_top} files ranked and saved to: {output_csv_rank}")

        # Print table to console
        print(f"\n  Top {num_top} Best Performing Files (by avg score):")
        print("  " + "-" * 100)
        print(f"  {'Rank':<6} {'Entity Count':<15} {'Type Seq':<15} {'Type Dist':<15} {'Avg Score':<12} {'File':<40}")
        print("  " + "-" * 100)

        for rank, file_data in enumerate(top_files, 1):
            entity_acc = float(file_data["entity_count_acc"])
            type_seq = float(file_data["type_seq_acc"])
            type_dist = float(file_data["type_dist_sim"])
            avg_score = float(file_data["avg_score_formatted"])
            filename = file_data["file"][-40:] if len(file_data["file"]) > 40 else file_data["file"]

            print(f"  {rank:<6} {entity_acc:<15.4f} {type_seq:<15.4f} {type_dist:<15.4f} {avg_score:<12.4f} {filename:<40}")

        print("  " + "-" * 100)


def generate_comparison_plots(rows, output_prefix):
    """Generate matplotlib plots for results comparison."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    versions = [r["output_version"] for r in rows]
    entity_count_accs = [float(r["entity_count_acc"]) for r in rows]
    type_seq_accs = [float(r["type_seq_acc"]) for r in rows]
    type_dist_sims = [float(r["type_dist_sim"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluation Results Comparison Across Output Versions', fontsize=16)

    # Plot 1: Sequence Metrics
    ax = axes[0, 0]
    x = np.arange(len(versions))
    width = 0.25
    ax.bar(x - width, entity_count_accs, width, label='Entity Count Acc')
    ax.bar(x, type_seq_accs, width, label='Type Seq Acc')
    ax.bar(x + width, type_dist_sims, width, label='Type Dist Sim')
    ax.set_ylabel('Accuracy')
    ax.set_title('Step 6: Sequence Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Entity Count Accuracy Trend
    ax = axes[0, 1]
    ax.plot(range(len(entity_count_accs)), entity_count_accs, 'o-', label='Entity Count Accuracy')
    ax.fill_between(range(len(entity_count_accs)), entity_count_accs, alpha=0.3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Entity Count Accuracy Trend')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: Type Sequence Accuracy Trend
    ax = axes[1, 0]
    ax.plot(range(len(type_seq_accs)), type_seq_accs, 's-', color='orange', label='Type Seq Accuracy')
    ax.fill_between(range(len(type_seq_accs)), type_seq_accs, alpha=0.3, color='orange')
    ax.set_ylabel('Accuracy')
    ax.set_title('Type Sequence Accuracy Trend')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 4: Type Distribution Similarity Trend
    ax = axes[1, 1]
    ax.plot(range(len(type_dist_sims)), type_dist_sims, '^-', color='green', label='Type Dist Sim')
    ax.fill_between(range(len(type_dist_sims)), type_dist_sims, alpha=0.3, color='green')
    ax.set_ylabel('Similarity')
    ax.set_title('Type Distribution Similarity Trend')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = f"{output_prefix}_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    plt.close()


def main():
    """Run evaluation pipeline for all output versions."""
    print("\n" + "=" * 70)
    print("BATCH EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output versions: {len(OUTPUT_VERSIONS)}")
    print("=" * 70)

    for i, output_version in enumerate(OUTPUT_VERSIONS, 1):
        print(f"\n[{i}/{len(OUTPUT_VERSIONS)}] Processing: {output_version}")
        print("-" * 70)

        output_dir = os.path.join(BASE_OUTPUT_DIR, output_version)

        # Check if output directory exists
        if not os.path.exists(output_dir):
            print(f"✗ Output directory not found: {output_dir}")
            results[output_version] = {"status": "error", "message": "Output directory not found"}
            continue

        # Count JSON files
        json_dir = os.path.join(output_dir, "json")
        json_count = 0
        if os.path.exists(json_dir):
            json_count = len([f for f in os.listdir(json_dir) if f.endswith('.json')])

        print(f"  JSON files: {json_count}")

        # Run evaluation steps
        step_results = {}
        step_results["step2_json_to_step"] = step2_json_to_step(output_version)
        step_results["step3_visualize_step"] = step3_visualize_step(output_version)
        step_results["step4_topology_eval"] = step4_topology_eval(output_version)
        step_results["step5_json_validation"] = step5_json_validation(output_version)
        step_results["step6_sequence_metrics"] = step6_sequence_metrics(output_version)
        step_results["step7_cad_sequence_eval"] = step7_cad_sequence_eval(output_version)

        # Count successful steps
        successful_steps = sum(1 for v in step_results.values() if v is True)
        total_steps = sum(1 for v in step_results.values() if v is not None)

        # Extract metrics
        metrics = format_metrics(output_version)

        results[output_version] = {
            "status": "completed",
            "json_count": json_count,
            "steps_passed": successful_steps,
            "steps_total": total_steps,
            "details": step_results,
            "metrics": metrics
        }

        print(f"  Result: {successful_steps}/{total_steps} steps passed")

        # Print detailed metrics
        if successful_steps > 0:
            print_detailed_results(metrics)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        status = result.get("status", "unknown")

        if status == "completed":
            passed = result.get("steps_passed", 0)
            total = result.get("steps_total", 0)
            json_count = result.get("json_count", 0)
            print(f"\n{output_version}")
            print(f"  Status: ✓ Completed")
            print(f"  JSON files: {json_count}")
            print(f"  Steps passed: {passed}/{total}")
        else:
            print(f"\n{output_version}")
            print(f"  Status: ✗ {result.get('message', status)}")

    print("\n" + "=" * 70)
    print("Note: Check individual output directories for detailed results")
    print("=" * 70)

    # Save comprehensive results summary
    print("\n" + "=" * 70)
    print("SAVING RESULTS SUMMARY")
    print("=" * 70)

    output_csv = "/root/cmu/16825_l43d/CMU16825_Final_project/evaluation_results_summary.csv"
    summary_rows = save_results_summary(results, output_csv)

    if summary_rows:
        print(f"\n✓ Summary generated for {len(summary_rows)} output versions")
    else:
        print("\n✗ No completed results to summarize")

    # Rank best CAD sequence files
    print("\n" + "=" * 70)
    print("RANKING BEST FILES")
    print("=" * 70)

    output_csv_rank = "/root/cmu/16825_l43d/CMU16825_Final_project/top_best_files.csv"
    rank_best_cad_sequence_files(results, output_csv_rank)

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
