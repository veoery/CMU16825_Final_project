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

# ============================================================================
# MATPLOTLIB PLOTTING CONFIGURATION - Standardized Colors and Sizes
# ============================================================================
PLOT_CONFIG = {
    # Color scheme for different metrics
    'colors': {
        'entity_count_acc': '#3498DB',      # Blue
        'type_seq_acc': '#E74C3C',          # Red
        'type_dist_sim': '#2ECC71',         # Green
        'topology_success': '#2ECC71',      # Green
        'sir': '#3498DB',                   # Blue
        'danglel': '#E74C3C',               # Red
        'fluxee': '#F39C12',                # Orange
        'conversion_rate': '#FDF5E6',       # Light beige (bar background)
    },
    # Marker styles for different metrics
    'markers': {
        'entity_count_acc': 'o',            # Circle
        'type_seq_acc': 's',                # Square
        'type_dist_sim': '^',               # Triangle
        'topology_success': 'o',            # Circle
        'sir': 's',                         # Square
        'danglel': '^',                     # Triangle
        'fluxee': 'd',                      # Diamond
    },
    # Standard line and marker sizes
    'line_widths': {
        'primary': 3.5,                     # Primary metrics
        'secondary': 2.5,                   # Secondary metrics
    },
    'marker_sizes': {
        'large': 10,                        # Primary metric markers
        'medium': 8,                        # Secondary metric markers
        'small': 40,                        # Scatter plot size (in points)
    },
    # Figure sizes
    'fig_sizes': {
        'single_plot': (6, 6),
        'dual_axis': (16, 9),
        'multi_plot': (16, 8),
        'grid_plot': (16, 9),
    },
    # Font sizes
    'font_sizes': {
        'title': 15,
        'label': 12,
        'tick': 10,
        'annotation': 10,
        'legend': 7,
    },
    # Alpha (transparency) values
    'alpha': {
        'line': 0.8,
        'fill': 0.1,
        'scatter': 0.6,
        'legend': 0.95,
    },
    # DPI settings
    'dpi': {
        'standard': 150,
        'high': 300,
    },
}

# Configuration
OUTPUT_VERSIONS = [
    "output_ckpt_2/output_eval_B2_10k",
    "output_ckpt_2/output_eval_B1_2048",
    # # "output_ckpt_4/output_ckpt_4_B1",
    # # "output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_B2_no_pc",
    # # "output_ckpt_4/output_ckpt_4_B2_test/output_ckpt_4_T5_txt_only",
    # # "output_ckpt_4/output_ckpt_4_B3_txt_8img",
    # # "output_ckpt_4/output_ckpt_4_B4_pc_txt_3img",
    # # "output_ckpt_4/output_ckpt_4_B4_txt_8img_fix",
    # # "output_ckpt_4/output_ckpt_4_B4_pc_txt_3img_fix",
    "output_ckpt_4/output_ckpt_4_B5_1img_pc_fix_case_2",
    "output_ckpt_5/output_ckpt_5_B5_1img_pc_fix_case_2",

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


def rank_best_cad_sequence_files(results, output_csv_rank, n=15):
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
    num_top = min(len(all_files), n)
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
        # print(f"\n  Top {num_top} Best Performing Files (by avg score):")
        # print("  " + "-" * 100)
        # print(f"  {'Rank':<6} {'Entity Count':<15} {'Type Seq':<15} {'Type Dist':<15} {'Avg Score':<12} {'File':<40}")
        # print("  " + "-" * 100)

        # for rank, file_data in enumerate(top_files, 1):
        #     entity_acc = float(file_data["entity_count_acc"])
        #     type_seq = float(file_data["type_seq_acc"])
        #     type_dist = float(file_data["type_dist_sim"])
        #     avg_score = float(file_data["avg_score_formatted"])
        #     filename = file_data["file"][-40:] if len(file_data["file"]) > 40 else file_data["file"]

        #     print(f"  {rank:<6} {entity_acc:<15.4f} {type_seq:<15.4f} {type_dist:<15.4f} {avg_score:<12.4f} {filename:<40}")

        # print("  " + "-" * 100)


def generate_comparison_plots(rows, output_prefix):
    """Generate matplotlib plots for results comparison."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Explicitly close any existing figures
    plt.close('all')

    versions = [r["output_version"] for r in rows]
    entity_count_accs = [float(r["entity_count_acc"]) for r in rows]
    type_seq_accs = [float(r["type_seq_acc"]) for r in rows]
    type_dist_sims = [float(r["type_dist_sim"]) for r in rows]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, ax = plt.subplots(1, 1, figsize=PLOT_CONFIG['fig_sizes']['single_plot'])
    fig.suptitle(f'Evaluation Results Comparison Across Output Versions_{timestamp}', fontsize=PLOT_CONFIG['font_sizes']['title'])

    # Plot 1: Sequence Metrics
    x = np.arange(len(versions))
    width = 0.25
    ax.bar(x - width, entity_count_accs, width, label='Entity Count Acc', color=PLOT_CONFIG['colors']['entity_count_acc'])
    ax.bar(x, type_seq_accs, width, label='Type Seq Acc', color=PLOT_CONFIG['colors']['type_seq_acc'])
    ax.bar(x + width, type_dist_sims, width, label='Type Dist Sim', color=PLOT_CONFIG['colors']['type_dist_sim'])
    ax.set_ylabel('Accuracy', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_title('Step 6: Sequence Metrics', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right', fontsize=PLOT_CONFIG['font_sizes']['tick'])
    ax.legend(fontsize=PLOT_CONFIG['font_sizes']['legend'], framealpha=PLOT_CONFIG['alpha']['legend'])
    ax.grid(axis='y', alpha=0.3)

    # # Plot 2: Entity Count Accuracy Trend
    # ax = axes[0, 1]
    # ax.plot(range(len(entity_count_accs)), entity_count_accs, 'o-', label='Entity Count Accuracy')
    # ax.fill_between(range(len(entity_count_accs)), entity_count_accs, alpha=0.3)
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Entity Count Accuracy Trend')
    # ax.set_xticks(range(len(versions)))
    # ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    # ax.grid(True, alpha=0.3)
    # ax.set_ylim([0, 1])

    # # Plot 3: Type Sequence Accuracy Trend
    # ax = axes[1, 0]
    # ax.plot(range(len(type_seq_accs)), type_seq_accs, 's-', color='orange', label='Type Seq Accuracy')
    # ax.fill_between(range(len(type_seq_accs)), type_seq_accs, alpha=0.3, color='orange')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Type Sequence Accuracy Trend')
    # ax.set_xticks(range(len(versions)))
    # ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    # ax.grid(True, alpha=0.3)
    # ax.set_ylim([0, 1])

    # # Plot 4: Type Distribution Similarity Trend
    # ax = axes[1, 1]
    # ax.plot(range(len(type_dist_sims)), type_dist_sims, '^-', color='green', label='Type Dist Sim')
    # ax.fill_between(range(len(type_dist_sims)), type_dist_sims, alpha=0.3, color='green')
    # ax.set_ylabel('Similarity')
    # ax.set_title('Type Distribution Similarity Trend')
    # ax.set_xticks(range(len(versions)))
    # ax.set_xticklabels([v.split('/')[-1] for v in versions], rotation=45, ha='right')
    # ax.grid(True, alpha=0.3)
    # ax.set_ylim([0, 1])

    plt.tight_layout()
    plot_path = f"{output_prefix}_comparison.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi']['standard'], bbox_inches='tight', facecolor='white')
    print(f"✓ Comparison plot saved to: {plot_path}")
    plt.close()

    # Generate topology evaluation plots
    # generate_topology_eval_plots(rows, output_prefix)

    # # Generate comprehensive step distribution plot
    # generate_step_distribution_plot(rows, output_prefix)

    # Generate the new combined visualization
    generate_combined_topology_plot(rows, output_prefix)

def generate_step_distribution_plot(rows, output_prefix):
    """Generate single comprehensive plot showing all topology metrics across output versions.

    Visualizes all topology evaluation results in one plot:
    - X-axis: Output versions
    - Y-axis: Metric values (normalized 0-100%)
    - Three metric lines: DangEL, SIR, FluxEE
    - Shows each metric's trend across all batches
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    # Explicitly close any existing figures
    plt.close('all')

    versions = [r["output_version"].split('/')[-1] for r in rows]
    x_pos = np.arange(len(versions))

    # Extract topology metrics
    danglel = [float(r.get("danglel", 0)) if r.get("danglel") != "N/A" else 0 for r in rows]
    sir = [float(r.get("sir", 0)) if r.get("sir") != "N/A" else 0 for r in rows]
    fluxee = [float(r.get("fluxee", 0)) if r.get("fluxee") != "N/A" else 0 for r in rows]

    # Normalize metrics to display scale (non-inverted, showing true values)
    # DangEL: show as-is (0 is perfect - no dangling edges)
    danglel_normalized = np.array(danglel)

    # SIR: show raw value (0-1 scale, where 1.0 = all faces intersect = worst)
    # Display as-is to show true problematic nature of self-intersections
    sir_normalized = np.array(sir) * 100

    # FluxEE: show as-is but scale for visibility (0 is perfect - perfectly enclosed)
    # Scale tiny values by 1e17 for better visibility in plots
    fluxee_normalized = np.array(fluxee) * 1e17

    # Create single comprehensive plot with explicit figure
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_sizes']['single_plot'], num=None)

    # Plot three metric lines with dots
    ax.plot(x_pos, danglel_normalized, PLOT_CONFIG['markers']['danglel']+'-', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['large'],
           label='DangEL (lower is better)', color=PLOT_CONFIG['colors']['danglel'], alpha=PLOT_CONFIG['alpha']['line'])

    ax.plot(x_pos, sir_normalized, PLOT_CONFIG['markers']['sir']+'-', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['large'],
           label='SIR % (0%=none, 100%=all faces intersect)', color=PLOT_CONFIG['colors']['sir'], alpha=PLOT_CONFIG['alpha']['line'])

    ax.plot(x_pos, fluxee_normalized, PLOT_CONFIG['markers']['fluxee']+'-', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['large'],
           label='FluxEE (lower is better)', color=PLOT_CONFIG['colors']['fluxee'], alpha=PLOT_CONFIG['alpha']['line'])

    # Add shaded regions for better readability
    ax.fill_between(x_pos, danglel_normalized, alpha=PLOT_CONFIG['alpha']['fill'], color=PLOT_CONFIG['colors']['danglel'])
    ax.fill_between(x_pos, sir_normalized, alpha=PLOT_CONFIG['alpha']['fill'], color=PLOT_CONFIG['colors']['sir'])
    ax.fill_between(x_pos, fluxee_normalized, alpha=PLOT_CONFIG['alpha']['fill'], color=PLOT_CONFIG['colors']['fluxee'])

    # Add value labels on dots
    for i, (d, s, f) in enumerate(zip(danglel_normalized, sir_normalized, fluxee_normalized)):
        ax.text(i, d + 2, f'{d:.0f}', ha='center', fontsize=PLOT_CONFIG['font_sizes']['annotation'], color=PLOT_CONFIG['colors']['danglel'], fontweight='bold')
        ax.text(i, s + 2, f'{s:.0f}', ha='center', fontsize=PLOT_CONFIG['font_sizes']['annotation'], color=PLOT_CONFIG['colors']['sir'], fontweight='bold')
        ax.text(i, f - 2, f'{f:.0f}', ha='center', fontsize=PLOT_CONFIG['font_sizes']['annotation'], color=PLOT_CONFIG['colors']['fluxee'], fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=PLOT_CONFIG['font_sizes']['tick'])
    ax.set_ylabel('Metric Values (DangEL & FluxEE: lower better | SIR: %)', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_xlabel('Output Version', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_title('Topology Metrics Across Output Versions', fontsize=PLOT_CONFIG['font_sizes']['title'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=PLOT_CONFIG['font_sizes']['legend'], framealpha=PLOT_CONFIG['alpha']['legend'])

    plt.tight_layout()
    plot_path = f"{output_prefix}_step_distribution.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi']['standard'], bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive topology metrics plot saved to: {plot_path}")
    plt.close()


def generate_topology_eval_plots(rows, output_prefix):
    """Generate simplified topology evaluation comparison plots.

    Visualizes:
    - topology_success: Success rate percentage
    - danglel: Dangling edges metric
    - sir: Shape Intersection Ratio
    - fluxee: Flux Evaluation Error
    - step/raw conversion rate: File processing success
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  Skipping topology evaluation plots - matplotlib not available")
        return

    # Explicitly close any existing figures
    plt.close('all')

    # Extract topology metrics
    versions = [r["output_version"].split('/')[-1] for r in rows]
    raw_files = [int(r.get("raw_files", 0)) for r in rows]
    step_files = [int(r.get("step_files", 0)) for r in rows]

    # Extract JSON file counts from json_validation field (e.g., "45/45" -> 45)
    def parse_json_count(val):
        if val == 'N/A' or val is None or val == '':
            return 0
        if isinstance(val, str) and '/' in val:
            try:
                parts = val.split('/')
                return int(parts[1])  # Total JSON files
            except:
                return 0
        return 0

    json_files = [parse_json_count(r.get("json_validation", "N/A")) for r in rows]

    # Parse topology_success (e.g., "8/8" -> 8, 8)
    def parse_success_rate(val):
        if val == 'N/A' or val is None or val == '':
            return 0, 1
        if isinstance(val, str) and '/' in val:
            try:
                parts = val.split('/')
                success = int(parts[0])
                total = int(parts[1])
                return success, total if total > 0 else 1
            except:
                return 0, 1
        return 0, 1

    topology_success_counts = [parse_success_rate(r.get("topology_success", "N/A")) for r in rows]
    topology_success = [success / total * 100 for success, total in topology_success_counts]
    danglel = [float(r.get("danglel", 0)) if r.get("danglel") != "N/A" else 0 for r in rows]
    sir = [float(r.get("sir", 0)) if r.get("sir") != "N/A" else 0 for r in rows]
    fluxee = [float(r.get("fluxee", 0)) if r.get("fluxee") != "N/A" else 0 for r in rows]

    # Calculate step/raw conversion rate (success rate of parsing raw to step)
    # If raw == 0, make it 1 to avoid zero division
    # Label format: (step files / json files / raw text files)
    step_raw_ratio = [step_files[i] / max(raw_files[i], 1) * 100 for i in range(len(raw_files))]
    step_raw_labels = [f"{step_files[i]}/{json_files[i] if json_files[i] > 0 else '?'}/{raw_files[i]}" for i in range(len(raw_files))]

    # Create simplified figure with 2x2 grid
    # Adjust figsize based on number of versions to keep reasonable aspect ratio
    # fig_width = max(10, min(14, 8 + len(versions) * 0.5))
    fig_width = 14
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

    # 1. DangEL (Dangling Edges)
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(range(len(versions)), danglel, color='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('DangEL Value', fontsize=11, fontweight='bold')
    ax.set_title('Dangling Edges (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if max(danglel) > 0:
        ax.set_yscale('log')

    # 2. SIR (Shape Intersection Ratio)
    ax = fig.add_subplot(gs[0, 1])
    colors_sir = plt.cm.RdYlGn(np.array(sir))
    bars = ax.bar(range(len(versions)), sir, color=colors_sir, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('SIR Value', fontsize=11, fontweight='bold')
    ax.set_title('SIR - Shape Intersection (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis='y', alpha=0.3)
    for i, val in enumerate(sir):
        ax.text(i, val + 0.03, f'{val:.3f}', ha='center', fontweight='bold', fontsize=8)

    # 3. Step/Raw Conversion Rate
    ax = fig.add_subplot(gs[1, 0])
    colors_ratio = plt.cm.RdYlGn(np.clip(np.array(step_raw_ratio) / 100, 0, 1))
    bars = ax.bar(range(len(versions)), step_raw_ratio, color=colors_ratio, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Step/Raw Conversion Rate', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 160)  # Increased to show labels above bars
    ax.grid(axis='y', alpha=0.3)
    for i, (val, label) in enumerate(zip(step_raw_ratio, step_raw_labels)):
        ax.text(i, val + 8, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)
        ax.text(i, val + 18, f'{label}', ha='center', fontweight='bold', fontsize=7, color='#333333')

    # 4. All Metrics Trend (Simplified)
    ax = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(versions))

    # Topology Success
    # ax.plot(x_pos, topology_success, 'o-', linewidth=2.5, markersize=8,
    #        label='Topology Success (%)', color='#2ECC71')

    # SIR (scaled to 0-100 for better visualization)
    ax.plot(x_pos, np.array(sir) * 100, 's-', linewidth=2.5, markersize=8,
           label='SIR (×100)', color='#3498DB')

    # DangEL (normalized inverse for comparison)
    danglel_normalized = 100 - np.clip(np.array(danglel) / (max(danglel) if max(danglel) > 0 else 1) * 100, 0, 100)
    ax.plot(x_pos, danglel_normalized, '^-', linewidth=2.5, markersize=8,
           label='Quality (inv DangEL %)', color='#E74C3C')

    # FluxEE (normalized inverse)
    fluxee_normalized = 100 - np.clip(np.array(fluxee) / (max(fluxee) if max(fluxee) > 0 else 1) * 100, 0, 100)
    ax.plot(x_pos, fluxee_normalized, 'd-', linewidth=2.5, markersize=8,
           label='Quality (inv FluxEE %)', color='#F39C12')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Output Version', fontsize=11, fontweight='bold')
    ax.set_title('Topology Metrics Trend', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 105)

    fig.suptitle('Topology Evaluation Comparison Across Output Versions',
                fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    plot_path = f"{output_prefix}_topology_eval.png"
    plt.savefig(plot_path, dpi=100, bbox_inches='tight', facecolor='white')
    print(f"✓ Topology evaluation plot saved to: {plot_path}")
    plt.close()

def generate_combined_topology_plot(rows, output_prefix):
    """
    Generates a combined plot showing the Step/Raw Conversion Rate as bars 
    and the Topology Metrics Trend as lines, sharing the same 0-100% Y-axis.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("⚠️  Skipping combined topology plot - matplotlib not available")
        return

    plt.close('all')

    # --- 1. Data Preparation ---
    versions = [r["output_version"].split('/')[-1] for r in rows]
    x_pos = np.arange(len(versions))

    # Parse STEP/Raw Conversion Rate (Bar data)
    raw_files = [int(r.get("raw_files", 0)) for r in rows]
    step_files = [int(r.get("step_files", 0)) for r in rows]
    step_raw_ratio = [step_files[i] / max(raw_files[i], 1) * 100 for i in range(len(raw_files))]
    step_raw_labels = [f"({step_files[i]}/{raw_files[i]})" for i in range(len(raw_files))]

    # Parse Topology Metrics Trend (Line data)
    def parse_success_rate(val):
        if isinstance(val, str) and '/' in val:
            try:
                parts = val.split('/')
                success = int(parts[0])
                total = int(parts[1])
                return success, total if total > 0 else 1
            except:
                return 0, 1
        return 0, 1

    topology_success_counts = [parse_success_rate(r.get("topology_success", "N/A")) for r in rows]
    topology_success = [success / total * 100 for success, total in topology_success_counts]

    # Normalize Inverse Topology Metrics to 0-100% scale (as done in existing code)
    danglel = [float(r.get("danglel", 0)) if r.get("danglel") != "N/A" else 0 for r in rows]
    sir = [float(r.get("sir", 0)) if r.get("sir") != "N/A" else 0 for r in rows]
    fluxee = [float(r.get("fluxee", 0)) if r.get("fluxee") != "N/A" else 0 for r in rows]

    # Normalize metrics (non-inverted, showing true values)
    # DangEL: show as-is (0 is perfect - no dangling edges)
    danglel_normalized = np.array(danglel)

    # SIR: show raw value (0-1 scale, where 1.0 = all faces intersect = worst)
    # Display as-is to show true problematic nature of self-intersections
    sir_normalized = np.array(sir) * 100

    # FluxEE: show as-is but scale for visibility (0 is perfect - perfectly enclosed)
    # Scale tiny values by 1e17 for better visibility in plots
    fluxee_normalized = np.array(fluxee) * 1e17

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_sizes']['single_plot'])

    # 2a. Plot Bars (Conversion Rate)
    # Use a lighter color scheme for the background bars
    bars = ax.bar(x_pos, step_raw_ratio, width=0.6, color=PLOT_CONFIG['colors']['conversion_rate'], edgecolor='black',
                  linewidth=1.0, label='Step/Raw Conversion Rate')
    
    # Add text labels for conversion rate
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{height:.1f}%\n{step_raw_labels[i]}',
                ha='center', va='bottom', fontsize=PLOT_CONFIG['font_sizes']['annotation'], fontweight='bold', color='black')

    # 2b. Plot Lines (Topology Metrics Trend)
    # Instantiate a second Axes that shares the same x-axis (for the lines to sit on top)
    # Note: Although we use twinx(), we set both axes to the same scale (0-100)
    ax2 = ax.twinx()

    # Topology Success (Green) - DISABLED
    # ax2.plot(x_pos, topology_success, PLOT_CONFIG['markers']['topology_success']+'-', linewidth=PLOT_CONFIG['line_widths']['primary'], markersize=PLOT_CONFIG['marker_sizes']['large'],
    #        label='Topology Success (%)', color=PLOT_CONFIG['colors']['topology_success'], zorder=5)

    # SIR % (raw value showing true self-intersection ratio)
    ax2.plot(x_pos, sir_normalized, PLOT_CONFIG['markers']['sir']+'--', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['medium'],
           label='SIR % (0%=none, 100%=all faces)', color=PLOT_CONFIG['colors']['sir'], zorder=5)

    # DangEL (lower is better)d
    ax2.plot(x_pos, danglel_normalized, PLOT_CONFIG['markers']['danglel']+'-', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['medium'],
           label='DangEL (lower is better)', color=PLOT_CONFIG['colors']['danglel'], zorder=5)

    # FluxEE (lower is better)
    ax2.plot(x_pos, fluxee_normalized, PLOT_CONFIG['markers']['fluxee']+'-', linewidth=PLOT_CONFIG['line_widths']['secondary'], markersize=PLOT_CONFIG['marker_sizes']['medium'],
           label='FluxEE (lower is better)', color=PLOT_CONFIG['colors']['fluxee'], zorder=5)

    # --- 3. Formatting ---

    # Configure Primary Axis (Bars)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Conversion Rate (%)', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold', color='black')
    ax.tick_params(axis='y', labelcolor='black')

    # Configure Secondary Axis (Lines) - auto-scale based on metric ranges
    ax2.set_ylabel('Topology Metrics', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Configure X-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(versions, rotation=45, ha='right', fontsize=PLOT_CONFIG['font_sizes']['tick'])
    ax.set_xlabel('Output Version', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Pipeline Success Rate (Bars) vs. Geometric Quality Trend (Lines)', fontsize=PLOT_CONFIG['font_sizes']['title'], fontweight='bold')

    # Combine legends from both axes - place outside chart on the right
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=PLOT_CONFIG['font_sizes']['legend'], framealpha=PLOT_CONFIG['alpha']['legend'])

    plt.tight_layout()
    plot_path = f"{output_prefix}_combined_topology_summary.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi']['standard'], bbox_inches='tight', facecolor='white')
    print(f"✓ Combined topology summary plot saved to: {plot_path}")
    plt.close()

def rank_best_files_per_version_cadseq(results, output_csv):
    """
    Rank best files per output version by CAD sequence accuracy.

    For each output_version with sequence evaluation results,
    extract and rank files by average of the three metrics:
    - entity_count_acc
    - type_seq_acc
    - type_dist_sim

    Save top 5 files per version to CSV with simple column names (1, 2, 3, etc.)
    """
    import csv

    all_rows = []

    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        if result.get("status") == "completed":
            metrics = result.get("metrics", {})
            seq = metrics.get("sequence", {})
            seq_results = seq.get("results", [])

            version_files = []

            for file_result in seq_results:
                if file_result.get("status") == "success":
                    file_metrics = file_result.get("metrics", {})

                    entity_count_acc = file_metrics.get("entity_count_acc", 0.0)
                    type_seq_acc = file_metrics.get("entity_type_sequence_acc", 0.0)
                    type_dist_sim = file_metrics.get("type_distribution_sim", 0.0)

                    avg_score = (entity_count_acc + type_seq_acc + type_dist_sim) / 3.0

                    version_files.append({
                        "output_version": output_version,
                        "file": file_result.get("file", "unknown"),
                        "entity_count_acc": entity_count_acc,
                        "type_seq_acc": type_seq_acc,
                        "type_dist_sim": type_dist_sim,
                        "avg_score": avg_score
                    })

            # Sort by average score (descending)
            version_files.sort(key=lambda x: x["avg_score"], reverse=True)

            # Take top 5 per version
            num_top = min(len(version_files), 5)
            top_files = version_files[:num_top]

            # Add to all_rows with simple column names (1, 2, 3, etc.)
            for rank, file_data in enumerate(top_files, 1):
                row = {
                    "output_version": file_data["output_version"],
                    str(rank): file_data["file"],
                    f"{rank}_entity_count_acc": f"{file_data['entity_count_acc']:.4f}",
                    f"{rank}_type_seq_acc": f"{file_data['type_seq_acc']:.4f}",
                    f"{rank}_type_dist_sim": f"{file_data['type_dist_sim']:.4f}",
                    f"{rank}_avg_score": f"{file_data['avg_score']:.4f}"
                }
                all_rows.append(row)

    if not all_rows:
        print("  ✗ No sequence evaluation results found per version")
        return

    # Group by output_version and pivot
    from collections import defaultdict
    version_data = defaultdict(dict)

    for file_data in all_rows:
        version = file_data["output_version"]
        for rank in range(1, 6):
            rank_str = str(rank)
            if rank_str in file_data:
                version_data[version][rank_str] = file_data[rank_str]
                version_data[version][f"{rank}_entity_count_acc"] = file_data.get(f"{rank}_entity_count_acc", "N/A")
                version_data[version][f"{rank}_type_seq_acc"] = file_data.get(f"{rank}_type_seq_acc", "N/A")
                version_data[version][f"{rank}_type_dist_sim"] = file_data.get(f"{rank}_type_dist_sim", "N/A")
                version_data[version][f"{rank}_avg_score"] = file_data.get(f"{rank}_avg_score", "N/A")

    # Write to CSV
    if version_data:
        # Build fieldnames dynamically
        fieldnames = ["output_version"]
        for rank in range(1, 6):
            fieldnames.append(str(rank))
            fieldnames.append(f"{rank}_entity_count_acc")
            fieldnames.append(f"{rank}_type_seq_acc")
            fieldnames.append(f"{rank}_type_dist_sim")
            fieldnames.append(f"{rank}_avg_score")

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for version in OUTPUT_VERSIONS:
                if version in version_data:
                    row_data = {"output_version": version}
                    row_data.update(version_data[version])
                    writer.writerow(row_data)

        print(f"✓ Per-version CAD sequence rankings saved to: {output_csv}")

        # Print summary
        # for version in OUTPUT_VERSIONS:
        #     if version in version_data:
        #         print(f"\n  {version.split('/')[-1]}:")
        #         for rank in range(1, 6):
        #             rank_str = str(rank)
        #             if rank_str in version_data[version]:
        #                 file_name = version_data[version][rank_str]
        #                 avg_score = version_data[version].get(f"{rank}_avg_score", "N/A")
        #                 print(f"    {rank}. {file_name} (score: {avg_score})")

def rank_best_files_per_version_topology(results, output_csv):
    """
    Rank best files per output version by topology evaluation metrics.

    For each output_version, extract files with topology evaluation results
    and rank by composite topology score:
    - SIR (Shape Intersection Ratio) - weighted 40%
    - Inverse DangEL (1 - normalized dangling edges) - weighted 30%
    - Inverse FluxEE (1 - normalized flux error) - weighted 30%

    Save top 5 files per version to CSV with simple column names (1, 2, 3, etc.)
    """
    import csv

    version_data = {}

    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        if result.get("status") == "completed":
            # Load topology_results.json directly
            src_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
            topo_results_file = os.path.join(src_dir, "topology_results", "topology_results.json")

            topo_results = []
            if os.path.exists(topo_results_file):
                try:
                    with open(topo_results_file, 'r') as f:
                        topo_results = json.load(f)
                except Exception:
                    topo_results = []

            version_files = []

            for file_result in topo_results:
                if file_result.get("status") == "success":
                    file_metrics = file_result.get("metrics", {})

                    sir = file_metrics.get("sir", 0.0)
                    danglel = file_metrics.get("danglel", 0.0)
                    fluxee = file_metrics.get("fluxee", 0.0)

                    # Normalize metrics for composite score
                    # SIR: already normalized (0-1), higher is better
                    sir_score = float(sir) if sir != 'N/A' else 0.0

                    # DangEL: lower is better, invert it
                    # Normalize by scaling: 0 is perfect, larger values are worse
                    danglel_val = float(danglel) if danglel != 'N/A' else 0.0
                    danglel_score = max(0, 1 - min(danglel_val / 1e-14, 1.0)) if danglel_val > 0 else 1.0

                    # FluxEE: lower is better, invert it
                    fluxee_val = float(fluxee) if fluxee != 'N/A' else 0.0
                    fluxee_score = max(0, 1 - min(fluxee_val / 1e-15, 1.0)) if fluxee_val > 0 else 1.0

                    # Composite score: weighted average
                    composite_score = (sir_score * 0.4 + danglel_score * 0.3 + fluxee_score * 0.3)

                    version_files.append({
                        "output_version": output_version,
                        "file": file_result.get("file", "unknown"),
                        "sir": sir_score,
                        "danglel": danglel_score,
                        "fluxee": fluxee_score,
                        "composite_score": composite_score
                    })

            if version_files:
                # Sort by composite score (descending)
                version_files.sort(key=lambda x: x["composite_score"], reverse=True)

                # Take top 5 per version
                num_top = min(len(version_files), 5)
                top_files = version_files[:num_top]

                version_data[output_version] = {
                    "top_files": top_files,
                    "count": num_top
                }

    if not version_data:
        print("  ✗ No topology evaluation results found per version")
        return

    # Write to CSV
    fieldnames = ["output_version"]
    for rank in range(1, 6):
        fieldnames.append(str(rank))
        fieldnames.append(f"{rank}_sir")
        fieldnames.append(f"{rank}_danglel")
        fieldnames.append(f"{rank}_fluxee")
        fieldnames.append(f"{rank}_composite_score")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for version in OUTPUT_VERSIONS:
            if version in version_data:
                row_data = {"output_version": version}

                for rank, file_data in enumerate(version_data[version]["top_files"], 1):
                    rank_str = str(rank)
                    row_data[rank_str] = file_data["file"]
                    row_data[f"{rank}_sir"] = f"{file_data['sir']:.4f}"
                    row_data[f"{rank}_danglel"] = f"{file_data['danglel']:.4f}"
                    row_data[f"{rank}_fluxee"] = f"{file_data['fluxee']:.4f}"
                    row_data[f"{rank}_composite_score"] = f"{file_data['composite_score']:.4f}"

                writer.writerow(row_data)

    print(f"✓ Per-version topology rankings saved to: {output_csv}")

    # Print summary
    # for version in OUTPUT_VERSIONS:
    #     if version in version_data:
    #         print(f"\n  {version.split('/')[-1]}:")
    #         for rank, file_data in enumerate(version_data[version]["top_files"], 1):
    #             file_name = file_data["file"]
    #             composite = file_data["composite_score"]
    #             print(f"    {rank}. {file_name} (composite score: {composite:.4f})")

def visualize_best_outputs_with_gt(csv_path, n=12):
    """
    Part 1: Visualize top N best generated CAD models alongside ground truth images. (done)

    Creates a grid showing:
    - Column 1: Model metrics and info
    - Column 2: Generated model visualization
    - Column 3: Ground truth reference image
    save as png/pdf report

    Part 2: Visualize all generated STEP img alongside ground truth images.
    - the imag will be at `CMU16825_Final_project/output_ckpt_4/output_ckpt_4_B5_1img_pc_fix_case_2/imgs/00900867_00001_repaired_20251202_0417.png`
    - find ground truth with base name '00900867_0000'
    Creates a grid showing:
    - Column 1: Model metrics and info
    - Column 2: Generated model visualization
    - Column 3: Ground truth reference image

    save as png/pdf report
    """
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("⚠️  Skipping visualization - PIL/matplotlib not available")
        return

    import csv

    BASE_DIR = "/root/cmu/16825_l43d/CMU16825_Final_project"
    GT_IMG_DIR = os.path.join(BASE_DIR, "data/gt/test/img/0090")

    def extract_model_id(filename):
        """Extract model ID from filename.

        Handles both JSON and PNG formats:
        - JSON: 00900730_00001_repaired.json -> 00900730_00001
        - PNG: 00900730_00001_repaired_20251202_0617.png -> 00900730_00001
        """
        # Remove extension
        name_without_ext = filename.replace('.json', '').replace('.png', '')

        # Handle _repaired_ separator (PNG with timestamp)
        if '_repaired_' in name_without_ext:
            return name_without_ext.split('_repaired_')[0]

        # Handle _repaired suffix (JSON)
        if name_without_ext.endswith('_repaired'):
            return name_without_ext.replace('_repaired', '')

        return name_without_ext

    def find_ground_truth_image(model_id):
        """Find ground truth image for model ID."""
        if not os.path.exists(GT_IMG_DIR):
            return None

        patterns = [f"{model_id}_000.png", f"{model_id}_001.png", f"{model_id}_002.png"]
        for pattern in patterns:
            img_path = os.path.join(GT_IMG_DIR, pattern)
            if os.path.exists(img_path):
                return img_path

        # Fuzzy match
        for file in os.listdir(GT_IMG_DIR):
            if model_id in file and file.endswith('.png'):
                return os.path.join(GT_IMG_DIR, file)
        return None

    def load_image(img_path, max_size=(400, 400)):
        """Load and resize image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        except:
            return None

    def create_blank_image(size=(400, 400), text="N/A"):
        """Create blank placeholder."""
        img = Image.new('RGB', size, color=(200, 200, 200))
        return img

    # Read best files
    best_files = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                best_files.append(row)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    if not best_files:
        print("❌ No best files found")
        return

    print(f"\n🎨 Creating comparison visualization for top {len(best_files)} results...")

    # Create figure
    figsize = (20, 5 * (len(best_files) // 3 + 1))
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(best_files), 3, figure=fig, hspace=0.4, wspace=0.05)

    for row, file_info in enumerate(best_files):
        rank = int(file_info['rank'])
        output_version = file_info['output_version'].split('/')[-1]
        filename = file_info['file']
        entity_acc = float(file_info['entity_count_acc'])
        type_seq_acc = float(file_info['type_seq_acc'])
        type_dist_sim = float(file_info['type_dist_sim'])
        avg_score = float(file_info['avg_score_formatted'])

        model_id = extract_model_id(filename)

        # Column 1: Info
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.axis('off')

        info_text = f"""RANK #{rank}

ID: {model_id}
Version: {output_version}

Metrics:
  Entity: {entity_acc:.4f}
  Type Seq: {type_seq_acc:.4f}
  Type Dist: {type_dist_sim:.4f}

Score: {avg_score:.4f}"""

        bg_color = '#90EE90' if avg_score >= 0.95 else '#FFD700' if avg_score >= 0.8 else '#FFA500' if avg_score >= 0.6 else '#FFB6C6'

        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8, pad=0.8))

        # Column 2: Generated Image
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.set_title('Generated Model', fontsize=10, fontweight='bold')

        gen_img_path = None
        try:
            output_path = os.path.join(BASE_DIR, file_info['output_version'])
            imgs_dir = os.path.join(output_path, "imgs")
            if os.path.exists(imgs_dir):
                for png_file in os.listdir(imgs_dir):
                    if model_id in png_file and png_file.endswith('.png'):
                        gen_img_path = os.path.join(imgs_dir, png_file)
                        break
        except:
            pass

        if gen_img_path and os.path.exists(gen_img_path):
            gen_img = load_image(gen_img_path)
            if gen_img:
                ax2.imshow(gen_img)
                ax2.set_xlabel('✓ Generated', fontsize=8, color='green', fontweight='bold')
            else:
                ax2.imshow(create_blank_image())
                ax2.set_xlabel('✗ Load error', fontsize=8, color='red')
        else:
            ax2.imshow(create_blank_image())
            ax2.set_xlabel('⚠️  Not found', fontsize=8, color='orange')

        ax2.axis('off')

        # Column 3: Ground Truth Image
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.set_title('Ground Truth', fontsize=10, fontweight='bold')
        
        gt_img_path = find_ground_truth_image(model_id)

        if gt_img_path:
            gt_img = load_image(gt_img_path)
            if gt_img:
                ax3.imshow(gt_img)
                ax3.set_xlabel(f'✓ {os.path.basename(gt_img_path)}', fontsize=7, color='green')
            else:
                ax3.imshow(create_blank_image())
                ax3.set_xlabel('✗ Load error', fontsize=8, color='red')
        else:
            ax3.imshow(create_blank_image())
            ax3.set_xlabel('⚠️  Not found', fontsize=8, color='orange')

        ax3.axis('off')

    fig.suptitle(f'Top {len(best_files)} Best Generated CAD Models vs Ground Truth\n(Ranked by CAD Sequence Matching Accuracy)',
                fontsize=14, fontweight='bold', y=0.995)

    legend_elements = [
        mpatches.Patch(facecolor='#90EE90', label='Excellent (≥0.95)'),
        mpatches.Patch(facecolor='#FFD700', label='Good (≥0.80)'),
        mpatches.Patch(facecolor='#FFA500', label='Fair (≥0.60)'),
        mpatches.Patch(facecolor='#FFB6C6', label='Poor (<0.60)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
              bbox_to_anchor=(0.5, -0.01))

    # Create viz directory if it doesn't exist
    viz_dir = os.path.join(BASE_DIR, "evaluation", "evaluation_result", "viz")
    os.makedirs(viz_dir, exist_ok=True)

    output_path = os.path.join(viz_dir, f"best_outputs_comparison_top{len(best_files)}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Comparison visualization saved: {output_path}")

    try:
        pdf_path = os.path.join(viz_dir, f"best_outputs_comparison_top{len(best_files)}.pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
        print(f"✓ High-quality PDF saved: {pdf_path}")
    except:
        pass

    plt.close()

    # Print summary
    print(f"\n✓ Generated comparison for {len(best_files)} best models")

def visualize_all_generated_images_with_gt():
    """
    Part 2: Visualize ALL generated STEP images from each output version alongside ground truth.

    For each output version:
    - Scans {output_version}/imgs/ directory for all PNG files
    - Extracts model ID from filename (handles timestamp suffixes)
    - Finds corresponding ground truth image
    - Creates 3-column grid visualization (metrics | generated | ground truth)
    - Saves as per-version png and pdf reports
    """
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("⚠️  Skipping visualization - PIL/matplotlib not available")
        return

    import csv
    import re
    from datetime import datetime

    BASE_DIR = "/root/cmu/16825_l43d/CMU16825_Final_project"
    GT_IMG_DIR = os.path.join(BASE_DIR, "data/gt/test/img/0090")

    def extract_model_id_from_png(filename):
        """Extract model ID from generated PNG filename.

        Example: '00900730_00001_repaired_20251202_0617.png' -> '00900730_00001'
        """
        # Remove extension
        name_without_ext = filename.replace('.png', '')

        # Pattern: model_id_number_repaired_timestamp_time.png
        # We want everything before '_repaired_'
        if '_repaired_' in name_without_ext:
            # Get everything before _repaired_
            model_id = name_without_ext.split('_repaired_')[0]
            return model_id

        # Fallback: if no _repaired_, just remove the last 3 parts if they look like timestamp
        parts = name_without_ext.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[:-2])

        return name_without_ext

    def find_ground_truth_image(model_id):
        """Find ground truth image for model ID."""
        if not os.path.exists(GT_IMG_DIR):
            return None
        
        patterns = [f"{model_id}_000.png", f"{model_id}_001.png", f"{model_id}_002.png"]
        for pattern in patterns:
            img_path = os.path.join(GT_IMG_DIR, pattern)
            if os.path.exists(img_path):
                return img_path

        # Fuzzy match
        for file in os.listdir(GT_IMG_DIR):
            if model_id in file and file.endswith('.png'):
                return os.path.join(GT_IMG_DIR, file)
        return None

    def load_image(img_path, max_size=(400, 400)):
        """Load and resize image."""
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
        except:
            return None

    def create_blank_image(size=(400, 400), text="N/A"):
        """Create blank placeholder."""
        img = Image.new('RGB', size, color=(200, 200, 200))
        return img

    # Process each output version
    for output_version in OUTPUT_VERSIONS:
        output_path = os.path.join(BASE_DIR, output_version)
        imgs_dir = os.path.join(output_path, "imgs")

        if not os.path.exists(imgs_dir):
            continue

        # Get all PNG files
        png_files = [f for f in os.listdir(imgs_dir) if f.endswith('.png')]
        png_files = png_files[:10]

        if not png_files:
            continue

        version_name = output_version.split('/')[-1]
        print(f"\n🎨 Visualizing {len(png_files)} generated images for {version_name}...")

        # Create figure with dynamic height based on number of images
        rows = len(png_files)
        figsize = (20, 5 * (rows // 3 + 1))
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        for row, png_filename in enumerate(sorted(png_files)):
            model_id = extract_model_id_from_png(png_filename)
            # print("DEBUG!!", png_filename)
            # print("DEBUG!!", model_id)
            # Column 1: Info
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.axis('off')

            info_text = f"""Generated Image #{row + 1}

ID: {model_id}
File: {png_filename[:40]}...
Version: {version_name}"""

            ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#E0E0FF', alpha=0.8, pad=0.8))

            # Column 2: Generated Image
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.set_title('Generated Model', fontsize=10, fontweight='bold')

            gen_img_path = os.path.join(imgs_dir, png_filename)
            if os.path.exists(gen_img_path):
                gen_img = load_image(gen_img_path)
                if gen_img:
                    ax2.imshow(gen_img)
                    ax2.set_xlabel('✓ Generated', fontsize=8, color='green', fontweight='bold')
                else:
                    ax2.imshow(create_blank_image())
                    ax2.set_xlabel('✗ Load error', fontsize=8, color='red')
            else:
                ax2.imshow(create_blank_image())
                ax2.set_xlabel('⚠️  Not found', fontsize=8, color='orange')

            ax2.axis('off')

            # Column 3: Ground Truth Image
            ax3 = fig.add_subplot(gs[row, 2])
            ax3.set_title('Ground Truth', fontsize=10, fontweight='bold')
            
            gt_img_path = find_ground_truth_image(model_id)

            if gt_img_path:
                gt_img = load_image(gt_img_path)
                if gt_img:
                    ax3.imshow(gt_img)
                    ax3.set_xlabel(f'✓ {os.path.basename(gt_img_path)}', fontsize=7, color='green')
                else:
                    ax3.imshow(create_blank_image())
                    ax3.set_xlabel('✗ Load error', fontsize=8, color='red')
            else:
                ax3.imshow(create_blank_image())
                ax3.set_xlabel('⚠️  Not found', fontsize=8, color='orange')

            ax3.axis('off')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        fig.suptitle(f'All Generated Images: {version_name} ({len(png_files)} images)\nvs Ground Truth',
                    fontsize=14, fontweight='bold', y=0.995)

        # Create viz directory if it doesn't exist
        viz_dir = os.path.join(BASE_DIR, "evaluation", "evaluation_result", "viz")
        os.makedirs(viz_dir, exist_ok=True)

        # Save PNG
        png_output = os.path.join(viz_dir, f"all_generated_images_{version_name}_{timestamp}.png")
        plt.savefig(png_output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Visualization saved: {os.path.basename(png_output)}")

        # Save PDF
        try:
            pdf_output = os.path.join(viz_dir, f"all_generated_images_{version_name}_{timestamp}.pdf")
            plt.savefig(pdf_output, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
            print(f"  ✓ PDF saved: {os.path.basename(pdf_output)}")
        except:
            pass

        plt.close()

    print(f"\n✓ All-images visualization complete for all versions")

def generate_cadseq_file_scatter_plot(results, output_prefix):
    """
    Generate scatter plot comparing individual file-level CAD sequence metrics
    (Entity Count Acc, Type Seq Acc, Type Dist Sim) across all output versions.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  Skipping CAD sequence scatter plot - matplotlib not available")
        return

    plt.close('all')

    # 1. Prepare Data Structure
    plot_data = [] # List of {'version': str, 'entity_acc': float, 'type_seq_acc': float, 'type_dist_sim': float}
    
    # Map version name to an index for plotting
    version_map = {v: i for i, v in enumerate(OUTPUT_VERSIONS)}
    version_labels = [v.split('/')[-1] for v in OUTPUT_VERSIONS]

    # 2. Extract Metrics from Results
    for output_version in OUTPUT_VERSIONS:
        result = results.get(output_version, {})
        if result.get("status") == "completed":
            metrics = result.get("metrics", {})
            seq = metrics.get("sequence", {})
            seq_results = seq.get("results", [])

            for file_result in seq_results:
                if file_result.get("status") == "success":
                    file_metrics = file_result.get("metrics", {})
                    
                    data_point = {
                        'version_index': version_map[output_version],
                        'entity_acc': file_metrics.get("entity_count_acc", 0.0),
                        'type_seq_acc': file_metrics.get("entity_type_sequence_acc", 0.0),
                        'type_dist_sim': file_metrics.get("type_distribution_sim", 0.0),
                    }
                    plot_data.append(data_point)

    if not plot_data:
        print("✗ No CAD sequence evaluation results found for scatter plot.")
        return

    # 3. Create Scatter Plot
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_sizes']['single_plot'])

    # Convert to NumPy arrays for easy plotting
    versions_indices = np.array([d['version_index'] for d in plot_data])
    entity_accs = np.array([d['entity_acc'] for d in plot_data])
    type_seq_accs = np.array([d['type_seq_acc'] for d in plot_data])
    type_dist_sims = np.array([d['type_dist_sim'] for d in plot_data])

    # Introduce small random jitter for better visibility of overlapping points (Optional but recommended)
    jitter = np.random.uniform(-0.1, 0.1, len(versions_indices))

    # Plot each metric
    ax.scatter(versions_indices + jitter, entity_accs, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='Entity Count Acc', marker=PLOT_CONFIG['markers']['entity_count_acc'], color=PLOT_CONFIG['colors']['entity_count_acc'])
    ax.scatter(versions_indices + jitter, type_seq_accs, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='Type Sequence Acc', marker=PLOT_CONFIG['markers']['type_seq_acc'], color=PLOT_CONFIG['colors']['type_seq_acc'])
    ax.scatter(versions_indices + jitter, type_dist_sims, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='Type Distribution Sim', marker=PLOT_CONFIG['markers']['type_dist_sim'], color=PLOT_CONFIG['colors']['type_dist_sim'])

    # 4. Formatting
    ax.set_xticks(range(len(version_labels)))
    ax.set_xticklabels(version_labels, rotation=45, ha='right', fontsize=PLOT_CONFIG['font_sizes']['tick'])
    ax.set_ylabel('Accuracy/Similarity Score', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_xlabel('Output Version', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_title('Individual File CAD Sequence Metrics Comparison', fontsize=PLOT_CONFIG['font_sizes']['title'], fontweight='bold')

    # Set Y-axis limits for score visualization
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.legend(loc='lower left', fontsize=PLOT_CONFIG['font_sizes']['legend'], framealpha=PLOT_CONFIG['alpha']['legend'])
    
    # Draw vertical lines to separate versions visually
    for i in range(len(version_labels)):
        ax.axvline(x=i, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plot_path = f"{output_prefix}_cadseq_file_scatter.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi']['standard'], bbox_inches='tight', facecolor='white')
    print(f"✓ Individual file CAD sequence scatter plot saved to: {plot_path}")
    plt.close()

def generate_topology_file_scatter_plot(results, output_prefix):
    """
    Generate scatter plot comparing individual file-level topology metrics
    (SIR, DangEL (Inverse), FluxEE (Inverse)) across all output versions.
    
    Metrics are normalized to 0-1 (or 0-100%) scale for comparison:
    - SIR: Higher is better.
    - DangEL (Inverse): Higher is better (less dangling edges).
    - FluxEE (Inverse): Higher is better (less enclosure error).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  Skipping topology scatter plot - matplotlib not available")
        return

    plt.close('all')
    
    # Constants for normalization (based on rank_best_files_per_version_topology)
    DANGEL_MAX_NORM = 1e-14
    FLUXEE_MAX_NORM = 1e-15
    
    # 1. Prepare Data Structure
    plot_data = [] # List of {'version_index': int, 'sir': float, 'inv_dangel': float, 'inv_fluxee': float}
    
    # Map version name to an index for plotting
    version_map = {v: i for i, v in enumerate(OUTPUT_VERSIONS)}
    version_labels = [v.split('/')[-1] for v in OUTPUT_VERSIONS]

    # 2. Extract and Normalize Metrics
    for output_version in OUTPUT_VERSIONS:
        output_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
        topo_results_file = os.path.join(output_dir, "topology_results", "topology_results.json")
        version_index = version_map[output_version]

        topo_results = load_json_safe(topo_results_file)

        if isinstance(topo_results, list):
            for file_result in topo_results:
                if file_result.get("status") == "success":
                    file_metrics = file_result.get("metrics", {})

                    sir = file_metrics.get("sir", 0.0)
                    danglel = file_metrics.get("danglel", 0.0)
                    fluxee = file_metrics.get("fluxee", 0.0)

                    # Normalize metrics: convert to float and handle 'N/A'
                    sir_val = float(sir) if sir != 'N/A' else 0.0
                    danglel_val = float(danglel) if danglel != 'N/A' else 0.0
                    fluxee_val = float(fluxee) if fluxee != 'N/A' else 0.0

                    # SIR score (higher is better)
                    sir_score = sir_val

                    # DangEL score (show as-is, lower is better)
                    dangel_score = danglel_val

                    # FluxEE score (show as-is but scale for visibility, lower is better)
                    fluxee_score = fluxee_val * 1e17

                    plot_data.append({
                        'version_index': version_index,
                        'sir': sir_score,
                        'dangel': dangel_score,
                        'fluxee': fluxee_score,
                    })

    if not plot_data:
        print("✗ No successful topology evaluation results found for scatter plot.")
        return

    # 3. Create Scatter Plot
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['fig_sizes']['single_plot'])

    # Convert to NumPy arrays for easy plotting
    versions_indices = np.array([d['version_index'] for d in plot_data])
    sir_scores = np.array([d['sir'] for d in plot_data])
    dangel_scores = np.array([d['dangel'] for d in plot_data])
    fluxee_scores = np.array([d['fluxee'] for d in plot_data])

    # Introduce small random jitter for better visibility of overlapping points
    jitter = np.random.uniform(-0.15, 0.15, len(versions_indices))

    # Plot each metric
    ax.scatter(versions_indices + jitter, sir_scores, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='SIR (0-1, 0=none, 1=all)', marker=PLOT_CONFIG['markers']['sir'], color=PLOT_CONFIG['colors']['sir'])
    ax.scatter(versions_indices + jitter, dangel_scores, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='DangEL (lower is better)', marker=PLOT_CONFIG['markers']['danglel'], color=PLOT_CONFIG['colors']['danglel'])
    ax.scatter(versions_indices + jitter, fluxee_scores, alpha=PLOT_CONFIG['alpha']['scatter'], s=PLOT_CONFIG['marker_sizes']['small'], label='FluxEE ×1e17 (lower is better)', marker=PLOT_CONFIG['markers']['fluxee'], color=PLOT_CONFIG['colors']['fluxee'])

    # 4. Formatting
    ax.set_xticks(range(len(version_labels)))
    ax.set_xticklabels(version_labels, rotation=45, ha='right', fontsize=PLOT_CONFIG['font_sizes']['tick'])
    ax.set_ylabel('Metric Values (see legend for interpretation)', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_xlabel('Output Version', fontsize=PLOT_CONFIG['font_sizes']['label'], fontweight='bold')
    ax.set_title('Individual File Topology Metrics Comparison', fontsize=PLOT_CONFIG['font_sizes']['title'], fontweight='bold')

    ax.grid(axis='y', alpha=0.5, linestyle='--')
    ax.legend(loc='upper left', fontsize=PLOT_CONFIG['font_sizes']['legend'], framealpha=PLOT_CONFIG['alpha']['legend'])
    
    # Draw vertical lines to separate versions visually
    for i in range(len(version_labels)):
        ax.axvline(x=i, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plot_path = f"{output_prefix}_topology_file_scatter.png"
    plt.savefig(plot_path, dpi=PLOT_CONFIG['dpi']['standard'], bbox_inches='tight', facecolor='white')
    print(f"✓ Individual file topology scatter plot saved to: {plot_path}")
    plt.close()

def organize_eval_results():
    """
    Organize evaluation results into structured directories:
    1. Per-version results: {output_version}_results/ (at project root level)
       Contains: sequence_eval_results.csv, .json, json_validation_results.json, .json, topology_results.json, .json
    2. Overall results: evaluation/evaluation_result/ (at project root level)
       Contains: evaluation_results_summary_comparison.png, .csv, top_best_files_*.csv
    """
    import shutil
    import glob

    # Organize per-version results
    for output_version in OUTPUT_VERSIONS:
        output_dir = os.path.join(BASE_OUTPUT_DIR, output_version)
        if not os.path.exists(output_dir):
            continue

        # Create results directory at project root with full path representation
        # e.g., "output_ckpt_2/output_eval_B1_2048_results"
        version_results_dir = os.path.join(BASE_OUTPUT_DIR, f"{output_version}/results")
        os.makedirs(version_results_dir, exist_ok=True)

        files_to_copy = [
            ("json/sequence_eval_results.csv", "sequence_eval_results.csv"),
            ("json/sequence_eval_results.json", "sequence_eval_results.json"),
            ("json_validation_results.json", "json_validation_results.json"),
            ("json_validation_summary.json", "json_validation_summary.json"),
            ("topology_results/topology_results.json", "topology_results.json"),
            ("topology_results/topology_summary.json", "topology_summary.json")
        ]

        moved_count = 0
        for src_rel, dst_name in files_to_copy:
            src_path = os.path.join(output_dir, src_rel)
            dst_path = os.path.join(version_results_dir, dst_name)
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception:
                    pass

        version_name = output_version.split('/')[-1]
        if moved_count > 0:
            print(f"  ✓ {version_name}: {moved_count} result files organized")

    # Organize overall evaluation results
    eval_result_dir = os.path.join(BASE_OUTPUT_DIR, "evaluation", "evaluation_result")
    os.makedirs(eval_result_dir, exist_ok=True)

    # Exact match files
    files_to_move_exact = [
        "evaluation_results_summary_comparison.png",
        "evaluation_results_summary.csv",
        "top_best_cadseq_global.csv",
        "top_best_files_cadseq_match_each.csv",
        "top_best_files_topo_each.csv",
        "top_best_files_each.csv",
        "evaluation_results_summary_step_distribution.png",
        "evaluation_results_summary_topology_eval.png",
        "evaluation_results_summary_cadseq_file_scatter.png",
        "evaluation_results_summary_topology_file_scatter.png",
        "evaluation_results_summary_combined_topology_summary.png"
    ]

    # Move exact match files
    for src_name in files_to_move_exact:
        src_path = os.path.join(BASE_OUTPUT_DIR, src_name)
        dst_path = os.path.join(eval_result_dir, src_name)
        if os.path.exists(src_path):
            try:
                shutil.move(src_path, dst_path)
            except Exception:
                pass

    # Note: Visualization files (best_outputs_comparison_top*.png/pdf and
    # all_generated_images_*.png/pdf) are now saved directly to the viz subfolder
    # by visualize_best_outputs_with_gt() and visualize_all_generated_images_with_gt()

    # Ensure viz subdirectory exists
    viz_dir = os.path.join(eval_result_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"✓ Overall results organized in evaluation/evaluation_result/")

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
    
    plot_prefix = "/root/cmu/16825_l43d/CMU16825_Final_project/evaluation_results_summary"
    
    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL FILE CAD SEQUENCE SCATTER PLOT")
    print("=" * 70)
    generate_cadseq_file_scatter_plot(results, plot_prefix)
    
    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL FILE TOPOLOGY SCATTER PLOT")
    print("=" * 70)
    generate_topology_file_scatter_plot(results, plot_prefix)

    # Rank best files by different metrics
    print("\n" + "=" * 70)
    print("RANKING BEST FILES")
    print("=" * 70)

    # Global CAD Sequence ranking
    output_csv_rank = "/root/cmu/16825_l43d/CMU16825_Final_project/top_best_cadseq_global.csv"
    rank_best_cad_sequence_files(results, output_csv_rank, 20)

    # Per-version CAD Sequence ranking
    print("\n>>> Per-Version CAD Sequence Rankings <<<")
    output_csv_cadseq_each = "/root/cmu/16825_l43d/CMU16825_Final_project/top_best_files_cadseq_match_each.csv"
    rank_best_files_per_version_cadseq(results, output_csv_cadseq_each)

    # Per-version Topology ranking
    print("\n>>> Per-Version Topology Rankings <<<")
    output_csv_topo_each = "/root/cmu/16825_l43d/CMU16825_Final_project/top_best_files_topo_each.csv"
    rank_best_files_per_version_topology(results, output_csv_topo_each)

    # Generate visualization of best outputs (Part 1: Top N best models)
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION - PART 1 (Top N Best CADSEQ Matches)")
    print("=" * 70)
    visualize_best_outputs_with_gt(output_csv_rank, n=20)

    # Generate visualization of all generated images (Part 2: All images per version)
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION - PART 2 (All Generated Images per Version)")
    print("=" * 70)
    visualize_all_generated_images_with_gt()

    # Organize evaluation results into structured directories
    print("\n" + "=" * 70)
    print("ORGANIZING EVALUATION RESULTS")
    print("=" * 70)
    organize_eval_results()

    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
