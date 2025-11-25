#!/usr/bin/env python
"""
Process Omni-CAD dataset: Convert JSON CAD sequences to STEP mesh files.

This script converts JSON-format CAD sequences from the Omni-CAD dataset 
into STEP mesh files using the DeepCAD processing pipeline.

Usage:
    conda activate DeepCAD
    python pipeline/process_cad.py

The script will:
1. Read all JSON files from data/Omni-CAD/json/
2. Convert them to STEP mesh files
3. Save output to data/Omni-CAD/step/
"""

import subprocess
import logging
import time
from pathlib import Path
from datetime import datetime
import sys

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"process_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ProcessingTimer:
    """Simple timer for tracking processing time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed_str(self):
        if self.start_time is None:
            return "0s"
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


def run_command(cmd, cwd=None):
    """Run command with filtered output (log every 5 min)."""
    timer = ProcessingTimer()
    timer.start()
    last_log_time = time.time()

    skip_patterns = [
        "Statistics on Transfer",
        "Transfer Mode",
        "Transferring Shape",
        "WorkSession",
        "Step File Name",
        "Write  Done",
        "*********"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                if any(pattern in line for pattern in skip_patterns):
                    continue

                is_progress = "Exporting" in line and "%" in line
                is_critical = any(kw in line for kw in ["ERROR", "SUMMARY", "Total", "Success", "Failed"])
                is_startup = any(kw in line for kw in ["Source directory", "Found", "Output directory"])

                if is_progress:
                    # Log only every 5 minutes
                    current_time = time.time()
                    if current_time - last_log_time >= 300:
                        logger.info(line)
                        last_log_time = current_time
                    print(f"\r{line}", end='', flush=True)
                elif is_critical or is_startup:
                    print(f"\n{line}")
                    logger.info(line)

        process.wait()
        logger.info(f"\n[Completed in {timer.elapsed_str()}]")
        return process.returncode == 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def main():
    logger.info("\n" + "="*80)
    logger.info("OMNI-CAD CAD PROCESSING: JSON -> STEP")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80 + "\n")

    overall_timer = ProcessingTimer()
    overall_timer.start()

    # Setup paths
    project_root = Path(__file__).parent.parent.resolve()
    data_root = project_root / "data" / "Omni-CAD"
    deepcad_utils = project_root / "3rd_party" / "DeepCAD" / "utils"
    json_dir = data_root / "json"
    step_output_dir = data_root / "json_step"
    step_final_dir = data_root / "step"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Data root: {data_root}\n")

    # Check if already done
    if step_final_dir.exists() and any(step_final_dir.iterdir()):
        logger.info("[SKIP] STEP files already exist")
        logger.info(f"Directory: {step_final_dir}")

        # Count files
        file_count = sum(1 for _ in step_final_dir.rglob("*.step"))
        logger.info(f"Files found: {file_count:,}")
        logger.info("\nProcessing already completed!")
        return True

    # Export to STEP
    logger.info("="*80)
    logger.info("Exporting JSON to STEP format")
    logger.info("="*80)
    logger.info("Estimated time: 4-6 hours")
    logger.info("Storage needed: ~29 GB\n")

    cmd = [
        "python",
        str(deepcad_utils / "export2step_progress.py"),
        "--src", str(json_dir),
        "--form", "json",
        "--num", "-1"
    ]

    logger.info(f"Command: {' '.join(str(c) for c in cmd)}\n")

    success = run_command(cmd, cwd=str(deepcad_utils))

    if not success:
        logger.error("\nERROR: STEP export failed!")
        return False

    # Rename output directory
    if step_output_dir.exists():
        if step_final_dir.exists():
            logger.info(f"\nRemoving old directory: {step_final_dir}")
            import shutil
            shutil.rmtree(step_final_dir)

        logger.info(f"Renaming: {step_output_dir} -> {step_final_dir}")
        step_output_dir.rename(step_final_dir)

    # Count final files
    file_count = sum(1 for _ in step_final_dir.rglob("*.step"))
    logger.info(f"\n[SUCCESS] Created {file_count:,} STEP files")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total time: {overall_timer.elapsed_str()}")
    logger.info("="*80)
    logger.info("\nDataset Summary:")
    logger.info("-" * 80)
    logger.info(f"JSON (input):        ✓  {sum(1 for _ in json_dir.rglob('*.json')):,} files (~14 GB)")
    logger.info(f"TXT (captions):      ✓  100 files (~165 MB)")
    logger.info(f"STEP (meshes):       ✓  {file_count:,} files (~29 GB)")
    logger.info("="*80)
    logger.info(f"\nLog saved to: {log_file}")
    logger.info("\n[SUCCESS] CAD processing completed successfully!")
    logger.info("="*80 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n[INTERRUPTED] Processing stopped by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"\n\n[ERROR] Unexpected error: {e}")
        sys.exit(1)





