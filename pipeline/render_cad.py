#!/usr/bin/env python
"""
Render CAD models: Convert STEP mesh files to rendered images.

This script renders STEP mesh files from the Omni-CAD dataset into 
multiple viewpoint images using OpenCASCADE. Each STEP file is rendered 
from 8 different viewpoints for comprehensive visualization.

Usage:
    conda activate DeepCAD  # Or learning3d
    python pipeline/render_cad.py --src data/Omni-CAD/step --output data/Omni-CAD/render

Options:
    --src: Source directory with STEP files (default: data/Omni-CAD/step)
    --output: Output directory for rendered images (default: data/Omni-CAD/render)
    --num_views: Number of viewpoints per model (default: 8)
    --idx: Start from index (for resuming) (default: 0)
    --num: Number of files to process, -1 for all (default: -1)
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "3rd_party" / "DeepCAD" / "utils"))

from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
import OCC.Core.V3d as V3d
from OCC.Core.Graphic3d import Graphic3d_TypeOfShadingModel_Phong

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"render_cad_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


class STEPRenderer:
    """Renderer for STEP files using OpenCASCADE."""

    def __init__(self):
        """Initialize the display (headless) with realistic Rhino-style lighting."""
        import os as os_module
        os_module.environ['PYOPENGL_PLATFORM'] = 'egl'  # Headless rendering
        self.display, _, _, _ = init_display(size=(800, 600))
        self.display.View.TriedronErase()

        # Set pure white background - disable gradient background first
        from OCC.Core.Aspect import Aspect_GFM_NONE
        self.display.View.SetBgGradientStyle(Aspect_GFM_NONE)
        self.display.View.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

        # Enable realistic Phong shading for better material appearance
        self.display.View.SetShadingModel(Graphic3d_TypeOfShadingModel_Phong)

        # Ensure lighting is on for clearer face differentiation
        viewer = self.display.View.Viewer()
        viewer.SetLightOn()

    def render_step_file(self, step_file, save_dir, num_views=8):
        """
        Render a STEP file to images from multiple viewpoints.

        Args:
            step_file: Path to STEP file
            save_dir: Directory to save images
            num_views: Number of views to render (default: 8)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read STEP file
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(str(step_file))
            if status != 1:  # IFSelect_RetDone
                return False

            step_reader.TransferRoot()
            shape = step_reader.Shape()

            # Set material for clearer face differentiation - use SILVER for strong lighting response
            from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NOM_SILVER
            material = Graphic3d_MaterialAspect(Graphic3d_NOM_SILVER)

            # Display with material
            ais_shape = self.display.DisplayShape(shape, update=False)[0]

            # Set medium gray color for good contrast against white background
            color = Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB)
            ais_shape.SetColor(color)
            ais_shape.SetMaterial(material)

            # Enable edge display with black color
            from OCC.Core.Prs3d import Prs3d_Drawer
            from OCC.Core.Aspect import Aspect_TOL_SOLID
            drawer = ais_shape.Attributes()
            drawer.SetFaceBoundaryDraw(True)
            drawer.FaceBoundaryAspect().SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB))  # Black edges
            drawer.FaceBoundaryAspect().SetWidth(1.0)
            ais_shape.SetAttributes(drawer)

            self.display.Context.Display(ais_shape, True)

            # Fit view with some margin (zoom out slightly)
            self.display.FitAll()
            self.display.View.ZFitAll()  # Fit depth

            # Zoom out 15% to avoid clipping geometry
            current_scale = self.display.View.Scale()
            self.display.View.SetScale(current_scale * 0.85)

            # Get base projection
            proj = list(self.display.View.Proj())

            # Generate filename base
            file_stem = Path(step_file).stem

            # Render from different viewpoints
            for i in range(num_views):
                new_proj = proj.copy()

                # Flip axes for different views
                if (i & 1) != 0:
                    new_proj[0] *= -1
                if (i & 2) != 0:
                    new_proj[1] *= -1
                if (i & 4) != 0:
                    new_proj[2] *= -1

                self.display.View.SetProj(new_proj[0], new_proj[1], new_proj[2])

                # Save image
                img_path = os.path.join(save_dir, f"{file_stem}_{i:03d}.png")
                self.display.View.Dump(img_path)

            # Clean up
            self.display.EraseAll()
            self.display.View.Reset()

            return True

        except Exception as e:
            logger.warning(f"Error rendering {Path(step_file).name}: {e}")
            self.display.EraseAll()
            self.display.View.Reset()
            return False


def main():
    parser = argparse.ArgumentParser(description="Render STEP files to images")
    parser.add_argument('--src', type=str,
                       default='data/Omni-CAD/step',
                       help="Source folder with STEP files")
    parser.add_argument('--output', type=str,
                       default='data/Omni-CAD/render',
                       help="Output folder for images")
    parser.add_argument('--num_views', type=int, default=8,
                       help="Number of viewpoints per model (default: 8)")
    parser.add_argument('--idx', type=int, default=0,
                       help="Start from index (for resuming)")
    parser.add_argument('--num', type=int, default=-1,
                       help="Number of files to process (-1 for all)")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("CAD RENDERING: STEP -> IMAGES")
    logger.info("="*80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    # Setup paths
    project_root = Path(__file__).parent.parent
    src_dir = project_root / args.src
    output_dir = project_root / args.output

    logger.info(f"Source: {src_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Views per model: {args.num_views}")

    # Get STEP files
    logger.info("\nScanning for STEP files...")
    step_files = sorted(glob.glob(str(src_dir / "**" / "*.step"), recursive=True))
    total_files = len(step_files)
    logger.info(f"Found {total_files:,} STEP files")

    # Apply idx/num filtering
    if args.num != -1:
        step_files = step_files[args.idx:args.idx + args.num]
        logger.info(f"Processing {len(step_files):,} files (from index {args.idx})")

    ensure_dir(output_dir)

    # Initialize renderer
    logger.info("\nInitializing renderer...")
    renderer = STEPRenderer()

    # Process files
    success_count = 0
    skip_count = 0
    error_count = 0

    start_time = time.time()

    logger.info("\n" + "="*80)
    logger.info("Starting rendering...")
    logger.info("="*80 + "\n")

    with tqdm(total=len(step_files), desc="Rendering", unit="file") as pbar:
        for step_file in step_files:
            step_path = Path(step_file)
            file_stem = step_path.stem

            # Preserve folder structure
            rel_path = step_path.relative_to(src_dir)
            img_subdir = output_dir / rel_path.parent
            ensure_dir(img_subdir)

            # Check if already rendered (check first view)
            first_img = img_subdir / f"{file_stem}_000.png"
            if first_img.exists():
                skip_count += 1
                pbar.set_postfix({
                    'success': success_count,
                    'skipped': skip_count,
                    'errors': error_count
                })
                pbar.update(1)
                continue

            # Render
            if renderer.render_step_file(step_file, str(img_subdir), args.num_views):
                success_count += 1
            else:
                error_count += 1

            pbar.set_postfix({
                'success': success_count,
                'skipped': skip_count,
                'errors': error_count
            })
            pbar.update(1)

            # Log progress every 100 files
            if (success_count + error_count) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (success_count + error_count) / elapsed
                remaining = len(step_files) - (success_count + error_count + skip_count)
                eta_seconds = remaining / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600

                logger.info(f"Progress: {success_count + error_count}/{len(step_files)} "
                          f"({rate:.1f} files/sec, ETA: {eta_hours:.1f}h)")

    # Summary
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    logger.info("\n" + "="*80)
    logger.info("RENDERING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total time: {hours}h {minutes}m")
    logger.info(f"Files processed: {len(step_files):,}")
    logger.info(f"Successfully rendered: {success_count:,}")
    logger.info(f"Skipped (existing): {skip_count:,}")
    logger.info(f"Errors: {error_count:,}")
    if (len(step_files) - skip_count) > 0:
        logger.info(f"Success rate: {success_count/(len(step_files)-skip_count)*100:.1f}%")
    logger.info(f"Images created: ~{success_count * args.num_views:,}")
    logger.info("="*80)
    logger.info(f"\nLog saved to: {log_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()





