"""
Visualize CAD Truncation: Side-by-side comparison of truncated vs complete geometry.
Uses OpenCASCADE to render JSON sequences via STEP conversion.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

# Add DeepCAD utils to path
project_root = Path(__file__).parent.parent
deepcad_utils = project_root / "3rd_party" / "DeepCAD" / "utils"
sys.path.append(str(deepcad_utils))

from OCC.Display.SimpleGui import init_display
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.Graphic3d import (
    Graphic3d_MaterialAspect,
    Graphic3d_NOM_SILVER,
    Graphic3d_TypeOfShadingModel_Phong
)


class CADVisualizer:
    """Visualize CAD sequences by rendering to images."""

    def __init__(self):
        """Initialize OpenCASCADE display for rendering."""
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Headless rendering

        self.display, _, _, _ = init_display(size=(800, 600))
        self.display.View.TriedronErase()

        # Set pure white background
        from OCC.Core.Aspect import Aspect_GFM_NONE
        self.display.View.SetBgGradientStyle(Aspect_GFM_NONE)
        self.display.View.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

        # Enable realistic Phong shading
        self.display.View.SetShadingModel(Graphic3d_TypeOfShadingModel_Phong)

        # Enable lighting
        viewer = self.display.View.Viewer()
        viewer.SetLightOn()

    def json_to_step(self, json_path: Path, step_path: Path) -> bool:
        """
        Convert JSON CAD sequence to STEP file using DeepCAD export utility.

        Args:
            json_path: Path to input JSON file
            step_path: Path to output STEP file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            step_path.parent.mkdir(parents=True, exist_ok=True)

            # Use DeepCAD export utility
            export_script = deepcad_utils / "export2step.py"

            # Create temporary directory for single file conversion
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Copy JSON to temp directory with expected structure
                temp_json = tmpdir_path / json_path.name
                import shutil
                shutil.copy(json_path, temp_json)

                # Run export (output will be in temp_json_step)
                cmd = [
                    sys.executable,
                    str(export_script),
                    "--src", str(tmpdir_path),
                    "--form", "json"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(deepcad_utils)
                )

                # Find generated STEP file
                step_output_dir = tmpdir_path.parent / f"{tmpdir_path.name}_step"
                generated_step = step_output_dir / f"{json_path.stem}.step"

                if generated_step.exists():
                    shutil.copy(generated_step, step_path)
                    return True
                else:
                    print(f"Warning: STEP file not generated for {json_path.name}")
                    # Try alternative location
                    alt_step = tmpdir_path / f"{json_path.stem}.step"
                    if alt_step.exists():
                        shutil.copy(alt_step, step_path)
                        return True

                    return False

        except Exception as e:
            print(f"Error converting {json_path.name} to STEP: {e}")
            return False

    def render_step_to_image(self, step_path: Path, img_path: Path) -> bool:
        """
        Render STEP file to PNG image.

        Args:
            step_path: Path to STEP file
            img_path: Path to output PNG file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read STEP file
            step_reader = STEPControl_Reader()
            status = step_reader.ReadFile(str(step_path))

            if status != 1:  # IFSelect_RetDone
                print(f"Failed to read STEP file: {step_path}")
                return False

            step_reader.TransferRoot()
            shape = step_reader.Shape()

            # Set material for clearer visualization
            material = Graphic3d_MaterialAspect(Graphic3d_NOM_SILVER)
            ais_shape = self.display.DisplayShape(shape, update=False)[0]

            # Set medium gray color
            color = Quantity_Color(0.6, 0.6, 0.6, Quantity_TOC_RGB)
            ais_shape.SetColor(color)
            ais_shape.SetMaterial(material)

            # Enable edge display with black edges
            from OCC.Core.Prs3d import Prs3d_Drawer
            drawer = ais_shape.Attributes()
            drawer.SetFaceBoundaryDraw(True)
            drawer.FaceBoundaryAspect().SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB))
            drawer.FaceBoundaryAspect().SetWidth(1.0)
            ais_shape.SetAttributes(drawer)

            self.display.Context.Display(ais_shape, True)

            # Fit view
            self.display.FitAll()
            self.display.View.ZFitAll()

            # Zoom out slightly
            current_scale = self.display.View.Scale()
            self.display.View.SetScale(current_scale * 0.85)

            # Save image
            img_path.parent.mkdir(parents=True, exist_ok=True)
            self.display.View.Dump(str(img_path))

            # Clean up
            self.display.EraseAll()
            self.display.View.Reset()

            return True

        except Exception as e:
            print(f"Error rendering STEP file {step_path}: {e}")
            self.display.EraseAll()
            self.display.View.Reset()
            return False

    def create_side_by_side_comparison(
        self,
        truncated_json: Path,
        original_json: Path,
        output_path: Path,
        temp_dir: Path = None
    ) -> bool:
        """
        Create side-by-side comparison image of truncated vs complete geometry.

        Args:
            truncated_json: Path to truncated JSON file
            original_json: Path to original JSON file
            output_path: Path to save comparison image
            temp_dir: Directory for temporary STEP files (optional)

        Returns:
            True if successful, False otherwise
        """
        if temp_dir is None:
            temp_dir = output_path.parent / "temp_step"

        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate STEP files
            truncated_step = temp_dir / f"{truncated_json.stem}.step"
            original_step = temp_dir / f"{original_json.stem}.step"

            print(f"  Converting truncated JSON to STEP...")
            if not self.json_to_step(truncated_json, truncated_step):
                print(f"  Failed to convert truncated JSON")
                return False

            print(f"  Converting original JSON to STEP...")
            if not self.json_to_step(original_json, original_step):
                print(f"  Failed to convert original JSON")
                return False

            # Render images
            truncated_img = temp_dir / f"{truncated_json.stem}.png"
            original_img = temp_dir / f"{original_json.stem}.png"

            print(f"  Rendering truncated geometry...")
            if not self.render_step_to_image(truncated_step, truncated_img):
                print(f"  Failed to render truncated STEP")
                return False

            print(f"  Rendering original geometry...")
            if not self.render_step_to_image(original_step, original_img):
                print(f"  Failed to render original STEP")
                return False

            # Create side-by-side comparison
            print(f"  Creating comparison image...")
            self.combine_images_side_by_side(
                truncated_img,
                original_img,
                output_path,
                truncated_json
            )

            # Clean up temp files
            truncated_step.unlink(missing_ok=True)
            original_step.unlink(missing_ok=True)
            truncated_img.unlink(missing_ok=True)
            original_img.unlink(missing_ok=True)

            return True

        except Exception as e:
            print(f"Error creating comparison: {e}")
            return False

    def combine_images_side_by_side(
        self,
        left_img_path: Path,
        right_img_path: Path,
        output_path: Path,
        truncated_json_path: Path
    ):
        """
        Combine two images side-by-side with labels.

        Args:
            left_img_path: Path to left (truncated) image
            right_img_path: Path to right (original) image
            output_path: Path to save combined image
            truncated_json_path: Path to truncated JSON (for metadata)
        """
        from PIL import Image, ImageDraw, ImageFont

        # Load images
        left_img = Image.open(left_img_path)
        right_img = Image.open(right_img_path)

        # Get dimensions
        width = left_img.width + right_img.width
        height = max(left_img.height, right_img.height) + 80  # Extra space for labels

        # Create combined image
        combined = Image.new('RGB', (width, height), color='white')

        # Paste images
        combined.paste(left_img, (0, 60))
        combined.paste(right_img, (left_img.width, 60))

        # Add labels
        draw = ImageDraw.Draw(combined)

        # Try to use a nice font, fall back to default
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        # Read truncation metadata
        with open(truncated_json_path, 'r') as f:
            trunc_data = json.load(f)
            metadata = trunc_data.get('truncation_metadata', {})

        # Labels
        left_label = f"Truncated: {metadata.get('kept_operations', '?')}/{metadata.get('original_operations', '?')} ops ({metadata.get('truncation_percentage', '?')}%)"
        right_label = f"Complete: {metadata.get('original_operations', '?')} ops (100%)"

        # Draw text
        draw.text((left_img.width // 2 - 100, 20), left_label, fill='black', font=font)
        draw.text((left_img.width + right_img.width // 2 - 100, 20), right_label, fill='black', font=font)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path)
        print(f"  Saved comparison to {output_path}")


def main():
    """Example usage."""
    visualizer = CADVisualizer()

    # Example: visualize a truncated file vs original
    original_json = Path("data/Omni-CAD-subset/json/0021/00210058_00006.json")
    truncated_json = Path("data/Omni-CAD-subset/json_truncated_test/0021/00210058_00006_tr_01.json")
    output_img = Path("data/Omni-CAD-subset/visualizations/00210058_00006_comparison.png")

    if truncated_json.exists() and original_json.exists():
        print(f"Creating visualization for {original_json.name}...")
        success = visualizer.create_side_by_side_comparison(
            truncated_json,
            original_json,
            output_img
        )
        if success:
            print(f"Successfully created comparison image!")
        else:
            print(f"Failed to create comparison image")
    else:
        print("Files not found. Run truncation first.")


if __name__ == "__main__":
    main()
