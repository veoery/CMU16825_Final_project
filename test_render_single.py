#!/usr/bin/env python
"""Quick test to render a single STEP file."""

import sys
from pathlib import Path

# Test rendering single file
step_file = Path("data/Omni-CAD-subset/step/0000/00000071_00005.step")
output_dir = Path("data/Omni-CAD-subset/test_img")
output_dir.mkdir(exist_ok=True)

print(f"Testing render of: {step_file}")
print(f"Output to: {output_dir}")

try:
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    import OCC.Core.V3d as V3d
    from OCC.Core.Graphic3d import (Graphic3d_MaterialAspect, Graphic3d_NOM_PLASTIC,
                                     Graphic3d_TypeOfShadingModel_Phong)

    print("✓ Imports successful")

    # Initialize display (offscreen headless rendering)
    print("Initializing display...")
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Headless rendering
    display, _, _, _ = init_display(size=(800, 600))
    display.View.TriedronErase()

    # Set pure white background - disable gradient background first
    from OCC.Core.Aspect import Aspect_GFM_NONE
    display.View.SetBgGradientStyle(Aspect_GFM_NONE)
    display.View.SetBackgroundColor(Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB))

    # Enable realistic Phong shading for better material appearance
    display.View.SetShadingModel(Graphic3d_TypeOfShadingModel_Phong)

    # Adjust lighting for clearer face differentiation
    viewer = display.View.Viewer()
    viewer.SetLightOn()  # Make sure light is on

    print("✓ Display initialized with pure white background and enhanced shading")

    # Read STEP file
    print(f"Reading STEP file...")
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(str(step_file))

    if status != 1:
        print(f"✗ Failed to read STEP file (status: {status})")
        sys.exit(1)

    step_reader.TransferRoot()
    shape = step_reader.Shape()
    print("✓ STEP file loaded")

    # Display shape with enhanced material for clearer face differentiation
    # Use SILVER material for strong directional lighting response
    from OCC.Core.Graphic3d import Graphic3d_NOM_SILVER
    material = Graphic3d_MaterialAspect(Graphic3d_NOM_SILVER)

    ais_shape = display.DisplayShape(shape, update=False)[0]

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

    display.Context.Display(ais_shape, True)
    display.FitAll()
    display.View.ZFitAll()

    # Zoom out slightly to avoid clipping (use SetScale instead)
    current_scale = display.View.Scale()
    display.View.SetScale(current_scale * 0.85)

    print("✓ Shape displayed with realistic shading")

    # Render single view
    img_path = output_dir / "test_render.png"
    display.View.Dump(str(img_path))
    print(f"✓ Image saved to: {img_path}")

    if img_path.exists():
        size_kb = img_path.stat().st_size / 1024
        print(f"✓ File size: {size_kb:.1f} KB")
        print("\n" + "="*60)
        print("SUCCESS! Rendering works correctly.")
        print("="*60)
    else:
        print("✗ Image file not created")
        sys.exit(1)

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nPlease install missing packages:")
    print("  conda install -c conda-forge pythonocc-core")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
