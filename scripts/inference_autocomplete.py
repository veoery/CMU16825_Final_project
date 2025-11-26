#!/usr/bin/env python
"""Autocompletion inference script for CAD-MLLM.

Given a truncated CAD JSON sequence plus its text caption, this script
formats the prompt the same way as training ("Complete this CAD sequence..."),
optionally attaches image / point cloud inputs, and generates the completion
tokens using a fine-tuned Stage 3 checkpoint.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from cad_mllm import CADMLLMModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run CAD autocompletion inference")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint directory (e.g., outputs_autocomplete/stage3_all_model)",
    )
    parser.add_argument(
        "--cad_id",
        type=str,
        required=False,
        help="CAD ID in format '0000/00000071_00005' (used to find caption/truncated JSON)",
    )
    parser.add_argument(
        "--text_caption",
        type=str,
        required=False,
        help="Text caption describing the CAD sample. Required if --cad_id is not provided.",
    )
    parser.add_argument(
        "--txt_json_root",
        type=str,
        required=False,
        help="Root directory for text caption JSON files (e.g., data/Omni-CAD-subset/txt). "
             "Needed if caption is not provided directly.",
    )
    parser.add_argument(
        "--truncated_json_path",
        type=str,
        required=False,
        help="Direct path to truncated JSON file. "
             "If not provided, will be constructed from --cad_id, --truncated_json_root, and --truncation_index.",
    )
    parser.add_argument(
        "--truncated_json_root",
        type=str,
        required=False,
        help="Root directory containing truncated JSON files (e.g., data/Omni-CAD-subset/json_truncated).",
    )
    parser.add_argument(
        "--truncation_index",
        type=int,
        default=1,
        help="Truncation suffix index (e.g., 1 -> _tr_01). Only used when --truncated_json_path is not provided.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional image path to condition on (PNG/JPG).",
    )
    parser.add_argument(
        "--pointcloud_path",
        type=str,
        default=None,
        help="Optional point cloud .npz path to condition on.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="Number of points to sample/pad point clouds to.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype.",
    )

    return parser.parse_args()


def load_caption_from_txt_root(cad_id: str, txt_root: Path) -> str:
    """Search txt_root/*.json files for the caption matching cad_id."""
    if not txt_root.exists():
        raise FileNotFoundError(f"Text root not found: {txt_root}")

    for json_file in sorted(txt_root.glob("*.json")):
        with open(json_file, "r") as f:
            entries = json.load(f)
        for entry in entries:
            if entry.get("id") == cad_id:
                return entry.get("text caption", "")

    raise ValueError(f"Could not find caption for CAD ID {cad_id} in {txt_root}")


def build_truncated_path(cad_id: str, trunc_root: Path, trunc_idx: int) -> Path:
    """Construct truncated JSON path like json_truncated/0000/XXXXXX_tr_01.json."""
    if not trunc_root.exists():
        raise FileNotFoundError(f"Truncated JSON root not found: {trunc_root}")

    parent, name = cad_id.split("/")
    trunc_name = f"{name}_tr_{trunc_idx:02d}.json"
    trunc_path = trunc_root / parent / trunc_name
    if not trunc_path.exists():
        raise FileNotFoundError(f"Truncated JSON not found: {trunc_path}")
    return trunc_path


def load_json_as_string(path: Path) -> str:
    with open(path, "r") as f:
        data = json.load(f)
    return json.dumps(data, separators=(",", ":"))


def load_image_tensor(image_path: Optional[str]) -> Optional[torch.Tensor]:
    if not image_path:
        return None
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor


def load_pointcloud_tensor(pointcloud_path: Optional[str], num_points: int) -> Optional[torch.Tensor]:
    if not pointcloud_path:
        return None
    pc_path = Path(pointcloud_path)
    if not pc_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {pc_path}")

    data = np.load(pc_path)
    if "points" in data:
        points = data["points"]
    elif "xyz" in data:
        points = data["xyz"]
    else:
        points = data[list(data.keys())[0]]

    points = points.astype(np.float32)

    if len(points) >= num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
    else:
        pad_idx = np.random.choice(len(points), num_points - len(points), replace=True)
        points = np.concatenate([points, points[pad_idx]], axis=0)

    return torch.from_numpy(points).unsqueeze(0)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = CADMLLMModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.config.device = device.type
    model.config.dtype = args.dtype

    # Resolve caption
    text_caption = args.text_caption
    if not text_caption:
        if not args.cad_id or not args.txt_json_root:
            raise ValueError("Either provide --text_caption or both --cad_id and --txt_json_root.")
        text_caption = load_caption_from_txt_root(args.cad_id, Path(args.txt_json_root))

    # Resolve truncated JSON path
    truncated_path = args.truncated_json_path
    if not truncated_path:
        if not args.cad_id or not args.truncated_json_root:
            raise ValueError(
                "Either provide --truncated_json_path or both --cad_id and --truncated_json_root."
            )
        truncated_path = build_truncated_path(
            args.cad_id,
            Path(args.truncated_json_root),
            args.truncation_index,
        )
    truncated_seq = load_json_as_string(Path(truncated_path))

    prompt = (
        f"Complete this CAD sequence: {text_caption}\n"
        f"Partial: {truncated_seq}\n"
        f"Complete: "
    )

    print("\n=== Autocomplete Prompt ===")
    print(prompt)
    print("==========================\n")

    # Optional modalities
    pixel_values = load_image_tensor(args.image_path)
    point_clouds = load_pointcloud_tensor(args.pointcloud_path, args.num_points)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)
    if point_clouds is not None:
        point_clouds = point_clouds.to(device)

    with torch.no_grad():
        generated = model.generate(
            text_prompt=prompt,
            pixel_values=pixel_values,
            point_clouds=point_clouds,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    completion = generated.split("Complete: ", 1)[-1].strip()

    print("=== Generated Completion ===")
    print(completion)
    print("============================")


if __name__ == "__main__":
    main()

