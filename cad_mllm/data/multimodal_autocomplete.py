"""Multimodal Autocomplete Dataset for CAD-MLLM.

Combines AutocompleteDataset (truncated/full JSON pairing) with
MultimodalCADDataset (image + point cloud support).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MultimodalAutocompleteDataset(Dataset):
    """Dataset for multimodal CAD autocompletion task.

    Combines:
    - AutocompleteDataset: Truncated/full JSON pairing
    - MultimodalCADDataset: Image + point cloud support, modality sampling

    Data structure:
        data/Omni-CAD-subset/
        ├── txt/                    # Text captions
        │   ├── 0000.json
        │   └── 0001.json
        ├── json/                   # FULL JSON sequences
        │   ├── 0000/
        │   │   └── 00000071_00005.json
        │   └── ...
        ├── json_truncated/         # TRUNCATED JSON sequences
        │   ├── 0000/
        │   │   ├── 00000071_00005_tr_01.json
        │   │   ├── 00000071_00005_tr_02.json
        │   │   └── ...
        │   └── ...
        ├── img/                    # Images
        │   └── 0000/
        │       └── 00000071_00005.jpg
        └── pointcloud/             # Point clouds
            └── 0000/
                └── 00000071_00005.npz

    Args:
        data_path: Path to text caption JSON files directory
        truncated_json_root: Root directory containing truncated JSON files
        full_json_root: Root directory containing full JSON files
        image_root: Root directory containing images (optional)
        pc_root: Root directory containing point clouds (optional)
        modality_probs: Probability distribution for sampling modality combinations
        max_samples: Maximum number of samples to load (for testing)
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        truncated_json_root: Union[str, Path],
        full_json_root: Union[str, Path],
        image_root: Optional[Union[str, Path]] = None,
        pc_root: Optional[Union[str, Path]] = None,
        modality_probs: Optional[Dict[str, float]] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.truncated_json_root = Path(truncated_json_root)
        self.full_json_root = Path(full_json_root)
        self.image_root = Path(image_root) if image_root else None
        self.pc_root = Path(pc_root) if pc_root else None

        # Modality sampling probabilities
        self.modality_probs = modality_probs or {
            "text": 0.1,
            "text+pc": 0.3,
            "text+img": 0.3,
            "text+pc+img": 0.3,
        }

        # Validate probabilities
        total_prob = sum(self.modality_probs.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Modality probabilities must sum to 1.0, got {total_prob}")

        # Load samples (pairs truncated JSONs with text captions)
        print(f"Loading samples from {self.truncated_json_root}...")
        self.samples = self._load_samples(max_samples)
        print(f"Loaded {len(self.samples)} samples")

        # Statistics
        self.missing_files = {"images": 0, "pointclouds": 0, "full_json": 0}

    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load all truncated JSON files and pair with text captions.

        Returns:
            List of sample dictionaries containing:
                - id: Original CAD ID (e.g., "0000/00000071_00005")
                - truncated_path: Path to truncated JSON file
                - text_caption: Text description
        """
        samples = []

        # Load text captions from JSON files
        text_captions = {}
        if self.data_path.is_dir():
            for caption_file in sorted(self.data_path.glob("*.json")):
                with open(caption_file, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        text_captions[entry["id"]] = entry["text caption"]

        # Iterate through all truncated JSONs
        truncated_files = sorted(self.truncated_json_root.rglob("*_tr_*.json"))

        for truncated_path in truncated_files:
            # Extract original cad_id from truncated filename
            # Example: "0000/00000071_00005_tr_01.json" → "0000/00000071_00005"
            rel_path = truncated_path.relative_to(self.truncated_json_root)
            stem = rel_path.stem  # "00000071_00005_tr_01"

            # Remove _tr_XX suffix
            base_name = "_".join(stem.split("_")[:-2])  # "00000071_00005"
            parent = rel_path.parent  # "0000"
            cad_id = str(parent / base_name)  # "0000/00000071_00005"

            # Get text caption
            caption = text_captions.get(cad_id, "")

            samples.append({
                "id": cad_id,
                "truncated_path": truncated_path,
                "text_caption": caption,
            })

            # Stop if max_samples reached
            if max_samples and len(samples) >= max_samples:
                break

        if not samples:
            raise ValueError(f"No truncated JSON files found in {self.truncated_json_root}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_modality_combination(self) -> str:
        """Sample a modality combination based on configured probabilities."""
        combinations = list(self.modality_probs.keys())
        probs = list(self.modality_probs.values())
        chosen = random.choices(combinations, weights=probs, k=1)[0]
        return chosen

    def _load_json(self, json_path: Path) -> Dict:
        """Load JSON file with error handling."""
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            return json.load(f)

    def _load_image(self, cad_id: str) -> Optional[np.ndarray]:
        """Load image for the given CAD ID with robustness checks.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Image as numpy array (H, W, C) or None if not found
        """
        if not self.image_root or not self.image_root.exists():
            return None

        # Try different extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = self.image_root / f"{cad_id}{ext}"
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert("RGB")
                    return np.array(img)
                except Exception as e:
                    print(f"[WARNING] Failed to load image {image_path}: {e}")
                    return None

        # If not found, log and return None
        self.missing_files["images"] += 1
        return None

    def _load_pointcloud(self, cad_id: str) -> Optional[np.ndarray]:
        """Load point cloud for the given CAD ID with robustness checks.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Point cloud as numpy array (N, 3) or None if not found
        """
        if not self.pc_root or not self.pc_root.exists():
            return None

        # Try .npz extension
        pc_path = self.pc_root / f"{cad_id}.npz"
        if pc_path.exists():
            try:
                data = np.load(pc_path)
                # Try common keys
                if 'points' in data:
                    return data['points'].astype(np.float32)
                elif 'xyz' in data:
                    return data['xyz'].astype(np.float32)
                else:
                    # Use first array
                    return data[list(data.files)[0]].astype(np.float32)
            except Exception as e:
                print(f"[WARNING] Failed to load point cloud {pc_path}: {e}")
                return None

        # If not found, log and return None
        self.missing_files["pointclouds"] += 1
        return None

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with multimodal data.

        Returns:
            Dictionary containing:
                - input_text: Text caption
                - truncated_seq: Truncated JSON sequence (string)
                - full_seq: Full JSON sequence (string)
                - cad_id: CAD ID
                - modality: Sampled modality combination
                - pixel_values: Image array (if modality includes img)
                - point_clouds: Point cloud array (if modality includes pc)
        """
        sample = self.samples[idx]
        cad_id = sample["id"]
        text_caption = sample["text_caption"]

        # Sample modality combination
        modality = self._sample_modality_combination()

        # Load truncated JSON
        truncated_json = self._load_json(sample["truncated_path"])
        truncated_seq = json.dumps(truncated_json, separators=(',', ':'))

        # Load full JSON
        full_json_path = self.full_json_root / f"{cad_id}.json"
        if not full_json_path.exists():
            self.missing_files["full_json"] += 1
            # Use truncated as fallback (shouldn't happen in valid data)
            full_seq = truncated_seq
        else:
            full_json = self._load_json(full_json_path)
            full_seq = json.dumps(full_json, separators=(',', ':'))

        result = {
            "input_text": text_caption,
            "truncated_seq": truncated_seq,
            "full_seq": full_seq,
            "cad_id": cad_id,
            "modality": modality,
        }

        # Load optional modalities based on sampled combination
        if "img" in modality and self.image_root:
            pixel_values = self._load_image(cad_id)
            if pixel_values is not None:
                result["pixel_values"] = pixel_values

        if "pc" in modality and self.pc_root:
            point_clouds = self._load_pointcloud(cad_id)
            if point_clouds is not None:
                result["point_clouds"] = point_clouds

        return result


class MultimodalAutocompleteCollator:
    """Collator for multimodal autocompletion task.

    Handles text + image + point cloud inputs with truncated/full JSON pairs.
    Implements proper loss masking (only compute loss on completion tokens).
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 32768,
        image_processor=None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_processor = image_processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Args:
            batch: List of samples from MultimodalAutocompleteDataset

        Returns:
            Dictionary containing batched tensors
        """
        # Format text inputs
        formatted_texts = []
        for sample in batch:
            # Instruction format:
            # "Complete this CAD sequence: {caption}\nPartial: {truncated}\nComplete: {full}"
            text = (
                f"Complete this CAD sequence: {sample['input_text']}\n"
                f"Partial: {sample['truncated_seq']}\n"
                f"Complete: {sample['full_seq']}"
            )
            formatted_texts.append(text)

        # Tokenize
        encodings = self.tokenizer(
            formatted_texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CRITICAL: Mask input/context tokens, only compute loss on completion
        labels = encodings["input_ids"].clone()
        for i, text in enumerate(formatted_texts):
            # Find where "Complete: " starts
            completion_marker = "Complete: "
            completion_start_char = text.find(completion_marker) + len(completion_marker)

            # Tokenize prefix to find token position
            prefix = text[:completion_start_char]
            prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            prefix_len = len(prefix_tokens)

            # Mask all tokens before completion
            labels[i, :prefix_len] = -100

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

        # Add optional modalities (batch them if present)
        # IMPORTANT: We don't modify labels here - the model's forward pass
        # handles padding labels to match inputs_embeds when adding image/PC tokens
        # CRITICAL: Must handle batches with inconsistent modalities by using zero-filled placeholders
        batch_size = len(batch)

        # Image batching with zero-filled placeholders for missing images
        if any("pixel_values" in s for s in batch):
            images = []
            for s in batch:
                if "pixel_values" in s:
                    img = torch.from_numpy(s["pixel_values"]).float()
                    # Convert (H, W, C) -> (C, H, W)
                    if img.dim() == 3 and img.shape[2] == 3:
                        img = img.permute(2, 0, 1)
                    # NO NORMALIZATION - model's ImageEncoder handles it internally
                    images.append(img)
                else:
                    # Zero-filled placeholder for missing image (3, 224, 224)
                    images.append(torch.zeros(3, 224, 224))
            result["pixel_values"] = torch.stack(images)  # Shape: [batch_size, 3, 224, 224]

        # Point cloud batching with zero-filled placeholders for missing point clouds
        if any("point_clouds" in s for s in batch):
            pcs = []
            for s in batch:
                if "point_clouds" in s:
                    pc = torch.from_numpy(s["point_clouds"]).float()
                    pcs.append(pc)
                else:
                    # Zero-filled placeholder for missing point cloud
                    # Assume point clouds are (num_points, 3), use same shape as present samples
                    # Get shape from first available sample
                    ref_shape = None
                    for ref_s in batch:
                        if "point_clouds" in ref_s:
                            ref_shape = ref_s["point_clouds"].shape
                            break
                    if ref_shape is not None:
                        pcs.append(torch.zeros(ref_shape))
                    else:
                        # Fallback: use default shape (8192, 3)
                        pcs.append(torch.zeros(8192, 3))
            result["point_clouds"] = torch.stack(pcs)  # Shape: [batch_size, num_points, 3]

        return result
