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

    def _load_image(self, cad_id: str) -> Optional[Image.Image]:
        """Load image for the given CAD ID with robustness checks.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Image as PIL Image or None if not found
            Note: Preprocessing (resize/normalize) will be done by image processor in collator
        """
        if not self.image_root or not self.image_root.exists():
            return None

        # Standard filename pattern: {cad_id}_000.png
        image_path = self.image_root / f"{cad_id}_000.png"
        if image_path.exists():
            try:
                img = Image.open(image_path).convert("RGB")
                return img  # Return PIL Image, not numpy array
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

        # Load truncated JSON to get metadata (how many operations to mask)
        truncated_json = self._load_json(sample["truncated_path"])
        kept_operations = truncated_json.get("truncation_metadata", {}).get("kept_operations", 0)

        # Load full JSON
        full_json_path = self.full_json_root / f"{cad_id}.json"
        if not full_json_path.exists():
            self.missing_files["full_json"] += 1
            # Use truncated as fallback (shouldn't happen in valid data)
            full_seq = json.dumps(truncated_json, separators=(',', ':'))
            kept_operations = len(truncated_json.get("sequence", []))
        else:
            full_json = self._load_json(full_json_path)
            full_seq = json.dumps(full_json, separators=(',', ':'))

        result = {
            "input_text": text_caption,
            "full_seq": full_seq,
            "kept_operations": kept_operations,  # Used for masking
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
        num_image_tokens: int = 256,  # DINOv2 with 224x224: 16x16 patches (CLS removed)
        num_pc_tokens: int = 1,  # Michelangelo outputs single global shape token
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_processor = image_processor
        self.num_image_tokens = num_image_tokens
        self.num_pc_tokens = num_pc_tokens

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Args:
            batch: List of samples from MultimodalAutocompleteDataset

        Returns:
            Dictionary containing batched tensors
        """
        # MEMORY OPTIMIZATION: Only include full sequence once (not truncated + full)
        # Use structural masking to mask "already seen" operations
        formatted_texts = []
        kept_ops_list = []
        for sample in batch:
            # Format: Caption + Full JSON (50% memory reduction vs including truncated)
            text = f"Complete this CAD sequence: {sample['input_text']}\n{sample['full_seq']}"
            formatted_texts.append(text)
            kept_ops_list.append(sample['kept_operations'])

        # Tokenize
        encodings = self.tokenizer(
            formatted_texts,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # CRITICAL: Structure-aware masking based on operation indices
        # Mask tokens corresponding to "seen" operations (in truncated version)
        labels = encodings["input_ids"].clone()
        for i, (text, sample) in enumerate(zip(formatted_texts, batch)):
            kept_operations = sample['kept_operations']

            # Parse full JSON to identify mask boundary
            try:
                full_json = json.loads(sample['full_seq'])

                # Create partial JSON (only operations 0 to kept_operations-1)
                partial_json = full_json.copy()
                if "sequence" in partial_json:
                    partial_json["sequence"] = partial_json["sequence"][:kept_operations]

                # Find where "new" content starts by tokenizing partial
                # Prompt: "Complete this CAD sequence: {caption}\n"
                prompt = f"Complete this CAD sequence: {sample['input_text']}\n"
                partial_json_str = json.dumps(partial_json, separators=(',', ':'))
                partial_text = prompt + partial_json_str

                # Tokenize to find cutoff
                partial_tokens = self.tokenizer(partial_text, add_special_tokens=False)["input_ids"]
                mask_until = len(partial_tokens)

                # Mask all tokens up to this point (prompt + seen operations)
                labels[i, :mask_until] = -100
            except (json.JSONDecodeError, KeyError):
                # Fallback: mask just the prompt if JSON parsing fails
                prompt = f"Complete this CAD sequence: {sample['input_text']}\n"
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                labels[i, :len(prompt_tokens)] = -100

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

        # Process images if present - use image_processor for proper preprocessing
        pixel_values_list = []
        has_image = []
        for sample in batch:
            if "pixel_values" in sample and self.image_processor is not None:
                # Preprocess PIL Image (returns dict with 'pixel_values' tensor)
                # This handles: resize to 224x224, convert to tensor, normalize
                processed = self.image_processor(sample["pixel_values"], return_tensors="pt")
                pixel_values_list.append(processed["pixel_values"].squeeze(0))  # Remove batch dim
                has_image.append(True)
            else:
                pixel_values_list.append(None)
                has_image.append(False)

        # Process point clouds if present - resample to exactly 2048 points
        point_cloud_list = []
        has_pc = []
        for sample in batch:
            if "point_clouds" in sample:
                pc = sample["point_clouds"]
                # Ensure consistent shape: exactly 2048 points
                if len(pc) > 2048:
                    # Random sample
                    idx = np.random.choice(len(pc), 2048, replace=False)
                    pc = pc[idx]
                elif len(pc) < 2048:
                    # Pad by repeating
                    pad_idx = np.random.choice(len(pc), 2048 - len(pc), replace=True)
                    pc = np.concatenate([pc, pc[pad_idx]], axis=0)
                point_cloud_list.append(torch.tensor(pc))
                has_pc.append(True)
            else:
                point_cloud_list.append(None)
                has_pc.append(False)

        # CRITICAL: Pre-pad labels to account for image/PC tokens (matching working version)
        # Order must match forward pass: [image] [point_cloud] [text]
        batch_size = len(batch)
        text_seq_len = labels.shape[1]

        # Calculate total sequence length including multimodal tokens
        img_tokens = self.num_image_tokens if any(has_image) else 0
        pc_tokens = self.num_pc_tokens if any(has_pc) else 0
        max_total_len = img_tokens + pc_tokens + text_seq_len

        # Pad labels with -100 for multimodal features (image/PC positions)
        if max_total_len > text_seq_len:
            # Create padded labels tensor, all initialized to -100
            padded_labels = torch.full(
                (batch_size, max_total_len),
                fill_value=-100,
                dtype=labels.dtype
            )
            # Copy text labels to the END (after image and PC tokens)
            text_start_idx = img_tokens + pc_tokens
            padded_labels[:, text_start_idx:text_start_idx + text_seq_len] = labels
            labels = padded_labels

        result["labels"] = labels

        # Add images to output if any present
        if any(has_image):
            # Fill missing with zeros (match shape of processed images)
            for i, pv in enumerate(pixel_values_list):
                if pv is None:
                    # Find a valid image to get the shape
                    valid_img = next((img for img in pixel_values_list if img is not None), None)
                    if valid_img is not None:
                        pixel_values_list[i] = torch.zeros_like(valid_img)
                    else:
                        pixel_values_list[i] = torch.zeros(3, 224, 224)
            result["pixel_values"] = torch.stack(pixel_values_list)

        # Add point clouds to output if any present
        if any(has_pc):
            # Fill missing with zeros
            for i, pc in enumerate(point_cloud_list):
                if pc is None:
                    point_cloud_list[i] = torch.zeros(2048, 3)
            result["point_clouds"] = torch.stack(point_cloud_list)

        return result
