"""Dataset classes for CAD-MLLM."""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image


class CADDataset(Dataset):
    """Dataset for loading CAD data with text descriptions.

    This dataset loads CAD command sequences and their corresponding text descriptions
    from the Omni-CAD dataset structure.

    Dataset structure:
    └── data_root
        └── Omni-CAD
            └── json
                ├── 0000
                ├── 0001
                └── ...
            └── txt
                ├── 0000.json
                ├── 0001.json
                └── ...

    Args:
        data_path: Path to the text description JSON file (e.g., "data/Omni-CAD/txt/0000.json")
                   or directory containing multiple JSON files (e.g., "data/Omni-CAD/txt/")
        json_root: Root directory containing CAD JSON files (e.g., "data/Omni-CAD/json").
                   If None, will be automatically inferred from data_path.
        tokenizer: Tokenizer for text processing
        max_seq_length: Maximum sequence length for tokenization
        modalities: List of modalities to load (currently only supports ["text"])
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 512,
        modalities: List[str] = ["text"],
        json_root: Optional[Union[str, Path]] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.modalities = modalities

        # Infer json_root if not provided
        if json_root is None:
            # Handle two cases:
            # Case 1: data_path is a file (.../Omni-CAD/txt/XXXX.json)
            #         -> json_root is .../Omni-CAD/json
            # Case 2: data_path is a directory (.../Omni-CAD/txt/)
            #         -> json_root is .../Omni-CAD/json
            if self.data_path.is_file():
                # Go up two levels: XXXX.json -> txt/ -> Omni-CAD/
                self.json_root = self.data_path.parent.parent / "json"
            elif self.data_path.is_dir():
                # Go up one level: txt/ -> Omni-CAD/
                self.json_root = self.data_path.parent / "json"
            else:
                raise ValueError(f"data_path must be a file or directory: {self.data_path}")
        else:
            self.json_root = Path(json_root)

        # Load text descriptions
        self.samples = self._load_text_descriptions()

    def _load_text_descriptions(self) -> List[Dict]:
        """Load text descriptions from JSON file(s)."""
        samples = []

        if self.data_path.is_file():
            # Single JSON file
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                samples.extend(data)
        elif self.data_path.is_dir():
            # Directory containing multiple JSON files
            for json_file in sorted(self.data_path.glob("*.json")):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    samples.extend(data)
        else:
            raise ValueError(f"data_path must be a file or directory: {self.data_path}")

        return samples

    def _load_cad_json(self, cad_id: str) -> Dict:
        """Load CAD command sequence from JSON file.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Dictionary containing CAD command sequence
        """
        # Parse cad_id to get file path
        # cad_id format: "0000/00000071_00005"
        cad_path = self.json_root / f"{cad_id}.json"

        if not cad_path.exists():
            raise FileNotFoundError(f"CAD file not found: {cad_path}")

        with open(cad_path, 'r') as f:
            cad_data = json.load(f)

        return cad_data

    def _serialize_cad_sequence(self, cad_data: Dict) -> str:
        """Convert CAD JSON structure to a serialized string format.

        This creates a text representation of the CAD command sequence
        that the model can learn to generate.

        Args:
            cad_data: Dictionary containing CAD command sequence

        Returns:
            Serialized string representation of CAD commands
        """
        # Simple serialization: convert to JSON string
        # For more sophisticated serialization, you can create a custom format
        return json.dumps(cad_data, separators=(',', ':'))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Returns:
            Dictionary containing:
                - input_text: Input text description
                - target_text: Target CAD sequence (serialized)
                - cad_id: CAD identifier
        """
        sample = self.samples[idx]

        cad_id = sample["id"]
        text_caption = sample["text caption"]

        # Load CAD command sequence
        cad_data = self._load_cad_json(cad_id)
        target_text = self._serialize_cad_sequence(cad_data)

        return {
            "input_text": text_caption,
            "target_text": target_text,
            "cad_id": cad_id,
        }


class CADCollator:
    """Collator for batching CAD dataset samples.

    This collator handles tokenization and padding for batched training.

    Args:
        tokenizer: Tokenizer for text processing
        max_seq_length: Maximum sequence length
        padding: Padding strategy ("max_length" or "longest")
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        padding: str = "max_length",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Args:
            batch: List of samples from CADDataset

        Returns:
            Dictionary containing:
                - input_ids: Tokenized input text
                - attention_mask: Attention mask for input
                - labels: Tokenized target text (for language modeling loss)
        """
        # Extract texts
        input_texts = [sample["input_text"] for sample in batch]
        target_texts = [sample["target_text"] for sample in batch]

        # Create instruction format
        # Format: "Generate a CAD model: <description>\n<CAD sequence>"
        formatted_texts = []
        for input_text, target_text in zip(input_texts, target_texts):
            # Combine instruction with description and target
            formatted_text = f"Generate a CAD model: {input_text}\n{target_text}"
            formatted_texts.append(formatted_text)

        # Tokenize
        encodings = self.tokenizer(
            formatted_texts,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Create labels (same as input_ids for causal language modeling)
        # Set padding tokens to -100 so they're ignored in loss
        labels = encodings["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }


def create_dummy_dataset(output_path: Union[str, Path], num_samples: int = 100):
    """Create a dummy dataset for testing.

    This creates a simplified dataset with dummy CAD sequences for testing
    the training pipeline without requiring the full Omni-CAD dataset.

    Args:
        output_path: Path to save the dummy dataset JSON file
        num_samples: Number of dummy samples to generate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_samples = []

    # Templates for dummy CAD descriptions
    templates = [
        "Generate a CAD model with a rectangular shape.",
        "Generate a CAD model with a cylindrical component.",
        "Generate a CAD model with a circular base and vertical walls.",
        "Generate a CAD model featuring curved surfaces and smooth edges.",
        "Generate a CAD model with multiple circular holes arranged symmetrically.",
    ]

    # Simple dummy CAD sequences
    dummy_cad_sequences = [
        {"type": "rectangle", "width": 10, "height": 5},
        {"type": "cylinder", "radius": 3, "height": 8},
        {"type": "circle", "radius": 5, "extrude": 10},
        {"type": "curve", "points": [[0, 0], [5, 5], [10, 0]]},
        {"type": "pattern", "base": "circle", "count": 4},
    ]

    for i in range(num_samples):
        template_idx = i % len(templates)
        cad_idx = i % len(dummy_cad_sequences)

        sample = {
            "input_text": templates[template_idx],
            "target_text": json.dumps(dummy_cad_sequences[cad_idx], separators=(',', ':')),
            "cad_id": f"dummy_{i:04d}",
        }
        dummy_samples.append(sample)

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(dummy_samples, f, indent=2)

    print(f"Created dummy dataset with {num_samples} samples at {output_path}")


class DummyCADDataset(Dataset):
    """Simplified dataset that works directly with dummy data file.

    This is a simpler version of CADDataset that works with the pre-processed
    dummy data format (doesn't need to load separate JSON files).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 512,
        modalities: List[str] = ["text"],
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.modalities = modalities

        # Load samples
        with open(self.data_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class MultimodalCADDataset(Dataset):
    """Dataset with multimodal curriculum support.

    Supports random modality combination sampling for curriculum training:
    - Text only
    - Text + Point Cloud
    - Text + Image
    - Text + Point Cloud + Image
    
    data/Omni-CAD/
    ├── json/
    │   ├── 0000/
    │   │   └── 00000071_00005.json
    │   └── 0001/
    ├── img/
    │   ├── 0000/
    │   │   └── 00000071_00005.jpg
    │   └── 0001/
    ├── pcd/
    │   ├── 0000/
    │   │   └── 00000071_00005.npz
    │   └── 0001/
    └── txt/
        ├── 0000.json
        └── 0001.json

    Args:
        data_path: Path to the text description JSON file or directory
        json_root: Root directory containing CAD JSON files
        image_root: Root directory containing images
        pc_root: Root directory containing point clouds
        tokenizer: Tokenizer for text processing
        max_seq_length: Maximum sequence length for tokenization
        available_modalities: List of available modalities in this dataset
        modality_sample_probs: Probability distribution for sampling modality combinations
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_seq_length: int = 512,
        json_root: Optional[Union[str, Path]] = None,
        image_root: Optional[Union[str, Path]] = None,
        pc_root: Optional[Union[str, Path]] = None,
        available_modalities: List[str] = ["text"],
        modality_sample_probs: Optional[Dict[str, float]] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.available_modalities = available_modalities
        self.modality_sample_probs = modality_sample_probs or {"text": 1.0}

        # Setup roots
        if json_root is None:
            if self.data_path.is_file():
                self.json_root = self.data_path.parent.parent / "json"
            elif self.data_path.is_dir():
                self.json_root = self.data_path.parent / "json"
            else:
                raise ValueError(f"data_path must be a file or directory: {self.data_path}")
        else:
            self.json_root = Path(json_root)

        # Setup image root
        if image_root is None and self.data_path.exists():
            if self.data_path.is_file():
                self.image_root = self.data_path.parent.parent / "img"
            else:
                self.image_root = self.data_path.parent / "img"
        else:
            self.image_root = Path(image_root) if image_root else None

        # Setup point cloud root
        if pc_root is None and self.data_path.exists():
            if self.data_path.is_file():
                self.pc_root = self.data_path.parent.parent / "pcd"
            else:
                self.pc_root = self.data_path.parent / "pcd"
        else:
            self.pc_root = Path(pc_root) if pc_root else None

        # Load samples
        self.samples = self._load_text_descriptions()

        # Validate modality combinations
        self._validate_modality_probs()

    def _load_text_descriptions(self) -> List[Dict]:
        """Load text descriptions from JSON file(s)."""
        samples = []

        if self.data_path.is_file():
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                samples.extend(data)
        elif self.data_path.is_dir():
            for json_file in sorted(self.data_path.glob("*.json")):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    samples.extend(data)
        else:
            raise ValueError(f"data_path must be a file or directory: {self.data_path}")

        return samples

    def _validate_modality_probs(self):
        """Validate that modality sampling probabilities are valid."""
        # Check probabilities sum to ~1.0
        total_prob = sum(self.modality_sample_probs.values())
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"Modality probabilities must sum to 1.0, got {total_prob}")

        # Check all modality combinations are valid
        for combo in self.modality_sample_probs.keys():
            modalities = combo.split('+')
            for mod in modalities:
                if mod not in self.available_modalities and mod != 'text':
                    raise ValueError(f"Invalid modality '{mod}' in combination '{combo}'")

    def _sample_modality_combination(self) -> List[str]:
        """Sample a modality combination based on configured probabilities."""
        combinations = list(self.modality_sample_probs.keys())
        probs = list(self.modality_sample_probs.values())

        chosen = random.choices(combinations, weights=probs, k=1)[0]
        return chosen.split('+')

    def _load_cad_json(self, cad_id: str) -> Dict:
        """Load CAD command sequence from JSON file."""
        cad_path = self.json_root / f"{cad_id}.json"

        if not cad_path.exists():
            raise FileNotFoundError(f"CAD file not found: {cad_path}")

        with open(cad_path, 'r') as f:
            cad_data = json.load(f)

        return cad_data

    def _load_image(self, cad_id: str) -> Optional[Image.Image]:
        """Load image for the given CAD ID.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Image as PIL Image or None if not found
            Note: Preprocessing (resize/normalize) will be done by image encoder
        """
        if self.image_root is None:
            return None

        # Try common image extensions
        # cad_id format: "0000/00000071_00005"
        # image path: img_root/0000/00000071_00005_000.png
        img_path = self.image_root / f"{cad_id}_000.png"
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except FileNotFoundError:
            print(f"{img_path} not found. Skipping...")
            return None
        except (OSError, IOError) as e:
            print(f"Warning: I/O error loading {img_path}: {e}. Skipping...")
            return None
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}. Skipping...")
            return None

        return None

    def _load_point_cloud(self, cad_id: str) -> Optional[np.ndarray]:
        """Load point cloud for the given CAD ID.

        Args:
            cad_id: CAD ID in format "0000/00000071_00005"

        Returns:
            Point cloud as numpy array (N, 3) or None if not found
        """
        if self.pc_root is None or not self.pc_root.exists():
            print(f"{self.pc_root} does not exist.")
            return None

        # cad_id format: "0000/00000071_00005"
        # pc path: pc_root/0000/00000071_00005.npz
        pc_path = self.pc_root / f"{cad_id}.npz"

        if not pc_path.exists():
            print(f"{pc_path} does not exist.")
            return None

        # Load point cloud from npz file
        data = np.load(pc_path)
        if 'points' in data:
            points = data['points']
        elif 'xyz' in data:
            points = data['xyz']
        else:
            # Assume first key is the point cloud
            points = data[list(data.keys())[0]]

        return points.astype(np.float32)

    def _serialize_cad_sequence(self, cad_data: Dict) -> str:
        """Convert CAD JSON structure to a serialized string format."""
        return json.dumps(cad_data, separators=(',', ':'))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with randomly sampled modalities.

        Returns:
            Dictionary containing:
                - input_text: Input text description
                - target_text: Target CAD sequence (serialized)
                - cad_id: CAD identifier
                - pixel_values: Image data (if sampled and available)
                - point_cloud: Point cloud data (if sampled and available)
                - active_modalities: List of modalities included in this sample
        """
        sample = self.samples[idx]
        cad_id = sample["id"]
        text_caption = sample["text caption"]

        # Load CAD command sequence
        cad_data = self._load_cad_json(cad_id)
        target_text = self._serialize_cad_sequence(cad_data)

        # Sample modality combination
        active_modalities = self._sample_modality_combination()
        # print(active_modalities)

        # Prepare output
        output = {
            "input_text": text_caption,
            "target_text": target_text,
            "cad_id": cad_id,
            "active_modalities": active_modalities,
        }

        # Load additional modalities if requested
        if "image" in active_modalities and self.image_root:
            # print('"image" in active_modalities')
            pixel_values = self._load_image(cad_id)
            if pixel_values is not None:
                output["pixel_values"] = pixel_values
                # print(f"pixel_values: {pixel_values.shape}")

        if "point_cloud" in active_modalities and self.pc_root:
            point_cloud = self._load_point_cloud(cad_id)
            if point_cloud is not None:
                output["point_cloud"] = point_cloud

        return output


class MultimodalCADCollator:
    """Collator for batching multimodal CAD dataset samples.

    Handles variable modality combinations in the same batch.

    Args:
        tokenizer: Tokenizer for text processing
        max_seq_length: Maximum sequence length for tokenization
        padding: Padding strategy
        num_image_tokens: Number of tokens image features will produce (after projection)
        num_pc_tokens: Number of tokens point cloud features will produce (after projection)
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        padding: str = "max_length",
        num_image_tokens: int = 256,  # DINOv2 with 224x224: 16x16 patches (CLS removed)
        num_pc_tokens: int = 1,  # Michelangelo outputs single global shape token
        image_processor=None,  # DINOv2 image processor for proper preprocessing
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.num_image_tokens = num_image_tokens
        self.num_pc_tokens = num_pc_tokens
        self.image_processor = image_processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples with variable modalities."""
        # Extract texts
        input_texts = [sample["input_text"] for sample in batch]
        target_texts = [sample["target_text"] for sample in batch]

        # Create instruction format and tokenize prompts separately to mask them
        formatted_texts = []
        prompt_texts = []
        for input_text, target_text in zip(input_texts, target_texts):
            prompt = f"Generate a CAD model: {input_text}\n"
            formatted_text = f"{prompt}{target_text}"
            formatted_texts.append(formatted_text)
            prompt_texts.append(prompt)

        # Tokenize full sequences
        encodings = self.tokenizer(
            formatted_texts,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize prompts to find their lengths (for masking)
        prompt_encodings = self.tokenizer(
            prompt_texts,
            max_length=self.max_seq_length,
            padding=False,
            truncation=True,
        )

        # Create labels - mask prompt part, only compute loss on target CAD sequence
        labels = encodings["input_ids"].clone()

        # Mask the prompt tokens (set to -100 so they're ignored in loss)
        for i, prompt_ids in enumerate(prompt_encodings["input_ids"]):
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = -100

        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Process images if present
        pixel_values_list = []
        has_image = []
        for sample in batch:
            if "pixel_values" in sample and self.image_processor is not None:
                # Preprocess PIL Image (returns dict with 'pixel_values' tensor)
                processed = self.image_processor(sample["pixel_values"], return_tensors="pt")
                pixel_values_list.append(processed["pixel_values"].squeeze(0))  # Remove batch dim
                has_image.append(True)
            else:
                pixel_values_list.append(None)
                has_image.append(False)

        # Process point clouds if present
        point_cloud_list = []
        has_pc = []
        for sample in batch:
            if "point_cloud" in sample:
                pc = sample["point_cloud"]
                # Ensure consistent shape
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

        # Calculate total sequence length per sample and pad labels accordingly
        # Order must match forward pass: [image] [point_cloud] [text]
        batch_size = len(batch)
        text_seq_len = labels.shape[1]

        # Calculate offsets for each modality
        img_tokens = self.num_image_tokens if any(has_image) else 0
        pc_tokens = self.num_pc_tokens if any(has_pc) else 0
        max_total_len = img_tokens + pc_tokens + text_seq_len

        # Pad labels with -100 for multimodal features
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

        output = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }

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
            output["pixel_values"] = torch.stack(pixel_values_list)

        # Add point clouds to output if any present
        if any(has_pc):
            # Fill missing with zeros
            for i, pc in enumerate(point_cloud_list):
                if pc is None:
                    point_cloud_list[i] = torch.zeros(2048, 3)
            output["point_clouds"] = torch.stack(point_cloud_list)

        return output
