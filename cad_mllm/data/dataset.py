"""Dataset classes for CAD-MLLM."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


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
                   or directory containing multiple JSON files
        json_root: Root directory containing CAD JSON files (e.g., "data/Omni-CAD/json")
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
            # Assume data_path is .../Omni-CAD/txt/XXXX.json
            # Then json_root is .../Omni-CAD/json
            self.json_root = self.data_path.parent.parent / "json"
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
