"""Autocomplete dataloader for CAD-MLLM.

Loads raw data (images, point clouds, JSON) and encodes them on-the-fly
using the encoders from cad_mllm.encoders.

Designed for Google Colab with data stored on Google Drive.

Data structure:
├── truncated_json/     # Truncated CAD sequences (.json)
├── full_json/          # Complete CAD sequences (.json)
├── images/             # CAD images (.png, .jpg)
└── point_clouds/       # Point cloud files (.ply, .obj, or .npy)
"""

import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from cad_mllm.encoders import ImageEncoder, MichelangeloPointEncoder


class AutocompleteDataset(Dataset):
    """Dataset for CAD autocomplete task with on-the-fly encoding.

    Loads truncated CAD sequences and their corresponding full sequences,
    along with raw images and point clouds. Encodings are computed during
    data loading (suitable for GPU-accelerated Colab).
    """

    def __init__(
        self,
        truncated_paths: List[str],
        full_paths: List[str],
        image_dir: str,
        pc_dir: str,
        tokenizer=None,
        image_encoder: Optional[ImageEncoder] = None,
        pc_encoder: Optional[MichelangeloPointEncoder] = None,
        max_seq_length: int = 512,
    ):
        """Initialize autocomplete dataset.

        Args:
            truncated_paths: List of paths to truncated CAD JSON files
            full_paths: List of paths to full CAD JSON files
            image_dir: Directory containing CAD images
            pc_dir: Directory containing point cloud files
            tokenizer: Tokenizer for text processing (optional)
            image_encoder: ImageEncoder instance (optional)
            pc_encoder: MichelangeloPointEncoder instance (optional)
            max_seq_length: Maximum sequence length (default: 512)
        """
        self.truncated_paths = truncated_paths
        self.full_paths = full_paths
        self.image_dir = image_dir
        self.pc_dir = pc_dir
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.pc_encoder = pc_encoder
        self.max_seq_length = max_seq_length

        assert len(truncated_paths) == len(full_paths), \
            "Number of truncated and full paths must match"

    def __len__(self) -> int:
        return len(self.truncated_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sample with raw data and encoded features.

        Returns:
            Dictionary containing:
                - truncated_seq: Truncated CAD sequence (JSON string)
                - full_seq: Full CAD sequence (JSON string)
                - img_embeds: Image embeddings or raw image tensor
                - pc_embeds: Point cloud embeddings or raw point cloud
        """
        # Load truncated and full CAD sequences
        with open(self.truncated_paths[idx], 'r') as f:
            truncated_seq = json.load(f)

        with open(self.full_paths[idx], 'r') as f:
            full_seq = json.load(f)

        # Get base filename for data lookup
        # Strip _tr_XX suffix from truncated filename to match image/pointcloud names
        full_stem = Path(self.truncated_paths[idx]).stem
        base_name = full_stem.split("_tr_")[0]

        # Load and encode image
        img_embeds = self._load_and_encode_image(base_name)

        # Load and encode point cloud
        pc_embeds = self._load_and_encode_pc(base_name)

        return {
            "truncated_seq": json.dumps(truncated_seq),
            "full_seq": json.dumps(full_seq),
            "img_embeds": img_embeds,
            "pc_embeds": pc_embeds,
        }

    def _load_and_encode_image(self, base_name: str) -> torch.Tensor:
        """Load and encode image to embeddings.

        Args:
            base_name: Base name of the file (without extension)

        Returns:
            Image embeddings of shape (num_patches, hidden_dim) or raw pixel tensor
        """
        # Try common image extensions with potential _XXX suffix variations
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            # First try exact match
            candidate = Path(self.image_dir) / f"{base_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break

            # If not found, search for files matching base_name_*ext pattern
            # (e.g., 00000171_00001_000.png, 00000171_00001_001.png, etc.)
            parent_dir = Path(self.image_dir)
            if parent_dir.exists():
                matching_files = sorted(parent_dir.glob(f"{base_name}_*{ext}"))
                if matching_files:
                    # Use the first matching file
                    image_path = matching_files[0]
                    break

        if image_path is None:
            raise FileNotFoundError(f"Image not found for {base_name} in {self.image_dir}")

        # Load image
        image = Image.open(str(image_path)).convert('RGB')

        # Encode if encoder available
        if self.image_encoder:
            with torch.no_grad():
                pixel_values = self.image_encoder.preprocess(image)
                img_embeds = self.image_encoder(pixel_values)
            return img_embeds.squeeze(0)  # Remove batch dimension
        else:
            # Return raw tensor if no encoder
            from torchvision import transforms
            transform = transforms.ToTensor()
            return transform(image)

    def _load_and_encode_pc(self, base_name: str) -> torch.Tensor:
        """Load and encode point cloud to embeddings.

        Args:
            base_name: Base name of the file (without extension)

        Returns:
            Point cloud embeddings of shape (1, hidden_dim) or raw point cloud
        """
        # Try common point cloud extensions
        pc_path = None
        for ext in ['.npz', '.npy', '.ply', '.obj']:
            candidate = Path(self.pc_dir) / f"{base_name}{ext}"
            if candidate.exists():
                pc_path = candidate
                break

        if pc_path is None:
            raise FileNotFoundError(f"Point cloud not found for {base_name} in {self.pc_dir}")

        # Load point cloud
        if str(pc_path).endswith('.npz'):
            data = np.load(str(pc_path))
            # NPZ files may contain multiple arrays, try common keys
            if 'points' in data:
                points = torch.from_numpy(data['points']).float()
            elif 'arr_0' in data:
                points = torch.from_numpy(data['arr_0']).float()
            else:
                # Use the first array in the npz file
                points = torch.from_numpy(data[list(data.files)[0]]).float()
        elif str(pc_path).endswith('.npy'):
            points = torch.from_numpy(np.load(str(pc_path))).float()
        elif str(pc_path).endswith('.ply'):
            points = self._load_ply(pc_path)
        elif str(pc_path).endswith('.obj'):
            points = self._load_obj(pc_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {pc_path}")

        # Ensure shape is (N, C) where C = 3 or 6
        if points.dim() == 2:
            pass  # Already (N, C)
        elif points.dim() == 1:
            points = points.unsqueeze(1)
        else:
            raise ValueError(f"Expected points of shape (N, C), got {points.shape}")

        # Add batch dimension for encoder
        points = points.unsqueeze(0)  # (1, N, C)

        # Encode if encoder available
        if self.pc_encoder:
            with torch.no_grad():
                pc_embeds = self.pc_encoder(points)  # (1, 1, hidden_dim)
            return pc_embeds.squeeze(0)  # (1, hidden_dim)
        else:
            # Return raw point cloud if no encoder
            return points.squeeze(0)

    @staticmethod
    def _load_ply(ply_path: Union[str, Path]) -> torch.Tensor:
        """Load point cloud from PLY file."""
        try:
            import trimesh
            mesh = trimesh.load(str(ply_path))
            points = mesh.vertices
            return torch.from_numpy(points).float()
        except ImportError:
            raise ImportError("trimesh required for PLY loading: pip install trimesh")

    @staticmethod
    def _load_obj(obj_path: Union[str, Path]) -> torch.Tensor:
        """Load point cloud from OBJ file."""
        try:
            import trimesh
            mesh = trimesh.load(str(obj_path))
            points = mesh.vertices
            return torch.from_numpy(points).float()
        except ImportError:
            raise ImportError("trimesh required for OBJ loading: pip install trimesh")


class AutocompleteCollator:
    """Collator for autocomplete dataset batches.

    Handles tokenization of CAD sequences and batching of embeddings.
    """

    def __init__(
        self,
        tokenizer=None,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples.

        Args:
            batch: List of samples from AutocompleteDataset

        Returns:
            Dictionary containing batched tensors and tokens
        """
        # Extract sequences
        truncated_seqs = [sample["truncated_seq"] for sample in batch]
        full_seqs = [sample["full_seq"] for sample in batch]

        # Create instruction format: "Continue this CAD sequence: <truncated>\n<full>"
        formatted_texts = []
        for tr_seq, full_seq in zip(truncated_seqs, full_seqs):
            formatted_text = f"Continue this CAD sequence: {tr_seq}\n{full_seq}"
            formatted_texts.append(formatted_text)

        # Tokenize if tokenizer provided
        if self.tokenizer:
            encodings = self.tokenizer(
                formatted_texts,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Create labels (set padding tokens to -100)
            labels = encodings["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            batch_dict = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }
        else:
            # If no tokenizer, return empty tensors
            batch_dict = {}

        # Stack embeddings
        batch_dict["img_embeds"] = torch.stack([s["img_embeds"] for s in batch])
        batch_dict["pc_embeds"] = torch.stack([s["pc_embeds"] for s in batch])

        return batch_dict


def get_autocomplete_dataloader(
    truncated_paths: List[str],
    full_paths: List[str],
    image_dir: str,
    pc_dir: str,
    tokenizer=None,
    image_encoder: Optional[ImageEncoder] = None,
    pc_encoder: Optional[MichelangeloPointEncoder] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    max_seq_length: int = 512,
) -> DataLoader:
    """Create dataloader for CAD autocomplete task with on-the-fly encoding.

    Designed for use on Google Colab with data stored on Google Drive.

    Args:
        truncated_paths: List of paths to truncated CAD JSON files
        full_paths: List of paths to full CAD JSON files
        image_dir: Directory path containing CAD images
        pc_dir: Directory path containing point cloud files
        tokenizer: Tokenizer for text processing (optional)
        image_encoder: ImageEncoder instance (optional, computed on-the-fly if None)
        pc_encoder: MichelangeloPointEncoder instance (optional, computed on-the-fly if None)
        batch_size: Batch size for loading (default: 4)
        shuffle: Whether to shuffle data (default: True)
        num_workers: Number of data loading workers (default: 0, recommended for Colab)
        max_seq_length: Maximum sequence length (default: 512)

    Returns:
        DataLoader instance

    Example (Google Colab):
        from google.colab import drive
        drive.mount('/content/drive')

        from cad_mllm.data import get_autocomplete_dataloader
        from cad_mllm.encoders import ImageEncoder, MichelangeloPointEncoder

        ROOT = "/content/drive/MyDrive/CAD_Project"
        truncated_dir = f"{ROOT}/truncated_json"
        full_dir = f"{ROOT}/full_json"
        image_dir = f"{ROOT}/images"
        pc_dir = f"{ROOT}/point_clouds"

        import glob
        truncated_paths = sorted(glob.glob(truncated_dir + "/*.json"))
        full_paths = []
        for tr in truncated_paths:
            base = os.path.basename(tr).split("_tr_")[0] + ".json"
            full_paths.append(os.path.join(full_dir, base))

        # Initialize encoders
        image_encoder = ImageEncoder(freeze=True)
        pc_encoder = MichelangeloPointEncoder(
            encoder_cfg_path="configs/michelangelo_point_encoder_cfg.yaml",
            encoder_sd_path="path/to/state_dict.pt"
        )

        train_loader = get_autocomplete_dataloader(
            truncated_paths,
            full_paths,
            image_dir,
            pc_dir,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            pc_encoder=pc_encoder,
            batch_size=4,
            shuffle=True
        )

        # Training loop
        device = "cuda"
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                img_embeds=batch["img_embeds"],
                pc_embeds=batch["pc_embeds"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            print("loss:", loss.item())
    """
    dataset = AutocompleteDataset(
        truncated_paths=truncated_paths,
        full_paths=full_paths,
        image_dir=image_dir,
        pc_dir=pc_dir,
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        pc_encoder=pc_encoder,
        max_seq_length=max_seq_length,
    )

    collator = AutocompleteCollator(tokenizer, max_seq_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )