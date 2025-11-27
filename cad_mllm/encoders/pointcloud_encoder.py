# pointcloud_encoder.py

import torch
import torch.nn as nn
from typing import Optional
from omegaconf import OmegaConf
import os
from pathlib import Path

# From Michelangelo repo
from Michelangelo.michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver


def download_michelangelo_checkpoint(checkpoint_path: str) -> None:
    """
    Auto-download Michelangelo checkpoint if it doesn't exist.

    Tries multiple reliable sources to avoid manual uploads.
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        print(f"✓ Michelangelo checkpoint already exists: {checkpoint_path}")
        return

    print(f"⚠ Michelangelo checkpoint not found at {checkpoint_path}")
    print("📥 Attempting automatic download...")

    # Create checkpoints directory
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Try multiple download sources in order of reliability
    sources = [
        {
            "name": "Hugging Face Mirror",
            "url": "https://huggingface.co/datasets/tiange/Michelangelo-checkpoints/resolve/main/shapenet.pth",
            "method": "urllib",
        },
        {
            "name": "Google Drive (gdown)",
            "file_id": "1wzfa4EoijmyfTpLPfma9r03wvQ_mbjFD",
            "method": "gdown",
        },
    ]

    for source in sources:
        try:
            print(f"\n🔄 Trying {source['name']}...")

            if source["method"] == "urllib":
                import urllib.request
                urllib.request.urlretrieve(source["url"], checkpoint_path)

            elif source["method"] == "gdown":
                try:
                    import gdown
                except ImportError:
                    print("   ⚠ gdown not installed, skipping...")
                    continue
                gdown.download(id=source["file_id"], output=str(checkpoint_path), quiet=False)

            # Verify the download
            if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1_000_000:  # > 1MB
                print(f"✓ Successfully downloaded from {source['name']}")
                print(f"✓ Saved to: {checkpoint_path}")
                return
            else:
                print(f"   ✗ Download failed or file too small")
                checkpoint_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            checkpoint_path.unlink(missing_ok=True)
            continue

    # All sources failed
    raise FileNotFoundError(
        f"\n❌ Could not auto-download Michelangelo checkpoint.\n"
        f"Please manually download from:\n"
        f"  https://huggingface.co/datasets/tiange/Michelangelo-checkpoints\n"
        f"And place it at: {checkpoint_path}"
    )


class MichelangeloPointEncoder(nn.Module):
    """
    Frozen Michelangelo point encoder wrapper.

    This implements g_p in Eq. (2) of the CAD-MLLM paper:
        X_p  ->  E_p = g_p(X_p)
    where g_p is the pretrained Michelangelo shape encoder.

    Args:
        encoder_cfg_path: path to michelangelo_point_encoder_cfg.yaml
                          (the tiny config you saved for AlignedShapeLatentPerceiver)
        encoder_sd_path:  path to michelangelo_point_encoder_state_dict.pt
                          (state_dict for AlignedShapeLatentPerceiver)
        num_points:       number of points to sample per shape for the encoder
        freeze:           if True, keep Michelangelo encoder frozen
        device:           torch.device or string ("cuda"/"cpu")
    """

    def __init__(
        self,
        encoder_cfg_path: str,
        encoder_sd_path: str,
        num_points: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.num_points = num_points
        self.freeze_encoder = freeze
        self.dtype = dtype

        # --- 1) Auto-download checkpoint if missing ---
        download_michelangelo_checkpoint(encoder_sd_path)

        # --- 2) Load config for AlignedShapeLatentPerceiver ---
        shape_cfg = OmegaConf.load(encoder_cfg_path)
        shape_params = dict(shape_cfg.params)  # OmegaConf -> plain dict

        # --- 3) Instantiate encoder (no Lightning, no CLIP) ---

        self.encoder = AlignedShapeLatentPerceiver(
            **shape_params,
            device=self.device,
            dtype=dtype,
        ).to(self.device)
        # print(self.encoder.embed_dim)

        # --- 4) Load weights ---
        state_dict = torch.load(encoder_sd_path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(state_dict, strict=True)

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                # print(p.dtype)
                p.requires_grad = False
            self.encoder.eval()

        # --- 4) Probe output dim with a dummy forward ---
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_points, 6, device=self.device).to(dtype)
            pc = dummy[..., :3]
            feats = dummy[..., 3:]
            global_embed, _ = self.encoder.encode_latents(pc, feats)
        self._output_dim = int(global_embed.shape[-1])

        print(
            f"[MichelangeloPointEncoder] loaded. "
            f"num_points={self.num_points}, embed_dim={self._output_dim}, freeze={self.freeze_encoder}"
        )

    @property
    def output_dim(self) -> int:
        """Dimension D of the global shape embedding."""
        return self._output_dim

    def _prepare_surface(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert (B, N, C) -> (B, num_points, 6) for Michelangelo.

        - If C == 3: assume xyz, append zero normals.
        - If C == 6: assume xyz + normals.
        - Randomly sample or pad to `self.num_points`.
        """
        if points.dim() != 3:
            raise ValueError(f"Expected points of shape (B, N, C), got {points.shape}")

        B, N, C = points.shape
        points = points.to(self.device)

        if C == 3:
            zeros = torch.zeros(B, N, 3, device=self.device, dtype=points.dtype)
            surface = torch.cat([points, zeros], dim=-1)  # (B, N, 6)
        elif C == 6:
            surface = points
        else:
            raise ValueError(f"Expected C=3 or 6 (xyz or xyz+normals), got C={C}")

        # Sample / pad along N
        if N > self.num_points:
            idx = torch.randperm(N, device=self.device)[: self.num_points]
            surface = surface[:, idx, :]
        elif N < self.num_points:
            pad_idx = torch.randint(0, N, (B, self.num_points - N), device=self.device)
            pad = torch.gather(
                surface,
                1,
                pad_idx.unsqueeze(-1).expand(-1, -1, surface.shape[-1]),
            )
            surface = torch.cat([surface, pad], dim=1)

        # Ensure final shape (B, num_points, 6)
        surface = surface[:, : self.num_points, :].contiguous().to(self.dtype)
        return surface

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, C) with C = 3 or 6.

        Returns:
            global shape token: (B, 1, D)
            (this is what goes into your point_cloud projector f_γ)
        """
        surface = self._prepare_surface(points)
        pc = surface[..., :3]
        feats = surface[..., 3:]
        # print(pc.dtype)
        # print(feats.dtype)

        if self.freeze_encoder:
            with torch.no_grad():
                shape_embed, _ = self.encoder.encode_latents(pc, feats)
        else:
            shape_embed, _ = self.encoder.encode_latents(pc, feats)

        # shape_embed: (B, D) → (B, 1, D)
        return shape_embed.unsqueeze(1)



# ----------------------------------------------------------------------
# Backwards-compat point cloud encoder (old API name).
# The new code uses MichelangeloPointEncoder; this stub only exists
# so that `from cad_mllm.encoders import PointCloudEncoder` keeps working.
class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, freeze: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, C)
        B, N, C = points.shape
        x = points.view(B * N, C)
        x = self.mlp(x)
        x = x.view(B, N, self.output_dim)
        return x
