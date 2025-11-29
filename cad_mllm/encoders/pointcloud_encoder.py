# pointcloud_encoder.py

import torch
import torch.nn as nn
from typing import Optional
from omegaconf import OmegaConf

# From Michelangelo repo
from Michelangelo.michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver


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

        # --- 1) Load config for AlignedShapeLatentPerceiver ---
        shape_cfg = OmegaConf.load(encoder_cfg_path)
        shape_params = dict(shape_cfg.params)  # OmegaConf -> plain dict

        # --- 2) Instantiate encoder (no Lightning, no CLIP) ---
        
        self.encoder = AlignedShapeLatentPerceiver(
            **shape_params,
            device=self.device,
            dtype=dtype,
        ).to(self.device)
        # print(self.encoder.embed_dim)

        # --- 3) Load weights ---
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
                shape_embed, latents = self.encoder.encode_latents(pc, feats)
        else:
            shape_embed, latents = self.encoder.encode_latents(pc, feats)

        # shape_embed: (B, D) → (B, 1, D)
        global_token = shape_embed.unsqueeze(1)

        # Result shape: (B, 1 + 512, D) = (B, 513, D)
        combined_tokens = torch.cat([global_token, latents], dim=1)
        # print(f"combined_tokens for pc: {combined_tokens.shape}")

        return combined_tokens



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
