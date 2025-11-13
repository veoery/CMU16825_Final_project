"""Point cloud encoder for processing 3D point cloud inputs.

Supports two backends:
  1) Simple PointNet-style encoder (original implementation).
  2) Pretrained Michelangelo shape encoder (recommended for CAD).

You choose the backend via the `use_michelangelo` flag (set in model.py via config).
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    # From the official Michelangelo repo:
    # https://github.com/NeuralCarver/Michelangelo
    from michelangelo.utils.misc import get_config_from_file, instantiate_from_config
except ImportError:
    get_config_from_file = None
    instantiate_from_config = None


class PointCloudEncoder(nn.Module):
    """Point cloud encoder for extracting features from 3D points.

    Two modes:

      • use_michelangelo = False  → simple PointNet-style encoder
      • use_michelangelo = True   → wrap Michelangelo's pretrained shape encoder

    In the Michelangelo mode, we implement g_p in Eq. (2) of the CAD-MLLM
    paper, i.e. we output a global shape embedding which is later fed
    into the point-projector f_γ to align with the LLM space.

    Args:
        input_dim:  Input point dimension (3 for xyz, 6 for xyz+normals, etc.).
                   For Michelangelo, you *ideally* want 6 (xyz + normals).
        hidden_dim: Internal feature size (used only in PointNet mode).
        output_dim: Output feature size (used only in PointNet mode; in
                    Michelangelo mode this is inferred from the pretrained model).
        freeze:     Whether to freeze encoder weights.
        use_michelangelo: If True, use Michelangelo shape encoder instead of PointNet.
        miche_config_path: Path to Michelangelo .yaml config.
        miche_ckpt_path:   Path to Michelangelo .ckpt weights.
        num_points: Number of points to sample per shape (for Michelangelo).
        device:     Device string for the Michelangelo model ("cuda" / "cpu").
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 512,
        output_dim: int = 1024,
        freeze: bool = True,
        use_michelangelo: bool = False,
        miche_config_path: Optional[str] = None,
        miche_ckpt_path: Optional[str] = None,
        num_points: int = 2048,
        device: str = "cuda",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_michelangelo = use_michelangelo
        self.num_points = num_points
        self.device = device

        if self.use_michelangelo:
            if get_config_from_file is None or instantiate_from_config is None:
                raise ImportError(
                    "michelangelo.utils.misc not found. "
                    "Make sure you have installed the Michelangelo repo "
                    "(e.g., `pip install -e .` in the cloned Michelangelo directory)."
                )
            if miche_config_path is None or miche_ckpt_path is None:
                raise ValueError(
                    "Michelangelo mode requires `miche_config_path` and `miche_ckpt_path`."
                )

            # --- Load Michelangelo shape encoder ---
            cfg = get_config_from_file(miche_config_path)
            # Official inference uses: model = instantiate_from_config(cfg, ckpt_path)
            self.michelangelo_model = instantiate_from_config(cfg, miche_ckpt_path)
            self.michelangelo_model = self.michelangelo_model.to(self.device)

            if freeze:
                for p in self.michelangelo_model.parameters():
                    p.requires_grad = False
                self.michelangelo_model.eval()

            # Probe the embedding dimension once with a dummy input
            with torch.no_grad():
                dummy = torch.zeros(1, num_points, 6, device=self.device, dtype=torch.float32)
                # The official API is: model.model.encode_shape_embed(surface, return_latents=True)
                shape_embed, _ = self.michelangelo_model.model.encode_shape_embed(
                    dummy, return_latents=True
                )
            # shape_embed: (1, D)
            self.output_dim = shape_embed.shape[-1]
            self.hidden_dim = self.output_dim  # for consistency

            print(f"[PointCloudEncoder] Michelangelo mode enabled, embed dim = {self.output_dim}")

        else:
            # --- Original simple PointNet-style encoder ---
            # Per-point feature extraction
            self.conv1 = nn.Conv1d(input_dim, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, hidden_dim, 1)

            # Global feature aggregation
            self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.conv5 = nn.Conv1d(hidden_dim, output_dim, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            self.bn4 = nn.BatchNorm1d(hidden_dim)
            self.bn5 = nn.BatchNorm1d(output_dim)

            self.relu = nn.ReLU()

            if freeze:
                for param in self.parameters():
                    param.requires_grad = False
                self.eval()

    def _prepare_surface_for_michelangelo(self, points: torch.Tensor) -> torch.Tensor:
        """Convert (B, N, C) points → (B, num_points, 6) surface tensor for Michelangelo.

        - If C == 3, we append zero normals.
        - If C == 6, we assume it is (xyz + normals).
        - We random-sample or pad to `self.num_points`.
        """
        B, N, C = points.shape

        if C == 3:
            zeros = torch.zeros(B, N, 3, device=points.device, dtype=points.dtype)
            surface = torch.cat([points, zeros], dim=-1)  # (B, N, 6)
        elif C == 6:
            surface = points
        else:
            raise ValueError(
                f"Michelangelo encoder expects 3 or 6 channels, got C={C}. "
                "Use xyz or xyz+normals."
            )

        # Sample / pad along N
        if N > self.num_points:
            idx = torch.randperm(N, device=points.device)[: self.num_points]
            surface = surface[:, idx, :]
        elif N < self.num_points:
            # Pad by sampling with replacement
            pad_idx = torch.randint(0, N, (B, self.num_points - N), device=points.device)
            pad = torch.gather(
                surface,
                1,
                pad_idx.unsqueeze(-1).expand(-1, -1, surface.shape[-1]),
            )
            surface = torch.cat([surface, pad], dim=1)

        # Ensure final shape (B, num_points, 6) and float32 for Michelangelo
        surface = surface[:, : self.num_points, :].contiguous().to(torch.float32)
        return surface

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to features.

        Args:
            points: (batch_size, num_points, C), C=3 or 6.

        Returns:
            Point features of shape (batch_size, 1, output_dim)
        """
        if self.use_michelangelo:
            # --- Michelangelo global shape embedding ---
            surface = self._prepare_surface_for_michelangelo(points)
            with torch.no_grad():
                shape_embed, _ = self.michelangelo_model.model.encode_shape_embed(
                    surface, return_latents=True
                )
            # shape_embed: (B, D) → (B, 1, D) as a single "point token"
            x = shape_embed.unsqueeze(1)
            return x

        # --- Original PointNet-style encoder path ---
        # (B, N, C) -> (B, C, N) for Conv1d
        x = points.transpose(1, 2)  # (B, input_dim, N)

        # Per-point feature extraction
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.relu(self.bn3(self.conv3(x)))  # (B, hidden_dim, N)

        # Global max pooling
        x = torch.max(x, dim=2, keepdim=True)[0]  # (B, hidden_dim, 1)

        # Global feature transformation
        x = self.relu(self.bn4(self.conv4(x)))  # (B, hidden_dim, 1)
        x = self.bn5(self.conv5(x))            # (B, output_dim, 1)

        # (B, output_dim, 1) -> (B, 1, output_dim)
        x = x.transpose(1, 2)
        return x
