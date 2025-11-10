"""Image encoder for processing visual inputs."""

import torch
import torch.nn as nn
from typing import Optional
from transformers import Dinov2Model, AutoImageProcessor


class ImageEncoder(nn.Module):
    """Image encoder using DINOv2 vision model.

    DINOv2 is better for geometric and structural understanding,
    making it ideal for CAD applications.

    Args:
        model_name: Name of the DINOv2 model to use
        freeze: Whether to freeze the encoder weights
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-large",
        torch_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
    ):
        super().__init__()

        # Load pre-trained DINOv2 vision encoder
        self.encoder = Dinov2Model.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image inputs to features.

        Args:
            pixel_values: Image tensor of shape (batch_size, channels, height, width)
                         or (batch_size, num_views, channels, height, width) for multi-view

        Returns:
            Image features of shape (batch_size, num_patches, hidden_dim)
            or (batch_size, num_views * num_patches, hidden_dim) for multi-view
        """
        # Handle multi-view images
        if pixel_values.dim() == 5:  # (B, V, C, H, W)
            batch_size, num_views = pixel_values.shape[:2]
            # Reshape to (B*V, C, H, W)
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])

            # Encode
            outputs = self.encoder(pixel_values=pixel_values)
            features = outputs.last_hidden_state  # (B*V, num_patches+1, hidden_dim)

            # Remove [CLS] token, keep only patch tokens
            features = features[:, 1:, :]  # (B*V, num_patches, hidden_dim)

            # Reshape back to (B, V*num_patches, hidden_dim)
            num_patches = features.shape[1]
            features = features.reshape(batch_size, num_views * num_patches, -1)
        else:
            # Single view
            outputs = self.encoder(pixel_values=pixel_values)
            features = outputs.last_hidden_state  # (B, num_patches+1, hidden_dim)

            # Remove [CLS] token, keep only patch tokens
            features = features[:, 1:, :]  # (B, num_patches, hidden_dim)

        return features

    def preprocess(self, images):
        """Preprocess images using DINOv2 processor.

        Args:
            images: PIL images or list of PIL images

        Returns:
            Preprocessed pixel values ready for the encoder
        """
        return self.processor(images=images, return_tensors="pt")["pixel_values"]
