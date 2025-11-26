"""Projection layers for aligning modality features to LLM space."""

import torch
import torch.nn as nn
from typing import Optional


class MLPProjector(nn.Module):
    """Multi-layer perceptron projector for feature alignment.
    
    This projector aligns features from different modalities (image, point cloud)
    into the LLM's feature space. It consists of multiple linear layers with
    activation functions and optional layer normalization.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension (should match LLM's hidden size)
        hidden_dim: Hidden dimension for intermediate layers
        num_layers: Number of layers in the MLP
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Final layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            # Single layer case
            layers = [nn.Linear(input_dim, output_dim)]
        
        self.projector = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features to output space.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Projected tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.projector(x)


class IdentityProjector(nn.Module):
    """Identity projector that returns input unchanged.
    
    This is useful for text inputs that are already in the LLM's feature space.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
