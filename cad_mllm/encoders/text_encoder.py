"""Text encoder using LLM's embedding layer."""

import torch
import torch.nn as nn
from typing import Optional


class TextEncoder(nn.Module):
    """Text encoder that uses the LLM's embedding layer.

    For text inputs, we directly use the LLM's word embeddings rather than
    a separate encoder. This is because text is already in a format that
    the LLM can naturally process.

    Args:
        embedding_layer: The embedding layer from the LLM
        freeze: Whether to freeze the embedding layer
    """

    def __init__(self, embedding_layer: nn.Embedding, freeze: bool = False):
        super().__init__()
        self.embedding = embedding_layer

        if freeze:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text input IDs to embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)

        Returns:
            Text embeddings of shape (batch_size, seq_len, hidden_dim)
        """
        return self.embedding(input_ids)
