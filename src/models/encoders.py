"""
Transformer-based temporal encoders for multi-timeframe data.

Memory-optimized for Apple M2 Silicon (8GB RAM):
- d_model: 64 (not 128/256)
- nhead: 4 (not 8)
- num_layers: 2 (not 6)
- All tensors float32
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.

    Adds position-dependent signals to the input embeddings so the
    transformer can understand the temporal order of the sequence.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but moves with model)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for a single timeframe.

    Takes a sequence of feature vectors and produces a single
    context vector summarizing the temporal patterns.

    Architecture:
        Input projection → Positional encoding → Transformer layers → Pooling

    Memory Optimization:
        - Uses d_model=64, nhead=4, num_layers=2 by default
        - All computations in float32
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Input projection: map features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [batch, seq, features]
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            Encoded representation [batch, d_model]
        """
        # Ensure float32
        x = x.float()

        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transform
        x = self.transformer(x, mask=mask)  # [batch, seq_len, d_model]

        # Layer norm
        x = self.layer_norm(x)

        # Pool: use mean pooling over sequence
        # Could also use last token or [CLS] token approach
        output = x.mean(dim=1)  # [batch, d_model]

        return output

    def forward_sequence(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass returning full sequence (for attention fusion).

        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            mask: Optional attention mask

        Returns:
            Encoded sequence [batch, seq_len, d_model]
        """
        x = x.float()
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        return x


class LightweightEncoder(nn.Module):
    """
    Even lighter encoder using Conv1D + attention for memory-constrained scenarios.

    Use this if TransformerEncoder causes OOM on M2.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.conv_layers.append(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

        # Additional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Self-attention for global context
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            [batch, hidden_dim]
        """
        # Conv1D expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            residual = x if x.shape[1] == conv.out_channels else None
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if residual is not None:
                x = x + residual

        # Back to [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Pool
        return x.mean(dim=1)
