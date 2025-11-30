"""
Attention-based fusion layer for multi-timeframe representations.

Combines the 15m, 1H, and 4H encoder outputs using cross-attention,
where the 15m (fastest) timeframe queries the higher timeframes for context.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AttentionFusion(nn.Module):
    """
    Cross-attention fusion for multi-timeframe representations.

    The 15m timeframe acts as the query (what to trade now),
    while 1H and 4H provide keys/values (broader context).

    Architecture:
        - Query: 15m representation
        - Keys/Values: Concatenated 1H and 4H representations
        - Output: Fused context vector
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (must match encoder output)
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Projection layers for each timeframe
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid()
        )

    def forward(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multi-timeframe representations.

        Args:
            enc_15m: 15-minute encoder output [batch, d_model]
            enc_1h: 1-hour encoder output [batch, d_model]
            enc_4h: 4-hour encoder output [batch, d_model]

        Returns:
            Fused context vector [batch, d_model]
        """
        batch_size = enc_15m.size(0)

        # Reshape to sequence format for attention: [batch, seq_len, d_model]
        # 15m is the query (what action to take now)
        query = self.query_proj(enc_15m).unsqueeze(1)  # [batch, 1, d_model]

        # 1H and 4H are keys/values (context)
        context = torch.stack([enc_1h, enc_4h], dim=1)  # [batch, 2, d_model]
        key = self.key_proj(context)
        value = self.value_proj(context)

        # Cross-attention: 15m queries higher timeframes
        attended, attention_weights = self.cross_attention(
            query, key, value
        )  # [batch, 1, d_model]

        # Remove sequence dimension
        attended = attended.squeeze(1)  # [batch, d_model]

        # Residual connection with 15m
        fused = self.norm1(attended + enc_15m)

        # Output projection with residual
        output = fused + self.output_proj(fused)
        output = self.norm2(output)

        # Adaptive gating: learn how much to use each timeframe
        gate_input = torch.cat([enc_15m, enc_1h, enc_4h], dim=1)
        gate_weights = self.gate(gate_input)  # [batch, d_model]

        # Apply gate
        output = output * gate_weights

        return output

    def forward_with_weights(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights for interpretability.

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = enc_15m.size(0)

        query = self.query_proj(enc_15m).unsqueeze(1)
        context = torch.stack([enc_1h, enc_4h], dim=1)
        key = self.key_proj(context)
        value = self.value_proj(context)

        attended, attention_weights = self.cross_attention(query, key, value)
        attended = attended.squeeze(1)

        fused = self.norm1(attended + enc_15m)
        output = fused + self.output_proj(fused)
        output = self.norm2(output)

        gate_input = torch.cat([enc_15m, enc_1h, enc_4h], dim=1)
        gate_weights = self.gate(gate_input)
        output = output * gate_weights

        return output, attention_weights.squeeze(1)  # [batch, 2] weights for 1H and 4H


class SimpleFusion(nn.Module):
    """
    Simpler fusion alternative using learned weighted sum.

    Use this if AttentionFusion causes memory issues.
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()

        # Learnable weights for each timeframe
        self.weights = nn.Parameter(torch.ones(3) / 3)

        # Projection and normalization
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> torch.Tensor:
        """Weighted sum fusion."""
        weights = torch.softmax(self.weights, dim=0)

        fused = (
            weights[0] * enc_15m +
            weights[1] * enc_1h +
            weights[2] * enc_4h
        )

        return self.projection(fused)


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion.

    Concatenates all timeframe representations and projects to context dimension.
    """

    def __init__(
        self,
        d_model: int = 64,
        context_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # 3 timeframes concatenated
        self.projection = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, context_dim),
            nn.LayerNorm(context_dim)
        )

    def forward(
        self,
        enc_15m: torch.Tensor,
        enc_1h: torch.Tensor,
        enc_4h: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate and project."""
        concat = torch.cat([enc_15m, enc_1h, enc_4h], dim=1)
        return self.projection(concat)
