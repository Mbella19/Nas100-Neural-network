"""
Market Analyst Model - Complete supervised learning module.

The Analyst produces a dense "Context Vector" that summarizes market state
across all timeframes. It predicts smoothed future returns to learn
sustained momentum rather than noise.

After training, the Analyst is frozen and its context vector is consumed
by the PPO Sniper Agent.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import gc

from .encoders import TransformerEncoder, LightweightEncoder
from .fusion import AttentionFusion, SimpleFusion, ConcatFusion


class MarketAnalyst(nn.Module):
    """
    Multi-timeframe Market Analyst with Transformer encoders.

    Architecture:
        - 3 TransformerEncoders (15m, 1H, 4H)
        - AttentionFusion layer
        - Trend prediction head (for training)

    Usage:
        Training: Use forward() to get both context and prediction
        Inference: Use get_context() for frozen context vector
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        context_dim: int = 64,
        dropout: float = 0.1,
        use_lightweight: bool = False
    ):
        """
        Args:
            feature_dims: Dict mapping timeframe to input feature dimension
                         e.g., {'15m': 12, '1h': 12, '4h': 12}
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            context_dim: Output context vector dimension
            dropout: Dropout rate
            use_lightweight: Use lighter Conv1D encoder instead of Transformer
        """
        super().__init__()

        self.d_model = d_model
        self.context_dim = context_dim

        # Choose encoder type
        EncoderClass = LightweightEncoder if use_lightweight else TransformerEncoder

        # Create encoder for each timeframe
        if use_lightweight:
            self.encoder_15m = LightweightEncoder(
                feature_dims.get('15m', 12),
                hidden_dim=d_model,
                dropout=dropout
            )
            self.encoder_1h = LightweightEncoder(
                feature_dims.get('1h', 12),
                hidden_dim=d_model,
                dropout=dropout
            )
            self.encoder_4h = LightweightEncoder(
                feature_dims.get('4h', 12),
                hidden_dim=d_model,
                dropout=dropout
            )
        else:
            self.encoder_15m = TransformerEncoder(
                feature_dims.get('15m', 12),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            self.encoder_1h = TransformerEncoder(
                feature_dims.get('1h', 12),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            self.encoder_4h = TransformerEncoder(
                feature_dims.get('4h', 12),
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )

        # Fusion layer
        self.fusion = AttentionFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )

        # Context projection (optional, if d_model != context_dim)
        if d_model != context_dim:
            self.context_proj = nn.Sequential(
                nn.Linear(d_model, context_dim),
                nn.LayerNorm(context_dim)
            )
        else:
            self.context_proj = nn.Identity()

        # Trend prediction head (for supervised training)
        self.trend_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim // 2, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for module in [self.trend_head, self.context_proj]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

    def forward(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]

        Returns:
            Tuple of:
                - context: Context vector [batch, context_dim]
                - prediction: Trend prediction [batch, 1]
        """
        # Ensure float32
        x_15m = x_15m.float()
        x_1h = x_1h.float()
        x_4h = x_4h.float()

        # Encode each timeframe
        enc_15m = self.encoder_15m(x_15m)  # [batch, d_model]
        enc_1h = self.encoder_1h(x_1h)
        enc_4h = self.encoder_4h(x_4h)

        # Fuse timeframes
        fused = self.fusion(enc_15m, enc_1h, enc_4h)  # [batch, d_model]

        # Project to context dimension
        context = self.context_proj(fused)  # [batch, context_dim]

        # Predict trend
        prediction = self.trend_head(context)  # [batch, 1]

        return context, prediction

    @torch.no_grad()
    def get_context(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Get context vector only (for inference/RL agent).

        This should be used after freezing the model.

        Args:
            x_15m: 15-minute features [batch, seq_len, features]
            x_1h: 1-hour features [batch, seq_len, features]
            x_4h: 4-hour features [batch, seq_len, features]

        Returns:
            Context vector [batch, context_dim]
        """
        context, _ = self.forward(x_15m, x_1h, x_4h)
        return context

    def freeze(self):
        """
        Freeze all parameters for use with RL agent.

        Call this after training is complete.
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters for fine-tuning."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True

    def get_attention_weights(
        self,
        x_15m: torch.Tensor,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor
    ) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Returns:
            Attention weights [batch, 2] showing attention to 1H and 4H
        """
        with torch.no_grad():
            enc_15m = self.encoder_15m(x_15m)
            enc_1h = self.encoder_1h(x_1h)
            enc_4h = self.encoder_4h(x_4h)
            _, weights = self.fusion.forward_with_weights(enc_15m, enc_1h, enc_4h)
        return weights


def create_analyst(
    feature_dims: Dict[str, int],
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> MarketAnalyst:
    """
    Factory function to create a Market Analyst with config.

    Args:
        feature_dims: Feature dimensions per timeframe
        config: AnalystConfig object (optional)
        device: Target device

    Returns:
        MarketAnalyst model
    """
    if config is None:
        # Default configuration
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            context_dim=64,
            dropout=0.1
        )
    else:
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            context_dim=config.context_dim,
            dropout=config.dropout
        )

    if device is not None:
        model = model.to(device)

    return model


def load_analyst(
    path: str,
    feature_dims: Dict[str, int],
    device: Optional[torch.device] = None,
    freeze: bool = True
) -> MarketAnalyst:
    """
    Load a trained Market Analyst from checkpoint.

    Args:
        path: Path to checkpoint file
        feature_dims: Feature dimensions (must match training)
        device: Target device
        freeze: Whether to freeze the model

    Returns:
        Loaded MarketAnalyst
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device or 'cpu')

    # Create model
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MarketAnalyst(
            feature_dims=feature_dims,
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dim_feedforward=config.get('dim_feedforward', 128),
            context_dim=config.get('context_dim', 64),
            dropout=config.get('dropout', 0.1)
        )
    else:
        model = create_analyst(feature_dims)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    if device is not None:
        model = model.to(device)

    if freeze:
        model.freeze()

    # Clear memory
    del checkpoint
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return model
