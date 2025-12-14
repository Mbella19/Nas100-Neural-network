import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, List
import gc

from .fusion import AttentionFusion, ConcatFusion
from .component_encoder import CrossAssetModule

class TCNBlock(nn.Module):
    """
    Residual TCN Block: Dilated Conv1D -> WeightNorm -> ReLU -> Dropout
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp1 = nn.Conv1d(out_channels, out_channels, 1) # Dummy chomp implemented via slicing below
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
        # Calculate amount to chomp (remove from end)
        self.padding = padding

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] # Chomp
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.padding] # Chomp
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """
    Stack of TCN Blocks.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(
                in_channels, out_channels, kernel_size, 
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, 
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNAnalyst(nn.Module):
    """
    Market Analyst using Temporal Convolutional Networks (TCN).
    Processing independent timeframes and fusing them.
    """
    def __init__(
        self,
        feature_dims: Dict[str, int],
        d_model: int = 64,
        context_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2,
        num_blocks: int = 3,  # Depth of TCN
        kernel_size: int = 3,
        tcn_num_channels: Optional[List[int]] = None,
        use_cross_asset_attention: bool = True,
        d_component: int = 16,
        component_seq_len: int = 12,
        n_components: int = 6,
        component_input_dim: int = 4
    ):
        super().__init__()
        self.architecture = "tcn"
        self.feature_dims = feature_dims
        self.d_model = d_model
        self.context_dim = context_dim
        self.num_classes = num_classes
        self.use_cross_asset_attention = use_cross_asset_attention
        # Store TCN hyperparams for checkpointing/loading
        self.tcn_num_blocks = num_blocks
        self.tcn_kernel_size = kernel_size
        # Store cross-asset params for checkpointing/loading
        self.d_component = d_component if use_cross_asset_attention else 0
        self.component_seq_len = component_seq_len
        self.n_components = n_components
        self.component_input_dim = component_input_dim

        # Feature Projections (Input -> d_model)
        self.projections = nn.ModuleDict({
            tf: nn.Linear(dim, d_model) 
            for tf, dim in feature_dims.items()
        })

        # TCN Encoders for each timeframe
        # Default: increasing channel depth per block for higher capacity.
        if tcn_num_channels is not None and len(tcn_num_channels) > 0:
            num_channels = list(tcn_num_channels)
            if len(num_channels) != num_blocks:
                # Backwards/defensive: if list length doesn't match, fall back to constant
                num_channels = [d_model] * num_blocks
        else:
            # Increasing channel depth: [d_model, d_model*2, d_model*4, ...]
            num_channels = [d_model * (2 ** i) for i in range(num_blocks)]
            # Ensure convergence back to d_model for fusion
            num_channels[-1] = d_model

        # Store for checkpointing / loading
        self.tcn_num_channels = num_channels
        
        self.tcns = nn.ModuleDict({
            tf: TemporalConvNet(
                num_inputs=d_model,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
            for tf in feature_dims.keys()
        })

        # Fusion Mechanism
        self.fusion = AttentionFusion(d_model=d_model, nhead=4)

        # Cross-Asset Attention Module
        if use_cross_asset_attention:
            self.cross_asset_module = CrossAssetModule(
                d_model=d_model,
                d_component=d_component,
                component_input_dim=component_input_dim,
                component_seq_len=component_seq_len,
                n_heads=2,
                dropout=dropout
            )
            # Input to context proj is fused + component_summary
            combined_dim = d_model + d_component
        else:
            self.cross_asset_module = None
            combined_dim = d_model

        # Final Context Projection
        self.context_proj = nn.Sequential(
            nn.Linear(combined_dim, context_dim),
            nn.ReLU(),
            nn.LayerNorm(context_dim)
        )

        # Prediction Heads
        self.direction_head = nn.Linear(context_dim, num_classes if num_classes > 2 else 1)
        self.volatility_head = nn.Linear(context_dim, 1)
        self.regime_head = nn.Linear(context_dim, 3) # Bull/Bear/Range
        
        # Multi-horizon heads (optional)
        self.horizon_1h_head = nn.Linear(context_dim, 1)
        self.horizon_2h_head = nn.Linear(context_dim, 1)

    def _process_timeframe(self, x: torch.Tensor, tf: str) -> torch.Tensor:
        """
        Process a single timeframe through Projection -> TCN -> Pooling.
        """
        if x.dim() == 2:
             x = x.unsqueeze(1)
        
        x_proj = self.projections[tf](x)
        x_tcn_in = x_proj.transpose(1, 2)
        x_tcn_out = self.tcns[tf](x_tcn_in)
        return x_tcn_out[:, :, -1]

    def _encode_and_fuse(
        self, 
        x_5m: torch.Tensor, 
        x_15m: torch.Tensor, 
        x_45m: torch.Tensor,
        component_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encodes timeframes and optionally applies cross-asset attention.
        """
        encodings = {}
        
        enc_5m = self._process_timeframe(x_5m, '5m')
        enc_15m = self._process_timeframe(x_15m, '15m')
        enc_45m = self._process_timeframe(x_45m, '45m')

        encodings['5m'] = enc_5m
        encodings['15m'] = enc_15m
        encodings['45m'] = enc_45m

        # Fuse multi-timeframe NAS100 context
        fused = self.fusion(enc_5m, enc_15m, enc_45m)  # [batch, d_model]

        # Cross-asset attention summary (optional)
        if self.use_cross_asset_attention and self.cross_asset_module is not None and component_data is not None:
            # UPDATED: unpacking tuple (summary, weights)
            component_summary, attn_weights = self.cross_asset_module(
                nas100_context=fused,
                component_data=component_data
            )  # [batch, d_component]
            enhanced = torch.cat([fused, component_summary], dim=-1)
            encodings['attention_weights'] = attn_weights
        else:
            enhanced = fused

        context = self.context_proj(enhanced)  # [batch, context_dim]

        return context, encodings

    def forward(
        self,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor,
        x_45m: torch.Tensor,
        component_data: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_multi_horizon: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass.
        """
        context, activations = self._encode_and_fuse(x_5m, x_15m, x_45m, component_data)

        direction = self.direction_head(context)

        if not return_aux and not return_multi_horizon:
            return context, direction

        result = [context, direction]

        if return_aux:
            volatility = self.volatility_head(context).squeeze(-1)
            regime = self.regime_head(context).squeeze(-1)
            result.extend([volatility, regime])

        if return_multi_horizon and self.horizon_1h_head is not None:
            horizon_1h = self.horizon_1h_head(context)
            horizon_2h = self.horizon_2h_head(context)
            result.extend([horizon_1h, horizon_2h])

        return tuple(result)

    @torch.no_grad()
    def get_context(
        self,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor,
        x_45m: torch.Tensor,
        component_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get context vector only."""
        context, _ = self.forward(x_5m, x_15m, x_45m, component_data=component_data)
        return context

    @torch.no_grad()
    def get_probabilities(
        self,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor,
        x_45m: torch.Tensor,
        component_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get context vector AND probabilities AND attention weights.
        Returns:
            context: [batch, context_dim]
            probs: [batch, num_classes]
            attn_weights: [batch, n_heads, 1, n_components] or None
        """
        context, activations = self._encode_and_fuse(x_5m, x_15m, x_45m, component_data=component_data)
        logits = self.direction_head(context)

        if self.num_classes == 2:
            p_up = torch.sigmoid(logits)
            p_down = 1 - p_up
            probs = torch.cat([p_down, p_up], dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)

        return context, probs, activations.get('attention_weights')

    @torch.no_grad()
    def get_activations(
        self,
        x_5m: torch.Tensor,
        x_15m: torch.Tensor,
        x_45m: torch.Tensor,
        component_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get context vector AND internal activations."""
        context, activations = self._encode_and_fuse(x_5m, x_15m, x_45m, component_data=component_data)
        return context, activations

    def freeze(self):
        """Freeze all parameters."""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze parameters."""
        self.train()
        for param in self.parameters():
            param.requires_grad = True

def create_tcn_analyst(
    feature_dims: Dict[str, int],
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> TCNAnalyst:
    """Factory function."""
    if config is None:
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=64,
            context_dim=64,
            dropout=0.3,
            num_classes=2,
            num_blocks=3,
            kernel_size=3,
            use_cross_asset_attention=True,
            d_component=16,
            component_seq_len=12,
            n_components=6,
            component_input_dim=4
        )
    else:
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=getattr(config, 'd_model', 64),
            context_dim=getattr(config, 'context_dim', 64),
            dropout=getattr(config, 'dropout', 0.3),
            num_classes=getattr(config, 'num_classes', 2),
            num_blocks=getattr(config, 'tcn_num_blocks', 3),
            kernel_size=getattr(config, 'tcn_kernel_size', 3),
            use_cross_asset_attention=getattr(config, 'use_cross_asset_attention', True),
            d_component=getattr(config, 'd_component', 16),
            component_seq_len=getattr(config, 'component_seq_len', 12),
            n_components=getattr(config, 'n_components', 6),
            component_input_dim=getattr(config, 'component_input_dim', 4)
        )

    if device is not None:
        model = model.to(device)

    return model

def load_tcn_analyst(
    path: str,
    feature_dims: Dict[str, int],
    device: Optional[torch.device] = None,
    freeze: bool = True,
    use_cross_asset_attention: Optional[bool] = None
) -> TCNAnalyst:
    """Load TCN Analyst from checkpoint."""
    checkpoint = torch.load(path, map_location=device or 'cpu')

    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        
        if use_cross_asset_attention is not None:
            cross_asset = use_cross_asset_attention
        else:
            cross_asset = saved_config.get('use_cross_asset_attention', False)

        d_model = saved_config.get('d_model', 64)
        num_blocks = saved_config.get('tcn_num_blocks', 3)

        # Legacy support: allow checkpoint to specify exact channel schedule.
        # If missing, infer from state_dict or fall back to constant channels.
        tcn_num_channels = saved_config.get('tcn_num_channels')
        if tcn_num_channels is None:
            inferred: List[int] = []
            state_dict = checkpoint.get('model_state_dict', {})
            # Try to find a timeframe key present in the checkpoint
            tf_key = None
            for k in state_dict.keys():
                if k.startswith("tcns.") and ".network.0.conv1.bias" in k:
                    parts = k.split(".")
                    if len(parts) > 1:
                        tf_key = parts[1]
                        break
            if tf_key is not None:
                for i in range(num_blocks):
                    key = f"tcns.{tf_key}.network.{i}.conv1.bias"
                    if key in state_dict:
                        inferred.append(int(state_dict[key].shape[0]))
            if len(inferred) == num_blocks:
                tcn_num_channels = inferred
            else:
                tcn_num_channels = [d_model] * num_blocks

        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=d_model,
            context_dim=saved_config.get('context_dim', 64),
            dropout=saved_config.get('dropout', 0.3),
            num_classes=saved_config.get('num_classes', 2),
            num_blocks=num_blocks,
            kernel_size=saved_config.get('tcn_kernel_size', 3),
            tcn_num_channels=tcn_num_channels,
            use_cross_asset_attention=cross_asset,
            d_component=saved_config.get('d_component', 16),
            component_seq_len=saved_config.get('component_seq_len', 12),
            n_components=saved_config.get('n_components', 6),
            component_input_dim=saved_config.get('component_input_dim', 4)
        )
    else:
        model = TCNAnalyst(
            feature_dims=feature_dims,
            d_model=64,
            context_dim=64,
            dropout=0.3,
            num_classes=2,
            num_blocks=3,
            kernel_size=3,
            use_cross_asset_attention=use_cross_asset_attention if use_cross_asset_attention is not None else False
        )

    # Load state dict with shape-mismatch tolerance
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        # Filter out incompatible shapes (e.g., architecture change) and retry.
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = len(state_dict) - len(compatible)
        if skipped > 0:
            print(f"⚠️ load_tcn_analyst: skipped {skipped} mismatched keys. Retraining recommended.")
        model.load_state_dict(compatible, strict=False)

    if device is not None:
        model = model.to(device)

    if freeze:
        model.freeze()

    del checkpoint
    gc.collect()
    try:
        torch.mps.empty_cache()
    except:
        pass

    return model
