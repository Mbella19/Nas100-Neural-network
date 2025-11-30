"""
Training script for the Market Analyst (supervised learning).

Trains the Analyst to predict smoothed future returns across
multiple timeframes. After training, the model is frozen for
use with the RL agent.

Memory-optimized for Apple M2 Silicon.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import gc
import logging

from ..models.analyst import MarketAnalyst, create_analyst
from ..data.features import create_smoothed_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTimeframeDataset(Dataset):
    """
    Dataset for multi-timeframe analyst training.

    Each sample contains:
    - 15m features with lookback window
    - 1H features with lookback window
    - 4H features with lookback window
    - Smoothed future return target
    """

    def __init__(
        self,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_4h: pd.DataFrame,
        feature_cols: List[str],
        target: pd.Series,
        lookback_15m: int = 48,
        lookback_1h: int = 24,
        lookback_4h: int = 12
    ):
        """
        Args:
            df_15m: 15-minute DataFrame
            df_1h: 1-hour DataFrame (aligned to 15m index)
            df_4h: 4-hour DataFrame (aligned to 15m index)
            feature_cols: Feature columns to use
            target: Smoothed future return target
            lookback_*: Lookback windows for each timeframe
        """
        self.lookback_15m = lookback_15m
        self.lookback_1h = lookback_1h
        self.lookback_4h = lookback_4h

        # Get feature matrices
        self.features_15m = df_15m[feature_cols].values.astype(np.float32)
        self.features_1h = df_1h[feature_cols].values.astype(np.float32)
        self.features_4h = df_4h[feature_cols].values.astype(np.float32)
        self.targets = target.values.astype(np.float32)

        # Valid indices (need enough lookback and valid target)
        self.start_idx = max(lookback_15m, lookback_1h * 4, lookback_4h * 16)
        self.valid_mask = ~np.isnan(self.targets[self.start_idx:])
        self.valid_indices = np.where(self.valid_mask)[0] + self.start_idx

        logger.info(f"Dataset created with {len(self.valid_indices)} valid samples")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]

        # Get lookback windows
        x_15m = self.features_15m[actual_idx - self.lookback_15m:actual_idx]
        x_1h = self.features_1h[actual_idx - self.lookback_1h:actual_idx]
        x_4h = self.features_4h[actual_idx - self.lookback_4h:actual_idx]

        # Target
        y = self.targets[actual_idx]

        return (
            torch.tensor(x_15m, dtype=torch.float32),
            torch.tensor(x_1h, dtype=torch.float32),
            torch.tensor(x_4h, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32)
        )


class AnalystTrainer:
    """
    Trainer class for the Market Analyst model.

    Features:
    - AdamW optimizer with weight decay
    - Huber loss for robustness
    - Early stopping
    - Memory-efficient batch processing
    - Checkpoint saving
    """

    def __init__(
        self,
        model: MarketAnalyst,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 10,
        cache_clear_interval: int = 50
    ):
        """
        Args:
            model: MarketAnalyst model
            device: Torch device
            learning_rate: Learning rate
            weight_decay: AdamW weight decay
            patience: Early stopping patience
            cache_clear_interval: Clear MPS cache every N batches
        """
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.cache_clear_interval = cache_clear_interval

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss function (Huber is more robust to outliers)
        self.criterion = nn.HuberLoss(delta=1.0)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x_15m, x_1h, x_4h, targets) in enumerate(pbar):
            # Move to device
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            _, predictions = self.model(x_15m, x_1h, x_4h)
            loss = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Memory cleanup
            if batch_idx % self.cache_clear_interval == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()

            pbar.set_postfix({'loss': loss.item()})

            # Clean up batch tensors
            del x_15m, x_1h, x_4h, targets, predictions, loss

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x_15m, x_1h, x_4h, targets in val_loader:
            x_15m = x_15m.to(self.device)
            x_1h = x_1h.to(self.device)
            x_4h = x_4h.to(self.device)
            targets = targets.to(self.device)

            _, predictions = self.model(x_15m, x_1h, x_4h)
            loss = self.criterion(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

            del x_15m, x_1h, x_4h, targets, predictions

        return total_loss / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum epochs
            save_path: Path to save best model

        Returns:
            Training history
        """
        logger.info(f"Starting training for up to {max_epochs} epochs")

        for epoch in range(1, max_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Update scheduler
            self.scheduler.step(val_loss)

            logger.info(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                if save_path:
                    self.save_checkpoint(save_path, epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Memory cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': {
                'd_model': self.model.d_model,
                'context_dim': self.model.context_dim
            }
        }

        filename = 'best.pt' if is_best else f'epoch_{epoch}.pt'
        torch.save(checkpoint, path / filename)
        logger.info(f"Saved checkpoint to {path / filename}")


def train_analyst(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    feature_cols: List[str],
    save_path: str,
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> Tuple[MarketAnalyst, Dict]:
    """
    Main function to train the Market Analyst.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features
        df_4h: 4-hour DataFrame with features
        feature_cols: Feature columns to use
        save_path: Path to save model
        config: AnalystConfig object
        device: Torch device

    Returns:
        Tuple of (trained model, training history)
    """
    # Default configuration
    if config is None:
        from config.settings import Config
        config = Config().analyst

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training on device: {device}")

    # Create target
    logger.info("Creating smoothed target...")
    target = create_smoothed_target(
        df_15m,
        future_window=config.future_window if hasattr(config, 'future_window') else 12,
        smooth_window=config.smooth_window if hasattr(config, 'smooth_window') else 12
    )

    # Create dataset
    logger.info("Creating dataset...")
    dataset = MultiTimeframeDataset(
        df_15m, df_1h, df_4h,
        feature_cols, target,
        lookback_15m=config.lookback_15m if hasattr(config, 'lookback_15m') else 48,
        lookback_1h=config.lookback_1h if hasattr(config, 'lookback_1h') else 24,
        lookback_4h=config.lookback_4h if hasattr(config, 'lookback_4h') else 12
    )

    # Split into train/validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Create data loaders
    batch_size = config.batch_size if hasattr(config, 'batch_size') else 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't support multiprocessing well
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Create model
    feature_dims = {
        '15m': len(feature_cols),
        '1h': len(feature_cols),
        '4h': len(feature_cols)
    }

    model = create_analyst(feature_dims, config, device)
    logger.info(f"Created MarketAnalyst with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = AnalystTrainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate if hasattr(config, 'learning_rate') else 1e-4,
        weight_decay=config.weight_decay if hasattr(config, 'weight_decay') else 1e-5,
        patience=config.patience if hasattr(config, 'patience') else 10
    )

    # Train
    history = trainer.train(
        train_loader,
        val_loader,
        max_epochs=config.max_epochs if hasattr(config, 'max_epochs') else 100,
        save_path=save_path
    )

    # Load best model
    best_path = Path(save_path) / 'best.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    return model, history


if __name__ == '__main__':
    # Example usage
    print("Use this module via: python -m src.training.train_analyst")
    print("Or import and call train_analyst() function")
