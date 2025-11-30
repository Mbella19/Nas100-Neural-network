"""
Metrics calculations for training monitoring and evaluation.

Features:
- Direction accuracy (up/down prediction)
- Regression metrics (MSE, MAE, R², MAPE)
- Trading-specific metrics (Sharpe, Sortino, Win Rate)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DirectionAccuracy:
    """Direction prediction accuracy metrics."""
    accuracy: float          # Overall direction accuracy
    up_precision: float      # Precision for up predictions
    up_recall: float         # Recall for up moves
    down_precision: float    # Precision for down predictions
    down_recall: float       # Recall for down moves
    neutral_rate: float      # Rate of near-zero predictions

    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'up_precision': self.up_precision,
            'up_recall': self.up_recall,
            'down_precision': self.down_precision,
            'down_recall': self.down_recall,
            'neutral_rate': self.neutral_rate
        }


@dataclass
class RegressionMetrics:
    """Standard regression metrics."""
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float  # Mean Absolute Percentage Error

    def to_dict(self) -> Dict[str, float]:
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'mape': self.mape
        }


@dataclass
class TradingMetrics:
    """Trading-specific performance metrics."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int

    def to_dict(self) -> Dict[str, float]:
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_trades': self.total_trades
        }


def calculate_direction_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.0
) -> DirectionAccuracy:
    """
    Calculate direction prediction accuracy.

    Args:
        predictions: Model predictions
        targets: Actual values
        threshold: Threshold for considering prediction as neutral

    Returns:
        DirectionAccuracy with all direction metrics
    """
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    # Direction classification
    pred_up = predictions > threshold
    pred_down = predictions < -threshold
    pred_neutral = ~pred_up & ~pred_down

    actual_up = targets > 0
    actual_down = targets < 0

    # Overall direction accuracy (ignoring magnitude)
    correct_direction = ((pred_up & actual_up) | (pred_down & actual_down))
    accuracy = correct_direction.sum() / len(targets) if len(targets) > 0 else 0.0

    # Up precision and recall
    up_true_positive = (pred_up & actual_up).sum()
    up_precision = up_true_positive / pred_up.sum() if pred_up.sum() > 0 else 0.0
    up_recall = up_true_positive / actual_up.sum() if actual_up.sum() > 0 else 0.0

    # Down precision and recall
    down_true_positive = (pred_down & actual_down).sum()
    down_precision = down_true_positive / pred_down.sum() if pred_down.sum() > 0 else 0.0
    down_recall = down_true_positive / actual_down.sum() if actual_down.sum() > 0 else 0.0

    # Neutral rate
    neutral_rate = pred_neutral.sum() / len(predictions) if len(predictions) > 0 else 0.0

    return DirectionAccuracy(
        accuracy=float(accuracy),
        up_precision=float(up_precision),
        up_recall=float(up_recall),
        down_precision=float(down_precision),
        down_recall=float(down_recall),
        neutral_rate=float(neutral_rate)
    )


def calculate_r2_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination).

    Args:
        predictions: Model predictions
        targets: Actual values

    Returns:
        R² score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def calculate_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> RegressionMetrics:
    """
    Calculate comprehensive regression metrics.

    Args:
        predictions: Model predictions
        targets: Actual values

    Returns:
        RegressionMetrics with all metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = predictions.flatten()
    targets = targets.flatten()

    # MSE and RMSE
    mse = float(np.mean((predictions - targets) ** 2))
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(predictions - targets)))

    # R²
    r2 = calculate_r2_score(predictions, targets)

    # MAPE (avoid division by zero)
    non_zero_mask = np.abs(targets) > 1e-10
    if non_zero_mask.sum() > 0:
        mape = float(np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100)
    else:
        mape = 0.0

    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mape=mape
    )


def calculate_trading_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> TradingMetrics:
    """
    Calculate trading performance metrics.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        TradingMetrics with all trading metrics
    """
    if isinstance(returns, torch.Tensor):
        returns = returns.detach().cpu().numpy()

    returns = returns.flatten()

    if len(returns) == 0:
        return TradingMetrics(
            total_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
            avg_win=0.0, avg_loss=0.0, total_trades=0
        )

    # Total return
    total_return = float(np.sum(returns))

    # Sharpe ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = 0.0
    if np.std(excess_returns) > 0:
        sharpe_ratio = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year))

    # Sortino ratio (downside deviation)
    # FIXED: Use correct formula - RMS of min(returns, 0), not std of negative returns only
    # This matches the formula in evaluation/metrics.py for consistency
    downside_returns = np.minimum(excess_returns, 0)  # Set positive returns to 0
    downside_std = np.sqrt(np.mean(downside_returns ** 2))  # RMS (root mean square)
    sortino_ratio = 0.0
    if downside_std > 1e-10:
        sortino_ratio = float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))
        # Cap to prevent inf in logs
        sortino_ratio = min(sortino_ratio, 100.0)

    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Win rate
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    total_trades = len(returns[returns != 0])
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

    # Average win/loss
    avg_win = float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0
    avg_loss = float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
    gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    return TradingMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=float(win_rate),
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=total_trades
    )


class MetricsTracker:
    """
    Track and aggregate metrics over training.

    Usage:
        tracker = MetricsTracker()
        tracker.update(predictions, targets)
        metrics = tracker.compute()
    """

    def __init__(self):
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.losses: List[float] = []

    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        loss: Optional[float] = None
    ):
        """Add batch predictions and targets."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(predictions.flatten())
        self.targets.append(targets.flatten())
        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated data."""
        if len(self.predictions) == 0:
            return {}

        all_predictions = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)

        # Direction accuracy
        dir_metrics = calculate_direction_accuracy(all_predictions, all_targets)

        # Regression metrics
        reg_metrics = calculate_regression_metrics(all_predictions, all_targets)

        result = {
            'direction_accuracy': dir_metrics.accuracy,
            'up_precision': dir_metrics.up_precision,
            'up_recall': dir_metrics.up_recall,
            'down_precision': dir_metrics.down_precision,
            'down_recall': dir_metrics.down_recall,
            'mse': reg_metrics.mse,
            'rmse': reg_metrics.rmse,
            'mae': reg_metrics.mae,
            'r2': reg_metrics.r2
        }

        if len(self.losses) > 0:
            result['avg_loss'] = float(np.mean(self.losses))

        return result

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get concatenated predictions and targets."""
        if len(self.predictions) == 0:
            return np.array([]), np.array([])
        return np.concatenate(self.predictions), np.concatenate(self.targets)


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm for a model.

    Args:
        model: PyTorch model

    Returns:
        Total gradient L2 norm
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return float(total_norm ** 0.5)


def compute_prediction_stats(predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics about predictions for debugging.

    Args:
        predictions: Model predictions

    Returns:
        Dictionary of statistics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    predictions = predictions.flatten()

    return {
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'median': float(np.median(predictions)),
        'pct_positive': float((predictions > 0).sum() / len(predictions)),
        'pct_negative': float((predictions < 0).sum() / len(predictions)),
        'pct_near_zero': float((np.abs(predictions) < 0.001).sum() / len(predictions))
    }
