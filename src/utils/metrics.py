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


@dataclass
class ClassificationMetrics:
    """Classification metrics for discrete return buckets."""
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    support: List[int]
    confusion: np.ndarray
    direction_accuracy: float
    up_precision: float
    up_recall: float
    down_precision: float
    down_recall: float
    neutral_precision: float
    neutral_recall: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'direction_accuracy': self.direction_accuracy,
            'up_precision': self.up_precision,
            'up_recall': self.up_recall,
            'down_precision': self.down_precision,
            'down_recall': self.down_recall,
            'neutral_precision': self.neutral_precision,
            'neutral_recall': self.neutral_recall
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


def _classes_to_direction(
    labels: np.ndarray,
    up_classes: Tuple[int, ...],
    down_classes: Tuple[int, ...]
) -> np.ndarray:
    """Map discrete classes to directional labels (-1, 0, 1)."""
    direction = np.zeros_like(labels, dtype=int)
    direction[np.isin(labels, up_classes)] = 1
    direction[np.isin(labels, down_classes)] = -1
    return direction


def calculate_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    up_classes: Tuple[int, ...] = (3, 4),
    down_classes: Tuple[int, ...] = (0, 1)
) -> ClassificationMetrics:
    """
    Compute detailed metrics for multi-class directional classification.

    Args:
        predictions: Predicted class indices or logits/probabilities
        targets: True class indices
        num_classes: Number of classes
        up_classes: Class indices representing upward moves
        down_classes: Class indices representing downward moves
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Convert logits/probabilities to class indices if needed
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=1)

    predictions = predictions.astype(int).flatten()
    targets = targets.astype(int).flatten()

    # Filter out invalid targets (NaN encoded as -2147483648 when cast to int)
    valid_mask = (targets >= 0) & (targets < num_classes)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, tgt in zip(predictions, targets):
        if 0 <= pred < num_classes:
            confusion[tgt, pred] += 1

    support = confusion.sum(axis=1)
    total = support.sum()

    accuracy = float((predictions == targets).mean()) if len(predictions) > 0 else 0.0

    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_precision.append(float(precision))
        per_class_recall.append(float(recall))
        per_class_f1.append(float(f1))

    macro_precision = float(np.mean(per_class_precision)) if per_class_precision else 0.0
    macro_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

    # Directional metrics (Up / Down / Neutral)
    pred_dir = _classes_to_direction(predictions, up_classes, down_classes)
    tgt_dir = _classes_to_direction(targets, up_classes, down_classes)

    direction_accuracy = float((pred_dir == tgt_dir).mean()) if len(pred_dir) > 0 else 0.0

    def _precision_recall(pred_mask: np.ndarray, tgt_mask: np.ndarray) -> Tuple[float, float]:
        tp = (pred_mask & tgt_mask).sum()
        precision = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0.0
        recall = tp / tgt_mask.sum() if tgt_mask.sum() > 0 else 0.0
        return float(precision), float(recall)

    up_precision, up_recall = _precision_recall(pred_dir == 1, tgt_dir == 1)
    down_precision, down_recall = _precision_recall(pred_dir == -1, tgt_dir == -1)
    neutral_precision, neutral_recall = _precision_recall(pred_dir == 0, tgt_dir == 0)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        support=support.tolist(),
        confusion=confusion,
        direction_accuracy=direction_accuracy,
        up_precision=up_precision,
        up_recall=up_recall,
        down_precision=down_precision,
        down_recall=down_recall,
        neutral_precision=neutral_precision,
        neutral_recall=neutral_recall
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

    def __init__(
        self,
        task_type: str = "regression",
        num_classes: Optional[int] = None,
        up_classes: Tuple[int, ...] = (3, 4),
        down_classes: Tuple[int, ...] = (0, 1)
    ):
        self.predictions: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.losses: List[float] = []
        self.task_type = task_type
        self.num_classes = num_classes
        self.up_classes = up_classes
        self.down_classes = down_classes

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

        if self.task_type == "classification":
            # Accept logits/probabilities or class indices
            if predictions.ndim > 1:
                predictions = predictions.argmax(axis=1)
            self.predictions.append(predictions.astype(int).flatten())
            self.targets.append(targets.astype(int).flatten())
        else:
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

        if self.task_type == "classification":
            if self.num_classes is None:
                raise ValueError("num_classes must be provided for classification metrics.")

            class_metrics = calculate_classification_metrics(
                all_predictions,
                all_targets,
                num_classes=self.num_classes,
                up_classes=self.up_classes,
                down_classes=self.down_classes
            )

            result = {
                'accuracy': class_metrics.accuracy,
                'macro_f1': class_metrics.macro_f1,
                'direction_accuracy': class_metrics.direction_accuracy,
                'up_precision': class_metrics.up_precision,
                'up_recall': class_metrics.up_recall,
                'down_precision': class_metrics.down_precision,
                'down_recall': class_metrics.down_recall,
                'neutral_precision': class_metrics.neutral_precision,
                'neutral_recall': class_metrics.neutral_recall
            }
        else:
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
