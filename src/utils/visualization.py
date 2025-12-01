"""
Training visualization utilities.

Features:
- Training curves (loss, accuracy, learning rate)
- Prediction vs target plots
- Confusion matrices for direction
- Distribution plots
- Memory and timing charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class TrainingVisualizer:
    """
    Comprehensive training visualization.

    Creates detailed plots for monitoring training progress.
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        style: str = 'seaborn-v0_8-darkgrid'
    ):
        """
        Args:
            save_dir: Directory to save plots
            style: Matplotlib style
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Try to set style, fall back to default if not available
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                pass  # Use default style

        # Color scheme
        self.colors = {
            'train': '#2E86AB',      # Blue
            'val': '#E94F37',         # Red
            'accent': '#F39C12',      # Orange
            'positive': '#27AE60',    # Green
            'negative': '#C0392B',    # Dark Red
            'neutral': '#95A5A6'      # Gray
        }

    @staticmethod
    def _prepare_class_arrays(
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Convert predictions/targets to flat int arrays and infer num_classes."""
        preds = predictions
        tgts = targets

        if isinstance(preds, np.ndarray) and preds.ndim > 1:
            preds = preds.argmax(axis=1)

        preds = np.asarray(preds).astype(int).flatten()
        tgts = np.asarray(tgts).astype(int).flatten()

        inferred_classes = num_classes
        if inferred_classes is None:
            max_pred = preds.max() if preds.size > 0 else 0
            max_tgt = tgts.max() if tgts.size > 0 else 0
            inferred_classes = int(max(max_pred, max_tgt)) + 1

        return preds, tgts, inferred_classes

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Progress",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive training curves.

        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', etc.
            title: Plot title
            save_name: Filename to save plot

        Returns:
            matplotlib Figure
        """
        # Determine which metrics we have
        has_loss = 'train_loss' in history and 'val_loss' in history
        has_acc = 'train_acc' in history and 'val_acc' in history
        has_dir_acc = 'train_direction_acc' in history and 'val_direction_acc' in history
        has_f1 = 'train_macro_f1' in history and 'val_macro_f1' in history
        has_lr = 'learning_rate' in history
        has_grad = 'grad_norm' in history
        has_memory = 'memory_mb' in history

        # Calculate number of subplots needed
        n_plots = sum([has_loss, has_acc, has_dir_acc, has_f1, has_lr, has_grad, has_memory])
        if n_plots == 0:
            n_plots = 1

        # Create figure
        fig, axes = plt.subplots(
            (n_plots + 1) // 2, 2,
            figsize=(14, 4 * ((n_plots + 1) // 2))
        )
        if n_plots == 1:
            axes = np.array([[axes, axes]])
        axes = axes.flatten()

        plot_idx = 0
        epochs = range(1, len(history.get('train_loss', history.get('val_loss', [0]))) + 1)

        # Loss plot
        if has_loss:
            ax = axes[plot_idx]
            ax.plot(epochs, history['train_loss'], label='Train Loss',
                   color=self.colors['train'], linewidth=2)
            ax.plot(epochs, history['val_loss'], label='Val Loss',
                   color=self.colors['val'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mark best validation loss
            best_epoch = np.argmin(history['val_loss']) + 1
            best_loss = min(history['val_loss'])
            ax.axvline(x=best_epoch, color=self.colors['accent'],
                      linestyle='--', alpha=0.7, label=f'Best: {best_loss:.6f}')
            ax.scatter([best_epoch], [best_loss], color=self.colors['accent'],
                      s=100, zorder=5, marker='*')
            plot_idx += 1

        # Accuracy plot
        if has_acc:
            ax = axes[plot_idx]
            ax.plot(epochs, [a * 100 for a in history['train_acc']],
                   label='Train Acc', color=self.colors['train'], linewidth=2)
            ax.plot(epochs, [a * 100 for a in history['val_acc']],
                   label='Val Acc', color=self.colors['val'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plot_idx += 1

        # Macro F1 plot
        if has_f1:
            ax = axes[plot_idx]
            ax.plot(epochs, [f * 100 for f in history['train_macro_f1']],
                   label='Train Macro F1', color=self.colors['train'], linewidth=2)
            ax.plot(epochs, [f * 100 for f in history['val_macro_f1']],
                   label='Val Macro F1', color=self.colors['val'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Macro F1 (%)')
            ax.set_title('Macro F1 Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plot_idx += 1

        # Direction accuracy plot
        if has_dir_acc:
            ax = axes[plot_idx]
            ax.plot(epochs, [a * 100 for a in history['train_direction_acc']],
                   label='Train Dir Acc', color=self.colors['train'], linewidth=2)
            ax.plot(epochs, [a * 100 for a in history['val_direction_acc']],
                   label='Val Dir Acc', color=self.colors['val'], linewidth=2)
            ax.axhline(y=50, color=self.colors['neutral'], linestyle='--',
                      alpha=0.7, label='Random (50%)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Direction Accuracy (%)')
            ax.set_title('Direction Prediction Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            plot_idx += 1

        # Learning rate plot
        if has_lr:
            ax = axes[plot_idx]
            ax.plot(epochs, history['learning_rate'],
                   color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Gradient norm plot
        if has_grad:
            ax = axes[plot_idx]
            ax.plot(epochs, history['grad_norm'],
                   color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm (Stability Indicator)')
            ax.grid(True, alpha=0.3)
            # Add warning threshold
            ax.axhline(y=10.0, color=self.colors['negative'], linestyle='--',
                      alpha=0.7, label='Warning Threshold')
            ax.legend()
            plot_idx += 1

        # Memory usage plot
        if has_memory:
            ax = axes[plot_idx]
            ax.plot(epochs, history['memory_mb'],
                   color=self.colors['train'], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage Over Training')
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def _plot_classification_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Classification Overview",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Confusion + distribution plot for classification targets."""
        preds, tgts, num_classes = self._prepare_class_arrays(
            predictions, targets
        )

        if class_names:
            class_labels = list(class_names)
            if len(class_labels) < num_classes:
                class_labels += [f"Class {i}" for i in range(len(class_labels), num_classes)]
        else:
            class_labels = [f"Class {i}" for i in range(num_classes)]

        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for pred, tgt in zip(preds, tgts):
            if 0 <= pred < num_classes and 0 <= tgt < num_classes:
                confusion[tgt, pred] += 1

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix heatmap
        ax = axes[0]
        im = ax.imshow(confusion, cmap='Blues')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticklabels(class_labels)
        ax.set_title('Confusion Matrix')

        for i in range(num_classes):
            for j in range(num_classes):
                total = confusion.sum() if confusion.sum() > 0 else 1
                pct = confusion[i, j] / total * 100
                ax.text(j, i, f"{confusion[i, j]}\n({pct:.1f}%)",
                        ha='center', va='center', fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Distribution comparison
        ax = axes[1]
        tgt_counts = np.bincount(tgts, minlength=num_classes)
        pred_counts = np.bincount(preds, minlength=num_classes)
        total_tgt = tgt_counts.sum() if tgt_counts.sum() > 0 else 1
        total_pred = pred_counts.sum() if pred_counts.sum() > 0 else 1
        x = np.arange(num_classes)

        ax.bar(x - 0.15, tgt_counts / total_tgt * 100,
               width=0.3, color=self.colors['train'], label='True')
        ax.bar(x + 0.15, pred_counts / total_pred * 100,
               width=0.3, color=self.colors['val'], label='Pred')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_ylabel('Share (%)')
        ax.set_title('Class Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_predictions_vs_targets(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Predictions vs Targets",
        save_name: Optional[str] = None,
        sample_size: int = 1000,
        task_type: str = "regression",
        class_names: Optional[List[str]] = None
    ) -> plt.Figure:
        """
        Plot predictions against actual targets.

        Args:
            predictions: Model predictions
            targets: Actual values
            title: Plot title
            save_name: Filename to save
            sample_size: Number of points to plot
            task_type: "regression" or "classification"
            class_names: Optional list of class names for classification

        Returns:
            matplotlib Figure
        """
        if task_type == "classification":
            return self._plot_classification_comparison(
                predictions,
                targets,
                class_names=class_names,
                title=title,
                save_name=save_name
            )

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Sample if too many points
        if len(predictions) > sample_size:
            indices = np.random.choice(len(predictions), sample_size, replace=False)
            predictions = predictions[indices]
            targets = targets[indices]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Scatter plot
        ax = axes[0]
        colors = np.where((predictions > 0) == (targets > 0),
                         self.colors['positive'], self.colors['negative'])
        ax.scatter(targets, predictions, c=colors, alpha=0.5, s=20)
        ax.plot([targets.min(), targets.max()],
                [targets.min(), targets.max()],
                'k--', linewidth=2, label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Scatter: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residual plot
        ax = axes[1]
        residuals = predictions - targets
        ax.scatter(targets, residuals, alpha=0.5, s=20, color=self.colors['train'])
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Residual (Pred - Actual)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

        # Distribution comparison
        ax = axes[2]
        ax.hist(targets, bins=50, alpha=0.5, label='Actual',
               color=self.colors['train'], density=True)
        ax.hist(predictions, bins=50, alpha=0.5, label='Predicted',
               color=self.colors['val'], density=True)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_direction_confusion(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        title: str = "Direction Classification",
        save_name: Optional[str] = None,
        task_type: str = "regression",
        up_classes: Tuple[int, ...] = (3, 4),
        down_classes: Tuple[int, ...] = (0, 1)
    ) -> plt.Figure:
        """
        Plot direction confusion matrix.

        Args:
            predictions: Model predictions
            targets: Actual values
            title: Plot title
            save_name: Filename to save
            task_type: "regression" or "classification"
            up_classes: Classes mapped to "up" when task_type == "classification"
            down_classes: Classes mapped to "down" when task_type == "classification"

        Returns:
            matplotlib Figure
        """
        if task_type == "classification":
            preds, tgts, _ = self._prepare_class_arrays(predictions, targets)
            pred_dir = np.zeros_like(preds)
            pred_dir[np.isin(preds, up_classes)] = 1
            pred_dir[np.isin(preds, down_classes)] = -1

            tgt_dir = np.zeros_like(tgts)
            tgt_dir[np.isin(tgts, up_classes)] = 1
            tgt_dir[np.isin(tgts, down_classes)] = -1

            labels = [-1, 0, 1]
            label_names = ['Down', 'Neutral', 'Up']
            idx_map = {val: i for i, val in enumerate(labels)}

            confusion = np.zeros((3, 3), dtype=int)
            for p, t in zip(pred_dir, tgt_dir):
                confusion[idx_map[t], idx_map[p]] += 1

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes[0]
            im = ax.imshow(confusion, cmap='Blues')
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(label_names)
            ax.set_yticklabels(label_names)
            ax.set_title('Directional Confusion')

            for i in range(3):
                for j in range(3):
                    total = confusion.sum() if confusion.sum() > 0 else 1
                    pct = confusion[i, j] / total * 100
                    ax.text(j, i, f"{confusion[i, j]}\n({pct:.1f}%)",
                            ha='center', va='center', fontsize=10)

            plt.colorbar(im, ax=ax)

            ax = axes[1]
            total = confusion.sum() if confusion.sum() > 0 else 1
            accuracy = np.trace(confusion) / total
            recall = np.divide(
                confusion.diagonal(),
                confusion.sum(axis=1, keepdims=False) + 1e-8
            )
            metrics = ['Accuracy', 'Recall Down', 'Recall Neutral', 'Recall Up']
            values = [accuracy, *recall.tolist()]
            colors = [
                self.colors['train'],
                self.colors['negative'],
                self.colors['neutral'],
                self.colors['positive']
            ]
            bars = ax.bar(metrics, values, color=colors)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score')
            ax.set_title('Directional Metrics')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=9)

            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_name and self.save_dir:
                plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

            return fig

        predictions = predictions.flatten()
        targets = targets.flatten()

        # Classify directions
        pred_up = predictions > 0
        pred_down = predictions <= 0
        actual_up = targets > 0
        actual_down = targets <= 0

        # Confusion matrix
        tp_up = (pred_up & actual_up).sum()
        fp_up = (pred_up & actual_down).sum()
        fn_up = (pred_down & actual_up).sum()
        tn_up = (pred_down & actual_down).sum()

        confusion = np.array([
            [tp_up, fp_up],
            [fn_up, tn_up]
        ])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix heatmap
        ax = axes[0]
        im = ax.imshow(confusion, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred Up', 'Pred Down'])
        ax.set_yticklabels(['Actual Up', 'Actual Down'])
        ax.set_title('Confusion Matrix')

        # Annotate
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{confusion[i, j]}\n({confusion[i,j]/confusion.sum()*100:.1f}%)',
                              ha='center', va='center', fontsize=12)

        plt.colorbar(im, ax=ax)

        # Metrics bar chart
        ax = axes[1]
        total = confusion.sum()
        accuracy = (tp_up + tn_up) / total if total > 0 else 0
        precision = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0
        recall = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        colors = [self.colors['train'], self.colors['val'],
                 self.colors['accent'], self.colors['positive']]

        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics')
        ax.axhline(y=0.5, color=self.colors['neutral'], linestyle='--',
                  alpha=0.7, label='Random Baseline')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        ax.legend()

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_epoch_summary(
        self,
        epoch: int,
        train_predictions: np.ndarray,
        train_targets: np.ndarray,
        val_predictions: np.ndarray,
        val_targets: np.ndarray,
        metrics: Dict[str, float],
        save_name: Optional[str] = None,
        task_type: str = "regression",
        class_names: Optional[List[str]] = None,
        num_classes: Optional[int] = None
    ) -> plt.Figure:
        """
        Create comprehensive epoch summary visualization.

        Args:
            epoch: Current epoch number
            train_predictions: Training predictions
            train_targets: Training targets
            val_predictions: Validation predictions
            val_targets: Validation targets
            metrics: Dictionary of metrics
            save_name: Filename to save

        Returns:
            matplotlib Figure
        """
        if task_type == "classification":
            preds_train, tgts_train, n_classes = self._prepare_class_arrays(
                train_predictions, train_targets, num_classes
            )
            preds_val, tgts_val, n_classes = self._prepare_class_arrays(
                val_predictions, val_targets, n_classes
            )

            if class_names:
                class_labels = list(class_names)
                if len(class_labels) < n_classes:
                    class_labels += [f"Class {i}" for i in range(len(class_labels), n_classes)]
            else:
                class_labels = [f"Class {i}" for i in range(n_classes)]

            def _confusion_matrix(preds: np.ndarray, tgts: np.ndarray) -> np.ndarray:
                conf = np.zeros((n_classes, n_classes), dtype=int)
                for p, t in zip(preds, tgts):
                    if 0 <= p < n_classes and 0 <= t < n_classes:
                        conf[t, p] += 1
                return conf

            train_conf = _confusion_matrix(preds_train, tgts_train)
            val_conf = _confusion_matrix(preds_val, tgts_val)

            fig = plt.figure(figsize=(14, 10))
            gs = gridspec.GridSpec(2, 2, figure=fig)

            # Train confusion
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(train_conf, cmap='Blues')
            ax1.set_title('Train Confusion')
            ax1.set_xticks(range(n_classes))
            ax1.set_yticks(range(n_classes))
            ax1.set_xticklabels(class_labels, rotation=45, ha='right')
            ax1.set_yticklabels(class_labels)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax1.text(j, i, str(train_conf[i, j]), ha='center', va='center', fontsize=9)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Val confusion
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(val_conf, cmap='Blues')
            ax2.set_title('Val Confusion')
            ax2.set_xticks(range(n_classes))
            ax2.set_yticks(range(n_classes))
            ax2.set_xticklabels(class_labels, rotation=45, ha='right')
            ax2.set_yticklabels(class_labels)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax2.text(j, i, str(val_conf[i, j]), ha='center', va='center', fontsize=9)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Distribution comparison
            ax3 = fig.add_subplot(gs[1, 0])
            train_counts = np.bincount(tgts_train, minlength=n_classes)
            val_counts = np.bincount(tgts_val, minlength=n_classes)
            total_train = train_counts.sum() if train_counts.sum() > 0 else 1
            total_val = val_counts.sum() if val_counts.sum() > 0 else 1
            x = np.arange(n_classes)
            ax3.bar(x - 0.15, train_counts / total_train * 100,
                    width=0.3, color=self.colors['train'], label='Train')
            ax3.bar(x + 0.15, val_counts / total_val * 100,
                    width=0.3, color=self.colors['val'], label='Val')
            ax3.set_xticks(x)
            ax3.set_xticklabels(class_labels, rotation=45, ha='right')
            ax3.set_ylabel('Share (%)')
            ax3.set_title('Target Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Metrics summary
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            metrics_text = f"Epoch {epoch} Metrics\n" + "=" * 30 + "\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    metrics_text += f"{key}: {value:.4f}\n"
                else:
                    metrics_text += f"{key}: {value}\n"
            ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            plt.suptitle(f'Epoch {epoch} Summary - {timestamp}',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            if save_name and self.save_dir:
                plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

            return fig

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # Train scatter
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(train_targets.flatten(), train_predictions.flatten(),
                   alpha=0.3, s=10, color=self.colors['train'])
        ax1.plot([train_targets.min(), train_targets.max()],
                [train_targets.min(), train_targets.max()], 'k--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Train: Predictions vs Targets')

        # Val scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(val_targets.flatten(), val_predictions.flatten(),
                   alpha=0.3, s=10, color=self.colors['val'])
        ax2.plot([val_targets.min(), val_targets.max()],
                [val_targets.min(), val_targets.max()], 'k--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Val: Predictions vs Targets')

        # Prediction distributions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(train_predictions.flatten(), bins=50, alpha=0.5,
                label='Train', color=self.colors['train'], density=True)
        ax3.hist(val_predictions.flatten(), bins=50, alpha=0.5,
                label='Val', color=self.colors['val'], density=True)
        ax3.set_xlabel('Prediction Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Prediction Distributions')
        ax3.legend()

        # Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        train_residuals = train_predictions.flatten() - train_targets.flatten()
        ax4.hist(train_residuals, bins=50, color=self.colors['train'], alpha=0.7)
        ax4.axvline(x=0, color='k', linestyle='--')
        ax4.set_xlabel('Residual')
        ax4.set_ylabel('Count')
        ax4.set_title(f'Train Residuals (mean={train_residuals.mean():.6f})')

        ax5 = fig.add_subplot(gs[1, 1])
        val_residuals = val_predictions.flatten() - val_targets.flatten()
        ax5.hist(val_residuals, bins=50, color=self.colors['val'], alpha=0.7)
        ax5.axvline(x=0, color='k', linestyle='--')
        ax5.set_xlabel('Residual')
        ax5.set_ylabel('Count')
        ax5.set_title(f'Val Residuals (mean={val_residuals.mean():.6f})')

        # Metrics summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        metrics_text = f"Epoch {epoch} Metrics\n" + "=" * 30 + "\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_text += f"{key}: {value:.6f}\n"
            else:
                metrics_text += f"{key}: {value}\n"
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Direction accuracy over time (sample)
        ax7 = fig.add_subplot(gs[2, :])
        sample_size = min(200, len(val_targets))
        indices = np.linspace(0, len(val_targets)-1, sample_size, dtype=int)

        correct = ((val_predictions[indices] > 0) == (val_targets[indices] > 0)).flatten()
        colors = np.where(correct, self.colors['positive'], self.colors['negative'])

        ax7.bar(range(len(indices)), val_predictions[indices].flatten(),
               color=colors, alpha=0.7, width=1.0)
        ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax7.set_xlabel('Sample Index')
        ax7.set_ylabel('Prediction')
        ax7.set_title('Sample Predictions (Green=Correct Direction, Red=Wrong)')

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        plt.suptitle(f'Epoch {epoch} Summary - {timestamp}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_learning_dynamics(
        self,
        batch_losses: List[float],
        batch_grad_norms: List[float],
        title: str = "Learning Dynamics",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot batch-level learning dynamics.

        Args:
            batch_losses: List of batch losses
            batch_grad_norms: List of gradient norms
            title: Plot title
            save_name: Filename to save

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        batches = range(1, len(batch_losses) + 1)

        # Raw loss
        ax = axes[0, 0]
        ax.plot(batches, batch_losses, alpha=0.5, color=self.colors['train'])
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title('Batch Loss (Raw)')
        ax.grid(True, alpha=0.3)

        # Smoothed loss
        ax = axes[0, 1]
        window = min(50, len(batch_losses) // 10 + 1)
        smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(batch_losses)+1), smoothed,
               color=self.colors['train'], linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Batch Loss (Smoothed, window={window})')
        ax.grid(True, alpha=0.3)

        # Gradient norms
        ax = axes[1, 0]
        ax.plot(batches, batch_grad_norms, alpha=0.5, color=self.colors['accent'])
        ax.axhline(y=1.0, color=self.colors['positive'], linestyle='--',
                  alpha=0.7, label='Ideal')
        ax.axhline(y=10.0, color=self.colors['negative'], linestyle='--',
                  alpha=0.7, label='Warning')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Batch Gradient Norms')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss vs gradient norm scatter
        ax = axes[1, 1]
        ax.scatter(batch_grad_norms, batch_losses, alpha=0.3,
                  s=10, color=self.colors['train'])
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Gradient Norm')
        ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def close_all(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Convenience function to plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure
    """
    visualizer = TrainingVisualizer()
    fig = visualizer.plot_training_curves(
        history,
        title="Training History",
        save_name=None
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig
