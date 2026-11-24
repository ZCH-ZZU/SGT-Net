"""Training-curve and confusion-matrix plotting."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_and_save_history(history_df, save_path):
    if history_df.empty:
        print('    - Warning: history DataFrame is empty, skipping plot.')
        return

    if 'epoch' not in history_df.columns:
        history_df = history_df.reset_index().rename(columns={'index': 'epoch'})
        history_df['epoch'] += 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Training History', fontsize=16)

    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='royalblue')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', color='darkorange')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.5)

    axes[1].plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy', color='royalblue')
    axes[1].plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy', color='darkorange')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.5)

    axes[2].plot(history_df['epoch'], history_df['lr'], label='Learning Rate', color='green', linestyle='--')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].legend()
    axes[2].grid(alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def plot_and_save_confusion_matrix(y_trues, y_preds, labels, save_dir,
                                   save_data=True, save_plot=True):
    """Save raw/normalized confusion matrices and/or a per-fold figure."""
    if not (save_data or save_plot):
        return
    n_splits = len(y_trues)
    if n_splits == 0:
        return

    all_cms, all_normalized = [], []
    for i in range(n_splits):
        cm = confusion_matrix(y_trues[i], y_preds[i])
        all_cms.append(cm)
        row_sums = cm.sum(axis=1, keepdims=True)
        all_normalized.append(np.divide(
            cm.astype(float), row_sums,
            out=np.zeros_like(cm, dtype=float), where=row_sums != 0))

    os.makedirs(save_dir, exist_ok=True)

    if save_plot:
        ncols = min(n_splits, 5)
        nrows = int(np.ceil(n_splits / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()
        for i in range(n_splits):
            cm, normalized = all_cms[i], all_normalized[i]
            annot = np.array([
                f'{count}\n({pct:.1%})'
                for count, pct in zip(cm.flatten(), normalized.flatten())
            ]).reshape(cm.shape)
            sns.heatmap(normalized, annot=annot, fmt='', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, cbar=False, ax=axes[i])
            axes[i].set_title(f'Fold {i + 1}')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        for i in range(n_splits, len(axes)):
            axes[i].axis('off')
        fig.suptitle('Confusion Matrices per Fold', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(save_dir, 'confusion_matrix.jpg')
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f'      -> Confusion matrix plot saved to: {plot_path}')

    if save_data:
        excel_path = os.path.join(save_dir, 'confusion_matrix_data.xlsx')
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for i in range(n_splits):
                pd.DataFrame(all_cms[i], index=labels, columns=labels).to_excel(
                    writer, sheet_name=f'Fold_{i + 1}_Raw_Counts')
                pd.DataFrame(all_normalized[i], index=labels, columns=labels).to_excel(
                    writer, sheet_name=f'Fold_{i + 1}_Normalized')
        print(f'      -> Confusion matrix data saved to: {excel_path}')
