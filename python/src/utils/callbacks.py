"""Early stopping with best-checkpoint saving."""

import os

import numpy as np
import torch


class EarlyStopping:
    """Stop training when the validation loss stops improving.

    The best model is saved to ``path``. Non-finite validation losses are
    ignored so that a diverged epoch cannot overwrite a valid checkpoint.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt',
                 min_epochs=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.min_epochs = min_epochs
        self.epoch = 0
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        if not np.isfinite(val_loss):
            return

        self.epoch += 1
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            if self.epoch > self.min_epochs:
                self.counter += 1
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            self.trace_func('Non-finite weights detected; checkpoint not saved.')
            return
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                'Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
