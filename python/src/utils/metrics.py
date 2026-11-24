"""Classification metrics used in all stages."""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, recall_score)


def calculate_all_metrics(y_true, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
    }

    specificities = []
    for label in np.unique(y_true):
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        fp = np.sum((y_true != label) & (y_pred == label))
        tn = np.sum((y_true != label) & (y_pred != label))
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    metrics['specificity'] = float(np.mean(specificities))
    return metrics
