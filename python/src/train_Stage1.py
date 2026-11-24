"""Stage I: intra-subject training with 5-fold GroupKFold."""

import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset

from .data_loader import HybridBCIDataset
from .models.models import get_model
from .utils.callbacks import EarlyStopping
from .utils.console import clear_console
from .utils.lr_scheduler import get_lr_scheduler
from .utils.metrics import calculate_all_metrics
from .utils.normalization import (StandardizedSubset, WindowZScoreSubset,
                                  fit_channel_stats)
from .utils.plotting import plot_and_save_confusion_matrix, plot_and_save_history
from .utils.training_utils import average_scores, evaluate_model, train_one_epoch


def _collate(batch):
    keys = batch[0]['data'].keys()
    return {
        'data': {k: torch.stack([s['data'][k] for s in batch]) for k in keys},
        'label': torch.tensor([s['label'] for s in batch]),
    }


def run_single_subject(config, subject_id, channels_dict, h5_path, output_dir):
    exp_cfg = config['experiment']
    train_cfg = config['training']
    task = exp_cfg['task']
    modality = exp_cfg['modalities']
    fusion_type = exp_cfg['fusion_type']
    dts_type = exp_cfg['dts_type']
    use_dts = dts_type == 'dts'

    trunk_params = copy.deepcopy(exp_cfg.get('trunk_params', {}).get('params',
                                                                    exp_cfg.get('trunk_params', {})))
    fusion_params = copy.deepcopy(exp_cfg.get('fusion_params', {}).get('params',
                                                                      exp_cfg.get('fusion_params', {})))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = HybridBCIDataset(h5_path, subject_id, task, modality)
    if len(dataset) == 0:
        print(f'  No data for subject {subject_id}, task {task}.')
        return None, 0.0, None

    mod_str = '+'.join(modality) if isinstance(modality, list) else modality
    exp_name = f'{task}_{dts_type}_{mod_str}_{fusion_type}'
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints', f'sub_{subject_id}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f'  Subject {subject_id} | {exp_name} | windows/trial={dataset.n_steps}')
    start_time = time.time()
    fold_scores, fold_trues, fold_preds, time_courses = [], [], [], []

    n_repeats = train_cfg.get('n_repeats', 1)
    use_group_kfold = exp_cfg.get('use_group_kfold', True)
    for repeat in range(n_repeats):
        groups = dataset.groups
        if use_group_kfold:
            kf = GroupKFold(n_splits=train_cfg['n_splits'])
            splits = kf.split(np.zeros(len(dataset)), dataset.labels.numpy(), groups)
        else:
            kf = StratifiedKFold(n_splits=train_cfg['n_splits'], shuffle=True,
                                 random_state=42 + repeat)
            splits = kf.split(np.zeros(len(dataset)), dataset.labels.numpy())

        for fold, (train_idx, test_idx) in enumerate(splits):
            val_cfg = train_cfg.get('validation_split', {})
            if val_cfg.get('method', 'default') == 'stratified':
                train_abs, val_abs = train_test_split(
                    train_idx, test_size=val_cfg.get('valid_ratio', 0.15),
                    stratify=dataset.labels.numpy()[train_idx],
                    random_state=42 + repeat)
                train_subset = Subset(dataset, train_abs)
                val_subset = Subset(dataset, val_abs)
                test_subset = Subset(dataset, test_idx)
            else:
                train_subset = Subset(dataset, train_idx)
                val_subset = Subset(dataset, test_idx)
                test_subset = Subset(dataset, test_idx)

            norm_mode = config.get('data', {}).get('normalization', 'zscore')
            if norm_mode == 'zscore':
                stats = fit_channel_stats(train_subset, dataset.base_modalities_needed)
                train_subset = StandardizedSubset(train_subset, stats)
                val_subset = StandardizedSubset(val_subset, stats)
                test_subset = StandardizedSubset(test_subset, stats)
            elif norm_mode == 'window_zscore':
                train_subset = WindowZScoreSubset(train_subset)
                val_subset = WindowZScoreSubset(val_subset)
                test_subset = WindowZScoreSubset(test_subset)

            train_loader = DataLoader(train_subset, batch_size=train_cfg['batch_size'],
                                      shuffle=True, collate_fn=_collate)
            val_loader = DataLoader(val_subset, batch_size=train_cfg['batch_size'],
                                    shuffle=False, collate_fn=_collate)
            test_loader = DataLoader(test_subset, batch_size=train_cfg['batch_size'],
                                     shuffle=False, collate_fn=_collate)

            model_config = {
                'channels_dict': channels_dict,
                'num_classes': config['num_classes'],
                'fusion_type': fusion_type,
                'fusion_params': fusion_params,
                'trunk_params': trunk_params,
                'use_dts': use_dts,
            }
            model = get_model(model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                         weight_decay=train_cfg.get('l2_weight_decay', 0.0))
            criterion = torch.nn.CrossEntropyLoss()
            scheduler = get_lr_scheduler(optimizer, train_cfg.get('lr_scheduler'),
                                         total_epochs=train_cfg['epochs'])

            es_cfg = train_cfg.get('early_stopping', {}).copy()
            es_enabled = es_cfg.pop('enabled', False)
            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold + 1}_best.pt')
            early_stopping = EarlyStopping(path=checkpoint_path, **es_cfg) if es_enabled else None

            history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
            for epoch in range(1, train_cfg['epochs'] + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch,
                    train_cfg['epochs'], train_cfg.get('l1_lambda', 0.0))
                val_labels, val_preds, val_loss = evaluate_model(model, val_loader, criterion, device)
                val_acc = float(np.mean(val_labels == val_preds))
                lr = optimizer.param_groups[0]['lr']
                for k, v in [('epoch', epoch), ('train_loss', train_loss), ('val_loss', val_loss),
                             ('train_acc', train_acc), ('val_acc', val_acc), ('lr', lr)]:
                    history[k].append(v)
                if scheduler and np.isfinite(val_loss):
                    scheduler.step(metrics=val_loss)
                if early_stopping:
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        break

            history_df = pd.DataFrame(history)
            plot_cfg = config.get('plotting', {})
            hist_cfg = plot_cfg.get('history', {})
            if hist_cfg.get('save_data', False):
                history_df.to_csv(os.path.join(checkpoint_dir, f'fold_{fold + 1}_history.csv'),
                                  index=False)
            if hist_cfg.get('save_plot', False):
                plot_and_save_history(history_df, os.path.join(checkpoint_dir,
                                                               f'fold_{fold + 1}_history.jpg'))

            if early_stopping and os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))

            labels, preds, _ = evaluate_model(model, test_loader, criterion, device)
            metrics = calculate_all_metrics(labels, preds)

            if dataset.n_steps > 1 and len(labels) % dataset.n_steps == 0:
                n_trials = len(labels) // dataset.n_steps
                correct = (labels.reshape(n_trials, dataset.n_steps) ==
                           preds.reshape(n_trials, dataset.n_steps)).astype(int)
                time_courses.append(correct)
                metrics['peak_accuracy'] = float(np.max(np.mean(correct, axis=0)))
            else:
                metrics['peak_accuracy'] = metrics['accuracy']

            fold_scores.append(metrics)
            fold_trues.append(labels)
            fold_preds.append(preds)
            print(f'    Fold {fold + 1}: acc={metrics["accuracy"]:.4f}, '
                  f'peak={metrics["peak_accuracy"]:.4f}')
            clear_console(config.get('logging', {}).get('clear_fold', False))

    if not fold_scores:
        return None, 0.0, None

    cm_cfg = config.get('plotting', {}).get('confusion_matrix', {})
    if cm_cfg.get('enabled', False):
        labels = config.get('plotting', {}).get('cm_labels', {}).get(
            task, [str(i) for i in range(config['num_classes'])])
        plot_and_save_confusion_matrix(
            fold_trues, fold_preds, labels,
            os.path.join(exp_dir, 'confusion_matrix', f'sub_{subject_id}'),
            save_data=cm_cfg.get('save_data', True),
            save_plot=cm_cfg.get('save_plot', True))

    avg = average_scores(fold_scores)
    avg['subject_id'] = subject_id
    avg['duration_seconds'] = time.time() - start_time
    tc = np.concatenate(time_courses, axis=0) if time_courses else None
    return avg, avg['duration_seconds'], tc
