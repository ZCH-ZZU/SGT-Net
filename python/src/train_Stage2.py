"""Stage II: LOSO cross-subject adaptation with source-only validation."""

import copy
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data_loader import CrossSubjectDataset, HybridBCIDataset
from .models.models import get_model
from .utils.callbacks import EarlyStopping
from .utils.losses import coral_loss, dcca_loss, mmd_loss
from .utils.lr_scheduler import get_lr_scheduler
from .utils.metrics import calculate_all_metrics
from .utils.normalization import StandardizedSubset, fit_channel_stats
from .utils.plotting import plot_and_save_confusion_matrix, plot_and_save_history
from .utils.training_utils import evaluate_model


def _collate(batch):
    keys = batch[0]['data'].keys()
    return {
        'data': {k: torch.stack([s['data'][k] for s in batch]) for k in keys},
        'label': torch.tensor([s['label'] for s in batch]),
    }


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_loso(config, target_id, source_ids, source_weights=None, output_dir=None, seed=42):
    _set_seed(seed)
    exp_cfg = config['experiment']
    train_cfg = config['training']
    task = exp_cfg['task']
    strategy = exp_cfg.get('strategy_current', 'source_only')
    use_dts = exp_cfg['dts_type'] == 'dts'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Subject-wise source validation split; target labels are never used for
    # early stopping or scheduling.
    val_ratio = train_cfg.get('validation_split', {}).get('valid_ratio', 0.15)
    if len(source_ids) >= 2 and val_ratio > 0:
        train_src, val_src = train_test_split(
            list(source_ids), test_size=val_ratio, random_state=seed)
    else:
        train_src, val_src = list(source_ids), list(source_ids)

    weight_map = {}
    if source_weights is not None:
        weight_map = {sid: float(w) for sid, w in zip(source_ids, source_weights)}

    train_weights = [weight_map.get(sid, 1.0) for sid in train_src]
    train_dataset = CrossSubjectDataset(
        config['data']['h5_path'], train_src, task, exp_cfg['modalities'],
        subject_weights=train_weights)
    val_dataset = CrossSubjectDataset(
        config['data']['h5_path'], val_src, task, exp_cfg['modalities'])
    target_dataset = HybridBCIDataset(
        config['data']['h5_path'], target_id, task, exp_cfg['modalities'])

    if len(train_dataset) == 0 or len(target_dataset) == 0:
        return None, None

    sampler = None
    shuffle = True
    if source_weights is not None:
        sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)
        shuffle = False

    if config.get('data', {}).get('normalization', 'zscore') == 'zscore':
        modalities = train_dataset.sub_datasets[0].base_modalities_needed
        stats = fit_channel_stats(train_dataset, modalities)
        train_dataset = StandardizedSubset(train_dataset, stats)
        val_dataset = StandardizedSubset(val_dataset, stats)
        target_dataset = StandardizedSubset(target_dataset, stats)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'],
                              shuffle=shuffle, sampler=sampler,
                              collate_fn=_collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'],
                            shuffle=False, collate_fn=_collate)
    target_loader = DataLoader(target_dataset, batch_size=train_cfg['batch_size'],
                               shuffle=True, collate_fn=_collate, drop_last=True)
    test_loader = DataLoader(target_dataset, batch_size=train_cfg['batch_size'],
                             shuffle=False, collate_fn=_collate)

    sample = train_dataset[0]['data']
    channels = {k: {'channels': v.shape[0]} for k, v in sample.items()}
    trunk_params = copy.deepcopy(exp_cfg.get('trunk_params', {}).get('params',
                                                                    exp_cfg.get('trunk_params', {})))
    fusion_params = copy.deepcopy(exp_cfg.get('fusion_params', {}).get('params',
                                                                      exp_cfg.get('fusion_params', {})))
    model = get_model({
        'channels_dict': channels,
        'num_classes': config['num_classes'],
        'fusion_type': exp_cfg['fusion_type'],
        'fusion_params': fusion_params,
        'trunk_params': trunk_params,
        'use_dts': use_dts,
    }).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'],
                                 weight_decay=train_cfg.get('l2_weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()
    scheduler = get_lr_scheduler(optimizer, train_cfg.get('lr_scheduler'),
                                 total_epochs=train_cfg['epochs'])

    result_dir = os.path.join(output_dir or 'results/Stage2', f'target_{target_id}')
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(result_dir, 'best_model.pt')
    es_cfg = train_cfg.get('early_stopping', {}).copy()
    es_enabled = es_cfg.pop('enabled', False)
    early_stopping = EarlyStopping(path=checkpoint_path, **es_cfg) if es_enabled else None

    lambda_da = exp_cfg.get('lambda_da_settings', {}).get(task, 0.1)
    lambda_dcca = exp_cfg.get('lambda_dcca', 0.1)
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start = time.time()
    target_iter = iter(target_loader)

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        correct_s, total_s = 0, 0
        for batch_s in train_loader:
            data_s = {k: v.to(device) for k, v in batch_s['data'].items()}
            label_s = batch_s['label'].to(device)
            optimizer.zero_grad()

            need_deep = strategy != 'source_only'
            need_mid = 'hierarchical' in strategy
            need_split = 'dcca' in strategy
            out_s = model(data_s, return_features=need_deep, return_mid_feats=need_mid,
                          return_split_feats=need_split)
            if isinstance(out_s, tuple):
                logits_s = out_s[0]
                idx = 1
                feat_s = out_s[idx] if need_deep else None
                if need_deep:
                    idx += 1
                mid_s = out_s[idx] if need_mid else None
                if need_mid:
                    idx += 1
                eeg_s, nirs_s = (out_s[idx], out_s[idx + 1]) if need_split else (None, None)
            else:
                logits_s = out_s
                feat_s = mid_s = eeg_s = nirs_s = None

            loss_cls = criterion(logits_s, label_s)
            loss_da = torch.tensor(0.0, device=device)
            loss_dcca = torch.tensor(0.0, device=device)

            if strategy != 'source_only':
                try:
                    batch_t = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    batch_t = next(target_iter)
                data_t = {k: v.to(device) for k, v in batch_t['data'].items()}
                out_t = model(data_t, return_features=need_deep, return_mid_feats=need_mid,
                              return_split_feats=need_split)
                if isinstance(out_t, tuple):
                    idx = 1
                    feat_t = out_t[idx] if need_deep else None
                    if need_deep:
                        idx += 1
                    mid_t = out_t[idx] if need_mid else None
                    if need_mid:
                        idx += 1
                    eeg_t, nirs_t = (out_t[idx], out_t[idx + 1]) if need_split else (None, None)
                else:
                    feat_t = mid_t = eeg_t = nirs_t = None

                if 'hierarchical' in strategy:
                    loss_da = mmd_loss(feat_s, feat_t)
                    if mid_s and mid_t:
                        shallow = []
                        for mod in mid_s:
                            if mid_s[mod] is not None and mid_t[mod] is not None:
                                shallow.append(coral_loss(mid_s[mod].mean(dim=-1),
                                                          mid_t[mod].mean(dim=-1)))
                        if shallow:
                            loss_da = loss_da + sum(shallow) / len(shallow)
                elif 'mmd' in strategy:
                    loss_da = mmd_loss(feat_s, feat_t)
                elif 'coral' in strategy:
                    loss_da = coral_loss(feat_s, feat_t)

                if need_split and eeg_s is not None and nirs_s is not None:
                    loss_dcca = dcca_loss(eeg_s, nirs_s)
                    if eeg_t is not None and nirs_t is not None:
                        loss_dcca = loss_dcca + dcca_loss(eeg_t, nirs_t)

            loss = loss_cls + lambda_da * loss_da + lambda_dcca * loss_dcca
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            _, pred_s = torch.max(logits_s, 1)
            correct_s += (pred_s == label_s).sum().item()
            total_s += label_s.size(0)

        val_labels, val_preds, val_loss = evaluate_model(model, val_loader, criterion, device)
        val_acc = float(np.mean(val_labels == val_preds))
        train_acc = correct_s / total_s if total_s else 0.0
        if not np.isfinite(val_loss):
            print(f'  Ep {epoch}: non-finite validation loss; skipping scheduler/early stop.')
        lr = optimizer.param_groups[0]['lr']
        for k, v in [('epoch', epoch), ('train_loss', total_loss / len(train_loader)),
                     ('train_acc', train_acc), ('val_loss', val_loss),
                     ('val_acc', val_acc), ('lr', lr)]:
            history[k].append(v)
        print(f'  Ep {epoch}: loss={total_loss / len(train_loader):.3f} '
              f'val_loss={val_loss:.3f} val_acc={val_acc:.3f}')
        if scheduler and np.isfinite(val_loss):
            scheduler.step(metrics=val_loss)
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break

    history_df = pd.DataFrame(history)
    hist_cfg = config.get('plotting', {}).get('history', {})
    if hist_cfg.get('save_data', False):
        history_df.to_csv(os.path.join(result_dir, 'history.csv'), index=False)
    if hist_cfg.get('save_plot', False):
        plot_and_save_history(history_df, os.path.join(result_dir, 'history.jpg'))

    if early_stopping and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    labels, preds, _ = evaluate_model(model, test_loader, criterion, device)
    metrics = calculate_all_metrics(labels, preds)
    n_steps = target_dataset.n_steps
    time_course = None
    if n_steps > 1 and len(labels) % n_steps == 0:
        n_trials = len(labels) // n_steps
        time_course = (labels.reshape(n_trials, n_steps) ==
                       preds.reshape(n_trials, n_steps)).astype(int)
        metrics['peak_accuracy'] = float(np.max(np.mean(time_course, axis=0)))
    else:
        metrics['peak_accuracy'] = metrics['accuracy']
    metrics['duration_seconds'] = time.time() - start

    cm_cfg = config.get('plotting', {}).get('confusion_matrix', {})
    if cm_cfg.get('enabled', False):
        cm_labels = config.get('plotting', {}).get('cm_labels', {}).get(task, ['0', '1'])
        plot_and_save_confusion_matrix(
            [labels], [preds], cm_labels, result_dir,
            save_data=cm_cfg.get('save_data', True),
            save_plot=cm_cfg.get('save_plot', True))

    return metrics, time_course
