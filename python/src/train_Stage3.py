"""Stage III: few-shot personalization with LoRA/DoRA/full fine-tuning."""

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data_loader import HybridBCIDataset
from .models.models import get_model
from .models.peft import inject_peft
from .utils.metrics import calculate_all_metrics
from .utils.normalization import StandardizedSubset, fit_channel_stats
from .utils.plotting import plot_and_save_confusion_matrix


def _collate(batch):
    keys = batch[0]['data'].keys()
    return {
        'data': {k: torch.stack([s['data'][k] for s in batch]) for k in keys},
        'label': torch.tensor([s['label'] for s in batch]),
    }


def _evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            data = {k: v.to(device) for k, v in batch['data'].items()}
            out = model(data)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(batch['label'].numpy())
    return np.array(labels), np.array(preds)


def _trial_split(dataset, shots, num_classes, seed):
    labels = dataset.labels.numpy()
    trial_ids = dataset.groups
    unique_trials = np.unique(trial_ids)
    trial_label = {t: int(labels[trial_ids == t][0]) for t in unique_trials}

    rng = np.random.RandomState(seed)
    selected = []
    if shots > 0:
        n_per_class = max(1, shots)
        min_class_trials = min(
            len([t for t in unique_trials if trial_label[t] == cls])
            for cls in range(num_classes))
        n_per_class = min(n_per_class, max(1, min_class_trials - 1))
        for cls in range(num_classes):
            cls_trials = [t for t in unique_trials if trial_label[t] == cls]
            rng.shuffle(cls_trials)
            selected.extend(cls_trials[:n_per_class])

    train_idx = np.where(np.isin(trial_ids, selected))[0]
    test_idx = np.where(~np.isin(trial_ids, selected))[0]
    if len(test_idx) == 0:
        test_idx = np.arange(len(dataset))
    return train_idx, test_idx


def run_finetune(config, target_id, pretrained_path, method, shots, seed,
                 target_modules=None, output_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_cfg = config['experiment']
    train_cfg = config['training']
    task = exp_cfg['task']
    dataset = HybridBCIDataset(config['data']['h5_path'], target_id, task,
                               exp_cfg['modalities'])
    if len(dataset) == 0:
        return None

    train_idx, test_idx = _trial_split(dataset, shots, config['num_classes'], seed)
    if method == 'none':
        train_idx = np.array([], dtype=int)
        test_idx = np.arange(len(dataset))

    train_loader = None
    train_subset = Subset(dataset, train_idx) if len(train_idx) > 0 else None
    test_subset = Subset(dataset, test_idx)
    if (config.get('data', {}).get('normalization', 'zscore') == 'zscore'
            and train_subset is not None):
        stats = fit_channel_stats(train_subset, dataset.base_modalities_needed)
        train_subset = StandardizedSubset(train_subset, stats)
        test_subset = StandardizedSubset(test_subset, stats)
    if train_subset is not None:
        train_loader = DataLoader(
            train_subset, batch_size=train_cfg['batch_size'],
            shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_subset,
                             batch_size=train_cfg['batch_size'],
                             shuffle=False, collate_fn=_collate)

    sample = dataset[0]['data']
    channels = {k: {'channels': v.shape[0]} for k, v in sample.items()}
    timepoints = {k: v.shape[1] for k, v in sample.items()}
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
        'use_dts': exp_cfg['dts_type'] == 'dts',
    })

    # Initialise lazy layers before loading weights.
    dummy = {k: torch.randn(2, channels[k]['channels'], timepoints[k]) for k in channels}
    with torch.no_grad():
        model(dummy)

    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state, strict=False)

    if method in ('lora', 'dora', 'linear_probing'):
        modules = [m for m in (target_modules or ['linear1', 'linear2'])
                   if m != 'classifier_head']
        model = inject_peft(model, method=method,
                            rank=exp_cfg.get('lora_rank', 2),
                            alpha=exp_cfg.get('lora_alpha', 8),
                            dropout=exp_cfg.get('lora_dropout', 0.1),
                            target_modules=modules)
        if 'classifier_head' in (target_modules or []):
            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
    elif method == 'full':
        for param in model.parameters():
            param.requires_grad = True
    elif method == 'none':
        for param in model.parameters():
            param.requires_grad = False

    # Keep the DTS offset trainable in all adaptation modes.
    for name, param in model.named_parameters():
        if 'dts_module.shift' in name:
            param.requires_grad = True

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
    start = time.time()

    if train_loader is not None and method != 'none':
        lr = train_cfg['learning_rate']
        if method == 'full':
            lr *= 0.1
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=train_cfg.get('l2_weight_decay', 0.0))
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(train_cfg['epochs']):
            for batch in train_loader:
                data = {k: v.to(device) for k, v in batch['data'].items()}
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                loss = criterion(model(data), labels)
                loss.backward()
                optimizer.step()

    labels, preds = _evaluate(model, test_loader, device)
    metrics = calculate_all_metrics(labels, preds)
    metrics.update({
        'train_time': time.time() - start,
        'trainable_params': trainable,
        'param_ratio': trainable / total * 100 if total else 0.0,
        'storage_mb': trainable * 4 / (1024 * 1024),
    })

    cm_cfg = config.get('plotting', {}).get('confusion_matrix', {})
    if output_dir and cm_cfg.get('enabled', False):
        cm_labels = config.get('plotting', {}).get('cm_labels', {}).get(
            task, [str(i) for i in range(config['num_classes'])])
        plot_and_save_confusion_matrix(
            [labels], [preds], cm_labels,
            os.path.join(output_dir, f'target_{target_id}'),
            save_data=cm_cfg.get('save_data', True),
            save_plot=cm_cfg.get('save_plot', True))
    return metrics
