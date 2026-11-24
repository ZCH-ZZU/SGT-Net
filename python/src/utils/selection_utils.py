"""Source-subject selection strategies for Stage II."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data_loader import HybridBCIDataset
from ..models.models import get_model
from .losses import mmd_loss
from .paths import resolve_path
from .physio_utils import calculate_heom_distance, load_physio_data


def _feature_mean(model, dataset, device, max_samples=50):
    indices = list(range(min(len(dataset), max_samples)))
    loader = DataLoader(dataset, batch_size=16, sampler=indices)
    feats = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            data = {k: v.to(device) for k, v in batch['data'].items()}
            outs = []
            for mod in model.modalities:
                out = model.trunks[mod](data[mod])
                if out.dim() == 3:
                    out = out.mean(dim=1)
                outs.append(out)
            feats.append(torch.cat(outs, dim=1))
    return torch.cat(feats, dim=0)


def _performance_path(sel_cfg, task):
    value = sel_cfg['performance_csv']
    if isinstance(value, dict):
        if task not in value:
            raise KeyError(f'No performance_csv entry for task {task}.')
        value = value[task]
    return resolve_path(value)


def get_selection_plan(config, target_id, all_sources, method, mode, k, device='cpu'):
    if method == 'baseline':
        return list(all_sources), None, {'all_scores': {str(s): 0.0 for s in all_sources}}

    sel_cfg = config['experiment']['source_selection']
    task = config['experiment']['task']
    k_val = len(all_sources) if str(k).lower() == 'all' else int(k)
    scores = []

    if method == 'physio':
        df = load_physio_data(resolve_path(sel_cfg['physio_path']))
        dists = calculate_heom_distance(df, target_id, all_sources)
        scores = list(zip(all_sources, dists))

    elif method == 'performance':
        path = _performance_path(sel_cfg, task)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Stage I summary not found: {path}')
        df = pd.read_csv(path)
        acc_col = 'accuracy_mean' if 'accuracy_mean' in df.columns else 'accuracy'
        for sid in all_sources:
            row = df[df['subject_id'] == sid]
            acc = float(row[acc_col].values[0]) if not row.empty else 0.5
            scores.append((sid, 1.0 - acc))

    elif method in ['data', 'hybrid']:
        exp = config['experiment']
        dummy = HybridBCIDataset(config['data']['h5_path'], target_id, task, exp['modalities'])
        channels = {k: {'channels': v.shape[0]} for k, v in dummy[0]['data'].items()}
        trunk_params = exp.get('trunk_params', {}).get('params', exp.get('trunk_params', {}))
        model = get_model({
            'channels_dict': channels,
            'num_classes': config['num_classes'],
            'fusion_type': 'concat',
            'fusion_params': {},
            'trunk_params': trunk_params,
            'use_dts': exp['dts_type'] == 'dts',
        }).to(device)

        target_feat = _feature_mean(model, dummy, device)
        perf = None
        if method == 'hybrid':
            path = _performance_path(sel_cfg, task)
            perf = pd.read_csv(path) if os.path.exists(path) else None

        for sid in all_sources:
            ds = HybridBCIDataset(config['data']['h5_path'], sid, task, exp['modalities'])
            src_feat = _feature_mean(model, ds, device)
            mmd = float(mmd_loss(src_feat, target_feat).item())
            if method == 'data':
                scores.append((sid, mmd))
            else:
                acc = 0.5
                if perf is not None:
                    row = perf[perf['subject_id'] == sid]
                    col = 'accuracy_mean' if 'accuracy_mean' in perf.columns else 'accuracy'
                    if not row.empty:
                        acc = float(row[col].values[0])
                scores.append((sid, 1.0 - acc / (1.0 + mmd)))
    else:
        raise ValueError(f'Unknown selection method: {method}')

    scores.sort(key=lambda x: x[1])
    selected = [s[0] for s in scores[:k_val]]
    vals = [s[1] for s in scores[:k_val]]
    debug = {'all_scores': {str(s): float(v) for s, v in scores}}

    if mode == 'hard':
        return selected, None, debug

    if mode == 'soft':
        vals_t = torch.tensor(vals, dtype=torch.float32)
        tau = float(vals_t.std()) if len(vals) > 1 else 1.0
        tau = tau if tau > 0 else 1.0
        weights = F.softmax(-vals_t / tau, dim=0).numpy() * len(vals)
        return selected, weights, debug

    return list(all_sources), None, debug
