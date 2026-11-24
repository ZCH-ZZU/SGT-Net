"""Physiological metadata utilities for source-subject selection."""

import os

import numpy as np
import pandas as pd


def load_physio_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Physio metadata not found at {csv_path}')

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = [c.strip().capitalize() for c in df.columns]

    if 'Subject' not in df.columns:
        candidates = [c for c in df.columns if 'sub' in c.lower()]
        if not candidates:
            raise KeyError(f"Column 'Subject' not found. Available: {df.columns.tolist()}")
        df.rename(columns={candidates[0]: 'Subject'}, inplace=True)

    df['Subject'] = df['Subject'].astype(int)
    return df


def calculate_heom_distance(df, target_id, source_ids):
    """Heterogeneous Euclidean-overlap metric between one target and sources."""
    target = df[df['Subject'] == target_id].iloc[0]
    sources = df[df['Subject'].isin(source_ids)]

    num_cols = ['Age']
    cat_cols = ['Gender', 'Handedness', 'Hair color', 'Hair length', 'Hair density']

    ranges = {}
    for col in num_cols:
        ranges[col] = df[col].max() - df[col].min()
        if ranges[col] == 0:
            ranges[col] = 1e-6

    distances = []
    for _, row in sources.iterrows():
        dist = 0.0
        for col in cat_cols:
            if row[col] != target[col]:
                dist += 1.0
        for col in num_cols:
            diff = abs(row[col] - target[col])
            dist += (diff / ranges[col]) ** 2
        distances.append(dist)
    return np.array(distances)


def get_source_weights(csv_path, target_id, source_ids, strategy='none', k=None,
                       temperature=1.0):
    debug = {'distances': [], 'all_source_ids': source_ids}
    if strategy == 'none':
        return source_ids, np.ones(len(source_ids)), debug

    df = load_physio_data(csv_path)
    distances = calculate_heom_distance(df, target_id, source_ids)
    debug['distances'] = distances

    if strategy == 'hard_selection':
        top = np.argsort(distances)[:k]
        selected = [source_ids[i] for i in top]
        debug['selected_distances'] = [distances[i] for i in top]
        return selected, np.ones(len(selected)), debug

    if strategy == 'soft_weighting':
        import torch
        import torch.nn.functional as F
        dist = torch.tensor(distances, dtype=torch.float32)
        weights = F.softmax(-dist / temperature, dim=0).numpy() * len(source_ids)
        return source_ids, weights, debug

    raise ValueError(f'Unknown strategy: {strategy}')
