"""HDF5 dataset loaders for SGT-Net.

The HDF5 files produced by ``matlab/run_preprocessing.m`` contain
``trial_ids`` for every sliding window, so trial-level grouping is
self-contained. All windows of a trial share the same group id.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def determine_base_modalities(strategy):
    """Map a modality strategy (e.g. ``meta4``) to base modalities."""
    needed = set()

    def parse(item):
        item = str(item)
        if item.startswith('meta'):
            if '1' in item:
                needed.update(['HbR', 'HbO'])
            if '2' in item:
                needed.update(['HbR', 'EEG'])
            if '3' in item:
                needed.update(['HbO', 'EEG'])
            if '4' in item:
                needed.update(['HbR', 'HbO', 'EEG'])
        else:
            needed.add(item)

    if isinstance(strategy, (list, tuple)):
        for item in strategy:
            parse(item)
    else:
        parse(strategy)
    return sorted(needed)


def get_subject_ids(h5_path, task):
    with h5py.File(h5_path, 'r') as f:
        if task not in f:
            return []
        return sorted(int(k.split('_')[1]) for k in f[task].keys()
                      if k.startswith('subject_'))


class HybridBCIDataset(Dataset):
    """One subject/task, windows in [samples, channels, time] layout."""

    def __init__(self, h5_path, subject_id, task, modality_strategy):
        self.h5_path = h5_path
        self.task = task
        self.subject_id = int(subject_id)
        self.modality_strategy = modality_strategy
        self.data_dict = {}
        self.labels = None
        self.groups = None
        self.num_samples = 0
        self.time_dims = {}
        self.base_modalities_needed = determine_base_modalities(modality_strategy)
        self._load_data()

    def _load_data(self):
        group_path = f'{self.task}/subject_{self.subject_id:02d}'
        with h5py.File(self.h5_path, 'r') as f:
            if group_path not in f:
                return

            for mod in self.base_modalities_needed:
                path = f'{group_path}/{mod.lower()}_data'
                if path not in f:
                    continue
                # File layout: [time, channels, samples] -> [samples, channels, time]
                data = torch.from_numpy(f[path][:]).float().permute(2, 1, 0)
                self.data_dict[mod] = data
                self.time_dims[mod] = data.shape[2]

            if not self.data_dict:
                return

            labels = torch.from_numpy(f[f'{group_path}/labels_decimal'][:]).long().squeeze()
            if labels.numel() and labels.min() > 0:
                labels = labels - 1
            self.labels = labels
            self.num_samples = len(labels)

            if f'{group_path}/trial_ids' not in f:
                raise KeyError(
                    f'{group_path}/trial_ids missing. Regenerate the HDF5 file '
                    'with the current MATLAB pipeline.')
            trial_ids = np.asarray(f[f'{group_path}/trial_ids'][:]).squeeze()
            self.groups = trial_ids
            self.n_trials = len(np.unique(trial_ids))
            if self.n_trials == 0 or self.num_samples % self.n_trials != 0:
                raise ValueError(
                    f'Cannot infer windows per trial for {group_path}: '
                    f'{self.num_samples} samples, {self.n_trials} trials.')
            self.n_steps = self.num_samples // self.n_trials

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'data': {mod: self.data_dict[mod][idx] for mod in self.base_modalities_needed},
            'label': self.labels[idx],
        }


class CrossSubjectDataset(Dataset):
    """Concatenate several subject datasets, optionally with sample weights."""

    def __init__(self, h5_path, subject_ids, task, modality_strategy,
                 subject_weights=None):
        self.sub_datasets = []
        self.sample_map = []
        self.weights = []
        self.total_samples = 0

        if subject_weights is None:
            subject_weights = np.ones(len(subject_ids))
        if len(subject_weights) != len(subject_ids):
            raise ValueError('subject_weights must match subject_ids.')

        for i, sid in enumerate(subject_ids):
            ds = HybridBCIDataset(h5_path, sid, task, modality_strategy)
            if len(ds) == 0:
                continue
            self.sub_datasets.append(ds)
            weight = float(subject_weights[i])
            for local_idx in range(len(ds)):
                self.sample_map.append((len(self.sub_datasets) - 1, local_idx))
                self.weights.append(weight)
            self.total_samples += len(ds)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        ds_idx, local_idx = self.sample_map[idx]
        return self.sub_datasets[ds_idx][local_idx]
