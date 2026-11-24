"""Per-channel standardization fitted on a training subset."""

import torch
from torch.utils.data import Dataset


def fit_channel_stats(subset, modalities, eps=1e-8):
    """Fit per-channel mean/std on ``subset``.

    Statistics are accumulated over every window and time point of the
    selected samples, independently for each channel and modality.
    """
    modalities = list(modalities)
    sums = {m: None for m in modalities}
    sumsq = {m: None for m in modalities}
    counts = {m: 0 for m in modalities}

    with torch.no_grad():
        for i in range(len(subset)):
            sample = subset[i]
            for m in modalities:
                x = sample['data'][m].double()
                s = x.sum(dim=1)
                q = (x * x).sum(dim=1)
                if sums[m] is None:
                    sums[m] = s
                    sumsq[m] = q
                else:
                    sums[m] += s
                    sumsq[m] += q
                counts[m] += x.shape[1]

    stats = {}
    for m in modalities:
        mean = sums[m] / counts[m]
        var = sumsq[m] / counts[m] - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0)) + eps
        stats[m] = (mean.float(), std.float())
    return stats


class StandardizedSubset(Dataset):
    """Apply fixed per-channel statistics to every sample of a subset."""

    def __init__(self, subset, stats):
        self.subset = subset
        self.stats = stats

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        data = {}
        for m, x in sample['data'].items():
            mean, std = self.stats[m]
            data[m] = (x - mean[:, None]) / std[:, None]
        return {'data': data, 'label': sample['label']}


class WindowZScoreSubset(Dataset):
    """Standardize every window per channel over time."""

    def __init__(self, subset, eps=1e-8):
        self.subset = subset
        self.eps = eps

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]
        data = {}
        for m, x in sample['data'].items():
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=True)
            data[m] = (x - mean) / (std + self.eps)
        return {'data': data, 'label': sample['label']}
