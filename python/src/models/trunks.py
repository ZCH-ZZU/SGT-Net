"""Transformer backbone with differentiable temporal synchronization (DTS).

Only the plain CNN + Transformer trunk used in the paper is kept here. The
DTS module applies a fixed 4 s advance (40 steps at 10 Hz after CNN pooling)
plus a bounded learnable residual of +-10 steps, i.e. an effective 3-5 s
alignment range. EEG is the anchor and is never shifted.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """Two-layer temporal CNN used in front of the Transformer."""

    def __init__(self, in_channels, out_channels=16, kernel_size1=32,
                 kernel_size2=16, pool_size1=4, pool_size2=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, kernel_size1), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_size1))
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(8, out_channels, (1, kernel_size2), padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, pool_size2))
        self.drop2 = nn.Dropout(dropout)

        self.out_channels = out_channels
        self.total_pool = pool_size1 * pool_size2

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.act1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.act2(self.bn2(self.conv2(x)))))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm-free Transformer layer with GELU feed-forward."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class TemporalShift(nn.Module):
    """Differentiable temporal shift with a fixed 4 s advance.

    ``total = base_shift + clamp(delta, -max_shift, max_shift)`` in time steps.
    With 80 fNIRS steps (8 s at 10 Hz), base_shift=40 corresponds to 4 s and
    the +-10-step residual gives a 3-5 s search range.
    """

    def __init__(self, sequence_length, init_shift=0.0, max_shift=10.0,
                 base_shift=40.0):
        super().__init__()
        self.seq_len = sequence_length
        self.shift = nn.Parameter(torch.tensor([init_shift], dtype=torch.float32))
        self.max_shift = max_shift
        self.base_shift = base_shift

    def forward(self, x):
        # x: (B, T, C)
        b, t, c = x.shape
        x = x.permute(0, 2, 1)

        total = torch.clamp(self.shift, -self.max_shift, self.max_shift) + self.base_shift
        norm_factor = 2.0 / (t - 1) if t > 1 else 1.0
        base_grid = torch.linspace(-1, 1, t, device=x.device).unsqueeze(0).expand(b, t)
        new_grid = base_grid + total * norm_factor

        x_4d = x.unsqueeze(2)
        zeros = torch.zeros_like(new_grid)
        grid = torch.stack([new_grid, zeros], dim=-1).unsqueeze(1)
        out = F.grid_sample(x_4d, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(2).permute(0, 2, 1)


class TransformerTrunk(nn.Module):
    """CNN + optional DTS + positional encoding + Transformer encoder."""

    def __init__(self, num_channels, timepoints, d_model=128, n_heads=8,
                 num_layers=2, dropout=0.3, use_cnn=True, cnn_params=None,
                 pe_type='spe', use_dts=False, init_shift=0.0,
                 output_mode='global'):
        super().__init__()
        if not use_cnn:
            raise ValueError('TransformerTrunk requires the CNN frontend.')

        self.use_dts = use_dts
        self.output_mode = output_mode
        self.timepoints = timepoints

        if cnn_params is None:
            cnn_params = dict(kernel_size1=32, kernel_size2=16,
                              pool_size1=4, pool_size2=2, out_channels=16)
        self.cnn_frontend = CNNFeatureExtractor(num_channels, dropout=dropout, **cnn_params)
        pool_factor = self.cnn_frontend.total_pool
        seq_len = max(1, timepoints // pool_factor)
        input_dim = cnn_params.get('out_channels', 16) * num_channels
        self.input_proj = nn.Linear(input_dim, d_model)

        if self.use_dts:
            self.dts_module = TemporalShift(seq_len, init_shift=init_shift)

        if pe_type.lower() in ['learnable', 'lpe']:
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=seq_len + 20, dropout=dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 20)

        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, return_mid_feats=False):
        # x: (B, C, T). Truncate to the configured window (2 s for w/o DTS).
        if x.shape[-1] > self.timepoints:
            x = x[:, :, :self.timepoints]
        elif x.shape[-1] < self.timepoints:
            raise ValueError(
                f'Input timepoints {x.shape[-1]} is smaller than expected {self.timepoints}.')

        x = self.cnn_frontend(x)
        mid_feats = None
        if return_mid_feats:
            b, p, c, t = x.shape
            mid_feats = x.reshape(b, p * c, t)

        b, p, c, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        x = self.input_proj(x)

        if self.use_dts:
            x = self.dts_module(x)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x if self.output_mode == 'sequence' else x.mean(dim=1)
        return (out, mid_feats) if return_mid_feats else out
