"""Parameter-efficient adaptation layers (LoRA and DoRA)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return (self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


class LinearWithLoRA(nn.Linear):
    """A frozen linear layer plus a trainable low-rank update."""

    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.1):
        super().__init__(linear_layer.in_features, linear_layer.out_features,
                         bias=linear_layer.bias is not None)
        self.weight.data = linear_layer.weight.data
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.data = linear_layer.bias.data
            self.bias.requires_grad = False
        self.lora = LoRALayer(self.in_features, self.out_features, rank, alpha, dropout)

    def forward(self, x):
        return super().forward(x) + self.lora(x)


class DoRALayer(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA).

    The pre-trained weight is frozen; a low-rank update changes its direction,
    while a trainable magnitude vector rescales each output row.
    """

    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.register_buffer('weight_0', linear_layer.weight.data.clone())
        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

        self.m = nn.Parameter(self.weight_0.norm(p=2, dim=1, keepdim=True))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    @property
    def weight(self):
        lora_weight = (self.lora_b @ self.lora_a) * self.scaling
        weight_prime = self.weight_0 + lora_weight
        norm = weight_prime.norm(p=2, dim=1, keepdim=True)
        return self.m * (weight_prime / (norm + 1e-6))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def inject_peft(model, method='lora', rank=4, alpha=16, dropout=0.1,
                target_modules=('input_proj',)):
    """Freeze the backbone and add the requested adaptation modules."""
    for param in model.parameters():
        param.requires_grad = False

    if method == 'none':
        return model
    if method == 'full':
        for param in model.parameters():
            param.requires_grad = True
        return model

    if method in ('lora', 'dora'):
        to_replace = []
        for name, module in model.named_modules():
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                to_replace.append((name, module))

        for name, module in to_replace:
            new_layer = (LinearWithLoRA(module, rank=rank, alpha=alpha, dropout=dropout)
                         if method == 'lora'
                         else DoRALayer(module, rank=rank, alpha=alpha, dropout=dropout))
            parent_name, child_name = name.rsplit('.', 1)
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, new_layer)

    if method in ('lora', 'dora'):
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            if method == 'dora' and name.endswith('.m'):
                param.requires_grad = True
            if 'bias' in name:
                param.requires_grad = True
    elif method == 'linear_probing':
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True

    return model
