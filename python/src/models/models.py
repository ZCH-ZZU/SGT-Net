"""Multi-branch Transformer model used in all three stages."""

import torch
import torch.nn as nn

from .fusion import (ConcatFusion, CoAttentionFusion, SequenceCoAttentionFusion,
                     GatedFusion, LMFusion)
from .trunks import TransformerTrunk

SEQUENCE_FUSIONS = {'tgca', 'sequence_co_attention'}


class MultiBranchNet(nn.Module):
    def __init__(self, channels_dict, num_classes, fusion_type='tgca',
                 fusion_params=None, trunk_params=None, use_dts=True):
        super().__init__()
        self.modalities = sorted(list(channels_dict.keys()))
        self.is_single_modality = len(self.modalities) == 1
        fusion_params = fusion_params or {}
        trunk_params = trunk_params or {}

        output_mode = 'sequence' if fusion_type in SEQUENCE_FUSIONS else 'global'

        self.trunks = nn.ModuleDict()
        for mod in self.modalities:
            num_channels = channels_dict[mod]['channels']
            mod_params = trunk_params.get(mod, {})
            if not mod_params and mod in ['HbO', 'HbR']:
                mod_params = trunk_params.get('NIRS', {})

            is_eeg = 'EEG' in mod
            mod_use_dts = False if is_eeg else bool(mod_params.get('use_dts', use_dts))

            # EEG is the anchor: 400 points (2 s @ 200 Hz), never shifted.
            # Single-modality fNIRS: 20 points (2 s @ 10 Hz).
            # Multi-modal + DTS: 80 points (8 s @ 10 Hz).
            # Multi-modal without DTS: 20 points (2 s @ 10 Hz).
            if is_eeg:
                timepoints = 400
            elif self.is_single_modality:
                timepoints = 20
            elif mod_use_dts:
                timepoints = 80
            else:
                timepoints = 20

            params = {k: v for k, v in mod_params.items() if k != 'use_dts'}
            self.trunks[mod] = TransformerTrunk(
                num_channels=num_channels, timepoints=timepoints,
                output_mode=output_mode, use_dts=mod_use_dts, **params)

        if self.is_single_modality:
            self.deep_feature_fusion = nn.Identity()
        elif fusion_type == 'concat':
            self.deep_feature_fusion = ConcatFusion()
        elif fusion_type == 'co_attention':
            self.deep_feature_fusion = CoAttentionFusion(**fusion_params)
        elif fusion_type in SEQUENCE_FUSIONS:
            self.deep_feature_fusion = SequenceCoAttentionFusion(**fusion_params)
        elif fusion_type == 'gated':
            self.deep_feature_fusion = GatedFusion(**fusion_params)
        elif fusion_type == 'lmf':
            self.deep_feature_fusion = LMFusion(**fusion_params)
        else:
            raise ValueError(f'Unsupported fusion_type: {fusion_type}')

        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, data_dict, return_features=False, return_mid_feats=False,
                return_split_feats=False):
        trunk_outputs = []
        mid_feats = {}
        for mod in self.modalities:
            if return_mid_feats:
                out, mid = self.trunks[mod](data_dict[mod], return_mid_feats=True)
                mid_feats[mod] = mid
            else:
                out = self.trunks[mod](data_dict[mod])
            trunk_outputs.append(out)

        if len(trunk_outputs) > 1:
            fused = self.deep_feature_fusion(*trunk_outputs)
        else:
            fused = trunk_outputs[0]
            if fused.dim() == 3:
                fused = fused.mean(dim=1)

        logits = self.classifier(fused)
        if return_features or return_mid_feats or return_split_feats:
            ret = [logits]
            if return_features:
                ret.append(fused)
            if return_mid_feats:
                ret.append(mid_feats)
            if return_split_feats:
                def _pool(out):
                    return out.mean(dim=1) if out.dim() == 3 else out
                eeg = _pool(trunk_outputs[0]) if len(trunk_outputs) > 0 else None
                nirs = _pool(trunk_outputs[1]) if len(trunk_outputs) > 1 else None
                ret.extend([eeg, nirs])
            return tuple(ret)
        return logits


def get_model(model_config):
    return MultiBranchNet(
        channels_dict=model_config['channels_dict'],
        num_classes=model_config['num_classes'],
        fusion_type=model_config.get('fusion_type', 'tgca'),
        fusion_params=model_config.get('fusion_params'),
        trunk_params=model_config.get('trunk_params'),
        use_dts=model_config.get('use_dts', True),
    )
