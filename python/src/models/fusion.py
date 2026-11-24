
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from torch.nn import MultiheadAttention


class ConcatFusion(nn.Module):
    """Concatenate the projected features of all modalities."""

    def forward(self, *features):
        valid_features = [f for f in features if f is not None]
        if not valid_features: return None
        if len(valid_features) == 1: return valid_features[0]
        return torch.cat(valid_features, dim=1)


class CoAttentionFusion(nn.Module):
    """Feature-level co-attention for two or three modalities."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_layers = None
        self.final_fusion_layer = None

        self.attention_maps = {}
        self.modality_names = ['EEG', 'HbO', 'HbR']

    def _initialize_layers(self, features: tuple):
        num_modalities = len(features)
        device = features[0].device
        self.proj_layers = nn.ModuleList([nn.LazyLinear(self.hidden_dim) for _ in range(num_modalities)])

        self.final_fusion_layer = nn.Sequential(
            nn.LazyLinear(self.hidden_dim * num_modalities),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.proj_layers.to(device)
        self.final_fusion_layer.to(device)
        print(f"CoAttentionFusion initialized for {num_modalities} modalities on device: {device}.")

    def forward(self, *features):
        """
         (Batch, Dim).
        """
        if self.final_fusion_layer is None:
            self._initialize_layers(features)

        self.attention_maps = {}

        num_modalities = len(features)
        if num_modalities < 2:
            return features[0]

        current_names = self.modality_names[:num_modalities]

        # 1. Projection (Non-linear)
        projected_features = [torch.tanh(self.proj_layers[i](features[i])) for i in range(num_modalities)]

        attended_feature_representations = []

        # 2. Co-Attention Calculation
        for i in range(num_modalities):
            target_feature = projected_features[i]  # (B, D)
            target_name = current_names[i]

            weighted_contexts = []

            # Loop over Context modalities
            for j in range(num_modalities):
                if i == j: continue

                context_feature = projected_features[j]  # (B, D)
                context_name = current_names[j]

                # [MODIFIED] Feature-Level Attention
                # Affinity: (D, B) @ (B, D) -> (D, D) (Feature Correlation)
                affinity_matrix = target_feature.T @ context_feature

                # Softmax on Feature Dimension (dim=1)
                attn_weights = F.softmax(affinity_matrix, dim=1)

                # [NEW] Save Map for Visualization
                map_key = f"{target_name}_from_{context_name}"
                self.attention_maps[map_key] = attn_weights.detach().cpu().numpy()

                # Apply Attention: (B, D) @ (D, D).T -> (B, D)
                # Explanation: Re-weight context features based on target preference
                weighted_context = context_feature @ attn_weights.T
                weighted_contexts.append(weighted_context)

            # 3. Aggregation (Mean)
            if weighted_contexts:
                attended_feature_representations.append(torch.stack(weighted_contexts, dim=0).mean(dim=0))

        # 4. Final Concatenation
        final_concat_list = []
        for i in range(num_modalities):
            final_concat_list.append(features[i])  # Original
            final_concat_list.append(attended_feature_representations[i])  # Co-Attended

        fused_features = torch.cat(final_concat_list, dim=1)
        final_output = self.final_fusion_layer(fused_features)

        return final_output




class SequenceCoAttentionFusion(nn.Module):
    """Temporally-guided co-attention (TGCA) for sequence features.

    A band mask restricts cross-modal attention to temporally nearby positions,
    which matches the alignment produced by DTS.
    """

    def __init__(self, hidden_dim=128, n_heads=4, temporal_window=5, dropout=0.3):
        super().__init__()
        self.d_model = hidden_dim
        self.n_heads = n_heads
        self.window = temporal_window

        self.proj_layers = None
        self.mha_layers = None
        self.final_fusion_layer = None

        self.attention_maps = {}
        self.modality_names = ['EEG', 'HbO', 'HbR']

    def _generate_temporal_mask(self, len_q, len_k, device):
        """
         ratio = len_k / len_q EEG (50)  fNIRS (80)  DTS
         Attention  temporal_window  DTS

         (Band Mask)
        : (len_q, len_k)
        : 0  (Keep), -inf  (Mask)

        :  Query  i   Key  i * (Lk/Lq)
               +/- window
        """
        if self.window <= 0:
            return None

        mask = torch.full((len_q, len_k), float('-inf'), device=device)

        ratio = len_k / len_q if len_q > 0 else 1.0

        for i in range(len_q):
            center = i * ratio

            start = max(0, int(math.floor(center - self.window)))
            end = min(len_k, int(math.ceil(center + self.window + 1)))

            mask[i, start:end] = 0.0

        return mask

    def _initialize_layers(self, features: tuple):
        num_modalities = len(features)
        device = features[0].device

        self.proj_layers = nn.ModuleList([
            nn.LazyLinear(self.d_model) for _ in range(num_modalities)
        ]).to(device)

        self.mha_layers = nn.ModuleList([
            MultiheadAttention(self.d_model, self.n_heads, dropout=0.1, batch_first=True)
            for _ in range(num_modalities)
        ]).to(device)

        self.final_fusion_layer = nn.Sequential(
            nn.Linear(2 * num_modalities * self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.d_model * 2, self.d_model)
        ).to(device)

        print(f"SequenceCoAttentionFusion (TGCA) initialized for {num_modalities} modalities.")

    def forward(self, *features):
        """
        Args:
            *features: tuple of (Batch, Seq_Len_i, Dim_i)
                        Seq_Len
        Returns:
            (Batch, Hidden_Dim) ->
        """
        if self.final_fusion_layer is None:
            self._initialize_layers(features)

        self.attention_maps = {}
        num_modalities = len(features)
        device = features[0].device

        current_names = self.modality_names[:num_modalities]

        projs = [torch.tanh(self.proj_layers[i](feat)) for i, feat in enumerate(features)]

        attended_outputs = []

        for i in range(num_modalities):
            query = projs[i]  # (B, Sq, D)
            B, Sq, D = query.shape

            context_accumulator = torch.zeros_like(query)

            count = 0
            for j in range(num_modalities):
                if i == j: continue

                key_val = projs[j]  # (B, Sk, D)
                Sk = key_val.shape[1]

                # mask shape: (Sq, Sk)
                attn_mask = self._generate_temporal_mask(Sq, Sk, device)

                # --- Cross Attention ---
                # output: (B, Sq, D), weights: (B, Sq, Sk)
                attn_out, attn_weights = self.mha_layers[i](
                    query, key_val, key_val,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True
                )

                context_accumulator += attn_out
                count += 1

                map_key = f"{current_names[i]}_attend_{current_names[j]}"
                self.attention_maps[map_key] = attn_weights[0].detach().cpu().numpy()

            if count > 0:
                context_accumulator /= count

            attended_outputs.append(query + context_accumulator)

        pooled_vectors = []
        for seq_feat in attended_outputs:
            # (B, S, D) -> (B, D)
            pooled_vectors.append(seq_feat.mean(dim=1))

        pooled_originals = []
        for seq_feat in projs:
            pooled_originals.append(seq_feat.mean(dim=1))

        concat_list = []
        for k in range(num_modalities):
            concat_list.append(pooled_originals[k])
            concat_list.append(pooled_vectors[k])

        global_feature = torch.cat(concat_list, dim=1)  # (B, M * 2 * D)

        final_output = self.final_fusion_layer(global_feature)  # (B, D)

        return final_output


class GatedFusion(nn.Module):
    """Gated multimodal fusion with learned per-modality weights."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_layers = None
        self.gate_network = None

    def _initialize_layers(self, features: tuple):
        """

        """
        num_modalities = len(features)
        device = features[0].device

        self.proj_layers = nn.ModuleList([
            nn.LazyLinear(self.hidden_dim) for _ in range(num_modalities)
        ]).to(device)

        concatenated_dim = num_modalities * self.hidden_dim
        self.gate_network = nn.Sequential(
            nn.Linear(concatenated_dim, num_modalities),
        ).to(device)
        print(f"GatedFusion initialized for {num_modalities} modalities with hidden_dim={self.hidden_dim}.")

    def forward(self, *features):
        """

        """
        num_modalities = len(features)
        if num_modalities == 0:
            return None
        if num_modalities == 1:
            return features[0]

        if self.proj_layers is None:
            self._initialize_layers(features)

        projected_features = [self.proj_layers[i](f) for i, f in enumerate(features)]

        concatenated_features = torch.cat(projected_features, dim=1)

        gate_logits = self.gate_network(concatenated_features)
        gate_weights = F.softmax(gate_logits, dim=1)  # (batch_size, num_modalities)

        stacked_features = torch.stack(projected_features, dim=1)

        gate_weights = gate_weights.unsqueeze(2)

        weighted_features = stacked_features * gate_weights
        fused_features = torch.sum(weighted_features, dim=1)

        return fused_features


class LMFusion(nn.Module):
    """Low-rank multimodal fusion (LMF)."""

    def __init__(self, rank=128, dropout_rate=0.5):
        super().__init__()
        self.rank = rank
        self.dropout_rate = dropout_rate
        self.proj_layers = None
        self.fusion_dropout = None

    def _initialize_layers(self, features: tuple):
        device = features[0].device
        num_modalities = len(features)

        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(self.rank),
                nn.ReLU()
            ) for _ in range(num_modalities)
        ]).to(device)

        self.fusion_dropout = nn.Dropout(self.dropout_rate).to(device)
        print(f"LMFusion initialized for {num_modalities} modalities with rank={self.rank}.")

    def forward(self, *features):
        num_modalities = len(features)
        if num_modalities == 0:
            return None
        if num_modalities == 1:
            return features[0]

        if self.proj_layers is None:
            self._initialize_layers(features)

        projected_features = [self.proj_layers[i](f) for i, f in enumerate(features)]

        fused_tensor = torch.ones_like(projected_features[0])
        for proj_feat in projected_features:
            fused_tensor = fused_tensor * proj_feat

        fused_tensor = self.fusion_dropout(fused_tensor)

        return fused_tensor
