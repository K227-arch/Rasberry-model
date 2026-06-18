"""
ContrastiveLoss
===============
Implements contrastive learning for NMT encoder representations.

The idea: encoder embeddings of source sentences should be closer to
their correct translations and further from incorrect translations
(negatives drawn from the same batch).

Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss,
the same objective used in SimCLR / CLIP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss on sentence-level encoder embeddings.

    Args:
        temperature: scaling factor (lower = sharper distribution)
        reduction: 'mean' or 'sum'
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        src_embeddings: torch.Tensor,   # (B, D) — source sentence representations
        tgt_embeddings: torch.Tensor,   # (B, D) — target sentence representations
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Positive pairs: (src_i, tgt_i)
        Negative pairs: all other (src_i, tgt_j) where j ≠ i within batch
        """
        batch_size = src_embeddings.size(0)

        # L2 normalise
        src_norm = F.normalize(src_embeddings, p=2, dim=-1)
        tgt_norm = F.normalize(tgt_embeddings, p=2, dim=-1)

        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(src_norm, tgt_norm.T) / self.temperature

        # Labels: diagonal is the positive pair
        labels = torch.arange(batch_size, device=src_embeddings.device)

        # Cross-entropy loss (source-to-target)
        loss_src = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)

        # Cross-entropy loss (target-to-source)
        loss_tgt = F.cross_entropy(sim_matrix.T, labels, reduction=self.reduction)

        return (loss_src + loss_tgt) / 2.0


def mean_pool_encoder(
    encoder_outputs: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool encoder hidden states (masked) to get sentence embedding.

    Args:
        encoder_outputs: (B, seq_len, hidden_dim)
        attention_mask: (B, seq_len)

    Returns:
        sentence_embedding: (B, hidden_dim)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    summed = (encoder_outputs * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts
