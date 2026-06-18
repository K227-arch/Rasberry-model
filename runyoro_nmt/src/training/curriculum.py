"""
CurriculumSampler
=================
Implements curriculum learning for NMT:
  - Stage 1: short / simple sentences (≤ 30 tokens)
  - Stage 2: medium sentences (≤ 80 tokens)
  - Stage 3: all sentences (≤ 200 tokens)

The sampler sorts training examples by difficulty (sentence length as proxy)
and progressively introduces harder examples as training advances.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class CurriculumSampler(Sampler):
    """
    A sampler that returns indices ordered by sentence complexity.
    Curriculum stage is controlled externally by the trainer.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        stage_max_tokens: int = 200,
        shuffle_within_stage: bool = True,
        seed: int = 42,
    ):
        self.pairs = pairs
        self.stage_max_tokens = stage_max_tokens
        self.shuffle = shuffle_within_stage
        self.seed = seed

        # Pre-compute difficulty (max of src/tgt token count)
        self.difficulties = [
            max(len(rny.split()), len(eng.split()))
            for rny, eng in pairs
        ]

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    def set_stage_max_tokens(self, max_tokens: int) -> None:
        self.stage_max_tokens = max_tokens
        logger.info("Curriculum stage updated: max_tokens=%d", max_tokens)

    def __iter__(self):
        eligible = [
            i for i, d in enumerate(self.difficulties)
            if d <= self.stage_max_tokens
        ]

        if self.shuffle:
            perm = torch.randperm(len(eligible), generator=self._generator).tolist()
            eligible = [eligible[p] for p in perm]

        logger.debug(
            "CurriculumSampler: %d/%d samples eligible at max_tokens=%d",
            len(eligible), len(self.pairs), self.stage_max_tokens,
        )
        return iter(eligible)

    def __len__(self) -> int:
        return sum(1 for d in self.difficulties if d <= self.stage_max_tokens)
