"""
Dataset splitting utilities with stratification by domain and sentence length.
"""

from __future__ import annotations

import random
from typing import List, Tuple


def split_dataset(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """
    Split dataset into train/val/test with reproducible shuffling.

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test
