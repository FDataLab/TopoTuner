"""Random K-way disjoint index subsets for eval (greedy decoding, mean±std across parts).

Not cross-validation: same fixed pool and splits for all checkpoints; no retraining per part.
"""

from __future__ import annotations

from typing import List

import numpy as np


def random_k_split_indices(n: int, k: int, seed: int) -> List[List[int]]:
    """Shuffle ``0..n-1`` with ``seed``, partition into ``k`` disjoint subsets (roughly equal size).

    Returns lists of **row indices** into the eval pool (after any limit/stratify). Indices refer to
    positions ``0 .. n-1`` in pool order.

    Empty subsets are dropped (can happen when ``n < k``).
    """
    if n <= 0:
        return []
    if k <= 0:
        raise ValueError("k must be positive")
    rng = np.random.RandomState(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    parts = np.array_split(idx, k)
    return [p.tolist() for p in parts if len(p) > 0]
