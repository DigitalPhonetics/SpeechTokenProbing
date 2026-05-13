"""Shared statistical helpers for minimal probe analyses.

This module contains reusable utilities for:
    - reproducible random seeding,
    - Jensen-Shannon divergence from count vectors,
    - token distribution aggregation,
    - class balancing helpers,
    - speaker-disjoint splitting with optional utterance fallback,
    - pairwise coefficient-map similarity summaries.

Assumptions:
    - Token ids are integers represented as Python ``int`` values.
    - Binary labels are encoded as ``0`` and ``1``.
    - Split functions should keep both classes in train and test when possible.

This module is a library utility module (no CLI entrypoint).
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds.

    Args:
        seed: Seed value applied to ``random`` and ``numpy.random``.
    """

    random.seed(seed)
    np.random.seed(seed)


def jsd_from_counts(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence from nonnegative count vectors.

    Args:
        v1: First count vector.
        v2: Second count vector.

    Returns:
        Jensen-Shannon divergence squared (base-2), matching the paper
        convention. Returns ``nan`` when either vector has zero total mass.
    """

    a = np.asarray(v1, dtype=np.float64)
    b = np.asarray(v2, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    a = a / a.sum()
    b = b / b.sum()
    return float(jensenshannon(a, b, base=2) ** 2)


def token_distribution(token_lists: Sequence[List[int]]) -> Tuple[Dict[int, float], Counter]:
    """Aggregate token sequences into normalized token frequencies.

    Args:
        token_lists: Sequence of token-id sequences.

    Returns:
        Tuple ``(distribution, counts)`` where ``distribution`` maps token id to
        relative frequency and ``counts`` is the raw :class:`collections.Counter`.
    """

    ctr = Counter()
    for seq in token_lists:
        ctr.update(seq)
    total = float(sum(ctr.values()))
    if total <= 0:
        return {}, ctr
    return {int(t): float(c) / total for t, c in ctr.items()}, ctr


def sample_balanced_groups(
    row_ids_pos: List[str],
    row_ids_neg: List[str],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Randomly downsample two id lists to equal size.

    Args:
        row_ids_pos: Candidate positive-row ids.
        row_ids_neg: Candidate negative-row ids.
        seed: RNG seed for deterministic sampling.

    Returns:
        Tuple ``(pos_ids, neg_ids)`` with matched lengths. Returns empty lists
        when either input list is empty.
    """

    n = min(len(row_ids_pos), len(row_ids_neg))
    if n <= 0:
        return [], []
    rng = random.Random(seed)
    return rng.sample(row_ids_pos, n), rng.sample(row_ids_neg, n)


def split_speaker_disjoint(
    y: Sequence[int],
    speakers: Sequence[str],
    test_size: float,
    seed: int,
    max_tries: int,
    allow_utterance_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Create train/test indices with speaker disjointness when possible.

    The function first attempts group-aware shuffling over speakers. A split is
    accepted only if both train and test contain both classes. If no such split
    is found and utterance fallback is allowed, it falls back to stratified
    utterance-level splitting.

    Args:
        y: Binary labels per sample.
        speakers: Speaker id per sample.
        test_size: Fraction assigned to test split.
        seed: Base random seed.
        max_tries: Number of random states attempted for group splitting.
        allow_utterance_fallback: Whether to allow non-speaker-disjoint
            stratified fallback.

    Returns:
        Tuple ``(train_indices, test_indices, split_meta)`` where ``split_meta``
        records effective split mode and class balance.

    Raises:
        ValueError: If labels are single-class, or when no valid speaker split
            is found and fallback is disabled.
    """

    y_arr = np.asarray(y, dtype=np.int64)
    idx = np.arange(len(y_arr))

    if len(set(y_arr.tolist())) < 2:
        raise ValueError("Need at least two classes for split")

    unique_spk = set(speakers)
    if len(unique_spk) >= 2:
        for k in range(max(1, int(max_tries))):
            rs = int(seed + k)
            gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=rs)
            tr, te = next(gss.split(idx, y_arr, groups=np.asarray(speakers, dtype=object)))
            ytr, yte = y_arr[tr], y_arr[te]
            if len(set(ytr.tolist())) >= 2 and len(set(yte.tolist())) >= 2:
                return tr, te, {
                    "mode": "speaker",
                    "random_state": rs,
                    "train_size": int(len(tr)),
                    "test_size": int(len(te)),
                    "train_pos_rate": float(np.mean(ytr)),
                    "test_pos_rate": float(np.mean(yte)),
                }

    if not allow_utterance_fallback:
        raise ValueError("Could not find valid speaker-disjoint split with both classes in train and test")

    tr, te = train_test_split(
        idx,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y_arr,
    )
    ytr, yte = y_arr[tr], y_arr[te]
    return tr, te, {
        "mode": "utterance_fallback",
        "random_state": int(seed),
        "train_size": int(len(tr)),
        "test_size": int(len(te)),
        "train_pos_rate": float(np.mean(ytr)),
        "test_pos_rate": float(np.mean(yte)),
    }


def pairwise_cosine_and_mad(weight_maps: Dict[str, Dict[int, float]]) -> List[dict]:
    """Compute pairwise cosine similarity and mean absolute delta across stages.

    Args:
        weight_maps: Mapping from stage name to token-weight dictionary.

    Returns:
        List of pairwise summary rows with keys
        ``stage_a``, ``stage_b``, ``cosine``, and ``mean_abs_delta``.
    """

    stages = sorted(weight_maps.keys())
    rows: List[dict] = []
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            a, b = stages[i], stages[j]
            wa, wb = weight_maps[a], weight_maps[b]
            vocab = sorted(set(wa.keys()) | set(wb.keys()))
            if not vocab:
                continue
            va = np.array([float(wa.get(t, 0.0)) for t in vocab], dtype=np.float64)
            vb = np.array([float(wb.get(t, 0.0)) for t in vocab], dtype=np.float64)
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            if na == 0.0 or nb == 0.0:
                cos = float("nan")
            else:
                cos = float(np.dot(va, vb) / (na * nb))
            mad = float(np.mean(np.abs(va - vb)))
            rows.append({"stage_a": a, "stage_b": b, "cosine": cos, "mean_abs_delta": mad})
    return rows
