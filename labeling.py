"""Label construction utilities for binary and quantile probe setups.

This module converts row-level labels into a binary target mapping used by all
three probe methods. It supports:
    - explicit binary labels (for example gender M/F),
    - continuous score quantile binning (for example VAD valence).

Assumptions:
    - Input rows are already normalized :class:`minimal_probe.io.ProbeRow`.
    - Exactly one labeling mode is selected per run.
    - Quantile labeling includes threshold ties according to the implemented
      ``<= lo`` and ``>= hi`` rules.

This module is a library utility module (no CLI entrypoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .io import ProbeRow, get_nested_label


@dataclass
class LabelingResult:
    """Container for resolved binary labels and labeling metadata.

    Attributes:
        labels_by_id: Mapping from row id to binary class id (1 positive,
            0 negative).
        positive_name: Human-readable positive class name used in reports.
        negative_name: Human-readable negative class name used in reports.
        mode: Labeling mode identifier (``"binary"`` or ``"quantile"``).
        meta: Additional method-specific metadata (field names, thresholds, and
            class counts).
    """

    labels_by_id: Dict[str, int]
    positive_name: str
    negative_name: str
    mode: str
    meta: dict


def labels_from_binary_field(rows: List[ProbeRow], label_field: str, pos_label: str, neg_label: str) -> LabelingResult:
    """Build binary labels from an explicit categorical label field.

    Args:
        rows: Probe rows to inspect.
        label_field: Label field path to read from each row.
        pos_label: Raw label value mapped to class ``1``.
        neg_label: Raw label value mapped to class ``0``.

    Returns:
        A :class:`LabelingResult` with binary labels and class counts.

    Raises:
        ValueError: If either class is empty after filtering.
    """

    out: Dict[str, int] = {}
    pos, neg = str(pos_label), str(neg_label)
    for r in rows:
        v = get_nested_label(r.labels, label_field)
        if v is None:
            continue
        sv = str(v).strip()
        if sv == pos:
            out[r.row_id] = 1
        elif sv == neg:
            out[r.row_id] = 0

    n_pos = sum(1 for y in out.values() if y == 1)
    n_neg = sum(1 for y in out.values() if y == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Binary label split empty for {label_field} with pos={pos!r} neg={neg!r}: pos={n_pos}, neg={n_neg}"
        )

    return LabelingResult(
        labels_by_id=out,
        positive_name=pos,
        negative_name=neg,
        mode="binary",
        meta={"label_field": label_field, "num_pos": n_pos, "num_neg": n_neg},
    )


def labels_from_quantiles(rows: List[ProbeRow], score_field: str, qlo: float, qhi: float) -> LabelingResult:
    """Build binary labels by low/high quantile binning of a numeric score.

    Values at or below the low threshold are assigned to class ``0`` and values
    at or above the high threshold are assigned to class ``1``. Values between
    thresholds are discarded.

    Args:
        rows: Probe rows to inspect.
        score_field: Numeric label field path.
        qlo: Low quantile in ``[0, 1]``.
        qhi: High quantile in ``[0, 1]``.

    Returns:
        A :class:`LabelingResult` with quantile thresholds and class counts.

    Raises:
        ValueError: If too few numeric labels are available or either class is
            empty after quantile filtering.
    """

    vals: List[Tuple[str, float]] = []
    for r in rows:
        v = get_nested_label(r.labels, score_field)
        try:
            vals.append((r.row_id, float(v)))
        except Exception:
            continue

    if len(vals) < 2:
        raise ValueError(f"Too few numeric labels for score field {score_field!r}")

    arr = np.array([v for _, v in vals], dtype=np.float64)
    lo = float(np.quantile(arr, qlo))
    hi = float(np.quantile(arr, qhi))

    out: Dict[str, int] = {}
    for rid, v in vals:
        if v <= lo:
            out[rid] = 0
        elif v >= hi:
            out[rid] = 1

    n_pos = sum(1 for y in out.values() if y == 1)
    n_neg = sum(1 for y in out.values() if y == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Quantile label split empty for {score_field!r} qlo={qlo} qhi={qhi}: pos={n_pos}, neg={n_neg}"
        )

    return LabelingResult(
        labels_by_id=out,
        positive_name="H",
        negative_name="L",
        mode="quantile",
        meta={
            "score_field": score_field,
            "qlo": float(qlo),
            "qhi": float(qhi),
            "lo_threshold": lo,
            "hi_threshold": hi,
            "num_pos": n_pos,
            "num_neg": n_neg,
        },
    )


def resolve_labeling(
    rows: List[ProbeRow],
    label_field: str,
    pos_label: str,
    neg_label: str,
    score_field: str,
    qlo: float,
    qhi: float,
) -> LabelingResult:
    """Resolve exactly one labeling mode from CLI arguments.

    Args:
        rows: Probe rows to label.
        label_field: Binary label field path. Empty string disables binary mode.
        pos_label: Positive class value for binary mode.
        neg_label: Negative class value for binary mode.
        score_field: Numeric score field path. Empty string disables quantile
            mode.
        qlo: Low quantile for quantile mode.
        qhi: High quantile for quantile mode.

    Returns:
        Resolved labeling result for the selected mode.

    Raises:
        ValueError: If zero or multiple labeling modes are requested.
    """

    has_binary = bool(str(label_field).strip())
    has_quant = bool(str(score_field).strip())
    if has_binary == has_quant:
        raise ValueError("Choose exactly one labeling mode: binary (--label-field...) or quantile (--score-field...)")

    if has_binary:
        return labels_from_binary_field(rows, label_field=label_field, pos_label=pos_label, neg_label=neg_label)
    return labels_from_quantiles(rows, score_field=score_field, qlo=qlo, qhi=qhi)
