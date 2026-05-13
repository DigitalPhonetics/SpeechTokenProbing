#!/usr/bin/env python3
"""CLI for Method 3: attribute-conditioned token-phone representation.

This script implements the paper's third probe method for stages that have
aligned token and phone sequences. It estimates phone distributions conditioned
on token and class, then reports cosine similarity, signed bias (lambda), and
token-level difference summaries under support/balance constraints.

Assumptions:
    - Input rows follow the canonical manifest schema.
    - Labels are binary after ``resolve_labeling``.
    - Aligned phone sequences are provided under ``phones[stage]`` and only
      exact length matches are considered usable.
    - Default thresholds follow paper settings unless overridden.

This module is both a library-style implementation (helper functions) and a CLI
entrypoint via :func:`main`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimal_probe.io import ensure_stage_exists, load_probe_rows, resolve_stage_list
from minimal_probe.labeling import resolve_labeling
from minimal_probe.stats import set_seed


def _cosine(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    """Compute cosine similarity between sparse phone distributions.

    Args:
        d1: First phone-probability mapping.
        d2: Second phone-probability mapping.

    Returns:
        Cosine similarity in ``[-1, 1]`` when both vectors have non-zero norm,
        otherwise ``nan``.
    """

    vocab = sorted(set(d1.keys()) | set(d2.keys()))
    if not vocab:
        return float("nan")
    v1 = np.array([float(d1.get(p, 0.0)) for p in vocab], dtype=np.float64)
    v2 = np.array([float(d2.get(p, 0.0)) for p in vocab], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    return float(np.dot(v1, v2) / (n1 * n2))


def _topk(d: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    """Return top-k items by descending value.

    Args:
        d: Mapping from key to numeric score.
        k: Maximum number of rows to return.

    Returns:
        Sorted ``(key, value)`` pairs.
    """

    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]


def _analyze_stage(
    rows,
    labels_by_id: Dict[str, int],
    stage: str,
    lambda_threshold: float,
    count_threshold: int,
    imbalance_threshold: float,
    top_k_phones: int,
    top_n_diff: int,
):
    """Run attribute-conditioned analysis for one stage.

    Args:
        rows: Loaded probe rows.
        labels_by_id: Binary labels keyed by row id.
        stage: Stage name to analyze (must have aligned phones).
        lambda_threshold: Absolute minimum signed-bias magnitude to count as
            biased.
        count_threshold: Minimum paired count ``cH + cL`` per token.
        imbalance_threshold: Minimum balance ratio
            ``min(cH, cL) / max(cH, cL)``.
        top_k_phones: Number of phones to include in per-token summaries.
        top_n_diff: Number of lowest-cosine tokens to keep.

    Returns:
        Stage result dictionary with ``ok`` and either:
            - success fields such as ``biased_positive``, ``biased_negative``,
              ``top_diff_tokens``, and skip/filter counters,
            - or ``error`` when no usable aligned rows exist.
    """

    ctr_pos = defaultdict(Counter)
    ctr_neg = defaultdict(Counter)
    c_pos = Counter()
    c_neg = Counter()

    used_rows = 0
    skipped_missing = 0
    skipped_len_mismatch = 0
    paired_positions = 0

    for r in rows:
        y = labels_by_id.get(r.row_id)
        if y is None:
            continue
        toks = r.token_sets.get(stage, [])
        phs = r.phones.get(stage, [])
        if not toks or not phs:
            skipped_missing += 1
            continue
        if len(toks) != len(phs):
            skipped_len_mismatch += 1
            continue

        used_rows += 1
        paired_positions += len(toks)
        for t, p in zip(toks, phs):
            t = int(t)
            p = str(p)
            if y == 1:
                ctr_pos[t][p] += 1
                c_pos[t] += 1
            else:
                ctr_neg[t][p] += 1
                c_neg[t] += 1

    tokens_all = sorted(set(c_pos.keys()) | set(c_neg.keys()))
    emb_pos: Dict[int, Dict[str, float]] = {}
    emb_neg: Dict[int, Dict[str, float]] = {}
    delta: Dict[int, Dict[str, float]] = {}
    cosine_by_token: Dict[int, float] = {}
    lambda_by_token: Dict[int, float] = {}
    dom_phone_by_token: Dict[int, str] = {}

    filtered_support = 0
    filtered_balance = 0

    if used_rows == 0:
        return {
            "ok": False,
            "stage": stage,
            "error": "no_usable_aligned_rows",
            "num_rows_used": 0,
            "paired_positions": 0,
            "skipped_missing": int(skipped_missing),
            "skipped_len_mismatch": int(skipped_len_mismatch),
        }

    for t in tokens_all:
        np_ = int(c_pos.get(t, 0))
        nn_ = int(c_neg.get(t, 0))
        if np_ + nn_ < int(count_threshold):
            filtered_support += 1
            continue
        mx = max(np_, nn_)
        bal_ratio = float(min(np_, nn_) / mx) if mx > 0 else 0.0
        if bal_ratio < float(imbalance_threshold):
            filtered_balance += 1
            continue

        dpos = ctr_pos[t]
        dneg = ctr_neg[t]
        sp = float(sum(dpos.values()))
        sn = float(sum(dneg.values()))
        if sp <= 0 or sn <= 0:
            continue

        ppos = {ph: float(c) / sp for ph, c in dpos.items()}
        pneg = {ph: float(c) / sn for ph, c in dneg.items()}
        emb_pos[t] = ppos
        emb_neg[t] = pneg

        cos = _cosine(ppos, pneg)
        cosine_by_token[t] = cos

        dd = {ph: float(ppos.get(ph, 0.0) - pneg.get(ph, 0.0)) for ph in sorted(set(ppos.keys()) | set(pneg.keys()))}
        delta[t] = dd
        dom_ph = max(dd.items(), key=lambda x: abs(x[1]))[0]
        lam = float(dd[dom_ph])
        dom_phone_by_token[t] = dom_ph
        lambda_by_token[t] = lam

    biased_pos = sorted([(int(t), dom_phone_by_token[t], lambda_by_token[t]) for t in lambda_by_token if lambda_by_token[t] >= lambda_threshold], key=lambda x: x[2], reverse=True)
    biased_neg = sorted([(int(t), dom_phone_by_token[t], lambda_by_token[t]) for t in lambda_by_token if lambda_by_token[t] <= -lambda_threshold], key=lambda x: x[2])

    diff_tokens = sorted([(int(t), float(cosine_by_token[t])) for t in cosine_by_token if not np.isnan(cosine_by_token[t])], key=lambda x: x[1])[: int(top_n_diff)]

    def _pack_bias(row):
        t, ph, lam = row
        np_ = int(c_pos.get(t, 0))
        nn_ = int(c_neg.get(t, 0))
        bal = float(min(np_, nn_) / max(np_, nn_)) if max(np_, nn_) > 0 else 0.0
        return {
            "token": int(t),
            "phone": str(ph),
            "lambda": float(lam),
            "count_positive": np_,
            "count_negative": nn_,
            "balance": bal,
            "top_positive_phones": _topk(emb_pos.get(t, {}), top_k_phones),
            "top_negative_phones": _topk(emb_neg.get(t, {}), top_k_phones),
            "cosine": float(cosine_by_token.get(t, float("nan"))),
        }

    top_bias_pos = [_pack_bias(x) for x in biased_pos[:5]]
    top_bias_neg = [_pack_bias(x) for x in biased_neg[:5]]

    top_diff_rows = []
    for t, cos in diff_tokens[:5]:
        dd = delta.get(t, {})
        top_abs = sorted(dd.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k_phones]
        top_diff_rows.append(
            {
                "token": int(t),
                "cosine": float(cos),
                "count_positive": int(c_pos.get(t, 0)),
                "count_negative": int(c_neg.get(t, 0)),
                "top_diffs": [(str(ph), float(v)) for ph, v in top_abs],
            }
        )

    return {
        "ok": True,
        "stage": stage,
        "num_rows_used": int(used_rows),
        "paired_positions": int(paired_positions),
        "skipped_missing": int(skipped_missing),
        "skipped_len_mismatch": int(skipped_len_mismatch),
        "count_threshold": int(count_threshold),
        "imbalance_threshold": float(imbalance_threshold),
        "lambda_threshold": float(lambda_threshold),
        "num_tokens_total": int(len(tokens_all)),
        "num_tokens_used": int(len(lambda_by_token)),
        "filtered_by_support": int(filtered_support),
        "filtered_by_balance": int(filtered_balance),
        "biased_positive": [(int(t), str(ph), float(lam)) for t, ph, lam in biased_pos],
        "biased_negative": [(int(t), str(ph), float(lam)) for t, ph, lam in biased_neg],
        "top_diff_tokens": [(int(t), float(c)) for t, c in diff_tokens],
        "biased_positive_top5": top_bias_pos,
        "biased_negative_top5": top_bias_neg,
        "top_diff_tokens_top5": top_diff_rows,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Method 3.

    Returns:
        Parsed namespace containing manifest path, labeling arguments, stage
        selection, Method 3 thresholds, and output prefix.
    """

    ap = argparse.ArgumentParser(description="Method 3: Attribute-conditioned token-phone representation")
    ap.add_argument("manifest_json")
    ap.add_argument("--stages", default="all", help="Comma-separated token set stages or 'all'")

    # labeling mode
    ap.add_argument("--label-field", default="")
    ap.add_argument("--pos-label", default="M")
    ap.add_argument("--neg-label", default="F")
    ap.add_argument("--score-field", default="")
    ap.add_argument("--qlo", type=float, default=0.2)
    ap.add_argument("--qhi", type=float, default=0.8)

    ap.add_argument("--lambda-threshold", type=float, default=0.015)
    ap.add_argument("--count-threshold", type=int, default=400)
    ap.add_argument("--imbalance-threshold", type=float, default=0.2)
    ap.add_argument("--top-k-phones", type=int, default=7)
    ap.add_argument("--top-n-diff", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-prefix", required=True)
    return ap.parse_args()


def main() -> None:
    """Run Method 3 end-to-end and write JSON/TXT outputs.

    The output JSON contains a stable schema with top-level keys:
    ``method``, ``manifest_json``, ``labeling``, ``stages``, ``per_stage``,
    and ``defaults``.
    """

    args = parse_args()
    set_seed(args.seed)

    rows = load_probe_rows(args.manifest_json)
    stages = resolve_stage_list(rows, args.stages)
    ensure_stage_exists(rows, stages)

    lab = resolve_labeling(
        rows=rows,
        label_field=args.label_field,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
        score_field=args.score_field,
        qlo=args.qlo,
        qhi=args.qhi,
    )

    per_stage: Dict[str, dict] = {}
    for stage in stages:
        per_stage[stage] = _analyze_stage(
            rows=rows,
            labels_by_id=lab.labels_by_id,
            stage=stage,
            lambda_threshold=args.lambda_threshold,
            count_threshold=args.count_threshold,
            imbalance_threshold=args.imbalance_threshold,
            top_k_phones=args.top_k_phones,
            top_n_diff=args.top_n_diff,
        )

    out = {
        "method": "attribute_conditioned_representation",
        "manifest_json": str(args.manifest_json),
        "labeling": {
            "mode": lab.mode,
            "positive_name": lab.positive_name,
            "negative_name": lab.negative_name,
            "meta": lab.meta,
        },
        "stages": stages,
        "per_stage": per_stage,
        "defaults": {
            "lambda_threshold": float(args.lambda_threshold),
            "count_threshold": int(args.count_threshold),
            "imbalance_threshold": float(args.imbalance_threshold),
        },
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(str(prefix) + ".json")
    out_txt = Path(str(prefix) + ".txt")
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("METHOD 3: ATTRIBUTE-CONDITIONED REPRESENTATION")
    lines.append("=" * 80)
    lines.append(f"Manifest: {args.manifest_json}")
    lines.append(f"Stages: {stages}")
    lines.append(f"Labeling: mode={lab.mode} pos={lab.positive_name} neg={lab.negative_name}")
    lines.append("")

    for s in stages:
        r = per_stage.get(s, {})
        if not r.get("ok"):
            lines.append(f"Stage {s}: error={r.get('error', 'no_result')}")
            continue
        lines.append(
            f"Stage {s}: used_rows={r['num_rows_used']} paired_positions={r['paired_positions']} "
            f"tokens_used={r['num_tokens_used']}"
        )
        top_pos = [int(x[0]) for x in r.get("biased_positive", [])[:5]]
        top_neg = [int(x[0]) for x in r.get("biased_negative", [])[:5]]
        lines.append(f"  Top5 +biased: {top_pos}")
        lines.append(f"  Top5 -biased: {top_neg}")
        top_diff = [int(x[0]) for x in r.get("top_diff_tokens", [])[:5]]
        lines.append(f"  Top5 differing-by-cosine: {top_diff}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
