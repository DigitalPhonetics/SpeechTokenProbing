#!/usr/bin/env python3
"""CLI for Method 1: distributional divergence over token identities.

This script implements the paper's first probe method. For each requested token
stage it compares class-conditional token distributions using JSD, computes a
shuffle baseline, and reports top positive/negative token differences.

Assumptions:
    - Input rows are provided in the canonical manifest schema.
    - Labels are binary after ``resolve_labeling``.
    - Default group policy uses full empirical class groups (paper default).

This module is both a library-style implementation (helper functions) and a CLI
entrypoint via :func:`main`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .io import ensure_stage_exists, load_probe_rows, resolve_stage_list, resolve_stage_vocab_ids
from .labeling import resolve_labeling
from .stats import jsd_from_counts, sample_balanced_groups, set_seed, token_distribution


def _rows_by_id(rows):
    """Index rows by row id.

    Args:
        rows: Iterable of probe rows.

    Returns:
        Dictionary mapping ``row_id`` to row object.
    """

    return {r.row_id: r for r in rows}


def _analyze_stage(
    rows,
    labels_by_id: Dict[str, int],
    stage: str,
    min_token_freq: int,
    num_random_trials: int,
    seed: int,
    pos_name: str,
    neg_name: str,
    group_policy: str,
):
    """Run divergence analysis for one token stage.

    Args:
        rows: Loaded probe rows.
        labels_by_id: Binary labels keyed by row id.
        stage: Token stage to analyze.
        min_token_freq: Minimum corpus count required for divergence vocabulary.
        num_random_trials: Number of shuffle baseline trials.
        seed: Random seed for balancing and shuffling.
        pos_name: Human-readable positive label name.
        neg_name: Human-readable negative label name.
        group_policy: One of ``"full_groups"`` or ``"balanced_downsample"``.

    Returns:
        Result dictionary with at least ``ok`` and either:
            - success payload keys such as ``jsd``, ``shuffle_mean``,
              ``top_positive``, ``top_negative``, ``per_token_diffs``,
            - or ``error`` when the stage cannot be analyzed.
    """

    usable = [r for r in rows if r.row_id in labels_by_id and stage in r.token_sets and r.token_sets[stage]]
    pos_ids_all = [r.row_id for r in usable if labels_by_id[r.row_id] == 1]
    neg_ids_all = [r.row_id for r in usable if labels_by_id[r.row_id] == 0]

    if group_policy == "balanced_downsample":
        pos_ids, neg_ids = sample_balanced_groups(pos_ids_all, neg_ids_all, seed=seed)
    elif group_policy == "full_groups":
        pos_ids, neg_ids = list(pos_ids_all), list(neg_ids_all)
    else:
        return {"ok": False, "error": f"unknown_group_policy:{group_policy}"}

    if not pos_ids or not neg_ids:
        return {"ok": False, "error": "empty_groups_after_balancing"}

    by_id = _rows_by_id(usable)
    pos_seqs = [by_id[rid].token_sets[stage] for rid in pos_ids]
    neg_seqs = [by_id[rid].token_sets[stage] for rid in neg_ids]

    all_d, all_ctr = token_distribution([r.token_sets[stage] for r in usable])
    vocab = sorted([int(t) for t, c in all_ctr.items() if int(c) >= int(min_token_freq)])
    if not vocab:
        return {"ok": False, "error": "no_vocab_after_threshold"}

    d_pos, _ = token_distribution(pos_seqs)
    d_neg, _ = token_distribution(neg_seqs)

    v_pos = np.array([d_pos.get(t, 0.0) for t in vocab], dtype=np.float64)
    v_neg = np.array([d_neg.get(t, 0.0) for t in vocab], dtype=np.float64)
    score = jsd_from_counts(v_pos, v_neg)

    combined = list(pos_ids) + list(neg_ids)
    rng = np.random.default_rng(seed)
    rand_vals: List[float] = []
    for _ in range(int(num_random_trials)):
        perm = rng.permutation(len(combined))
        shuffled = [combined[int(i)] for i in perm]
        s_pos = shuffled[: len(pos_ids)]
        s_neg = shuffled[len(pos_ids) : len(pos_ids) + len(neg_ids)]
        sd_pos, _ = token_distribution([by_id[rid].token_sets[stage] for rid in s_pos])
        sd_neg, _ = token_distribution([by_id[rid].token_sets[stage] for rid in s_neg])
        sv_pos = np.array([sd_pos.get(t, 0.0) for t in vocab], dtype=np.float64)
        sv_neg = np.array([sd_neg.get(t, 0.0) for t in vocab], dtype=np.float64)
        rv = jsd_from_counts(sv_pos, sv_neg)
        if not np.isnan(rv):
            rand_vals.append(float(rv))

    diffs: List[Tuple[int, float, float, float]] = []
    for t in vocab:
        p = float(d_pos.get(t, 0.0))
        n = float(d_neg.get(t, 0.0))
        d = p - n
        if p + n > 0:
            diffs.append((int(t), p, n, d))
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    return {
        "ok": True,
        "stage": stage,
        "groups": {"positive": pos_name, "negative": neg_name},
        "num_usable_rows": int(len(usable)),
        "num_pos": int(len(pos_ids)),
        "num_neg": int(len(neg_ids)),
        "group_policy": str(group_policy),
        "vocab_size": int(len(vocab)),
        "min_token_freq": int(min_token_freq),
        "jsd": float(score),
        "shuffle_mean": float(np.mean(rand_vals)) if rand_vals else 0.0,
        "shuffle_std": float(np.std(rand_vals)) if rand_vals else 0.0,
        "top_positive": [x for x in diffs if x[3] > 0][:30],
        "top_negative": [x for x in diffs if x[3] < 0][:30],
        "per_token_diffs": diffs,
    }


def _cross_stage_jsd(stage_results: Dict[str, dict]) -> List[dict]:
    """Compare class-conditional distributions across stages.

    Args:
        stage_results: Per-stage outputs from :func:`_analyze_stage`.

    Returns:
        Pairwise rows comparing positive and negative class distributions across
        stages with ``jsd`` and ``mean_abs_delta``.
    """

    valid = {s: r for s, r in stage_results.items() if isinstance(r, dict) and r.get("ok")}
    out: List[dict] = []
    stages = sorted(valid.keys())
    for i in range(len(stages)):
        for j in range(i + 1, len(stages)):
            sa, sb = stages[i], stages[j]
            ra, rb = valid[sa], valid[sb]
            vocab = sorted(
                set(int(t) for t, *_ in ra.get("per_token_diffs", []))
                | set(int(t) for t, *_ in rb.get("per_token_diffs", []))
            )
            if not vocab:
                continue
            a_pos = {int(t): float(p) for t, p, _, _ in ra.get("per_token_diffs", [])}
            b_pos = {int(t): float(p) for t, p, _, _ in rb.get("per_token_diffs", [])}
            a_neg = {int(t): float(n) for t, _, n, _ in ra.get("per_token_diffs", [])}
            b_neg = {int(t): float(n) for t, _, n, _ in rb.get("per_token_diffs", [])}
            va_pos = np.array([a_pos.get(t, 0.0) for t in vocab], dtype=np.float64)
            vb_pos = np.array([b_pos.get(t, 0.0) for t in vocab], dtype=np.float64)
            va_neg = np.array([a_neg.get(t, 0.0) for t in vocab], dtype=np.float64)
            vb_neg = np.array([b_neg.get(t, 0.0) for t in vocab], dtype=np.float64)
            out.append(
                {
                    "stage_a": sa,
                    "stage_b": sb,
                    "bin": "positive",
                    "jsd": jsd_from_counts(va_pos, vb_pos) if va_pos.sum() > 0 and vb_pos.sum() > 0 else float("nan"),
                    "mean_abs_delta": float(np.mean(np.abs(va_pos - vb_pos))),
                }
            )
            out.append(
                {
                    "stage_a": sa,
                    "stage_b": sb,
                    "bin": "negative",
                    "jsd": jsd_from_counts(va_neg, vb_neg) if va_neg.sum() > 0 and vb_neg.sum() > 0 else float("nan"),
                    "mean_abs_delta": float(np.mean(np.abs(va_neg - vb_neg))),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Method 1.

    Returns:
        Parsed namespace containing manifest path, labeling arguments, stage
        selection, divergence hyperparameters, and output prefix.
    """

    ap = argparse.ArgumentParser(description="Method 1: Distributional divergence analysis")
    ap.add_argument("manifest_json")
    ap.add_argument("--stages", default="all", help="Comma-separated token set stages or 'all'")

    # labeling mode (choose one)
    ap.add_argument("--label-field", default="")
    ap.add_argument("--pos-label", default="M")
    ap.add_argument("--neg-label", default="F")
    ap.add_argument("--score-field", default="")
    ap.add_argument("--qlo", type=float, default=0.2)
    ap.add_argument("--qhi", type=float, default=0.8)

    ap.add_argument("--min-token-freq", type=int, default=50)
    ap.add_argument("--num-random-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stage-vocab-ids",
        default="",
        help="Required for multi-stage comparisons. Format: stage:id,stage:id,... "
        "with identical ids across compared stages.",
    )
    ap.add_argument(
        "--balanced-downsample",
        action="store_true",
        help="Use class-balanced downsampling before divergence (non-paper alternative)",
    )
    ap.add_argument("--output-prefix", required=True)
    return ap.parse_args()


def main() -> None:
    """Run Method 1 end-to-end and write JSON/TXT outputs.

    The output JSON contains a stable schema with top-level keys:
    ``method``, ``manifest_json``, ``labeling``, ``stages``, ``per_stage``,
    ``cross_stage_jsd``, and ``defaults``.
    """

    args = parse_args()
    set_seed(args.seed)

    rows = load_probe_rows(args.manifest_json)
    stages = resolve_stage_list(rows, args.stages)
    ensure_stage_exists(rows, stages)
    resolved_stage_vocab_ids = resolve_stage_vocab_ids(args.stage_vocab_ids, stages=stages)

    lab = resolve_labeling(
        rows=rows,
        label_field=args.label_field,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
        score_field=args.score_field,
        qlo=args.qlo,
        qhi=args.qhi,
    )

    # Paper-default: full empirical groups.
    group_policy = "balanced_downsample" if bool(args.balanced_downsample) else "full_groups"

    per_stage: Dict[str, dict] = {}
    for i, stage in enumerate(stages):
        per_stage[stage] = _analyze_stage(
            rows=rows,
            labels_by_id=lab.labels_by_id,
            stage=stage,
            min_token_freq=args.min_token_freq,
            num_random_trials=args.num_random_trials,
            seed=args.seed + i,
            pos_name=lab.positive_name,
            neg_name=lab.negative_name,
            group_policy=group_policy,
        )

    cross = _cross_stage_jsd(per_stage)

    out = {
        "method": "distributional_divergence",
        "manifest_json": str(args.manifest_json),
        "stage_vocab_ids": resolved_stage_vocab_ids,
        "shared_vocab_id": next(iter(set(resolved_stage_vocab_ids.values())), None) if resolved_stage_vocab_ids else None,
        "labeling": {
            "mode": lab.mode,
            "positive_name": lab.positive_name,
            "negative_name": lab.negative_name,
            "meta": lab.meta,
        },
        "stages": stages,
        "per_stage": per_stage,
        "cross_stage_jsd": cross,
        "defaults": {
            "min_token_freq": int(args.min_token_freq),
            "num_random_trials": int(args.num_random_trials),
            "group_policy": group_policy,
        },
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(str(prefix) + ".json")
    out_txt = Path(str(prefix) + ".txt")
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("METHOD 1: DISTRIBUTIONAL DIVERGENCE")
    lines.append("=" * 80)
    lines.append(f"Manifest: {args.manifest_json}")
    lines.append(f"Group policy: {group_policy}")
    lines.append(f"Stages: {stages}")
    lines.append(f"Labeling: mode={lab.mode} pos={lab.positive_name} neg={lab.negative_name}")
    lines.append("")
    lines.append(f"{'Stage':<16} {'JSD':>10} {'Shuffle':>10} {'JSD/Shuffle':>12} {'Top5 +':<24} {'Top5 -'}")
    for s in stages:
        r = per_stage.get(s, {})
        if not r.get("ok"):
            lines.append(f"{s:<16} error={r.get('error', 'no_result')}")
            continue
        jsd = float(r.get("jsd", 0.0))
        sh = float(r.get("shuffle_mean", 0.0))
        ratio = jsd / sh if sh > 0 else float("nan")
        top_pos = [int(t) for t, *_ in r.get("top_positive", [])[:5]]
        top_neg = [int(t) for t, *_ in r.get("top_negative", [])[:5]]
        lines.append(f"{s:<16} {jsd:>10.4f} {sh:>10.4f} {ratio:>12.2f} {str(top_pos):<24} {str(top_neg)}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
