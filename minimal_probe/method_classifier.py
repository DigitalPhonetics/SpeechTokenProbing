#!/usr/bin/env python3
"""CLI for Method 2: token-identity classifiers.

This script implements the paper's second probe method. It trains logistic
regression classifiers on BOW/SHARE/SET token features, reports accuracy and
balanced accuracy, exposes top tokens for both polarity directions, and computes
cross-stage coefficient similarity.

Assumptions:
    - Input rows follow the canonical manifest schema.
    - Labels are binary after ``resolve_labeling``.
    - Default split policy is shared speaker-disjoint splitting across stages.
    - Train-only token filtering uses a relative frequency threshold and keeps
      tokens seen in both classes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize

from .io import ensure_stage_exists, load_probe_rows, resolve_stage_list, resolve_stage_vocab_ids
from .labeling import resolve_labeling
from .stats import pairwise_cosine_and_mad, set_seed, split_speaker_disjoint


def _prepare_stage_rows(rows, labels_by_id: Dict[str, int], stage: str, allowed_row_ids: Optional[Set[str]] = None):
    """Collect usable rows and serialized samples for one stage."""

    usable = [
        r
        for r in rows
        if r.row_id in labels_by_id
        and stage in r.token_sets
        and r.token_sets[stage]
        and (allowed_row_ids is None or r.row_id in allowed_row_ids)
    ]
    y = [int(labels_by_id[r.row_id]) for r in usable]
    speakers = [r.speaker for r in usable]
    samples = [" ".join(str(t) for t in r.token_sets[stage]) for r in usable]
    return usable, samples, y, speakers


def _token_counts(samples: List[str], y: List[int]) -> Dict[int, Counter]:
    """Count token occurrences per class from serialized token samples."""

    pos = Counter()
    neg = Counter()
    for s, yy in zip(samples, y):
        toks = [int(x) for x in s.split()] if s else []
        if int(yy) == 1:
            pos.update(toks)
        else:
            neg.update(toks)
    return {1: pos, 0: neg}


def _train_token_filter(train_samples: List[str], train_y: List[int], min_token_rel_freq: float):
    """Build train-only vocabulary filter used by all feature modes."""

    ctrs = _token_counts(train_samples, train_y)
    pos, neg = ctrs[1], ctrs[0]
    all_ctr = pos + neg
    total = int(sum(all_ctr.values()))
    thr = max(1, int(np.ceil(float(min_token_rel_freq) * float(total))))
    keep = {
        int(t)
        for t, c in all_ctr.items()
        if int(c) >= int(thr) and int(t) in pos and int(t) in neg
    }
    meta = {
        "min_token_rel_freq": float(min_token_rel_freq),
        "total_train_token_occurrences": int(total),
        "effective_threshold": int(thr),
        "kept_vocab_size": int(len(keep)),
    }
    return keep, meta


def _apply_filter(samples: List[str], keep: set[int]) -> List[str]:
    """Filter token samples to a retained vocabulary."""

    out = []
    for s in samples:
        if not s:
            out.append("")
            continue
        out.append(" ".join(tok for tok in s.split() if int(tok) in keep))
    return out


def _fit_one_mode(train_samples, test_samples, y_train, y_test, mode: str, class_weight: Optional[str]):
    """Fit one logistic-regression feature mode and collect diagnostics.

    The implementation fits one model only (H=1 vs L=0). The inverse polarity
    outputs are derived by sign inversion and label inversion.
    """

    vec = CountVectorizer(binary=(mode == "set"), token_pattern=r"(?u)\b\S+\b")
    try:
        xtr = vec.fit_transform(train_samples)
        xte = vec.transform(test_samples)
    except ValueError:
        return {"ok": False, "error": "empty_vocab_after_filter"}

    if xtr.shape[1] == 0:
        return {"ok": False, "error": "empty_vocab_after_filter"}

    if mode == "share":
        xtr = normalize(xtr, norm="l1", axis=1)
        xte = normalize(xte, norm="l1", axis=1)

    cw = None if class_weight in (None, "none") else str(class_weight)
    clf_h = LogisticRegression(max_iter=1000, class_weight=cw)
    clf_h.fit(xtr, y_train)
    pred_h = clf_h.predict(xte)

    y_test_l = [1 - int(y) for y in y_test]
    pred_l = [1 - int(y) for y in pred_h]

    vocab = vec.get_feature_names_out()
    w_h = {int(tok): float(w) for tok, w in zip(vocab, clf_h.coef_[0])}
    w_l = {int(tok): float(-w) for tok, w in zip(vocab, clf_h.coef_[0])}

    top_h = sorted(w_h.items(), key=lambda x: x[1], reverse=True)[:30]
    top_l = sorted(w_l.items(), key=lambda x: x[1], reverse=True)[:30]

    return {
        "ok": True,
        "mode": mode,
        "class_weight": "none" if cw is None else str(cw),
        "metrics_h_vs_l": {
            "accuracy": float(accuracy_score(y_test, pred_h)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_h)),
            "confusion_matrix": confusion_matrix(y_test, pred_h).tolist(),
            "classification_report": classification_report(y_test, pred_h, digits=3, zero_division=0),
        },
        "metrics_l_vs_h": {
            "accuracy": float(accuracy_score(y_test_l, pred_l)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_l, pred_l)),
            "confusion_matrix": confusion_matrix(y_test_l, pred_l).tolist(),
            "classification_report": classification_report(y_test_l, pred_l, digits=3, zero_division=0),
        },
        "top_tokens_h_model": [{"token": int(t), "weight": float(w)} for t, w in top_h],
        "top_tokens_l_model": [{"token": int(t), "weight": float(w)} for t, w in top_l],
        "weights_h_model": w_h,
        "weights_l_model": w_l,
    }


def _fit_stage_with_fixed_split(
    usable,
    samples: Sequence[str],
    y: Sequence[int],
    tr_idx: Sequence[int],
    te_idx: Sequence[int],
    split_meta: dict,
    min_token_rel_freq: float,
    class_weight: str,
):
    """Run all classifier modes for a stage using a precomputed split."""

    train_samples_raw = [samples[int(i)] for i in tr_idx]
    test_samples_raw = [samples[int(i)] for i in te_idx]
    y_train = [int(y[int(i)]) for i in tr_idx]
    y_test = [int(y[int(i)]) for i in te_idx]

    keep, filter_meta = _train_token_filter(train_samples_raw, y_train, min_token_rel_freq=min_token_rel_freq)
    train_samples = _apply_filter(train_samples_raw, keep)
    test_samples = _apply_filter(test_samples_raw, keep)

    chance = 0.5
    maj = float(max(sum(y_test), len(y_test) - sum(y_test)) / len(y_test)) if y_test else None

    modes = {}
    for mode in ("bow", "share", "set"):
        rr = _fit_one_mode(train_samples, test_samples, y_train, y_test, mode, class_weight=class_weight)
        if rr.get("ok"):
            rr["chance_baseline"] = float(chance)
            rr["majority_baseline"] = float(maj) if maj is not None else None
        modes[mode] = rr

    any_mode_ok = any(bool(rr.get("ok")) for rr in modes.values())
    if not any_mode_ok:
        return {
            "ok": False,
            "error": "no_classifier_mode_succeeded",
            "num_usable_rows": int(len(usable)),
            "split": split_meta,
            "token_filter": filter_meta,
            "modes": modes,
        }

    return {
        "ok": True,
        "num_usable_rows": int(len(usable)),
        "split": split_meta,
        "token_filter": filter_meta,
        "modes": modes,
    }


def _summarize_values(values: List[float]) -> dict:
    """Summarize a list with mean/std and percentile CI."""

    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
    }


def _aggregate_mode_runs(mode_runs: List[dict]) -> dict:
    """Aggregate metric distributions across repeated splits for one mode."""

    if not mode_runs:
        return {"ok": False, "error": "no_successful_mode_runs"}

    h_bal = [float(r["metrics_h_vs_l"]["balanced_accuracy"]) for r in mode_runs]
    h_acc = [float(r["metrics_h_vs_l"]["accuracy"]) for r in mode_runs]
    l_bal = [float(r["metrics_l_vs_h"]["balanced_accuracy"]) for r in mode_runs]
    l_acc = [float(r["metrics_l_vs_h"]["accuracy"]) for r in mode_runs]
    maj = [float(r["majority_baseline"]) for r in mode_runs if r.get("majority_baseline") is not None]
    chance = [float(r["chance_baseline"]) for r in mode_runs if r.get("chance_baseline") is not None]

    return {
        "ok": True,
        "num_runs": int(len(mode_runs)),
        "metrics_h_vs_l": {
            "balanced_accuracy": _summarize_values(h_bal),
            "accuracy": _summarize_values(h_acc),
        },
        "metrics_l_vs_h": {
            "balanced_accuracy": _summarize_values(l_bal),
            "accuracy": _summarize_values(l_acc),
        },
        "majority_baseline": _summarize_values(maj),
        "chance_baseline": _summarize_values(chance),
    }


def _aggregate_stage_splits(stage: str, split_runs: List[dict], num_splits_requested: int) -> dict:
    """Aggregate stage-level results across repeated splits."""

    successful = [r for r in split_runs if r.get("ok")]
    failed = [r for r in split_runs if not r.get("ok")]

    if not successful:
        err = failed[0].get("error", "no_successful_splits") if failed else "no_successful_splits"
        return {
            "ok": False,
            "stage": stage,
            "error": err,
            "num_splits_requested": int(num_splits_requested),
            "num_splits_succeeded": 0,
            "num_splits_failed": int(len(failed)),
            "split_series": split_runs,
        }

    ref = successful[0]
    out = {
        "ok": True,
        "stage": stage,
        "num_usable_rows": int(ref.get("num_usable_rows", 0)),
        "split": ref.get("split", {}),
        "token_filter": ref.get("token_filter", {}),
        "modes": ref.get("modes", {}),
        "num_splits_requested": int(num_splits_requested),
        "num_splits_succeeded": int(len(successful)),
        "num_splits_failed": int(len(failed)),
        "split_series": split_runs,
    }

    aggregate_modes: Dict[str, dict] = {}
    for mode in ("bow", "share", "set"):
        mode_runs = [r["modes"][mode] for r in successful if r.get("modes", {}).get(mode, {}).get("ok")]
        aggregate_modes[mode] = _aggregate_mode_runs(mode_runs)
    out["aggregate_modes"] = aggregate_modes
    return out


def _shared_rows_for_stages(rows, labels_by_id: Dict[str, int], stages: Sequence[str]):
    """Return rows usable in all requested stages."""

    return [
        r
        for r in rows
        if r.row_id in labels_by_id
        and all(stage in r.token_sets and r.token_sets[stage] for stage in stages)
    ]


def _collect_cross_stage_consistency(stage_split_runs: Dict[str, List[dict]], num_splits: int) -> Dict[str, List[dict]]:
    """Compute cross-stage SHARE consistency and summarize across splits."""

    share_h_series: List[dict] = []
    share_l_series: List[dict] = []

    for split_idx in range(int(num_splits)):
        share_h: Dict[str, Dict[int, float]] = {}
        share_l: Dict[str, Dict[int, float]] = {}
        for stage, runs in stage_split_runs.items():
            if split_idx >= len(runs):
                continue
            res = runs[split_idx]
            if not res.get("ok"):
                continue
            share = res.get("modes", {}).get("share", {})
            if share.get("ok"):
                share_h[stage] = {int(k): float(v) for k, v in share.get("weights_h_model", {}).items()}
                share_l[stage] = {int(k): float(v) for k, v in share.get("weights_l_model", {}).items()}

        for row in pairwise_cosine_and_mad(share_h):
            rr = dict(row)
            rr["split_index"] = int(split_idx)
            share_h_series.append(rr)
        for row in pairwise_cosine_and_mad(share_l):
            rr = dict(row)
            rr["split_index"] = int(split_idx)
            share_l_series.append(rr)

    def _aggregate(rows: List[dict]) -> tuple[List[dict], List[dict]]:
        grouped: Dict[tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: {"cos": [], "mad": []})
        for row in rows:
            key = (str(row["stage_a"]), str(row["stage_b"]))
            grouped[key]["cos"].append(float(row["cosine"]))
            grouped[key]["mad"].append(float(row["mean_abs_delta"]))

        compact: List[dict] = []
        rich: List[dict] = []
        for (a, b), vals in sorted(grouped.items()):
            cos_summary = _summarize_values(vals["cos"])
            mad_summary = _summarize_values(vals["mad"])
            compact.append(
                {
                    "stage_a": a,
                    "stage_b": b,
                    "cosine": float(cos_summary["mean"]) if cos_summary["mean"] is not None else float("nan"),
                    "mean_abs_delta": float(mad_summary["mean"]) if mad_summary["mean"] is not None else float("nan"),
                }
            )
            rich.append(
                {
                    "stage_a": a,
                    "stage_b": b,
                    "cosine_summary": cos_summary,
                    "mean_abs_delta_summary": mad_summary,
                }
            )
        return compact, rich

    share_h_model, share_h_model_aggregate = _aggregate(share_h_series)
    share_l_model, share_l_model_aggregate = _aggregate(share_l_series)

    return {
        "share_h_model": share_h_model,
        "share_l_model": share_l_model,
        "share_h_model_series": share_h_series,
        "share_l_model_series": share_l_series,
        "share_h_model_aggregate": share_h_model_aggregate,
        "share_l_model_aggregate": share_l_model_aggregate,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Method 2."""

    ap = argparse.ArgumentParser(description="Method 2: Token-based classifiers")
    ap.add_argument("manifest_json")
    ap.add_argument("--stages", default="all", help="Comma-separated token set stages or 'all'")

    # labeling mode
    ap.add_argument("--label-field", default="")
    ap.add_argument("--pos-label", default="M")
    ap.add_argument("--neg-label", default="F")
    ap.add_argument("--score-field", default="")
    ap.add_argument("--qlo", type=float, default=0.2)
    ap.add_argument("--qhi", type=float, default=0.8)

    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--split-max-tries", type=int, default=100)
    ap.add_argument(
        "--min-token-rel-freq",
        type=float,
        default=0.00002,
        help="Train-only token filter threshold as fraction of total train token occurrences (default 0.002%%)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--per-stage-split",
        action="store_true",
        help="Use stage-local speaker-disjoint splits (legacy behavior).",
    )
    ap.add_argument(
        "--shared-split-across-stages",
        action="store_true",
        help="Deprecated compatibility flag; shared split is now the default.",
    )
    ap.add_argument(
        "--num-splits",
        type=int,
        default=10,
        help="Number of repeated split/evaluation runs for uncertainty estimates.",
    )
    ap.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting passed to LogisticRegression.",
    )
    ap.add_argument(
        "--stage-vocab-ids",
        default="",
        help="Required for multi-stage comparisons. Format: stage:id,stage:id,... "
        "with identical ids across compared stages.",
    )
    ap.add_argument(
        "--allow-utterance-fallback",
        action="store_true",
        help="Allow utterance-level stratified fallback if no valid speaker-disjoint split is found",
    )
    ap.add_argument("--output-prefix", required=True)
    return ap.parse_args()


def main() -> None:
    """Run Method 2 end-to-end and write JSON/TXT outputs."""

    args = parse_args()
    if int(args.num_splits) < 1:
        raise ValueError("--num-splits must be >= 1")
    if bool(args.per_stage_split) and bool(args.shared_split_across_stages):
        raise ValueError("--per-stage-split conflicts with --shared-split-across-stages")

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

    use_shared_split = not bool(args.per_stage_split)
    split_policy = "shared_speaker_split" if use_shared_split else "per_stage_split"

    stage_split_runs: Dict[str, List[dict]] = {stage: [] for stage in stages}

    if use_shared_split:
        shared_rows = _shared_rows_for_stages(rows=rows, labels_by_id=lab.labels_by_id, stages=stages)
        shared_y = [int(lab.labels_by_id[r.row_id]) for r in shared_rows]
        shared_speakers = [r.speaker for r in shared_rows]

        if len(shared_rows) < 20 or len(set(shared_y)) < 2:
            for stage in stages:
                stage_split_runs[stage] = [
                    {
                        "ok": False,
                        "stage": stage,
                        "error": "too_few_samples_for_shared_split",
                        "num_usable_rows": int(len(shared_rows)),
                        "split_index": int(i),
                        "split_seed": int(args.seed + i),
                    }
                    for i in range(int(args.num_splits))
                ]
        else:
            shared_ids = {r.row_id for r in shared_rows}
            stage_prepared: Dict[str, dict] = {}
            for stage in stages:
                usable, samples, y, _speakers = _prepare_stage_rows(
                    rows=rows,
                    labels_by_id=lab.labels_by_id,
                    stage=stage,
                    allowed_row_ids=shared_ids,
                )
                stage_prepared[stage] = {
                    "usable": usable,
                    "samples": samples,
                    "y": y,
                }
                if len(usable) != len(shared_rows):
                    raise ValueError(
                        f"Shared split preparation mismatch for stage {stage!r}: expected {len(shared_rows)} rows, got {len(usable)}"
                    )

            for split_idx in range(int(args.num_splits)):
                split_seed = int(args.seed + split_idx)
                try:
                    tr_idx, te_idx, split_meta = split_speaker_disjoint(
                        y=shared_y,
                        speakers=shared_speakers,
                        test_size=args.test_size,
                        seed=split_seed,
                        max_tries=args.split_max_tries,
                        allow_utterance_fallback=bool(args.allow_utterance_fallback),
                    )
                except ValueError as e:
                    for stage in stages:
                        stage_split_runs[stage].append(
                            {
                                "ok": False,
                                "stage": stage,
                                "error": f"speaker_split_failed:{str(e)}",
                                "num_usable_rows": int(len(shared_rows)),
                                "split_index": int(split_idx),
                                "split_seed": split_seed,
                            }
                        )
                    continue

                for stage in stages:
                    prep = stage_prepared[stage]
                    rr = _fit_stage_with_fixed_split(
                        usable=prep["usable"],
                        samples=prep["samples"],
                        y=prep["y"],
                        tr_idx=tr_idx,
                        te_idx=te_idx,
                        split_meta=split_meta,
                        min_token_rel_freq=args.min_token_rel_freq,
                        class_weight=str(args.class_weight),
                    )
                    rr["stage"] = stage
                    rr["split_index"] = int(split_idx)
                    rr["split_seed"] = split_seed
                    stage_split_runs[stage].append(rr)
    else:
        for stage_index, stage in enumerate(stages):
            usable, samples, y, speakers = _prepare_stage_rows(rows, lab.labels_by_id, stage)
            for split_idx in range(int(args.num_splits)):
                split_seed = int(args.seed + split_idx + stage_index * int(args.num_splits))
                if len(usable) < 20 or len(set(y)) < 2:
                    stage_split_runs[stage].append(
                        {
                            "ok": False,
                            "stage": stage,
                            "error": "too_few_samples",
                            "num_usable_rows": int(len(usable)),
                            "split_index": int(split_idx),
                            "split_seed": split_seed,
                        }
                    )
                    continue

                try:
                    tr_idx, te_idx, split_meta = split_speaker_disjoint(
                        y=y,
                        speakers=speakers,
                        test_size=args.test_size,
                        seed=split_seed,
                        max_tries=args.split_max_tries,
                        allow_utterance_fallback=bool(args.allow_utterance_fallback),
                    )
                except ValueError as e:
                    stage_split_runs[stage].append(
                        {
                            "ok": False,
                            "stage": stage,
                            "error": f"speaker_split_failed:{str(e)}",
                            "num_usable_rows": int(len(usable)),
                            "split_index": int(split_idx),
                            "split_seed": split_seed,
                        }
                    )
                    continue

                rr = _fit_stage_with_fixed_split(
                    usable=usable,
                    samples=samples,
                    y=y,
                    tr_idx=tr_idx,
                    te_idx=te_idx,
                    split_meta=split_meta,
                    min_token_rel_freq=args.min_token_rel_freq,
                    class_weight=str(args.class_weight),
                )
                rr["stage"] = stage
                rr["split_index"] = int(split_idx)
                rr["split_seed"] = split_seed
                stage_split_runs[stage].append(rr)

    per_stage = {
        stage: _aggregate_stage_splits(stage, split_runs=stage_split_runs[stage], num_splits_requested=int(args.num_splits))
        for stage in stages
    }

    cross = _collect_cross_stage_consistency(stage_split_runs=stage_split_runs, num_splits=int(args.num_splits))

    out = {
        "method": "token_classifiers",
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
        "cross_stage_consistency": cross,
        "defaults": {
            "test_size": float(args.test_size),
            "split_max_tries": int(args.split_max_tries),
            "min_token_rel_freq": float(args.min_token_rel_freq),
            "split_policy": split_policy,
            "allow_utterance_fallback": bool(args.allow_utterance_fallback),
            "num_splits": int(args.num_splits),
            "class_weight": str(args.class_weight),
        },
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(str(prefix) + ".json")
    out_txt = Path(str(prefix) + ".txt")
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("METHOD 2: TOKEN-BASED CLASSIFIERS")
    lines.append("=" * 80)
    lines.append(f"Manifest: {args.manifest_json}")
    lines.append(
        f"Split policy: {split_policy} | num_splits={int(args.num_splits)} "
        f"| class_weight={args.class_weight} | allow_utterance_fallback={bool(args.allow_utterance_fallback)}"
    )
    lines.append(f"Stages: {stages}")
    lines.append(f"Labeling: mode={lab.mode} pos={lab.positive_name} neg={lab.negative_name}")
    if resolved_stage_vocab_ids:
        lines.append(f"Shared vocab id: {next(iter(set(resolved_stage_vocab_ids.values())))}")
    lines.append("")
    lines.append(f"{'Stage':<16} {'BOW bal':>10} {'SHARE bal':>10} {'SET bal':>10} {'Maj':>8} {'Splits':>8} {'Top5 H':<24} {'Top5 L'}")
    for s in stages:
        r = per_stage.get(s, {})
        if not r.get("ok"):
            lines.append(f"{s:<16} error={r.get('error', 'no_result')}")
            continue

        def bal_mean(mode: str) -> float:
            return float(r["aggregate_modes"][mode]["metrics_h_vs_l"]["balanced_accuracy"]["mean"])

        bow = bal_mean("bow")
        share = bal_mean("share")
        sett = bal_mean("set")
        maj = r["aggregate_modes"]["bow"].get("majority_baseline", {}).get("mean", float("nan"))
        top_h = [int(x["token"]) for x in r["modes"]["share"]["top_tokens_h_model"][:5]] if r["modes"].get("share", {}).get("ok") else []
        top_l = [int(x["token"]) for x in r["modes"]["share"]["top_tokens_l_model"][:5]] if r["modes"].get("share", {}).get("ok") else []
        lines.append(
            f"{s:<16} {bow:>10.3f} {share:>10.3f} {sett:>10.3f} {float(maj):>8.3f} "
            f"{int(r.get('num_splits_succeeded', 0)):>8}/{int(r.get('num_splits_requested', 0)):<3} "
            f"{str(top_h):<24} {str(top_l)}"
        )

    lines.append("")
    lines.append("Cross-stage SHARE consistency (H-model, mean across splits)")
    for row in cross.get("share_h_model", []):
        lines.append(
            f"  {row['stage_a']} vs {row['stage_b']}: cosine={float(row['cosine']):.4f} mean|Δ|={float(row['mean_abs_delta']):.4f}"
        )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
