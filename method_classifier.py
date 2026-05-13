#!/usr/bin/env python3
"""CLI for Method 2: token-identity classifiers.

This script implements the paper's second probe method. It trains logistic
regression classifiers on BOW/SHARE/SET token features, reports accuracy and
balanced accuracy, exposes top tokens for both polarity directions, and computes
cross-stage coefficient similarity.

Assumptions:
    - Input rows follow the canonical manifest schema.
    - Labels are binary after ``resolve_labeling``.
    - Default split policy is per-stage speaker-disjoint 80/20 where possible.
    - Train-only token filtering uses a relative frequency threshold and keeps
      tokens seen in both classes.

This module is both a library-style implementation (helper functions) and a CLI
entrypoint via :func:`main`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import normalize

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimal_probe.io import ensure_stage_exists, load_probe_rows, resolve_stage_list
from minimal_probe.labeling import resolve_labeling
from minimal_probe.stats import pairwise_cosine_and_mad, set_seed, split_speaker_disjoint


def _prepare_stage_rows(rows, labels_by_id: Dict[str, int], stage: str, allowed_row_ids: Optional[Set[str]] = None):
    """Collect usable rows and serialized samples for one stage.

    Args:
        rows: Loaded probe rows.
        labels_by_id: Binary labels keyed by row id.
        stage: Token stage to extract.
        allowed_row_ids: Optional whitelist of row ids, used for shared-split
            runs.

    Returns:
        Tuple ``(usable_rows, samples, y, speakers)`` where ``samples`` are
        whitespace-delimited token-id strings.
    """

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
    """Count token occurrences per class from serialized token samples.

    Args:
        samples: Whitespace-delimited token strings.
        y: Binary labels aligned with ``samples``.

    Returns:
        Dictionary mapping class id to token Counter: ``{1: pos_counter,
        0: neg_counter}``.
    """

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
    """Build train-only vocabulary filter used by all feature modes.

    A token is retained only if:
        1) its training count meets the effective threshold, and
        2) it appears in both classes.

    Args:
        train_samples: Training token samples.
        train_y: Training binary labels.
        min_token_rel_freq: Relative threshold as a fraction of total training
            token occurrences.

    Returns:
        Tuple ``(keep_set, meta)`` where ``keep_set`` contains retained token
        ids and ``meta`` records threshold details.
    """

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
    """Filter token samples to a retained vocabulary.

    Args:
        samples: Whitespace-delimited token strings.
        keep: Retained token id set.

    Returns:
        Filtered samples with dropped tokens removed.
    """

    out = []
    for s in samples:
        if not s:
            out.append("")
            continue
        out.append(" ".join(tok for tok in s.split() if int(tok) in keep))
    return out


def _fit_one_mode(train_samples, test_samples, y_train, y_test, mode: str):
    """Fit one logistic-regression feature mode and collect diagnostics.

    Args:
        train_samples: Filtered training samples.
        test_samples: Filtered test samples.
        y_train: Binary training labels.
        y_test: Binary test labels.
        mode: One of ``"bow"``, ``"share"``, or ``"set"``.

    Returns:
        Result dictionary with ``ok`` plus either:
            - success fields: metrics for both polarity models, top tokens, and
              full weight maps,
            - or ``error`` when the vectorizer vocabulary is empty.
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

    # model A: H=1 vs L=0
    clf_h = LogisticRegression(max_iter=1000)
    clf_h.fit(xtr, y_train)
    pred_h = clf_h.predict(xte)

    # model B: L=1 vs H=0
    y_train_l = [1 - int(y) for y in y_train]
    y_test_l = [1 - int(y) for y in y_test]
    clf_l = LogisticRegression(max_iter=1000)
    clf_l.fit(xtr, y_train_l)
    pred_l = clf_l.predict(xte)

    vocab = vec.get_feature_names_out()
    w_h = {int(tok): float(w) for tok, w in zip(vocab, clf_h.coef_[0])}
    w_l = {int(tok): float(w) for tok, w in zip(vocab, clf_l.coef_[0])}

    top_h = sorted(w_h.items(), key=lambda x: x[1], reverse=True)[:30]
    top_l = sorted(w_l.items(), key=lambda x: x[1], reverse=True)[:30]

    return {
        "ok": True,
        "mode": mode,
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
):
    """Run all classifier modes for a stage using a precomputed split.

    Args:
        usable: Usable stage rows.
        samples: Serialized token samples aligned with ``usable``.
        y: Binary labels aligned with ``usable``.
        tr_idx: Train indices into ``usable``.
        te_idx: Test indices into ``usable``.
        split_meta: Metadata produced by split selection.
        min_token_rel_freq: Relative train-token frequency threshold.

    Returns:
        Stage result dictionary containing ``ok``, split metadata, token filter
        metadata, and per-mode outputs under ``modes``.
    """

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
        rr = _fit_one_mode(train_samples, test_samples, y_train, y_test, mode)
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


def _analyze_stage(
    rows,
    labels_by_id: Dict[str, int],
    stage: str,
    test_size: float,
    seed: int,
    split_max_tries: int,
    min_token_rel_freq: float,
    allow_utterance_fallback: bool,
):
    """Run classifier analysis for one stage with stage-local split.

    Args:
        rows: Loaded probe rows.
        labels_by_id: Binary labels keyed by row id.
        stage: Token stage to analyze.
        test_size: Test split fraction.
        seed: Base random seed.
        split_max_tries: Max attempts for speaker-disjoint split search.
        min_token_rel_freq: Relative train-token frequency threshold.
        allow_utterance_fallback: Whether utterance-level fallback split is
            allowed when speaker split fails.

    Returns:
        Stage result dictionary with ``ok`` and either success payload or an
        ``error`` code string.
    """

    usable, samples, y, speakers = _prepare_stage_rows(rows, labels_by_id, stage)
    if len(usable) < 20 or len(set(y)) < 2:
        return {"ok": False, "error": "too_few_samples", "num_usable_rows": int(len(usable))}

    try:
        tr_idx, te_idx, split_meta = split_speaker_disjoint(
            y=y,
            speakers=speakers,
            test_size=test_size,
            seed=seed,
            max_tries=split_max_tries,
            allow_utterance_fallback=allow_utterance_fallback,
        )
    except ValueError as e:
        return {
            "ok": False,
            "stage": stage,
            "error": f"speaker_split_failed:{str(e)}",
            "num_usable_rows": int(len(usable)),
        }

    out = _fit_stage_with_fixed_split(
        usable=usable,
        samples=samples,
        y=y,
        tr_idx=tr_idx,
        te_idx=te_idx,
        split_meta=split_meta,
        min_token_rel_freq=min_token_rel_freq,
    )
    out["stage"] = stage
    return out


def _cross_stage_consistency(stage_results: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Compute pairwise SHARE-weight consistency across successful stages.

    Args:
        stage_results: Per-stage classifier outputs.

    Returns:
        Dictionary with pairwise summary rows for both polarity models:
        ``share_h_model`` and ``share_l_model``.
    """

    share_h: Dict[str, Dict[int, float]] = {}
    share_l: Dict[str, Dict[int, float]] = {}
    for stage, res in stage_results.items():
        if not res.get("ok"):
            continue
        share = res.get("modes", {}).get("share", {})
        if share.get("ok"):
            share_h[stage] = {int(k): float(v) for k, v in share.get("weights_h_model", {}).items()}
            share_l[stage] = {int(k): float(v) for k, v in share.get("weights_l_model", {}).items()}

    return {
        "share_h_model": pairwise_cosine_and_mad(share_h),
        "share_l_model": pairwise_cosine_and_mad(share_l),
    }


def _shared_rows_for_stages(rows, labels_by_id: Dict[str, int], stages: Sequence[str]):
    """Return rows usable in all requested stages.

    Args:
        rows: Loaded probe rows.
        labels_by_id: Binary labels keyed by row id.
        stages: Stage names required simultaneously.

    Returns:
        Rows that are labeled and contain non-empty token sequences for every
        requested stage.
    """

    return [
        r
        for r in rows
        if r.row_id in labels_by_id
        and all(stage in r.token_sets and r.token_sets[stage] for stage in stages)
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Method 2.

    Returns:
        Parsed namespace containing manifest path, labeling arguments, split and
        filtering settings, stage selection, and output prefix.
    """

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
        "--shared-split-across-stages",
        action="store_true",
        help="Use one common speaker-disjoint split across requested stages (non-paper alternative)",
    )
    ap.add_argument(
        "--allow-utterance-fallback",
        action="store_true",
        help="Allow utterance-level stratified fallback if no valid speaker-disjoint split is found",
    )
    ap.add_argument("--output-prefix", required=True)
    return ap.parse_args()


def main() -> None:
    """Run Method 2 end-to-end and write JSON/TXT outputs.

    The output JSON contains a stable schema with top-level keys:
    ``method``, ``manifest_json``, ``labeling``, ``stages``, ``per_stage``,
    ``cross_stage_consistency``, and ``defaults``.
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

    split_policy = "shared_speaker_split" if bool(args.shared_split_across_stages) else "per_stage_split"

    per_stage: Dict[str, dict] = {}
    if bool(args.shared_split_across_stages):
        shared_rows = _shared_rows_for_stages(rows=rows, labels_by_id=lab.labels_by_id, stages=stages)
        if len(shared_rows) < 20:
            for stage in stages:
                per_stage[stage] = {
                    "ok": False,
                    "stage": stage,
                    "error": "too_few_samples_for_shared_split",
                    "num_usable_rows": int(len(shared_rows)),
                }
        else:
            shared_y = [int(lab.labels_by_id[r.row_id]) for r in shared_rows]
            shared_speakers = [r.speaker for r in shared_rows]
            try:
                tr_idx, te_idx, split_meta = split_speaker_disjoint(
                    y=shared_y,
                    speakers=shared_speakers,
                    test_size=args.test_size,
                    seed=args.seed,
                    max_tries=args.split_max_tries,
                    allow_utterance_fallback=bool(args.allow_utterance_fallback),
                )
            except ValueError as e:
                for stage in stages:
                    per_stage[stage] = {
                        "ok": False,
                        "stage": stage,
                        "error": f"speaker_split_failed:{str(e)}",
                        "num_usable_rows": int(len(shared_rows)),
                    }
            else:
                shared_ids = {r.row_id for r in shared_rows}
                for stage in stages:
                    usable, samples, y, _speakers = _prepare_stage_rows(
                        rows=rows,
                        labels_by_id=lab.labels_by_id,
                        stage=stage,
                        allowed_row_ids=shared_ids,
                    )
                    if len(usable) != len(shared_rows):
                        per_stage[stage] = {
                            "ok": False,
                            "stage": stage,
                            "error": "shared_split_stage_row_mismatch",
                            "num_usable_rows": int(len(usable)),
                        }
                        continue

                    rr = _fit_stage_with_fixed_split(
                        usable=usable,
                        samples=samples,
                        y=y,
                        tr_idx=tr_idx,
                        te_idx=te_idx,
                        split_meta=split_meta,
                        min_token_rel_freq=args.min_token_rel_freq,
                    )
                    rr["stage"] = stage
                    per_stage[stage] = rr
    else:
        for i, stage in enumerate(stages):
            per_stage[stage] = _analyze_stage(
                rows=rows,
                labels_by_id=lab.labels_by_id,
                stage=stage,
                test_size=args.test_size,
                seed=args.seed + i,
                split_max_tries=args.split_max_tries,
                min_token_rel_freq=args.min_token_rel_freq,
                allow_utterance_fallback=bool(args.allow_utterance_fallback),
            )

    cross = _cross_stage_consistency(per_stage)

    out = {
        "method": "token_classifiers",
        "manifest_json": str(args.manifest_json),
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
    lines.append(f"Split policy: {split_policy} | allow_utterance_fallback={bool(args.allow_utterance_fallback)}")
    lines.append(f"Stages: {stages}")
    lines.append(f"Labeling: mode={lab.mode} pos={lab.positive_name} neg={lab.negative_name}")
    lines.append("")
    lines.append(f"{'Stage':<16} {'BOW bal':>10} {'SHARE bal':>10} {'SET bal':>10} {'Maj':>8} {'Top5 H':<24} {'Top5 L'}")
    for s in stages:
        r = per_stage.get(s, {})
        if not r.get("ok"):
            lines.append(f"{s:<16} error={r.get('error', 'no_result')}")
            continue

        def bal(mode: str) -> float:
            return float(r["modes"][mode]["metrics_h_vs_l"]["balanced_accuracy"]) if r["modes"][mode].get("ok") else float("nan")

        bow = bal("bow")
        share = bal("share")
        sett = bal("set")
        maj = r["modes"]["bow"].get("majority_baseline", float("nan")) if r["modes"]["bow"].get("ok") else float("nan")
        top_h = [int(x["token"]) for x in r["modes"]["share"]["top_tokens_h_model"][:5]] if r["modes"]["share"].get("ok") else []
        top_l = [int(x["token"]) for x in r["modes"]["share"]["top_tokens_l_model"][:5]] if r["modes"]["share"].get("ok") else []
        lines.append(f"{s:<16} {bow:>10.3f} {share:>10.3f} {sett:>10.3f} {float(maj):>8.3f} {str(top_h):<24} {str(top_l)}")

    lines.append("")
    lines.append("Cross-stage SHARE consistency (H-model)")
    for row in cross.get("share_h_model", []):
        lines.append(
            f"  {row['stage_a']} vs {row['stage_b']}: cosine={float(row['cosine']):.4f} mean|Δ|={float(row['mean_abs_delta']):.4f}"
        )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
