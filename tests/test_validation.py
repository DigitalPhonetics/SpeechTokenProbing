from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from minimal_probe.io import load_probe_rows, resolve_stage_vocab_ids
from minimal_probe.labeling import validate_quantile_bounds

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples" / "tiny_manifest.json"


def test_duplicate_row_ids_raise(tmp_path: Path) -> None:
    manifest = tmp_path / "dup.json"
    rows = [
        {
            "id": "utt_1",
            "speaker": "spk_1",
            "token_sets": {"tokenizer": [1, 2, 3]},
            "labels": {"gender": "M"},
        },
        {
            "id": "utt_1",
            "speaker": "spk_2",
            "token_sets": {"tokenizer": [4, 5, 6]},
            "labels": {"gender": "F"},
        },
    ]
    manifest.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate row id"):
        load_probe_rows(manifest)


def test_invalid_quantiles_raise() -> None:
    with pytest.raises(ValueError, match="0 <= qlo < qhi <= 1"):
        validate_quantile_bounds(0.8, 0.2)
    with pytest.raises(ValueError, match="0 <= qlo < qhi <= 1"):
        validate_quantile_bounds(-0.1, 0.8)
    with pytest.raises(ValueError, match="0 <= qlo < qhi <= 1"):
        validate_quantile_bounds(0.2, 1.2)


def test_resolve_stage_vocab_ids_missing_and_mismatch() -> None:
    stages = ["tokenizer", "zero"]
    with pytest.raises(ValueError, match="Multi-stage comparison requires"):
        resolve_stage_vocab_ids("", stages)
    with pytest.raises(ValueError, match="identical vocabulary ids"):
        resolve_stage_vocab_ids("tokenizer:tok,zero:other", stages)


def test_divergence_cli_fails_without_stage_vocab_ids(tmp_path: Path) -> None:
    out_prefix = tmp_path / "divergence"
    cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_divergence",
        str(MANIFEST),
        "--stages",
        "tokenizer,zero",
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--output-prefix",
        str(out_prefix),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "stage-vocab-ids" in proc.stderr


def test_divergence_cli_fails_with_mismatched_stage_vocab_ids(tmp_path: Path) -> None:
    out_prefix = tmp_path / "divergence"
    cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_divergence",
        str(MANIFEST),
        "--stages",
        "tokenizer,zero",
        "--stage-vocab-ids",
        "tokenizer:tok,zero:other",
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--output-prefix",
        str(out_prefix),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "identical vocabulary ids" in proc.stderr


def test_classifier_cli_fails_without_stage_vocab_ids(tmp_path: Path) -> None:
    out_prefix = tmp_path / "classifier"
    cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_classifier",
        str(MANIFEST),
        "--stages",
        "tokenizer,zero",
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--output-prefix",
        str(out_prefix),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "stage-vocab-ids" in proc.stderr
