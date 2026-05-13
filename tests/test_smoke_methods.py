from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "examples" / "tiny_manifest.json"


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="scikit-learn is not installed")
def test_method_smoke_runs(tmp_path: Path) -> None:
    divergence_prefix = tmp_path / "out" / "divergence"
    classifier_prefix = tmp_path / "out" / "classifier"
    attr_prefix = tmp_path / "out" / "attr_repr"

    vocab_map = "tokenizer:shared,zero:shared,text_only:shared"

    div_cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_divergence",
        str(MANIFEST),
        "--stages",
        "tokenizer,zero,text_only",
        "--stage-vocab-ids",
        vocab_map,
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--min-token-freq",
        "1",
        "--output-prefix",
        str(divergence_prefix),
    ]
    div = subprocess.run(div_cmd, cwd=ROOT, text=True, capture_output=True)
    assert div.returncode == 0, div.stderr

    clf_cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_classifier",
        str(MANIFEST),
        "--stages",
        "tokenizer,zero,text_only",
        "--stage-vocab-ids",
        vocab_map,
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--num-splits",
        "3",
        "--min-token-rel-freq",
        "0.0",
        "--output-prefix",
        str(classifier_prefix),
    ]
    clf = subprocess.run(clf_cmd, cwd=ROOT, text=True, capture_output=True)
    assert clf.returncode == 0, clf.stderr

    attr_cmd = [
        sys.executable,
        "-m",
        "minimal_probe.method_attr_repr",
        str(MANIFEST),
        "--stages",
        "tokenizer",
        "--label-field",
        "gender",
        "--pos-label",
        "M",
        "--neg-label",
        "F",
        "--count-threshold",
        "1",
        "--imbalance-threshold",
        "0.0",
        "--lambda-threshold",
        "0.0",
        "--output-prefix",
        str(attr_prefix),
    ]
    attr = subprocess.run(attr_cmd, cwd=ROOT, text=True, capture_output=True)
    assert attr.returncode == 0, attr.stderr

    for prefix in (divergence_prefix, classifier_prefix, attr_prefix):
        assert Path(str(prefix) + ".json").exists()
        assert Path(str(prefix) + ".txt").exists()

    div_out = json.loads(Path(str(divergence_prefix) + ".json").read_text(encoding="utf-8"))
    assert div_out["method"] == "distributional_divergence"
    assert div_out["stage_vocab_ids"]["tokenizer"] == "shared"

    clf_out = json.loads(Path(str(classifier_prefix) + ".json").read_text(encoding="utf-8"))
    assert clf_out["method"] == "token_classifiers"
    assert clf_out["defaults"]["split_policy"] == "shared_speaker_split"
    tok_stage = clf_out["per_stage"]["tokenizer"]
    assert tok_stage["num_splits_requested"] == 3
    assert "aggregate_modes" in tok_stage
    assert "ci95" in tok_stage["aggregate_modes"]["share"]["metrics_h_vs_l"]["balanced_accuracy"]
    assert "share_h_model_aggregate" in clf_out["cross_stage_consistency"]

    attr_out = json.loads(Path(str(attr_prefix) + ".json").read_text(encoding="utf-8"))
    assert attr_out["method"] == "attribute_conditioned_representation"
    assert attr_out["per_stage"]["tokenizer"]["ok"] is True
