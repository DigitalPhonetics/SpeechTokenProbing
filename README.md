# SpeechTokenProbing

Minimal, model-agnostic implementation of three probe families from the paper:

1. Distributional divergence (`minimal_probe.method_divergence`)
2. Token-based classifiers (`minimal_probe.method_classifier`)
3. Attribute-conditioned token-phone representation (`minimal_probe.method_attr_repr`)

This repository is a probing toolkit release. It does **not** include the full paper pipeline artifacts (token extraction, MFA/VAD preprocessing, shared-text/silence controls, table-generation scripts, or full paper outputs).

## Installation

```bash
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

## Package Layout

Code lives in the `minimal_probe/` package and is intended to run via:

```bash
python -m minimal_probe.method_divergence ...
python -m minimal_probe.method_classifier ...
python -m minimal_probe.method_attr_repr ...
```

## Canonical Input Schema

Input manifest is JSON list (or dict of rows), each row following:

```json
{
  "id": "utt_0001",
  "speaker": "spk_01",
  "text": "optional text",
  "token_sets": {
    "tokenizer": [101, 203, 55],
    "zero": [77, 91, 12],
    "text_only": [9, 14, 82]
  },
  "labels": {
    "gender": "M",
    "valence": 0.63,
    "arousal": 0.41,
    "dominance": 0.52
  },
  "phones": {
    "tokenizer": ["AH0", "N", "T"]
  }
}
```

Required fields:

- `id` (must be unique)
- `speaker`
- `token_sets`
- `labels`

Optional fields:

- `text`
- `phones` (required by Method 3 for aligned token-phone analysis)

## Labeling Modes

All scripts support exactly one labeling mode per run:

- Binary mode: `--label-field`, `--pos-label`, `--neg-label`
- Quantile mode: `--score-field`, `--qlo`, `--qhi`

Quantile bounds are validated globally and must satisfy `0 <= qlo < qhi <= 1`.

## Strict Cross-Stage Vocabulary Guard

Method 1 and Method 2 perform cross-stage comparisons. For multi-stage runs you must pass:

```bash
--stage-vocab-ids tokenizer:shared,zero:shared,text_only:shared
```

All compared stages must resolve to the same vocab id, otherwise execution fails.

## Method 1 Example (Divergence)

```bash
python -m minimal_probe.method_divergence examples/tiny_manifest.json \
  --stages tokenizer,zero,text_only \
  --stage-vocab-ids tokenizer:shared,zero:shared,text_only:shared \
  --label-field gender --pos-label M --neg-label F \
  --output-prefix out/divergence
```

## Method 2 Example (Classifier)

Default policy uses **shared speaker-disjoint split across stages** and repeated runs for uncertainty.

```bash
python -m minimal_probe.method_classifier examples/tiny_manifest.json \
  --stages tokenizer,zero,text_only \
  --stage-vocab-ids tokenizer:shared,zero:shared,text_only:shared \
  --score-field valence --qlo 0.2 --qhi 0.8 \
  --num-splits 10 \
  --class-weight balanced \
  --output-prefix out/classifier
```

Legacy split policy can be requested explicitly:

```bash
python -m minimal_probe.method_classifier examples/tiny_manifest.json \
  --stages tokenizer,zero,text_only \
  --stage-vocab-ids tokenizer:shared,zero:shared,text_only:shared \
  --score-field valence --qlo 0.2 --qhi 0.8 \
  --per-stage-split \
  --output-prefix out/classifier_per_stage
```

## Method 3 Example (Attribute-Conditioned)

```bash
python -m minimal_probe.method_attr_repr examples/tiny_manifest.json \
  --stages tokenizer \
  --label-field gender --pos-label M --neg-label F \
  --output-prefix out/attr_repr
```

## Outputs

Each method writes:

- `<output-prefix>.json`
- `<output-prefix>.txt`

Method 2 JSON now includes:

- per-split series metadata
- aggregate mean/std/95% CI for classifier metrics
- aggregate cross-stage consistency summaries

## Example Data and Tests

- Example manifest: `examples/tiny_manifest.json`
- Tests: `tests/`

Run tests:

```bash
pytest -q
```
