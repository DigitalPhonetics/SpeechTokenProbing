# SpeechTokenProbing

Model-independent implementation of:

1. **Distributional divergence** (`method_divergence.py`)
2. **Token-based classifiers** (`method_classifier.py`)
3. **Attribute-conditioned representation** (`method_attr_repr.py`)


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
- `id` (string)
- `speaker` (string)
- `token_sets` (dict stage -> token sequence)
- `labels` (dict)

Optional:
- `phones` (dict stage -> aligned phone sequence), used only by Method 3.

Notes:
- Stage names in `token_sets` are arbitrary and user-defined.
- Method 3 only applies to stages that also exist in `phones` with aligned same-length sequences.

## Labeling Modes

All three scripts support both:
- **binary mode**: `--label-field`, `--pos-label`, `--neg-label`
- **quantile mode**: `--score-field`, `--qlo`, `--qhi`

Choose exactly one mode per run.

## Paper-Default Behavior

- **Distributional divergence** uses full-group empirical distributions by default (no class downsampling).
- **Token-based classifiers** uses per-stage speaker-disjoint splits by default and marks only failing stages as `ok=false` (run continues).
- **Attribute-conditioned representation** skips missing/misaligned aligned rows, reports skip counts, and fails a stage only if no usable aligned rows remain.

## Optional Flags

- Distributional divergence:
  - `--balanced-downsample` to use class-balanced downsampling before divergence.
- Token-based classifiers:
  - `--shared-split-across-stages` to force one shared split across requested stages.
  - `--allow-utterance-fallback` to allow utterance-level fallback if speaker-disjoint split fails.

## Distributional divergence Example

```bash
python3 -m minimal_probe.method_divergence data.json \
  --stages tokenizer,zero,text_only \
  --label-field gender --pos-label M --neg-label F \
  --output-prefix out/divergence
```

Alternative:

```bash
python3 -m minimal_probe.method_divergence data.json \
  --stages tokenizer,zero,text_only \
  --label-field gender --pos-label M --neg-label F \
  --balanced-downsample \
  --output-prefix out/divergence_downsampled
```

## Token-based classifiers Example

```bash
python3 -m minimal_probe.method_classifier data.json \
  --stages tokenizer,zero,text_only \
  --score-field valence --qlo 0.2 --qhi 0.8 \
  --output-prefix out/classifier
```

Optional shared-split alternative:

```bash
python3 -m minimal_probe.method_classifier data.json \
  --stages tokenizer,zero,text_only \
  --score-field valence --qlo 0.2 --qhi 0.8 \
  --shared-split-across-stages \
  --output-prefix out/classifier_shared_split
```

## Attribute-conditioned representation Example

```bash
python3 -m minimal_probe.method_attr_repr data.json \
  --stages tokenizer \
  --label-field gender --pos-label M --neg-label F \
  --output-prefix out/attr_repr
```

## Outputs

Each method writes:
- `<output-prefix>.json` (compact machine-readable result)
- `<output-prefix>.txt` (short human summary)

## Defaults 

- Distributional divergence: `--num-random-trials 20`, `--min-token-freq 50`
- Token-based classifiers: speaker-disjoint split `0.8/0.2`, train-only frequency filter `0.002%`
- Attribute-conditioned representation: `count>=400`, `balance>=0.2`, `|lambda|>=0.015`
