"""Input loading and schema helpers for minimal probe manifests.

This module provides a lightweight manifest contract used by all three probe
methods. It normalizes row-level fields, converts token and phone sequences
into canonical Python types, and reports malformed items via warnings.

Assumptions:
    - Each usable row has a speaker id, at least one non-empty token stage, and
      a non-empty labels dictionary.
    - Token stages are stored under ``token_sets`` and are integer-like
      sequences.
    - Phone stages are optional and only required by Method 3.

This module is a library utility module (no CLI entrypoint).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ProbeRow:
    """Normalized probe row used across all analysis methods.

    Attributes:
        row_id: Stable utterance identifier. Falls back to ``row_{index}`` when
            absent in the source manifest.
        speaker: Speaker identifier used for speaker-disjoint splitting.
        token_sets: Mapping from arbitrary stage names (for example
            ``tokenizer``, ``zero``) to integer token sequences.
        labels: Label payload dictionary. Supports both flat and nested keys
            through dotted path lookup.
        phones: Optional mapping from stage names to aligned phone sequences.
            Method 3 only applies to stages that exist here and match token
            sequence lengths.
        text: Optional transcript text preserved as-is.
    """

    row_id: str
    speaker: str
    token_sets: Dict[str, List[int]]
    labels: Dict[str, Any]
    phones: Dict[str, List[str]]
    text: str


def _to_int_sequence(x: Any) -> Tuple[List[int], int]:
    """Convert a sequence-like object to integers while dropping bad items.

    Args:
        x: Candidate sequence value from the manifest.

    Returns:
        A tuple ``(values, dropped_count)`` where ``values`` contains successful
        integer conversions and ``dropped_count`` counts elements that failed
        conversion.
    """

    if not isinstance(x, (list, tuple)):
        return [], 0
    out: List[int] = []
    dropped = 0
    for v in x:
        try:
            out.append(int(v))
        except Exception:
            dropped += 1
            continue
    return out, dropped


def _to_str_sequence(x: Any) -> Tuple[List[str], int]:
    """Convert a sequence-like object to non-empty stripped strings.

    Args:
        x: Candidate sequence value from the manifest.

    Returns:
        A tuple ``(values, dropped_count)`` where ``values`` contains stripped
        non-empty strings and ``dropped_count`` counts empty-string elements.
    """

    if not isinstance(x, (list, tuple)):
        return [], 0
    out: List[str] = []
    dropped = 0
    for v in x:
        s = str(v).strip()
        if s:
            out.append(s)
        else:
            dropped += 1
    return out, dropped


def _normalize_row(raw: dict, fallback_id: str) -> Tuple[ProbeRow, Dict[str, int]]:
    """Validate and normalize one raw manifest row.

    Args:
        raw: Unnormalized row dictionary.
        fallback_id: Identifier used when ``raw["id"]`` is missing or blank.

    Returns:
        A tuple ``(row, stats)`` where ``row`` is a normalized :class:`ProbeRow`
        and ``stats`` contains dropped sequence-item counters.

    Raises:
        ValueError: If required fields are missing or unusable (speaker,
            token_sets, labels).
    """

    row_id = str(raw.get("id", fallback_id)).strip()
    if not row_id:
        row_id = fallback_id

    speaker = str(raw.get("speaker", "")).strip()
    if not speaker:
        raise ValueError(f"row {row_id}: missing required 'speaker'")

    token_sets_raw = raw.get("token_sets")
    if not isinstance(token_sets_raw, dict) or not token_sets_raw:
        raise ValueError(f"row {row_id}: missing required non-empty 'token_sets' dict")

    token_sets: Dict[str, List[int]] = {}
    dropped_token_items = 0
    for stage, seq in token_sets_raw.items():
        st = str(stage).strip()
        if not st:
            continue
        toks, dropped = _to_int_sequence(seq)
        dropped_token_items += int(dropped)
        if toks:
            token_sets[st] = toks
    if not token_sets:
        raise ValueError(f"row {row_id}: no valid token sequences in 'token_sets'")

    labels = raw.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"row {row_id}: missing required non-empty 'labels' dict")

    phones_raw = raw.get("phones", {})
    phones: Dict[str, List[str]] = {}
    dropped_phone_items = 0
    if isinstance(phones_raw, dict):
        for stage, seq in phones_raw.items():
            st = str(stage).strip()
            if not st:
                continue
            ph, dropped = _to_str_sequence(seq)
            dropped_phone_items += int(dropped)
            if ph:
                phones[st] = ph

    text = str(raw.get("text", "")).strip()
    return ProbeRow(
        row_id=row_id,
        speaker=speaker,
        token_sets=token_sets,
        labels=labels,
        phones=phones,
        text=text,
    ), {
        "dropped_token_items": int(dropped_token_items),
        "dropped_phone_items": int(dropped_phone_items),
    }


def load_probe_rows(path: str | Path) -> List[ProbeRow]:
    """Load and normalize a probe manifest from JSON.

    Supported top-level layouts are:
        - list of row dicts
        - dict mapping ids to row dicts

    Non-dict rows are skipped with warnings. Malformed sequence items inside
    valid rows are dropped and summarized via warnings.

    Args:
        path: Path to the JSON manifest.

    Returns:
        List of normalized :class:`ProbeRow` objects.

    Raises:
        ValueError: If the manifest top-level JSON type is unsupported or a row
            fails required-field validation.
    """

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    non_dict_rows_skipped = 0
    if isinstance(obj, dict):
        raw_rows = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rr = dict(v)
                if "id" not in rr:
                    rr["id"] = str(k)
                raw_rows.append(rr)
            else:
                non_dict_rows_skipped += 1
    elif isinstance(obj, list):
        raw_rows = []
        for r in obj:
            if isinstance(r, dict):
                raw_rows.append(r)
            else:
                non_dict_rows_skipped += 1
    else:
        raise ValueError(f"Unsupported manifest type: {type(obj).__name__}")

    rows: List[ProbeRow] = []
    dropped_token_items_total = 0
    dropped_phone_items_total = 0
    for i, raw in enumerate(raw_rows):
        row, row_stats = _normalize_row(raw, fallback_id=f"row_{i}")
        rows.append(row)
        dropped_token_items_total += int(row_stats.get("dropped_token_items", 0))
        dropped_phone_items_total += int(row_stats.get("dropped_phone_items", 0))

    if non_dict_rows_skipped > 0:
        warnings.warn(
            f"Skipped {non_dict_rows_skipped} non-dict rows while loading manifest {p}",
            UserWarning,
            stacklevel=2,
        )
    if dropped_token_items_total > 0 or dropped_phone_items_total > 0:
        warnings.warn(
            "Dropped malformed sequence items while loading manifest "
            f"{p}: tokens={dropped_token_items_total}, phones={dropped_phone_items_total}",
            UserWarning,
            stacklevel=2,
        )
    return rows


def get_nested_label(labels: Dict[str, Any], field: str) -> Any:
    """Resolve a possibly dotted label path from a label dictionary.

    Args:
        labels: Label dictionary from one row.
        field: Label field path, for example ``"gender"`` or
            ``"vad.valence"``.

    Returns:
        The resolved value when the path exists; otherwise ``None``.
    """

    cur: Any = labels
    for part in str(field).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve_stage_list(rows: List[ProbeRow], stages_csv: str) -> List[str]:
    """Resolve requested stage names from CLI-style stage input.

    Args:
        rows: Loaded probe rows.
        stages_csv: Comma-separated stage names or ``"all"``.

    Returns:
        Ordered list of stage names to analyze.

    Raises:
        ValueError: If no stage name can be resolved from explicit input.
    """

    if stages_csv.strip().lower() == "all":
        seen = set()
        out: List[str] = []
        for r in rows:
            for s in r.token_sets.keys():
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out

    stages = [s.strip() for s in stages_csv.split(",") if s.strip()]
    if not stages:
        raise ValueError("No stages resolved from --stages")
    return stages


def ensure_stage_exists(rows: List[ProbeRow], stages: List[str]) -> None:
    """Validate that all requested stages exist in at least one row.

    Args:
        rows: Loaded probe rows.
        stages: Requested stage names.

    Raises:
        ValueError: If one or more requested stages are absent from
            ``token_sets`` across all rows.
    """

    known = set()
    for r in rows:
        known.update(r.token_sets.keys())
    missing = [s for s in stages if s not in known]
    if missing:
        raise ValueError(f"Requested stages not found in token_sets: {missing}")
