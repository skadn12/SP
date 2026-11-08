"""Freeze unused events from existing local source files without model calls.

Selection excludes whole dataset/split/item identities before excerpt creation.
A different card, span or excerpt never makes a consumed event available again.
This v4 module owns the new selection logic; historical v1 code stays unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "h4_external_event_seed_selection_v1"
SCHEMA_VERSION = "h4.r02.dataset_cards.v1"
_spec = importlib.util.spec_from_file_location(
    "r02_source_pool_legacy_cards", HERE.parent / "h4_r02_dualroute_feedback_v1/dataset_cards.py"
)
assert _spec and _spec.loader
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
# Only the existing byte reader and canonical hashing helpers are shared.
_read_source, _canonical, _sha = _legacy._read_source, _legacy._canonical, _legacy._sha


def _soda_item(entry: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    row = entry["row"]
    if row["split"] != "train" or entry.get("truncated_cells"):
        raise ValueError("SODA sample must be complete train data")
    if len(row["dialogue"]) != len(row["speakers"]):
        raise ValueError("SODA speaker/dialogue lengths differ")
    units = [{"unit_id": "narrative", "kind": "narrative", "speaker": None,
              "text": row["narrative"]}]
    units.extend(
        {"unit_id": f"dialogue:{index}", "kind": "dialogue", "speaker": speaker,
         "text": text}
        for index, (speaker, text) in enumerate(zip(row["speakers"], row["dialogue"]))
    )
    return {
        "source_id": f"soda:train:{row['original_index']}",
        "dataset": "allenai/soda", "split": "train", "item_id": str(row["original_index"]),
        "local_row_index": entry["row_idx"],
        "source_file_sha256": receipt["local_sha256"],
        "source_item_sha256": _sha(_canonical(row)),
        "source_span": {"narrative": "full", "dialogue_start": 0,
                        "dialogue_end": len(row["dialogue"]) - 1},
        "source_speakers": sorted(set(row["speakers"])), "units": units,
        "excerpt_sha256": _sha(_canonical(units)),
    }


def _multiwoz_item(original: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    # Match the bounded excerpt used by the initial unused-source selector.
    end = min(11, len(original["turns"]) - 1)
    turns = [turn for turn in original["turns"] if 0 <= int(turn["turn_id"]) <= end]
    if end < 0 or [int(turn["turn_id"]) for turn in turns] != list(range(end + 1)):
        raise ValueError("MultiWOZ source span is not complete")
    units = [{"unit_id": f"turn:{turn['turn_id']}", "kind": "dialogue",
              "speaker": turn["speaker"], "text": turn["utterance"]} for turn in turns]
    return {
        "source_id": f"multiwoz22:train:{original['dialogue_id']}",
        "dataset": "MultiWOZ_2.2", "split": "train", "item_id": original["dialogue_id"],
        "source_file_sha256": receipt["local_sha256"],
        "source_item_sha256": _sha(_canonical(original)),
        "source_span": {"turn_start": 0, "turn_end": end},
        "source_speakers": sorted({unit["speaker"] for unit in units}), "units": units,
        "excerpt_sha256": _sha(_canonical(units)),
    }


def freeze_unused_source_subset(
    consumed_event_keys: set[tuple[str, str, str]],
    count: int,
    source_dir: str | Path = SOURCE_DIR,
) -> dict[str, Any]:
    """Return up to ``count`` unused events, deterministically and without writes.

    Selection sorts by dataset and item ID, as in the initial unused-source
    selector. All lineage hashes bind actual source bytes, full raw items, and
    selected excerpts. Exhaustion returns the remaining items (possibly none),
    never reuses historical events or invents replacements. This does not
    assess source complexity, training value or semantic suitability.
    """
    if type(count) is not int or count < 1:
        raise ValueError("source_count_invalid")
    if not isinstance(consumed_event_keys, (set, frozenset)) or any(
        not isinstance(key, tuple) or len(key) != 3
        or not all(isinstance(part, str) and part for part in key)
        for key in consumed_event_keys
    ):
        raise ValueError("consumed_event_keys_invalid")
    directory = Path(source_dir)
    soda, soda_receipt = _read_source(
        directory / "soda_train_sample.json", directory / "soda_train_sample_source.json", "allenai/soda"
    )
    multiwoz, multiwoz_receipt = _read_source(
        directory / "multiwoz_train_sample.json", directory / "multiwoz_train_source.json", "MultiWOZ_2.2"
    )
    candidates = {}
    for dataset, entries in (("allenai/soda", soda["rows"]), ("MultiWOZ_2.2", multiwoz)):
        for entry in entries:
            item_id = str(entry["row"]["original_index"]) if dataset == "allenai/soda" else entry["dialogue_id"]
            key = (dataset, "train", item_id)
            if key in candidates:
                raise ValueError("duplicate_source_event_identity")
            candidates[key] = entry
    keys = [key for key in sorted(candidates, key=lambda key: (key[0], key[2]))
            if key not in consumed_event_keys
            and (key[0] != "MultiWOZ_2.2" or candidates[key]["turns"])][:count]
    # Preserve the existing manifest's dataset-grouped output order; selection
    # itself is determined by the ordered candidate keys above.
    items = [_soda_item(candidates[key], soda_receipt) for key in keys if key[0] == "allenai/soda"]
    items.extend(_multiwoz_item(candidates[key], multiwoz_receipt)
                 for key in keys if key[0] == "MultiWOZ_2.2")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": "deterministic_unused_event_pool_from_frozen_local_sources",
        "selection_is_complexity_or_difficulty_certificate": False,
        "files": [soda_receipt, multiwoz_receipt], "items": items,
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    return manifest
