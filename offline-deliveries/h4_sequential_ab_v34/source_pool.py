"""Freeze unused events from existing local source files without model calls.

Selection excludes whole dataset/split/item identities before excerpt creation.
A different card, span or excerpt never makes a consumed event available again.
This v24 module stratifies on observed dataset metadata, never guessed topics.
"""

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "h4_external_event_seed_selection_v1"
SCHEMA_VERSION = "h4.r02.dataset_cards.v1"
SELECTION_VERSION = "v24.dataset_metadata_stratified_sha256.v1"
SELECTION_SEED = "h4.r02.v24.unused_sources.v1"
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


def _metadata_stratum(dataset: str, entry: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Use native metadata labels, not inferred event topics or difficulty.

    SODA's relation is a commonsense relation label, not an event-type label.
    MultiWOZ's services are whole-dialogue domain labels; a bounded excerpt
    is not guaranteed to contain every listed service. Preserve those limits
    explicitly in the manifest instead of presenting either as a taxonomy.
    """
    if dataset == "allenai/soda":
        relation = entry["row"].get("relation")
        return "row.relation", (relation,) if isinstance(relation, str) and relation else ()
    services = entry.get("services", [])
    if not isinstance(services, list) or any(not isinstance(x, str) or not x for x in services):
        raise ValueError("source_services_metadata_invalid")
    return "services", tuple(sorted(set(services)))


def _rank(kind: str, value: Any) -> str:
    return _sha(_canonical([SELECTION_SEED, kind, value]))


def _stratified_order(candidates: dict, count: int) -> list[tuple[str, str, str]]:
    """Balance datasets, then their observed strata, with stable hash ties.

    The seed is versioned and fixed, not tuned to any root or model result.
    Exhausted strata/datasets gracefully yield their slots to remaining ones.
    Prefix stability permits small and larger dry runs to be compared directly.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    for key, entry in candidates.items():
        buckets[key[0]][_metadata_stratum(key[0], entry)].append(key)
    for strata in buckets.values():
        for keys in strata.values():
            keys.sort(key=lambda key: (_rank("event", key), key))
    dataset_counts, stratum_counts = Counter(), Counter()
    ordered = []
    while buckets and len(ordered) < count:
        dataset = min(buckets, key=lambda name: (dataset_counts[name], _rank("dataset", name), name))
        stratum = min(buckets[dataset], key=lambda label: (
            stratum_counts[(dataset, label)], _rank("stratum", [dataset, label]), label))
        keys = buckets[dataset][stratum]
        ordered.append(keys.pop(0))
        dataset_counts[dataset] += 1
        stratum_counts[(dataset, stratum)] += 1
        if not keys:
            del buckets[dataset][stratum]
        if not buckets[dataset]:
            del buckets[dataset]
    return ordered


def freeze_unused_source_subset(
    consumed_event_keys: set[tuple[str, str, str]],
    count: int,
    source_dir: str | Path = SOURCE_DIR,
) -> dict[str, Any]:
    """Return up to ``count`` unused events, deterministically and without writes.

    Dataset-balanced round robin replaces the old lexicographic head slice.
    Within each dataset, round robin covers native metadata strata; a fixed
    SHA-256 ranking breaks ties independently of item/file ordering. Metadata
    and selection order are recorded in the hashed manifest. Raw-item and
    excerpt hashes retain their existing byte definitions. Exhaustion returns
    remaining items, never reused events or invented replacements. Diversity
    here does not assess complexity, training value or semantic suitability.
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
    available = {key: entry for key, entry in candidates.items()
                 if key not in consumed_event_keys
                 and (key[0] != "MultiWOZ_2.2" or entry["turns"])}
    keys = _stratified_order(available, count)
    items = [_soda_item(candidates[key], soda_receipt) if key[0] == "allenai/soda"
             else _multiwoz_item(candidates[key], multiwoz_receipt) for key in keys]
    available_strata = Counter((key[0], _metadata_stratum(key[0], entry))
                              for key, entry in available.items())
    provenance = {
        "algorithm_version": SELECTION_VERSION, "seed": SELECTION_SEED,
        "requested_count": count, "selected_count": len(items),
        "consumed_event_keys_sha256": _sha(_canonical(sorted(consumed_event_keys))),
        "consumed_event_count": len(consumed_event_keys),
        "local_item_counts_by_dataset": dict(sorted(Counter(key[0] for key in candidates).items())),
        "available_counts_by_dataset": dict(sorted(Counter(key[0] for key in available).items())),
        "selected_counts_by_dataset": dict(sorted(Counter(key[0] for key in keys).items())),
        "stratification_limits": [
            "SODA row.relation is native commonsense relation metadata, not a topic/event-type certificate.",
            "MultiWOZ services are native whole-dialogue domain metadata; a bounded excerpt may cover fewer services.",
            "Missing labels use an empty metadata stratum; no synthetic event labels are assigned.",
        ],
        "available_strata": [
            {"dataset": dataset, "metadata_field": stratum[0],
             "metadata_values": list(stratum[1]), "count": n}
            for (dataset, stratum), n in sorted(available_strata.items())
        ],
        "selected_order": [
            {"selection_rank": index, "source_id": item["source_id"],
             "dataset": key[0], "metadata_field": _metadata_stratum(key[0], candidates[key])[0],
             "metadata_values": list(_metadata_stratum(key[0], candidates[key])[1]),
             "event_order_sha256": _rank("event", key)}
            for index, (key, item) in enumerate(zip(keys, items), 1)
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_method": SELECTION_VERSION,
        "selection_is_complexity_or_difficulty_certificate": False,
        "selection_provenance": provenance,
        "files": [soda_receipt, multiwoz_receipt], "items": items,
    }
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    return manifest
