"""Create research-only shuffled display packets for the ten legacy candidates.

The source candidates are immutable historical records.  This adapter only
creates a deterministic presentation mapping; it does not convert the legacy
schema into a V2.2.1 candidate, grant source admission, or run a Reviewer.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_ROOTS = HERE / "roots"
OUTPUT = HERE / "shuffled_display_research"
RENDERER = "base_qa_option_freeze_v2_2_1"
SEED = "independent-review-v2.2.1"


def typed_key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(typed_key(value).encode("utf-8")).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(root_dir: Path) -> dict:
    candidate = json.loads((root_dir / "HISTORICAL_CANDIDATE.json").read_text(encoding="utf-8"))
    source = json.loads((root_dir / "FULL_SOURCE.json").read_text(encoding="utf-8"))
    candidate_id = candidate["candidate_id"]
    labels = sorted(candidate["options"])
    canonical_options = {f"option_{label}": candidate["options"][label] for label in labels}
    canonical_gold = [f"option_{label}" for label in candidate["gold"]]
    ids = sorted(canonical_options)
    seed_material = [RENDERER, SEED, candidate_id, ids]
    rng = random.Random(int(digest(seed_material), 16))
    rng.shuffle(ids)
    display_to_canonical = {chr(65 + i): option_id for i, option_id in enumerate(ids)}
    canonical_to_display = {option_id: label for label, option_id in display_to_canonical.items()}
    display_options = {label: canonical_options[option_id] for label, option_id in display_to_canonical.items()}
    display_gold = sorted(canonical_to_display[option_id] for option_id in canonical_gold)
    mapping = {
        "schema_version": "base_qa_v2_2_1_legacy_display_mapping_v1",
        "renderer": RENDERER,
        "renderer_mode": "algorithm_compatible_projection_not_formal_v221_bundle",
        "seed": SEED,
        "candidate_id": candidate_id,
        "canonical_option_ids": sorted(canonical_options),
        "display_to_canonical": display_to_canonical,
        "canonical_to_display": canonical_to_display,
        "canonical_gold": canonical_gold,
        "display_gold": display_gold,
        "mapping_sha256": digest(display_to_canonical),
        "source_candidate_sha256": digest(candidate),
        "formal_qa_allowed": False,
        "status": "RESEARCH_ONLY_SHUFFLED_DISPLAY",
    }
    packet = {
        "schema_version": "base_qa_v2_2_1_legacy_blind_packet_projection_v1",
        "stage": "INDEPENDENT_SOLVE",
        "reviewer_mode": "same_model_blind",
        "candidate_id": candidate_id,
        "root_id": candidate["root_id"],
        "question_type": candidate["question_type"],
        "question": candidate["question"],
        "options": display_options,
        "source": source,
        "source_sha256": digest(source),
        "mapping_sha256": mapping["mapping_sha256"],
        "formal_qa_allowed": False,
        "status": "RESEARCH_ONLY_NO_REVIEWER_RUN",
    }
    # The display packet intentionally has no Gold, item analysis, or author
    # proof.  The private mapping is separate and is not a review result.
    out = OUTPUT / candidate["root_id"]
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "DISPLAY_PACKET.json", packet)
    write_new(out / "DISPLAY_MAPPING_PRIVATE.json", mapping)
    write_new(out / "DISPLAY_SOURCE_BINDING.json", {
        "path": str(root_dir / "FULL_SOURCE.json"),
        "sha256": digest(source),
        "bytes": (root_dir / "FULL_SOURCE.json").stat().st_size,
        "formal_qa_allowed": False,
    })
    return {
        "root_id": candidate["root_id"],
        "candidate_id": candidate_id,
        "question_type": candidate["question_type"],
        "display_to_canonical": display_to_canonical,
        "canonical_gold": canonical_gold,
        "display_gold": display_gold,
        "mapping_sha256": mapping["mapping_sha256"],
        "packet_sha256": digest(packet),
        "formal_qa_allowed": False,
        "status": "RESEARCH_ONLY_SHUFFLED_DISPLAY",
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"output already exists: {OUTPUT}")
    rows = []
    for root_dir in sorted(INPUT_ROOTS.iterdir()):
        if root_dir.is_dir() and (root_dir / "HISTORICAL_CANDIDATE.json").is_file():
            rows.append(build(root_dir))
    manifest = {
        "schema_version": "base_qa_v2_2_1_ten_root_shuffled_display_manifest_v1",
        "renderer": RENDERER,
        "renderer_mode": "algorithm_compatible_projection_not_formal_v221_bundle",
        "seed": SEED,
        "candidate_count": len(rows),
        "formal_pass_count": 0,
        "reviewer_run_count": 0,
        "solver_run_count": 0,
        "purpose": "repair presentation-label omission for legacy candidate inspection",
        "source_candidates_unchanged": True,
        "formal_admission_unchanged": True,
        "rows": rows,
        "interpretation": (
            "Stable canonical option IDs are mapped to a deterministic display order. "
            "Gold is private and remapped only for checking the display. These packets "
            "are not current V2.2.1 candidates and have not been independently reviewed."
        ),
    }
    write_new(OUTPUT / "SHUFFLED_DISPLAY_MANIFEST.json", manifest)
    print(json.dumps({"output": str(OUTPUT), "candidate_count": len(rows), "formal_pass_count": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
