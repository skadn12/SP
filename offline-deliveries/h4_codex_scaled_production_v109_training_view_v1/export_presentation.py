#!/usr/bin/env python3
"""Create an exclusive, deterministic presentation-only training candidate view.

This tool uses Python's standard library, makes no model calls, and never writes
to its source package. The output directory must not exist.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ALGORITHM = "sha256_counter_fisher_yates_v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def messages_sha(messages):
    # Match the source FRONTIER_MANIFEST's documented JSON serialization.
    return sha_bytes(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def snapshot(directory):
    return [
        {"relative_path": str(path.relative_to(directory)), "sha256": sha_file(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def permutation(namespace, candidate_id, labels):
    """Fisher-Yates with a fully specified SHA256 random stream."""
    seed = hashlib.sha256(namespace.encode("utf-8") + b"\0" + candidate_id.encode("utf-8")).digest()
    counter = 0
    order = list(labels)
    for i in range(len(order) - 1, 0, -1):
        bound = i + 1
        limit = (1 << 256) // bound * bound
        while True:
            value = int.from_bytes(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest(), "big")
            counter += 1
            if value < limit:
                j = value % bound
                break
        order[i], order[j] = order[j], order[i]
    return order, seed.hex()


def pattern_audit(records, field):
    groups = defaultdict(list)
    for record in records:
        groups[record["question_type"]].append(record)
    result = {}
    for kind, group in sorted(groups.items()):
        sequences = Counter(",".join(record[field]) for record in group)
        positions = defaultdict(Counter)
        for record in group:
            for index, label in enumerate(record[field], 1):
                positions[str(index)][label] += 1
        result[kind] = {
            "question_count": len(group),
            "gold_label_sequence_counts": dict(sorted(sequences.items())),
            "unique_gold_label_sequences": len(sequences),
            "maximum_identical_gold_label_sequence_count": max(sequences.values(), default=0),
            "answer_position_label_counts": {p: dict(sorted(c.items())) for p, c in sorted(positions.items())},
        }
    return result


def export(args):
    source = Path(args.source_package).resolve()
    pool_path = Path(args.pool).resolve()
    output = Path(args.output).resolve()
    tool = Path(__file__).resolve()
    require(source.is_dir() and pool_path.is_file(), "Source package/pool must exist")
    require(output != source and source not in output.parents and output not in source.parents,
            "Output and source package must be separate directory trees")
    require(output != pool_path.parent and output not in pool_path.parents,
            "Output cannot contain an existing input pool")
    # Refuse overwrite, including partially completed earlier exports.
    output.mkdir(parents=False, exist_ok=False)
    before = snapshot(source)
    pool_hash = sha_file(pool_path)
    config = {
        "status": "FIXED_BEFORE_NEW_PRESENTATION_DISTRIBUTION_COMPUTED",
        "created_at_utc": now(),
        "seed_namespace": args.seed,
        "candidate_seed": "SHA256(UTF8(seed_namespace) + NUL + UTF8(candidate_id))",
        "algorithm": ALGORITHM,
        "algorithm_detail": "Start with A-F or I1-I6. For i=5..1 use SHA256(seed_bytes + uint64be(counter)), counter from 0, with rejection above floor(2^256/(i+1))*(i+1); swap i with unbiased draw modulo i+1.",
        "seed_search_performed": False,
        "source_package": str(source),
        "pool_path": str(pool_path),
        "pool_sha256": pool_hash,
        "source_package_snapshot": before,
        "tool": {"path": str(tool), "sha256": sha_file(tool)},
        "scope": "Presentation-only version, not a new semantic batch or preregistration; no model call, training, new question, selection change, weight change, or replay mixing.",
    }
    # Persist configuration before computing any output permutation/distribution.
    write_json(output / "CONFIG_FREEZE.json", config)

    old_complete = json.loads((source / "COMPLETE_MANIFEST.json").read_text())
    for entry in old_complete["files"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = source / path
        require(sha_file(path) == entry["sha256"], "Old package manifest hash mismatch: " + str(path))
    manifest = json.loads((source / "FRONTIER_MANIFEST.json").read_text())
    rows = [json.loads(line) for line in (source / "frontier_candidates.jsonl").read_text().splitlines() if line.strip()]
    candidates = manifest["candidates"]
    require(len(rows) == len(candidates) == manifest["candidate_count"], "Source row/manifest count mismatch")
    require(len({c["candidate_id"] for c in candidates}) == len(candidates), "Duplicate source candidate ID")
    pool = json.loads(pool_path.read_text())
    questions = {q["candidate_id"]: q for q in pool["questions"]}
    require(len(questions) == len(pool["questions"]), "Duplicate pool candidate ID")

    new_rows, new_candidates, mappings = [], [], []
    for index, (row, candidate) in enumerate(zip(rows, candidates)):
        cid = candidate["candidate_id"]
        require(cid in questions, "Candidate absent from pool: " + cid)
        question = questions[cid]
        kind = candidate["question_type"]
        require(kind in ("multi_select", "ordering") and question["question_type"] == kind, "Question type mismatch: " + cid)
        messages = row["messages"]
        require(len(messages) == 3 and [m["role"] for m in messages] == ["system", "user", "assistant"], "Expected three message roles: " + cid)
        require(messages_sha(messages) == candidate["messages_sha256"], "Old message hash mismatch: " + cid)
        require(candidate["source_sha256"] == question["source_sha256"], "Source hash binding mismatch: " + cid)
        labels = list("ABCDEF") if kind == "multi_select" else ["I" + str(i) for i in range(1, 7)]
        content = question["options"] if kind == "multi_select" else question["items"]
        require(set(content) == set(labels), "Expected six canonical labels: " + cid)
        require(all(isinstance(content[label], str) and "\n" not in content[label] for label in labels), "Items must be single-line text: " + cid)
        old_gold = list(candidate["canonical_gold"])
        require(old_gold == question["gold"], "Pool/source Gold mismatch: " + cid)
        require(messages[2]["content"] == "<box>" + ",".join(old_gold) + "</box>", "Assistant/Gold mismatch: " + cid)
        header = "Options:" if kind == "multi_select" else "Items:"
        old_block = "\n\n" + header + "\n" + "\n".join(label + ". " + content[label] for label in labels) + "\n\n"
        user = messages[1]["content"]
        require(user.count(old_block) == 1, "Replacement block must occur exactly once: " + cid)
        prefix, suffix = user.split(old_block)
        require("Question (" + kind + "):\n" + question["question"] in prefix, "Question text mismatch: " + cid)
        require(suffix.startswith("Output only "), "Output instruction boundary missing: " + cid)

        old_labels_in_new_order, seed_hash = permutation(args.seed, cid, labels)
        new_to_old = dict(zip(labels, old_labels_in_new_order))
        old_to_new = {old: new for new, old in new_to_old.items()}
        new_content = {new: content[old] for new, old in new_to_old.items()}
        new_gold = [old_to_new[old] for old in old_gold]
        if kind == "multi_select":
            new_gold.sort(key=labels.index)
        new_block = "\n\n" + header + "\n" + "\n".join(label + ". " + new_content[label] for label in labels) + "\n\n"
        new_row = copy.deepcopy(row)
        new_row["messages"][1]["content"] = prefix + new_block + suffix
        new_row["messages"][2]["content"] = "<box>" + ",".join(new_gold) + "</box>"
        new_messages = new_row["messages"]
        require(new_messages[1]["content"].count(new_block) == 1, "New block count mismatch: " + cid)
        new_prefix, new_suffix = new_messages[1]["content"].split(new_block)
        old_gold_texts = [content[label] for label in old_gold]
        new_gold_texts = [new_content[label] for label in new_gold]
        content_same = Counter(content.values()) == Counter(new_content.values())
        gold_same = Counter(old_gold_texts) == Counter(new_gold_texts) if kind == "multi_select" else old_gold_texts == new_gold_texts
        require(content_same and gold_same, "Content/Gold meaning changed: " + cid)
        require(prefix == new_prefix and suffix == new_suffix, "User message outside block changed: " + cid)
        require(messages[0] == new_messages[0] and [m["role"] for m in new_messages] == ["system", "user", "assistant"], "System/roles changed: " + cid)
        require({k: v for k, v in row.items() if k != "messages"} == {k: v for k, v in new_row.items() if k != "messages"}, "Row metadata changed: " + cid)
        require({k: v for k, v in messages[1].items() if k != "content"} == {k: v for k, v in new_messages[1].items() if k != "content"}, "User metadata changed: " + cid)
        require({k: v for k, v in messages[2].items() if k != "content"} == {k: v for k, v in new_messages[2].items() if k != "content"}, "Assistant metadata changed: " + cid)
        mapped = {
            "row_index_zero_based": index, "candidate_id": cid, "root_id": candidate["root_id"],
            "question_type": kind, "candidate_seed_sha256": seed_hash,
            "old_label_to_new_label": old_to_new, "new_label_to_old_label": new_to_old,
            "old_content_by_label": content, "new_content_by_label": new_content,
            "old_gold": old_gold, "new_gold": new_gold,
            "old_gold_texts": old_gold_texts, "new_gold_texts": new_gold_texts,
            "old_messages_sha256": messages_sha(messages), "new_messages_sha256": messages_sha(new_messages),
            "old_message_content_sha256": {m["role"]: sha_bytes(m["content"].encode("utf-8")) for m in messages},
            "new_message_content_sha256": {m["role"]: sha_bytes(m["content"].encode("utf-8")) for m in new_messages},
            "unchanged_user_prefix_sha256": sha_bytes(prefix.encode("utf-8")),
            "unchanged_user_suffix_sha256": sha_bytes(suffix.encode("utf-8")),
            "content_multiset_unchanged": content_same, "correct_content_unchanged": gold_same,
            "three_roles_unchanged": True, "user_outside_replacement_block_unchanged": True,
            "system_message_unchanged": True, "replacement_block_occurrences": 1,
        }
        updated = copy.deepcopy(candidate)
        updated["source_canonical_gold"] = old_gold
        updated["source_messages_sha256"] = candidate["messages_sha256"]
        updated["canonical_gold"] = new_gold
        updated["messages_sha256"] = mapped["new_messages_sha256"]
        updated["training_view_content"] = new_content
        updated["presentation_mapping_row"] = index
        updated["solver_question_statistics_scope"] = "Unchanged historical solver statistics use source canonical labels and randomized solver displays; do not reinterpret their gold/content fields as new training-view labels."
        new_rows.append(new_row)
        new_candidates.append(updated)
        mappings.append(mapped)

    types = Counter(c["question_type"] for c in candidates)
    multigold = Counter(str(len(c["canonical_gold"])) for c in candidates if c["question_type"] == "multi_select")
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update({
        "status": "PRESENTATION_ONLY_CANDIDATE_VIEW_NO_TRAINING_EXECUTED",
        "presentation_version": ALGORITHM,
        "seed_namespace": args.seed,
        "candidates": new_candidates,
        "source_package": str(source),
        "source_frontier_manifest_sha256": sha_file(source / "FRONTIER_MANIFEST.json"),
        "semantic_batch_or_selection_changed": False,
        "weights_changed": False,
        "replay_mixed": False,
    })
    with (output / "frontier_candidates.jsonl").open("x", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "FRONTIER_MANIFEST.json", new_manifest)
    write_json(output / "PRESENTATION_MAPPING.json", {"message_hash_serialization": "json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')", "records": mappings})
    audit = {
        "status": "PASS_PRESENTATION_INVARIANTS",
        "created_at_utc": now(), "rows": len(rows), "type_counts": dict(types), "multi_gold_counts": dict(sorted(multigold.items())),
        "candidate_id_order_unchanged": [c["candidate_id"] for c in candidates] == [c["candidate_id"] for c in new_candidates],
        "content_multisets_unchanged": len(mappings), "correct_content_sets_or_ordered_sequences_unchanged": len(mappings),
        "system_messages_unchanged": len(mappings), "three_role_rows": len(mappings), "single_block_replacements": len(mappings),
        "selection_eligibility_and_weights_unchanged": True,
        "before": pattern_audit(mappings, "old_gold"), "after": pattern_audit(mappings, "new_gold"),
        "seed_search_performed": False, "one_row_per_source_question": True,
        "note": "The after distribution is descriptive. There is no rejection, seed search, or promise of exact label balance; all export permutations are accepted.",
    }
    write_json(output / "LABEL_PATTERN_AUDIT.json", audit)
    require(snapshot(source) == before, "Source package changed during export")
    require(sha_file(pool_path) == pool_hash, "Pool changed during export")
    write_json(output / "INPUT_BINDINGS.json", {"source_package": str(source), "source_package_before": before, "source_package_after": snapshot(source), "all_source_package_hashes_unchanged": True, "pool_path": str(pool_path), "pool_sha256_before": pool_hash, "pool_sha256_after": sha_file(pool_path), "original_complete_manifest_sha256": sha_file(source / "COMPLETE_MANIFEST.json")})
    readme = """# Presentation-only training candidate view

The original sealed candidate package is retained for provenance. This view is
the presentation entry point for later training candidates: one row for every
source candidate, with the same eligibility, order, weight, text content, and
correct content set or chronological Gold text sequence. No replay is mixed.

Only the option/item block and assistant label answer are relabeled. The system
message, supplied dialogue, question, and output instruction remain byte-for-byte
identical. Labels remain A-F or I1-I6. Multi-select answers are sorted by their
new labels; ordering answers retain the original correct content sequence.

The fixed namespace and candidate ID seed independent deterministic SHA256
Fisher-Yates shuffles. CONFIG_FREEZE.json was written before any new permutation
or after-distribution calculation. There is one seed, no seed search, rejection
of unattractive distributions, semantic revision, new model call, or training.
This is a presentation version, not a new semantic batch or preregistration.
Training benefit has not been measured or established.

FRONTIER_MANIFEST.json updates canonical_gold/messages_sha256 for this view.
source_canonical_gold/source_messages_sha256 and the unchanged historical
solver_question_statistics preserve source provenance; historical solver
statistics still use their source labels and randomized displays. The native
solver evaluation already randomized display positions; this export does not
revise either model's completed evaluation.

PRESENTATION_MAPPING.json records both directions of each label map, exact old
and new Gold, exact option/item text, and message hashes. LABEL_PATTERN_AUDIT.json
describes before/after label patterns and invariant checks. INPUT_BINDINGS.json
verifies every original package file and the pool remained unchanged.
COMPLETE_MANIFEST.json binds every export file except itself and the exporter.

Reproduce in a new, nonexistent output directory:

    python export_presentation.py --source-package SOURCE_PACKAGE --pool POOL.json --output NEW_DIRECTORY --seed v107_training_presentation_v1

The tool requires a source FRONTIER_MANIFEST.json, frontier_candidates.jsonl and
COMPLETE_MANIFEST.json plus an evaluation pool with canonical options/items and
Gold. It refuses any existing output directory and writes no input files.
"""
    with (output / "README.md").open("x", encoding="utf-8") as handle:
        handle.write(readme.replace("--seed v107_training_presentation_v1", "--seed " + args.seed))
    complete = {"status": "PRESENTATION_ONLY_TRAINING_VIEW_SEALED", "created_at_utc": now(), "files": snapshot(output), "self_hash_excluded": True, "exporter": {"path": str(tool), "sha256": sha_file(tool)}, "training_started": False, "new_model_calls": 0, "replay_mixed": False}
    write_json(output / "COMPLETE_MANIFEST.json", complete)
    print(json.dumps({"status": complete["status"], "output": str(output), "rows": len(rows), "type_counts": dict(types), "multi_gold_counts": dict(sorted(multigold.items())), "ordering_before_max_same_sequence": audit["before"].get("ordering", {}).get("maximum_identical_gold_label_sequence_count"), "ordering_after_max_same_sequence": audit["after"].get("ordering", {}).get("maximum_identical_gold_label_sequence_count")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, help="Fixed namespace, frozen before examining new label distributions")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
