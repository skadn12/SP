"""Gold-blind display permutation; never change propositions, truth or canonical IDs.

Canonical pool rows remain the provenance record. These views are the default
human question display and future training rendering input, still provisional.
Changing the display does not repair all-true questions or leaking stems.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

VERSION = "v53_gold_blind_display_export_v1"
DEFAULT_SEED = "h4-v53-display-20261011"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def display_view(question, *, seed=DEFAULT_SEED):
    kind = question["question_type"]
    if kind not in ("multi_select", "ordering"):
        raise ValueError("unsupported_question_type")
    field = "options" if kind == "multi_select" else "items"
    labels = list("ABCDEF") if kind == "multi_select" else ["I" + str(i) for i in range(1, 7)]
    original = question[field]
    if not isinstance(original, dict) or set(original) != set(labels):
        raise ValueError("six_canonical_labels_required")
    if not all(isinstance(t, str) and t.strip() for t in original.values()):
        raise ValueError("invalid_statement_text")
    if len(set(original.values())) != 6:
        raise ValueError("duplicate_statement_text")
    gold = question["gold"]
    if (not isinstance(gold, list) or len(gold) != len(set(gold))
            or not set(gold) <= set(labels) or not gold
            or (kind == "ordering" and set(gold) != set(labels))):
        raise ValueError("invalid_canonical_gold")
    # Sampling is based only on public identity/content and seed. It never
    # searches for an answer pattern or consumes Gold, evidence or scores.
    public = {"candidate_id": question["candidate_id"], "question_type": kind,
              "question": question["question"], "statements": sorted(original.values()),
              "seed": seed}
    values = sorted(original.values())
    random.Random(int(digest(public), 16)).shuffle(values)
    old_by_text = {text: label for label, text in original.items()}
    display_to_original = {label: old_by_text[text] for label, text in zip(labels, values)}
    original_to_display = {v: k for k, v in display_to_original.items()}
    mapped = [original_to_display[label] for label in gold]
    if kind == "multi_select":
        mapped.sort(key=labels.index)
    return {"display_version": VERSION, "display_seed": seed,
            "candidate_id": question["candidate_id"], "root_id": question["root_id"],
            "question_type": kind, "question": question["question"],
            field: dict(zip(labels, values)), "gold": mapped,
            "display_to_original": display_to_original,
            "original_to_display": original_to_display,
            "canonical_question_sha256": digest(question),
            "answer_count": len(gold), "training_allowed": False,
            "presentation_only": True,
            "content_issues_not_repaired_by_shuffle": True}


def replay_view(question, view):
    if view != display_view(question, seed=view["display_seed"]):
        raise ValueError("display_export_or_gold_mapping_changed")
    return True


def display_stats(views):
    multi = [q for q in views if q["question_type"] == "multi_select"]
    ordering = [q for q in views if q["question_type"] == "ordering"]
    return {"multi_select": len(multi),
            "multi_correct_count_histogram": dict(sorted(Counter(len(q["gold"]) for q in multi).items())),
            "multi_display_gold_patterns": dict(Counter(",".join(q["gold"]) for q in multi)),
            "multi_correct_position_counts": {k: sum(k in q["gold"] for q in multi) for k in "ABCDEF"},
            "ordering_display_gold_patterns": dict(Counter(",".join(q["gold"]) for q in ordering)),
            "multi_all_six": sum(len(q["gold"]) == 6 for q in multi),
            "multi_outside_two_to_four": sum(not 2 <= len(q["gold"]) <= 4 for q in multi),
            "ordering": len(ordering),
            "ordering_display_identity": sum(q["gold"] == list(q["items"]) for q in ordering)}


def markdown(views, title="v53 打乱呈现的候选题"):
    lines = ["# " + title, "",
             "这是按内容同步映射答案的呈现版本；原始题面、Gold 和来源保留在冻结候选记录中。",
             "仅打乱选项不解决全选、错误陈述或题干泄露顺序。所有题仍为未正式准入的候选。", ""]
    for q in views:
        field = "options" if q["question_type"] == "multi_select" else "items"
        lines.extend(["## " + q["candidate_id"], "", q["question"], ""])
        lines.extend(f"- {label}: {text}" for label, text in q[field].items())
        lines.extend(["", "呈现 Gold：" + json.dumps(q["gold"], ensure_ascii=False), ""])
        if q["question_type"] == "multi_select" and not 2 <= q["answer_count"] <= 4:
            lines.extend(["答案数量不符合新 2–4 项正确的出题规范；未修改旧题真假。", ""])
        lines.extend(["原编号映射：" + json.dumps(q["display_to_original"], ensure_ascii=False), ""])
    return "\n".join(lines)


def write_views(folder, questions, *, seed=DEFAULT_SEED):
    folder = Path(folder)
    views = [display_view(q, seed=seed) for q in questions]
    for q, view in zip(questions, views):
        replay_view(q, view)
    with (folder / "DISPLAY_CANDIDATES.jsonl").open("x") as stream:
        for view in views:
            stream.write(json.dumps(view, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (folder / "SELECTED_QUESTIONS.md").open("x") as stream:
        stream.write(markdown(views) + "\n")
    return views
