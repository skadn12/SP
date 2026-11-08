"""Gold-aware export only; inference never imports this module."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def main():
    freeze = json.loads((ROOT / "QUESTION_FREEZE.json").read_text())
    assert freeze["gates"]["content_review_status"] == "PASS"
    public, private = [], []
    for item in freeze["questions"]:
        path = Path(item["question_file"]["path"])
        assert binding(path)["sha256"] == item["question_file"]["sha256"]
        q = json.loads(path.read_text())
        field = "options" if q["question_type"] == "multi_select" else "items"
        order = q["display_order"]
        assert len(order) == len(set(order)) == len(q[field]) and set(order) == set(q[field])
        assert order != list(q[field])
        mapping = {chr(65+i): key for i, key in enumerate(order)}
        reverse = {key: label for label, key in mapping.items()}
        row = {key: q[key] for key in ("candidate_id", "root_id", "question_type", "task_operation", "question", "source_binding")}
        row["display"] = {label: q[field][key] for label, key in mapping.items()}
        public.append(row)
        gold = [reverse[key] for key in q["gold"]]
        private.append({"candidate_id": q["candidate_id"], "question_type": q["question_type"],
            "display_to_canonical": mapping, "canonical_to_display": reverse,
            "canonical_gold": q["gold"], "display_gold": sorted(gold) if q["question_type"] == "multi_select" else gold,
            "question_binding": binding(path)})
    assert len(public) == len(private) == 14
    save(HERE / "inputs/PUBLIC_QUESTIONS.json", {"questions": public})
    private_path = HERE / "inputs/PRIVATE_SCORING.json"
    save(private_path, {"questions": private})
    private_path.chmod(0o600)
    save(HERE / "inputs/EXPORT_FREEZE.json", {"requests": 14,
        "selection": "Every v5 admitted question, original frozen display, before solver results",
        "exporter_gold_aware": True, "inference_gold_aware": False,
        "bindings": [binding(HERE / "inputs/PUBLIC_QUESTIONS.json"), binding(ROOT / "QUESTION_FREEZE.json"),
            binding(ROOT / "questions/CONTENT_REVIEW.json"), binding(HERE / "PROTOCOL_ZH.md"), binding(Path(__file__))],
        "private_binding": binding(private_path)})
    print("EXPORTED: 14 public questions and private scoring; no model call")

if __name__ == "__main__":
    main()
