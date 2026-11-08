"""Commit one reviewed scene; never edit a source prefix or accepted artifact."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

P = Path(__file__).resolve().parent
SEQUENCE = [f"T4R:{route}:S{i:02d}" for route in "AB" for i in range(1, 4)]


def read(p):
    return json.loads(Path(p).read_text())


def binding(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def write_new(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        f.write(json.dumps(d, ensure_ascii=False, indent=2) + "\n")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--draft", type=Path, required=True)
    a.add_argument("--review", type=Path, required=True)
    args = a.parse_args()
    draft, review = read(args.draft), read(args.review)
    assert review["status"] == "PASS"
    assert review["draft"] == binding(args.draft)
    root, scene = draft["root_id"], draft["scene_ref"]
    assert root in ["cc-train-episode-104814", "cc-train-episode-109164"]
    index = SEQUENCE.index(scene)
    outdir = P / "extensions" / root / "accepted"
    prior = P / "inputs" / f"{root}.json" if index == 0 else outdir / (SEQUENCE[index - 1].replace(":", "_") + "_SOURCE.json")
    assert draft["input_source"] == binding(prior)
    prefix = read(prior)
    turns = draft["turns"]
    assert 12 <= len(turns) <= 16
    assert {t["speaker"] for t in turns} <= {t["speaker"] for t in prefix["turns"]}
    for i, t in enumerate(turns, 1):
        assert t["turn_ref"] == f"{scene}:T{i:03d}"
        assert isinstance(t["text"], str) and t["text"].strip()
    refs = [t["turn_ref"] for t in prefix["turns"] + turns]
    assert len(refs) == len(set(refs))
    reservation = read(P / "SOURCE_RESERVATIONS.json")
    if ":B:" in scene:
        matched = [r for r in reservation["reservations"] if r["root_id"] == root and r["scene_ref"] == scene]
        assert len(matched) == 1 and draft["native_event"] == matched[0]["native_record"]
    else:
        assert draft.get("native_event") is None
    original = read(P / "inputs" / f"{root}.json")
    history = prefix.get("targeted_extension_scenes", [])
    result = {"schema_version": "targeted_four_root_source_v1", "root_id": root,
              "frozen_prefix": binding(P / "inputs" / f"{root}.json"),
              "targeted_extension_scenes": history + [{"scene_ref": scene, "draft": binding(args.draft), "review": binding(args.review), "native_event": draft.get("native_event")}],
              "turns": prefix["turns"] + turns}
    assert result["turns"][:len(original["turns"])] == original["turns"]
    out = outdir / (scene.replace(":", "_") + "_SOURCE.json")
    write_new(out, result)
    log = {"status": "ACCEPTED", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "root_id": root, "scene_ref": scene, "source": binding(out), "draft": binding(args.draft), "review": binding(args.review),
           "prefix_turn_count": len(original["turns"]), "new_turn_count": len(turns),
           "total_turn_count": len(result["turns"]), "complete_extension": index == 5}
    write_new(outdir / (scene.replace(":", "_") + "_ACCEPTANCE.json"), log)
    if index == 5:
        final = P / "sources" / f"{root}.json"
        assert not final.exists()
        final.write_bytes(out.read_bytes())
    print(json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    main()
