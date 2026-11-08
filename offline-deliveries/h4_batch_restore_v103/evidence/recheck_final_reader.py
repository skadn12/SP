"""Resolve the initial reader's EOF separator false failures; no content edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

DELIVERY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    initial_path = DELIVERY / "evidence/FINAL_REPORT_SECOND_READER.json"
    initial_binding = bind(initial_path)
    initial = read(initial_path)
    expected_initial_failures = [f"readable_exact_ordered_turns:cc-train-episode-{n}" for n in (14304, 24719, 38601)]
    assert initial["status"] == "FAIL"
    assert initial["failures"] == expected_initial_failures
    assert len(initial["checks"]) == 49
    assert sum(c["status"] == "PASS" for c in initial["checks"]) == 46
    changed_bindings = [b for b in initial["bindings"] if bind(b["path"]) != b]
    assert not changed_bindings, changed_bindings
    outcomes = []
    for n in (14304, 24719, 38601):
        source_path = DELIVERY / f"sources/root{n}_SOURCE.json"
        review_path = DELIVERY / f"inputs/root{n}_SOURCE_REVIEW.json"
        readable_path = DELIVERY / f"sources/root{n}_READABLE.md"
        source, review = read(source_path), read(review_path)
        root_id = f"cc-train-episode-{n}"
        lines = [f"# {root_id} — {review['decision']}", "", "此文件逐字展示冻结累计对话；来源判断见配套 SOURCE_REVIEW.json。", ""]
        for turn in source["turns"]:
            lines.extend([f"**{turn['turn_ref']} · {turn['speaker']}**", "", turn["text"], ""])
        expected = "\n".join(lines).encode("utf-8")
        actual_bytes = readable_path.read_bytes()
        assert actual_bytes == expected, f"Full renderer byte mismatch for {root_id}"
        actual = actual_bytes.decode("utf-8")
        headers = list(re.finditer(r"^\*\*([^\n]+) · ([^\n]+)\*\*\n\n", actual, re.MULTILINE))
        assert len(headers) == len(source["turns"])
        for i, (match, turn) in enumerate(zip(headers, source["turns"])):
            assert match.group(1) == turn["turn_ref"] and match.group(2) == turn["speaker"]
            last = i == len(headers) - 1
            block_end = len(actual) if last else headers[i + 1].start()
            separator = "\n" if last else "\n\n"
            body = actual[match.end():block_end]
            assert body == turn["text"] + separator, (root_id, turn["turn_ref"])
        final = source["turns"][-1]
        outcomes.append({"root_id": root_id, "status": "PASS", "turns": len(headers),
                         "all_turn_refs_speakers_and_text_exact_and_ordered": True,
                         "full_renderer_bytes_equal": True,
                         "last_turn_ref": final["turn_ref"],
                         "initial_false_expectation_tail_repr": repr((final["text"] + "\n\n")[-100:]),
                         "actual_correct_eof_tail_repr": repr(actual[-99:]),
                         "source": bind(source_path), "readable": bind(readable_path)})
    before_path = DELIVERY / "PRESERVATION_BEFORE.json"
    before = read(before_path)
    assert len(before["files"]) == 4127
    historical_changes = [b["path"] for b in before["files"] if not Path(b["path"]).is_file()
                          or bind(b["path"])["sha256"] != b["sha256"]
                          or Path(b["path"]).stat().st_size != b["bytes"]]
    assert not historical_changes, historical_changes[:5]
    receipt = {
        "version": "v103_final_report_second_reader_eof_recheck_v1", "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_receipt_preserved": initial_binding,
        "correction": "The initial reader required two separator newlines after every turn. The existing renderer joins an empty final list element, leaving one newline at EOF. Only the three last-turn EOF separators caused the initial FAIL; no source text differed. The recheck compares the complete renderer output byte-for-byte and independently parses every turn header, speaker and exact text, with two separator newlines between turns and one at EOF.",
        "initial_passed_checks_retained": 46, "initial_false_failures_corrected": expected_initial_failures,
        "effective_final_check_count": 49, "effective_final_pass_count": 49,
        "all_initial_bound_files_unchanged": True, "initial_bound_files_rechecked": len(initial["bindings"]),
        "initial_bindings": initial["bindings"], "readable_sources": outcomes,
        "historical_protected_files_rechecked": 4127, "historical_changes": historical_changes,
        "preservation_before": bind(before_path), "recheck_script": bind(__file__),
        "source_question_answer_report_or_export_edits": [],
        "model_calls": 0, "http_calls": 0,
        "recheck_implementation_note": "First script execution stopped before writing a receipt because its preservation comparison compared a path/hash dictionary to a path/hash/bytes dictionary. The comparison was corrected to explicit SHA256 and size fields; no historical or delivery content was changed.",
        "limitation": initial["limitation"],
    }
    output = DELIVERY / "evidence/FINAL_REPORT_SECOND_READER_RECHECK.json"
    with output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    assert bind(initial_path) == initial_binding
    print(json.dumps({"status": "PASS", "receipt": bind(output), "initial_receipt": initial_binding,
                      "checks": 49, "historical_files_unchanged": 4127, "source_turns": [r["turns"] for r in outcomes]}))


if __name__ == "__main__":
    main()
