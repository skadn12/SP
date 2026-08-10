"""Compare frozen Memorax recall with authoritative transcript state histories.

This module is intentionally a fixed four-event diagnostic. It is not a state
resolver, does not infer new event chains, and never calls a live service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_DIR = Path(__file__).resolve().parent
FROZEN_RUN_ROOT = Path(
    "/root/autodl-tmp/gxy-selfplay/restart-n5-runs/"
    "conversation_chronicles_episode_146700_simple_append_memorax_"
    "fullr1_durable_devdiv_local_anchor_span_probe_v1_live_1"
)
PRE_OVERVIEW_PATH = FROZEN_RUN_ROOT / "memory/pre_generation_overview.json"
POST_OVERVIEW_PATH = FROZEN_RUN_ROOT / "memory/post_generation_overview.json"
EXPANDED_DIALOGUE_PATH = FROZEN_RUN_ROOT / "expanded_dialogue.json"

PROTECTED_SOURCE_DIRS = (
    REPO_ROOT
    / "scripts/current_method/restart_n5_spell/simple_append_memorax_v1",
    REPO_ROOT
    / "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_fullr1_durable_devdiv_probe_v1",
    REPO_ROOT
    / "scripts/current_method/restart_n5_spell/"
    "simple_append_memorax_v1_qa_fact_first_probe_v1",
)


# The four event lines, their state descriptions, and their semantic comparison
# outcomes are deliberately fixed by offline human audit. The program only binds
# these decisions to frozen source material and checks their reproducibility.
EVENT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "event": "soccer",
        "overview_terms": ("soccer", "football", "\u8db3\u7403", "\u8e22\u7403"),
        "state_history": (
            {
                "stage": "earlier",
                "state": "A missed a kick and felt disappointed about the miss.",
                "source_positions": (1, 3),
            },
            {
                "stage": "earlier",
                "state": (
                    "A and B recalled their former after-work practice, and B "
                    "suggested starting again."
                ),
                "source_positions": (5, 6, 7),
            },
            {
                "stage": "latest",
                "state": (
                    "A accepted a renewed daily after-work practice plan; B made "
                    "it concrete by setting the start for tomorrow and bringing the ball."
                ),
                "source_positions": (68, 69, 72),
            },
        ),
        "assessment": {
            "memory_fact_coverage": "sufficient",
            "latest_state_fact_present_in_memorax": True,
            "memory_latest_update_missing": False,
            "outdated_or_earlier_states_also_present": True,
            "memory_event_identity_unstable": True,
            "memory_summary_state_conflict": False,
            "state_resolution_layer_needed": True,
            "memory_alone_supports_latest_state_resolution": False,
            "memory_plus_transcript_supports_latest_state_resolution": True,
            "case": "C",
            "diagnosis": (
                "The latest restart decision is recalled, but it is split across "
                "two near-duplicate event keys and mixed with earlier soccer facts."
            ),
        },
    },
    {
        "event": "report_mistake",
        "overview_terms": ("\u62a5\u544a\u9519\u8bef",),
        "state_history": (
            {
                "stage": "earlier",
                "state": (
                    "A placed or attempted an urgent call and was waiting for the "
                    "boss to answer; A still needed to disclose the report mistake."
                ),
                "source_positions": (48, 50),
            },
            {
                "stage": "earlier",
                "state": "A committed to taking responsibility and making the mistake right.",
                "source_positions": (56,),
            },
            {
                "stage": "latest",
                "state": (
                    "A said they would call the boss now and explain what happened; "
                    "the transcript never confirms that the explanation was delivered."
                ),
                "source_positions": (96, 110),
            },
        ),
        "assessment": {
            "memory_fact_coverage": "partial",
            "latest_state_fact_present_in_memorax": True,
            "memory_latest_update_missing": False,
            "outdated_or_earlier_states_also_present": True,
            "memory_event_identity_unstable": False,
            "memory_summary_state_conflict": True,
            "state_resolution_layer_needed": True,
            "memory_alone_supports_latest_state_resolution": False,
            "memory_plus_transcript_supports_latest_state_resolution": True,
            "case": "C",
            "diagnosis": (
                "Memorax includes the decision to contact the boss, but the same "
                "summary also overstates the disclosure/apology as completed. The "
                "authoritative transcript ends at an imminent intention."
            ),
        },
    },
    {
        "event": "health_doctor_work_coverage",
        "overview_terms": ("\u751f\u75c5\u8bf7\u5047", "\u53d1\u70e7", "doctor", "temperature"),
        "state_history": (
            {
                "stage": "earlier",
                "state": (
                    "A had an elevated temperature and headache, and paracetamol "
                    "had not helped much."
                ),
                "source_positions": (34, 36),
            },
            {
                "stage": "latest",
                "state": (
                    "A planned to see a doctor after work and asked B to cover the "
                    "shift; B accepted. No doctor visit is shown as completed."
                ),
                "source_positions": (37, 38, 39),
            },
        ),
        "assessment": {
            "memory_fact_coverage": "sufficient",
            "latest_state_fact_present_in_memorax": True,
            "memory_latest_update_missing": False,
            "outdated_or_earlier_states_also_present": True,
            "memory_event_identity_unstable": False,
            "memory_summary_state_conflict": False,
            "state_resolution_layer_needed": True,
            "memory_alone_supports_latest_state_resolution": True,
            "memory_plus_transcript_supports_latest_state_resolution": True,
            "case": "A",
            "diagnosis": (
                "One stable event entry contains both the symptoms and the later "
                "doctor/coverage plan. Its latest plan can be recovered cautiously "
                "without claiming that the doctor visit occurred."
            ),
        },
    },
    {
        "event": "new_cashier",
        "overview_terms": ("new cashier", "\u65b0\u6536\u94f6", "\u65b0\u7684\u6536\u94f6"),
        "state_history": (
            {
                "stage": "earlier",
                "state": (
                    "A had heard that the store had a new cashier; B had not heard, "
                    "and A had not worked with her."
                ),
                "source_positions": (9, 10, 11),
            },
            {
                "stage": "latest",
                "state": (
                    "The cashier remained discussed rather than directly observed: "
                    "B had not met her, and A planned to talk to her later. No contact "
                    "is shown as completed."
                ),
                "source_positions": (80, 81, 88),
            },
        ),
        "assessment": {
            "memory_fact_coverage": "insufficient",
            "latest_state_fact_present_in_memorax": False,
            "memory_latest_update_missing": True,
            "outdated_or_earlier_states_also_present": False,
            "memory_event_identity_unstable": False,
            "memory_summary_state_conflict": False,
            "state_resolution_layer_needed": False,
            "memory_alone_supports_latest_state_resolution": False,
            "memory_plus_transcript_supports_latest_state_resolution": True,
            "case": "B",
            "diagnosis": (
                "Neither overview recalls the new-cashier event or A's later plan "
                "to make contact. This is a recall/write coverage gap, not merely "
                "a missing latest-state marker."
            ),
        },
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, *, ignore_caches: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if ignore_caches and (
            "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        digest.update(f"{_sha256_file(path)}  {relative.as_posix()}\n".encode("utf-8"))
    return digest.hexdigest()


def _integrity_snapshot() -> dict[str, Any]:
    return {
        "frozen_run_tree_sha256": _tree_sha256(FROZEN_RUN_ROOT, ignore_caches=False),
        "frozen_inputs": {
            str(PRE_OVERVIEW_PATH): _sha256_file(PRE_OVERVIEW_PATH),
            str(POST_OVERVIEW_PATH): _sha256_file(POST_OVERVIEW_PATH),
            str(EXPANDED_DIALOGUE_PATH): _sha256_file(EXPANDED_DIALOGUE_PATH),
        },
        "protected_source_manifests": {
            str(path): _tree_sha256(path, ignore_caches=True)
            for path in PROTECTED_SOURCE_DIRS
        },
    }


def _overview_entries(overview: dict[str, Any], phase: str) -> Iterable[dict[str, str]]:
    for section in ("people", "locations", "events", "items", "themes"):
        records = overview.get(section, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            key = str(record.get("canonical_key", ""))
            display_key = str(record.get("display_key", ""))
            content = record.get("content", [])
            if isinstance(content, str):
                content = [content]
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, str):
                    yield {
                        "phase": phase,
                        "section": section,
                        "canonical_key": key,
                        "display_key": display_key,
                        "content": item,
                    }


def _match_memory_facts(
    entries: Iterable[dict[str, str]], terms: tuple[str, ...]
) -> list[dict[str, str]]:
    lowered_terms = tuple(term.casefold() for term in terms)
    matches = []
    for entry in entries:
        haystack = " ".join(
            (entry["canonical_key"], entry["display_key"], entry["content"])
        ).casefold()
        if any(term in haystack for term in lowered_terms):
            matches.append(entry)
    return matches


def _materialize_state_history(
    turns: list[dict[str, Any]], history: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    materialized = []
    for state in history:
        evidence = []
        for position in state["source_positions"]:
            if position < 1 or position > len(turns):
                raise ValueError(f"source position out of range: {position}")
            turn = turns[position - 1]
            evidence.append(
                {
                    "position": position,
                    "turn_ref": turn.get("turn_ref"),
                    "speaker": turn.get("speaker"),
                    "source_text": turn.get("text"),
                }
            )
        materialized.append(
            {
                "stage": state["stage"],
                "state": state["state"],
                "evidence": evidence,
            }
        )
    return materialized


def build_diagnostic() -> dict[str, Any]:
    pre = _load_json(PRE_OVERVIEW_PATH)
    post = _load_json(POST_OVERVIEW_PATH)
    dialogue = _load_json(EXPANDED_DIALOGUE_PATH)
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("expanded dialogue has no turns")
    if dialogue.get("root_id") != "cc-train-episode-146700":
        raise ValueError("unexpected frozen root")

    entries = list(_overview_entries(pre, "pre_generation")) + list(
        _overview_entries(post, "post_generation")
    )
    event_results = []
    for spec in EVENT_SPECS:
        history = _materialize_state_history(turns, spec["state_history"])
        latest = [state for state in history if state["stage"] == "latest"]
        if len(latest) != 1:
            raise ValueError(f"{spec['event']} must have exactly one latest state")
        event_results.append(
            {
                "event": spec["event"],
                "memorax_recalled_facts": _match_memory_facts(
                    entries, spec["overview_terms"]
                ),
                "transcript_state_history": history,
                "transcript_latest_state": latest[0],
                **spec["assessment"],
                "assessment_basis": "fixed_human_offline_comparison",
            }
        )

    present_count = sum(
        result["latest_state_fact_present_in_memorax"] for result in event_results
    )
    case_counts = {
        case: sum(result["case"] == case for result in event_results)
        for case in ("A", "B", "C")
    }
    return {
        "diagnostic_version": "memorax_state_resolution_diagnostic_v1",
        "scope": {
            "root_id": dialogue["root_id"],
            "turn_count": len(turns),
            "events": [spec["event"] for spec in EVENT_SPECS],
            "frozen_run_root": str(FROZEN_RUN_ROOT),
            "api_calls": 0,
            "state_resolver_implemented": False,
            "state_ledger_implemented": False,
        },
        "events": event_results,
        "summary": {
            "event_count": len(event_results),
            "latest_update_present_count": present_count,
            "case_counts": case_counts,
            "memory_contains_update_facts": "partial",
            "memory_alone_supports_latest_state_resolution": False,
            "memory_plus_transcript_supports_latest_state_resolution": True,
            "primary_problem": "mixed",
            "dominant_component": "state_resolution_and_event_identity",
            "secondary_component": "memory_fact_coverage",
            "conclusion": (
                "Memorax recalls the latest update for three of four fixed event "
                "lines, but duplicate/conflicting summaries prevent reliable "
                "Memory-only latest-state resolution. The transcript resolves all "
                "four lines, while the new-cashier update is missing from recall."
            ),
        },
        "integrity": {"before": _integrity_snapshot()},
    }


def _render_markdown(result: dict[str, Any]) -> str:
    rows = []
    for event in result["events"]:
        latest = event["transcript_latest_state"]["state"].replace("|", "\\|")
        rows.append(
            "| {event} | {latest} | {present} | {old} | Case {case}: {diagnosis} |".format(
                event=event["event"],
                latest=latest,
                present="yes" if event["latest_state_fact_present_in_memorax"] else "no",
                old=(
                    "yes"
                    if event["outdated_or_earlier_states_also_present"]
                    else "no"
                ),
                case=event["case"],
                diagnosis=event["diagnosis"].replace("|", "\\|"),
            )
        )

    details = []
    for event in result["events"]:
        memory_lines = []
        for fact in event["memorax_recalled_facts"]:
            memory_lines.append(
                f"- `{fact['phase']}/{fact['section']}/{fact['canonical_key']}`: "
                f"{fact['content']}"
            )
        if not memory_lines:
            memory_lines.append("- No matching recalled fact was found.")
        history_lines = []
        for state in event["transcript_state_history"]:
            refs = ", ".join(
                f"{item['position']} (`{item['turn_ref']}`)"
                for item in state["evidence"]
            )
            history_lines.append(f"- **{state['stage']}**: {state['state']} [{refs}]")
            history_lines.extend(
                f"  - {item['speaker']}: {item['source_text']}"
                for item in state["evidence"]
            )
        details.append(
            "\n".join(
                [
                    f"## {event['event']}",
                    "",
                    "### Memorax recalled facts",
                    "",
                    *memory_lines,
                    "",
                    "### Authoritative transcript history",
                    "",
                    *history_lines,
                    "",
                    f"**Diagnosis:** Case {event['case']}. {event['diagnosis']}",
                    "",
                    f"- Memory fact coverage: `{event['memory_fact_coverage']}`",
                    "- State-resolution layer needed: "
                    f"`{str(event['state_resolution_layer_needed']).lower()}`",
                    "- Memory alone resolves latest state: "
                    f"`{str(event['memory_alone_supports_latest_state_resolution']).lower()}`",
                    "- Memory + transcript resolves latest state: "
                    f"`{str(event['memory_plus_transcript_supports_latest_state_resolution']).lower()}`",
                ]
            )
        )

    integrity = result["integrity"]
    frozen_input_lines = [
        f"- Frozen input `{path}`: `{sha}`"
        for path, sha in integrity["after"]["frozen_inputs"].items()
    ]
    protected_source_lines = [
        f"- Protected source `{path}`: `{sha}`"
        for path, sha in integrity["after"]["protected_source_manifests"].items()
    ]
    return "\n".join(
        [
            "# State Resolution Diagnostic",
            "",
            "This is a fixed, zero-API comparison for `episode-146700`. It does not "
            "implement a resolver or write any state back to Memorax.",
            "",
            "## Summary",
            "",
            "| Event | Latest transcript state | Present in Memorax? | Earlier states also present? | Diagnosis |",
            "|---|---|---:|---:|---|",
            *rows,
            "",
            f"- `memory_contains_update_facts = {result['summary']['memory_contains_update_facts']}`",
            "- `memory_alone_supports_latest_state_resolution = false`",
            "- `memory_plus_transcript_supports_latest_state_resolution = true`",
            f"- `primary_problem = {result['summary']['primary_problem']}`",
            f"- Dominant component: `{result['summary']['dominant_component']}`",
            f"- Secondary component: `{result['summary']['secondary_component']}`",
            f"- Latest update recalled: `{result['summary']['latest_update_present_count']}/4`",
            "",
            result["summary"]["conclusion"],
            "",
            "The result is mixed: a thin transcript-assisted resolver would help the "
            "soccer/report lines, but it cannot recover a missing new-cashier recall "
            "from Memory alone. No State Ledger or write-back is justified by this probe.",
            "",
            *details,
            "",
            "## Integrity And Request Discipline",
            "",
            f"- API calls: `{result['scope']['api_calls']}`",
            "- Expansion / Planner / Realizer / QA / Solver calls: `0`",
            "- State Resolver implemented: `false`",
            "- State Ledger implemented: `false`",
            "- Frozen run SHA before: "
            f"`{integrity['before']['frozen_run_tree_sha256']}`",
            "- Frozen run SHA after: "
            f"`{integrity['after']['frozen_run_tree_sha256']}`",
            f"- Frozen run unchanged: `{str(integrity['unchanged']).lower()}`",
            *frozen_input_lines,
            *protected_source_lines,
            "",
        ]
    )


def run_diagnostic(output_dir: Path = MODULE_DIR) -> dict[str, Any]:
    result = build_diagnostic()
    output_dir.mkdir(parents=True, exist_ok=True)
    result["integrity"]["after"] = _integrity_snapshot()
    result["integrity"]["unchanged"] = (
        result["integrity"]["before"] == result["integrity"]["after"]
    )
    if not result["integrity"]["unchanged"]:
        raise RuntimeError("protected frozen inputs or source manifests changed")

    json_path = output_dir / "state_resolution_diagnostic.json"
    md_path = output_dir / "STATE_RESOLUTION_DIAGNOSTIC.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR)
    args = parser.parse_args()
    result = run_diagnostic(args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
