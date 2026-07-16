import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.current_method import audit_stage2_canonical_rebuild as audit


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class CanonicalRebuildAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source.jsonl"
        self.rebuild = self.root / "rebuild"
        self.rebuild.mkdir()
        self.expectations = audit.AuditExpectations(
            graph_count=1,
            parent_count=1,
            canonical_event_count=2,
            source_event_count=2,
            preexcluded_source_event_count=0,
            semantic_audit_exclusion_count=0,
            fallback_event_count=0,
            ineligible_source_row_count=0,
            eligible_rejection_count=0,
            minimum_events_per_row=2,
            lineage_count=1,
        )
        self._write_valid_fixture()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _canonical_events() -> list[dict]:
        return [
            {
                "event_id": "rebuilt:event:1",
                "entity": "Maria",
                "slot": "donated_item",
                "value": "her old car",
                "status": "active",
                "time_index": 1,
                "evidence_turn_ids": ["D1:1"],
                "required_for_dialogue": True,
                "provenance": {
                    "source_event_id": "old:event:1",
                    "supporting_span": "Maria donated her old car",
                },
            },
            {
                "event_id": "rebuilt:event:2",
                "entity": "John",
                "slot": "attended_class",
                "value": "a yoga class",
                "status": "active",
                "time_index": 2,
                "evidence_turn_ids": ["D1:2"],
                "required_for_dialogue": True,
                "provenance": {
                    "source_event_id": "old:event:2",
                    "supporting_span": "John attended a yoga class",
                },
            },
        ]

    def _write_valid_fixture(self) -> None:
        source_row = {
            "sample_id": "source-sample-1",
            "split": "train",
            "training_use": "train_candidate",
            "dialogue": {
                "dialogue_hash": "source-dialogue-hash",
                "turns": [
                    {"turn_id": "D1:1", "speaker": "Maria", "text": "Maria donated her old car"},
                    {"turn_id": "D1:2", "speaker": "John", "text": "John attended a yoga class"},
                ],
            },
            "graph_update": {
                "events": [
                    {"event_id": "old:event:1"},
                    {"event_id": "old:event:2"},
                ]
            },
        }
        write_jsonl(self.source, [source_row])

        request = {
            "schema_version": audit.REBUILD_SCHEMA_VERSION,
            "request_id": "request-1",
            "source_sample_id": "source-sample-1",
            "source_line_number": 1,
            "source_event_count": 2,
            "fallback_event_count": 0,
            "events": [
                {
                    "source_event_id": "old:event:1",
                    "source_event_order": 1,
                    "reconstruction_source": "source_event_evidence_link",
                    "evidence": [
                        {"turn_id": "D1:1", "speaker": "Maria", "text": "Maria donated her old car"}
                    ],
                },
                {
                    "source_event_id": "old:event:2",
                    "source_event_order": 2,
                    "reconstruction_source": "source_event_evidence_link",
                    "evidence": [
                        {"turn_id": "D1:2", "speaker": "John", "text": "John attended a yoga class"}
                    ],
                },
            ],
            "skipped_events": [],
        }
        write_jsonl(self.rebuild / "extraction_requests.jsonl", [request])
        write_jsonl(self.rebuild / "ineligible_source_rows.jsonl", [])
        write_jsonl(
            self.rebuild / "semantic_audit_exclusions_applied.jsonl", []
        )
        write_jsonl(self.rebuild / "source_event_exclusions.jsonl", [])
        write_jsonl(self.rebuild / "rejected_events.jsonl", [])

        events = self._canonical_events()
        graph_row = {
            "schema_version": audit.REBUILD_SCHEMA_VERSION,
            "source_sample_id": "source-sample-1",
            "source_line_number": 1,
            "source_event_count": 2,
            "excluded_source_event_count": 0,
            "request_id": "request-1",
            "semantic_parent_graph": {
                "schema_version": "semantic_graph_v2",
                "graph_id": "canonical-graph-1",
                "parent_graph_id": "root-1",
                "events": events,
                "event_count": 2,
            },
            "coverage": {
                "complete_for_eligible_evidence": True,
                "old_graph_events_inherited": False,
                "legacy_semantic_labels_exposed_to_model": False,
            },
        }
        write_jsonl(self.rebuild / "stage2_semantic_graphs.jsonl", [graph_row])

        graph_hash = audit.sha256_json(events)
        turns = copy.deepcopy(source_row["dialogue"]["turns"])
        parent = {
            "schema_version": audit.PARENT_SCHEMA_VERSION,
            "sample_id": "parent-1",
            "source_id": "parent-1",
            "lineage_group": "conv-1",
            "dialogue": {"dialogue_hash": "clean-dialogue-hash", "turns": turns},
            "turns": copy.deepcopy(turns),
            "semantic_parent_graph": {
                "schema_version": "semantic_graph_v2",
                "graph_id": "parent-graph-1",
                "parent_graph_id": "root-1",
                "events": copy.deepcopy(events),
                "event_count": 2,
                "empty_parent_snapshot_allowed": False,
                "graph_sha256": graph_hash,
            },
            "source_snapshot": {
                "source_sample_id": "source-sample-1",
                "reconstructed_graph_sha256": graph_hash,
            },
            "metadata": {"source_sample_id": "source-sample-1"},
        }
        write_jsonl(self.rebuild / "stage3_parent_v3.jsonl", [parent])
        write_jsonl(self.rebuild / "stage3_parent_v3_rejected.jsonl", [])
        write_jsonl(self.rebuild / "stage3_parent_v3_audit.jsonl", [])

        summary = {
            "input_row_count": 1,
            "selected_request_count": 1,
            "eligible_request_count": 1,
            "eligible_selected_request_count": 1,
            "accepted_canonical_event_count": 2,
            "excluded_source_event_count": 0,
            "fallback_event_count": 0,
            "semantic_audit_exclusion_count": 0,
            "semantic_audit_exclusion_schema_version": (
                "stage2_semantic_audit_exclusion_v1"
            ),
            "ineligible_source_row_count": 0,
            "canonical_graph_count": 1,
            "row_atomic_rejected_graph_count": 0,
            "rejection_record_count": 0,
            "parent_v3_count": 1,
            "parent_v3_rejected_count": 0,
            "complete_selection": True,
            "legacy_qa_in_model_prompt": False,
            "legacy_semantic_label_fields_in_model_request": False,
            "old_graph_events_inherited": False,
            "row_atomic_admission": True,
        }
        write_json(self.rebuild / "rebuild_summary.json", summary)

    def _audit(self, *, forbidden: list[Path] | None = None) -> dict:
        return audit.audit_rebuild(
            source_path=self.source,
            rebuild_dir=self.rebuild,
            expectations=self.expectations,
            forbidden_jsonl=forbidden or [],
        )

    @staticmethod
    def _codes(report: dict) -> set[str]:
        return {item["code"] for item in report["failures"]}

    def _replace_canonical_and_parent_events(self, events: list[dict]) -> None:
        graphs = audit.read_jsonl(self.rebuild / "stage2_semantic_graphs.jsonl")
        graphs[0]["semantic_parent_graph"]["events"] = copy.deepcopy(events)
        graphs[0]["semantic_parent_graph"]["event_count"] = len(events)
        write_jsonl(self.rebuild / "stage2_semantic_graphs.jsonl", graphs)

        graph_hash = audit.sha256_json(events)
        parents = audit.read_jsonl(self.rebuild / "stage3_parent_v3.jsonl")
        parents[0]["semantic_parent_graph"]["events"] = copy.deepcopy(events)
        parents[0]["semantic_parent_graph"]["event_count"] = len(events)
        parents[0]["semantic_parent_graph"]["graph_sha256"] = graph_hash
        parents[0]["source_snapshot"]["reconstructed_graph_sha256"] = graph_hash
        write_jsonl(self.rebuild / "stage3_parent_v3.jsonl", parents)

    def _append_valid_ineligible_source_row(self) -> None:
        source_rows = audit.read_jsonl(self.source)
        source_rows.append(
            {
                "sample_id": "source-sample-2",
                "split": "train",
                "training_use": "train_candidate",
                "dialogue": {
                    "dialogue_hash": "source-dialogue-hash-2",
                    "turns": [
                        {
                            "turn_id": "D2:1",
                            "speaker": "Analyst",
                            "text": "Q: legacy prompt A: legacy answer",
                        }
                    ],
                },
                "graph_update": {
                    "events": [
                        {"event_id": "old:event:3"},
                        {"event_id": "old:event:4"},
                    ]
                },
            }
        )
        write_jsonl(self.source, source_rows)

        requests = audit.read_jsonl(self.rebuild / "extraction_requests.jsonl")
        requests.append(
            {
                "schema_version": audit.REBUILD_SCHEMA_VERSION,
                "request_id": "request-2",
                "source_sample_id": "source-sample-2",
                "source_line_number": 2,
                "source_event_count": 2,
                "fallback_event_count": 0,
                "events": [],
                "skipped_events": [
                    {
                        "source_event_id": "old:event:3",
                        "source_event_order": 1,
                        "reason": "legacy_qa_evidence_turn",
                    },
                    {
                        "source_event_id": "old:event:4",
                        "source_event_order": 2,
                        "reason": "legacy_qa_evidence_turn",
                    },
                ],
            }
        )
        write_jsonl(self.rebuild / "extraction_requests.jsonl", requests)
        write_jsonl(
            self.rebuild / "ineligible_source_rows.jsonl",
            [
                {
                    "schema_version": audit.REBUILD_SCHEMA_VERSION,
                    "request_id": "request-2",
                    "source_sample_id": "source-sample-2",
                    "source_line_number": 2,
                    "source_event_count": 2,
                    "excluded_source_event_count": 2,
                    "reason": "no_clean_linked_semantic_events",
                    "dialogue_fallback_used": False,
                    "disposition": "source_row_quarantined",
                }
            ],
        )
        write_jsonl(
            self.rebuild / "source_event_exclusions.jsonl",
            [
                {
                    "schema_version": audit.REBUILD_SCHEMA_VERSION,
                    "request_id": "request-2",
                    "source_sample_id": "source-sample-2",
                    "source_event_id": event_id,
                    "reason": "legacy_qa_evidence_turn",
                    "disposition": "quarantined_before_model",
                }
                for event_id in ("old:event:3", "old:event:4")
            ],
        )
        summary = audit.read_json(self.rebuild / "rebuild_summary.json")
        summary.update(
            {
                "input_row_count": 2,
                "selected_request_count": 2,
                "excluded_source_event_count": 2,
                "ineligible_source_row_count": 1,
            }
        )
        write_json(self.rebuild / "rebuild_summary.json", summary)
        self.expectations = audit.AuditExpectations(
            graph_count=1,
            parent_count=1,
            canonical_event_count=2,
            source_event_count=4,
            preexcluded_source_event_count=2,
            semantic_audit_exclusion_count=0,
            fallback_event_count=0,
            ineligible_source_row_count=1,
            eligible_rejection_count=0,
            minimum_events_per_row=2,
            lineage_count=1,
        )

    def _append_valid_semantic_audit_exclusion(self) -> None:
        source_rows = audit.read_jsonl(self.source)
        source_rows[0]["graph_update"]["events"].append(
            {"event_id": "old:event:3"}
        )
        write_jsonl(self.source, source_rows)

        requests = audit.read_jsonl(self.rebuild / "extraction_requests.jsonl")
        requests[0]["source_event_count"] = 3
        requests[0]["skipped_events"] = [
            {
                "source_event_id": "old:event:3",
                "source_event_order": 3,
                "reason": "independent_semantic_audit_exclusion",
            }
        ]
        write_jsonl(self.rebuild / "extraction_requests.jsonl", requests)
        write_jsonl(
            self.rebuild / "semantic_audit_exclusions_applied.jsonl",
            [
                {
                    "schema_version": "stage2_semantic_audit_exclusion_v1",
                    "source_line_number": 1,
                    "source_event_id": "old:event:3",
                    "reason": "predicate_slot_mismatch",
                    "detail": "The slot does not preserve the evidence predicate.",
                    "decision": "quarantine",
                    "request_id": "request-1",
                    "source_sample_id": "source-sample-1",
                    "disposition": "quarantined_before_canonical_admission",
                }
            ],
        )
        write_jsonl(
            self.rebuild / "source_event_exclusions.jsonl",
            [
                {
                    "schema_version": audit.REBUILD_SCHEMA_VERSION,
                    "request_id": "request-1",
                    "source_sample_id": "source-sample-1",
                    "source_line_number": 1,
                    "source_event_id": "old:event:3",
                    "source_event_order": 3,
                    "reason": "independent_semantic_audit_exclusion",
                    "semantic_audit_reason": "predicate_slot_mismatch",
                    "semantic_audit_detail": (
                        "The slot does not preserve the evidence predicate."
                    ),
                    "disposition": "quarantined_before_model",
                }
            ],
        )
        summary = audit.read_json(self.rebuild / "rebuild_summary.json")
        summary.update(
            {
                "excluded_source_event_count": 1,
                "semantic_audit_exclusion_count": 1,
            }
        )
        write_json(self.rebuild / "rebuild_summary.json", summary)
        self.expectations = audit.AuditExpectations(
            graph_count=1,
            parent_count=1,
            canonical_event_count=2,
            source_event_count=3,
            preexcluded_source_event_count=1,
            semantic_audit_exclusion_count=1,
            fallback_event_count=0,
            ineligible_source_row_count=0,
            eligible_rejection_count=0,
            minimum_events_per_row=2,
            lineage_count=1,
        )

    def test_valid_fixture_passes_complete_fail_closed_audit(self) -> None:
        report = self._audit()
        self.assertTrue(report["pass"], report["failures"])
        self.assertTrue(report["fail_closed"])
        self.assertFalse(report["inputs_mutated"])
        self.assertEqual(report["observed"]["legacy_event_disposition_count"], 2)
        self.assertEqual(report["observed"]["canonical_event_count"], 2)

    def test_one_explicit_no_fallback_ineligible_row_is_audited(self) -> None:
        self._append_valid_ineligible_source_row()
        report = self._audit()
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["observed"]["ineligible_source_row_count"], 1)
        self.assertEqual(
            report["observed"]["canonical_plus_preexcluded_source_event_count"],
            4,
        )

    def test_ineligible_reason_or_dialogue_fallback_fails(self) -> None:
        self._append_valid_ineligible_source_row()
        rows = audit.read_jsonl(self.rebuild / "ineligible_source_rows.jsonl")
        rows[0]["reason"] = "model_did_not_like_row"
        rows[0]["dialogue_fallback_used"] = True
        write_jsonl(self.rebuild / "ineligible_source_rows.jsonl", rows)
        codes = self._codes(self._audit())
        self.assertIn("invalid_ineligible_source_row_reason", codes)
        self.assertIn("ineligible_source_row_used_dialogue_fallback", codes)

    def test_dialogue_fallback_request_event_fails(self) -> None:
        rows = audit.read_jsonl(self.rebuild / "extraction_requests.jsonl")
        rows[0]["events"][0]["reconstruction_source"] = "clean_dialogue_fallback"
        write_jsonl(self.rebuild / "extraction_requests.jsonl", rows)
        report = self._audit()
        self.assertIn("unsupported_reconstruction_source", self._codes(report))

    def test_semantic_audit_exclusion_has_exact_applied_source_ledger(self) -> None:
        self._append_valid_semantic_audit_exclusion()
        report = self._audit()
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["observed"]["semantic_audit_exclusion_count"], 1)
        self.assertEqual(
            report["observed"]["semantic_audit_source_exclusion_count"], 1
        )

        rows = audit.read_jsonl(
            self.rebuild / "semantic_audit_exclusions_applied.jsonl"
        )
        rows[0]["detail"] = "A different audit rationale."
        write_jsonl(
            self.rebuild / "semantic_audit_exclusions_applied.jsonl", rows
        )
        self.assertIn(
            "semantic_audit_exclusion_payload_mismatch",
            self._codes(self._audit()),
        )

    def test_dependencies_must_be_local_prior_and_terminal_source_audited(self) -> None:
        events = self._canonical_events()
        events[1]["depends_on"] = [events[0]["event_id"]]
        events[1]["provenance"]["deterministic_terminal_recovery"] = True
        events[1]["provenance"]["dependency_source"] = (
            "deterministic_prior_active_entity_fact_overlap"
        )
        self._replace_canonical_and_parent_events(events)
        report = self._audit()
        self.assertTrue(report["pass"], report["failures"])
        self.assertEqual(report["observed"]["dependency_count"], 1)
        self.assertEqual(
            report["observed"]["dangling_or_nonprior_dependency_count"], 0
        )

        self._write_valid_fixture()
        events = self._canonical_events()
        events[0]["depends_on"] = [events[1]["event_id"]]
        events[0]["provenance"]["deterministic_terminal_recovery"] = True
        self._replace_canonical_and_parent_events(events)
        codes = self._codes(self._audit())
        self.assertIn("canonical_dependency_dangling_or_nonprior", codes)
        self.assertIn("terminal_dependency_source_provenance_missing", codes)

    def test_missing_legacy_event_disposition_fails(self) -> None:
        rows = audit.read_jsonl(self.rebuild / "extraction_requests.jsonl")
        rows[0]["events"].pop()
        write_jsonl(self.rebuild / "extraction_requests.jsonl", rows)
        report = self._audit()
        self.assertFalse(report["pass"])
        self.assertIn("incomplete_legacy_event_disposition", self._codes(report))

    def test_request_semantic_hint_fails(self) -> None:
        rows = audit.read_jsonl(self.rebuild / "extraction_requests.jsonl")
        rows[0]["events"][0]["entity_hint"] = "Maria"
        write_jsonl(self.rebuild / "extraction_requests.jsonl", rows)
        report = self._audit()
        self.assertIn("forbidden_request_payload_key", self._codes(report))

    def test_legacy_qa_or_padding_in_parent_dialogue_fails(self) -> None:
        parents = audit.read_jsonl(self.rebuild / "stage3_parent_v3.jsonl")
        bad_turn = {"turn_id": "D1:3", "speaker": "Analyst", "text": "Q: Who? A: Maria"}
        parents[0]["turns"].append(bad_turn)
        parents[0]["dialogue"]["turns"].append(copy.deepcopy(bad_turn))
        write_jsonl(self.rebuild / "stage3_parent_v3.jsonl", parents)
        report = self._audit()
        self.assertIn("legacy_qa_dialogue_not_cleaned", self._codes(report))

        self._write_valid_fixture()
        parents = audit.read_jsonl(self.rebuild / "stage3_parent_v3.jsonl")
        padding = {
            "turn_id": "D1:3",
            "speaker": "Analyst",
            "text": build_padding_text(),
        }
        parents[0]["turns"].append(padding)
        parents[0]["dialogue"]["turns"].append(copy.deepcopy(padding))
        write_jsonl(self.rebuild / "stage3_parent_v3.jsonl", parents)
        report = self._audit()
        self.assertIn("deterministic_padding_not_cleaned", self._codes(report))

    def test_parent_graph_mismatch_or_forbidden_payload_fails(self) -> None:
        parents = audit.read_jsonl(self.rebuild / "stage3_parent_v3.jsonl")
        parents[0]["semantic_parent_graph"]["events"][0]["value"] = "changed"
        parents[0]["qa"] = {"answer": "leak"}
        write_jsonl(self.rebuild / "stage3_parent_v3.jsonl", parents)
        report = self._audit()
        codes = self._codes(report)
        self.assertIn("parent_canonical_graph_events_mismatch", codes)
        self.assertIn("forbidden_parent_payload_key", codes)

    def test_optional_heldout_exact_overlap_fails(self) -> None:
        heldout = self.root / "heldout.jsonl"
        write_jsonl(heldout, [{"dialogue": {"dialogue_hash": "clean-dialogue-hash"}}])
        report = self._audit(forbidden=[heldout])
        self.assertIn("heldout_exact_identity_overlap", self._codes(report))

    def test_missing_required_artifact_fails_closed(self) -> None:
        (self.rebuild / "stage3_parent_v3.jsonl").unlink()
        report = self._audit()
        self.assertFalse(report["pass"])
        self.assertIn("required_rebuild_files_missing", self._codes(report))


def build_padding_text() -> str:
    # Use the production exact allowlist so this test also guards detector drift.
    from scripts.current_method import build_stage3_parent_v2

    return build_stage3_parent_v2.KNOWN_DETERMINISTIC_PADDING_TEXTS[0]


if __name__ == "__main__":
    unittest.main()
