"""Fresh native engineering tests; synthetic fixtures only, no model/API calls."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import public_anchor_v1 as anchors
import baseline
import prepare
HERE = Path(__file__).resolve().parent
ROOT = 'cc-train-episode-1'
AUTHOR, REVIEWER = '/root/fixture_author', '/root/fixture_reviewer'
BASE_READERS = ['/root/fixture_base1', '/root/fixture_base2']
CONSTRAINT = 'Synthetic fixture only; establish no real content judgment.'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def binding(path):
    path = path.resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class PolicyValidationTests(unittest.TestCase):
    def test_hold_needs_two_distinct_exact_spans_not_distinct_turns(self):
        turns = {'R01:S01:T001': {'speaker': 'A', 'text': 'The ticket is active. The ticket is cancelled.'}}
        judgment = {'full_source_read': True, 'covered_turn_refs': list(turns),
                    'checks': {c: {'complete': True, 'status': 'HOLD' if c == 'fact_state_time' else 'PASS',
                                   'reason': 'Synthetic known expectation.'} for c in baseline.CHECKS},
                    'findings': [{'id': 'f1', 'category': 'fact_state_time', 'classification': 'HARD_CONTRADICTION',
                                  'disposition': 'HOLD', 'reason': 'Both explicit states concern the same ticket.',
                                  'citations': [{'turn_ref': 'R01:S01:T001', 'speaker': 'A', 'quote': 'The ticket is active.'},
                                                {'turn_ref': 'R01:S01:T001', 'speaker': 'A', 'quote': 'The ticket is cancelled.'}],
                                  'temporal_evidence': []}],
                    'continuation_constraints': ['Do not continue the unresolved ticket state.'], 'decision': 'HOLD'}
        baseline.validate(judgment, turns)
        first = judgment['findings'][0]['citations'][0]
        for citations in ([first], [first, dict(first)]):
            bad = json.loads(json.dumps(judgment)); bad['findings'][0]['citations'] = citations
            with self.assertRaisesRegex(ValueError, 'two distinct exact citation spans'):
                baseline.validate(bad, turns)

    def test_adapted_corresponding_r01_requires_actual_scene_and_matching_evidence(self):
        source = {'turns': [{'turn_ref': f'R01:S{i:02d}:T001', 'scene_ref': f'R01:S{i:02d}',
                             'speaker': 'A', 'text': f'Synthetic event {i}.'} for i in (1, 2, 3)]}
        row = {'cohort': 'adapted', 'a_plan_mode': 'CORRESPONDING_R01',
               'existing_r01_scene_refs': [t['scene_ref'] for t in source['turns']]}
        plans = [{'scene_ref': scene, 'baseline_scene_ref': t['scene_ref'], 'thread': f'Continue event {i}.',
                  'evidence': [{'turn_ref': t['turn_ref'], 'speaker': t['speaker'], 'quote': t['text']}]}
                 for i, (scene, t) in enumerate(zip(anchors.SCENES[:3], source['turns']), 1)]
        prepare.validate_a_continuations(row, source, plans)
        # Missing mode falls back to actual source scenes, never cohort.
        prepare.validate_a_continuations({'cohort': 'adapted'}, source, plans)
        bad = json.loads(json.dumps(plans)); bad[1]['baseline_scene_ref'] = 'R01:S01'; bad[1]['evidence'] = bad[0]['evidence']
        with self.assertRaisesRegex(AssertionError, 'CORRESPONDING_R01_SCENE'):
            prepare.validate_a_continuations(row, source, bad)
        bad = json.loads(json.dumps(plans)); bad[1]['evidence'] = bad[0]['evidence']
        with self.assertRaisesRegex(AssertionError, 'EVIDENCE_MUST_SUPPORT'):
            prepare.validate_a_continuations(row, source, bad)
        partial = {'turns': source['turns'][:1]}
        row = {'cohort': 'adapted', 'a_plan_mode': 'THREE_EXISTING_EVENT_THREADS_OR_STAGES', 'existing_r01_scene_refs': ['R01:S01']}
        for plan in plans:
            plan['baseline_scene_ref'] = 'R01:S01'; plan['evidence'] = plans[0]['evidence']
        prepare.validate_a_continuations(row, partial, plans)
        bad = json.loads(json.dumps(plans)); bad[2]['baseline_scene_ref'] = 'R01:S03'
        with self.assertRaisesRegex(AssertionError, 'BASELINE_SCENE_NOT_IN_SOURCE'):
            prepare.validate_a_continuations(row, partial, bad)


class FreshWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_fresh_v124_', dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name) / 'fresh_batch'
        self.tools = self.batch / 'tools'
        self.unit = self.batch / 'units/root1'
        self.folder = self.unit / 'roots/root1'
        self.tools.mkdir(parents=True)
        self.stages = []
        for name in (*anchors.RUNTIME_TOOLS, 'baseline.py', 'prepare.py'):
            shutil.copy2(HERE / name, self.tools / name)
        shutil.copytree(HERE.parent / 'scripts', self.batch / 'scripts')
        rows, roles, assignments = [], [], []
        for i in range(1, 32):
            root = f'cc-train-episode-{i}'
            unit = self.batch / f'units/root{i}'
            baseline = unit / 'inputs/BASELINE.json'
            turns = [{'turn_ref': f'R01:S{j:02d}:T{k:03d}', 'session_ref': f'R01:S{j:02d}',
                      'scene_ref': f'R01:S{j:02d}', 'speaker': 'A' if k % 2 else 'B',
                      'text': f'Synthetic baseline event {j} utterance {k}.', 'origin': 'historical_fixture_only'}
                     for j in (1, 2, 3) for k in (1, 2)]
            write(baseline, {'root_id': root, 'turns': turns, 'synthetic_fixture_only': True})
            roles.append({'root_id': root, 'author': AUTHOR, 'reviewer': REVIEWER, 'baseline_reviewers': BASE_READERS})
            rows.append({'root_id': root, 'split': 'dev' if i <= 15 else 'test', 'action': 'SCREEN',
                         'cohort': 'standard', 'unit_path': str(unit), 'baseline_path': str(baseline),
                         'baseline_sha256': binding(baseline)['sha256'], 'baseline_binding': binding(baseline),
                         'baseline_reviewers': BASE_READERS})
            events = [{'dialogue_id': f'synthetic-{i}-{j}', 'turns': [
                        {'turn_id': f'native-{i}-{j}-1', 'speaker': 'A', 'text': f'Synthetic native event {j} begins.'},
                        {'turn_id': f'native-{i}-{j}-2', 'speaker': 'B', 'text': f'Synthetic native event {j} completed.'}]}
                      for j in (1, 2, 3)]
            pool = unit / 'inputs/EVENT_POOL.json'; write(pool, {'events': events})
            assignments.append({'root_id': root, 'event_pool_binding': binding(pool),
                'candidate_source_ids': ['multiwoz22:train:' + e['dialogue_id'] for e in events]})
        write(self.batch / 'MANIFEST.json', {'roots': rows, 'synthetic_fixture_only': True})
        write(self.batch / 'ROLE_ASSIGNMENTS.json', {'roots': roles, 'synthetic_fixture_only': True})
        evidence = self.batch / 'evidence'; evidence.mkdir()
        for name in ('BASELINE_POLICY.md', 'CONTENT_CONTRACT.md'):
            (evidence / name).write_text('Synthetic fixture policy; no formal content claims.\n')
        write(evidence / 'SOURCE_POOL_FREEZE.json', {'consumed_source_ids': [], 'candidate_assignments': assignments})
        frozen = [p for p in self.batch.rglob('*') if p.is_file()]
        write(self.batch / 'MASTER_FREEZE.json', {'status': 'FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': [binding(p) for p in frozen],
              'synthetic_fixture_only': True})
        source = read(self.unit / 'inputs/BASELINE.json')
        judgment = {'full_source_read': True, 'covered_turn_refs': [t['turn_ref'] for t in source['turns']],
                    'checks': {c: {'complete': True, 'status': 'PASS', 'reason': 'Known synthetic fixture expectation.'}
                               for c in ('identity_coreference', 'gender_relationship_titles', 'fact_state_time', 'unsupported_reconciliation')},
                    'findings': [], 'continuation_constraints': [CONSTRAINT], 'decision': 'PASS'}
        payload = self.unit / 'synthetic_baseline_judgment.json'; write(payload, judgment)
        for executor in BASE_READERS:
            self.call('baseline.py', 'read', '--root', ROOT, '--executor', executor)
            self.call('baseline.py', 'save', '--root', ROOT, '--executor', executor, '--judgment', payload)
        self.call('baseline.py', 'gate', '--root', ROOT, '--executor', '/root/fixture_gate')
        self.call('prepare.py', 'match-read', '--root', ROOT, '--executor', AUTHOR)
        matching = {'read_attestation': {'full_source_read': True, 'all_continuation_constraints_read': True,
                     'all_candidate_events_fully_read': True},
                    'a_continuations': [{'scene_ref': scene, 'baseline_scene_ref': f'R01:S{i:02d}', 'thread': f'Synthetic event {i} continuation.',
                        'evidence': [{'turn_ref': source['turns'][(i-1)*2]['turn_ref'], 'speaker': source['turns'][(i-1)*2]['speaker'],
                                      'quote': source['turns'][(i-1)*2]['text']}]} for i, scene in enumerate(anchors.SCENES[:3], 1)],
                    'allocations': [{'scene_ref': scene, 'source_id': f'multiwoz22:train:synthetic-1-{i}',
                        'compatibility_reason': 'Synthetic known-compatible event.', 'main_purpose': f'Synthetic event {i}.'}
                        for i, scene in enumerate(anchors.SCENES[3:], 1)]}
        payload = self.unit / 'synthetic_matching_payload.json'; write(payload, matching)
        self.call('prepare.py', 'match-save', '--root', ROOT, '--executor', AUTHOR, '--payload', payload)

    def call(self, tool, *args, json_output=False, expect=0):
        if tool == "review_v4.py" and len(args) > 1 and str(args[1]).endswith("-save") and expect == 0:
            pre_args = list(args); pre_args[1] = str(pre_args[1]).replace("-save", "-preflight")
            self.call(tool, *pre_args, expect=0)
        completed = subprocess.run([sys.executable, "-B", str(self.tools / tool), *map(str, args)],
                                   cwd=self.batch, capture_output=True, text=True)
        self.stages.append({"tool": tool, "args": list(map(str, args)), "returncode": completed.returncode})
        self.assertEqual(completed.returncode, expect, completed.stdout + "\n" + completed.stderr)
        return json.loads(completed.stdout) if json_output else completed.stdout

    def scene_review(self, scene):
        target = self.folder / "scenes" / scene.removeprefix("R02:").replace(":", "_")
        source = read(target / "CUMULATIVE_SOURCE.json")
        raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True},
               "decision": "READY", "checks": {name: {"status": "PASS", "reason": "Known synthetic fixture expectation only."}
                                                for name in ("time", "location", "actor_identity", "plan_vs_completion", "cross_scene_continuity",
                                                             "semantic_repetition", "naturalness", "event_fidelity")},
               "findings": [], "baseline_constraint_review": {"status": "PASS", "reason": "Synthetic expected judgment.",
                   "constraint_results": [{"index": 1, "constraint": "Synthetic fixture only; establish no real content judgment.",
                                           "status": "PASS", "reason": "Fixture only.", "evidence_turn_refs": [source["turns"][-1]["turn_ref"]]}],
                   "new_blocking_findings": []}}
        if ":B:" in scene:
            raw["read_attestation"]["native_event_full_read"] = True
        judgment = self.unit / f"synthetic_{scene.replace(':', '_')}_review.json"; write(judgment, raw)
        base = [ROOT, "scene-read", "--batch", self.batch, "--executor", REVIEWER, "--scene", scene]
        self.call("review_v4.py", *base)
        base[1] = "scene-preflight"
        self.call("review_v4.py", *base, "--judgment", judgment)
        base[1] = "scene-save"
        self.call("review_v4.py", *base, "--judgment", judgment)
        result = self.call("review_provenance_gate_v2.py", "check", "--batch", self.batch, "--root", ROOT,
                           "--phase", "scene", "--scene", scene, json_output=True)
        self.assertTrue(result["pass"], result["errors"])

    def expand(self, scenes):
        for scene in scenes:
            self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene)
            messages = [{"speaker": "A" if i % 2 else "B", "text": f"Synthetic {scene} utterance {i}."} for i in range(1, 13)]
            payload = self.unit / f"synthetic_{scene.replace(':', '_')}_payload.json"
            write(payload, {"turns": messages})
            self.call("author_v120.py", "scene-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR,
                      "--scene", scene, "--payload", payload)
            self.scene_review(scene)
            state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
            index = anchors.SCENES.index(scene)
            self.assertEqual(state["scenes"][index]["status"], "READY", state)
            self.assertEqual(state["ready_scene_count"], index + 1)

    def qa_payloads(self):
        source = read(self.folder / "CUMULATIVE_SOURCE.json")
        turns = {row["turn_ref"]: row for row in source["turns"]}
        source_sha = binding(self.folder / "CUMULATIVE_SOURCE.json")["sha256"]
        cfg = read(self.unit / "RUN_CONFIG.json")
        plan = {"schema_version": anchors.SCHEMA, "root_id": ROOT, "source_sha256": source_sha,
                "scope_scene_refs": list(anchors.SCENES), "slots": []}
        draft = {"stem": "Select the correctly attributed exact utterances within these six scenes: " + ", ".join(anchors.SCENES),
                 "facts": [], "ordering": []}
        self.statement_expected = {}
        for i, scene in enumerate(anchors.SCENES, 1):
            turn = turns[scene + ":T003"]
            true = f'In {scene}, speaker A says "{turn["text"]}".'
            false = f'In {scene}, speaker B says "{turn["text"]}".'
            draft["facts"].append({"fact_id": f"synthetic-f{i}", "source_turn_ref": turn["turn_ref"], "true_text": true, "false_text": false})
            self.statement_expected[true] = ("supports", turn)
            self.statement_expected[false] = ("contradicts", turn)
        for slot_i, turn_i in ((1, 4), (2, 8)):
            slot = {"slot_id": str(slot_i), "candidate_id": cfg["batch_id"] + "-root1-ordering" + str(slot_i), "anchors": []}
            ordering = {"slot_id": str(slot_i), "items": []}
            for i, scene in enumerate(anchors.SCENES):
                turn = turns[scene + f":T{turn_i:03d}"]
                anchor = {"item_id": f"I{(i + slot_i) % 6 + 1}", "anchor_id": f"synthetic-anchor-{slot_i}-{i}",
                          "turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "full_turn_quote": turn["text"],
                          "full_turn_sha256": anchors.text_sha(turn["text"])}
                slot["anchors"].append(anchor)
                ordering["items"].append({"item_id": anchor["item_id"], "anchor_id": anchor["anchor_id"],
                                          "text": turn["text"], "support_quote": turn["text"]})
            plan["slots"].append(slot); draft["ordering"].append(ordering)
        self.plan = plan
        plan_path, draft_path = self.unit / "synthetic_anchor_payload.json", self.unit / "synthetic_qa_payload.json"
        write(plan_path, plan); write(draft_path, draft)
        return plan_path, draft_path

    def statement_judgment(self, text):
        relation, turn = self.statement_expected[text]
        return {"proposition": text, "relation": relation, "scope_valid": True, "fact_distinct": True,
                "reason": "Synthetic known speaker attribution expectation.",
                "citations": [{"turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "quote": turn["text"]}]}

    def ordering_judgment(self, question):
        slot = next(s for s in self.plan["slots"] if s["candidate_id"] == question["candidate_id"])
        by_item = {a["item_id"]: a for a in slot["anchors"]}
        result = {"candidate_id": question["candidate_id"], **{k: True for k in ("stem_valid", "scope_valid", "coherent", "unique_order", "neutral_stem_ok")},
                  "reason": "Exact synthetic events in distinct turns.", "derived_order": [a["item_id"] for a in slot["anchors"]],
                  "alternative_orders": [], "items": {}}
        for label, text in question["items"].items():
            a = by_item[label]
            result["items"][label] = {"proposition": text, **{k: True for k in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "single_event_ok")},
                "reason": "Synthetic exact event.", "anchor_id": a["anchor_id"], "frozen_full_turn_sha256": a["full_turn_sha256"],
                "order_anchor": {"turn_ref": a["turn_ref"], "speaker": a["speaker"], "quote": a["full_turn_quote"]},
                "support_refs": [a["turn_ref"]], "alternative_matching_anchors": []}
        return result

    def qa_review(self, phase):
        attest = {"full_source_read": True, "all_continuation_constraints_read": True, "private_answers_read_before_save": False,
                  "public_anchor_plan_full_read": True, "source_read_before_anchor_plan_and_public_content": True}
        raw = {"read_attestation": attest, "findings": []}
        if phase == "asset":
            assets = read(self.folder / "PUBLIC_ASSETS.json")
            raw.update(multiselect_stem_valid=True, statements={r["statement_id"]: self.statement_judgment(r["proposition"]) for r in assets["statements"]},
                       ordering_questions=[self.ordering_judgment(q) for q in assets["ordering_questions"]])
        else:
            attest["final_public_questions_actually_reread"] = True
            raw["questions"] = {}
            for q in read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]:
                if q["question_type"] == "ordering":
                    judgment = self.ordering_judgment(q)
                else:
                    judgment = {"candidate_id": q["candidate_id"], **{k: True for k in ("stem_valid", "scope_valid", "fact_distinct", "no_entangled_options")},
                                "reason": "Known synthetic attribution facts.", "options": {label: self.statement_judgment(text) for label, text in q["options"].items()}}
                raw["questions"][q["candidate_id"]] = judgment
        path = self.unit / f"synthetic_{phase}_judgment.json"; write(path, raw)
        self.call("review_v4.py", ROOT, phase + "-read", "--batch", self.batch, "--executor", REVIEWER)
        self.call("review_v4.py", ROOT, phase + "-save", "--batch", self.batch, "--executor", REVIEWER, "--judgment", path)

    def test_full_native_retry_expansion_and_qa(self):
        self.expand(anchors.SCENES)
        self.call("author_v120.py", "qa-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR)
        plan, draft = self.qa_payloads()
        self.call("author_v120.py", "anchor-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", plan)
        self.call("author_v120.py", "qa-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--payload", draft)
        self.call("question_controller_v120.py", "publish", "--batch-dir", self.unit)
        self.qa_review("asset")
        compose = self.call("run_guarded_questions_v2.py", "compose", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertTrue(compose["downstream_invoked"]); self.assertEqual(compose["status"], "GUARDED_STAGE_COMPLETED")
        self.assertEqual(len(read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]), 7)
        self.qa_review("public")
        finalize = self.call("run_guarded_questions_v2.py", "finalize", "--batch", self.batch, "--root", ROOT, json_output=True)
        self.assertTrue(finalize["downstream_invoked"]); self.assertEqual(finalize["status"], "GUARDED_STAGE_COMPLETED")
        crosscheck = read(self.folder / "FINAL_CROSSCHECK.json")
        self.assertEqual(len(crosscheck["results"]), 7)
        self.assertTrue(all(row["status"] == "CONTENT_PASS" for row in crosscheck["results"]), crosscheck)
        gold = read(self.folder / "PRIVATE_GOLD.json")
        self.assertTrue(gold["gold_constructed_after_complete_public_review"])
        by_id = {q["candidate_id"]: q for q in read(self.folder / "PUBLIC_QUESTIONS.json")["questions"]}
        self.assertEqual(sorted(len(q["gold"]) for q in gold["questions"] if by_id[q["candidate_id"]]["question_type"] == "multi_select"), [2, 3, 4, 5, 6])
        self.assertGreaterEqual(len(self.stages), 67)

    def test_author_read_rejects_changed_execution_config_before_render(self):
        path = self.unit / "RUN_CONFIG.json"
        cfg = read(path); cfg["new_turn_count"] = {"min": 1, "max": 100}; write(path, cfg)
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        result = self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT,
                           "--executor", AUTHOR, "--scene", "R02:A:S01", expect=1)
        self.assertEqual(result, "")
        self.assertFalse((self.unit / "evidence/AUTHOR_R02_A_S01_READ_CONTEXT.json").exists())
        self.assertFalse((self.unit / "evidence/AUTHOR_R02_A_S01_FULL_READ.md").exists())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_author_entry_rejects_foreign_self_or_serializer_origins(self):
        # A separate interpreter imports the real frozen author helper and then
        # substitutes only an origin; no production function implementation is mocked.
        for field, expected in (("self", "RUNTIME_LOADED_CODE_MISMATCH"), ("serializer", "FOREIGN_SCENE_SERIALIZER")):
            foreign = self.tools / ("FOREIGN_" + field + ".py")
            shutil.copy2(self.tools / "author_v120.py" if field == "self" else self.batch / "scripts/codex_scene_payload.py", foreign)
            before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
            script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import author_v120 as author
if sys.argv[2] == "self": author.__file__ = sys.argv[3]
else: author.serializer.__file__ = sys.argv[3]
try:
    author.assigned(Path(sys.argv[4]),sys.argv[5],sys.argv[6])
except ValueError as exc:
    print(str(exc)); raise SystemExit(2)
raise SystemExit(9)
'''
            completed = subprocess.run([sys.executable, "-B", "-c", script, str(self.tools), field, str(foreign),
                                        str(self.batch), ROOT, AUTHOR], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
            self.assertIn(expected, completed.stdout)
            self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_real_native_a2_review_unlocks_a3_then_missing_receipt_blocks(self):
        self.expand(anchors.SCENES[:2])
        scene = anchors.SCENES[2]
        self.call("author_v120.py", "scene-read", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene)
        payload = self.unit / "synthetic_A3_payload.json"
        write(payload, {"turns": [{"speaker": "A" if i % 2 else "B", "text": f"Synthetic A3 check {i}."} for i in range(1, 13)]})
        self.call("author_v120.py", "scene-save", "--batch-dir", self.batch, "--root", ROOT, "--executor", AUTHOR, "--scene", scene, "--payload", payload)
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertEqual(state["scenes"][2]["status"], "AWAITING_REVIEW")
        digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
        receipt = self.unit / "evidence/reads" / f"scene_R02_A_S02_{digest}" / "SAVE_RECEIPT.json"
        receipt.unlink()
        state = self.call("scene_controller_v120.py", "progress", "--batch-dir", self.unit, "--root-id", ROOT, json_output=True)
        self.assertEqual(state["scenes"][1]["status"], "INTEGRITY_FAILURE")
        self.assertIn("SCENE_REVIEW_PROVENANCE_BLOCKED", state["scenes"][1]["reason"])
        self.assertFalse(state["full_expansion_complete"])

    def phase_call(self, root, expected=0):
        code = "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from phase_gate_v124 import require_phase; print(require_phase(Path(sys.argv[2]),sys.argv[3]))"
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.tools), str(self.batch), root], capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_test_phase_closed_until_complete_audited_dev_accounting(self):
        root = 'cc-train-episode-16'
        blocked = self.phase_call(root, 1)
        self.assertIn('TEST_PHASE_CLOSED_UNTIL_ALL_DEV_TERMINAL', blocked.stderr)
        result = self.call('baseline.py', 'read', '--root', root, '--executor', BASE_READERS[0], expect=1)
        self.assertEqual(result, '')
        self.assertFalse((self.batch / 'units/root16/baseline_reviews').exists())
        result = self.call('prepare.py', 'match-read', '--root', root, '--executor', AUTHOR, expect=1)
        self.assertEqual(result, '')
        evidence = self.batch / 'evidence'
        rows = []
        for i in range(1, 16):
            path = evidence / f'synthetic_terminal_{i}.json'
            write(path, {'synthetic_fixture_only': True, 'terminal_state': 'BASELINE_HOLD'})
            rows.append({'root_id': f'cc-train-episode-{i}', 'terminal_state': 'BASELINE_HOLD', 'terminal_evidence_bindings': [binding(path)]})
        accounting = evidence / 'DEV_TERMINAL_ACCOUNTING.json'
        write(accounting, {'schema_version': 'v124_dev_terminal_accounting_v1', 'status': 'ALL_DEV_TERMINAL',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'solver_run': False, 'score_based_selection': False, 'roots': rows})
        audit = evidence / 'DEV_PHASE_CLOSE_AUDIT.json'
        write(audit, {'status': 'PASS', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'terminal_accounting_binding': binding(accounting)})
        closure = evidence / 'DEV_PHASE_CLOSED.json'
        write(closure, {'schema_version': 'v124_dev_phase_closed_v1', 'status': 'DEV_COMPLETE_BEFORE_TEST_READS',
              'batch_id': self.batch.name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'solver_run': False, 'score_based_selection': False, 'master_freeze_binding': binding(self.batch / 'MASTER_FREEZE.json'),
              'manifest_binding': binding(self.batch / 'MANIFEST.json'), 'terminal_accounting_binding': binding(accounting),
              'independent_audit_binding': binding(audit)})
        self.phase_call(root)
        self.call('baseline.py', 'read', '--root', root, '--executor', BASE_READERS[0])
        path = evidence / 'synthetic_terminal_1.json'; path.write_text('{}\n')
        self.assertIn('PHASE_BINDING_CHANGED', self.phase_call(root, 1).stderr)

    def test_failed_anchor_first_raw_survives_and_blocks_rewrite(self):
        self.expand(anchors.SCENES)
        self.call('author_v120.py', 'qa-read', '--batch-dir', self.batch, '--root', ROOT, '--executor', AUTHOR)
        plan_path, _ = self.qa_payloads()
        good = read(plan_path)
        bad = dict(good); bad['source_sha256'] = 'bad-source'
        write(plan_path, bad)
        self.call('author_v120.py', 'anchor-save', '--batch-dir', self.batch, '--root', ROOT, '--executor', AUTHOR, '--payload', plan_path, expect=1)
        self.assertEqual(read(self.folder / 'ORDERING_ANCHOR_PAYLOAD.json'), bad)
        self.assertTrue((self.folder / 'ORDERING_ANCHOR_FAILURE.json').exists())
        write(plan_path, good)
        self.call('author_v120.py', 'anchor-save', '--batch-dir', self.batch, '--root', ROOT, '--executor', AUTHOR, '--payload', plan_path, expect=1)
        self.assertEqual(read(self.folder / 'ORDERING_ANCHOR_PAYLOAD.json'), bad)

    def test_baseline_first_semantic_payload_locked_and_failures_preserved(self):
        root = 'cc-train-episode-2'
        unit = self.batch / 'units/root2'
        payload = unit / 'synthetic_failed_baseline.json'
        original = read(self.unit / 'synthetic_baseline_judgment.json')
        bad = json.loads(json.dumps(original)); bad['checks']['identity_coreference'].pop('complete')
        write(payload, bad)
        self.call('baseline.py', 'read', '--root', root, '--executor', BASE_READERS[0])
        self.call('baseline.py', 'save', '--root', root, '--executor', BASE_READERS[0], '--judgment', payload, expect=1)
        folder = unit / 'baseline_reviews/reader1'
        self.assertEqual(read(folder / 'FIRST_JUDGMENT.json'), bad)
        changed = json.loads(json.dumps(original)); changed['continuation_constraints'] = ['Changed semantic content.']
        write(payload, changed)
        self.call('baseline.py', 'save', '--root', root, '--executor', BASE_READERS[0], '--judgment', payload, expect=1)
        self.assertFalse((folder / 'REVIEW.json').exists())
        write(payload, original)
        self.call('baseline.py', 'save', '--root', root, '--executor', BASE_READERS[0], '--judgment', payload)
        self.assertEqual(len(list((folder / 'submissions').glob('*.failure.json'))), 2)

    def test_matching_first_raw_failure_preserved_and_no_retry(self):
        raw = self.unit / 'evidence/MATCHING/MATCHING_PAYLOAD.json'
        # Remove only synthetic artifacts to exercise a first failed save in
        # this isolated fixture; no formal artifact is ever changed by tests.
        raw.unlink()
        bad = self.unit / 'synthetic_bad_matching.json'; write(bad, {})
        self.call('prepare.py', 'match-save', '--root', ROOT, '--executor', AUTHOR, '--payload', bad, expect=1)
        self.assertEqual(read(raw), {})
        self.assertTrue((raw.parent / 'MATCHING_FAILURE.json').exists())
        self.call('prepare.py', 'match-save', '--root', ROOT, '--executor', AUTHOR,
                  '--payload', self.unit / 'synthetic_matching_payload.json', expect=1)
        self.assertEqual(read(raw), {})


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main(verbosity=2)
