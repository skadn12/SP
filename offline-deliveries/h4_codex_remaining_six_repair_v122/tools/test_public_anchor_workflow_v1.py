"""Synthetic end-to-end read/save/gate tests; no real roots or model calls.

Only scene-controller state is synthetic. Review rendering, exact bindings,
semantic schema validation, first-judgment preservation, lineage and guarded
dispatch execute the production v120 code. Synthetic fixtures live below this
tools directory and are deleted on completion.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import public_anchor_v1 as anchors
import review_provenance_gate_v2 as gate
import run_guarded_questions_v2 as wrapper

HERE = Path(__file__).resolve().parent
ROOT = "cc-train-episode-1"
AUTHOR, REVIEWER = "/root/fixture_author", "/root/fixture_reviewer"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper(path):
    spec = importlib.util.spec_from_file_location("synthetic_v120_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fixture_module(name, tools):
    spec = importlib.util.spec_from_file_location(name, tools / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_fixture(batch):
    unit = batch / "units/root1"
    folder = unit / "roots/root1"
    tools = batch / "tools"
    tools.mkdir(parents=True)
    for name in anchors.RUNTIME_TOOLS:
        shutil.copy2(HERE / name, tools / name)
    shutil.copytree(HERE.parent / "scripts", batch / "scripts")
    # Explicit synthetic test state, never used or accepted for real roots.
    (tools / "scene_controller_v120.py").write_text('''
SYNTHETIC_FIXTURE_ONLY = True
import json
def load_config(unit):
    return json.loads((unit / "RUN_CONFIG.json").read_text())
def inspect_root(unit, cfg, root, record_reviews=False):
    folder = unit / "roots/root1"
    scenes = ["R02:%s:S%02d" % (r, i) for r in "AB" for i in (1, 2, 3)]
    awaiting = unit / "SYNTHETIC_SCENE_AWAITING.json"
    current = json.loads(awaiting.read_text())["scene"] if awaiting.exists() else None
    current_index = scenes.index(current) if current else 6
    return {"root_stopped": False, "full_expansion_complete": current is None,
            "scenes": [{"status": "READY" if n < current_index else "AWAITING_REVIEW" if n == current_index else "NOT_STARTED",
                        "folder": str(folder / "scenes" / s.removeprefix("R02:").replace(":", "_"))}
                       for n, s in enumerate(scenes)]}
''', encoding="utf-8")
    role = {"root_id": ROOT, "author": AUTHOR, "reviewer": REVIEWER,
            "baseline_reviewers": ["/root/fixture_base1", "/root/fixture_base2"]}
    write(batch / "ROLE_ASSIGNMENTS.json", {"synthetic_fixture_only": True, "roots": [role]})
    write(unit / "RUN_CONFIG.json", {"synthetic_fixture_only": True, "roots": [{"root_id": ROOT}]})
    write(unit / "selection.json", {"root_id": ROOT, "a_continuations": [
        {"scene_ref": scene, "baseline_scene_ref": "R01:S01", "thread": "Synthetic thread.", "evidence": ["Synthetic evidence."]}
        for scene in anchors.SCENES[:3]]})
    write(unit / "inputs/BASELINE.json", {"synthetic_fixture_only": True})
    baseline_reviews = []
    for i, reviewer in enumerate(role["baseline_reviewers"], 1):
        path = unit / f"baseline_reviews/reader{i}/REVIEW.json"
        write(path, {"reviewer": reviewer, "synthetic_fixture_only": True})
        baseline_reviews.append({**gate.binding(path), "reviewer": reviewer})
    write(unit / "inputs/CONTINUATION_CONSTRAINTS.json", {
        "root_id": ROOT, "decision": "PASS", "continuation_constraints": ["Synthetic events remain separate."],
        "baseline_binding": gate.binding(unit / "inputs/BASELINE.json"), "review_bindings": baseline_reviews})
    source = {"root_id": ROOT, "turns": [
        {"turn_ref": scene + f":T{j:03d}", "scene_ref": scene, "speaker": "Fixture",
         "text": f"Synthetic event {scene} number {j} was completed."}
        for scene in anchors.SCENES for j in (1, 2)]}
    write(folder / "CUMULATIVE_SOURCE.json", source)
    prefix = unit / "synthetic_prefix"
    for name, obj in (("CANDIDATE.json", source), ("RAW_JUDGMENT.json", {}), ("READ_CONTEXT.json", {}), ("SAVE_RECEIPT.json", {})):
        write(prefix / name, obj)
    (prefix / "INPUT_RENDER.txt").write_text("Explicitly synthetic prefix metadata.\n", encoding="utf-8")
    write(prefix / "REVIEW.json", {"input_bindings": [],
          "read_context_binding": gate.binding(prefix / "READ_CONTEXT.json"),
          "input_render_binding": gate.binding(prefix / "INPUT_RENDER.txt"),
          "raw_judgment_binding": gate.binding(prefix / "RAW_JUDGMENT.json"),
          "candidate_prefix_binding": gate.binding(prefix / "CANDIDATE.json")})
    write(unit / "PREFIX_IMPORT.json", {"synthetic_fixture_only": True, "scenes": [],
          "prefix_review_binding": gate.binding(prefix / "REVIEW.json"),
          "candidate_prefix_binding": gate.binding(prefix / "CANDIDATE.json"),
          "prefix_read_receipt_binding": gate.binding(prefix / "SAVE_RECEIPT.json")})
    for scene in anchors.SCENES:
        short = scene.removeprefix("R02:").replace(":", "_")
        write(folder / "scenes" / short / "SOURCE_REVIEW.json", {"synthetic_fixture_only": True})
    write(folder / "scenes/B_S03/CUMULATIVE_SOURCE.json", source)
    plan = {"schema_version": anchors.SCHEMA, "root_id": ROOT,
            "source_sha256": gate.sha(folder / "CUMULATIVE_SOURCE.json"),
            "scope_scene_refs": list(anchors.SCENES), "slots": []}
    for j in (1, 2):
        slot = {"slot_id": str(j), "candidate_id": f"synthetic-ordering{j}", "anchors": []}
        for i, scene in enumerate(anchors.SCENES):
            turn = source["turns"][i * 2 + j - 1]
            slot["anchors"].append({"item_id": f"I{(i + j) % 6 + 1}", "anchor_id": f"synthetic-a{j}-{i}",
                "turn_ref": turn["turn_ref"], "speaker": turn["speaker"], "full_turn_quote": turn["text"],
                "full_turn_sha256": anchors.text_sha(turn["text"])})
        plan["slots"].append(slot)
    write(folder / "ORDERING_ANCHOR_PLAN.json", plan)
    write(folder / "QA_AUTHOR_RECEIPT.json", {"author": AUTHOR, "source_sha256": plan["source_sha256"],
          "ordering_anchor_plan_binding": gate.binding(folder / "ORDERING_ANCHOR_PLAN.json")})
    questions = [{"candidate_id": slot["candidate_id"], "root_id": ROOT, "question_type": "ordering",
                  "question": "Order the synthetic events as they appear in the supplied dialogue.",
                  "items": {a["item_id"]: a["full_turn_quote"] for a in slot["anchors"]},
                  "source_path": str(folder / "CUMULATIVE_SOURCE.json"), "source_sha256": plan["source_sha256"]}
                 for slot in plan["slots"]]
    write(folder / "PUBLIC_ASSETS.json", {"root_id": ROOT, "source_sha256": plan["source_sha256"],
          "statements": [], "ordering_questions": questions})
    write(folder / "PUBLIC_QUESTIONS.json", {"questions": questions})
    frozen = [batch / "ROLE_ASSIGNMENTS.json"] + [tools / name for name in anchors.RUNTIME_TOOLS]
    frozen += [batch / "scripts" / name for name in anchors.RUNTIME_SCRIPTS]
    write(batch / "MASTER_FREEZE.json", {"synthetic_fixture_only": True, "files": [gate.binding(p) for p in frozen]})
    # The actual executing review/gate/wrapper dependencies must come from the
    # fixture's frozen tool paths, not a cached implementation in the outer batch.
    global_names = {"public_anchor_v1": "anchors", "review_provenance_gate_v2": "gate", "run_guarded_questions_v2": "wrapper"}
    for name in (*global_names, "scene_review_provenance_v1"):
        module = load_fixture_module(name, tools)
        if name in global_names:
            globals()[global_names[name]] = module
    return unit, folder, load_helper(tools / "review_v4.py")


def semantic_judgment(folder, phase):
    plan = read(folder / "ORDERING_ANCHOR_PLAN.json")
    reviews = []
    for slot in plan["slots"]:
        review = {"candidate_id": slot["candidate_id"], "stem_valid": True, "scope_valid": True,
                  "coherent": True, "unique_order": True, "neutral_stem_ok": True,
                  "reason": "Synthetic fixture events are exact and separate.", "items": {},
                  "derived_order": [a["item_id"] for a in slot["anchors"]], "alternative_orders": []}
        for anchor in slot["anchors"]:
            review["items"][anchor["item_id"]] = {"proposition": anchor["full_turn_quote"],
                **{name: True for name in ("supported", "actor_ok", "modality_ok", "state_ok", "neutral_stem_ok", "single_event_ok")},
                "reason": "Exact synthetic event.", "anchor_id": anchor["anchor_id"],
                "frozen_full_turn_sha256": anchor["full_turn_sha256"],
                "order_anchor": {"turn_ref": anchor["turn_ref"], "speaker": anchor["speaker"], "quote": anchor["full_turn_quote"]},
                "support_refs": [anchor["turn_ref"]], "alternative_matching_anchors": []}
        reviews.append(review)
    attestation = {"full_source_read": True, "all_continuation_constraints_read": True,
                   "private_answers_read_before_save": False, "public_anchor_plan_full_read": True,
                   "source_read_before_anchor_plan_and_public_content": True}
    if phase == "public":
        attestation["final_public_questions_actually_reread"] = True
        return {"read_attestation": attestation, "questions": {r["candidate_id"]: r for r in reviews}, "findings": []}
    return {"read_attestation": attestation, "multiselect_stem_valid": True, "statements": {},
            "ordering_questions": reviews, "findings": []}


def review_paths(unit, phase):
    digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
    return unit / "evidence/reads" / f"{phase}_root_{digest}"


def record(batch, unit, folder, helper, phase):
    with redirect_stdout(io.StringIO()):
        helper.start(batch, ROOT, phase, REVIEWER)
        raw = unit / f"synthetic_{phase}_judgment.json"
        write(raw, semantic_judgment(folder, phase))
        assert helper.preflight_review(batch, ROOT, phase, REVIEWER, raw)["pass"]
        helper.save_review(batch, ROOT, phase, REVIEWER, raw)


def rebind(unit, folder, phase):
    """Simulate internally consistent tampering; gate must check content too."""
    directory = review_paths(unit, phase)
    review_path = folder / gate.OUTPUTS[phase]
    review = read(review_path)
    for field, name in (("read_context_binding", "READ_CONTEXT.json"), ("raw_judgment_binding", "RAW_JUDGMENT.json"),
                        ("input_render_binding", "INPUT_RENDER.txt")):
        review[field] = gate.binding(directory / name)
    write(review_path, review)
    receipt = read(directory / "SAVE_RECEIPT.json")
    receipt.update(read_context_binding=gate.binding(directory / "READ_CONTEXT.json"),
                   raw_judgment_binding=gate.binding(directory / "RAW_JUDGMENT.json"), output_binding=gate.binding(review_path))
    write(directory / "SAVE_RECEIPT.json", receipt)


class PublicAnchorWorkflowTests(unittest.TestCase):
    def setUp(self):
        names = ("public_anchor_v1", "review_provenance_gate_v2", "run_guarded_questions_v2", "scene_review_provenance_v1")
        original = {name: sys.modules.get(name) for name in names}
        original_globals = {name: globals()[name] for name in ("anchors", "gate", "wrapper")}
        def restore_modules():
            for name, module in original.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            globals().update(original_globals)
        self.addCleanup(restore_modules)
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic_anchor_", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name)
        self.unit, self.folder, self.helper = make_fixture(self.batch)

    def codes(self, phase="asset"):
        return {row["code"] for row in gate.check(self.batch, ROOT, phase)["errors"]}

    def test_real_read_save_gate_end_to_end(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        outcome = gate.check(self.batch, ROOT, "asset")
        self.assertTrue(outcome["pass"], outcome["errors"])
        self.assertFalse(outcome["semantic_quality_certified"])
        render = (review_paths(self.unit, "asset") / "INPUT_RENDER.txt").read_text()
        self.assertLess(render.index("ALL_CONTINUATION_CONSTRAINTS"), render.index("FULL_CUMULATIVE_SOURCE"))
        self.assertLess(render.index("FULL_CUMULATIVE_SOURCE"), render.index("PUBLIC_ORDERING_ANCHOR_PLAN_FULL"))
        self.assertLess(render.index("PUBLIC_ORDERING_ANCHOR_PLAN_FULL"), render.index("PUBLIC_CONTENT_FULL"))

    def test_public_end_to_end_recurses_asset_lineage(self):
        for phase in ("asset", "public"):
            record(self.batch, self.unit, self.folder, self.helper, phase)
        result = gate.check(self.batch, ROOT, "public")
        self.assertTrue(result["pass"], result["errors"])

    def test_identity_mismatch_prevents_read(self):
        with self.assertRaisesRegex(ValueError, "EXECUTOR_NOT_ASSIGNED_REVIEWER"):
            self.helper.start(self.batch, ROOT, "asset", AUTHOR)
        self.assertFalse((self.unit / "evidence/reads").exists())

    def test_private_plan_field_rejected_before_render(self):
        p = self.folder / "ORDERING_ANCHOR_PLAN.json"
        obj = read(p); obj["true_text"] = "Forbidden answer hint"; write(p, obj)
        with self.assertRaisesRegex(ValueError, "PLAN_FIELDS_NOT_EXACT_PUBLIC_SCHEMA"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)
        self.assertFalse((self.unit / "evidence/reads").exists())

    def test_nested_private_plan_field_rejected(self):
        p = self.folder / "ORDERING_ANCHOR_PLAN.json"
        obj = read(p); obj["slots"][0]["anchors"][0]["false_text"] = "Forbidden"; write(p, obj)
        with self.assertRaisesRegex(ValueError, "ANCHOR_FIELDS_NOT_EXACT_PUBLIC_SCHEMA"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)

    def test_public_asset_private_metadata_never_renders(self):
        path = self.folder / "PUBLIC_ASSETS.json"
        original = read(path)
        variants = []
        obj = deepcopy(original); obj["answer_key"] = ["I1"]; variants.append(obj)
        obj = deepcopy(original); obj["statements"] = [{"statement_id": "s1", "proposition": "Fixture", "private": True}]; variants.append(obj)
        obj = deepcopy(original); obj["ordering_questions"][0]["Gold"] = ["I1"]; variants.append(obj)
        obj = deepcopy(original); obj["ordering_questions"][0]["items"]["I1"] = {"text": "Fixture", "answer_key": True}; variants.append(obj)
        obj = deepcopy(original); obj["ordering_failures"] = [{"question_type": "ordering", "slot_id": "1", "candidate_id": "fixture", "reason": "author_failure", "failure": {"private": True}}]; variants.append(obj)
        for obj in variants:
            with self.subTest(injected=obj):
                write(path, obj)
                with self.assertRaisesRegex(ValueError, "PUBLIC_"):
                    self.helper.start(self.batch, ROOT, "asset", REVIEWER)
                self.assertFalse((self.unit / "evidence/reads").exists())

    def test_public_question_private_metadata_never_renders(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        path = self.folder / "PUBLIC_QUESTIONS.json"
        original = read(path)
        variants = []
        obj = deepcopy(original); obj["private"] = {"gold": []}; variants.append(obj)
        obj = deepcopy(original); obj["questions"][0]["answer_key"] = ["I1"]; variants.append(obj)
        obj = deepcopy(original); obj["questions"][0]["items"]["I1"] = {"text": "Fixture", "gold": True}; variants.append(obj)
        for value in ({"A": "Fixture", "B": "Fixture", "C": "Fixture", "D": "Fixture", "E": "Fixture", "F": "Fixture", "Gold": "A"},
                      {label: {"text": "Fixture", "private": True} for label in "ABCDEF"}):
            obj = deepcopy(original)
            q = deepcopy(obj["questions"][0]); q.pop("items")
            q.update(candidate_id="synthetic-ms", question_type="multi_select", options=value)
            obj["questions"].append(q); variants.append(obj)
        for obj in variants:
            with self.subTest(injected=obj):
                write(path, obj)
                with self.assertRaisesRegex(ValueError, "PUBLIC_"):
                    self.helper.start(self.batch, ROOT, "public", REVIEWER)
                self.assertFalse(review_paths(self.unit, "public").exists())

    def test_gate_independently_rejects_public_private_metadata(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        path = self.folder / "PUBLIC_ASSETS.json"
        obj = read(path); obj["ordering_questions"][0]["private"] = {"answer_key": "I1"}; write(path, obj)
        self.assertIn("ORDERING_ANCHOR_PLAN:INVALID_PUBLIC_ANCHORS", self.codes())

    def test_wrong_source_anchor_hash_prevents_read(self):
        p = self.folder / "ORDERING_ANCHOR_PLAN.json"
        obj = read(p); obj["slots"][0]["anchors"][0]["full_turn_sha256"] = "0" * 64; write(p, obj)
        with self.assertRaisesRegex(ValueError, "ANCHOR_FULL_TURN_HASH_MISMATCH"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)

    def test_qa_receipt_must_bind_plan(self):
        p = self.folder / "QA_AUTHOR_RECEIPT.json"
        obj = read(p); del obj["ordering_anchor_plan_binding"]; write(p, obj)
        with self.assertRaisesRegex(ValueError, "QA_AUTHOR_ANCHOR_PLAN_BINDING"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)

    def test_wrong_frozen_turn_rejected_and_first_judgment_preserved(self):
        with redirect_stdout(io.StringIO()):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)
        raw = semantic_judgment(self.folder, "asset")
        item = next(iter(raw["ordering_questions"][0]["items"].values()))
        item["order_anchor"]["turn_ref"] = "R02:A:S02:T001"
        p = self.unit / "synthetic_invalid_judgment.json"; write(p, raw)
        with redirect_stdout(io.StringIO()):
            check = self.helper.preflight_review(self.batch, ROOT, "asset", REVIEWER, p)
        self.assertFalse(check["pass"])
        self.assertIn("REVIEW_ANCHOR_OUTSIDE_FROZEN_TURN", check["error"])
        with self.assertRaisesRegex(ValueError, "PASSING_SCHEMA_PRECHECK_IDENTITY_REQUIRED"):
            self.helper.save_review(self.batch, ROOT, "asset", REVIEWER, p)
        directory = review_paths(self.unit, "asset")
        self.assertEqual(read(Path(check["candidate_binding"]["path"])), raw)
        self.assertFalse((directory / "RAW_JUDGMENT.json").exists())
        self.assertFalse((directory / "SAVE_FAILURE.json").exists())
        self.assertFalse((self.folder / "ASSET_SOURCE_JUDGMENTS.json").exists())

    def test_no_receipt_blocks_even_with_valid_saved_review(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        (review_paths(self.unit, "asset") / "SAVE_RECEIPT.json").unlink()
        self.assertIn("SAVE_RECEIPT:FILE_MISSING", self.codes())

    def test_private_extra_input_rejected_even_when_hashed(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        private = self.folder / "PRIVATE_DESIGN.json"; write(private, {"synthetic_forbidden": True})
        p = review_paths(self.unit, "asset") / "READ_CONTEXT.json"
        obj = read(p); obj["input_bindings"].append(gate.binding(private)); write(p, obj); rebind(self.unit, self.folder, "asset")
        self.assertIn("READ_CONTEXT:INPUT_SET_MISMATCH", self.codes())

    def test_wrong_anchor_id_even_consistent_raw_review_is_blocked(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        for p in (review_paths(self.unit, "asset") / "RAW_JUDGMENT.json", self.folder / "ASSET_SOURCE_JUDGMENTS.json"):
            obj = read(p); next(iter(obj["ordering_questions"][0]["items"].values()))["anchor_id"] = "wrong-id"; write(p, obj)
        rebind(self.unit, self.folder, "asset")
        self.assertTrue(any(code.startswith("REVIEW:INVALID_FROZEN_ANCHOR:") for code in self.codes()))

    def test_forged_anchor_quote_even_consistent_raw_review_is_blocked(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        for path in (review_paths(self.unit, "asset") / "RAW_JUDGMENT.json", self.folder / "ASSET_SOURCE_JUDGMENTS.json"):
            obj = read(path)
            next(iter(obj["ordering_questions"][0]["items"].values()))["order_anchor"]["quote"] = "Forged event absent from frozen source."
            write(path, obj)
        rebind(self.unit, self.folder, "asset")
        result = gate.check(self.batch, ROOT, "asset")
        self.assertFalse(result["pass"])
        self.assertTrue(any(row.get("detail") == "REVIEW_ANCHOR_QUOTE_NOT_EXACT" for row in result["errors"]), result)

    def test_empty_or_repeated_review_anchor_quote_is_rejected(self):
        plan = read(self.folder / "ORDERING_ANCHOR_PLAN.json")
        planned = plan["slots"][0]["anchors"][0]
        for quote, expected in (("", "REVIEW_ANCHOR_QUOTE_NOT_EXACT"), (" ", "REVIEW_ANCHOR_QUOTE_NOT_EXACT"), ("e", "REVIEW_ANCHOR_QUOTE_NOT_UNIQUE")):
            item = deepcopy(semantic_judgment(self.folder, "asset")["ordering_questions"][0]["items"][planned["item_id"]])
            item["order_anchor"]["quote"] = quote
            with self.subTest(quote=quote), self.assertRaisesRegex(ValueError, expected):
                anchors.validate_review_anchor(item, planned)

    def test_private_gold_after_asset_save_blocks_compose_without_writes(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        write(self.folder / "PRIVATE_GOLD.json", {"gold": "injected after save"})
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        calls = []
        result, code = wrapper.run(self.batch, ROOT, "compose", downstream=lambda *a: calls.append(a))
        self.assertEqual(code, 2); self.assertFalse(result["downstream_invoked"]); self.assertEqual(calls, [])
        self.assertIn("QA_STAGE:PRIVATE_GOLD_EXISTS_BEFORE_CONSUMPTION", self.codes())
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_private_gold_after_public_save_blocks_finalize_without_writes(self):
        for phase in ("asset", "public"):
            record(self.batch, self.unit, self.folder, self.helper, phase)
        write(self.folder / "PRIVATE_GOLD.json", {"gold": "injected after save"})
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        calls = []
        result, code = wrapper.run(self.batch, ROOT, "finalize", downstream=lambda *a: calls.append(a))
        self.assertEqual(code, 2); self.assertFalse(result["downstream_invoked"]); self.assertEqual(calls, [])
        self.assertIn("QA_STAGE:PRIVATE_GOLD_EXISTS_BEFORE_CONSUMPTION", self.codes("public"))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_runtime_closure_rejects_changed_tool_or_native_script(self):
        paths = [self.batch / "tools" / n for n in anchors.RUNTIME_TOOLS]
        paths += [self.batch / "scripts" / n for n in anchors.RUNTIME_SCRIPTS]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes(); path.write_bytes(original + b"\n# changed frozen implementation\n")
                try:
                    with self.assertRaisesRegex(ValueError, "RUNTIME_FROZEN_HASH_MISMATCH"):
                        self.helper.start(self.batch, ROOT, "asset", REVIEWER)
                    self.assertIn("GATE_RUNTIME:NOT_FROZEN_OR_LOADED_CODE_CHANGED", self.codes())
                    calls = []
                    result, code = wrapper.run(self.batch, ROOT, "compose", downstream=lambda *a: calls.append(a))
                    self.assertEqual(code, 2); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])
                finally:
                    path.write_bytes(original)
        self.assertFalse((self.unit / "evidence/reads").exists())

    def test_wrapper_rejects_loaded_foreign_gate_function(self):
        original = wrapper.check
        wrapper.check = lambda *a: {"pass": True, "errors": []}
        try:
            calls = []
            result, code = wrapper.run(self.batch, ROOT, "compose", downstream=lambda *a: calls.append(a))
            self.assertEqual(code, 2); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])
            self.assertIn("RUNTIME_LOADED_CODE_MISMATCH", result["gate"]["errors"][0]["detail"])
        finally:
            wrapper.check = original

    def test_subprocess_changed_gate_cannot_turn_denial_into_dispatch(self):
        path = self.batch / "tools/review_provenance_gate_v2.py"
        with path.open("a", encoding="utf-8") as stream:
            stream.write('\ndef check(*args, **kwargs):\n    return {"pass": True, "errors": [], "status": "FORGED_PASS"}\n')
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        completed = subprocess.run([sys.executable, "-B", str(self.batch / "tools/run_guarded_questions_v2.py"),
                                    "compose", "--batch", str(self.batch), "--root", ROOT], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertFalse(result["downstream_invoked"])
        self.assertEqual(result["status"], "BLOCKED_BEFORE_DOWNSTREAM")
        self.assertIn("RUNTIME_FROZEN_HASH_MISMATCH", result["gate"]["errors"][0]["detail"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_changed_render_even_rehashed_is_blocked(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        directory = review_paths(self.unit, "asset")
        p = directory / "INPUT_RENDER.txt"; p.write_text(p.read_text() + "Unexpected private hint\n")
        c = directory / "READ_CONTEXT.json"; obj = read(c); obj["input_render_binding"] = gate.binding(p); write(c, obj)
        rebind(self.unit, self.folder, "asset")
        self.assertIn("INPUT_RENDER:NOT_EXACT_CURRENT_INPUT_RENDER", self.codes())

    def test_public_cannot_hide_broken_asset_lineage(self):
        for phase in ("asset", "public"):
            record(self.batch, self.unit, self.folder, self.helper, phase)
        (review_paths(self.unit, "asset") / "SAVE_RECEIPT.json").unlink()
        self.assertIn("PRIOR_ASSET_PROVENANCE:SAVE_RECEIPT:FILE_MISSING", self.codes("public"))

    def test_source_first_attestation_required_by_save(self):
        with redirect_stdout(io.StringIO()):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)
        obj = semantic_judgment(self.folder, "asset"); obj["read_attestation"]["source_read_before_anchor_plan_and_public_content"] = False
        raw = self.unit / "synthetic_invalid_attestation.json"; write(raw, obj)
        with redirect_stdout(io.StringIO()):
            check = self.helper.preflight_review(self.batch, ROOT, "asset", REVIEWER, raw)
        self.assertFalse(check["pass"])
        self.assertIn("EXPLICIT_SOURCE_FIRST_SEQUENCE_REQUIRED", check["error"])
        with self.assertRaisesRegex(ValueError, "PASSING_SCHEMA_PRECHECK_IDENTITY_REQUIRED"):
            self.helper.save_review(self.batch, ROOT, "asset", REVIEWER, raw)

    def test_failed_gate_has_zero_downstream_calls_and_zero_writes(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        (review_paths(self.unit, "asset") / "SAVE_RECEIPT.json").unlink()
        before = {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()}
        calls = []
        result, code = wrapper.run(self.batch, ROOT, "compose", downstream=lambda *a: calls.append(a))
        self.assertEqual(code, 2); self.assertFalse(result["downstream_invoked"]); self.assertEqual(calls, [])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.batch.rglob("*") if p.is_file()})

    def test_success_dispatch_and_check_only(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        calls = []
        def downstream(*args):
            calls.append(args); return SimpleNamespace(returncode=0, stdout="synthetic", stderr="")
        result, code = wrapper.run(self.batch, ROOT, "compose", check_only=True, downstream=downstream)
        self.assertEqual(code, 0); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])
        result, code = wrapper.run(self.batch, ROOT, "compose", downstream=downstream)
        self.assertEqual(code, 0); self.assertEqual(len(calls), 1); self.assertTrue(result["downstream_invoked"])

    def test_failed_public_gate_has_zero_finalize_calls(self):
        for phase in ("asset", "public"):
            record(self.batch, self.unit, self.folder, self.helper, phase)
        (review_paths(self.unit, "public") / "SAVE_RECEIPT.json").unlink()
        calls = []
        result, code = wrapper.run(self.batch, ROOT, "finalize", downstream=lambda *a: calls.append(a))
        self.assertEqual(code, 2); self.assertEqual(calls, []); self.assertFalse(result["downstream_invoked"])

    def test_missing_scene_adapter_prevents_read_no_fallback(self):
        (self.batch / "tools/scene_controller_v120.py").unlink()
        with self.assertRaisesRegex(ValueError, "V120_SCENE_CONTROLLER_ADAPTER_REQUIRED"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)

    def test_selection_a_plan_mismatch_prevents_read(self):
        p = self.unit / "selection.json"
        obj = read(p); obj["a_continuations"][1]["scene_ref"] = "R02:A:S01"; write(p, obj)
        with self.assertRaisesRegex(ValueError, "A_SELECTION_THREE_SCENE_PLAN_REQUIRED"):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)

    def test_selection_changed_after_read_cannot_save(self):
        with redirect_stdout(io.StringIO()):
            self.helper.start(self.batch, ROOT, "asset", REVIEWER)
        p = self.unit / "selection.json"
        obj = read(p); obj["a_continuations"][1]["thread"] = "Changed to unrelated synthetic thread."; write(p, obj)
        judgment = self.unit / "synthetic_asset_judgment.json"; write(judgment, semantic_judgment(self.folder, "asset"))
        with self.assertRaisesRegex(ValueError, "READ_INPUTS_CHANGED"):
            self.helper.save_review(self.batch, ROOT, "asset", REVIEWER, judgment)

    def test_selection_omission_blocks_gate_even_with_refreshed_hashes(self):
        record(self.batch, self.unit, self.folder, self.helper, "asset")
        path = review_paths(self.unit, "asset") / "READ_CONTEXT.json"
        obj = read(path); obj["selection_binding"] = None
        obj["input_bindings"] = [r for r in obj["input_bindings"] if r["path"] != str(self.unit / "selection.json")]
        write(path, obj); rebind(self.unit, self.folder, "asset")
        self.assertIn("READ_CONTEXT:selection_binding:BINDING_MISSING", self.codes())
        self.assertIn("READ_CONTEXT:INPUT_SET_MISMATCH", self.codes())

    def prepare_scene(self, scene):
        write(self.unit / "SYNTHETIC_SCENE_AWAITING.json", {"scene": scene})
        target = self.folder / "scenes" / scene.removeprefix("R02:").replace(":", "_")
        (target / "SOURCE_REVIEW.json").unlink()
        source = read(self.folder / "CUMULATIVE_SOURCE.json")
        source["turns"] = source["turns"][:(anchors.SCENES.index(scene) + 1) * 2]
        write(target / "CUMULATIVE_SOURCE.json", source)
        draft = {"author": AUTHOR, "source_id": "synthetic-native-id" if ":B:" in scene else None,
                 "turns": [row for row in source["turns"] if row["scene_ref"] == scene]}
        write(target / "EXPANSION_DRAFT.json", draft)
        if ":B:" in scene:
            native_path = self.unit / "synthetic_native.json"
            write(native_path, {"turns": [{"turn_id": "synthetic-t1", "speaker": "Fixture", "text": "Synthetic source event."}]})
            write(self.unit / "SOURCE_RESERVATIONS.json", {"allocation": [{"root_id": ROOT, "scene_ref": scene,
                  "source_item_path": str(native_path), "source_item_sha256": gate.sha(native_path),
                  "selection_path": str(self.unit / "selection.json"), "selection_sha256": gate.sha(self.unit / "selection.json")}]})
        raw = {"read_attestation": {"full_source_read": True, "all_continuation_constraints_read": True},
               "decision": "READY", "checks": {name: {"status": "PASS", "reason": "Synthetic fixture."} for name in self.helper.CHECKS},
               "findings": [], "baseline_constraint_review": {"status": "PASS", "reason": "Synthetic fixture.",
                   "constraint_results": [{"index": 1, "constraint": "Synthetic events remain separate.", "status": "PASS",
                                           "reason": "Synthetic fixture.", "evidence_turn_refs": [source["turns"][-1]["turn_ref"]]}],
                   "new_blocking_findings": []}}
        if ":B:" in scene:
            raw["read_attestation"]["native_event_full_read"] = True
        raw_path = self.unit / "synthetic_scene_judgment.json"; write(raw_path, raw)
        with redirect_stdout(io.StringIO()):
            self.helper.start(self.batch, ROOT, "scene", REVIEWER, scene)
            self.assertTrue(self.helper.preflight_review(self.batch, ROOT, "scene", REVIEWER, raw_path, scene)["pass"])
            self.helper.save_review(self.batch, ROOT, "scene", REVIEWER, raw_path, scene)
        return target

    def test_scene_a_read_save_gate_slice(self):
        self.prepare_scene("R02:A:S02")
        result = gate.check_scene(self.batch, ROOT, "R02:A:S02")
        self.assertTrue(result["pass"], result["errors"])

    def test_scene_b_read_save_native_binding_slice(self):
        self.prepare_scene("R02:B:S01")
        result = gate.check_scene(self.batch, ROOT, "R02:B:S01")
        self.assertTrue(result["pass"], result["errors"])
        p = self.unit / "synthetic_native.json"; obj = read(p); obj["turns"][0]["text"] = "Changed"; write(p, obj)
        result = gate.check_scene(self.batch, ROOT, "R02:B:S01")
        self.assertFalse(result["pass"])
        self.assertTrue(any("HASH_MISMATCH" in r["code"] for r in result["errors"]))

    def test_ready_cannot_hide_failed_constraints_even_consistently_rehashed(self):
        scene = "R02:A:S02"
        target = self.prepare_scene(scene)
        digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
        directory = self.unit / "evidence/reads" / f"scene_R02_A_S02_{digest}"
        raw_path, review_path = directory / "RAW_JUDGMENT.json", target / "SOURCE_REVIEW.json"
        receipt_path = directory / "SAVE_RECEIPT.json"
        original_raw, original_review, original_receipt = read(raw_path), read(review_path), read(receipt_path)
        variants = [lambda c: c.update(status="FAIL"),
                    lambda c: c["constraint_results"][0].update(status="FAIL"),
                    lambda c: c.update(new_blocking_findings=["Synthetic explicit blocker."])]
        for change in variants:
            raw, review = deepcopy(original_raw), deepcopy(original_review)
            change(raw["baseline_constraint_review"]); change(review["baseline_constraint_review"])
            write(raw_path, raw)
            review["raw_judgment_binding"] = gate.binding(raw_path); write(review_path, review)
            receipt = deepcopy(original_receipt)
            receipt.update(raw_judgment_binding=gate.binding(raw_path), output_binding=gate.binding(review_path)); write(receipt_path, receipt)
            result = gate.check_scene(self.batch, ROOT, scene)
            self.assertFalse(result["pass"], result)
            self.assertIn("REVIEW:READY_WITH_FAILED_CONSTRAINT", {r["code"] for r in result["errors"]})

    def test_scene_wrong_executor_and_missing_receipt_block(self):
        target = self.prepare_scene("R02:A:S02")
        p = target / "SOURCE_REVIEW.json"; obj = read(p); obj["reviewer"] = AUTHOR; write(p, obj)
        digest = hashlib.sha256(REVIEWER.encode()).hexdigest()[:16]
        (self.unit / "evidence/reads" / f"scene_R02_A_S02_{digest}" / "SAVE_RECEIPT.json").unlink()
        result = gate.check_scene(self.batch, ROOT, "R02:A:S02")
        codes = {r["code"] for r in result["errors"]}
        self.assertIn("REVIEW:FIELD_MISMATCH:reviewer", codes)
        self.assertIn("SAVE_RECEIPT:FILE_MISSING", codes)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    unittest.main(verbosity=2)
