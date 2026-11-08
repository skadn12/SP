"""Fail-closed Day2 admission over the immutable Day1 schema.

This checks recorded bytes/chronology, not process isolation or causal truth.
No model inference, network call, or model-weight loading occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
DAY1 = HERE.parents[1] / "day1"
PINS = {
    "PROTOCOL_V1.json": "79c649b9234caec7d060a84ae6d3ccf3e58d10cd5a3cfc7c627e7e0a92a418fe",
    "MODEL_BINDINGS.json": "4f00ac34f2717785f1147e9c10ad02eac63fb2284c5d1b081d3c744a9a6b64d2",
    "TRAIN_ROOTS.json": "b331aee609fe95516dd69be41726c571c580db45a2dedf50138e5fbb1962d8e7",
    "diagnosis/validate_diagnosis.py": "9f3666ef80a59db0b060475f68b05cb4e99cd4ed2a14f3a539a959ad920de524",
    "diagnosis/MEMORY_SCHEMA.json": "0b12bc8feaec657238151c194162c9d84ee3eb0ea3f57cf14dc7169f3bb4de63",
    "diagnosis/FACT_MEMORY_SCHEMA.json": "35247c30361d95679c2ec0bdc50b3ba68e62a38e87aab66a588853749dd6a198",
    "diagnosis/WEAKNESS_TAXONOMY.json": "4af46fc29ac79828c90c5354b58d5d5f9ab09f86831f58ed30dd4bde77a08b1f",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


class Rejected(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise Rejected(message)


def stamp(value):
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        need(at.tzinfo is not None, "timestamp must have a timezone")
        return at
    except (AttributeError, TypeError, ValueError) as e:
        raise Rejected("invalid timezone-aware timestamp: " + str(value)) from e


def family(provenance):
    return provenance["root_id"] + "::" + provenance["family_id"]


def all_refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for v in value.values():
                yield from all_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_refs(v)


class Controller:
    def __init__(self, target_path, sidecar_path=None):
        self.target_path = Path(target_path).resolve()
        self.root = self.target_path.parent
        self.sidecar_path = Path(sidecar_path).resolve() if sidecar_path else None
        self.checked_refs = {}
        self.models = {}
        self.run_cache = {}
        self.observation_cache = {}
        self.tokenizers = {}
        self.check_count = 0
        for name, expected in PINS.items():
            need(sha(DAY1 / name) == expected, "frozen Day1 input changed: " + name)
        spec = importlib.util.spec_from_file_location("immutable_day1_diagnosis", DAY1 / "diagnosis/validate_diagnosis.py")
        self.legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.legacy)
        self.protocol = json.loads((DAY1 / "PROTOCOL_V1.json").read_text())
        self.frozen_cap = self.protocol["diagnosis"]["max_evidence_output_tokens"]
        need(self.frozen_cap == 1024, "frozen diagnosis cap must be 1024")
        self.roots = set(json.loads((DAY1 / "TRAIN_ROOTS.json").read_text())["root_ids"])
        self.binding = json.loads((DAY1 / "MODEL_BINDINGS.json").read_text())
        self.target = json.loads(self.target_path.read_text())
        self.sidecar = json.loads(self.sidecar_path.read_text()) if self.sidecar_path else None

    def check(self, condition, message):
        self.check_count += 1
        need(condition, message)

    def path(self, ref):
        self.check(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "artifact reference requires exactly path and sha256")
        p = Path(ref["path"])
        p = (self.root / p).resolve() if not p.is_absolute() else p.resolve()
        self.check(p.is_file(), "missing bound artifact: " + str(p))
        key = (str(p), ref["sha256"])
        if key not in self.checked_refs:
            self.check(sha(p) == ref["sha256"], "artifact byte hash mismatch: " + str(p))
            self.checked_refs[key] = True
        return p

    def load(self, ref, record_type=None):
        p = self.path(ref)
        try:
            value = json.loads(p.read_text())
        except (UnicodeError, json.JSONDecodeError) as e:
            raise Rejected("expected JSON artifact: " + str(p)) from e
        self.check(isinstance(value, dict), "bound JSON record must be an object")
        if record_type:
            self.check(value.get("record_type") == record_type, "wrong raw record type; expected " + record_type)
        return value

    def same_ref(self, a, b):
        return self.path(a) == self.path(b) and a["sha256"] == b["sha256"]

    def provenance(self, value):
        self.check(value.get("split") == "train" and value.get("namespace") == "train.feedback", "only train.feedback provenance is permitted")
        self.check(value.get("root_id") in self.roots, "root is absent from frozen Train manifest")
        self.check(all(value.get(k) for k in ["family_id", "source_id", "source_sha256"]), "incomplete source/family provenance")

    def register_models(self):
        entries = self.sidecar["models"]
        self.check(isinstance(entries, list), "models must be a list")
        for model in entries:
            mid = model["model_id"]
            self.check(mid not in self.models, "duplicate model registry identifier")
            self.path(model["binding"])
            self.models[mid] = model
        for mid, model in self.models.items():
            parent = model["parent_model_id"]
            if parent is None:
                self.check(model["canonical_model_id"] == "S0_M_clean_merged", "unbound or unrelated root model")
                self.check(self.same_ref(model["binding"], reference(DAY1 / "MODEL_BINDINGS.json")), "S0 registry must bind frozen model manifest")
                self.check(model.get("training_record") is None, "S0 must not invent a new training record")
            else:
                self.check(parent in self.models, "missing parent model registry entry")
                manifest = self.load(model["binding"], "model_file_manifest")
                self.check(manifest["model_id"] == mid and manifest["canonical_model_id"] == model["canonical_model_id"], "derived model manifest identity mismatch")
                self.check(bool(manifest["files"]) and any(Path(f["path"]).suffix in {".safetensors", ".pt", ".bin"} for f in manifest["files"]), "derived model requires bound weight files")
                for ref in manifest["files"]:
                    self.path(ref)
                training = self.load(model["training_record"], "training_lineage")
                self.check(training["parent_model_id"] == parent and training["child_model_id"] == mid, "training lineage identifiers mismatch")
                self.check(self.same_ref(training["parent_binding"], self.models[parent]["binding"]), "training parent bytes mismatch")
                self.check(self.same_ref(training["child_binding"], model["binding"]), "training child bytes mismatch")
                self.check(isinstance(training["training_question_ids"], list) and isinstance(training["training_family_keys"], list), "training exclusion manifests must be explicit")
                self.check(stamp(training["started_at"]) < stamp(training["completed_at"]), "invalid training chronology")
        for mid in self.models:
            self.ancestors(mid)

    def ancestors(self, model_id):
        result = []
        while model_id is not None:
            self.check(model_id in self.models and model_id not in result, "unknown/cyclic model lineage")
            result.append(model_id)
            model_id = self.models[model_id]["parent_model_id"]
        return result

    def equivalent(self, one, two):
        self.check(one in self.models and two in self.models, "model missing from bound registry")
        return self.models[one]["canonical_model_id"] == self.models[two]["canonical_model_id"] and self.same_ref(self.models[one]["binding"], self.models[two]["binding"])

    def descendant(self, child, ancestor):
        return any(self.equivalent(x, ancestor) for x in self.ancestors(child))

    def tokenizer(self, ref):
        p = self.path(ref)
        bound = [f for f in self.binding["models"]["S0_M_clean_merged"]["files"] if Path(f["path"]).name == "tokenizer.json"]
        self.check(len(bound) == 1 and ref["sha256"] == bound[0]["sha256"], "tokenizer must match the frozen S0 tokenizer bytes")
        if str(p) not in self.tokenizers:
            from tokenizers import Tokenizer
            self.tokenizers[str(p)] = Tokenizer.from_file(str(p))
        return self.tokenizers[str(p)]

    def solver_run(self, ref):
        key = str(self.path(ref))
        if key in self.run_cache:
            return self.run_cache[key]
        run = self.load(ref, "solver_run")
        request = self.load(run["request"], "solver_request")
        response = self.load(run["response"], "solver_response")
        fields = {"record_type", "request_id", "model_id", "model_binding", "question_id", "provenance", "task_type", "mode", "max_output_tokens", "public_question", "source_input", "display_mapping", "prompt", "gold_visible", "author_evidence_visible", "started_at"}
        self.check(set(request) == fields, "request must use exact public-input field whitelist")
        self.provenance(request["provenance"])
        self.check(request["model_id"] in self.models, "request uses unknown model")
        self.check(self.same_ref(request["model_binding"], self.models[request["model_id"]]["binding"]), "actual Solver request binds a different model artifact")
        self.check(request["mode"] in {"direct", "diagnostic"}, "unsupported Solver mode")
        self.check(request["gold_visible"] is False and request["author_evidence_visible"] is False, "Solver request exposes Gold or author evidence")
        public = self.load(request["public_question"])
        banned = {"gold", "gold_answer", "gold_content_ids", "answer", "answers", "correct_answer", "correct_indices", "author_evidence", "reference_evidence", "solver_scores"}
        def check_public(obj):
            if isinstance(obj, dict):
                self.check(not (set(obj) & banned), "private field in public question input")
                for val in obj.values(): check_public(val)
            elif isinstance(obj, list):
                for val in obj: check_public(val)
        check_public(public)
        source = self.path(request["source_input"])
        self.check(sha(source) == request["provenance"]["source_sha256"], "request source bytes differ from provenance")
        mapping = self.load(request["display_mapping"])
        displayed = mapping["display_to_content"]
        texts = mapping["content_to_text"]
        self.check(4 <= len(displayed) <= 6 and len(set(displayed.values())) == len(displayed) and set(displayed.values()) == set(texts), "display/content mapping is not a 4/5/6-item bijection")
        keys = list(displayed)
        expected_keys = list("ABCDEF"[:len(keys)]) if request["task_type"] == "multi_select" else ["I" + str(i + 1) for i in range(len(keys))]
        self.check(set(keys) == set(expected_keys), "display mapping has illegal labels")
        public_texts = public.get("options", public.get("items", {}))
        if isinstance(public_texts, dict): public_texts = list(public_texts.values())
        self.check(set(public_texts) == set(texts.values()), "display mapping changes public question contents")
        self.check(isinstance(request["prompt"], str) and request["prompt"], "actual rendered prompt is missing")
        self.check(all(text in request["prompt"] for text in texts.values()) and public["question"] in request["prompt"], "rendered prompt omits an item or question")
        source_json = json.loads(source.read_text())
        self.check(isinstance(source_json.get("turns"), list), "bound source must expose complete accepted turns")
        self.check(all(t["text"] in request["prompt"] for t in source_json["turns"]), "rendered prompt omits part of complete source")
        self.check(response["request_id"] == request["request_id"] and response["model_id"] == request["model_id"] and response["question_id"] == request["question_id"], "request/response identifiers mismatch")
        self.check(stamp(request["started_at"]) < stamp(response["committed_at"]), "response must be committed after request start")
        train_ref = self.models[request["model_id"]].get("training_record")
        if train_ref:
            training = self.load(train_ref, "training_lineage")
            self.check(stamp(training["completed_at"]) < stamp(request["started_at"]), "Solver response uses a checkpoint before its training completed")
        ids = response["output_token_ids"]
        self.check(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), "original output token IDs are required")
        count = len(ids)
        self.check(type(request["max_output_tokens"]) is int and 0 <= count <= request["max_output_tokens"], "actual output exceeds request budget")
        cap = self.frozen_cap if request["mode"] == "diagnostic" else self.protocol["evaluation"]["primary_direct_protocol"]["max_output_tokens"]
        self.check(0 < request["max_output_tokens"] <= cap and count <= cap, "output budget exceeds frozen " + str(cap) + " token cap")
        decoded = self.tokenizer(run["tokenizer"]).decode(ids, skip_special_tokens=True)
        self.check(decoded == response["text"], "raw text is inconsistent with bound original output token IDs")
        self.check(response["format_valid"] is True and response["full_evidence_visible"] is True, "invalid format or incomplete evidence cannot establish a weakness")
        self.check(isinstance(response["prediction_content_ids"], list), "response must retain content prediction")
        if request["mode"] == "direct":
            match = re.fullmatch(r"\s*<box>\s*(.*?)\s*</box>\s*", response["text"], re.DOTALL)
            self.check(match is not None, "direct raw answer is not one valid answer box")
            labels = [x.strip().upper() for x in match.group(1).split(",")]
        else:
            answer = json.loads(response["text"])
            self.check(set(answer) == {"evidence", "answer"}, "diagnostic response is not the frozen evidence/answer object")
            self.check(len(answer["evidence"]) == len(displayed) and {e["label"] for e in answer["evidence"]} == set(displayed), "diagnostic evidence does not cover each displayed item once")
            self.check(all(set(e) == {"label", "status", "turn_ref", "quote"} and e["status"] in {"supported", "contradicted", "not_supported", "ambiguous"} for e in answer["evidence"]), "diagnostic item evidence has invalid fields/status")
            labels = answer["answer"]
        self.check(bool(labels) and len(labels) == len(set(labels)) and set(labels) <= set(displayed), "raw prediction has unknown or repeated display labels")
        if request["task_type"] == "ordering":
            self.check(set(labels) == set(displayed), "raw ordering is not a complete permutation")
        self.check([displayed[label] for label in labels] == response["prediction_content_ids"], "parsed content prediction differs from original raw answer/mapping")
        result = {"record": run, "request": request, "response": response, "ref": ref, "actual_tokens": count}
        self.run_cache[key] = result
        return result

    def observation(self, ref):
        key = str(self.path(ref))
        if key in self.observation_cache:
            return self.observation_cache[key]
        obs = self.load(ref, "gold_evaluation")
        run = self.solver_run(obs["solver_run"])
        request, response = run["request"], run["response"]
        self.check(obs["question_id"] == request["question_id"] and obs["provenance"] == request["provenance"] and obs["model_id"] == request["model_id"], "Gold evaluation is unrelated to bound Solver run")
        self.check(obs.get("gold_read_scope") == "this_scoring_run" and isinstance(obs.get("gold_read_scope_id"), str) and bool(obs["gold_read_scope_id"]), "Gold-read chronology must identify this scoring run; no global first-read claim")
        self.check(stamp(response["committed_at"]) < stamp(obs["gold_first_read_at"]) <= stamp(obs["completed_at"]), "Solver answer must be committed strictly before Gold read")
        gold = self.load(obs["gold_reference"], "private_gold")
        self.check(gold["question_id"] == request["question_id"] and gold["task_type"] == request["task_type"], "Gold record question/type mismatch")
        predicted, correct = response["prediction_content_ids"], gold["gold_content_ids"]
        mapping = self.load(request["display_mapping"])["display_to_content"]
        self.check(bool(correct) and len(set(correct)) == len(correct) and set(correct) <= set(mapping.values()), "Gold is not a valid nonempty content-ID answer")
        if request["task_type"] == "ordering":
            self.check(set(correct) == set(mapping.values()), "ordering Gold is not a complete content permutation")
        exact = (predicted == correct) if request["task_type"] == "ordering" else (set(predicted) == set(correct))
        self.check(obs["exact_correct"] is exact, "Gold evaluation exact_correct is inconsistent with raw answer")
        gate = self.load(obs["content_gate"], "content_admission")
        self.check(gate["question_id"] == request["question_id"] and gate["provenance"] == request["provenance"], "content gate is unrelated to question/source")
        self.check(gate["decision"] == "PASS" and gate["score_used"] is False, "question lacks score-independent content admission")
        self.check(stamp(gate["completed_at"]) < stamp(request["started_at"]), "content admission must precede Solver request")
        self.check(self.same_ref(gate["public_question"], request["public_question"]) and self.same_ref(gate["source_input"], request["source_input"]), "content gate applies to another question/source version")
        self.check(self.same_ref(gate["gold_reference"], obs["gold_reference"]), "evaluated Gold differs from admitted Gold version")
        self.path(gate["review_record"])
        result = {"evaluation": obs, "run": run, "exact": exact, "ref": ref}
        self.observation_cache[key] = result
        return result

    def evaluation_group(self, ref):
        group = self.load(ref, "evaluation_group")
        self.check(bool(group["observations"]), "evaluation group is empty")
        observations = [self.observation(r) for r in group["observations"]]
        self.check(len({str(self.path(r)) for r in group["observations"]}) == len(observations), "duplicate observations inflate a denominator")
        return observations

    def groups_rate(self, observations):
        return sum(o["exact"] for o in observations) / len(observations)

    def aligned(self, left, right, same_model):
        def identity(obs):
            r = obs["run"]["request"]
            return (r["question_id"], family(r["provenance"]), r["task_type"])
        self.check(sorted(map(identity, left)) == sorted(map(identity, right)), "paired comparison uses different questions/families or denominators")
        lm = {o["run"]["request"]["model_id"] for o in left}
        rm = {o["run"]["request"]["model_id"] for o in right}
        self.check(len(lm) == len(rm) == 1, "comparison groups mix models")
        def paired_identity(obs):
            r = obs["run"]["request"]
            return (r["question_id"], family(r["provenance"]), r["task_type"], r["mode"], r["max_output_tokens"], r["source_input"]["sha256"], r["display_mapping"]["sha256"])
        if not same_model:
            self.check(sorted(map(paired_identity, left)) == sorted(map(paired_identity, right)), "before/after collection changes mode, source, mapping or output budget")
            self.check(sorted(o["run"]["request"]["public_question"]["sha256"] for o in left) == sorted(o["run"]["request"]["public_question"]["sha256"] for o in right), "before/after changes public question versions")
        if same_model:
            self.check(self.equivalent(next(iter(lm)), next(iter(rm))), "controlled probe changes model")
        return next(iter(lm)), next(iter(rm))

    def validate_trace(self, trace):
        self.check(trace["output_budget_tokens"] <= self.frozen_cap and trace["actual_output_tokens"] <= self.frozen_cap, "diagnostic trace declares/exceeds frozen 1024 cap")

    def verify_evidence(self, evidence, source, provenance):
        self.check(evidence["source_id"] == provenance["source_id"] and evidence["source_sha256"] == provenance["source_sha256"], "weakness evidence cites another source")
        turns = source["turns"]
        first, last = evidence["turn_start"], evidence["turn_end"]
        self.check(0 <= first <= last < len(turns), "weakness evidence uses an invalid zero-based turn interval")
        quoted = "\n".join(t["text"] for t in turns[first:last + 1] if t["speaker"] == evidence["speaker"])
        self.check(bool(evidence["quote"]) and evidence["quote"] in quoted, "weakness quote/speaker does not match bound source turns")

    def trace_against_run(self, trace, diagnostic):
        request, response = diagnostic["request"], diagnostic["response"]
        self.check(request["mode"] == "diagnostic" and trace["request_id"] == request["request_id"] and trace["question_id"] == request["question_id"] and trace["provenance"] == request["provenance"] and trace["task_type"] == request["task_type"], "diagnostic trace identity differs from original run")
        self.check(self.equivalent(trace["model_id"], request["model_id"]), "diagnostic trace model differs from original run")
        for field in ["public_question", "source_input", "display_mapping"]:
            self.check(self.same_ref(trace[field], request[field]), "diagnostic trace input byte binding mismatch: " + field)
        self.check(trace["actual_output_tokens"] == diagnostic["actual_tokens"] and trace["output_budget_tokens"] == request["max_output_tokens"], "diagnostic trace token counts differ from original output")
        self.check(self.same_ref(trace["raw_response"], diagnostic["record"]["response"]), "diagnostic trace raw bytes mismatch")
        self.check(stamp(response["committed_at"]) <= stamp(trace["created_at"]), "parsed trace predates original response")
        self.check(trace["final_prediction_content_ids"] == response["prediction_content_ids"], "diagnostic parsed final prediction differs from original response")
        answer = json.loads(response["text"])
        mapping = self.load(request["display_mapping"])["display_to_content"]
        raw_items = {mapping[e["label"]]: e for e in answer["evidence"]}
        self.check({i["content_id"] for i in trace["per_item"]} == set(raw_items), "diagnostic parsed items differ from original evidence")
        source = self.load(request["source_input"])
        turns = {t.get("turn_ref"): (n, t) for n, t in enumerate(source["turns"]) if t.get("turn_ref")}
        status_map = {"supported": "supported", "contradicted": "refuted", "not_supported": "uncertain", "ambiguous": "uncertain"}
        for item in trace["per_item"]:
            raw = raw_items[item["content_id"]]
            expected = "ordering_occurrence" if request["task_type"] == "ordering" else status_map[raw["status"]]
            self.check(item["judgment"] == expected, "diagnostic parsed judgment differs from original response")
            for citation in item["evidence_refs"]:
                self.check(raw["turn_ref"] in turns and bool(raw["quote"]), "parsed diagnostic invents a source citation absent from raw response")
                n, turn = turns[raw["turn_ref"]]
                self.check(citation["quote"] == raw["quote"] and raw["quote"] in turn["text"] and citation["speaker"] == turn["speaker"], "parsed diagnostic changes original quote/speaker")
                self.check(citation["turn_start"] == citation["turn_end"] == n and citation["source_id"] == request["provenance"]["source_id"] and citation["source_sha256"] == request["provenance"]["source_sha256"], "parsed diagnostic source/index binding mismatch")

    def validate_memory(self):
        doc = self.target
        self.check(self.same_ref(doc["allowed_train_roots_manifest"], reference(DAY1 / "TRAIN_ROOTS.json")), "memory uses another Train root manifest")
        nonempty = bool(doc["failure_cases"] or doc["probes"] or doc["causes"])
        if not nonempty:
            self.check(doc["model_id"] in {"S0_M_clean", "S0_M_clean_merged"}, "empty memory uses an unbound model")
            return
        self.check(self.sidecar is not None, "nonempty memory requires a controller evidence sidecar")
        c = self.sidecar
        self.check(c.get("record_type") == "diagnosis_controller_manifest" and c.get("schema_version") == 1, "wrong controller manifest format")
        self.check(self.same_ref(c["target"], reference(self.target_path)), "controller does not bind exact target snapshot bytes")
        self.check(self.same_ref(c["frozen_protocol"], reference(DAY1 / "PROTOCOL_V1.json")), "controller changes frozen experiment protocol")
        self.register_models()
        self.check(doc["model_id"] in self.models, "memory model is absent from registry")
        cases = {f["failure_id"]: f for f in doc["failure_cases"]}
        bindings = {f["failure_id"]: f for f in c["failure_bindings"]}
        self.check(len(bindings) == len(c["failure_bindings"]) and set(bindings) == set(cases), "failure bindings do not cover exact failure inventory")
        case_observations = {}
        for failure_id, case in cases.items():
            bound = bindings[failure_id]
            self.check(self.same_ref(bound["direct_group"], case["direct_answer_record"]), "failure raw direct group binding differs")
            obs = self.evaluation_group(case["postgold_evaluation"])
            direct = self.load(case["direct_answer_record"], "solver_run_group")
            self.check(len(direct["runs"]) == len(obs), "direct run/evaluation denominator mismatch")
            self.check({str(self.path(x)) for x in direct["runs"]} == {str(self.path(x["evaluation"]["solver_run"])) for x in obs}, "direct raw runs differ from scored runs")
            for observation in obs:
                request = observation["run"]["request"]
                self.check(request["mode"] == "direct" and request["question_id"] == case["question_id"] and request["provenance"] == case["provenance"], "failure uses unrelated raw observation")
                self.check(self.equivalent(request["model_id"], doc["model_id"]), "failure has a different bound current model")
                self.check(self.same_ref(observation["evaluation"]["content_gate"], case["content_gate"]["record"]), "failure gate differs from original content review")
            for evidence in case["evidence_refs"]:
                self.verify_evidence(evidence, self.load(obs[0]["run"]["request"]["source_input"]), case["provenance"])
            mappings = {o["run"]["request"]["display_mapping"]["sha256"] for o in obs}
            failed_mappings = {o["run"]["request"]["display_mapping"]["sha256"] for o in obs if not o["exact"]}
            aggregate = case["repeat_aggregation"]
            self.check(aggregate["valid_calls"] == len(obs) and aggregate["content_failures"] == sum(not o["exact"] for o in obs), "failure numerator/denominator differs from original records")
            self.check(aggregate["total_display_mappings"] == len(mappings) and aggregate["failed_display_mappings"] == len(failed_mappings), "display mapping aggregates differ from original records")
            if case["diagnostic_trace"] is not None:
                trace = self.load(case["diagnostic_trace"])
                self.check(not self.legacy.validate(trace, self.roots), "diagnostic trace fails frozen schema")
                self.validate_trace(trace)
                diagnostic = self.solver_run(bound["diagnostic_run"])
                diagnostic_evaluation = self.observation(bound["diagnostic_evaluation"])
                self.check(self.same_ref(diagnostic_evaluation["evaluation"]["solver_run"], bound["diagnostic_run"]), "diagnostic evaluation scores a different raw response")
                self.check(diagnostic["request"]["mode"] == "diagnostic" and diagnostic["request"]["question_id"] == case["question_id"] and diagnostic["request"]["provenance"] == case["provenance"], "diagnostic raw run is unrelated to case")
                self.check(self.equivalent(diagnostic["request"]["model_id"], doc["model_id"]), "diagnostic trace uses unrelated model")
                self.trace_against_run(trace, diagnostic)
            case_observations[failure_id] = obs
        probes = {p["probe_id"]: p for p in doc["probes"]}
        probe_bindings = {p["probe_id"]: p for p in c["probe_bindings"]}
        self.check(len(probe_bindings) == len(c["probe_bindings"]) and set(probe_bindings) == set(probes), "probe bindings do not cover exact probe inventory")
        probe_data = {}
        for probe_id, probe in probes.items():
            binding = probe_bindings[probe_id]
            self.check(self.same_ref(binding["record"], probe["record"]), "controlled probe binding mismatch")
            raw = self.load(probe["record"], "controlled_probe")
            self.check(raw["probe_id"] == probe_id and raw["changed_factor"] == probe["type"], "controlled probe identity/type mismatch")
            self.check(bool(raw["cause_ids"]) and set(raw["cause_ids"]) <= {x["cause_id"] for x in doc["causes"]}, "controlled probe cites a missing cause")
            self.check(bool(raw["failure_ids"]) and set(raw["failure_ids"]) <= set(cases), "probe has no related original failures")
            related = [cases[i] for i in raw["failure_ids"]]
            self.check(any(f["provenance"] == probe["provenance"] for f in related), "probe provenance is unrelated to failure support")
            self.check(self.equivalent(raw["model_id"], doc["model_id"]) and self.equivalent(probe["compared_model_id"], doc["model_id"]), "supporting probe changes current model")
            self.check(self.same_ref(raw["protocol"], probe["frozen_protocol"]), "controlled probe uses a different frozen protocol")
            protocol = self.load(raw["protocol"], "controlled_probe_protocol")
            self.check(protocol["probe_id"] == probe_id and protocol["cause_ids"] == raw["cause_ids"] and protocol["failure_ids"] == raw["failure_ids"] and protocol["changed_factor"] == raw["changed_factor"], "probe did not freeze its related cause/failure/contrast before collection")
            self.check(protocol["expected_exact_direction"] == raw["expected_exact_direction"] and protocol["frozen_at"] == raw["frozen_at"], "probe expected effect was changed after protocol freeze")
            before = self.evaluation_group(raw["baseline_evaluations"])
            after = self.evaluation_group(raw["intervention_evaluations"])
            self.aligned(before, after, same_model=True)
            self.check(all(o["run"]["request"]["question_id"] in {f["question_id"] for f in related} for o in before + after), "controlled probe does not test cited failure questions")
            self.check({o["run"]["request"]["prompt"] for o in before} != {o["run"]["request"]["prompt"] for o in after}, "controlled probe has no actual changed input")
            self.check({(o["run"]["request"]["mode"], o["run"]["request"]["max_output_tokens"]) for o in before} == {(o["run"]["request"]["mode"], o["run"]["request"]["max_output_tokens"]) for o in after}, "controlled probe changes Solver mode or output budget")
            self.check(all(stamp(raw["frozen_at"]) < stamp(o["run"]["request"]["started_at"]) for o in before + after), "probe protocol was frozen after collection")
            self.check(all(stamp(o["evaluation"]["completed_at"]) <= stamp(raw["completed_at"]) for o in before + after), "probe conclusion precedes observations")
            controls = set(raw["unchanged_controls"])
            self.check({"model", "source", "question_semantics", "scoring"} <= controls, "controlled probe lacks common controls")
            self.check({o["run"]["request"]["source_input"]["sha256"] for o in before} == {o["run"]["request"]["source_input"]["sha256"] for o in after}, "controlled probe changes source")
            review = self.load(raw["review"], "probe_review")
            self.check(review["decision"] == "PASS" and review["probe_id"] == probe_id and review["controlled"] is True and set(review["cause_ids"]) == set(raw["cause_ids"]), "controlled probe lacks related review")
            self.check(stamp(review["completed_at"]) >= stamp(raw["completed_at"]), "probe review predates result")
            self.check(raw["outcome"] == probe["outcome"] == review["outcome"], "probe outcome disagrees with reviewed original record")
            delta = self.groups_rate(after) - self.groups_rate(before)
            direction = raw["expected_exact_direction"]
            self.check(direction in {"increase", "decrease", "change"}, "probe expected direction is missing")
            if probe["outcome"] == "supports_hypothesis":
                self.check((direction == "increase" and delta > 0) or (direction == "decrease" and delta < 0) or (direction == "change" and delta != 0), "supporting probe has no recorded predicted effect")
            probe_data[probe_id] = raw
        result_bindings = {r["result_id"]: r for r in c["result_bindings"]}
        self.check(len(result_bindings) == len(c["result_bindings"]), "duplicate result bindings")
        expected_result_ids = [r["result_id"] for cause in doc["causes"] for r in cause["new_question_results"]]
        self.check(len(expected_result_ids) == len(set(expected_result_ids)) and set(result_bindings) == set(expected_result_ids), "result bindings do not cover exact new-question inventory")
        for cause in doc["causes"]:
            fid = set(cause["supporting_failure_ids"])
            for probe_id in cause["probe_ids"]:
                raw = probe_data[probe_id]
                self.check(cause["cause_id"] in raw["cause_ids"] and set(raw["failure_ids"]) <= fid, "cause cites an unrelated controlled probe")
            families = {family(cases[i]["provenance"]) for i in fid}
            questions = {cases[i]["question_id"] for i in fid}
            for evidence in cause["evidence_refs"]:
                related = [cases[i] for i in fid if cases[i]["provenance"]["source_sha256"] == evidence["source_sha256"]]
                self.check(bool(related), "cause evidence has no supporting bound failure source")
                observed = case_observations[related[0]["failure_id"]][0]
                self.verify_evidence(evidence, self.load(observed["run"]["request"]["source_input"]), related[0]["provenance"])
            results = {}
            for result in cause["new_question_results"]:
                self.check(self.same_ref(result_bindings[result["result_id"]]["record"], result["record"]), "new-question comparison binding mismatch")
                raw = self.load(result["record"], "new_question_comparison")
                self.check(raw["result_id"] == result["result_id"] and raw["cause_id"] == cause["cause_id"], "new-question result is unrelated to cause")
                self.check(self.same_ref(raw["before_protocol"], raw["after_protocol"]), "new-question before/after protocols differ")
                protocol = self.load(raw["before_protocol"], "new_question_comparison_protocol")
                self.check(protocol["metric"] == "exact_rate", "new-question comparison uses an unbound metric")
                before, after = self.evaluation_group(raw["before_evaluations"]), self.evaluation_group(raw["after_evaluations"])
                self.check(all(stamp(protocol["frozen_at"]) < stamp(o["run"]["request"]["started_at"]) for o in before + after), "new-question comparison protocol was frozen after collection")
                before_model, after_model = self.aligned(before, after, same_model=False)
                self.check(before_model == result["before_model_id"] and after_model == result["after_model_id"], "comparison result model labels mismatch")
                self.check(not self.equivalent(before_model, after_model) and self.descendant(after_model, before_model), "improvement/recurrence must follow a bound training lineage")
                self.check(self.descendant(before_model, doc["model_id"]) or self.descendant(doc["model_id"], before_model), "comparison is unrelated to current memory model lineage")
                qids = {o["run"]["request"]["question_id"] for o in before}
                new_families = {family(o["run"]["request"]["provenance"]) for o in before}
                self.check(not (qids & questions) and not (new_families & families), "new-question comparison reuses supporting question/family")
                self.check(new_families == {family(result["provenance"])}, "new-question comparison provenance mismatch")
                for model_id in self.ancestors(after_model):
                    train_ref = self.models[model_id].get("training_record")
                    if train_ref:
                        train = self.load(train_ref, "training_lineage")
                        self.check(not qids.intersection(train["training_question_ids"]) and not new_families.intersection(train["training_family_keys"]), "new-question comparison includes a trained-on question/family")
                before_rate, after_rate = self.groups_rate(before), self.groups_rate(after)
                self.check(result["before_exact_rate"] == before_rate and result["after_exact_rate"] == after_rate, "reported improvement rate differs from original predictions")
                self.check(all(stamp(o["evaluation"]["completed_at"]) <= stamp(raw["completed_at"]) for o in before + after), "comparison result predates observations")
                review = self.load(raw["review"], "comparison_review")
                self.check(review["decision"] == "PASS" and review["result_id"] == result["result_id"] and review["new_untrained_family"] is True, "new-question comparison lacks source/newness review")
                self.check(stamp(review["completed_at"]) >= stamp(raw["completed_at"]), "comparison review predates result")
                results[str(self.path(result["record"]))] = {"delta": after_rate - before_rate, "at": stamp(review["completed_at"]), "before": before_model, "after": after_model, "questions": qids}
            allowed = {("none", "hypothesis"), ("hypothesis", "reproduced"), ("hypothesis", "retired"), ("reproduced", "improving"), ("reproduced", "retired"), ("improving", "recurred"), ("improving", "retired"), ("recurred", "improving"), ("recurred", "retired")}
            last_time, last_to, last_improvement = None, None, None
            for transition in cause["state_history"]:
                at = stamp(transition["at"])
                self.check((transition["from"], transition["to"]) in allowed, "invalid cause-state transition")
                self.check(last_time is None or (at > last_time and transition["from"] == last_to), "cause history must be chronological and connected")
                self.check(self.descendant(transition["model_id"], doc["model_id"]) or self.descendant(doc["model_id"], transition["model_id"]), "cause transition belongs to unrelated model lineage")
                record_path = str(self.path(transition["record"]))
                if transition["to"] == "reproduced":
                    related = [probe_data[p] for p in cause["probe_ids"] if probes[p]["outcome"] == "supports_hypothesis" and str(self.path(probes[p]["record"])) == record_path]
                    self.check(bool(related), "reproduced state must cite its related supporting controlled probe")
                    self.check(all(stamp(self.load(p["review"], "probe_review")["completed_at"]) <= at for p in related), "reproduced state precedes controlled-probe review")
                    self.check(self.equivalent(transition["model_id"], doc["model_id"]), "reproduced state changes the current frozen model")
                elif transition["to"] == "improving":
                    self.check(record_path in results, "improving lacks a bound reviewed new-question result")
                    item = results[record_path]
                    self.check(item["delta"] > 0 and item["at"] <= at, "improving requires earlier reviewed positive effect on new questions")
                    self.check(self.equivalent(transition["model_id"], item["after"]), "improving transition model differs from improved model")
                    last_improvement = item
                elif transition["to"] == "recurred":
                    self.check(last_improvement is not None, "recurred requires a chronological prior improvement")
                    self.check(record_path in results, "recurred lacks a bound reviewed later decline")
                    item = results[record_path]
                    self.check(item["delta"] < 0 and last_improvement["at"] < item["at"] <= at, "recurred requires subsequent negative change")
                    self.check(self.descendant(item["before"], last_improvement["after"]) and item["questions"] == last_improvement["questions"], "recurrence is not a later regression of the improved held-out questions")
                    self.check(self.equivalent(transition["model_id"], item["after"]), "recurrence transition model mismatch")
                last_time, last_to = at, transition["to"]
            self.check(stamp(doc["created_at"]) >= last_time, "snapshot predates its cause-state history")

    def validate(self):
        schema_errors = self.legacy.validate(self.target, self.roots)
        self.check(not schema_errors, "Day1 schema/semantic validation failed: " + "; ".join(schema_errors))
        for ref in all_refs(self.target):
            self.path(ref)
        kind = self.target["document_type"]
        if kind == "challenger_weakness_memory":
            self.validate_memory()
        elif kind == "solver_diagnostic_trace":
            self.validate_trace(self.target)
            self.check(self.sidecar is not None, "standalone trace requires controller evidence, not declarations only")
            self.register_models()
            run = self.solver_run(self.sidecar["diagnostic_run"])
            self.check(self.same_ref(self.sidecar["target"], reference(self.target_path)), "standalone trace binding mismatch")
            self.trace_against_run(self.target, run)
        else:
            raise Rejected("controller publication supports weakness memory or diagnostic trace; other documents require their dedicated content controller")
        return {"status": "PASS", "checks": self.check_count, "verified_artifact_count": len(self.checked_refs), "document_type": kind, "target_sha256": sha(self.target_path), "frozen_diagnostic_cap": self.frozen_cap, "model_calls": 0, "training_runs": 0, "limits": ["Recorded chronology is checked; this does not prove OS-level isolation or hidden backend behavior.", "Review records remain internal semantic attestations, not an automatic proof of causal mechanisms."]}


def validate_file(target_path, sidecar_path=None):
    try:
        controller = Controller(target_path, sidecar_path)
        return controller.validate()
    except (Rejected, KeyError, IndexError, AttributeError, TypeError, ValueError, OSError) as e:
        return {"status": "FAIL", "error": str(e), "model_calls": 0, "training_runs": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--controller", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    report = validate_file(args.artifact, args.controller)
    if args.report:
        with args.report.open("x") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
