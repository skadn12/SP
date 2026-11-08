from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
PREV = WEEK / "day5_training_v1" / "train_diagnosis_prepare"
TOKENIZER = Path("/root/autodl-tmp/Qwen3-4B")
BACKEND_SOURCE = Path("/root/autodl-tmp/gxy-selfplay/worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell/vllm_backends.py")
MAX_OUTPUT = 1024
MAX_MODEL_LEN = 16384


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def schema(labels: list[str], turn_refs: list[str], question_type: str) -> dict:
    assessments = {
        label: {
            "type": "object",
            "additionalProperties": False,
            "required": ["judgment", "turn_refs"],
            "properties": {
                "judgment": {"type": "string", "enum": ["supported", "contradicted", "not_supported", "ambiguous"]},
                "turn_refs": {
                    "type": "array", "maxItems": 3,
                    "items": {"type": "string", "enum": turn_refs},
                },
            },
        }
        for label in labels
    }
    answer = {"type": "array", "items": {"type": "string", "enum": labels}}
    if question_type == "ordering":
        answer.update(minItems=len(labels), maxItems=len(labels))
    else:
        answer.update(minItems=1, maxItems=len(labels))
    return {
        "type": "object", "additionalProperties": False,
        "required": ["item_evidence", "final_answer"],
        "properties": {
            "item_evidence": {
                "type": "object", "additionalProperties": False,
                "required": labels, "properties": assessments,
            },
            "final_answer": answer,
        },
    }


def make_prompt(tokenizer, q: dict, display: dict, source: dict) -> str:
    field = "options" if q["question_type"] == "multi_select" else "items"
    labels = list(display[field])
    turns = "\n".join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source["turns"])
    task = (
        "For each displayed label, fill exactly one item_evidence property. "
        "Judge whether the statement is supported, contradicted, not_supported, or ambiguous "
        "under the exact question. Cite up to three source turn_ref values that support your "
        "judgment; use an empty list when no source turn supports it. Do not quote or explain. "
        "For multi_select, final_answer contains every selected label. For ordering, it contains "
        "every label exactly once in the requested order. Return only the required JSON object."
    )
    system = "Use only the supplied dialogue. Do not use outside knowledge, hidden reasoning, or unstated event order."
    user = (
        f"Dialogue:\n{turns}\n\nQuestion ({q['question_type']}):\n{q['question']}\n\n"
        + field.title() + ":\n"
        + "\n".join(f"{label}. {text}" for label, text in display[field].items())
        + "\n\n" + task
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def prepare():
    from transformers import AutoTokenizer

    questions = read(PREV / "public" / "PUBLIC_QUESTIONS.json")["questions"]
    brief_rows = read(PREV / "public" / "BRIEF_EVIDENCE_REQUESTS.json")["requests"]
    selection = read(PREV / "SELECTION.json")
    prior_selection = selection["selected_roots"]
    if len(prior_selection) != 6:
        raise RuntimeError("EXPECTED_SIX_FROZEN_TRAIN_ROOTS")
    q_by = {q["candidate_id"]: q for q in questions}
    choices = {}
    for row in brief_rows:
        if row["variant_id"] == "perm_00" and row["slot"] == 0:
            choices.setdefault(row["candidate_id"], row)
    selected = []
    for root in prior_selection:
        for typ in ("multi_select", "ordering"):
            candidates = [q for q in questions if q["root_id"] == root and q["question_type"] == typ]
            if not candidates:
                raise RuntimeError(f"MISSING_FIXED_TYPE:{root}:{typ}")
            q = min(candidates, key=lambda x: hashlib.sha256(("day6-diagnostic-reliability-v1|" + x["candidate_id"]).encode()).hexdigest())
            if q["candidate_id"] not in choices:
                raise RuntimeError("MISSING_ORIGINAL_PERM00_REQUEST")
            selected.append(q)
    if len(selected) != 12 or len({q["candidate_id"] for q in selected}) != 12:
        raise RuntimeError("SELECTION_COUNT_OR_UNIQUENESS_FAILURE")

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    rows, source_bindings, prompt_bindings = [], [], []
    for q in selected:
        row = choices[q["candidate_id"]]
        source_path = Path(q["source_path"])
        if sha(source_path) != q["source_sha256"]:
            raise RuntimeError("SOURCE_HASH_MISMATCH:" + q["candidate_id"])
        source = read(source_path)
        display = row["display"]
        field = "options" if q["question_type"] == "multi_select" else "items"
        labels = list(display[field])
        turn_refs = [t["turn_ref"] for t in source["turns"]]
        if len(turn_refs) != len(set(turn_refs)):
            raise RuntimeError("DUPLICATE_SOURCE_TURN_REF")
        out_schema = schema(labels, turn_refs, q["question_type"])
        prompt = make_prompt(tokenizer, q, display, source)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) + MAX_OUTPUT > MAX_MODEL_LEN:
            raise RuntimeError("CONTEXT_OVERFLOW:" + q["candidate_id"])
        request_id = row["request_id"].replace("__brief_evidence__", "__diag_json_v1__")
        prompt_path = HERE / "prompts" / (request_id + ".txt")
        schema_path = HERE / "schemas" / (request_id + ".json")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt)
        save(schema_path, out_schema)
        prompt_bindings.append(binding(prompt_path))
        source_bindings.append(binding(source_path))
        old = {
            role: binding(PREV / "models" / role / "brief_evidence" / "responses" / (row["request_id"] + ".json"))
            for role in ("A1", "B1")
        }
        rows.append({
            "request_id": request_id, "candidate_id": q["candidate_id"], "root_id": q["root_id"],
            "question_type": q["question_type"], "seed": row["seed"], "slot": row["slot"],
            "variant_id": row["variant_id"], "display": display, "labels": labels,
            "prompt_path": str(prompt_path), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": len(prompt_tokens), "schema_path": str(schema_path),
            "schema_sha256": sha(schema_path), "source_path": str(source_path),
            "source_sha256": q["source_sha256"], "source_turn_refs": turn_refs,
            "max_output_tokens": MAX_OUTPUT, "legacy_outputs": old,
        })

    endpoint_bindings = {}
    for role in ("A1", "B1"):
        ep = read(PREV / "models" / role / "EXECUTION_MANIFEST.json")
        endpoint_bindings[role] = {
            "manifest": binding(PREV / "models" / role / "EXECUTION_MANIFEST.json"),
            "endpoint_manifest": ep["endpoint_manifest"], "endpoint_proof": ep["endpoint_proof"],
            "model_bindings": ep["model_bindings"], "tokenizer_bindings": ep["tokenizer_bindings"],
            "code_bindings": ep["code_bindings"],
            "endpoint_evidence_bindings": ep["endpoint_evidence_bindings"],
            "model_directory": ep["model_directory"], "tokenizer_directory": ep["tokenizer_directory"],
            "runtime_versions": ep["runtime_versions"], "python_prefix": ep["python_prefix"],
        }
    save(HERE / "REQUESTS.json", {"status": "FIXED_SCORE_INDEPENDENT_SELECTION", "requests": rows})
    runtime_versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "vllm", "jsonschema")}
    save(HERE / "PROTOCOL.json", {
        "status": "PREPARED_PENDING_INDEPENDENT_PREFLIGHT",
        "selection": "For each of the six previously frozen Train roots, choose the lexicographically smallest SHA256(day6-diagnostic-reliability-v1|candidate_id) within each type; no Solver scores, raw outputs, or Gold are read.",
        "question_count": 12, "roots": prior_selection, "types": {"multi_select": 6, "ordering": 6},
        "requests_per_model": 12, "max_total_requests": 24,
        "decoder": {"temperature": 0.8, "top_p": 0.9, "max_tokens": MAX_OUTPUT, "max_model_len": MAX_MODEL_LEN,
            "enable_thinking": False, "structured_outputs": "StructuredOutputsParams(json=schema)", "automatic_retry": False},
        "runtime_versions": runtime_versions,
        "comparison": "Format, label completeness, turn-ref validity only; no score/correctness claim from this pilot.",
        "gold_opened_by_prepare": False, "dev_test_inputs": False,
    })
    files = [binding(p) for p in sorted(HERE.rglob("*")) if p.is_file() and p.name not in ("PREPARATION_FREEZE.json", "PREPARATION_FREEZE_V2.json")]
    upstream = [binding(PREV / name) for name in [
        "POLICY.json", "EXECUTION_ADDENDUM_V2.json", "SELECTION.json", "DIAGNOSIS_RECORDS_FREEZE.json",
        "public/PUBLIC_QUESTIONS.json", "public/BRIEF_EVIDENCE_REQUESTS.json", "public/DIRECT_REQUESTS.json",
    ]]
    code_bindings = [binding(Path(__file__)), binding(HERE / "run.py"), binding(BACKEND_SOURCE)]
    save(HERE / "PREPARATION_FREEZE_V2.json", {
        "status": "CPU_PREPARED_AWAITING_INDEPENDENT_PREFLIGHT",
        "files": files, "upstream": upstream, "sources": source_bindings,
        "prompts": prompt_bindings, "endpoints": endpoint_bindings, "code_bindings": code_bindings,
        "tokenizer_path": str(TOKENIZER), "tokenizer_bindings": [binding(TOKENIZER / n) for n in ("tokenizer.json", "tokenizer_config.json")],
        "runner_runtime_versions": runtime_versions,
        "selection_root_file": binding(WEEK / "CURRENT_STATUS.json"),
        "new_solver_calls": 0, "gold_opened": False, "old_records_modified": False,
    })
    print(json.dumps({"status": "CPU_PREPARED_AWAITING_INDEPENDENT_PREFLIGHT", "questions": len(rows),
        "roots": len(prior_selection), "types": {k: sum(r["question_type"] == k for r in rows) for k in ("multi_select", "ordering")},
        "requests_if_run": 24, "input_tokens_min": min(r["prompt_tokens"] for r in rows),
        "input_tokens_max": max(r["prompt_tokens"] for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "prepare":
        raise SystemExit("usage: python prepare.py prepare")
    if any((HERE / name).exists() for name in ("REQUESTS.json", "PREPARATION_FREEZE_V2.json", "prompts", "schemas")):
        raise SystemExit("NO_OVERWRITE: package directory is not empty")
    prepare()
