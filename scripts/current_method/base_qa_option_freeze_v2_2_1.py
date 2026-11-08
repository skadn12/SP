"""V2.2.1 typed answers and one authoritative frozen presentation.

This offline file adapter grants no source permission and invokes no model.
Formal use must enter through the pipeline's source/difficulty/target gates.
The stored map is used for delivery, blind parsing, comparison and scoring;
loading a bundle never rerenders or reassigns a display label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

try:
    from . import base_qa_pipeline_v2_1 as legacy
    from . import base_qa_answers_v2_2_1 as answers
except ImportError:
    import base_qa_pipeline_v2_1 as legacy
    import base_qa_answers_v2_2_1 as answers

VERSION = "base_qa_option_freeze_v2_2_1"
BLIND_SCHEMA = "base_qa_blind_response_v2_2_1"
BLIND_FIELDS = frozenset((legacy.BLIND_FIELDS - {"independent_gold"}) | {"independent_answer"})
SEED = "independent-review-v2.2.1"
ENVELOPE = "V2_2_1_FREEZE_ENVELOPE.json"
CONTRACT = "RENDER_CONTRACT.json"
PACKET = "PUBLIC_PACKET.json"
RECEIPT = "ORCHESTRATION_RECEIPT.json"
MANIFEST = "IMPLEMENTATION_MANIFEST.json"
_FILES = ("AUTHOR_CANDIDATE.json", "AUTHOR_CANDIDATE_FREEZE.json", PACKET,
          CONTRACT, "SOURCE.json", "SOURCE_SCOPE.json", RECEIPT, MANIFEST)


class FreezeError(ValueError):
    """A typed answer or immutable orchestration boundary is invalid."""


def _bytes(value):
    return answers.typed_key(value).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return _sha(_bytes(value))


def _loads(raw):
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"),
                           object_pairs_hook=legacy._reject_duplicate_keys,
                           parse_constant=legacy._reject_constant)
        # parse_constant alone misses overflow such as JSON's 1e9999.
        answers.typed_key(value)
        return value
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise FreezeError("STRICT_JSON_INVALID") from exc


def _clone(value):
    return _loads(_bytes(value))


def _read(path):
    try:
        return _loads(Path(path).read_bytes())
    except OSError as exc:
        raise FreezeError("FREEZE_FILE_MISSING_OR_UNREADABLE:" + Path(path).name) from exc


def _write(path, value):
    with Path(path).open("xb") as handle:
        handle.write(_bytes(value) + b"\n")


def _file_binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def normalize_candidate(candidate):
    try:
        return answers.normalize_candidate(candidate)
    except ValueError as exc:
        raise FreezeError(str(exc)) from exc


def implementation_bindings(extra_paths=()):
    """Bind new helpers, pipeline, schemas/prompts and every reused dependency."""
    here = Path(__file__).resolve()
    required = [here, Path(answers.__file__).resolve(),
                here.with_name("base_qa_pipeline_v2_2_1.py"),
                here.with_name("base_qa_schema_v2_2_1.json"),
                here.with_name("base_qa_pipeline_v2_2.py"),
                here.with_name("base_qa_schema_v2_2.json"),
                here.with_name("base_qa_option_freeze_v2_2.py")]
    new_prompts = sorted(here.with_name("base_qa_prompts_v2_2_1").glob("*.txt"))
    if not new_prompts or any(not path.is_file() for path in required):
        raise FreezeError("V2_2_1_IMPLEMENTATION_OR_PROMPTS_MISSING")
    paths = [*required, *new_prompts,
             *sorted(here.with_name("base_qa_prompts_v2_2").glob("*.txt")),
             *sorted(here.with_name("base_qa_prompts_v1").glob("*.txt")),
             *sorted(here.with_name("base_qa_prompts_v2").glob("*.txt")),
             *(Path(row["path"]) for row in legacy.implementation_bindings()),
             *(Path(path) for path in extra_paths)]
    try:
        rows = [_file_binding(path) for path in sorted(set(path.resolve() for path in paths))]
    except OSError as exc:
        raise FreezeError("IMPLEMENTATION_FILE_UNREADABLE") from exc
    return {"schema_version": VERSION + ":implementation_manifest", "files": rows}


def _verify_implementation(manifest):
    if (type(manifest) is not dict or set(manifest) != {"schema_version", "files"}
            or manifest.get("schema_version") != VERSION + ":implementation_manifest"
            or type(manifest.get("files")) is not list):
        raise FreezeError("IMPLEMENTATION_MANIFEST_SCHEMA")
    paths = []
    for row in manifest["files"]:
        if type(row) is not dict or set(row) != {"path", "sha256", "bytes"}:
            raise FreezeError("IMPLEMENTATION_BINDING_SCHEMA")
        try:
            actual = _file_binding(row["path"])
        except (OSError, TypeError, ValueError) as exc:
            raise FreezeError("IMPLEMENTATION_FILE_UNREADABLE") from exc
        if not answers.typed_equal(row, actual):
            raise FreezeError("IMPLEMENTATION_DRIFT:" + Path(row["path"]).name)
        paths.append(actual["path"])
    if len(paths) != len(set(paths)):
        raise FreezeError("IMPLEMENTATION_DUPLICATE_PATH")
    if not {row["path"] for row in implementation_bindings()["files"]} <= set(paths):
        raise FreezeError("IMPLEMENTATION_BINDING_INCOMPLETE")


def blind_instruction():
    return Path(__file__).with_name("base_qa_prompts_v2_2_1").joinpath("blind.txt").read_text(encoding="utf-8")


def _packet_from_mapping(candidate, source, scope, mapping):
    payload = {
        "stage": "INDEPENDENT_SOLVE", "instruction": blind_instruction(),
        "candidate_id": candidate["candidate_id"], "reviewer_mode": "same_model_blind",
        "source": legacy.v2.source_view(source, scope), "source_scope": _clone(scope),
        "question_type": candidate["question_type"], "question": candidate["question"],
        "options": {label: candidate["options"][mapping[label]] for label in sorted(mapping)},
    }
    return {"packet": payload, "packet_sha256": fingerprint(payload),
            "source_scope_sha256": fingerprint(scope)}


def _render_once(candidate):
    ids = sorted(candidate["options"])
    rng = random.Random(int(fingerprint([VERSION, SEED, candidate["candidate_id"], ids]), 16))
    rng.shuffle(ids)
    return {chr(65 + index): option_id for index, option_id in enumerate(ids)}


def _source_binding(source, scope):
    bound, error = legacy._verify_source_binding(source, scope.get("source_binding"))
    if not bound:
        raise FreezeError(error)
    binding = scope.get("source_binding")
    if isinstance(binding, dict) and not answers.typed_equal(_read(binding["path"]), source):
        raise FreezeError("SOURCE_BINDING_TYPED_OBJECT_MISMATCH")
    if (scope.get("source_id") != source.get("source_id", source.get("root_id"))
            or scope.get("source_sha256") != legacy._source_sha256(source, binding)):
        raise FreezeError("SOURCE_SCOPE_BINDING_MISMATCH")


def freeze_bundle(run_dir, candidate, source, source_scope, *, reviewer_id,
                  context_id, issuer_id, receipt_id, extra_paths=()):
    """Persist one immutable offline bundle; this does not approve any material."""
    candidate = normalize_candidate(candidate)
    if type(source) is not dict or type(source_scope) is not dict:
        raise FreezeError("SOURCE_AND_SCOPE_OBJECT_REQUIRED")
    if any(type(value) is not str or not value.strip()
           for value in (reviewer_id, context_id, issuer_id, receipt_id)):
        raise FreezeError("ORCHESTRATION_IDENTITIES_REQUIRED")
    if reviewer_id in (candidate["author_id"], issuer_id):
        raise FreezeError("ORCHESTRATION_IDENTITY_COLLISION")
    source, scope = _clone(source), _clone(source_scope)
    _source_binding(source, scope)
    manifest = implementation_bindings(extra_paths)
    mapping = _render_once(candidate)
    packet = _packet_from_mapping(candidate, source, scope, mapping)
    contract = {
        "schema_version": VERSION + ":render_contract", "renderer_version": VERSION,
        "seed": SEED, "candidate_id": candidate["candidate_id"],
        "candidate_sha256": fingerprint(candidate), "canonical_option_ids": sorted(candidate["options"]),
        "source_scope_sha256": packet["source_scope_sha256"], "packet_sha256": packet["packet_sha256"],
        "display_to_canonical": mapping, "canonical_to_display": {value: key for key, value in mapping.items()},
        "mapping_sha256": fingerprint(mapping), "presentation_policy": "frozen_once_no_reassignment",
    }
    receipt = legacy.build_orchestration_receipt(
        candidate["candidate_id"], packet["packet_sha256"], packet["source_scope_sha256"],
        reviewer_id, context_id, issuer_id, receipt_id, fingerprint(candidate))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FreezeError("NEW_EMPTY_FREEZE_DIRECTORY_REQUIRED")
    _write(run_dir / "AUTHOR_CANDIDATE.json", candidate)
    _write(run_dir / "AUTHOR_CANDIDATE_FREEZE.json", {
        "schema_version": VERSION + ":author_candidate_freeze", "candidate_sha256": fingerprint(candidate),
        "packet_sha256": packet["packet_sha256"], "source_scope_sha256": packet["source_scope_sha256"]})
    for name, value in ((PACKET, packet), (CONTRACT, contract), ("SOURCE.json", source),
                        ("SOURCE_SCOPE.json", scope), (RECEIPT, receipt), (MANIFEST, manifest)):
        _write(run_dir / name, value)
    envelope = {
        "schema_version": VERSION + ":freeze_envelope", "candidate_id": candidate["candidate_id"],
        "candidate_sha256": fingerprint(candidate), "packet_sha256": packet["packet_sha256"],
        "source_scope_sha256": packet["source_scope_sha256"], "render_contract_sha256": fingerprint(contract),
        "inner_receipt_sha256": fingerprint(receipt), "implementation_manifest_sha256": fingerprint(manifest),
        "files": {name: {key: value for key, value in _file_binding(run_dir / name).items() if key != "path"}
                  for name in _FILES},
    }
    _write(run_dir / ENVELOPE, envelope)
    load_bundle(run_dir, source, scope)
    return _clone(envelope)


def load_bundle(run_dir, source=None, source_scope=None):
    """Verify all files then return the stored authority, without rerendering."""
    run_dir = Path(run_dir)
    envelope = _read(run_dir / ENVELOPE)
    keys = {"schema_version", "candidate_id", "candidate_sha256", "packet_sha256", "source_scope_sha256",
            "render_contract_sha256", "inner_receipt_sha256", "implementation_manifest_sha256", "files"}
    if (type(envelope) is not dict or set(envelope) != keys
            or envelope.get("schema_version") != VERSION + ":freeze_envelope"
            or type(envelope.get("files")) is not dict or set(envelope["files"]) != set(_FILES)):
        raise FreezeError("FREEZE_ENVELOPE_SCHEMA")
    values = {}
    for name in _FILES:
        try:
            raw = (run_dir / name).read_bytes()
        except OSError as exc:
            raise FreezeError("FREEZE_FILE_MISSING:" + name) from exc
        if not answers.typed_equal(envelope["files"][name], {"sha256": _sha(raw), "bytes": len(raw)}):
            raise FreezeError("FREEZE_FILE_DRIFT:" + name)
        values[name] = _loads(raw)
    _verify_implementation(values[MANIFEST])
    candidate = values["AUTHOR_CANDIDATE.json"]
    if not answers.typed_equal(candidate, normalize_candidate(candidate)):
        raise FreezeError("FROZEN_CANDIDATE_NOT_CANONICAL")
    scope, stored_source = values["SOURCE_SCOPE.json"], values["SOURCE.json"]
    if type(scope) is not dict or type(stored_source) is not dict:
        raise FreezeError("FROZEN_SOURCE_OR_SCOPE_SCHEMA")
    _source_binding(stored_source, scope)
    if source is not None and not answers.typed_equal(source, stored_source):
        raise FreezeError("SOURCE_INPUT_DRIFT")
    if source_scope is not None and not answers.typed_equal(source_scope, scope):
        raise FreezeError("SOURCE_SCOPE_INPUT_DRIFT")
    contract, packet, receipt = values[CONTRACT], values[PACKET], values[RECEIPT]
    if type(contract) is not dict:
        raise FreezeError("RENDER_CONTRACT_OBJECT_REQUIRED")
    mapping = contract.get("display_to_canonical")
    labels = [chr(65 + index) for index in range(len(candidate["options"]))]
    if (type(mapping) is not dict or set(mapping) != set(labels)
            or any(type(value) is not str for value in mapping.values())
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) != set(candidate["options"])):
        raise FreezeError("FROZEN_MAPPING_INVALID")
    expected_packet = _packet_from_mapping(candidate, stored_source, scope, mapping)
    if not answers.typed_equal(packet, expected_packet):
        raise FreezeError("FROZEN_MAPPING_PACKET_MISMATCH")
    expected_contract = {
        "schema_version": VERSION + ":render_contract", "renderer_version": VERSION,
        "seed": SEED, "candidate_id": candidate["candidate_id"],
        "candidate_sha256": fingerprint(candidate), "canonical_option_ids": sorted(candidate["options"]),
        "source_scope_sha256": packet["source_scope_sha256"], "packet_sha256": packet["packet_sha256"],
        "display_to_canonical": mapping, "canonical_to_display": {value: key for key, value in mapping.items()},
        "mapping_sha256": fingerprint(mapping), "presentation_policy": "frozen_once_no_reassignment",
    }
    if not answers.typed_equal(contract, expected_contract):
        raise FreezeError("RENDER_CONTRACT_SCHEMA_OR_BINDING")
    expected_author = {
        "schema_version": VERSION + ":author_candidate_freeze", "candidate_sha256": fingerprint(candidate),
        "packet_sha256": packet["packet_sha256"], "source_scope_sha256": packet["source_scope_sha256"]}
    if not answers.typed_equal(values["AUTHOR_CANDIDATE_FREEZE.json"], expected_author):
        raise FreezeError("AUTHOR_CANDIDATE_FREEZE_DRIFT")
    if type(receipt) is not dict:
        raise FreezeError("ORCHESTRATION_RECEIPT_OBJECT_REQUIRED")
    for key in ("reviewer_id", "context_id", "issuer_id", "receipt_id"):
        if type(receipt.get(key)) is not str or not receipt[key].strip():
            raise FreezeError("INNER_RECEIPT_IDENTITY_INVALID")
    if receipt["reviewer_id"] in (candidate["author_id"], receipt["issuer_id"]):
        raise FreezeError("INNER_RECEIPT_IDENTITY_COLLISION")
    expected_receipt = legacy.build_orchestration_receipt(
        candidate["candidate_id"], packet["packet_sha256"], packet["source_scope_sha256"],
        receipt["reviewer_id"], receipt["context_id"], receipt["issuer_id"], receipt["receipt_id"], fingerprint(candidate))
    if not answers.typed_equal(receipt, expected_receipt):
        raise FreezeError("INNER_RECEIPT_BINDING_DRIFT")
    bindings = {"candidate_id": candidate["candidate_id"], "candidate_sha256": fingerprint(candidate),
                "packet_sha256": packet["packet_sha256"], "source_scope_sha256": packet["source_scope_sha256"],
                "render_contract_sha256": fingerprint(contract), "inner_receipt_sha256": fingerprint(receipt),
                "implementation_manifest_sha256": fingerprint(values[MANIFEST])}
    if any(not answers.typed_equal(envelope[key], value) for key, value in bindings.items()):
        raise FreezeError("OUTER_ENVELOPE_BINDING_DRIFT")
    return {"envelope": envelope, "candidate": candidate, "source": stored_source,
            "source_scope": scope, "packet": packet, "render_contract": contract,
            "orchestration_receipt": receipt}


def frozen_packet(run_dir, source=None, source_scope=None):
    return _clone(load_bundle(run_dir, source, source_scope)["packet"])


def frozen_mapping(run_dir, source=None, source_scope=None):
    return _clone(load_bundle(run_dir, source, source_scope)["render_contract"]["display_to_canonical"])


def reviewer_receipt_projection(run_dir, source=None, source_scope=None):
    return legacy.reviewer_receipt_projection(load_bundle(run_dir, source, source_scope)["orchestration_receipt"])


def parse_blind_response(raw_bytes, bundle):
    """Strict typed blind protocol; no inferred field, answer, ID, or approval."""
    raw_hash = _sha(raw_bytes) if isinstance(raw_bytes, bytes) else None
    if not isinstance(raw_bytes, bytes):
        return {"status": "INVALID", "raw_sha256": None, "response": None, "errors": ["RAW_INPUT:TYPE"]}
    try:
        response = _loads(raw_bytes)
    except FreezeError:
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None, "errors": ["JSON:STRICT_PARSE_INVALID"]}
    if type(response) is not dict:
        return {"status": "INVALID", "raw_sha256": raw_hash, "response": None, "errors": ["TOP_LEVEL:OBJECT_REQUIRED"]}
    candidate, packet = bundle["candidate"], bundle["packet"]
    source, scope, receipt = bundle["source"], bundle["source_scope"], bundle["orchestration_receipt"]
    errors = ["FIELD_MISSING:" + key for key in sorted(BLIND_FIELDS - set(response))]
    errors += ["FIELD_EXTRA:" + key for key in sorted(set(response) - BLIND_FIELDS)]
    for key in ("schema_version", "candidate_id", "packet_sha256", "source_scope_sha256",
                "reviewer_id", "model_id", "reviewer_mode", "decision"):
        if key in response and (type(response[key]) is not str or not response[key].strip()):
            errors.append("FIELD_TYPE:" + key)
    bindings = {"schema_version": BLIND_SCHEMA, "candidate_id": candidate["candidate_id"],
                "packet_sha256": packet["packet_sha256"], "source_scope_sha256": packet["source_scope_sha256"],
                "reviewer_mode": "same_model_blind", "reviewer_id": receipt["reviewer_id"]}
    for key, value in bindings.items():
        if not answers.typed_equal(response.get(key), value):
            errors.append("BINDING_MISMATCH:" + key)
    if response.get("decision") not in ("SOLVED", "HOLD"):
        errors.append("DECISION:INVALID")
    if response.get("decision") == "SOLVED":
        try:
            answers.canonical_answer(response.get("independent_answer"), candidate["question_type"],
                                     bundle["render_contract"]["display_to_canonical"])
        except ValueError as exc:
            errors.append("INDEPENDENT_ANSWER:" + str(exc))
    elif response.get("decision") == "HOLD" and response.get("independent_answer") is not None:
        errors.append("INDEPENDENT_ANSWER:HOLD_REQUIRES_NULL")
    legacy._valid_text_list(response.get("reasoning_steps"), "REASONING_STEPS", errors)
    legacy._valid_text_list(response.get("ambiguities"), "AMBIGUITIES", errors, allow_empty=True)
    legacy._valid_text_list(response.get("source_limitations"), "SOURCE_LIMITATIONS", errors, allow_empty=True)
    if response.get("decision") == "SOLVED" and response.get("ambiguities") != []:
        errors.append("AMBIGUITIES:UNRESOLVED_SOLVED")
    if response.get("decision") == "HOLD" and not response.get("ambiguities"):
        errors.append("AMBIGUITIES:HOLD_REASON_REQUIRED")
    evidence = response.get("evidence")
    if type(evidence) is not list or not evidence:
        errors.append("EVIDENCE:NONEMPTY_LIST_REQUIRED")
    else:
        by_ref = {turn.get("turn_ref"): turn for turn in source.get("turns", [])
                  if type(turn) is dict and type(turn.get("turn_ref")) is str}
        refs = []
        for index, item in enumerate(evidence):
            prefix = "EVIDENCE:" + str(index)
            if type(item) is not dict or set(item) != {"turn_ref", "speaker", "quote"}:
                errors.append(prefix + ":FIELDS_INVALID")
                continue
            if any(type(item[key]) is not str or not item[key].strip() for key in item):
                errors.append(prefix + ":FIELD_TYPE")
                continue
            turn = by_ref.get(item["turn_ref"])
            refs.append(item["turn_ref"])
            if (not turn or turn.get("speaker") != item["speaker"]
                    or type(turn.get("text")) is not str or item["quote"] not in turn["text"]):
                errors.append(prefix + ":NOT_EXACT_SOURCE_EVIDENCE")
        if len(refs) != len(set(refs)):
            errors.append("EVIDENCE:DUPLICATE_TURN_REF")
        gate = legacy.formal_evidence_gate(scope, refs)
        if not gate["eligible"]:
            errors += ["EVIDENCE:NOT_FORMALLY_ADMITTED:" + error for error in gate["errors"]]
            errors += ["EVIDENCE:TURN_NOT_FORMALLY_ALLOWED:" + ref for ref in gate["ineligible_turn_refs"]]
    if not answers.typed_equal(response.get("isolation_receipt"), legacy.reviewer_receipt_projection(receipt)):
        errors.append("ISOLATION_RECEIPT:RESPONSE_MISMATCH")
    return {"status": "VALID" if not errors else "INVALID", "raw_sha256": raw_hash,
            "response": response, "errors": sorted(set(errors))}


def freeze_blind_response(run_dir, raw_bytes, source=None, source_scope=None):
    """Keep exact bytes before parsing; never retry or silently repair output."""
    if not isinstance(raw_bytes, bytes):
        raise TypeError("raw_bytes must be exact Reviewer bytes")
    run_dir = Path(run_dir)
    # Preserve received bytes even when an already-frozen context has drifted.
    with (run_dir / "BLIND_RAW.bin").open("xb") as handle:
        handle.write(raw_bytes)
    _write(run_dir / "BLIND_RAW_FREEZE.json", {
        "version": VERSION, "schema_version": VERSION + ":raw_freeze",
        "sha256": _sha(raw_bytes), "bytes": len(raw_bytes), "encoding": "exact input bytes; not normalized"})
    try:
        bundle = load_bundle(run_dir, source, source_scope)
        parsed = parse_blind_response(raw_bytes, bundle)
    except (FreezeError, OSError, ValueError, TypeError, KeyError) as exc:
        parsed = {"status": "INVALID", "raw_sha256": _sha(raw_bytes), "response": None,
                  "errors": ["V2_2_1_FREEZE_INVALID:" + str(exc)]}
    _write(run_dir / "BLIND_PARSE.json", parsed)
    if parsed["status"] == "VALID":
        _write(run_dir / "BLIND_SOLUTION.json", parsed["response"])
        _write(run_dir / "BLIND_SOLUTION_FREEZE.json", {
            "version": VERSION, "schema_version": VERSION + ":solution_freeze",
            "sha256": fingerprint(parsed["response"]), "raw_sha256": parsed["raw_sha256"],
            "parse_file_sha256": _sha((run_dir / "BLIND_PARSE.json").read_bytes()),
            "solution_file_sha256": _sha((run_dir / "BLIND_SOLUTION.json").read_bytes())})
    return parsed


def blind_disposition(run_dir, source=None, source_scope=None):
    """Reparse frozen bytes, verify all freezes, and only then compare Gold."""
    try:
        run_dir = Path(run_dir)
        bundle = load_bundle(run_dir, source, source_scope)
        raw = (run_dir / "BLIND_RAW.bin").read_bytes()
        raw_freeze = _read(run_dir / "BLIND_RAW_FREEZE.json")
        parse = _read(run_dir / "BLIND_PARSE.json")
        solution = _read(run_dir / "BLIND_SOLUTION.json")
        solution_freeze = _read(run_dir / "BLIND_SOLUTION_FREEZE.json")
        expected_raw = {"version": VERSION, "schema_version": VERSION + ":raw_freeze",
                        "sha256": _sha(raw), "bytes": len(raw), "encoding": "exact input bytes; not normalized"}
        expected_solution = {
            "version": VERSION, "schema_version": VERSION + ":solution_freeze", "sha256": fingerprint(solution),
            "raw_sha256": _sha(raw), "parse_file_sha256": _sha((run_dir / "BLIND_PARSE.json").read_bytes()),
            "solution_file_sha256": _sha((run_dir / "BLIND_SOLUTION.json").read_bytes())}
        if not answers.typed_equal(raw_freeze, expected_raw) or not answers.typed_equal(solution_freeze, expected_solution):
            raise FreezeError("BLIND_FREEZE_BINDING_MISMATCH")
        reparsed = parse_blind_response(raw, bundle)
        if (reparsed["status"] != "VALID" or not answers.typed_equal(parse, reparsed)
                or not answers.typed_equal(solution, reparsed["response"])):
            raise FreezeError("BLIND_RAW_REPARSE_OR_SOLUTION_MISMATCH")
        if solution["decision"] != "SOLVED":
            return {"decision": "HOLD", "errors": ["REVIEWER_HOLD"], "gold_compared": False}
        qtype = bundle["candidate"]["question_type"]
        reviewer = answers.map_answer(solution["independent_answer"], qtype,
                                      bundle["render_contract"]["display_to_canonical"])
        author = bundle["candidate"]["gold_answer"]
        match = answers.typed_equal(reviewer, author)
        return {"decision": "REVIEWER_MATCH" if match else "HOLD", "gold_compared": True,
                "errors": [] if match else ["REVIEWER_GOLD_DISAGREEMENT"],
                "author_gold": author, "reviewer_gold": reviewer,
                "reviewer_display_answer": solution["independent_answer"],
                "mapping_sha256": bundle["render_contract"]["mapping_sha256"]}
    except (FreezeError, OSError, ValueError, TypeError, KeyError) as exc:
        return {"decision": "HOLD", "errors": ["V2_2_1_FREEZE_INVALID:" + str(exc)], "gold_compared": False}


def score_frozen_answer(run_dir, display_answer, source=None, source_scope=None):
    """Score typed labels against the same stored map used by the Reviewer."""
    try:
        bundle = load_bundle(run_dir, source, source_scope)
        qtype = bundle["candidate"]["question_type"]
        canonical = answers.map_answer(display_answer, qtype, bundle["render_contract"]["display_to_canonical"])
        return {"status": "VALID", "correct": answers.typed_equal(canonical, bundle["candidate"]["gold_answer"]),
                "canonical_answer": canonical, "mapping_sha256": bundle["render_contract"]["mapping_sha256"]}
    except (FreezeError, OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "HOLD", "correct": None, "errors": [str(exc)]}
