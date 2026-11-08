"""One-shot V2.2 presentation freeze over the unchanged V2.1 review protocol.

The stored public packet and display mapping are authoritative. Legacy rendering
is only re-executed to detect drift; its output never replaces a frozen label.
This is an offline file/parse adapter, not a Reviewer or Solver executor.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

try:
    from . import base_qa_pipeline_v2_1 as legacy
except ImportError:
    import base_qa_pipeline_v2_1 as legacy

VERSION = "base_qa_option_freeze_v2_2"
SEED = "independent-review-v2.1"
ENVELOPE = "V2_2_FREEZE_ENVELOPE.json"
CONTRACT = "RENDER_CONTRACT.json"
PACKET = "PUBLIC_PACKET.json"
RECEIPT = "ORCHESTRATION_RECEIPT.json"
MANIFEST = "IMPLEMENTATION_MANIFEST.json"
_FILES = (
    "AUTHOR_CANDIDATE.json", "AUTHOR_CANDIDATE_FREEZE.json", PACKET,
    CONTRACT, "SOURCE.json", "SOURCE_SCOPE.json", RECEIPT, MANIFEST,
)
_OPTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class FreezeError(ValueError):
    """A freeze boundary is missing, malformed, or no longer consistent."""


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return _sha(_bytes(value))


def _clone(value):
    # JSON serialization checks non-finite values before any file is created.
    return _loads(_bytes(value))


def _loads(raw):
    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=legacy._reject_duplicate_keys,
                          parse_constant=legacy._reject_constant)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise FreezeError("STRICT_JSON_INVALID") from exc


def _read(path):
    try:
        return _loads(Path(path).read_bytes())
    except OSError as exc:
        raise FreezeError("FREEZE_FILE_MISSING_OR_UNREADABLE:" + Path(path).name) from exc


def _write(path, value):
    # x mode is intentional: failed or partial attempts cannot be overwritten.
    raw = _bytes(value) + b"\n"
    with Path(path).open("xb") as handle:
        handle.write(raw)


def _file_binding(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def normalize_candidate(candidate):
    """Copy and canonicalize option-key storage; preserve all ordered arrays.

    Option IDs are private stable identities. They are never inferred from text
    or a display label, and duplicate/invalid IDs are rejected before freezing.
    """
    if not isinstance(candidate, dict):
        raise FreezeError("CANDIDATE_OBJECT_REQUIRED")
    options = candidate.get("options")
    if not isinstance(options, dict) or not 2 <= len(options) <= 26:
        raise FreezeError("OPTIONS_REQUIRE_2_TO_26_STABLE_IDS")
    if any(not isinstance(key, str) or not _OPTION_ID.fullmatch(key) for key in options):
        raise FreezeError("OPTION_ID_INVALID")
    if any(not isinstance(text, str) or not text.strip() for text in options.values()):
        raise FreezeError("OPTION_TEXT_INVALID")
    if candidate.get("question_type") not in ("single_choice", "multi_select", "ordering"):
        raise FreezeError("QUESTION_TYPE_INVALID")
    if any(not isinstance(candidate.get(key), str) or not candidate[key].strip()
           for key in ("candidate_id", "author_id", "question")):
        raise FreezeError("CANDIDATE_ID_AUTHOR_OR_QUESTION_INVALID")
    gold = candidate.get("gold")
    if (not isinstance(gold, list) or not gold
            or any(not isinstance(item, str) for item in gold)
            or len(gold) != len(set(gold)) or not set(gold) <= set(options)):
        raise FreezeError("GOLD_IDS_INVALID")
    if candidate["question_type"] == "single_choice" and len(gold) != 1:
        raise FreezeError("GOLD_SINGLE_CARDINALITY")
    if candidate["question_type"] == "ordering" and set(gold) != set(options):
        raise FreezeError("GOLD_ORDERING_COVERAGE")
    normalized = _clone(candidate)
    normalized["options"] = {key: normalized["options"][key] for key in sorted(options)}
    return normalized


def implementation_bindings(extra_paths=()):
    """Bind new code/prompts and every reused legacy dependency by exact bytes."""
    here = Path(__file__).resolve()
    main = here.with_name("base_qa_pipeline_v2_2.py")
    prompts = sorted(here.with_name("base_qa_prompts_v2_2").glob("*.txt"))
    if not main.is_file() or not prompts:
        raise FreezeError("V2_2_IMPLEMENTATION_OR_PROMPTS_MISSING")
    paths = [here, main, *prompts,
             *sorted(here.with_name("base_qa_prompts_v1").glob("*.txt")),
             *sorted(here.with_name("base_qa_prompts_v2").glob("*.txt")),
             *(Path(row["path"]) for row in legacy.implementation_bindings()),
             *(Path(path) for path in extra_paths)]
    try:
        bindings = [_file_binding(path) for path in sorted(set(p.resolve() for p in paths))]
    except OSError as exc:
        raise FreezeError("IMPLEMENTATION_FILE_UNREADABLE") from exc
    return {"schema_version": VERSION + ":implementation_manifest", "files": bindings}


def _verify_implementation(manifest):
    if (not isinstance(manifest, dict)
            or set(manifest) != {"schema_version", "files"}
            or manifest.get("schema_version") != VERSION + ":implementation_manifest"
            or not isinstance(manifest.get("files"), list)):
        raise FreezeError("IMPLEMENTATION_MANIFEST_SCHEMA")
    rows = manifest["files"]
    paths = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
            raise FreezeError("IMPLEMENTATION_BINDING_SCHEMA")
        try:
            actual = _file_binding(row["path"])
        except (OSError, TypeError, ValueError) as exc:
            raise FreezeError("IMPLEMENTATION_FILE_UNREADABLE") from exc
        if row != actual:
            raise FreezeError("IMPLEMENTATION_DRIFT:" + Path(row["path"]).name)
        paths.append(actual["path"])
    if len(paths) != len(set(paths)):
        raise FreezeError("IMPLEMENTATION_DUPLICATE_PATH")
    mandatory = implementation_bindings()["files"]
    if not {row["path"] for row in mandatory} <= set(paths):
        raise FreezeError("IMPLEMENTATION_BINDING_INCOMPLETE")


def _packet_from_public(candidate, source, source_scope, public):
    payload = {
        "stage": "INDEPENDENT_SOLVE", "instruction": legacy.blind_instruction(),
        "candidate_id": candidate["candidate_id"], "reviewer_mode": "same_model_blind",
        "source": legacy.v2.source_view(source, source_scope),
        "source_scope": _clone(source_scope), "question_type": candidate["question_type"],
        "question": candidate["question"], "options": _clone(public["options"]),
    }
    return {"packet": payload, "packet_sha256": fingerprint(payload),
            "source_scope_sha256": fingerprint(source_scope)}


def freeze_bundle(run_dir, candidate, source, source_scope, *, reviewer_id,
                  context_id, issuer_id, receipt_id, extra_paths=()):
    """Create one new private freeze; no source admission is granted by this API.

    This may be used on offline fixtures. Real orchestration must separately
    establish formal source permission and the pre-TARGET difficulty gate.
    """
    normalized = normalize_candidate(candidate)
    if not isinstance(source, dict) or not isinstance(source_scope, dict):
        raise FreezeError("SOURCE_AND_SCOPE_OBJECT_REQUIRED")
    if any(not isinstance(value, str) or not value.strip()
           for value in (reviewer_id, context_id, issuer_id, receipt_id)):
        raise FreezeError("ORCHESTRATION_IDENTITIES_REQUIRED")
    if reviewer_id in (normalized["author_id"], issuer_id):
        raise FreezeError("ORCHESTRATION_IDENTITY_COLLISION")
    source, source_scope = _clone(source), _clone(source_scope)
    manifest = implementation_bindings(extra_paths)
    # Generate the actual presentation once. Thereafter it is stored authority.
    public, private = legacy.v1.prepare_display(normalized, seed=SEED)
    packet = _packet_from_public(normalized, source, source_scope, public)
    mapping = private["display_to_canonical"]
    contract = {
        "schema_version": VERSION + ":render_contract", "renderer_version": legacy.v1.VERSION,
        "seed": SEED, "candidate_id": normalized["candidate_id"],
        "candidate_sha256": fingerprint(normalized), "canonical_option_ids": sorted(normalized["options"]),
        "source_scope_sha256": packet["source_scope_sha256"],
        "packet_sha256": packet["packet_sha256"], "display_to_canonical": _clone(mapping),
        "canonical_to_display": {value: key for key, value in mapping.items()},
        "mapping_sha256": fingerprint(mapping), "presentation_policy": "frozen_once_no_reassignment",
    }
    receipt = legacy.build_orchestration_receipt(
        normalized["candidate_id"], packet["packet_sha256"], packet["source_scope_sha256"],
        reviewer_id, context_id, issuer_id, receipt_id, fingerprint(normalized))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FreezeError("NEW_EMPTY_FREEZE_DIRECTORY_REQUIRED")
    legacy.freeze_author_candidate(run_dir, normalized, packet["packet_sha256"],
                                   packet["source_scope_sha256"])
    for name, value in ((PACKET, packet), (CONTRACT, contract), ("SOURCE.json", source),
                        ("SOURCE_SCOPE.json", source_scope), (RECEIPT, receipt), (MANIFEST, manifest)):
        _write(run_dir / name, value)
    envelope = {
        "schema_version": VERSION + ":freeze_envelope", "candidate_id": normalized["candidate_id"],
        "candidate_sha256": fingerprint(normalized), "packet_sha256": packet["packet_sha256"],
        "source_scope_sha256": packet["source_scope_sha256"],
        "render_contract_sha256": fingerprint(contract), "inner_receipt_sha256": fingerprint(receipt),
        "implementation_manifest_sha256": fingerprint(manifest),
        "files": {name: {k: v for k, v in _file_binding(run_dir / name).items() if k != "path"}
                  for name in _FILES},
    }
    _write(run_dir / ENVELOPE, envelope)
    # Disk read-back happens before any packet can be delivered.
    load_bundle(run_dir, source, source_scope)
    return _clone(envelope)


def load_bundle(run_dir, source=None, source_scope=None):
    """Validate exact persisted files and return copies of the frozen authority."""
    run_dir = Path(run_dir)
    envelope = _read(run_dir / ENVELOPE)
    expected_keys = {"schema_version", "candidate_id", "candidate_sha256", "packet_sha256",
                     "source_scope_sha256", "render_contract_sha256", "inner_receipt_sha256",
                     "implementation_manifest_sha256", "files"}
    if (not isinstance(envelope, dict) or set(envelope) != expected_keys
            or envelope.get("schema_version") != VERSION + ":freeze_envelope"
            or not isinstance(envelope.get("files"), dict)
            or set(envelope["files"]) != set(_FILES)):
        raise FreezeError("FREEZE_ENVELOPE_SCHEMA")
    values = {}
    for name in _FILES:
        try:
            raw = (run_dir / name).read_bytes()
        except OSError as exc:
            raise FreezeError("FREEZE_FILE_MISSING:" + name) from exc
        if envelope["files"][name] != {"sha256": _sha(raw), "bytes": len(raw)}:
            raise FreezeError("FREEZE_FILE_DRIFT:" + name)
        values[name] = _loads(raw)
    _verify_implementation(values[MANIFEST])
    candidate = values["AUTHOR_CANDIDATE.json"]
    normalized = normalize_candidate(candidate)
    if list(candidate["options"]) != sorted(candidate["options"]):
        raise FreezeError("FROZEN_OPTION_ORDER_NOT_CANONICAL")
    stored_source, stored_scope = values["SOURCE.json"], values["SOURCE_SCOPE.json"]
    if not isinstance(stored_source, dict) or not isinstance(stored_scope, dict):
        raise FreezeError("FROZEN_SOURCE_OR_SCOPE_SCHEMA")
    bound, binding_error = legacy._verify_source_binding(stored_source, stored_scope.get("source_binding"))
    if not bound:
        raise FreezeError(binding_error)
    if (stored_scope.get("source_id") != stored_source.get("source_id", stored_source.get("root_id"))
            or stored_scope.get("source_sha256") != legacy._source_sha256(
                stored_source, stored_scope.get("source_binding"))):
        raise FreezeError("FROZEN_SOURCE_SCOPE_BINDING_MISMATCH")
    if source is not None and source != stored_source:
        raise FreezeError("SOURCE_INPUT_DRIFT")
    if source_scope is not None and source_scope != stored_scope:
        raise FreezeError("SOURCE_SCOPE_INPUT_DRIFT")
    packet, contract, receipt = values[PACKET], values[CONTRACT], values[RECEIPT]
    if not isinstance(packet, dict) or set(packet) != {"packet", "packet_sha256", "source_scope_sha256"}:
        raise FreezeError("FROZEN_PACKET_SCHEMA")
    if (fingerprint(packet["packet"]) != packet["packet_sha256"]
            or fingerprint(stored_scope) != packet["source_scope_sha256"]):
        raise FreezeError("FROZEN_PACKET_HASH_MISMATCH")
    expected_contract = {
        "schema_version": VERSION + ":render_contract", "renderer_version": legacy.v1.VERSION,
        "seed": SEED, "candidate_id": candidate["candidate_id"],
        "candidate_sha256": fingerprint(candidate), "canonical_option_ids": sorted(candidate["options"]),
        "source_scope_sha256": packet["source_scope_sha256"], "packet_sha256": packet["packet_sha256"],
        "display_to_canonical": contract.get("display_to_canonical") if isinstance(contract, dict) else None,
        "canonical_to_display": contract.get("canonical_to_display") if isinstance(contract, dict) else None,
        "mapping_sha256": contract.get("mapping_sha256") if isinstance(contract, dict) else None,
        "presentation_policy": "frozen_once_no_reassignment",
    }
    if contract != expected_contract:
        raise FreezeError("RENDER_CONTRACT_SCHEMA_OR_BINDING")
    mapping = contract["display_to_canonical"]
    labels = [chr(65 + i) for i in range(len(candidate["options"]))]
    if (not isinstance(mapping, dict) or set(mapping) != set(labels)
            or any(not isinstance(value, str) for value in mapping.values())
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) != set(candidate["options"])
            or contract["mapping_sha256"] != fingerprint(mapping)
            or contract["canonical_to_display"] != {value: key for key, value in mapping.items()}):
        raise FreezeError("FROZEN_MAPPING_INVALID")
    displayed = {label: candidate["options"][mapping[label]] for label in labels}
    expected_public = {"options": displayed}
    if _packet_from_public(candidate, stored_source, stored_scope, expected_public) != packet:
        raise FreezeError("FROZEN_MAPPING_PACKET_MISMATCH")
    # Consistency check only. Neither recomputed object becomes delivery authority.
    check_public, check_private = legacy.v1.prepare_display(normalized, seed=SEED)
    if (check_private["display_to_canonical"] != mapping or check_public["options"] != displayed
            or legacy.blind_packet(normalized, stored_source, stored_scope) != packet):
        raise FreezeError("LEGACY_RENDERER_DRIFT")
    if not isinstance(values["AUTHOR_CANDIDATE_FREEZE.json"], dict):
        raise FreezeError("LEGACY_AUTHOR_FREEZE_SCHEMA")
    frozen_author, errors = legacy._load_frozen_author_candidate(
        run_dir, packet["packet_sha256"], packet["source_scope_sha256"])
    if errors or frozen_author != candidate:
        raise FreezeError("LEGACY_AUTHOR_FREEZE_DRIFT:" + ",".join(errors))
    receipt_keys = {"schema_version", "receipt_id", "candidate_id", "packet_sha256", "source_scope_sha256",
                    "author_candidate_sha256", "reviewer_id", "context_id", "context_mode",
                    "author_material_visible", "issuer_id"}
    if not isinstance(receipt, dict) or set(receipt) != receipt_keys:
        raise FreezeError("INNER_RECEIPT_EXACT_SCHEMA_REQUIRED")
    if (any(not isinstance(receipt.get(key), str) or not receipt[key].strip()
            for key in ("reviewer_id", "context_id", "issuer_id", "receipt_id"))
            or receipt["reviewer_id"] in (candidate["author_id"], receipt["issuer_id"])):
        raise FreezeError("INNER_RECEIPT_IDENTITY_INVALID")
    expected_receipt = legacy.build_orchestration_receipt(
        candidate["candidate_id"], packet["packet_sha256"], packet["source_scope_sha256"],
        receipt["reviewer_id"], receipt["context_id"], receipt["issuer_id"], receipt["receipt_id"],
        fingerprint(candidate))
    if receipt != expected_receipt:
        raise FreezeError("INNER_RECEIPT_BINDING_DRIFT")
    bindings = {"candidate_id": candidate["candidate_id"], "candidate_sha256": fingerprint(candidate),
                "packet_sha256": packet["packet_sha256"], "source_scope_sha256": packet["source_scope_sha256"],
                "render_contract_sha256": fingerprint(contract), "inner_receipt_sha256": fingerprint(receipt),
                "implementation_manifest_sha256": fingerprint(values[MANIFEST])}
    if any(envelope[key] != value for key, value in bindings.items()):
        raise FreezeError("OUTER_ENVELOPE_BINDING_DRIFT")
    return {"envelope": envelope, "candidate": candidate, "source": stored_source,
            "source_scope": stored_scope, "packet": packet, "render_contract": contract,
            "orchestration_receipt": receipt}


def frozen_packet(run_dir, source=None, source_scope=None):
    """Reviewer and later Solver use this same stored public presentation."""
    return _clone(load_bundle(run_dir, source, source_scope)["packet"])


def frozen_mapping(run_dir, source=None, source_scope=None):
    """Single frozen display-to-canonical mapping for Reviewer, Solver and scoring."""
    return _clone(load_bundle(run_dir, source, source_scope)["render_contract"]["display_to_canonical"])


def reviewer_receipt_projection(run_dir, source=None, source_scope=None):
    bundle = load_bundle(run_dir, source, source_scope)
    return legacy.reviewer_receipt_projection(bundle["orchestration_receipt"])


def freeze_blind_response(run_dir, raw_bytes, source=None, source_scope=None):
    """Freeze exact raw bytes once, then delegate strict no-repair parsing."""
    if not isinstance(raw_bytes, bytes):
        raise TypeError("raw_bytes must be exact Reviewer bytes")
    try:
        bundle = load_bundle(run_dir, source, source_scope)
    except (FreezeError, OSError, TypeError, KeyError, ValueError) as exc:
        # Even a drifted context must not discard the received Reviewer bytes.
        run_dir = Path(run_dir)
        with (run_dir / "BLIND_RAW.bin").open("xb") as handle:
            handle.write(raw_bytes)
        _write(run_dir / "BLIND_RAW_FREEZE.json", {
            "version": legacy.VERSION, "schema_version": legacy.VERSION + ":raw_freeze",
            "sha256": _sha(raw_bytes), "bytes": len(raw_bytes),
            "encoding": "exact input bytes; not normalized"})
        parsed = {"status": "INVALID", "raw_sha256": _sha(raw_bytes), "response": None,
                  "errors": ["V2_2_FREEZE_INVALID:" + str(exc)]}
        _write(run_dir / "BLIND_PARSE.json", parsed)
        return parsed
    packet = bundle["packet"]
    return legacy.freeze_blind_response(
        run_dir, raw_bytes, packet["packet_sha256"], packet["source_scope_sha256"],
        bundle["candidate"], bundle["source"], bundle["source_scope"], bundle["orchestration_receipt"])


def blind_disposition(run_dir, source=None, source_scope=None):
    """Gold comparison only after bundle and raw/solution freezes verify."""
    try:
        bundle = load_bundle(run_dir, source, source_scope)
        for name in ("BLIND_RAW_FREEZE.json", "BLIND_PARSE.json", "BLIND_SOLUTION.json",
                     "BLIND_SOLUTION_FREEZE.json"):
            # Reject duplicate keys and wrong top-level types before old readers.
            if not isinstance(_read(Path(run_dir) / name), dict):
                raise FreezeError("BLIND_STORED_OBJECT_REQUIRED:" + name)
        packet = bundle["packet"]
        result = legacy.blind_disposition(
            run_dir, packet["packet_sha256"], packet["source_scope_sha256"],
            bundle["source"], bundle["source_scope"], bundle["orchestration_receipt"])
        if result.get("gold_compared"):
            mapping = bundle["render_contract"]["display_to_canonical"]
            canonical = [mapping[label] for label in result["reviewer_display_gold"]]
            if result.get("reviewer_gold") != canonical:
                raise FreezeError("LEGACY_COMPARISON_MAPPING_DRIFT")
        return result
    except (FreezeError, OSError, TypeError, KeyError, ValueError) as exc:
        return {"decision": "HOLD", "errors": ["V2_2_FREEZE_INVALID:" + str(exc)],
                "gold_compared": False}


def score_frozen_answer(run_dir, display_answer, source=None, source_scope=None):
    """Pure offline scoring adapter; never invokes a model or infers an answer."""
    try:
        bundle = load_bundle(run_dir, source, source_scope)
        mapping = bundle["render_contract"]["display_to_canonical"]
        if (not isinstance(display_answer, list)
                or any(not isinstance(label, str) for label in display_answer)
                or len(display_answer) != len(set(display_answer))
                or not set(display_answer) <= set(mapping)):
            raise FreezeError("ANSWER_LABELS_INVALID")
        kind = bundle["candidate"]["question_type"]
        if ((kind == "single_choice" and len(display_answer) != 1)
                or (kind == "ordering" and set(display_answer) != set(mapping))):
            raise FreezeError("ANSWER_CARDINALITY_INVALID")
        canonical = [mapping[label] for label in display_answer]
        return {"status": "VALID", "correct": legacy.v1.same_answer(
            kind, bundle["candidate"]["gold"], canonical), "canonical_answer": canonical,
            "mapping_sha256": bundle["render_contract"]["mapping_sha256"]}
    except (FreezeError, OSError, TypeError, KeyError, ValueError) as exc:
        return {"status": "HOLD", "correct": None, "errors": [str(exc)]}
