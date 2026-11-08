#!/usr/bin/env python3
"""Read-only, fail-closed preflight for the anonymous v3 input layer.

The checker only validates controller artifacts. It never starts an author,
selector, Solver, training, export, or question generation process. Root,
group, arm, and feedback bindings remain in PRIVATE_BINDINGS.json; role
packets receive anonymous source projections and selector packets receive no
opportunity identity.
The v3 boundary is stricter than the legacy checker: public SOURCE files may
contain only anonymous turn/session data.  Controller-private bindings and the
admission audit retain provenance, and this checker reconciles those records
against the canonical files without exposing them to role packets.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
SRC_TOKEN = re.compile(r"^SRC_[0-9]{2,}$")
OPP_CODE = re.compile(r"^OPP_[0-9]{3}(?:_[0-9a-f]{6,})?$")

ROLE_ALLOWED = {
    "schema_version", "role", "opportunity_code", "workflow", "source",
    "neutral", "contract", "output", "opportunity_ceiling", "slots_ceiling",
    "generation_authorization",
    "context_card",
}
ROLE_REQUIRED = {
    "schema_version", "role", "opportunity_code", "workflow", "source",
    "neutral", "contract", "output", "opportunity_ceiling", "slots_ceiling",
    "generation_authorization",
}
SELECTOR_ALLOWED = {
    "schema_version", "role", "source", "neutral", "history", "slots",
    "contract", "generation_authorization", "expected_result_schema",
}
SELECTOR_REQUIRED = {
    "schema_version", "role", "source", "neutral", "history", "slots",
    "contract", "generation_authorization", "expected_result_schema",
}
IDENTITY_KEYS = {
    "root", "root_id", "root_code", "group", "group_id", "arm", "workflow",
    "feedback", "feedback_card", "author", "author_id", "opportunity_id",
    "opportunity_code", "candidate_id", "source_id", "canonical_source_binding",
}
NEUTRAL_FORBIDDEN = IDENTITY_KEYS | {
    "path", "source_path", "scene_ref", "risk_id", "raw_train_line_index_1_based",
}
DOWNSTREAM_KEYS = {
    "solver_started", "training_started", "export_started", "author_started",
    "selector_started", "question_generation_started", "generation_started",
}
PRIVATE_VALUE_TOKENS = {
    "root_id", "canonical_path", "source_id", "scene_ref", "group_id",
    "arm", "candidate_id", "solver", "gold", "history_question",
    "absolute_path", "controller_private", "private_bindings",
    "private_opportunity_mapping", "schedule_restricted",
}

# These values are allowed in controller-private artifacts, but are evidence of
# a boundary violation in a public SOURCE/NEUTRAL projection or role packet.
PROVENANCE_VALUE_RE = re.compile(
    r"(?:cc-train-episode-\d+|(?<![A-Za-z0-9])R\d{2}(?::[AB]:S\d{2}(?::T\d{3})?)?|"
    r"codex_synthetic_extension|generated_extension|source_grounded)",
    re.IGNORECASE,
)
PUBLIC_SOURCE_TOP_LEVEL = {
    "schema_version", "projection_policy", "session_ref_policy",
    "turn_count", "session_count", "turns",
}
PUBLIC_TURN_FIELDS = {"turn_ref", "speaker", "text"}
SYNTHETIC_ORIGIN = "codex_synthetic_extension"
SYNTHETIC_ROOTS = {
    "cc-train-episode-25652", "cc-train-episode-75241", "cc-train-episode-77882",
}
SYNTHETIC_SCENES = [
    f"R03:{route}:S{scene:02d}"
    for route in ("A", "B")
    for scene in range(1, 4)
]


class DuplicateKey(ValueError):
    pass


def reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")


def no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKey(f"duplicate key {key!r}")
        obj[key] = value
    return obj


def read_json(path: Path):
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys,
                       parse_constant=reject_constant)
    return value, raw


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_at(path: Path):
    value, raw = read_json(path)
    return value, raw, digest(raw)


def inside(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def add(issues, code: str, message: str, **detail):
    issues.append({"code": code, "message": message, **detail})


def _identity_value(value: str) -> bool:
    """Return whether a packet-visible string looks like private identity.

    This deliberately stays narrower than a recursive source-text scan.  It is
    used only for controller packets/schedules, where absolute paths, root
    names, arm/group labels and feedback labels must never cross the boundary.
    Anonymous SRC tokens and OPP codes are transport identifiers and are
    therefore allowed here.
    """
    stripped = value.strip()
    lower = stripped.lower()
    if lower in PRIVATE_VALUE_TOKENS:
        return True
    if stripped.startswith("/") or re.match(r"^[A-Za-z]:[\\\\/]", stripped):
        return True
    if re.search(r"\bcc-train-episode-\d+\b", lower):
        return True
    if re.fullmatch(r"r\d{2}", lower):
        return True
    if lower in {"old_f", "old_g", "new_f", "new_g", "context_card", "f", "g"}:
        return True
    return False


def forbidden_hits(value, forbidden, path="$", hits=None, *, check_values=False,
                   allowed_values=()):
    hits = [] if hits is None else hits
    forbidden_lower = {str(x).lower() for x in forbidden}
    allowed_lower = {str(x).lower() for x in allowed_values}
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            if key_s.lower() in forbidden_lower:
                hits.append({"path": f"{path}.{key_s}", "key": key_s})
            if check_values and isinstance(child, str) and child.strip().lower() not in allowed_lower and _identity_value(child):
                hits.append({"path": f"{path}.{key_s}", "value": child})
            forbidden_hits(child, forbidden, f"{path}.{key_s}", hits,
                           check_values=check_values, allowed_values=allowed_values)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if check_values and isinstance(child, str) and child.strip().lower() not in allowed_lower and _identity_value(child):
                hits.append({"path": f"{path}[{i}]", "value": child})
            forbidden_hits(child, forbidden, f"{path}[{i}]", hits,
                           check_values=check_values, allowed_values=allowed_values)
    return hits


def provenance_hits(value, path="$", hits=None):
    """Find internal provenance keys and values in a public projection."""
    hits = [] if hits is None else hits
    text_provenance_re = re.compile(
        r"(?:cc-train-episode-\d+|R\d{2}:[AB]:S\d{2}(?::T\d{3})?|"
        r"codex_synthetic_extension|generated_extension|source_grounded)",
        re.IGNORECASE,
    )
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key)
            key_l = key_s.lower()
            if key_l in {
                "origin", "root", "root_id", "root_code", "scene_ref", "source_id",
                "source_round", "session_ref", "session_ref_derivation",
                "continuation_provenance", "resume_provenance", "canonical_path",
                "canonical_source", "input_source_sha256", "baseline_sha256",
                "draft_sha256", "source_item_sha256",
            }:
                hits.append({"path": f"{path}.{key_s}", "key": key_s})
            # Free-form dialogue text may legitimately contain identifiers such
            # as ``TR5077``.  Scan metadata and structured references, but do
            # not interpret ordinary speaker/text content as controller
            # provenance.
            value_re = text_provenance_re if key_l in {"speaker", "text"} else PROVENANCE_VALUE_RE
            if isinstance(child, str) and value_re.search(child):
                hits.append({"path": f"{path}.{key_s}", "value": child})
            if not isinstance(child, str):
                provenance_hits(child, f"{path}.{key_s}", hits)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, str) and PROVENANCE_VALUE_RE.search(child):
                hits.append({"path": f"{path}[{i}]", "value": child})
            if not isinstance(child, str):
                provenance_hits(child, f"{path}[{i}]", hits)
    elif isinstance(value, str) and PROVENANCE_VALUE_RE.search(value):
        hits.append({"path": path, "value": value})
    return hits


def path_value(base: Path, value: Any) -> Path | None:
    """Resolve a packet path, accepting only relative paths inside v2."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    path = (base / value).resolve()
    return path if inside(path, base) else None


def check_hash(issues, label: str, path: Path, declared: Any, base: Path):
    if not isinstance(declared, str) or not HEX64.fullmatch(declared):
        add(issues, f"{label.upper()}_HASH_FIELD", f"{label} sha256 is missing or malformed",
            declared=declared)
        return None
    try:
        actual = digest(path.read_bytes())
    except Exception as exc:
        add(issues, f"{label.upper()}_INVALID", str(exc), path=rel(path, base))
        return None
    if actual != declared:
        add(issues, f"{label.upper()}_HASH", f"{label} byte hash mismatch",
            path=rel(path, base), declared=declared, actual=actual)
    return actual


def bool_fields(value, path="$", found=None):
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, child in value.items():
            key_s = str(key).lower()
            if key_s in DOWNSTREAM_KEYS:
                found.append((f"{path}.{key}", child))
            bool_fields(child, f"{path}.{key}", found)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            bool_fields(child, f"{path}[{i}]", found)
    return found


class V3Preflight:
    def __init__(self, base: Path):
        self.base = base
        self.issues: list[dict[str, Any]] = []
        self.role_rows: list[dict[str, Any]] = []
        self.selector_rows: list[dict[str, Any]] = []
        self.refs: dict[str, Any] = {}
        self.external_bindings: dict[Any, dict[str, Any]] = {}

    def load_ref(self, name: str):
        path = self.base / name
        try:
            value, _, _ = json_at(path)
            self.refs[name] = value
            return value
        except Exception as exc:
            add(self.issues, "REFERENCE_INVALID", f"{name}: {exc}", path=rel(path, self.base))
            self.refs[name] = None
            return None

    def external_bindings_check(self):
        """Audit the controller's canonical INPUT_BINDINGS before projections."""
        path = self.base.parent / "INPUT_BINDINGS.json"
        try:
            bindings, _, _ = json_at(path)
        except Exception as exc:
            add(self.issues, "EXTERNAL_BINDINGS_INVALID", str(exc), path=str(path))
            return
        roots = bindings.get("roots") if isinstance(bindings, dict) else None
        if not isinstance(roots, list):
            add(self.issues, "EXTERNAL_BINDINGS_SCHEMA", "INPUT_BINDINGS.roots must be a list")
            return
        if len(roots) != 8:
            add(self.issues, "EXTERNAL_ROOT_COUNT", "canonical INPUT_BINDINGS must contain eight roots",
                expected=8, actual=len(roots))
        seen = set()
        for i, row in enumerate(roots):
            if not isinstance(row, dict):
                add(self.issues, "EXTERNAL_BINDING_ROW", "canonical binding row must be an object", index=i)
                continue
            root = row.get("root_id")
            if root in seen:
                add(self.issues, "EXTERNAL_ROOT_DUPLICATE", "duplicate canonical root_id", value=root)
            seen.add(root)
            canonical = row.get("canonical_source")
            if not isinstance(canonical, dict):
                add(self.issues, "EXTERNAL_CANONICAL_SCHEMA", "canonical_source must be an object", root=root)
                continue
            source_path = Path(canonical.get("path", ""))
            if not source_path.is_absolute() or not source_path.exists():
                add(self.issues, "EXTERNAL_CANONICAL_PATH", "canonical source path missing", root=root,
                    path=canonical.get("path"))
                continue
            try:
                raw = source_path.read_bytes()
                actual_hash = digest(raw)
                if actual_hash != canonical.get("sha256"):
                    add(self.issues, "EXTERNAL_CANONICAL_HASH", "canonical source hash mismatch", root=root,
                        declared=canonical.get("sha256"), actual=actual_hash)
                if len(raw) != canonical.get("bytes"):
                    add(self.issues, "EXTERNAL_CANONICAL_BYTES", "canonical source bytes mismatch", root=root,
                        declared=canonical.get("bytes"), actual=len(raw))
                source = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_keys,
                                    parse_constant=reject_constant)
                if not isinstance(source, dict) or source.get("root_id") != root:
                    add(self.issues, "EXTERNAL_CANONICAL_ROOT", "canonical source root_id mismatch", root=root,
                        actual=source.get("root_id") if isinstance(source, dict) else None)
                actual_turns = len(source.get("turns", [])) if isinstance(source, dict) and isinstance(source.get("turns"), list) else None
                if actual_turns != row.get("turn_count"):
                    add(self.issues, "EXTERNAL_TURN_COUNT", "canonical turn_count mismatch", root=root,
                        declared=row.get("turn_count"), actual=actual_turns)
            except Exception as exc:
                add(self.issues, "EXTERNAL_CANONICAL_INVALID", str(exc), root=root)
        if len(seen) != len(roots) or None in seen:
            add(self.issues, "EXTERNAL_ROOT_ID", "canonical root_id values must be present and unique")
        self.external_bindings = {row.get("root_id"): row for row in roots
                                  if isinstance(row, dict) and row.get("root_id") is not None}

    def auth_and_downstream_gate(self):
        names = ("PROTOCOL_FREEZE.json", "AUTHORIZATION.json", "ACCESS_RULES.json",
                 "PRIVATE_BINDINGS.json", "INPUT_MANIFEST.json", "SCHEDULE_PUBLIC.json",
                 "SCHEDULE_RESTRICTED.json", "AUTHOR_LAUNCH_AUTHORIZATION.json",
                 "PACKET_MANIFEST.json", "BUILD_MANIFEST.json")
        states = {}
        for name in names:
            value = self.refs.get(name)
            if not isinstance(value, dict):
                continue
            for key in ("generation_authorization", "current_generation_authorization",
                        "author_launch_authorization"):
                if key in value:
                    states[f"{name}:{key}"] = value[key]
            for path, child in bool_fields(value):
                if child is not False:
                    add(self.issues, "DOWNSTREAM_STARTED", "downstream execution state must be false",
                        path=f"{name}{path[1:]}", value=child)
        required_states = {
            "PROTOCOL_FREEZE.json:generation_authorization",
            "AUTHORIZATION.json:generation_authorization",
            "ACCESS_RULES.json:current_generation_authorization",
            "SCHEDULE_PUBLIC.json:generation_authorization",
            "SCHEDULE_RESTRICTED.json:generation_authorization",
            "AUTHOR_LAUNCH_AUTHORIZATION.json:generation_authorization",
            "PACKET_MANIFEST.json:generation_authorization",
        }
        missing = sorted(required_states - set(states))
        if missing:
            add(self.issues, "AUTHORIZATION_STATE_MISSING", "required controller authorization fields are missing",
                fields=missing)
        if any(value != "NONE" for value in states.values()):
            add(self.issues, "GENERATION_AUTHORIZATION_NONZERO",
                "all v2 controller authorization states must be NONE", states=states)
        if states and len(set(states.values())) != 1:
            add(self.issues, "AUTHORIZATION_STATE_CONFLICT",
                "controller authorization states disagree", states=states)

        # Stage A is a metadata-only freeze.  Keep the launch and every
        # execution scope explicitly disabled; a missing or truthy flag is a
        # fail-closed protocol error rather than an implicit default.
        launch = self.refs.get("AUTHOR_LAUNCH_AUTHORIZATION.json")
        if not isinstance(launch, dict):
            add(self.issues, "AUTHOR_LAUNCH_AUTH_SCHEMA",
                "AUTHOR_LAUNCH_AUTHORIZATION must be an object")
        else:
            if launch.get("author_launch_authorized") is not False:
                add(self.issues, "AUTHOR_LAUNCH_AUTHORIZED",
                    "author launch must remain explicitly unauthorized",
                    value=launch.get("author_launch_authorized"))
            scope = launch.get("scope")
            required_scope = ("author", "selector", "solver", "training",
                              "export", "retry", "fallback")
            if not isinstance(scope, dict):
                add(self.issues, "AUTHOR_SCOPE_SCHEMA",
                    "author launch scope must be an object")
            else:
                for key in required_scope:
                    if scope.get(key) is not False:
                        add(self.issues, "AUTHOR_SCOPE_ENABLED",
                            "all execution scopes must be explicitly false",
                            scope=key, value=scope.get(key))

        authorization = self.refs.get("AUTHORIZATION.json")
        if isinstance(authorization, dict):
            scope = authorization.get("scope")
            if not isinstance(scope, dict):
                add(self.issues, "AUTHORIZATION_SCOPE_SCHEMA",
                    "AUTHORIZATION.scope must be an object")
            else:
                for key in ("author", "selector", "solver", "training",
                            "export", "retry", "fallback"):
                    if scope.get(key) is not False:
                        add(self.issues, "AUTHORIZATION_SCOPE_ENABLED",
                            "authorization scope must be explicitly false",
                            scope=key, value=scope.get(key))

        # Catch a generation authorization hidden in a newly-added controller
        # artifact instead of relying only on the current allow-list.
        def generation_values(value, path="$", found=None):
            found = [] if found is None else found
            if isinstance(value, dict):
                for key, child in value.items():
                    if str(key).lower().endswith("generation_authorization"):
                        found.append((f"{path}.{key}", child))
                    generation_values(child, f"{path}.{key}", found)
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    generation_values(child, f"{path}[{i}]", found)
            return found
        for name, value in self.refs.items():
            for path, child in generation_values(value):
                if child != "NONE":
                    add(self.issues, "GENERATION_AUTHORIZATION_NONZERO",
                        "every generation authorization must be NONE",
                        path=f"{name}{path[1:]}", value=child)

    def execution_identity_check(self):
        """Reject a copied freeze whose execution id names another run."""
        freeze = self.refs.get("PROTOCOL_FREEZE.json")
        if not isinstance(freeze, dict):
            return
        declared = freeze.get("execution_id")
        expected = self.base.name
        if declared != expected:
            add(self.issues, "EXECUTION_ID_MISMATCH",
                "protocol freeze execution_id must match the execution directory",
                declared=declared, expected=expected)

    def input_manifest_check(self):
        manifest = self.refs.get("INPUT_MANIFEST.json")
        if not isinstance(manifest, dict):
            return
        sources = manifest.get("sources")
        if not isinstance(sources, list):
            add(self.issues, "SOURCE_MANIFEST_SCHEMA", "INPUT_MANIFEST.sources must be a list")
            return
        if len(sources) != 8:
            add(self.issues, "ROOT_COUNT", "v2 requires exactly eight admitted source roots",
                expected=8, actual=len(sources))
        seen = set()
        for i, row in enumerate(sources):
            if not isinstance(row, dict):
                add(self.issues, "SOURCE_MANIFEST_ROW", "source entry must be an object", index=i)
                continue
            token = row.get("source_code")
            if not isinstance(token, str) or not SRC_TOKEN.fullmatch(token):
                add(self.issues, "SOURCE_TOKEN", "source_code must be an anonymous SRC token",
                    index=i, value=token)
            if token in seen:
                add(self.issues, "SOURCE_TOKEN_DUPLICATE", "duplicate source_code", value=token)
            seen.add(token)
            for field in ("turn_count", "source_sha256", "neutral_sha256", "source_path", "neutral_path"):
                if field not in row:
                    add(self.issues, "SOURCE_MANIFEST_FIELD", "source entry field missing",
                        index=i, field=field)
            src_path = path_value(self.base, row.get("source_path"))
            neutral_path = path_value(self.base, row.get("neutral_path"))
            if src_path is None or neutral_path is None:
                add(self.issues, "SOURCE_MANIFEST_PATH", "manifest paths must be relative and inside execution",
                    index=i, source_path=row.get("source_path"), neutral_path=row.get("neutral_path"))
                continue
            if token and src_path.parent.name != token:
                add(self.issues, "SOURCE_TOKEN_PATH", "source path is not bound to source_code",
                    source_code=token, path=rel(src_path, self.base))
            check_hash(self.issues, "source", src_path, row.get("source_sha256"), self.base)
            check_hash(self.issues, "neutral", neutral_path, row.get("neutral_sha256"), self.base)
            try:
                source, _, _ = json_at(src_path)
                self.source_projection_check(source, src_path, row.get("turn_count"))
            except Exception as exc:
                add(self.issues, "SOURCE_INVALID", str(exc), path=rel(src_path, self.base))
            try:
                neutral, _, _ = json_at(neutral_path)
                self.neutral_check(neutral, neutral_path)
                if isinstance(neutral, dict) and neutral.get("turn_count") != row.get("turn_count"):
                    add(self.issues, "NEUTRAL_TURN_COUNT", "neutral turn_count disagrees with source manifest",
                        source_code=token, declared=row.get("turn_count"), actual=neutral.get("turn_count"))
            except Exception as exc:
                add(self.issues, "NEUTRAL_INVALID", str(exc), path=rel(neutral_path, self.base))
        historical = manifest.get("historical_coverage")
        if not isinstance(historical, dict):
            add(self.issues, "HISTORICAL_MANIFEST_SCHEMA", "historical_coverage must be an object")
        else:
            hpath = path_value(self.base, historical.get("path"))
            if hpath is None:
                add(self.issues, "HISTORICAL_MANIFEST_PATH", "historical coverage path must stay in execution")
            else:
                check_hash(self.issues, "historical_coverage", hpath, historical.get("sha256"), self.base)
                try:
                    history, _, _ = json_at(hpath)
                    if not isinstance(history, dict):
                        raise ValueError("historical coverage projection must be an object")
                    self.public_artifact_scan("historical coverage", history, hpath)
                    for field in ("record_count", "unknown_count"):
                        if history.get(field) != historical.get(field):
                            add(self.issues, "HISTORICAL_MANIFEST_COUNT",
                                "historical manifest count disagrees with projection",
                                field=field, declared=historical.get(field), actual=history.get(field))
                except Exception as exc:
                    add(self.issues, "HISTORICAL_PROJECTION_INVALID", str(exc), path=rel(hpath, self.base))
        context = manifest.get("context_card")
        if not isinstance(context, dict):
            add(self.issues, "CONTEXT_MANIFEST_SCHEMA", "context_card must be an object")
        else:
            cpath = path_value(self.base, context.get("path"))
            if cpath is None:
                add(self.issues, "CONTEXT_MANIFEST_PATH", "context card path must stay in execution")
            else:
                check_hash(self.issues, "context_card", cpath, context.get("sha256"), self.base)
                try:
                    context_value, _, _ = json_at(cpath)
                    self.public_artifact_scan("context card", context_value, cpath)
                except Exception as exc:
                    add(self.issues, "CONTEXT_PROJECTION_INVALID", str(exc), path=rel(cpath, self.base))

        # Reconcile private controller bindings against both the canonical
        # INPUT_BINDINGS and anonymous manifest after both have been loaded.
        private = self.refs.get("PRIVATE_BINDINGS.json")
        private_sources = private.get("sources") if isinstance(private, dict) else None
        if not isinstance(private_sources, dict):
            add(self.issues, "PRIVATE_BINDING_SCHEMA", "PRIVATE_BINDINGS.sources must be an object")
        else:
            expected_roots = set(self.external_bindings)
            if set(private_sources) != expected_roots:
                add(self.issues, "PRIVATE_ROOT_COVERAGE",
                    "private bindings must cover exactly the canonical source roots",
                    expected_count=len(expected_roots), actual_count=len(private_sources))
            source_by_token = {row.get("source_code"): row for row in sources
                               if isinstance(row, dict) and isinstance(row.get("source_code"), str)}
            for root, binding in private_sources.items():
                if not isinstance(binding, dict):
                    continue
                canonical = self.external_bindings.get(root, {}).get("canonical_source", {})
                for field, canonical_field in (("canonical_path", "path"),
                                               ("canonical_sha256", "sha256"),
                                               ("canonical_bytes", "bytes")):
                    if binding.get(field) != canonical.get(canonical_field):
                        add(self.issues, "PRIVATE_CANONICAL_BINDING_MISMATCH",
                            "private binding disagrees with canonical INPUT_BINDINGS",
                            root=root, field=field)
                token = binding.get("source_code")
                manifest_row = source_by_token.get(token)
                if not isinstance(token, str) or not SRC_TOKEN.fullmatch(token) or manifest_row is None:
                    add(self.issues, "PRIVATE_SOURCE_BINDING_MISMATCH",
                        "private source token must resolve to one manifest source", root=root, source_code=token)
                    continue
                for private_field, manifest_field in (("projection_path", "source_path"),
                                                       ("projection_sha256", "source_sha256"),
                                                       ("neutral_path", "neutral_path"),
                                                       ("neutral_sha256", "neutral_sha256"),
                                                       ("turn_count", "turn_count")):
                    if binding.get(private_field) != manifest_row.get(manifest_field):
                        add(self.issues, "PRIVATE_PROJECTION_BINDING_MISMATCH",
                            "private binding disagrees with anonymous input manifest",
                            root=root, field=private_field)
            if set(source_by_token) != {b.get("source_code") for b in private_sources.values()
                                      if isinstance(b, dict)}:
                add(self.issues, "PRIVATE_SOURCE_COVERAGE",
                    "private bindings and input manifest must cover the same source tokens")

    def source_projection_check(self, source: Any, path: Path, expected_turn_count: Any):
        if not isinstance(source, dict):
            add(self.issues, "SOURCE_SCHEMA", "anonymous SOURCE.json must be an object",
                path=rel(path, self.base))
            return
        required = {"schema_version", "projection_policy", "turn_count", "session_count", "turns"}
        missing = sorted(required - set(source))
        if missing:
            add(self.issues, "SOURCE_REQUIRED_FIELDS", "anonymous source projection is incomplete",
                path=rel(path, self.base), fields=missing)
        if source.get("schema_version") != "anonymous_source_projection_v3":
            add(self.issues, "SOURCE_SCHEMA_VERSION", "unexpected anonymous source schema",
                path=rel(path, self.base), value=source.get("schema_version"))
        # v3 permits only a small public metadata surface.
        extra = sorted(set(source) - PUBLIC_SOURCE_TOP_LEVEL)
        if extra:
            add(self.issues, "SOURCE_METADATA_EXTRA", "public SOURCE contains non-anonymous metadata fields",
                path=rel(path, self.base), fields=extra)
        turns = source.get("turns")
        if not isinstance(turns, list):
            add(self.issues, "SOURCE_TURNS_SCHEMA", "source.turns must be a list", path=rel(path, self.base))
            return
        if source.get("turn_count") != len(turns) or source.get("turn_count") != expected_turn_count:
            add(self.issues, "SOURCE_TURN_COUNT", "source turn_count does not match bytes/list or manifest",
                path=rel(path, self.base), declared=source.get("turn_count"), list_count=len(turns), manifest=expected_turn_count)
        if not isinstance(source.get("session_count"), int) or source.get("session_count") < 1:
            add(self.issues, "SOURCE_SESSION_COUNT", "source.session_count must be a positive integer",
                path=rel(path, self.base))
        refs = set()
        sessions: dict[str, list[int]] = {}
        field_violations = []
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                add(self.issues, "SOURCE_TURN_ROW", "source turn must be an object", path=rel(path, self.base), index=i)
                continue
            fields = set(turn)
            if fields != PUBLIC_TURN_FIELDS:
                field_violations.append({"index": i,
                                        "extra": sorted(fields - PUBLIC_TURN_FIELDS),
                                        "missing": sorted(PUBLIC_TURN_FIELDS - fields)})
            for field in PUBLIC_TURN_FIELDS:
                if not isinstance(turn.get(field), str):
                    add(self.issues, "SOURCE_TURN_FIELD", "source turn field must be a string",
                        path=rel(path, self.base), index=i, field=field)
            ref = turn.get("turn_ref")
            if isinstance(ref, str):
                match = re.fullmatch(r"S([0-9]{2}):T([0-9]{3})", ref)
                if not match:
                    add(self.issues, "SOURCE_TURN_REF", "turn_ref is not an anonymous resolvable reference",
                        path=rel(path, self.base), index=i, value=ref)
                else:
                    sessions.setdefault(match.group(1), []).append(int(match.group(2)))
                if ref in refs:
                    add(self.issues, "SOURCE_TURN_REF_DUPLICATE", "turn_ref is not unique",
                        path=rel(path, self.base), value=ref)
                refs.add(ref)
        if field_violations:
            add(self.issues, "SOURCE_TURN_FIELDS", "public source turns must contain only anonymous fields",
                path=rel(path, self.base), count=len(field_violations), samples=field_violations[:20])
        leaks = provenance_hits(source)
        if leaks:
            add(self.issues, "SOURCE_PROVENANCE_LEAK",
                "public SOURCE contains internal provenance or origin markers",
                path=rel(path, self.base), hits=leaks[:50])
        if isinstance(source.get("session_count"), int) and len(sessions) != source.get("session_count"):
            add(self.issues, "SOURCE_SESSION_COUNT_MISMATCH",
                "source.session_count does not match turn_ref session cardinality",
                path=rel(path, self.base), declared=source.get("session_count"), actual=len(sessions))
        for session, indices in sessions.items():
            expected = list(range(1, len(indices) + 1))
            if indices != expected:
                add(self.issues, "SOURCE_SESSION_TURN_SEQUENCE",
                    "turn_ref indices must be contiguous and ordered per anonymous session",
                    path=rel(path, self.base), session=f"S{session}", actual=indices[:20])

    def neutral_check(self, neutral: Any, path: Path):
        if not isinstance(neutral, dict):
            add(self.issues, "NEUTRAL_SCHEMA", "NEUTRAL.json must be an object", path=rel(path, self.base))
            return
        required = {"schema_version", "turn_count", "session_count",
                    "complete_dialogue_preserved", "evidence_policy"}
        missing = sorted(required - set(neutral))
        if missing:
            add(self.issues, "NEUTRAL_REQUIRED_FIELDS", "anonymous neutral schema is incomplete",
                path=rel(path, self.base), fields=missing)
        if neutral.get("schema_version") != "anonymous_neutral_projection_v3":
            add(self.issues, "NEUTRAL_SCHEMA_VERSION", "unexpected anonymous neutral schema",
                path=rel(path, self.base), value=neutral.get("schema_version"))
        if (not isinstance(neutral.get("turn_count"), int) or neutral.get("turn_count") < 1 or
                not isinstance(neutral.get("session_count"), int) or neutral.get("session_count") < 1):
            add(self.issues, "NEUTRAL_COUNTS_SCHEMA", "neutral turn/session counts must be integers",
                path=rel(path, self.base))
        if neutral.get("complete_dialogue_preserved") is not True:
            add(self.issues, "NEUTRAL_PRESERVATION_FLAG", "neutral complete_dialogue_preserved must be true",
                path=rel(path, self.base), value=neutral.get("complete_dialogue_preserved"))
        hits = forbidden_hits(neutral, NEUTRAL_FORBIDDEN)
        hits.extend(provenance_hits(neutral))
        if hits:
            add(self.issues, "ANONYMOUS_PROJECTION_IDENTITY_LEAK",
                "anonymous neutral projection contains identity/provenance fields",
                path=rel(path, self.base), hits=hits[:20])

    def public_artifact_scan(self, label: str, value: Any, path: Path):
        """Apply the same identity/provenance boundary to role-visible JSON."""
        hits = forbidden_hits(value, NEUTRAL_FORBIDDEN, check_values=True,
                              allowed_values=("NONE", "UNKNOWN", "UNVERIFIED_PROVISIONAL"))
        hits.extend(provenance_hits(value))
        if hits:
            add(self.issues, "PUBLIC_ARTIFACT_PROVENANCE_LEAK",
                f"anonymous {label} contains internal identity/provenance markers",
                path=rel(path, self.base), hits=hits[:50])

    def synthetic_audit_check(self):
        """Reconcile private synthetic provenance with canonical and public data."""
        private = self.refs.get("PRIVATE_BINDINGS.json")
        if not isinstance(private, dict) or not isinstance(private.get("sources"), dict):
            return
        admission = self.refs.get("SOURCE_ADMISSION_AUDIT.json")
        warning_rows = admission.get("warnings", []) if isinstance(admission, dict) else []
        if not isinstance(warning_rows, list):
            add(self.issues, "ADMISSION_AUDIT_SCHEMA", "SOURCE_ADMISSION_AUDIT.warnings must be a list")
            warning_rows = []
        by_source = {
            row.get("source_code"): row for row in warning_rows
            if isinstance(row, dict) and isinstance(row.get("source_code"), str)
        }
        manifest = self.refs.get("INPUT_MANIFEST.json")
        manifest_sources = {
            row.get("source_code"): row for row in manifest.get("sources", [])
            if isinstance(row, dict) and isinstance(row.get("source_code"), str)
        } if isinstance(manifest, dict) and isinstance(manifest.get("sources"), list) else {}

        for root, binding in private["sources"].items():
            if not isinstance(binding, dict):
                continue
            canonical_path = Path(binding.get("canonical_path", ""))
            try:
                canonical, _ = read_json(canonical_path)
            except Exception as exc:
                add(self.issues, "SYNTHETIC_CANONICAL_INVALID", str(exc), root=root)
                continue
            canonical_turns = canonical.get("turns") if isinstance(canonical, dict) else None
            if not isinstance(canonical_turns, list):
                add(self.issues, "SYNTHETIC_CANONICAL_TURNS", "canonical turns must be a list", root=root)
                continue
            # Only extension rows without a usable session_ref are unbound
            # synthetic segments.  Some historical canonical sources label
            # ordinary, session-bound rows with the same origin value; those
            # rows must not be reclassified as unbound provenance.
            extension = [t for t in canonical_turns
                         if (isinstance(t, dict) and
                             t.get("origin") == SYNTHETIC_ORIGIN and
                             (not isinstance(t.get("session_ref"), str) or
                              not t.get("session_ref", "").strip()))]
            extension_count = len(extension)
            derivation = binding.get("session_ref_derivation")
            if not isinstance(derivation, dict):
                add(self.issues, "SYNTHETIC_DERIVATION_SCHEMA",
                    "private binding must declare session_ref_derivation", root=root)
                derivation = {}
            declared_count = derivation.get("synthetic_turn_count",
                                            derivation.get("derived_count"))
            if declared_count != extension_count:
                add(self.issues, "SYNTHETIC_DERIVED_COUNT_MISMATCH",
                    "private derived_count disagrees with canonical extension count",
                    root=root, declared=declared_count, actual=extension_count)
            grid_validated = derivation.get("synthetic_grid_validated")
            if extension_count and grid_validated is not True:
                add(self.issues, "SYNTHETIC_GRID_AUDIT",
                    "private synthetic grid must be explicitly validated", root=root,
                    value=grid_validated)
            if not extension_count and grid_validated not in (False, None):
                add(self.issues, "SYNTHETIC_GRID_AUDIT",
                    "non-synthetic source must not claim synthetic grid validation", root=root,
                    value=grid_validated)
            synthetic_rows = derivation.get("synthetic_turns")
            if extension_count and (not isinstance(synthetic_rows, list) or len(synthetic_rows) != extension_count):
                add(self.issues, "SYNTHETIC_TURN_AUDIT_COUNT",
                    "private synthetic_turns audit must cover every derived turn", root=root,
                    declared=(len(synthetic_rows) if isinstance(synthetic_rows, list) else None),
                    actual=extension_count)
            if extension_count and isinstance(synthetic_rows, list) and len(synthetic_rows) == extension_count:
                for index, (audit_row, turn) in enumerate(zip(synthetic_rows, extension)):
                    expected_ref = turn.get("turn_ref")
                    if not isinstance(audit_row, dict) or audit_row.get("turn_ref") != expected_ref:
                        add(self.issues, "SYNTHETIC_TURN_AUDIT_REF",
                            "private synthetic_turns audit order/ref disagrees with canonical",
                            root=root, index=index, declared=(audit_row.get("turn_ref") if isinstance(audit_row, dict) else None),
                            actual=expected_ref)
                    if isinstance(audit_row, dict) and audit_row.get("session_ref_present") is not False:
                        add(self.issues, "SYNTHETIC_TURN_AUDIT_SESSION",
                            "private synthetic turn audit must record missing session_ref",
                            root=root, index=index, value=audit_row.get("session_ref_present"))
            if derivation.get("provenance_claim") is not False:
                add(self.issues, "SYNTHETIC_PROVENANCE_CLAIM",
                    "synthetic session derivation must explicitly make no provenance claim",
                    root=root, value=derivation.get("provenance_claim"))
            if extension_count and root not in SYNTHETIC_ROOTS:
                add(self.issues, "SYNTHETIC_ROOT_NOT_ADMITTED",
                    "synthetic extension is present on a root outside the admitted allow-list", root=root)
            if root in SYNTHETIC_ROOTS and extension_count != 48:
                add(self.issues, "SYNTHETIC_EXTENSION_COUNT",
                    "each admitted synthetic root must contain exactly 48 extension turns",
                    root=root, actual=extension_count)

            # Every approved extension is six complete 8-turn segments in a
            # fixed order.  This catches partial, reordered, or guessed joins.
            if extension_count:
                scene_rows: dict[str, list[dict[str, Any]]] = {}
                for turn in extension:
                    ref = turn.get("turn_ref")
                    match = re.fullmatch(r"(R03:[AB]:S[0-9]{2}):T([0-9]{3})", ref or "")
                    if not match:
                        add(self.issues, "SYNTHETIC_TURN_REF", "synthetic turn_ref is not a structured segment reference", root=root, value=ref)
                        continue
                    scene_rows.setdefault(match.group(1), []).append(turn)
                    if isinstance(turn.get("session_ref"), str) and turn.get("session_ref", "").strip():
                        add(self.issues, "SYNTHETIC_SESSION_REF_PRESENT",
                            "synthetic canonical turn unexpectedly has session_ref", root=root, value=ref)
                if list(scene_rows) != SYNTHETIC_SCENES:
                    add(self.issues, "SYNTHETIC_SCENE_SEQUENCE",
                        "synthetic scenes must be R03 A/B S01-S03 in order",
                        root=root, actual=list(scene_rows))
                for scene in SYNTHETIC_SCENES:
                    rows = scene_rows.get(scene, [])
                    refs = [row.get("turn_ref") for row in rows]
                    expected = [f"{scene}:T{i:03d}" for i in range(1, 9)]
                    if refs != expected:
                        add(self.issues, "SYNTHETIC_SCENE_TURNS",
                            "each synthetic scene must contain T001-T008 exactly once",
                            root=root, scene=scene, actual=refs)
                    for row in rows:
                        if not isinstance(row.get("speaker"), str) or not isinstance(row.get("text"), str):
                            add(self.issues, "SYNTHETIC_TURN_CONTENT", "synthetic turn speaker/text must be strings", root=root, scene=scene)

            source_code = binding.get("source_code")
            projection_path = path_value(self.base, binding.get("projection_path"))
            projection = None
            if projection_path is not None:
                try:
                    projection, _ = read_json(projection_path)
                except Exception as exc:
                    add(self.issues, "SYNTHETIC_PROJECTION_INVALID", str(exc), root=root)
            if isinstance(projection, dict):
                public_turns = projection.get("turns")
                if not isinstance(public_turns, list):
                    add(self.issues, "SYNTHETIC_PROJECTION_TURNS", "public projection turns must be a list", root=root)
                elif len(public_turns) != len(canonical_turns):
                    add(self.issues, "SYNTHETIC_PROJECTION_COUNT",
                        "public projection must preserve canonical turn count", root=root,
                        canonical=len(canonical_turns), projected=len(public_turns))
                else:
                    canonical_content = [(t.get("speaker"), t.get("text")) for t in canonical_turns if isinstance(t, dict)]
                    public_content = [(t.get("speaker"), t.get("text")) for t in public_turns if isinstance(t, dict)]
                    if canonical_content != public_content:
                        add(self.issues, "SYNTHETIC_PROJECTION_CONTENT",
                            "public projection speaker/text sequence differs from canonical", root=root)
                    if extension_count:
                        tail = public_turns[-extension_count:]
                        groups: dict[str, list[int]] = {}
                        for turn in tail:
                            ref = turn.get("turn_ref") if isinstance(turn, dict) else None
                            match = re.fullmatch(r"(S[0-9]{2}):T([0-9]{3})", ref or "")
                            if match:
                                groups.setdefault(match.group(1), []).append(int(match.group(2)))
                        if len(groups) != 6 or any(nums != list(range(1, 9)) for nums in groups.values()):
                            add(self.issues, "SYNTHETIC_PROJECTION_SEGMENTS",
                                "public projection synthetic tail must be six anonymous 8-turn sessions",
                                root=root, groups=groups)

            audit_row = by_source.get(source_code)
            if extension_count and not isinstance(audit_row, dict):
                add(self.issues, "ADMISSION_AUDIT_MISSING",
                    "synthetic source is missing its private admission audit row", root=root, source_code=source_code)
            if isinstance(audit_row, dict):
                issues = audit_row.get("issues", [])
                if not isinstance(issues, list):
                    add(self.issues, "ADMISSION_AUDIT_ISSUES_SCHEMA", "audit row issues must be a list", root=root)
                    issues = []
                synthetic_issues = [x for x in issues if isinstance(x, dict) and x.get("code") == "SESSION_REF_DERIVED_SYNTHETIC_SEGMENT"]
                if bool(extension_count) != bool(synthetic_issues):
                    add(self.issues, "ADMISSION_AUDIT_SYNTHETIC_FLAG",
                        "audit synthetic warning presence disagrees with canonical extension", root=root)
                for item in synthetic_issues:
                    if item.get("derived_count") != extension_count or item.get("provenance_claim") is not False:
                        add(self.issues, "ADMISSION_AUDIT_SYNTHETIC_COUNT",
                            "audit synthetic warning disagrees with canonical extension", root=root, item=item)
            manifest_row = manifest_sources.get(source_code)
            if isinstance(manifest_row, dict) and manifest_row.get("turn_count") != len(canonical_turns):
                add(self.issues, "SYNTHETIC_MANIFEST_COUNT",
                    "input manifest turn_count disagrees with canonical", root=root,
                    declared=manifest_row.get("turn_count"), actual=len(canonical_turns))

    def private_binding_check(self):
        private = self.refs.get("PRIVATE_BINDINGS.json")
        if not isinstance(private, dict):
            return
        sources = private.get("sources")
        if not isinstance(sources, dict):
            add(self.issues, "PRIVATE_BINDING_SCHEMA", "PRIVATE_BINDINGS.sources must be an object")
            return
        for root, binding in sources.items():
            if not isinstance(binding, dict):
                add(self.issues, "PRIVATE_BINDING_ROW", "private source binding must be an object", root=root)
                continue
            for field in ("source_code", "canonical_sha256", "canonical_bytes", "projection_path", "projection_sha256",
                          "neutral_path", "neutral_sha256", "turn_count"):
                if field not in binding:
                    add(self.issues, "PRIVATE_BINDING_FIELD", "private source binding field missing",
                        root=root, field=field)
            for field in ("projection_path", "neutral_path"):
                path = path_value(self.base, binding.get(field))
                if path is None:
                    add(self.issues, "PRIVATE_BINDING_PATH", "private binding path must remain in execution",
                        root=root, field=field, value=binding.get(field))
                else:
                    check_hash(self.issues, field, path, binding.get(field.replace("_path", "_sha256")), self.base)
            canonical = Path(binding.get("canonical_path", ""))
            if not canonical.is_absolute() or not canonical.exists():
                add(self.issues, "PRIVATE_CANONICAL_PATH", "canonical_path must point to an existing controller source",
                    root=root, value=binding.get("canonical_path"))
            else:
                try:
                    raw = canonical.read_bytes()
                    actual = digest(raw)
                    if actual != binding.get("canonical_sha256"):
                        add(self.issues, "PRIVATE_CANONICAL_HASH", "canonical source hash mismatch",
                            root=root, declared=binding.get("canonical_sha256"), actual=actual)
                    if len(raw) != binding.get("canonical_bytes"):
                        add(self.issues, "PRIVATE_CANONICAL_BYTES", "canonical source byte count mismatch",
                            root=root, declared=binding.get("canonical_bytes"), actual=len(raw))
                except Exception as exc:
                    add(self.issues, "PRIVATE_CANONICAL_INVALID", str(exc), root=root)
            if not isinstance(binding.get("canonical_sha256"), str) or not HEX64.fullmatch(binding.get("canonical_sha256", "")):
                add(self.issues, "PRIVATE_CANONICAL_HASH", "canonical_sha256 must be a SHA256", root=root)
            if not isinstance(binding.get("canonical_bytes"), int) or binding.get("canonical_bytes") < 0:
                add(self.issues, "PRIVATE_CANONICAL_BYTES", "canonical_bytes must be a nonnegative integer", root=root)

    def contract_check(self, issues, entry: Any, label: str):
        if not isinstance(entry, dict):
            add(issues, f"{label.upper()}_CONTRACT_SCHEMA", "contract must be an object")
            return
        path = path_value(self.base, entry.get("path"))
        if path is None:
            add(issues, f"{label.upper()}_CONTRACT_PATH", "contract path must be relative and inside execution",
                value=entry.get("path"))
            return
        check_hash(issues, f"{label}_contract", path, entry.get("sha256"), self.base)

    def role_check(self, packet_path: Path):
        row = {"path": rel(packet_path, self.base), "issues": []}
        try:
            obj, _, _ = json_at(packet_path)
        except Exception as exc:
            add(row["issues"], "ROLE_PACKET_INVALID", str(exc))
            row["pass"] = False
            return row
        if not isinstance(obj, dict):
            add(row["issues"], "ROLE_PACKET_TYPE", "role packet must be an object")
            row["pass"] = False
            return row
        missing = sorted(ROLE_REQUIRED - set(obj))
        extra = sorted(set(obj) - ROLE_ALLOWED)
        if missing:
            add(row["issues"], "ROLE_REQUIRED_FIELDS", "role packet fields missing", fields=missing)
        if extra:
            add(row["issues"], "ROLE_EXTRA_FIELDS", "unexpected role packet fields", fields=extra)
        if obj.get("role") != "author":
            add(row["issues"], "ROLE_VALUE", "role packet role must be author", value=obj.get("role"))
        code = obj.get("opportunity_code")
        row["opportunity_code"] = code
        if not isinstance(code, str) or not OPP_CODE.fullmatch(code):
            add(row["issues"], "OPPORTUNITY_CODE", "opportunity_code is malformed", value=code)
        if obj.get("generation_authorization") != "NONE":
            add(row["issues"], "ROLE_AUTHORIZATION", "role packet authorization must be NONE")
        for key in ("source", "neutral"):
            section = obj.get(key)
            if not isinstance(section, dict):
                add(row["issues"], "ROLE_SOURCE_SCHEMA", f"{key} must be an object")
                continue
            path = path_value(self.base, section.get("path"))
            if path is None:
                add(row["issues"], "ROLE_PATH", f"{key}.path must be relative anonymous path",
                    value=section.get("path"))
                continue
            if key == "source" and not SRC_TOKEN.fullmatch(path.parent.name):
                add(row["issues"], "ROLE_SOURCE_TOKEN", "source path must use anonymous SRC token",
                    path=rel(path, self.base))
            check_hash(row["issues"], key, path, section.get("sha256"), self.base)
            if key == "source" and not isinstance(section.get("turn_count"), int):
                add(row["issues"], "ROLE_TURN_COUNT", "source.turn_count must be an integer")
            if key == "neutral":
                try:
                    neutral, _, _ = json_at(path)
                    self.neutral_check(neutral, path)
                except Exception as exc:
                    add(row["issues"], "ROLE_NEUTRAL_INVALID", str(exc))
        self.contract_check(row["issues"], obj.get("contract"), "role")
        if not isinstance(obj.get("output"), dict):
            add(row["issues"], "ROLE_OUTPUT_SCHEMA", "output must be an object")
        if obj.get("opportunity_ceiling") != 1:
            add(row["issues"], "ROLE_CEILING", "opportunity_ceiling must be 1",
                value=obj.get("opportunity_ceiling"))
        if obj.get("workflow") not in ("OLD", "NEW"):
            add(row["issues"], "ROLE_WORKFLOW", "workflow must be OLD or NEW",
                value=obj.get("workflow"))
        expected_slots = 3 if obj.get("workflow") == "NEW" else 0
        if obj.get("slots_ceiling") != expected_slots:
            add(row["issues"], "ROLE_SLOTS_CEILING", "slots_ceiling does not match workflow",
                expected=expected_slots, value=obj.get("slots_ceiling"))
        context = obj.get("context_card")
        if context is not None:
            if not isinstance(context, dict):
                add(row["issues"], "ROLE_CONTEXT_SCHEMA", "context_card must be an object")
            else:
                cpath = path_value(self.base, context.get("path"))
                if cpath is None:
                    add(row["issues"], "ROLE_CONTEXT_PATH", "context_card path must be relative and in execution")
                else:
                    check_hash(row["issues"], "context_card", cpath, context.get("sha256"), self.base)
        hits = forbidden_hits(obj, IDENTITY_KEYS - {"opportunity_code", "workflow", "context_card"},
                              check_values=True, allowed_values=("NONE", "OLD", "NEW", "author"))
        if hits:
            add(row["issues"], "ROLE_IDENTITY_LEAK", "role packet contains private identity fields", hits=hits[:20])
        row["pass"] = not row["issues"]
        return row

    def selector_check(self, packet_path: Path):
        row = {"path": rel(packet_path, self.base), "issues": []}
        try:
            obj, _, _ = json_at(packet_path)
        except Exception as exc:
            add(row["issues"], "SELECTOR_PACKET_INVALID", str(exc))
            row["pass"] = False
            return row
        if not isinstance(obj, dict):
            add(row["issues"], "SELECTOR_PACKET_TYPE", "selector packet must be an object")
            row["pass"] = False
            return row
        missing = sorted(SELECTOR_REQUIRED - set(obj))
        extra = sorted(set(obj) - SELECTOR_ALLOWED)
        if missing:
            add(row["issues"], "SELECTOR_REQUIRED_FIELDS", "selector packet fields missing", fields=missing)
        if extra:
            add(row["issues"], "SELECTOR_EXTRA_FIELDS", "unexpected selector packet fields", fields=extra)
        if obj.get("role") != "selector":
            add(row["issues"], "SELECTOR_ROLE_VALUE", "selector packet role must be selector")
        if obj.get("generation_authorization") != "NONE":
            add(row["issues"], "SELECTOR_AUTHORIZATION", "selector packet authorization must be NONE")
        hits = forbidden_hits(obj, IDENTITY_KEYS,
                              check_values=True, allowed_values=("NONE", "selector"))
        if hits:
            add(row["issues"], "SELECTOR_IDENTITY_LEAK", "selector packet contains identity fields", hits=hits[:20])
        for key in ("source", "neutral"):
            section = obj.get(key)
            if not isinstance(section, dict):
                add(row["issues"], "SELECTOR_SOURCE_SCHEMA", f"{key} must be an object")
                continue
            path = path_value(self.base, section.get("path"))
            if path is None or not SRC_TOKEN.fullmatch(path.parent.name):
                add(row["issues"], "SELECTOR_ANONYMOUS_PATH", f"{key}.path must point to anonymous SRC token",
                    value=section.get("path"))
            else:
                check_hash(row["issues"], key, path, section.get("sha256"), self.base)
        history = obj.get("history")
        if not isinstance(history, dict) or path_value(self.base, history.get("path")) is None:
            add(row["issues"], "SELECTOR_HISTORY_PATH", "history.path must be a relative in-execution path")
        elif not str(history.get("path")).endswith("inputs/anonymous_history/HISTORICAL_COVERAGE_ANON.json"):
            add(row["issues"], "SELECTOR_HISTORY_BINDING", "history.path must bind anonymous historical projection")
        else:
            check_hash(row["issues"], "history", path_value(self.base, history.get("path")), history.get("sha256"), self.base)
        slots = obj.get("slots")
        if not isinstance(slots, dict) or not isinstance(slots.get("path"), str) or Path(slots.get("path")).is_absolute():
            add(row["issues"], "SELECTOR_SLOTS_PATH", "slots.path must be a relative in-execution path")
        elif slots.get("path") != "NEW_SLOTS.json":
            add(row["issues"], "SELECTOR_SLOTS_PLACEHOLDER", "selector slots path must remain NEW_SLOTS.json placeholder",
                value=slots.get("path"))
        if not isinstance(slots, dict) or slots.get("sha256") is not None:
            add(row["issues"], "SELECTOR_SLOTS_HASH", "pre-generation selector slots sha256 must be null")
        expected = {"selected_slot_id", "status", "reason", "input_hashes"}
        if not isinstance(obj.get("expected_result_schema"), dict) or set(obj["expected_result_schema"]) != expected:
            add(row["issues"], "SELECTOR_RESULT_SCHEMA", "selector expected_result_schema has incorrect keys")
        self.contract_check(row["issues"], obj.get("contract"), "selector")
        row["pass"] = not row["issues"]
        return row

    def schedule_check(self):
        public_codes = set()
        restricted_rows = []
        for name in ("SCHEDULE_PUBLIC.json", "SCHEDULE_RESTRICTED.json"):
            value = self.refs.get(name)
            if not isinstance(value, dict):
                continue
            rows = value.get("opportunities")
            if not isinstance(rows, list):
                add(self.issues, "SCHEDULE_SCHEMA", f"{name}.opportunities must be a list")
                continue
            if len(rows) != 32:
                add(self.issues, "SCHEDULE_ROW_COUNT", f"{name} must contain exactly 32 rows",
                    expected=32, actual=len(rows))
            if name == "SCHEDULE_PUBLIC.json":
                hits = forbidden_hits(rows, IDENTITY_KEYS - {"opportunity_code"},
                                      check_values=True, allowed_values=("OPP",))
                if hits:
                    add(self.issues, "PUBLIC_SCHEDULE_IDENTITY_LEAK",
                        "public schedule contains private identity fields", hits=hits[:20])
            codes = []
            for i, item in enumerate(rows):
                if not isinstance(item, dict):
                    add(self.issues, "SCHEDULE_ROW_SCHEMA", "schedule row must be an object", schedule=name, index=i)
                    continue
                code = item.get("opportunity_code")
                if not isinstance(code, str):
                    add(self.issues, "SCHEDULE_CODE", "schedule row missing opportunity_code", schedule=name, index=i)
                elif not OPP_CODE.fullmatch(code):
                    add(self.issues, "SCHEDULE_CODE", "opportunity_code is malformed", schedule=name, index=i, value=code)
                codes.append(code)
                if name == "SCHEDULE_PUBLIC.json" and set(item) != {"opportunity_code"}:
                    add(self.issues, "PUBLIC_SCHEDULE_FIELDS", "public schedule row must contain only opportunity_code",
                        index=i, fields=sorted(set(item) - {"opportunity_code"}))
                if name == "SCHEDULE_RESTRICTED.json":
                    restricted_rows.append(item)
            if len(codes) != len(set(codes)):
                add(self.issues, "SCHEDULE_DUPLICATE_CODE", "schedule opportunity_code values must be unique", schedule=name)
            if name == "SCHEDULE_PUBLIC.json":
                public_codes = set(c for c in codes if isinstance(c, str))
        restricted_codes = set()
        source_counts = {}
        group_counts = {}
        for item in restricted_rows:
            code = item.get("opportunity_code")
            restricted_codes.add(code)
            source = item.get("source_code")
            group = item.get("group")
            if not isinstance(source, str) or not SRC_TOKEN.fullmatch(source):
                add(self.issues, "RESTRICTED_SOURCE_CODE", "restricted schedule source_code must be SRC token", value=source)
            source_counts[source] = source_counts.get(source, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1
            if group not in ("OLD_F", "OLD_G", "NEW_F", "NEW_G"):
                add(self.issues, "RESTRICTED_GROUP", "restricted schedule group is invalid", value=group)
            expected_workflow = "NEW" if isinstance(group, str) and group.startswith("NEW") else "OLD"
            if item.get("workflow") != expected_workflow:
                add(self.issues, "RESTRICTED_WORKFLOW", "restricted workflow does not match group", code=code)
            expected_feedback = "CONTEXT_CARD" if isinstance(group, str) and group.endswith("F") else "NONE"
            if item.get("feedback") != expected_feedback:
                add(self.issues, "RESTRICTED_FEEDBACK", "restricted feedback does not match arm", code=code)
        if public_codes and restricted_codes != public_codes:
            add(self.issues, "SCHEDULE_CODE_SET", "public and restricted opportunity code sets differ",
                public_count=len(public_codes), restricted_count=len(restricted_codes))
        if restricted_rows:
            if set(source_counts) and any(n != 4 for n in source_counts.values()):
                add(self.issues, "SCHEDULE_SOURCE_BALANCE", "each of 8 sources must have four group rows", counts=source_counts)
            if set(group_counts) and any(n != 8 for n in group_counts.values()):
                add(self.issues, "SCHEDULE_GROUP_BALANCE", "each group must have eight source rows", counts=group_counts)

    def packet_check(self):
        role_dir = self.base / "role_packets"
        if not role_dir.is_dir():
            add(self.issues, "ROLE_PACKET_DIR_MISSING", "role_packets directory is missing")
        else:
            for path in sorted(role_dir.glob("*/INPUT.json")):
                self.role_rows.append(self.role_check(path))
            if not self.role_rows:
                add(self.issues, "ROLE_PACKETS_MISSING", "no role packet INPUT.json found")
            elif len(self.role_rows) != 32:
                add(self.issues, "ROLE_PACKET_COUNT", "v2 requires 32 role packets", expected=32, actual=len(self.role_rows))
        selector_dir = self.base / "selector_packets"
        if not selector_dir.is_dir():
            add(self.issues, "SELECTOR_PACKET_DIR_MISSING", "selector_packets directory is missing")
        else:
            for path in sorted(selector_dir.glob("*/INPUT.json")):
                self.selector_rows.append(self.selector_check(path))
            if not self.selector_rows:
                add(self.issues, "SELECTOR_PACKETS_MISSING", "no selector packet INPUT.json found")
            elif len(self.selector_rows) != 16:
                add(self.issues, "SELECTOR_PACKET_COUNT", "selector packets must contain NEW opportunities only (16)",
                    expected=16, actual=len(self.selector_rows))
        restricted = self.refs.get("SCHEDULE_RESTRICTED.json")
        new_codes = set()
        if isinstance(restricted, dict) and isinstance(restricted.get("opportunities"), list):
            new_codes = {x.get("opportunity_code") for x in restricted["opportunities"]
                         if isinstance(x, dict) and x.get("workflow") == "NEW"}
        for row in self.selector_rows:
            code = Path(row["path"]).parent.name
            if new_codes and code not in new_codes:
                add(self.issues, "SELECTOR_OLD_OPPORTUNITY", "selector packet exists for OLD opportunity", code=code)

    def artifact_manifest_check(self):
        """Reconcile packet manifest and selector summary with on-disk packets."""
        manifest = self.refs.get("PACKET_MANIFEST.json")
        if not isinstance(manifest, dict):
            return
        packets = manifest.get("packets")
        if not isinstance(packets, list):
            add(self.issues, "PACKET_MANIFEST_SCHEMA", "PACKET_MANIFEST.packets must be a list")
            return
        if manifest.get("packet_count") != 32 or len(packets) != 32:
            add(self.issues, "PACKET_MANIFEST_COUNT", "packet manifest must contain exactly 32 packets",
                declared=manifest.get("packet_count"), actual=len(packets))
        selector_count = sum(1 for item in packets if isinstance(item, dict) and "selector_input_path" in item)
        if manifest.get("selector_packet_count") != 16 or selector_count != 16:
            add(self.issues, "PACKET_MANIFEST_SELECTOR_COUNT",
                "packet manifest must contain exactly 16 selector packets",
                declared=manifest.get("selector_packet_count"), actual=selector_count)
        seen = set()
        for i, item in enumerate(packets):
            if not isinstance(item, dict):
                add(self.issues, "PACKET_MANIFEST_ROW", "packet manifest row must be an object", index=i)
                continue
            code = item.get("opportunity_code")
            if not isinstance(code, str) or code in seen:
                add(self.issues, "PACKET_MANIFEST_CODE", "packet opportunity codes must be unique", index=i, code=code)
            seen.add(code)
            role_path = path_value(self.base, item.get("role_input_path"))
            if role_path is None:
                add(self.issues, "PACKET_MANIFEST_ROLE_PATH", "role input path must be inside execution", code=code)
            else:
                check_hash(self.issues, "packet_role_input", role_path,
                           item.get("role_input_sha256"), self.base)
                if role_path != self.base / "role_packets" / str(code) / "INPUT.json":
                    add(self.issues, "PACKET_MANIFEST_ROLE_BINDING", "role input path does not match opportunity code", code=code)
            selector_path_value = item.get("selector_input_path")
            if selector_path_value is not None:
                selector_path = path_value(self.base, selector_path_value)
                if selector_path is None:
                    add(self.issues, "PACKET_MANIFEST_SELECTOR_PATH", "selector input path must be inside execution", code=code)
                else:
                    check_hash(self.issues, "packet_selector_input", selector_path,
                               item.get("selector_input_sha256"), self.base)
                    if selector_path != self.base / "selector_packets" / str(code) / "INPUT.json":
                        add(self.issues, "PACKET_MANIFEST_SELECTOR_BINDING", "selector input path does not match opportunity code", code=code)
        summary = self.refs.get("SELECTOR_OPERATION_SUMMARY.json")
        if not isinstance(summary, dict):
            add(self.issues, "SELECTOR_SUMMARY_SCHEMA", "SELECTOR_OPERATION_SUMMARY must be an object")
        else:
            if summary.get("status") != "PREPARED_NOT_STARTED":
                add(self.issues, "SELECTOR_SUMMARY_STATUS", "selector operation summary must be PREPARED_NOT_STARTED", value=summary.get("status"))
            if summary.get("generation_authorization") != "NONE":
                add(self.issues, "SELECTOR_SUMMARY_AUTHORIZATION", "selector operation summary authorization must be NONE")
            if summary.get("result_count") != 0:
                add(self.issues, "SELECTOR_SUMMARY_RESULTS", "selector operation summary must have zero results", value=summary.get("result_count"))
            if summary.get("selector_opportunity_count") != 16:
                add(self.issues, "SELECTOR_SUMMARY_COUNT", "selector operation summary must cover 16 NEW opportunities", value=summary.get("selector_opportunity_count"))

        mapping = self.refs.get("PRIVATE_OPPORTUNITY_MAPPING.json")
        if not isinstance(mapping, dict) or not isinstance(mapping.get("opportunities"), list):
            add(self.issues, "PRIVATE_OPPORTUNITY_MAPPING_SCHEMA", "private opportunity mapping must contain opportunities list")
        elif len(mapping["opportunities"]) != 32:
            add(self.issues, "PRIVATE_OPPORTUNITY_MAPPING_COUNT", "private opportunity mapping must cover 32 opportunities", actual=len(mapping["opportunities"]))

    def run(self):
        for name in ("PROTOCOL_FREEZE.json", "AUTHORIZATION.json", "ACCESS_RULES.json",
                     "PRIVATE_BINDINGS.json", "INPUT_MANIFEST.json", "SCHEDULE_PUBLIC.json",
                     "SCHEDULE_RESTRICTED.json", "SOURCE_ADMISSION_AUDIT.json",
                     "AUTHOR_LAUNCH_AUTHORIZATION.json", "PACKET_MANIFEST.json", "BUILD_MANIFEST.json",
                     "PRIVATE_OPPORTUNITY_MAPPING.json", "SELECTOR_OPERATION_SUMMARY.json"):
            self.load_ref(name)
        self.external_bindings_check()
        self.auth_and_downstream_gate()
        self.execution_identity_check()
        self.input_manifest_check()
        self.synthetic_audit_check()
        self.private_binding_check()
        self.schedule_check()
        self.packet_check()
        self.artifact_manifest_check()
        return self

    def report(self):
        role_pass = sum(x.get("pass", False) for x in self.role_rows)
        selector_pass = sum(x.get("pass", False) for x in self.selector_rows)
        all_pass = not self.issues and role_pass == len(self.role_rows) and selector_pass == len(self.selector_rows)
        manifest = self.refs.get("INPUT_MANIFEST.json")
        return {
            "schema_version": "cc_v3_anonymous_preflight_diagnostic_v1",
            "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "execution_directory": str(self.base),
            "read_only": True,
            "author_or_selector_started": False,
            "question_generation_started": False,
            "solver_started": False,
            "training_started": False,
            "export_started": False,
            "global_issues": self.issues,
            "role_packets": self.role_rows,
            "selector_packets": self.selector_rows,
            "counts": {
                "source_manifest_rows": len(manifest.get("sources", [])) if isinstance(manifest, dict) and isinstance(manifest.get("sources"), list) else 0,
                "role_packets": len(self.role_rows),
                "role_packets_pass": role_pass,
                "selector_packets": len(self.selector_rows),
                "selector_packets_pass": selector_pass,
            },
            "decision": "READY_FOR_MANUAL_REVIEW" if all_pass else "HOLD_AND_QUARANTINE",
            "all_pass": all_pass,
            "limitations": [
                "This preflight does not authorize generation, selector, Solver, training, or export.",
                "Quality, target fidelity, history overlap, and structure diversity remain separate reviews.",
            ],
        }


def exclusive_write(path: Path, raw: bytes):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    base, out = args.base.resolve(), args.out.resolve()
    if out == base or not inside(out, base):
        raise SystemExit("--out must be a separate directory inside --base")
    out.mkdir(mode=0o700, parents=False, exist_ok=True)
    report = V3Preflight(base).run().report()
    report["checker_sha256"] = digest(Path(__file__).read_bytes())
    raw = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    report_path = out / "PREFLIGHT_DIAGNOSTIC_20261107_v3.json"
    exclusive_write(report_path, raw)
    print(json.dumps({"report": str(report_path), "all_pass": report["all_pass"],
                      "decision": report["decision"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
