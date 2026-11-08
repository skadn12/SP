"""Offline by default. The live branch requires a separately frozen authorization.

No transport is constructed in offline mode. This task executes offline only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
from contextlib import contextmanager, ExitStack

from . import POLICY_ID
from .budget import validate_budget, ROLES
from .bundle import validate_bundle_feedback
from .capacity import local_counter, VERSION, API_ROLES
from .planning import sha
from .workflow import execute_policy
from .transport_config import validate_complete_endpoint, credential_environment
from .supplement_contract_alignment import select_supplement_version
from .experiment_authorization import load_experiment_scope, experiment_registry
from .root_scope import is_root_scope


@contextmanager
def offline_network_guard():
    # Keep socket a real type so importing ssl/transformers still works; block
    # the operations that can reach a remote address, including datagrams.
    with ExitStack() as stack:
        for target in ("socket.create_connection", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.socket.sendto"):
            stack.enter_context(patch(target, side_effect=RuntimeError("OFFLINE_NETWORK_BLOCKED")))
        yield


def implementation_manifest():
    package = Path(__file__).resolve().parent
    base = package.parent
    legacy = base / "simple_append_memorax_v1_memory_event_continuation_probe_v1"
    paths = list(package.glob("*.py")) + [legacy / name for name in (
        "contracts.py", "promotion.py", "profiles.py", "prompts.py", "incremental_round.py",
        "gate_output_contract.py", "realizer_style.py")]
    paths += [base / "h4_round1_r03_feedback_view_v2" / name for name in (
        "templates.py", "renderers.py", "compiler.py")]
    paths.append(base / "simple_append_v1/assembly.py")
    # Each grant has its own pinned SHA. Adding unrelated grants is not a code change.
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def validate_frozen_sources(bundle):
    from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.incremental_round import build_round_history_parent
    sources = bundle.get("frozen_sources", {})
    if set(sources) != {"history", "plan", "candidates", "tracks"}:
        raise ValueError("frozen_sources_required")
    values = {}
    for name, identity in sources.items():
        content = Path(identity["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise ValueError("frozen_source_sha_mismatch:" + name)
        values[name] = json.loads(content)
    parent = build_round_history_parent(values["history"], round_index=3)
    parent.pop("prior_expanded_dialogue", None)
    if (sha(parent) != sha(bundle["parent"]) or values["plan"]["value"] != bundle["original_plan"]
            or values["candidates"]["candidates"] != bundle["candidates"]
            or values["tracks"]["verified_tracks"] != bundle["verified_tracks"]):
        raise ValueError("bundle_not_identical_to_frozen_sources")


def validate_authorization(*, authorization, bundle, budget, model_config):
    validate_complete_endpoint(model_config.get("endpoint"))
    is_root_scope(bundle)
    validate_budget(budget)
    validate_bundle_feedback(bundle)
    if budget.get("readiness_version") != VERSION:
        raise ValueError("current_readiness_role_capacity_required_for_live")
    if any(budget["role_capacity"][r]["model"] != model_config.get("model") for r in API_ROLES):
        raise ValueError("transport_model_does_not_match_role_capacity")
    if authorization.get("explicitly_authorized") is not True:
        raise ValueError("separate_real_execution_authorization_required")
    expected = {"policy_id": bundle["policy_id"], "identity": bundle["identity"],
                "bundle_sha256": sha(bundle), "budget_sha256": sha(budget),
                "model_config_sha256": sha(model_config), "implementation_manifest": implementation_manifest()}
    if any(authorization.get(k) != v for k, v in expected.items()):
        raise ValueError("authorization_frozen_identity_mismatch")
    if "fixture_notice" in bundle or "offline" in bundle["identity"]["branch_id"].lower():
        raise ValueError("offline_fixture_is_not_real_run_authorization")
    validate_frozen_sources(bundle)


class SingleShotTransport:
    """One HTTP attempt, no redirects/retry/repair/fallback or Memory capability."""
    single_shot_authorized = True
    def __init__(self, config, budget):
        required = {"endpoint", "model", "temperature", "timeout_seconds",
                    "api_key_env", "tokenizer_path", "tokenizer_json_sha256"}
        if set(config) != required:
            raise ValueError("complete_model_config_https_required")
        validate_complete_endpoint(config["endpoint"])
        if not isinstance(config["temperature"], (int, float)) or not 0 <= config["temperature"] <= 2 or config["timeout_seconds"] <= 0:
            raise ValueError("explicit_model_parameters_required")
        path = Path(config["tokenizer_path"])
        if hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest() != config["tokenizer_json_sha256"]:
            raise ValueError("local_tokenizer_sha_mismatch")
        self.counter = local_counter(str(path), config["tokenizer_json_sha256"])
        self.tokenizer = self.counter.tokenizer
        self.config = config
        self.budget = validate_budget(budget)
        self.last_response_envelope = None
        if not os.environ.get(config["api_key_env"]):
            raise ValueError("credential_environment_missing")

    def count_tokens(self, messages):
        return self.counter.count_tokens(messages)

    def request(self, request):
        from .http_evidence import request_once
        return request_once(self, request)


def production_registry():
    return Path(__file__).resolve().parents[5] / "restart-n5-runs/policy_state/expanded_single_plan_incremental_retention_v1"


def main(argv=None):
    from .task_selection import VERSION as SELECTION_VERSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--credential-config", type=Path,
                        help="Explicit legacy project API config; path/SHA must be pinned in authorization")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-mode", choices=("production", "extra-experiment"), default="production")
    parser.add_argument("--experiment-id", help="Previously pinned extra-experiment authorization ID; never registers an ID")
    parser.add_argument("--fresh-normalization-version", choices=(SELECTION_VERSION,),
                        help="Explicit same-task selection after a fresh root supplement")
    args = parser.parse_args(argv)
    if (args.execution_mode == "extra-experiment") != bool(args.experiment_id):
        parser.error("explicit_extra_experiment_mode_and_id_required_together")
    if args.experiment_id and not args.execute:
        parser.error("extra_experiment_mode_requires_explicit_execute_entry")
    read = lambda p: json.loads(p.read_text())
    if args.execute:
        if not all((args.bundle, args.budget, args.model_config, args.authorization)):
            parser.error("BLOCK: live requires explicit bundle, complete budget, model config and separate authorization")
        bundle, budget, model = read(args.bundle), read(args.budget), read(args.model_config)
        authorization = read(args.authorization)
        validate_authorization(authorization=authorization, bundle=bundle, budget=budget, model_config=model)
        if authorization.get("fresh_normalization_version") != args.fresh_normalization_version:
            raise ValueError("fresh_selection_authorization_mismatch")
        if args.fresh_normalization_version is not None and not is_root_scope(bundle):
            raise ValueError("fresh_selection_requires_explicit_fresh_root_entry")
        scope = None
        if args.execution_mode == "extra-experiment":
            scope = load_experiment_scope(args.experiment_id, bundle=bundle, budget=budget,
                model_config=model, production_registry=production_registry())
            if authorization.get("experiment_scope") != scope:
                raise ValueError("execution_authorization_experiment_scope_mismatch")
        elif "experiment_scope" in authorization:
            raise ValueError("experiment_authorization_requires_explicit_mode")
        source = authorization.get("credential_source")
        if args.credential_config:
            if not isinstance(source, dict) or source.get("path") != str(args.credential_config.resolve()):
                raise ValueError("credential_source_requires_matching_authorization")
        elif source is not None:
            raise ValueError("authorized_credential_source_requires_explicit_cli_argument")
        with credential_environment(model, source):
            backend = SingleShotTransport(model, budget)
            # Fixed registry independent of run/branch directory: no attempt reset.
            registry = experiment_registry() if scope is not None else production_registry()
            result = execute_policy(bundle=bundle, budget=budget, backend=backend,
                                    state_registry=registry, artifact_dir=args.run_dir, live=True,
                                    experiment_scope=scope,
                                    fresh_normalization_version=args.fresh_normalization_version)
    else:
        if args.credential_config:
            parser.error("offline mode must not load real credential config")
        from .fixtures import failed_bundle, FixtureBackend, example_budget
        bundle = read(args.bundle) if args.bundle else failed_bundle()
        budget = read(args.budget) if args.budget else example_budget()
        # Both sockets and Memory are unavailable; mocks cannot accidentally
        # become real requests through an imported helper.
        with offline_network_guard():
            validate_frozen_sources(bundle)
            validate_bundle_feedback(bundle)
            result = execute_policy(bundle=bundle, budget=budget, backend=FixtureBackend(bundle),
                                    state_registry=args.run_dir.parent/(args.run_dir.name+"_offline_state"),
                                    artifact_dir=args.run_dir,
                                    fresh_normalization_version=args.fresh_normalization_version)
    print(json.dumps({"status": result["status"], "external_call_count": result["external_call_count"],
                      "offline_fixture_only": result["offline_fixture_only"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
