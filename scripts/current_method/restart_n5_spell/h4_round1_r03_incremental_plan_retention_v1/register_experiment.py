"""Explicit local authorization registration. Never sends model/Memory requests."""
import argparse
import json
from pathlib import Path

from .experiment_authorization import register_experiment_record
from .run import offline_network_guard, production_registry, validate_frozen_sources
from .bundle import validate_bundle_feedback
from .budget import validate_budget
from .transport_config import validate_complete_endpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "bundle", "budget", "model-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--explicitly-authorized", action="store_true", required=True)
    args = parser.parse_args(argv)
    with offline_network_guard():
        read = lambda path: json.loads(path.read_text())
        bundle, budget, model = read(args.bundle), read(args.budget), read(args.model_config)
        validate_frozen_sources(bundle)
        validate_bundle_feedback(bundle)
        validate_budget(budget)
        validate_complete_endpoint(model["endpoint"])
        result = register_experiment_record(args.record, expected_sha256=args.expected_sha256,
            explicitly_authorized=args.explicitly_authorized,
            bundle=bundle, budget=budget, model_config=model, production_registry=production_registry())
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
