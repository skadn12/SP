#!/usr/bin/env python3
"""Future unified entrypoint with an additional backend completion guard.

V1 remains frozen for the already launched v1 live probe. This wrapper adds a
separately sealed completion protocol and never changes the original v1 code.
"""
from pathlib import Path
import sys

import run_production as v1
from completion_guard import validate_completion
from production_core import bind, read


class ProductionPipeline(v1.ProductionPipeline):
    def _freeze(self):
        super()._freeze()
        v1.seal_equal(self.run_dir / "COMPLETION_PROTOCOL_V2.json", {
            "schema_version": "oet1_followup_completion_protocol_v2",
            "additional_code_bindings": [bind(__file__), bind(Path(__file__).with_name("completion_guard.py"))],
            "require_backend_returncode": 0, "require_no_timeout": True, "require_turn_completed_event": True,
            "old_v1_probe_reclassified": False})

    def _checked_call(self, call_id, prompt):
        response, result = super()._checked_call(call_id, prompt)
        _, events = self.integrity.validate_call_result(self.run_dir / "cli_calls" / call_id / "RESULT.json", prompt)
        validate_completion(result, events)
        return response, result

    def _validate_accepted(self, sd, packet, previous_path, scene):
        super()._validate_accepted(sd, packet, previous_path, scene)
        for role in ("AUTHOR", "REVIEW"):
            call_id = f"{packet['root_id']}__{packet['condition']}__{scene.replace(':', '_')}__{role}"
            result, events = self.integrity.validate_call_result(self.run_dir / "cli_calls" / call_id / "RESULT.json")
            validate_completion(result, events)


if __name__ == "__main__":
    v1.ProductionPipeline = ProductionPipeline
    sys.exit(v1.main())
