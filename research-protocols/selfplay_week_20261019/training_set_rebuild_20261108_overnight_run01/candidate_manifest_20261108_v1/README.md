# F/G candidate manifest (2026-11-08)

This directory is a new immutable downstream manifest. Frozen authoring, packet, binding, and canonical-review inputs were read and hash-checked; none were edited. `CANDIDATE_MANIFEST.json` embeds one candidate per review packet and records canonical PASS/HOLD status. `pre_solver_eligible` is only a review gate; Solver, lineage/semantic/exposure gates, and training admission remain pending.

The two canonical HOLD records (both root `cc-train-episode-171871`) are retained for audit but have `pre_solver_eligible=false`.
