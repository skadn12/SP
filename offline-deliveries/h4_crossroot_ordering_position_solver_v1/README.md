# Position-bound ordering Solver input v1

This offline adapter reads the 10 deterministic position-bound v5 ordering candidates. It reruns the source binding and intersects the existing DeepSeek-V3.2 first pass, DeepSeek-V3.2 adversarial pass, and independent GLM-5.2 review. Only `cc-train-episode-194259-order-v5-003` passed those content checks. Its question wording and six item/evidence pairs are byte-equivalent in content to the reviewed version; the adapter maps each review verdict by exact item/evidence content and verifies the new Gold against source turn order. The other nine candidates are excluded from Solver input.

The original review and prior Solver result are historical records. The changed labels alter the prompts and canonical Gold, so prior Solver responses and selection cannot be reused. This adapter performs no generation, semantic repair, inference, export, or training.

The frozen `READY_FOR_EVALUATION.json` and `REVIEW_REBIND_AUDIT.json` are built once with:

```bash
.venv-local4b/bin/python offline-deliveries/h4_crossroot_ordering_position_solver_v1/build.py
```

The versioned runner uses the existing Qwen3-4B ordering protocol and ground/full policy. From the project directory, run:

```bash
.venv-local4b/bin/python offline-deliveries/h4_crossroot_ordering_position_solver_v1/run_solver.py preflight
.venv-local4b/bin/python offline-deliveries/h4_crossroot_ordering_position_solver_v1/run_solver.py live
.venv-local4b/bin/python offline-deliveries/h4_crossroot_ordering_position_solver_v1/report_results.py
```

`preflight` freezes a new run only if no `run/` directory exists. `live` cannot automatically rerun a started evaluation. The report is written independently into this directory. No Solver call has been launched by adapter construction.

Focused checks: `.venv-local4b/bin/python -m unittest discover -s offline-deliveries/h4_crossroot_ordering_position_solver_v1 -p 'test_*.py'`.
