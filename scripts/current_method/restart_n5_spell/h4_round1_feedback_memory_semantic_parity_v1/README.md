# Round-1 R01 Memory Semantic Parity Audit

This package is offline-only. It audits the already completed control and
feedback R01 reconstruction smoke artifacts. It does not call Memorax, models,
Solver, QA, or semantic auditor, and it never writes to a namespace.

The audit is fail-closed: the live smoke stored bucket deltas rather than the
full post-R01 bucket response, so relationship, state, temporal, and complete
pollution checks remain `UNKNOWN` until a read-only snapshot is captured.
