# QA Difficulty Integration Probe v1

This is an isolated bridge from the scene-release Realizer probe to the
existing QA measurement stack. It materializes a derived
complete_parent_plus_child dialogue, runs zero-API contract checks, and then
permits a fixed QA Writer/Solver measurement on that derived input.

It does not change production prompts, QA admission, Solver code, Parent/Child
history, or Memorax data. The intended endpoint is composite-QA admission,
not Child Add success.
