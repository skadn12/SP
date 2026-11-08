# H4 R03 initial-only Planner repair v3

This offline bundle regenerates the initial Planner request after the continuation relation repair. It has `append_enabled=false` and has made zero external calls. A continuation is grounded by `verified_track_ref`; `dependencies` is normally empty and may reference only an earlier same-response candidate when a real ordering/prerequisite exists. Self references are rejected.

The request is prepared under `prepared/initial/`; no Planner, Gate, or Realizer call was made.
