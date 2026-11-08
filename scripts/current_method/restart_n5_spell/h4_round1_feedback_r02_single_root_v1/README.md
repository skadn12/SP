# Round-1 feedback R02 single-root preflight

This package freezes the first feedback branch before any R02 generation or
Memorax write. It is offline-only: it reads the completed R01 reconstruction,
semantic parity report, feedback card, and token budget, then emits immutable
contracts and a launch plan.

The existing continuation runner is deliberately not launched by this package.
It does not accept a feedback card and its default R02 session is `:R02`, so a
small adapter must be reviewed before a live `:R02-feedback` run.
