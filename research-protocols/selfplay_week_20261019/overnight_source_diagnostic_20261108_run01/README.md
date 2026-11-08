# Overnight source diagnostic run 01

This directory is an isolated preparation and execution workspace for
`OVERNIGHT_PLAN_20261108_v1`.  It never mutates the Conversation Chronicles
source, the historical admission ledgers, the 135-row sealed export, or the
140-candidate review directory.

Current state: `PREPARED_NOT_STARTED`.  The diagnostic has no formal question
generation, Solver calls, or training runs.  A run controller may advance to
sampling only after the training process has been independently observed as
started and healthy, and only with the frozen input manifest written here.

Files created by the preparation script are immutable snapshots.  If a source
hash or exclusion registry changes, the controller must stop and create a new
run directory rather than silently refreshing this one.
