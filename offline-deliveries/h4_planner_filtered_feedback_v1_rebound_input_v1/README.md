# Rebound R03 Planner input bundle

This bundle removes the runtime dependency on the old aligned run's input snapshot. It copies only the Parent+R01+R02 history identity, Memory planning inputs, verified tracks, freeze metadata, and capacity budget. The old Planner plan, old R03 dialogue, prompts, requests, and responses are not copied.

The filtered feedback export remains the separate authoritative export and is referenced by its immutable artifact inventory. No model or HTTP call was made.
