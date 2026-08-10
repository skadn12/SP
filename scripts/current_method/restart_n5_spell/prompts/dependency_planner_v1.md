# Dependency Planner v1

Design future parent–child dependencies. Do not mine a relationship from an existing
child dialogue: no child dialogue is present in this task.

For each valid slot, select a real parent graph event and its matching parent turn,
then design one new future child event. The concrete joint result must require both:

```text
A = the selected parent operand
B = the planned future child event
C = the concrete joint result

A alone cannot determine C.
B alone cannot determine C when the child reader does not know the parent history.
A + B uniquely determine C.
```

Allowed operators only:

```text
commitment_outcome
prediction_outcome
qualification_requirement
claim_verification
prior_state_later_decision
```

Reject topic overlap and vague results such as “aligns with”, “is related to”,
“provides a foundation”, “supports the general goal”, “both emphasize”, or “is
relevant to”. Use `no_valid_dependency` rather than inventing a value or relation.

Return stable parent event IDs and P aliases only. Never copy parent quotations.
The local program retrieves the complete frozen parent text.

The planned child event must be genuinely new. It must not repeat the parent fact,
state the inference bridge, or directly state the joint result. List short phrases
whose appearance in the future child dialogue would improperly restate the parent.

All four support bundles must differ. Changing only the operator does not create a
new dependency.
