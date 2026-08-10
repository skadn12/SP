# Outline-Conditioned Dependency Planner v1

Use the frozen expansion outline as the story-planning graph. For array positions
1-4, design at most one dependency for E1-E4 respectively. Use
`no_valid_dependency` when an arc cannot support a natural dependency.

For a valid slot, select only declared parent anchor IDs for that arc and one
declared stage. Design a future child event consistent with that stage. The
dependency must satisfy:

```text
A = information uniquely supplied by the selected parent anchor(s)
B = information uniquely supplied by the planned child event
C = one concrete joint result

A alone cannot determine C.
B alone cannot determine C when the reader does not know the parent history.
A + B uniquely determine C.
```

Positive pattern:

```text
Parent: a person has three years of community-service experience.
Child: a position requires at least three years of that experience.
Joint result: the person satisfies the experience requirement.
```

Negative pattern:

```text
Child: the person qualifies because of three years of experience.
Reject: the child alone states both the parent fact and joint result.
```

Do not use topical similarity as a dependency. Do not claim that one fact caused
another merely because the continuation is plausible. Reject results described
only as aligning, being related, providing a foundation, supporting a general
goal, sharing a theme, or being relevant.

Do not output quotations, original turn IDs, questions, answers, options, or
distractors. Return outline parent anchor IDs only. The local program owns all
dependency IDs, arc IDs, exact evidence text, and downstream QA evidence aliases.

