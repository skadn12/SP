# Manual question-yield diagnostic

This is a small, purposive source audit, not a population yield estimate. "Manual" means Codex authored and independently reviewed the item; there was no external human annotator. No item in this directory is promoted to training.

Sources reviewed before question writing:

| Root | Dialogue SHA-256 | Latest round | Conservative source finding |
| --- | --- | --- | --- |
| `cc-train-episode-91716` | `9f5081c74844cf37c5bbace426c8ad466465d9190f94e752235ca71ffda6401b` | R02, 14 turns | 0 independent multi-select items, 0 ordering items under the current-round state-change rule. |
| `cc-train-episode-197635` | `7b291fcf058d81520af49077c292e990b2e91af32e2356509928b6820e922e74` | R02, 52 turns | At most 1 independent multi-select item to test, 0 ordering items. |

Count an item only after option-level source review. Each option must have direct supporting or contradicting turns; silence is not a false-option proof. A question should combine at least two independently checkable cross-round event relations with a latest-round state, and near-duplicates within a root count once. Attempt at most two independent multi-select items per root. Ordering needs four performed actions and explicit chronological edges; plans, intended invitations and unordered workshop activities do not qualify.

Freeze an accepted question before the M0 probe. Run six distinct balanced option displays in each of no-context and full-context modes. One item remains one item, regardless of the 12 responses. The predeclared diagnostic band is no-context <= 1/6, full-context 2-4/6 and a gain of >= 2/6. This band is a difficulty probe, not a training admission rule. Preserve semantic failures and revisions instead of selecting only favorable scores.

The earlier `41625` manual V2 result (1/6 without context, 3/6 with context) is an exploratory positive control, reported separately. It is not a new root in this sample.
