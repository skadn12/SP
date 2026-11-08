# Structured QA pilot V3

This isolated pilot uses the frozen 188-turn R03 source and V2's ten G02
utterances. The extraction prompt sees only `turn_ref`, `speaker`, and `text`
from eleven fixed historical turns plus all ten G02 turns. It never sees the
event graph or `event_key` metadata. It produces source-cited atomic claims and
their direct opposite claims, both pending human semantic review.

Run from this directory:

```bash
python run.py prepare-extraction
# Inspect extraction/prompt.txt and extraction/manifest.json before any call.
python run.py run-extraction
```

The extraction attempt saves raw output even when validation fails. Structural
validation rejects duplicate JSON keys and checks each cited ref, speaker,
checkpoint, and exact spoken quote. It cannot certify entailment, whether an
action was performed, or whether `opposite_claim` is a direct same-scope
contradiction.

After inspecting `extraction/pending_ledger.json` against the source, a human
reviewer may create `reviewed_ids.json` containing only the accepted IDs:

```json
{
  "status": "SOURCE_REVIEWED",
  "attestation": "SOURCE_AND_OPPOSITE_CLAIM_REVIEWED",
  "reviewer": "reviewer-name",
  "pending_ledger_sha256": "sha256 of extraction/pending_ledger.json",
  "accepted_fact_ids": ["F001", "F002"]
}
```

The attestation covers both each source-supported claim and its opposite
wording, including the same actor, object, version, predicate, and checkpoint.
Reject an ID when either wording is unsupported or ambiguous. This file is a
review record, not a gold answer or hand-authored question.

```bash
python run.py prepare-composition
# Inspect composition/prompt.txt and composition/manifest.json.
python run.py run-composition
python test_run.py
```

The second call selects only reviewed IDs and orientations. The program
renders two-clause options, evidence, candidate ID, and gold deterministically;
it preserves an explicit abstention. All stages use create-only files and one
HTTP attempt without retry. A successful candidate is still unreviewed as a
whole, not Solver-tested, and not training data.
