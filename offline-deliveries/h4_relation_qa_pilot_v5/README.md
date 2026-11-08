# Relation-first QA pilot V5

V5 is a same-root diagnostic on the frozen R03+G02 spoken turns. It extracts
two-checkpoint relational propositions directly from dialogue, then selects
five reviewed relation IDs for one multi-select question. It does not reuse
V3's accepted atomic facts, V4's curated pair pool, or hidden event metadata.
Ordering abstains on this source. Preparation never calls the API.

```bash
python run.py prepare-extraction
# Review extraction/prompt.txt and extraction/manifest.json.
python run.py run-extraction
```

One extraction attempt writes its raw response before structural validation.
The structural check verifies unique IDs, two different exact quoted turns
from different evidence windows, source hashes, and absence of turn IDs in
proposition text. It cannot establish semantic entailment, two-span necessity,
strict contradiction, or training value. Review every raw relation against
both spoken turns; reject unsupported, ambiguous, trivial, or compound claims.
For example, do not infer approval happened after a revision from transcript
position, a scheduled study time from Sarah's availability, or a completed
talk from outline approval.

The only permitted review artifact is `reviewed_relations.json` with one
assessment for **every** raw ID. Use exactly this schema, with the SHA-256 of
`extraction/pending_relations.json`:

```json
{
  "status": "SOURCE_REVIEWED",
  "attestation": "TWO_SPAN_CLAIM_AND_STRICT_CONTRADICTION_REVIEWED",
  "reviewer": "reviewer name and review scope",
  "pending_relations_sha256": "64-character SHA-256",
  "assessments": [
    {"relation_id": "R001", "decision": "supported_nontrivial", "reason": "exact two-span justification"},
    {"relation_id": "R002", "decision": "rejected_ambiguous", "reason": "specific ambiguity"}
  ]
}
```

Decisions are `supported_nontrivial`, `rejected_unsupported`,
`rejected_ambiguous`, or `rejected_trivial`. Review must cover the supported
claim and its opposite as written. No textual corrections or new relations
may enter this artifact. The loader revalidates the pending file against the
untouched raw response and checks both hashes before using accepted IDs.

```bash
python run.py prepare-composition
# If fewer than five accepted relations, this writes only an abstention.
# Otherwise inspect composition/prompt.txt and composition/manifest.json.
python run.py run-composition
python -m unittest -v test_run.py
```

The composition call is limited to one HTTP attempt, with no retry. It may
abstain. The program derives all displayed wording, citations, and gold from
reviewed raw relations. The resulting candidate is explicitly unreviewed;
audit each option and its training value before any M0 test. This same-root
diagnostic cannot formally promote a training item or measure production yield.
