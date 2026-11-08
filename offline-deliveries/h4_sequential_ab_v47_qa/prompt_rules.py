"""Evidence-first v47 generation; unchanged v45 independent-review guidance."""

CONTRACT = "v47_evidence_before_proposition_same_fields_v1"

ACTION_EVIDENCE = """
Use the actual source_turns as the sole evidence. Preserve who does or says what,
to which object, in which event, and whether it is proposed, intended, promised,
in progress, reported, or completed. A statement about preparing, intending,
agreeing, or being asked to do something does not establish that it was done.
Conversely, a plan alone does not establish that the action never happened.
Do not convert another speaker's advice into the addressee's action or a report
into an independently confirmed outcome. Retain restrictions, uncertainty and
negation. Check the proposition against its evidence in the full dialogue.
"""

EXACT_QUOTES = """
Copy every quote from ONE continuous span inside its cited source turn. Use a
short single clause with enough words to establish the claim and its modality.
Do not join separated fragments, insert ellipses, replace words or pronouns,
normalize punctuation, or quote a paraphrase of the source. The proposition may
be a faithful paraphrase; the quotation must be copied character-for-character.
Before returning JSON, check that every quoted string appears verbatim inside
the named turn and identifies the same actor, object, action and state.
"""

MULTI_RELATIONS = """
For each multi-select proposition, distinguish three evidence situations before
choosing a relation: the dialogue entails it, explicitly contradicts it in the
same actor/event/time scope, or leaves it unresolved. Emit supports only for
the first and contradicts only for the second. Unmentioned, unconfirmed or
merely more detailed claims are unresolved; they are not false options.
A quote describing a plan neither supports completion nor, on its own,
contradicts completion. A conditional suggestion neither proves its condition
nor the suggested result. For contradicts, cite words that establish a fact
incompatible with the entire proposition in the same scope; explain that
incompatibility briefly. For supports, all asserted details must be supported.
Do not output an unresolved relation or add schema fields: replace an unresolved
proposition with a fully evidenced one, or abstain for that candidate if six
distinct coherent options are unavailable. All six options may be supported;
do not invent false options to meet an assumed answer count.
"""

ORDERING_ANCHORS = """
For ordering, phrase each item as the statement actually made, retaining its
action state and modality. Locate the item's own primary clause, not a later
completion, earlier context, or another speaker's reaction. Every exact anchor
must occur once within its cited turn and must not overlap another item's
anchor. If two items use one turn, copy their separate clauses and order them
left to right. Order transcript appearances, not inferred real-world events.
Do not pad the six items with duplicate claims, unsupported completion, or
unrelated details. Abstain when six coherent, distinct anchors cannot be found.
"""

MULTI_EVIDENCE_FIRST = """
Construct each option from its source evidence. Emit its existing fields in this
order: turn_ref, speaker, quote, proposition, relation, rationale. First copy a
short exact source clause, then write a proposition and its relation grounded in
that clause and the full dialogue. Do not start from an old candidate statement
and search for a quote to justify it. Preserve the quoted action's state and
qualifications in the proposition; use an explicitly incompatible proposition
only when the evidence establishes contradicts. Do not add fields or reasoning.
"""

ORDERING_EVIDENCE_FIRST = """
Construct the question from six source anchors. Emit the existing item_evidence
object for I1-I6 before the items object, then emit gold and order_reason. First
copy six distinct exact source clauses in one coherent scope. Then write each
corresponding item as a faithful statement of its anchor, preserving action state
and modality. Derive gold from the clauses' positions in the supplied dialogue.
Do not start from old candidate items and search for quotes to justify them. Use
only the existing fields; these evidence annotations are not reasoning traces.
"""


def generation_guidance(question_type):
    if question_type not in ("multi_select", "ordering"):
        raise ValueError("unknown_question_type:" + str(question_type))
    specific = MULTI_RELATIONS if question_type == "multi_select" else ORDERING_ANCHORS
    evidence_first = (MULTI_EVIDENCE_FIRST if question_type == "multi_select"
                      else ORDERING_EVIDENCE_FIRST)
    return ("\nCURRENT QA TASK — EVIDENCE CHECK BEFORE FINAL JSON:\n"
            + ACTION_EVIDENCE + specific + EXACT_QUOTES
            + evidence_first
            + "\nKeep the required JSON schema and candidate IDs. Do the checks silently; "
              "return only final questions or abstentions, with short evidence annotations.\n")


def review_guidance():
    return ("\nCURRENT INDEPENDENT REVIEW — ACTION STATE AND EXACT ANCHORS:\n"
            + ACTION_EVIDENCE + EXACT_QUOTES
            + "\nJudge the supplied items independently without rewriting them. An item "
              "claiming completion needs completion evidence; a plan alone is insufficient. "
              "If a claim is unresolved, mark it unsupported and explain the evidence gap "
              "briefly; do not invent an explicit contradiction. Copy each primary and "
              "alternative anchor from its own exact contiguous clause. Preserve the "
              "required review schema and report ambiguity in its existing fields.\n")
