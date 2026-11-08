"""Evidence-first generation with explicit referent, tense and ordering scope checks."""

CONTRACT = "v54_scoped_fact_contradictions_and_public_scope_v1"

ORDERING_STEM_PREFIX = (
    "Order the statements, using left-to-right clause order within a turn, "
    "as they APPEAR IN THE SUPPLIED DIALOGUE")
ORDERING_STEM = ORDERING_STEM_PREFIX + "."

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
proposition with a fully evidenced one. Each question must contain exactly six
distinct coherent options: 2, 3 or 4 supports and the remaining options
contradicts. If that evidenced mixture is unavailable, abstain for the candidate.
Never remove a true option from Gold or mislabel a supported proposition to meet
the count; construct a different explicitly contradicted proposition instead.
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

CONDITION_PRESERVATION = """
Preserve every condition and scope marker in the evidence and proposition,
including if, unless, only when, when, before, after, and similar qualifiers.
Keep the distinction between an offer, promise, plan, intention, ongoing action,
and completed action; do not strengthen or weaken that level. Preserve nested
intentions too: planning to note or remember an action is not committing to the
action itself. Do not turn a conditional commitment into an unconditional one.
"""

EVIDENCED_RELATION_MIX = """
Build 2-4 supports options and 2-4 contradicts options from scoped facts:
1. Copy a source clause and identify its speaker, exact event, time, condition,
and action state. Keep those fixed in the proposition. A past counterfactual
('would have', 'had I', 'if I had') stays a past hypothetical; do not paraphrase
it as a current intention. Consult the full history for corrections and already
established outcomes before assigning a relation.
2. For a false option, negate an explicit claim at that same time, or substitute
an incompatible value of ONE attribute of that same identified event/object.
Use a different time for the same stated appointment, for example, only when
the evidence fixes its time. Keep recommendations, plans and reports attributed
as recommendations, plans and reports. A prohibition does not disprove a past
action; evidence of one past incident does not rule out a different incident.
3. Test whether the quoted fact and the proposed false option could BOTH be
true in the stated scope. If yes, or if incompatibility requires an unstated
assumption, discard that option and choose another fact. In the existing
rationale, name the explicit conflicting fact and why the two claims cannot
both hold; 'not mentioned', 'not confirmed' or 'different object' is insufficient.
Use explicit incompatibility, never an unmentioned detail or missing confirmation.
Prefer distinct facts; do not fill the set with true/false twins of the same claim.
Do not choose a count by deleting supported Gold labels, randomly flipping
relations, or converting an unresolved claim into contradicts. Rebuild the
option set or abstain when the required evidenced mix cannot be constructed.
Do not arrange truth into a prefix; display permutation is handled separately.
"""

MULTI_PUBLIC_SCOPE = """
Write the public question after constructing the six options. Check every option
against its public speaker, topic, event and time restrictions, not just the
private evidence. A fact may be true in the dialogue yet outside the question.
Prefer a neutral stem such as 'Which statements about TOPIC are supported by
the dialogue? Select all that apply.' Add an exact scene scope when needed.
If options concern several speakers, do not ask only what one speaker supplied.
If a speaker-specific question is useful, all six options must concern that
speaker's statements or actions. Revise the scope or replace the mismatched
option before final JSON; never remove a true option from Gold to hide a mismatch.
"""

VARIED_FACT_COMBINATIONS = """
Within one narrow coherent scope, combine different supported facts and actors
from the supplied turns. Seek a different combination of facts from prior whole
question sets. Supported paraphrases and partial overlap remain allowed; only an
exact copy of the entire six-statement set is excluded by the existing policy.
"""

REFERENT_BINDING = """
Before naming any person or relationship in a proposition, resolve the quoted
pronoun or description using the surrounding turns: who is speaking, who is
being addressed, and who is being discussed can be three different people.
Trace possessives and descriptions such as my friend, our friendship, the other
friend, and that colleague to the same person or relationship throughout the
exchange. An unnamed third person must not become the named listener merely
because the listener is present. Check nearby replies for a different referent
or an explicit correction. If the name is not uniquely established, keep the
source's relational description or choose a different evidenced proposition.
Do not infer gender from a name. Preserve the exact object and scope of a
concern, promise or claim; evidence about one relationship does not establish
the same claim about another relationship.
"""

TEMPORAL_STATUS = """
Compare the proposition's main verb with the quoted clause before choosing its
relation. Will be waiting describes future waiting, not is waiting now; will
meet describes an intended meeting, not has arrived or has already met. Use
plans to, promises to, or says they will when that is what the evidence states.
Current presence and completed actions need explicit evidence of that state at
the relevant time. A later occurrence is not proof that it had already happened
at the earlier quoted turn. Likewise a reported past action must not become an
unperformed plan. Preserve the evidence's time, condition and uncertainty in
every asserted detail, including the option's main verb rather than only its
rationale. Missing confirmation alone never establishes contradicts.
"""

ORDERING_SCOPE = """
Choose the ordering match scope from the public question stem, not from private
item_evidence or a soft slot focus. Questions may cover one scene, several
scenes, or the full dialogue. Use exactly this one-sentence public stem:
""" + ORDERING_STEM + """
To explicitly limit named source scenes, replace the final period with
' within the SCENE_REF scene.' or ' within the SCENE_REF and SCENE_REF scenes.',
using exact scene refs from the source. Keep every other word unchanged.
This is an optional scope declaration, never a required single-scene format.
Do not add a topic summary, event list, sequence words, turn positions, item
labels, answers, or any other text to the public stem. Put the coherent topic
only in the existing private topic field; the stem must not narrate the answer.
Within the stated scope,
check every item's possible occurrences. Replace a vague item such as offering
support or asking for details with the specific source-grounded topic, object
and action that distinguish the intended statement. Do not invent details to
make an anchor unique or silently use a first-occurrence rule. If several
matching occurrences can change the order, choose clearer supported items or
abstain. For a full-dialogue question, matching occurrences in other scenes
remain relevant. Keep the full transcript for contextual reference resolution.
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
            + CONDITION_PRESERVATION + REFERENT_BINDING + TEMPORAL_STATUS
            + VARIED_FACT_COMBINATIONS
            + (ORDERING_SCOPE if question_type == "ordering" else "")
            + (EVIDENCED_RELATION_MIX + MULTI_PUBLIC_SCOPE
               if question_type == "multi_select" else "")
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
