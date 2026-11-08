"""Whole-event comparison shapes and literal-reference checks, not semantics.

All purpose, relevance, role and action labels are supplied by the model. This
module never derives them from words in a card/dialogue. It checks the explicit
contract and routes explicit negative judgments without rewriting responses.
"""
from copy import deepcopy


IDENTITY_FIELDS = {"purpose", "start_state", "end_state", "roles", "current_trigger", "nearest_old_event"}
NEAREST_FIELDS = {"summary", "history_evidence", "difference"}
ROLE_FIELDS = {"actor_id", "participant", "role"}
COMPARISON_FIELDS = IDENTITY_FIELDS | {"request_acceptance", "experience_state"}
PAIR_FIELDS = {"source", "actual"}
REQUEST_FIELDS = {"required", "verdict", "request_evidence", "acceptance_evidence"}
EXPERIENCE_FIELDS = {"source", "actual", "verdict"}
EXPERIENCE_STATES = {"completed_experience", "future_arrangement", "other"}

PLANNER_INSTRUCTION = """
WHOLE NEW EVENT IDENTITY (dataset_new_event): Return event_identity with EXACTLY
purpose,start_state,end_state,roles,current_trigger,nearest_old_event. Use short
concrete strings naming the action object and its actual status, not merely
'request/answer/thank'. roles is one {actor_id,participant,role} per supplied
card actor, mapped to the actual root participant. nearest_old_event is exactly
{summary,history_evidence,difference}: select the most similar old event from
the FULL history, cite 1-3 exact {turn_ref,quote} spans, and explain the concrete
new event/occasion/trigger and change beyond it. If genuinely no related old
event exists, history_evidence may be [], with a concrete explanation in summary.
Do not claim the new facts were already in history. This is a comparison, not
permission to substitute the old event. Every new claim must be expressed in
the planned developments and subsequently the dialogue. An old task can make
real progress yet still fail the assigned distinct new-event route.

Preserve the supplied card's whole purpose, start/end status and participant
roles. A seed may change the service/activity domain; spa, gardening, comedy or
other source nouns are NOT mandatory keywords. The adapted actual event must
still instantiate the material function and mode. Generic asking/advice/gratitude
cannot justify any unrelated event. If a card maps a NEW request and acceptance,
plan that actual corresponding request before the acceptance, with the right
requester, responder and object. Presenting a list, desiring help, an old request,
or another narrower/later request cannot silently supply it. Steps may share
one development only if its text explicitly preserves their actual order.
Satisfaction after a completed experience is not satisfaction with a future
arrangement. Preserve that distinction even when changing the activity domain.
"""

REVIEW_INSTRUCTION = """
WHOLE-EVENT COMPARISON: Independently return comparison on the event_function
check (function_check in the plan review). Its EXACT fields are purpose,
start_state,end_state,roles,current_trigger,nearest_old_event,request_acceptance,
experience_state. purpose/start_state/end_state each have exactly {source,actual}
short strings: derive source from the assigned card and actual from the actual
plan developments / new dialogue respectively, not from the planner's identity
labels. roles is one {actor_id,participant,role} per card actor. current_trigger
states what initiates THIS event. nearest_old_event is exactly
{summary,history_evidence,difference}; compare the FULL history's most similar
old event using 1-3 exact historical {turn_ref,quote} spans. [] is allowed only
if no related old event was found and summary explains that finding. Difference
must name actual new content, not the mere word 'new' or attached card IDs.

request_acceptance has exactly {required,verdict,request_evidence,acceptance_evidence}.
required is a boolean judgment whether this card requires an actual new request
and corresponding acceptance. verdict is compatible or revise. If required and
compatible, cite BOTH the actual request and the corresponding later acceptance:
1-3 exact plan-development spans in plan review, or 1-3 exact NEW-turn spans in
scene review. In the plan they may share a development ONLY if its actual text
establishes request then acceptance; in dialogue the request must occur in an
earlier NEW turn. Identify the same requester/responder and action object in
the overall function reason. An earlier historical request, merely presenting
risks, an implied desire, or a different later scheduling request cannot fill a
missing current request. Use revise when missing, reversed or mismatched; cite
the available evidence, allowing [] for an absent action. If not required,
both evidence arrays are [] and verdict=compatible; explain applicability in
the overall function reason when material. Do not hide a missing request by
calling a clearly request-and-accept card not required.

experience_state has exactly {source,actual,verdict}; source and actual are
completed_experience, future_arrangement or other. Classify the relevant event
status, not the review time: a reported completed outing remains completed even
though its satisfaction is discussed now. A future reservation/plan is not an
experienced activity. Use revise for a material status substitution; otherwise
compatible. This judgment is separate from whether the scene has some progress.
Every nested revise binds even with positive whole-function and step verdicts.

The whole-function reason explains both sides' actual purposes, status/roles,
the claimed newness versus the nearest old event and any nested negative. Source
domains may change at seed level; no spa/garden/comedy noun test is authorized.
Quotes must both occur verbatim at the cited reference AND actually support the
specific claim, actor, state and relation. Mere substring existence, repeated
request/answer/thank shapes, an old positive reviewer or plan labels are not
semantic support. Do not assert historical hesitation from a quote only about
reconnecting, or accept a service request from a quote only presenting a list.
Do not rewrite an input, fill an omitted action or erase a negative to pass.
Use compact strings and shortest sufficient exact spans within 4096 tokens.
"""


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def actor_ids(card):
    """Enumerate explicit IDs only; never infer a role from action wording."""
    actors = card.get("actors") if isinstance(card, dict) else None
    if isinstance(actors, list) and actors:
        return [a["actor_id"] for a in actors if isinstance(a, dict) and _text(a.get("actor_id"))]
    return list(dict.fromkeys(s["actor_id"] for s in card.get("steps", [])
                             if isinstance(s, dict) and _text(s.get("actor_id"))))


def identity_schema(card):
    return {
        "purpose": "<concrete main event purpose>",
        "start_state": "<state before this event>", "end_state": "<state after this event>",
        "roles": [{"actor_id": actor, "participant": "<exact root participant name>",
                   "role": "<actual role>"} for actor in actor_ids(card)],
        "current_trigger": "<actual new trigger>",
        "nearest_old_event": {"summary": "<nearest old event or explained absence>",
                              "history_evidence": [], "difference": "<concrete newness beyond it>"},
    }


def comparison_schema(card, *, scene=False):
    result = identity_schema(card)
    for key in ("purpose", "start_state", "end_state"):
        result[key] = {"source": "<card requirement>", "actual": "<actual event>"}
    ref = {"turn_ref": "<exact NEW:T reference>", "quote": "<exact new-turn span>"} if scene else {
        "development_index": "<integer development position>", "quote": "<exact development span>"}
    result["request_acceptance"] = {"required": "<boolean>", "verdict": "<compatible|revise>",
                                    "request_evidence": [ref], "acceptance_evidence": [deepcopy(ref)]}
    result["experience_state"] = {"source": "<completed_experience|future_arrangement|other>",
                                  "actual": "<completed_experience|future_arrangement|other>",
                                  "verdict": "<compatible|revise>"}
    return result


def collect_errors(value, card, history, *, path="event_identity", comparison=False,
                   scene=None, turns=None):
    """Check schema, model-supplied role IDs, citations and explicit temporal order.

    A plausible fake claim attached to a real quote still needs semantic review.
    required/applicability and every source/actual label remain model judgments.
    """
    errors = []

    def add(code, target, actual, **details):
        errors.append({"code": code, "path": target, "actual": deepcopy(actual), **deepcopy(details)})

    def fields(item, names, target):
        if not isinstance(item, dict) or set(item) != names:
            add("event_identity_fields_invalid", target, item, allowed=sorted(names))
            return False
        return True

    def text(item, target):
        if not _text(item): add("event_identity_text_required", target, item)

    history = history if isinstance(history, list) else []
    history_map = {h["turn_ref"]: h["text"] for h in history if isinstance(h, dict)
                   and _text(h.get("turn_ref")) and _text(h.get("text"))}
    allowed_people = {h["speaker"] for h in history if isinstance(h, dict) and _text(h.get("speaker"))}
    if not allowed_people and isinstance(scene, dict):
        allowed_people = {name for name in scene.get("actors", []) if _text(name)}

    def citations(items, target, corpus, reference, *, maximum=3):
        refs = []
        if not isinstance(items, list) or len(items) > maximum:
            add("event_identity_evidence_count_invalid", target, items, allowed=f"0..{maximum}")
        seen = set()
        for index, span in enumerate(items if isinstance(items, list) else []):
            at = f"{target}[{index}]"
            if not fields(span, {reference, "quote"}, at): continue
            ref, quote = span.get(reference), span.get("quote")
            ref_type_ok = type(ref) is int if reference == "development_index" else _text(ref)
            actual_text = corpus.get(ref) if ref_type_ok else None
            if actual_text is None:
                add("event_identity_reference_invalid", f"{at}.{reference}", ref, allowed=list(corpus))
            elif not _text(quote) or quote not in actual_text:
                add("event_identity_quote_invalid", f"{at}.quote", quote,
                    cited_reference=ref, cited_text=actual_text)
            else:
                refs.append(ref)
                if (ref, quote) in seen: add("event_identity_duplicate_evidence", at, span)
                seen.add((ref, quote))
        return refs

    if not fields(value, COMPARISON_FIELDS if comparison else IDENTITY_FIELDS, path):
        return errors
    for key in ("purpose", "start_state", "end_state"):
        if comparison:
            if fields(value[key], PAIR_FIELDS, f"{path}.{key}"):
                for side in PAIR_FIELDS: text(value[key][side], f"{path}.{key}.{side}")
        else: text(value[key], f"{path}.{key}")
    text(value["current_trigger"], f"{path}.current_trigger")
    roles = value["roles"]
    expected, observed = actor_ids(card), []
    if not isinstance(roles, list): add("event_identity_roles_not_array", f"{path}.roles", roles)
    for index, role in enumerate(roles if isinstance(roles, list) else []):
        at = f"{path}.roles[{index}]"
        if not fields(role, ROLE_FIELDS, at): continue
        for key in ROLE_FIELDS: text(role[key], f"{at}.{key}")
        if _text(role["actor_id"]): observed.append(role["actor_id"])
        if allowed_people and (not _text(role["participant"]) or role["participant"] not in allowed_people):
            add("event_identity_participant_invalid", f"{at}.participant", role["participant"],
                allowed=sorted(allowed_people))
    if sorted(observed) != sorted(expected):
        add("event_identity_actor_coverage_invalid", f"{path}.roles", observed, allowed=expected)
    nearest = value["nearest_old_event"]
    if fields(nearest, NEAREST_FIELDS, f"{path}.nearest_old_event"):
        for key in ("summary", "difference"): text(nearest[key], f"{path}.nearest_old_event.{key}")
        citations(nearest["history_evidence"], f"{path}.nearest_old_event.history_evidence",
                  history_map, "turn_ref")
    if not comparison: return errors

    request = value["request_acceptance"]
    if fields(request, REQUEST_FIELDS, f"{path}.request_acceptance"):
        required = request["required"]
        if type(required) is not bool:
            add("event_identity_required_not_boolean", f"{path}.request_acceptance.required", required)
        if request["verdict"] not in ("compatible", "revise"):
            add("event_identity_verdict_invalid", f"{path}.request_acceptance.verdict", request["verdict"])
        if turns is None:
            developments = scene.get("developments", []) if isinstance(scene, dict) else []
            corpus = {i: t for i, t in enumerate(developments, 1) if _text(t)}
            reference = "development_index"
        else:
            corpus = {f"NEW:T{i:03d}": t["text"] for i, t in enumerate(turns, 1)
                      if isinstance(t, dict) and _text(t.get("text"))}
            reference = "turn_ref"
        req = citations(request["request_evidence"], f"{path}.request_acceptance.request_evidence", corpus, reference)
        acc = citations(request["acceptance_evidence"], f"{path}.request_acceptance.acceptance_evidence", corpus, reference)
        if required is False and (request["request_evidence"] != [] or request["acceptance_evidence"] != []
                                  or request["verdict"] != "compatible"):
            add("event_identity_not_required_shape_invalid", f"{path}.request_acceptance", request,
                allowed="required=false: empty evidence arrays and compatible")
        if required is True and request["verdict"] == "compatible":
            if not req or not acc:
                add("event_identity_request_acceptance_missing_evidence", f"{path}.request_acceptance", request)
            elif (turns is not None and max(req) >= min(acc)) or (turns is None and max(req) > min(acc)):
                add("event_identity_request_not_before_acceptance", f"{path}.request_acceptance", request,
                    allowed="actual corresponding new request before acceptance; same plan development requires semantic order review")
    experience = value["experience_state"]
    if fields(experience, EXPERIENCE_FIELDS, f"{path}.experience_state"):
        for key in ("source", "actual"):
            if not isinstance(experience[key], str) or experience[key] not in EXPERIENCE_STATES:
                add("event_identity_experience_state_invalid", f"{path}.experience_state.{key}", experience[key],
                    allowed=sorted(EXPERIENCE_STATES))
        if experience["verdict"] not in ("compatible", "revise"):
            add("event_identity_verdict_invalid", f"{path}.experience_state.verdict", experience["verdict"])
        if (all(isinstance(experience.get(key), str) for key in ("source", "actual"))
                and {experience["source"], experience["actual"]} ==
                {"completed_experience", "future_arrangement"} and experience["verdict"] == "compatible"):
            add("event_identity_explicit_experience_status_mismatch", f"{path}.experience_state", experience,
                allowed="revise for these explicitly different experience states")
    return errors


def explicit_negatives(comparison):
    """Only propagate explicit model judgments, with no lexical classifier."""
    return [key for key in ("request_acceptance", "experience_state")
            if isinstance(comparison, dict) and isinstance(comparison.get(key), dict)
            and comparison[key].get("verdict") == "revise"]
