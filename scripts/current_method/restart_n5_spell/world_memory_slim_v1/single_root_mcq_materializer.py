"""Fresh distractor generation and MCQ materialization for one expanded root.

This stage deliberately does not run a paid Solver.  The materialized MCQ is
handed to the local SPELL filter, which owns no-context grounding and the
remaining Solver audits.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from transformers import AutoTokenizer

from ..contrastive_distractor_canary import build_contrastive_distractor_prompt
from ..core import atomic_json, atomic_text, sha256_bytes, sha256_file
from ..ledger import RequestLedger
from ..lineage_distractor_writer import validate_distractor_object
from ..lineage_three_stage_canary import ApiBudget, run_single_object_task
from ..live_backends import (
    LiveSemanticRepairBackend,
    OpenAICompatibleChallengerClient,
    load_challenger_config,
)
from ..single_item_qa_canary import (
    deterministic_answer_placement,
    materialize_single_item_mcq,
)
from .mcq_option_diagnostics import diagnose_option_shape
from .provenance import git_value, hash_tree, project_diff_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DISTRACTOR_MAX_TOKENS = 2048
MODEL_CONTEXT_LIMIT = 40_960
MATERIALIZER_VERSION = (
    "slim_v11_single_root_mcq_materializer_v22_unambiguous_variable_component")
DISTRACTOR_PROMPT_CONTRACT_VERSION = (
    "slim_v11_distractor_contract_v20_factorial_components_"
    "unambiguous_variable_component")

SOLVER_GUIDED_RETRY_DECISIONS = frozenset({
    "grounding_inconclusive", "bad_ground_semantic_leakage",
    "no_context_position_bias",
})

_COUNTERFACTUAL_EXTREME_CUES = re.compile(
    r"\b(?:always|never|all|entirely|completely|without\s+any|refused\s+to)\b",
    flags=re.IGNORECASE,
)


def _component_token_count_word_list(value: str) -> list[str]:
    return [token.casefold() for token in re.findall(
        r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", value)]


def _component_token_count(value: str) -> int:
    return len(_component_token_count_word_list(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def evidence_alignment_audit(gold: dict[str, Any]) -> dict[str, Any]:
    """Preserve evidence admission status outside the strict MCQ schema."""
    mismatch = bool(
        gold.get("evidence_mismatch_marked") or gold.get("evidence_mismatch"))
    reasons = gold.get("evidence_mismatch_reasons") or []
    return {
        "evidence_mismatch_policy": (
            gold.get("evidence_mismatch_policy") or "strict"),
        "evidence_alignment_status": (
            "mismatch_marked" if mismatch else "aligned"),
        "evidence_mismatch_marked": mismatch,
        "evidence_mismatch_reasons": [str(value) for value in reasons],
        "semantic_entailment_independently_verified": bool(
            gold.get("semantic_entailment_independently_verified")),
    }


def validate_distractors_with_shape_gate(
    raw: Any, *, core_qa: dict[str, Any], span_catalog: dict[str, Any],
    placement: dict[str, Any],
) -> dict[str, Any]:
    validation = validate_distractor_object(
        raw, correct_answer_text=core_qa["correct_answer_text"])
    if validation.get("decision") != "distractor_accept":
        return validation
    materialization = materialize_single_item_mcq(
        core_qa=core_qa,
        distractors=validation["distractors"],
        placement=placement,
        span_catalog=span_catalog,
    )
    mcq = materialization["materialized_qa"]
    diagnostic = diagnose_option_shape(
        options=mcq["options"],
        reference_answer_letter=mcq["reference_answer_letter"],
        question=mcq["question"],
    )
    if diagnostic["answer_shape_leakage_risk"]:
        reasons = []
        if diagnostic["correct_unique_medoid_leakage"]:
            reasons.append("correct_answer_unique_option_medoid")
        if diagnostic["single_mutation_star"]:
            reasons.append("gold_plus_three_small_mutations")
        if diagnostic["question_cue_leakage_risk"]:
            reasons.append("question_uniquely_cues_gold_content")
        if diagnostic["gold_length_leakage"]:
            reasons.append("correct_answer_unique_length_outlier")
        if diagnostic["shared_false_trajectory_risk"]:
            reasons.append("distractors_share_one_false_trajectory_component")
        return {**validation, "decision": "distractor_shape_invalid",
                "reasons": reasons, "option_shape": diagnostic}
    return {**validation, "option_shape": diagnostic}


def decompose_cross_boundary_answer(answer: str) -> dict[str, str]:
    """Split an audited answer at its parent/child transition connector."""
    value = " ".join(str(answer or "").strip().split())
    patterns = (
        r"^(?P<parent>.+?)(?P<connector>,\s+then\s+later[,]?\s+)(?P<child>.+)$",
        r"^(?P<parent>.+?)(?P<connector>\s+and\s+later[,]?\s+)(?P<child>.+)$",
        r"^(?P<parent>.+?)(?P<connector>\s+Later[:,]\s*)(?P<child>.+)$",
        r"^(?P<parent>.+?)(?P<connector>\s+Later\s+)(?P<child>.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, value, flags=re.IGNORECASE)
        if match:
            return {
                "parent_component": match.group("parent").strip(),
                "connector": match.group("connector"),
                "child_component": match.group("child").strip(),
            }
    raise ValueError("cross_boundary_answer_not_decomposable")


def compile_factorial_distractors(
    raw: Any, *, correct_answer_text: str,
) -> dict[str, Any]:
    """Compile two false components into a deterministic 2x2 distractor grid."""
    required = {"counterfactual_parent_component",
                "counterfactual_child_component"}
    if not isinstance(raw, dict) or set(raw) != required:
        return {"decision": "distractor_schema_invalid",
                "reasons": ["factorial_component_top_level_schema"]}
    false_parent = raw.get("counterfactual_parent_component")
    false_child = raw.get("counterfactual_child_component")
    if (not isinstance(false_parent, str) or not false_parent.strip() or
            not isinstance(false_child, str) or not false_child.strip()):
        return {"decision": "distractor_schema_invalid",
                "reasons": ["factorial_component_nonempty_strings_required"]}
    try:
        true = decompose_cross_boundary_answer(correct_answer_text)
    except ValueError as exc:
        return {"decision": "distractor_schema_invalid", "reasons": [str(exc)]}
    false_parent = " ".join(false_parent.strip().split())
    false_child = " ".join(false_child.strip().split())
    if false_parent.casefold() == true["parent_component"].casefold():
        return {"decision": "distractor_schema_invalid",
                "reasons": ["counterfactual_parent_matches_gold"]}
    if false_child.casefold() == true["child_component"].casefold():
        return {"decision": "distractor_schema_invalid",
                "reasons": ["counterfactual_child_matches_gold"]}
    parity: dict[str, dict[str, int]] = {}
    parity_errors: list[str] = []
    for name, false_value, true_value in (
            ("parent", false_parent, true["parent_component"]),
            ("child", false_child, true["child_component"])):
        true_count = _component_token_count(true_value)
        false_count = _component_token_count(false_value)
        allowed_delta = max(3, int(true_count * 0.35))
        parity[name] = {
            "true_token_count": true_count,
            "counterfactual_token_count": false_count,
            "allowed_delta": allowed_delta,
        }
        if abs(false_count - true_count) > allowed_delta:
            parity_errors.append(f"counterfactual_{name}_length_mismatch")
        if (_COUNTERFACTUAL_EXTREME_CUES.search(false_value) and
                not _COUNTERFACTUAL_EXTREME_CUES.search(true_value)):
            parity_errors.append(f"counterfactual_{name}_extreme_cue")
    if parity_errors:
        return {
            "decision": "distractor_shape_invalid",
            "reasons": parity_errors,
            "factorial_component_parity": parity,
        }
    connector = true["connector"]
    distractors = [
        false_parent + connector + false_child,
        false_parent + connector + true["child_component"],
        true["parent_component"] + connector + false_child,
    ]
    if len({value.casefold() for value in distractors}) != 3:
        return {"decision": "distractor_schema_invalid",
                "reasons": ["factorial_distractors_not_distinct"]}
    return {
        "decision": "factorial_components_compiled", "reasons": [],
        "factorial_components": {
            "true_parent_component": true["parent_component"],
            "true_child_component": true["child_component"],
            "counterfactual_parent_component": false_parent,
            "counterfactual_child_component": false_child,
            "connector": connector,
        },
        "factorial_component_parity": parity,
        "distractors": distractors,
    }


def validate_factorial_components_with_shape_gate(
    raw: Any, *, core_qa: dict[str, Any], span_catalog: dict[str, Any],
    placement: dict[str, Any],
    solver_guidance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if solver_guidance is not None and isinstance(raw, dict):
        prior = solver_guidance.get("prior_counterfactual_components") or {}
        reused = []
        for component in ("parent", "child"):
            key = f"counterfactual_{component}_component"
            current = " ".join(str(raw.get(key) or "").strip().split()).casefold()
            previous = " ".join(str(prior.get(key) or "").strip().split()).casefold()
            if current and previous and current == previous:
                reused.append(key)
        if reused:
            return {
                "decision": "distractor_shape_invalid",
                "reasons": [f"solver_guided_reused_prior_{key}" for key in reused],
                "solver_guided_reused_components": reused,
            }
    compiled = compile_factorial_distractors(
        raw, correct_answer_text=core_qa["correct_answer_text"])
    if compiled.get("decision") != "factorial_components_compiled":
        return compiled
    validation = validate_distractors_with_shape_gate(
        {"distractors": compiled["distractors"]}, core_qa=core_qa,
        span_catalog=span_catalog, placement=placement)
    option_shape = validation.get("option_shape") or {}
    local_mutations = [
        profile for profile in
        (option_shape.get("gold_to_distractor_profiles") or [])
        if profile.get("small_local_mutation")
    ]
    if (validation.get("decision") == "distractor_accept" and
            local_mutations):
        # A 2x2 grid exposes each counterfactual component in isolation.  If
        # either hybrid differs from the gold by only one local edit, the model
        # may have supplied a synonym (ambiguous) or an obvious slot swap
        # (leaky).  Require a whole-proposition counterfactual before SPELL.
        return {
            **validation,
            "decision": "distractor_shape_invalid",
            "reasons": ["factorial_component_small_local_mutation"],
            "factorial_component_local_mutations": local_mutations,
            "factorial_components": compiled["factorial_components"],
        }
    return {**validation,
            "factorial_components": compiled["factorial_components"]}


def _shape_retry_suffix(raw: Any, validation: dict[str, Any]) -> str:
    parity = validation.get("factorial_component_parity") or {}
    length_lines: list[str] = []
    for name in ("parent", "child"):
        row = parity.get(name) or {}
        true_count = int(row.get("true_token_count") or 0)
        allowed_delta = int(row.get("allowed_delta") or 0)
        if true_count and allowed_delta:
            length_lines.append(
                f"- counterfactual_{name}_component: "
                f"{max(1, true_count - allowed_delta)} to "
                f"{true_count + allowed_delta} word tokens (target {true_count}).")
    previous_lines: list[str] = []
    if isinstance(raw, dict):
        for key in ("counterfactual_parent_component",
                    "counterfactual_child_component"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                previous_lines.append(
                    f"- Do not repeat this previous invalid {key}: "
                    + json.dumps(value.strip(), ensure_ascii=False))
    return "\n".join([
        "OPTION SHAPE CONTRACT RETRY.",
        "The previous counterfactual components were structurally invalid or leaked the gold "
        "through option shape: " + ", ".join(validation.get("reasons") or []),
        "LENGTH RANGES (count word tokens exactly):",
        *(length_lines or ["- Keep each component within the true component's stated range."]),
        *previous_lines,
        "Replace both counterfactual components. The program, not you, will "
        "combine them into three 2x2 distractors. Keep both components in the "
        "SAME people/event/task domain named by the question. "
        "Keep question-stated people and contextual anchors represented in every "
        "choice, on both the earlier and later side. If the question is about a "
        "weather contingency and later logistics, every choice must remain about "
        "a weather contingency and later logistics; do not substitute budget, "
        "venue size, or performer lineup as the earlier issue. Change the decisive "
        "state, cause, action, decision, or outcome; "
        "do not evade the question by switching to unrelated topics.",
        "Each component must change a whole proposition, not one token. Never "
        "swap only a date, place, amount, person, or adjective.",
        "A component that merely replaces one task subtype, concern noun, "
        "activity, or short predicate will be rejected as a local mutation. "
        "The false proposition must be mutually incompatible with the true "
        "proposition in the frozen context, not a synonym or an additional fact "
        "that could be true at the same time.",
        "Keep each replacement component close to the corresponding true "
        "component's token length. Do not make false choices conspicuously longer "
        "or shorter.",
        "Do not signal falsity with categorical or melodramatic wording such as "
        "always, never, all challenges, completely, refused to, or without any "
        "support unless the true component uses the same degree of certainty.",
        "Keep all four choices comparable in length, specificity, detail, and "
        "causal coherence. The gold must not be the uniquely central or most "
        "complete option.",
        "Match evaluative tone across all four choices: social desirability, "
        "emotional warmth, competence, responsibility, and certainty must not make "
        "the gold uniquely admirable or the false choices uniquely cold, careless, "
        "irrelevant, or inept. When the gold describes support, use a different "
        "credible form or focus of support as the counterfactual, not withdrawal or "
        "indifference merely to become false.",
        "Match the question's abstraction level in both counterfactual components. "
        "Do not make the gold the only combination that repeats or semantically "
        "matches category labels from the question, such as personal/professional, "
        "emotional/practical, or relationship/task focus.",
        "The two replacement components must be CROSS-COMPOSABLE: false-parent "
        "+ true-child, true-parent + false-child, and false-parent + false-child "
        "must each read as a coherent, plausible trajectory without the dialogue. "
        "Do not create a false component that makes its hybrid option internally "
        "contradictory or obviously less natural than the gold.",
        "Do not retain a framing phrase from the true component when that phrase "
        "contradicts the replacement behavior. The wording inside each false "
        "component must accurately describe its own alternative, not label "
        "practical advice as emotional reflection or label withdrawal as support.",
        "Do not return complete distractors. Do not keep most of a gold component "
        "and swap only a few tokens.",
        "Do not return the question, gold, evidence, diagnostics, or explanations.",
        "Do not reproduce or minimally edit any previous distractor.",
        'OUTPUT ONLY {"counterfactual_parent_component":"...",'
        '"counterfactual_child_component":"..."}',
    ])


def build_shape_balanced_distractor_prompt(
    *, frozen_dialogue: list[dict[str, Any]], span_catalog: dict[str, Any],
    base_qa: dict[str, Any], solver_guidance: Mapping[str, Any] | None = None,
) -> str:
    decomposition = decompose_cross_boundary_answer(
        str(base_qa["correct_answer_text"]))
    parent_count = _component_token_count(decomposition["parent_component"])
    child_count = _component_token_count(decomposition["child_component"])
    parent_delta = max(3, int(parent_count * 0.35))
    child_delta = max(3, int(child_count * 0.35))
    payload = {
        "frozen_question": base_qa["question"],
        "true_parent_component": decomposition["parent_component"],
        "true_child_component": decomposition["child_component"],
        "connector": decomposition["connector"],
        "counterfactual_component_word_token_limits": {
            "parent": {"minimum": max(1, parent_count - parent_delta),
                       "target": parent_count,
                       "maximum": parent_count + parent_delta},
            "child": {"minimum": max(1, child_count - child_delta),
                      "target": child_count,
                      "maximum": child_count + child_delta},
        },
        "selected_evidence": base_qa.get("selected_evidence") or [],
        "full_frozen_dialogue": frozen_dialogue,
    }
    if solver_guidance is not None:
        payload["prior_solver_grounding_failure"] = dict(solver_guidance)
    contract = [
        f"PROMPT_CONTRACT_VERSION: {DISTRACTOR_PROMPT_CONTRACT_VERSION}",
        "FACTORIAL COUNTERFACTUAL COMPONENT CONTRACT",
        "The program has split the correct answer into one parent-side component "
        "and one child-side component. Return exactly one plausible false parent "
        "component and one plausible false child component.",
        "Freeze the people, event, task, and broad context explicitly named by the "
        "question. Every distractor must answer that same question in that same "
        "domain; never substitute unrelated event types merely to become false.",
        "Preserve the question's parent-side topic category and child-side topic "
        "category in every distractor. For example, a question about an earlier "
        "weather contingency and later logistics requires all choices to describe "
        "an earlier weather contingency and later logistics. Vary the forecast, "
        "response, state, or outcome, not the topic category.",
        "Each false component must preserve its corresponding subject and broad "
        "topic but replace the entire state, decision, action, or outcome with a "
        "credible alternative contradicted by the frozen dialogue.",
        "Prefer a hard negative assembled from a concrete same-domain detail "
        "that appears elsewhere in the frozen dialogue but belongs to a different "
        "speaker, episode, time, or outcome. It must still be false for the exact "
        "subject and comparison asked here.",
        "The false component must be mutually incompatible with the true "
        "component in this context. Do not return a synonym, a compatible extra "
        "detail, or a one-slot substitution such as changing only the activity, "
        "concern noun, location, date, or object.",
        "Preserve grammar, tense, answer type, specificity, and approximate length "
        "so either false component can substitute for its true counterpart.",
        "Obey the exact minimum/target/maximum word-token ranges in the frozen "
        "input. Count tokens before returning; use one concise clause per component.",
        "All four compiled choices must sound equally moderate and plausible "
        "without the dialogue. Avoid absolutes, exaggerated confidence, blanket "
        "refusal, and explicit meta-phrases such as 'no shift'.",
        "EVALUATIVE PARITY: match social desirability, emotional warmth, competence, "
        "responsibility, and certainty across true and false components. Do not make "
        "the gold uniquely supportive, empathetic, mature, balanced, or professionally "
        "appropriate while false choices sound withdrawn, careless, unrelated, or "
        "incompetent. Prefer a different credible kind or target of support.",
        "QUESTION-LABEL PARITY: every compiled choice must answer the same abstract "
        "dimensions named by the question. The gold must not be the only combination "
        "that repeats or semantically matches the question's category labels.",
        "CROSS-COMPOSABILITY REQUIREMENT: the false parent and false child must "
        "remain coherent both together and when each is paired with the unchanged "
        "true component.",
        "Before returning, mentally compose all four combinations. Each must be "
        "grammatical, internally coherent, equally responsive to the question, and "
        "plausible without access to the frozen dialogue. The correct combination "
        "must not be identifiable merely because it is the only coherent one.",
        "A false component must describe its alternative honestly. Do not keep a "
        "true framing label when the new predicate no longer fits that label; for "
        "example, do not call practical problem-solving 'emotional reflection'.",
        "Do not merely negate the gold or swap one name, noun, date, adjective, or "
        "short phrase. Do not include the connector in either component.",
        "The program will create false-parent/false-child, false-parent/true-child, "
        "and true-parent/false-child options. Do not return complete options.",
        "Return no question, gold, evidence, explanation, or additional field.",
        "FROZEN INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        'OUTPUT ONLY {"counterfactual_parent_component":"...",'
        '"counterfactual_child_component":"..."}',
    ]
    if solver_guidance is not None:
        contract[1:1] = [
            "SOLVER-GUIDED RETRY: the prior option set is frozen in the input and "
            "failed no-context grounding. The question, gold, evidence, and source "
            "dialogue remain immutable; regenerate both false components only.",
            "Do not reuse either prior counterfactual component or any complete prior "
            "false option. The prior gold-selection count is diagnostic, not a target "
            "to manipulate by position.",
            "For every noun phrase or comparison dimension in the frozen question, "
            "make each false component supply a credible same-category alternative. "
            "A false option must not become dismissible merely because it switches "
            "from team to individual, support to indifference, challenge to success, "
            "or another category excluded by the question.",
            "Use no_context_component_selection_feedback as position-neutral evidence. "
            "A false factor selected zero or rarely across the 12 balanced rollouts is "
            "too easy to dismiss. Replace it with a genuinely plausible alternative "
            "that satisfies the question's presuppositions and evaluative stance while "
            "remaining contradicted by the dialogue. Change the substantive reason, "
            "method, target, or outcome; do not merely replace concern with hesitation "
            "or another near-synonym.",
        ]
    return "\n\n".join(contract)


def build_solver_guided_full_distractor_prompt(
    *, frozen_dialogue: list[dict[str, Any]], span_catalog: dict[str, Any],
    base_qa: dict[str, Any], solver_guidance: Mapping[str, Any],
) -> str:
    payload = {
        "frozen_question": base_qa["question"],
        "frozen_correct_answer": base_qa["correct_answer_text"],
        "selected_evidence": base_qa.get("selected_evidence") or [],
        "prior_solver_grounding_failure": dict(solver_guidance),
        "full_frozen_dialogue": frozen_dialogue,
    }
    locked_factor = str(solver_guidance.get("locked_factor") or "")
    locked_mode = locked_factor in {"parent", "child"}
    if locked_mode:
        side = "earlier/parent" if locked_factor == "child" else "later/child"
        rules = [
            "Return exactly three one-sided replacement components, not complete "
            "answers or complete two-time trajectories.",
            f"Each string is only the {side} proposition. The program will add the "
            "connector and the frozen opposite-side proposition after your response.",
            "The question, correct answer, evidence, dialogue, connector, and locked "
            "factor are immutable.",
            "Generate alternatives only for variable_factor. Never repeat the locked "
            "factor and never write the second side of the comparison.",
            "When variable_factor is parent, stop at the end of the earlier proposition. "
            "When variable_factor is child, write only the later proposition. Do not "
            "include the connector or words such as Later that start another time side.",
            "Each variable component must be a credible same-category alternative to "
            "the true variable component, false under the frozen dialogue, and plausible "
            "without the dialogue.",
            "Preserve the named person, answer type, abstraction level, and evaluative "
            "category for this one side. Change the substantive method, reason, target, "
            "or outcome rather than negating it or switching categories.",
            "Use no_context_component_selection_feedback to diagnose the prior leak. A "
            "factor selected zero or rarely was too easy to dismiss; do not paraphrase it.",
            "Obey variable_component_word_token_limits exactly. Use the true variable "
            "component only as a length/type reference and do not copy it.",
            "The program will check the compiled four options for comparable length, "
            "grammar, specificity, social desirability, competence, and coherence.",
            "Do not reuse any prior false option or prior counterfactual component. Do "
            "not copy the correct answer, create a synonym, or make a second correct answer.",
            "Do not use absolutes, melodrama, omitted required facets, or one-word slot swaps.",
            "Return no complete answer, question, evidence, explanation, diagnostics, "
            "or extra field.",
        ]
        contract_title = "SOLVER-GUIDED VARIABLE-COMPONENT CONTRACT"
    else:
        rules = [
            "Return exactly three complete distractor answers. Do not return components.",
            "The question, correct answer, evidence, and dialogue are immutable.",
            "Every distractor must directly answer the exact frozen question using the "
            "same named people, time comparison, answer type, abstraction level, and "
            "number of causal or developmental facets as the correct answer.",
            "Preserve every presupposition and evaluative category in the question.",
            "Each distractor must be false under the frozen dialogue but plausible "
            "without the dialogue. Change the substantive method, reason, target, "
            "sequence, or outcome rather than negating the correct answer.",
            "Keep all four options comparable in length, grammar, specificity, social "
            "desirability, emotional warmth, competence, and causal coherence.",
            "Do not reuse any prior false option. Do not copy the correct answer, create "
            "a synonym, or make a second correct answer.",
            "Do not use absolutes, melodrama, category switches, omitted facets, or "
            "one-word slot swaps. Each distractor must be a complete independent trajectory.",
            "Return no question, evidence, explanation, diagnostics, or extra field.",
        ]
        contract_title = "SOLVER-GUIDED FULL-OPTION DISTRACTOR CONTRACT"
    return "\n\n".join([
        f"PROMPT_CONTRACT_VERSION: {DISTRACTOR_PROMPT_CONTRACT_VERSION}",
        contract_title + "\n- " + "\n- ".join(rules),
        "FROZEN INPUT\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ('OUTPUT ONLY {"counterfactual_variable_components":["...","...","..."]}'
         if locked_mode else
         'OUTPUT ONLY {"distractors":["...","...","..."]}'),
    ])


def compile_solver_guided_variable_components(
    raw: Any, *, solver_guidance: Mapping[str, Any],
) -> dict[str, Any]:
    if (not isinstance(raw, dict) or
            set(raw) != {"counterfactual_variable_components"}):
        return {"decision": "distractor_schema_invalid",
                "reasons": ["solver_guided_variable_component_schema"]}
    values = raw.get("counterfactual_variable_components")
    if (not isinstance(values, list) or len(values) != 3 or
            any(not isinstance(value, str) or not value.strip()
                for value in values)):
        return {"decision": "distractor_schema_invalid",
                "reasons": ["solver_guided_three_variable_components_required"]}
    values = [" ".join(value.strip().split()) for value in values]
    if len({value.casefold() for value in values}) != 3:
        return {"decision": "distractor_schema_invalid",
                "reasons": ["solver_guided_variable_components_not_distinct"]}
    components = solver_guidance.get("prior_counterfactual_components") or {}
    locked = str(solver_guidance.get("locked_factor") or "")
    connector = str(components.get("connector") or "")
    if locked == "child":
        shared = str(components.get("true_child_component") or "")
        distractors = [value + connector + shared for value in values]
    elif locked == "parent":
        shared = str(components.get("true_parent_component") or "")
        distractors = [shared + connector + value for value in values]
    else:
        return {"decision": "distractor_schema_invalid",
                "reasons": ["solver_guided_locked_factor_required"]}
    if not connector or not shared:
        return {"decision": "distractor_schema_invalid",
                "reasons": ["solver_guided_locked_component_incomplete"]}
    true_variable = str(solver_guidance.get("true_variable_component") or "")
    prior_variable = str(solver_guidance.get("prior_variable_component") or "")
    limits = solver_guidance.get("variable_component_word_token_limits") or {}
    minimum = int(limits.get("minimum") or 0)
    maximum = int(limits.get("maximum") or 0)
    connector_words = set(_component_token_count_word_list(connector))
    reasons = []
    for value in values:
        count = _component_token_count(value)
        value_words = set(_component_token_count_word_list(value))
        if minimum and maximum and not minimum <= count <= maximum:
            reasons.append("solver_guided_variable_component_length_mismatch")
        if connector_words and connector_words.issubset(value_words):
            reasons.append("solver_guided_variable_component_contains_connector")
        if shared.casefold() in value.casefold():
            reasons.append("solver_guided_variable_component_contains_locked_factor")
        if true_variable and value.casefold() == true_variable.casefold():
            reasons.append("solver_guided_variable_component_matches_gold")
        if prior_variable and value.casefold() == prior_variable.casefold():
            reasons.append("solver_guided_variable_component_reuses_prior")
    if reasons:
        return {"decision": "distractor_shape_invalid",
                "reasons": sorted(set(reasons)),
                "counterfactual_variable_components": values}
    return {"decision": "solver_guided_components_compiled", "reasons": [],
            "counterfactual_variable_components": values,
            "distractors": distractors}


def validate_solver_guided_distractors_with_shape_gate(
    raw: Any, *, core_qa: dict[str, Any], span_catalog: dict[str, Any],
    placement: dict[str, Any], solver_guidance: Mapping[str, Any],
) -> dict[str, Any]:
    locked_factor = str(solver_guidance.get("locked_factor") or "")
    compiled = None
    if locked_factor in {"parent", "child"}:
        compiled = compile_solver_guided_variable_components(
            raw, solver_guidance=solver_guidance)
        if compiled["decision"] != "solver_guided_components_compiled":
            return compiled
        raw = {"distractors": compiled["distractors"]}
    if isinstance(raw, dict) and isinstance(raw.get("distractors"), list):
        prior_options = solver_guidance.get("prior_options") or {}
        prior_gold = str(
            solver_guidance.get("prior_reference_answer_letter") or "")
        prior_false = {
            " ".join(str(text).strip().split()).casefold()
            for letter, text in prior_options.items() if letter != prior_gold
        }
        reused = [
            value for value in raw["distractors"]
            if isinstance(value, str) and
            " ".join(value.strip().split()).casefold() in prior_false
        ]
        if reused:
            return {
                "decision": "distractor_shape_invalid",
                "reasons": ["solver_guided_reused_prior_full_option"],
                "solver_guided_reused_options": reused,
            }
    validation = validate_distractors_with_shape_gate(
        raw, core_qa=core_qa, span_catalog=span_catalog,
        placement=placement)
    if compiled is not None:
        validation = {**validation,
                      "counterfactual_variable_components": compiled[
                          "counterfactual_variable_components"]}
    return validation


def build_distractor_validator(
    *, core_qa: dict[str, Any], span_catalog: dict[str, Any],
    placement: dict[str, Any], solver_guidance: Mapping[str, Any] | None,
) -> Callable[[Any], dict[str, Any]]:
    if solver_guidance is not None:
        return lambda raw: validate_solver_guided_distractors_with_shape_gate(
            raw, core_qa=core_qa, span_catalog=span_catalog,
            placement=placement, solver_guidance=solver_guidance)
    return lambda raw: validate_factorial_components_with_shape_gate(
        raw, core_qa=core_qa, span_catalog=span_catalog,
        placement=placement)


def _solver_guided_full_retry_suffix(
    raw: Any, validation: dict[str, Any], *,
    solver_guidance: Mapping[str, Any],
) -> str:
    previous = []
    output = 'OUTPUT ONLY {"distractors":["...","...","..."]}'
    field = "distractors"
    if str(solver_guidance.get("locked_factor") or "") in {"parent", "child"}:
        field = "counterfactual_variable_components"
        output = ('OUTPUT ONLY {"counterfactual_variable_components":'
                  '["...","...","..."]}')
    if isinstance(raw, dict) and isinstance(raw.get(field), list):
        previous = [str(value) for value in raw[field]
                    if isinstance(value, str)]
    locked_mode = field != "distractors"
    variable_factor = str(solver_guidance.get("variable_factor") or "")
    return "\n".join([
        ("VARIABLE-COMPONENT RETRY." if locked_mode else
         "FULL-OPTION DISTRACTOR RETRY."),
        "The previous three answers failed the strict schema or option-shape gate: "
        + ", ".join(validation.get("reasons") or []),
        "Replace all three. Do not repeat these invalid answers: "
        + json.dumps(previous, ensure_ascii=False),
        (("Each replacement is only the " + variable_factor +
          "-side proposition, not a full answer. Do not include a connector, a second "
          "time clause, or the locked factor. It must remain false in the dialogue.")
         if locked_mode else
         "Every replacement must independently answer every premise and comparison "
         "dimension in the frozen question while remaining false in the dialogue."),
        "VARIABLE COMPONENT TOKEN LIMITS: " + json.dumps(
            solver_guidance.get("variable_component_word_token_limits") or {},
            ensure_ascii=False, sort_keys=True),
        ("Return only unlocked one-sided components; the program preserves and adds "
         "the locked factor." if locked_mode else
         "Do not return components, question, gold, evidence, or explanations."),
        output,
    ])


def solver_guided_retry_eligible(final_decision: str) -> bool:
    return str(final_decision or "") in SOLVER_GUIDED_RETRY_DECISIONS


def derive_component_selection_feedback(
    *, spell_result: Mapping[str, Any],
    factorial_components: Mapping[str, Any],
) -> dict[str, Any]:
    """Map permuted Solver choices back to the deterministic 2x2 factors."""
    true_parent = str(factorial_components.get("true_parent_component") or "")
    true_child = str(factorial_components.get("true_child_component") or "")
    false_parent = str(
        factorial_components.get("counterfactual_parent_component") or "")
    false_child = str(
        factorial_components.get("counterfactual_child_component") or "")
    connector = str(factorial_components.get("connector") or "")
    if not all((true_parent, true_child, false_parent, false_child, connector)):
        raise ValueError("solver_guidance_factorial_components_incomplete")

    def normalized(value: str) -> str:
        return " ".join(value.strip().split()).casefold()

    combination_text = {
        "false_parent_false_child": false_parent + connector + false_child,
        "false_parent_true_child": false_parent + connector + true_child,
        "true_parent_false_child": true_parent + connector + false_child,
        "true_parent_true_child": true_parent + connector + true_child,
    }
    by_text = {normalized(value): key for key, value in combination_text.items()}
    combination_counts = {key: 0 for key in combination_text}
    unknown = 0
    for order in spell_result.get("balanced_no_context_orders") or []:
        rollout = order.get("rollout") or {}
        for row in rollout.get("rollouts") or []:
            selected = normalized(str(row.get("parsed_option_text") or ""))
            combination = by_text.get(selected)
            if combination is None:
                unknown += 1
            else:
                combination_counts[combination] += 1
    factor_counts = {
        "true_parent": (combination_counts["true_parent_false_child"] +
                        combination_counts["true_parent_true_child"]),
        "false_parent": (combination_counts["false_parent_false_child"] +
                         combination_counts["false_parent_true_child"]),
        "true_child": (combination_counts["false_parent_true_child"] +
                       combination_counts["true_parent_true_child"]),
        "false_child": (combination_counts["false_parent_false_child"] +
                        combination_counts["true_parent_false_child"]),
    }
    return {
        "combination_selection_counts": combination_counts,
        "factor_selection_counts": factor_counts,
        "mapped_selection_count": sum(combination_counts.values()),
        "unknown_selection_count": unknown,
    }


def load_solver_guidance(
    *, prior_mcq_run: Path, prior_spell_run: Path,
    expected_qa_id: str, expected_question: str, expected_answer: str,
) -> dict[str, Any]:
    mcq = _load(prior_mcq_run / "qa/final_mcq.json")
    spell_rows = json.loads(
        (prior_spell_run / "filter_results.json").read_text(encoding="utf-8"))
    if not isinstance(spell_rows, list) or len(spell_rows) != 1 or not isinstance(
            spell_rows[0], dict):
        raise ValueError("solver_guidance_filter_result_invalid")
    result = spell_rows[0]
    if (str(mcq.get("qa_id") or "") != expected_qa_id or
            str(mcq.get("question") or "") != expected_question or
            str(mcq.get("canonical_answer_text") or "") != expected_answer):
        raise ValueError("solver_guidance_frozen_qa_mismatch")
    final_decision = str(result.get("final_decision") or "")
    if not solver_guided_retry_eligible(final_decision):
        raise ValueError("solver_guidance_decision_not_retryable")
    grounding = result.get("balanced_grounding") or {}
    materializer_summary = _load(
        prior_mcq_run / "reports/MCQ_MATERIALIZER_SUMMARY.json")
    factorial_components = (
        (materializer_summary.get("distractor_result") or {}).get(
            "accepted_validation", {}).get("factorial_components") or {})
    component_feedback = derive_component_selection_feedback(
        spell_result=result, factorial_components=factorial_components)
    factor_counts = component_feedback["factor_selection_counts"]
    mapped_count = component_feedback["mapped_selection_count"]
    required_shared_components = []
    locked_factor = None
    if mapped_count and factor_counts["true_parent"] == mapped_count:
        required_shared_components.append(
            factorial_components["true_parent_component"])
        locked_factor = "parent"
    if mapped_count and factor_counts["true_child"] == mapped_count:
        required_shared_components.append(
            factorial_components["true_child_component"])
        locked_factor = "child" if locked_factor is None else "both"
    true_variable_component = None
    prior_variable_component = None
    variable_limits = None
    if locked_factor == "child":
        true_variable_component = factorial_components["true_parent_component"]
        prior_variable_component = factorial_components[
            "counterfactual_parent_component"]
    elif locked_factor == "parent":
        true_variable_component = factorial_components["true_child_component"]
        prior_variable_component = factorial_components[
            "counterfactual_child_component"]
    if true_variable_component:
        true_count = _component_token_count(true_variable_component)
        delta = max(3, int(true_count * 0.35))
        variable_limits = {"minimum": max(1, true_count - delta),
                           "target": true_count,
                           "maximum": true_count + delta}
    return {
        "prior_final_decision": final_decision,
        "semantic_correct_count": int(grounding.get("semantic_correct_count") or 0),
        "total_valid_rollouts": int(grounding.get("total_valid_rollouts") or 0),
        "semantic_correct_rate": grounding.get("semantic_correct_rate"),
        "prior_question": mcq["question"],
        "prior_options": dict(mcq["options"]),
        "prior_reference_answer_letter": mcq["reference_answer_letter"],
        "prior_counterfactual_components": factorial_components,
        "no_context_component_selection_feedback": component_feedback,
        "required_shared_components": required_shared_components,
        "locked_factor": locked_factor,
        "variable_factor": ({"parent": "child", "child": "parent"}.get(
            str(locked_factor or ""))),
        "true_variable_component": true_variable_component,
        "prior_variable_component": prior_variable_component,
        "variable_component_word_token_limits": variable_limits,
    }


def preflight(
    *, semantic_audit_run: Path, qa_source_run: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path, root_id: str,
    prior_mcq_run: Path | None = None, prior_spell_run: Path | None = None,
) -> dict[str, Any]:
    gold_path = semantic_audit_run / "qa/admitted_gold.json"
    audit_path = semantic_audit_run / "reports/QA_SEMANTIC_AUDIT_SUMMARY.json"
    source_view_path = qa_source_run / "frozen/qa_source_view.json"
    catalog_path = qa_source_run / "frozen/span_catalog.json"
    checks: dict[str, bool] = {
        "run_root_absent": not run_root.exists(),
        "semantic_audit_run_exists": semantic_audit_run.is_dir(),
        "qa_source_run_exists": qa_source_run.is_dir(),
        "admitted_gold_exists": gold_path.is_file(),
        "semantic_audit_summary_exists": audit_path.is_file(),
        "source_view_exists": source_view_path.is_file(),
        "span_catalog_exists": catalog_path.is_file(),
        "api_config_exists": api_config_path.is_file(),
        "tokenizer_exists": tokenizer_path.exists(),
        "solver_guidance_paths_paired": (
            (prior_mcq_run is None) == (prior_spell_run is None)),
    }
    values = None
    prompt = ""
    prompt_tokens = 0
    if all(checks.values()):
        gold = _load(gold_path)
        audit = _load(audit_path)
        source_view = _load(source_view_path)
        catalog = _load(catalog_path)
        turns = source_view.get("combined_dialogue") or []
        ready = audit.get("readiness") or {}
        semantic_audit_admitted = bool(
            ready.get("semantic_entailment_independently_verified") or
            (ready.get("evidence_mismatch_marked") and
             ready.get("open_answer_gold_ready")))
        checks.update({
            "semantic_gold_or_marked_candidate_ready": semantic_audit_admitted,
            "open_answer_gold_ready": ready.get("open_answer_gold_ready") is True,
            "gold_has_no_options": "options" not in gold,
            "root_identity": str(gold.get("qa_id") or "").startswith(
                root_id.replace("-", "_")),
            "complete_dialogue_nonempty": bool(turns),
        })
        solver_guidance = None
        try:
            if prior_mcq_run is not None and prior_spell_run is not None:
                checks["prior_mcq_run_exists"] = prior_mcq_run.is_dir()
                checks["prior_spell_run_exists"] = prior_spell_run.is_dir()
                if checks["prior_mcq_run_exists"] and checks[
                        "prior_spell_run_exists"]:
                    solver_guidance = load_solver_guidance(
                        prior_mcq_run=prior_mcq_run,
                        prior_spell_run=prior_spell_run,
                        expected_qa_id=str(gold.get("qa_id") or ""),
                        expected_question=str(gold.get("question") or ""),
                        expected_answer=str(gold.get("correct_answer_text") or ""),
                    )
                    checks["solver_guidance_contract_valid"] = True
            if solver_guidance is None:
                prompt = build_shape_balanced_distractor_prompt(
                    frozen_dialogue=turns, span_catalog=catalog, base_qa=gold)
            else:
                prompt = build_solver_guided_full_distractor_prompt(
                    frozen_dialogue=turns, span_catalog=catalog, base_qa=gold,
                    solver_guidance=solver_guidance)
            checks["factorial_answer_decomposable"] = True
        except ValueError:
            prompt = ""
            checks["factorial_answer_decomposable"] = False
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        checks["distractor_prompt_within_context"] = (
            prompt_tokens + DISTRACTOR_MAX_TOKENS <= MODEL_CONTEXT_LIMIT)
        values = {"gold": gold, "audit": audit, "source_view": source_view,
                  "span_catalog": catalog, "solver_guidance": solver_guidance}
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "preflight_version": MATERIALIZER_VERSION,
        "valid": not failed,
        "failed_checks": failed,
        "checks": checks,
        "root_id": root_id,
        "semantic_audit_run": str(semantic_audit_run),
        "qa_source_run": str(qa_source_run),
        "semantic_audit_tree": hash_tree(semantic_audit_run)
            if semantic_audit_run.is_dir() else None,
        "qa_source_tree": hash_tree(qa_source_run)
            if qa_source_run.is_dir() else None,
        "prior_mcq_tree": hash_tree(prior_mcq_run)
            if prior_mcq_run is not None and prior_mcq_run.is_dir() else None,
        "prior_spell_tree": hash_tree(prior_spell_run)
            if prior_spell_run is not None and prior_spell_run.is_dir() else None,
        "gold_sha256": sha256_file(gold_path) if gold_path.is_file() else None,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")) if prompt else None,
        "prompt_tokens": prompt_tokens,
        "values": values,
        "api_calls": 0,
    }


def run(
    *, semantic_audit_run: Path, qa_source_run: Path, run_root: Path,
    api_config_path: Path, tokenizer_path: Path, root_id: str,
    candidate_index: int = 1,
    prior_mcq_run: Path | None = None, prior_spell_run: Path | None = None,
    preflight_only: bool = False, no_api: bool = False,
    maximum_attempts: int = 3,
    before_paid_request: Callable[[str, bool, int], None] | None = None,
) -> dict[str, Any]:
    if candidate_index < 1:
        raise ValueError("mcq_candidate_index_invalid")
    if maximum_attempts not in {1, 2, 3}:
        raise ValueError("distractor_maximum_attempts_invalid")
    frozen = preflight(
        semantic_audit_run=semantic_audit_run, qa_source_run=qa_source_run,
        run_root=run_root, api_config_path=api_config_path,
        tokenizer_path=tokenizer_path, root_id=root_id,
        prior_mcq_run=prior_mcq_run, prior_spell_run=prior_spell_run)
    public = {key: value for key, value in frozen.items()
              if key not in {"prompt", "values"}}
    if preflight_only or no_api:
        return public
    if not frozen["valid"]:
        raise RuntimeError(
            "PRE_API_PREFLIGHT_FAILED:" + ",".join(frozen["failed_checks"]))
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ("frozen", "qa", "raw/prompts", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    api = replace(load_challenger_config(api_config_path), timeout_seconds=600,
                  temperature=0.25, enable_thinking=False)
    freeze = {
        "freeze_version": MATERIALIZER_VERSION,
        "created_at": _now(),
        "root_id": root_id,
        "candidate_index": candidate_index,
        "run_root": str(run_root),
        "semantic_audit_run": str(semantic_audit_run),
        "qa_source_run": str(qa_source_run),
        "semantic_audit_tree_sha256": frozen["semantic_audit_tree"]["tree_sha256"],
        "qa_source_tree_sha256": frozen["qa_source_tree"]["tree_sha256"],
        "gold_sha256": frozen["gold_sha256"],
        "prompt_sha256": frozen["prompt_sha256"],
        "prompt_tokens": frozen["prompt_tokens"],
        "solver_guided_retry": prior_mcq_run is not None,
        "prior_mcq_run": str(prior_mcq_run) if prior_mcq_run else None,
        "prior_spell_run": str(prior_spell_run) if prior_spell_run else None,
        "prior_mcq_tree_sha256": ((frozen.get("prior_mcq_tree") or {}).get(
            "tree_sha256")),
        "prior_spell_tree_sha256": ((frozen.get("prior_spell_tree") or {}).get(
            "tree_sha256")),
        "model_configuration": {
            "model": api.model, "base_url": api.base_url,
            "temperature": api.temperature,
            "timeout_seconds": api.timeout_seconds,
            "enable_thinking": api.enable_thinking,
            "max_completion_tokens": DISTRACTOR_MAX_TOKENS,
            "api_key_recorded": False,
        },
        "request_budget": {"normal": 1,
                           "format_only_retry": maximum_attempts - 1,
                           "physical_maximum": maximum_attempts},
        "code_manifest": {
            "git_head": git_value(PROJECT_ROOT, "rev-parse", "HEAD"),
            "project_diff_sha256": project_diff_sha256(PROJECT_ROOT),
            "runner_sha256": sha256_file(Path(__file__)),
        },
        "preflight": public,
    }
    atomic_json(run_root / "run_freeze_manifest.json", freeze)
    atomic_json(run_root / "run_status.json", {
        "status": "initialized", "created_at": _now(), "api_calls": 0})
    atomic_json(run_root / "frozen/admitted_core_qa.json",
                frozen["values"]["gold"])
    atomic_json(run_root / "frozen/span_catalog.json",
                frozen["values"]["span_catalog"])
    atomic_text(run_root / "frozen/distractor_prompt.txt", frozen["prompt"])
    ledger = RequestLedger(
        run_root, sha256_file(run_root / "run_freeze_manifest.json"), "live")
    solver_guidance = frozen["values"].get("solver_guidance")
    distractor_validator = build_distractor_validator(
        core_qa=frozen["values"]["gold"],
        span_catalog=frozen["values"]["span_catalog"],
        placement=deterministic_answer_placement(
            f"{root_id}-round-1", candidate_index - 1),
        solver_guidance=solver_guidance)
    result = run_single_object_task(
        run_root=run_root, ledger=ledger,
        backend=LiveSemanticRepairBackend(
            OpenAICompatibleChallengerClient(api), DISTRACTOR_MAX_TOKENS),
        budget=ApiBudget(normal_caps={"distractor_writer": 1},
                         format_retry_cap=maximum_attempts - 1,
                         physical_cap=maximum_attempts),
        stage="distractor_writer",
        logical_request_id=(
            f"{root_id}-round1-distractors-{candidate_index:02d}" +
            ("-solver-guided" if prior_mcq_run is not None else "")),
        semantic_prompt=frozen["prompt"],
        lineage_view={
            "combined_dialogue": frozen["values"]["source_view"]["combined_dialogue"],
            "span_catalog": frozen["values"]["span_catalog"],
            "view_sha256": frozen["values"]["source_view"]["view_sha256"],
        },
        validator=distractor_validator,
        schema_retry_decisions=("distractor_schema_invalid",
                                "distractor_shape_invalid"),
        retry_prompt_suffix_factory=(
            (lambda raw, validation: _solver_guided_full_retry_suffix(
                raw, validation, solver_guidance=solver_guidance))
            if solver_guidance is not None
            else _shape_retry_suffix),
        max_attempts=maximum_attempts,
        ledger_budget_limit=maximum_attempts,
        ledger_total_budget_limit=maximum_attempts,
        before_paid_request=before_paid_request,
        json_invalid_decision="distractor_json_invalid")
    mcq = materialization = option_shape = None
    if result["accepted"]:
        materialization = materialize_single_item_mcq(
            core_qa=frozen["values"]["gold"],
            distractors=result["accepted_validation"]["distractors"],
            placement=deterministic_answer_placement(
                f"{root_id}-round-1", candidate_index - 1),
            span_catalog=frozen["values"]["span_catalog"])
        mcq = materialization["materialized_qa"]
        option_shape = diagnose_option_shape(
            options=mcq["options"],
            reference_answer_letter=mcq["reference_answer_letter"],
            question=mcq["question"],
        )
        atomic_json(run_root / "qa/final_mcq.json", mcq)
        atomic_json(run_root / "qa/materialization_record.json", materialization)
        atomic_json(run_root / "reports/option_shape_diagnostic.json", option_shape)
    completed = ledger.completed()
    usage = [row.get("response_metadata", {}).get("usage") or {}
             for row in completed]
    audit_after = hash_tree(semantic_audit_run)
    source_after = hash_tree(qa_source_run)
    prior_mcq_after = hash_tree(prior_mcq_run) if prior_mcq_run else None
    prior_spell_after = hash_tree(prior_spell_run) if prior_spell_run else None
    ready = mcq is not None
    evidence_alignment = evidence_alignment_audit(frozen["values"]["gold"])
    atomic_json(run_root / "reports/evidence_alignment_audit.json",
                evidence_alignment)
    summary = {
        "report_version": MATERIALIZER_VERSION,
        "status": "completed" if ready else "failed",
        "failure_stage": None if ready else "distractor_writer",
        "root_id": root_id,
        "candidate_index": candidate_index,
        "run_root": str(run_root),
        "distractor_result": result,
        "question": mcq.get("question") if mcq else None,
        "options": mcq.get("options") if mcq else None,
        "reference_answer_letter": mcq.get("reference_answer_letter") if mcq else None,
        "option_shape": option_shape,
        "evidence_alignment": evidence_alignment,
        "api_statistics": {
            "physical_requests": len(completed),
            "prompt_tokens": sum(int(x.get("prompt_tokens") or 0) for x in usage),
            "completion_tokens": sum(int(x.get("completion_tokens") or 0) for x in usage),
            "total_tokens": sum(int(x.get("total_tokens") or 0) for x in usage),
        },
        "integrity": {
            "semantic_audit_run_unchanged": audit_after["tree_sha256"] ==
                frozen["semantic_audit_tree"]["tree_sha256"],
            "qa_source_run_unchanged": source_after["tree_sha256"] ==
                frozen["qa_source_tree"]["tree_sha256"],
            "prior_mcq_run_unchanged": (
                prior_mcq_after is None or prior_mcq_after["tree_sha256"] ==
                frozen["prior_mcq_tree"]["tree_sha256"]),
            "prior_spell_run_unchanged": (
                prior_spell_after is None or prior_spell_after["tree_sha256"] ==
                frozen["prior_spell_tree"]["tree_sha256"]),
        },
        "solver_guidance": frozen["values"].get("solver_guidance"),
        "readiness": {"mcq_ready": ready, "qa_pipeline_ready": False,
                      "local_spell_filter_pending": ready,
                      "training_ready": False},
    }
    atomic_json(run_root / "reports/MCQ_MATERIALIZER_SUMMARY.json", summary)
    # Compatibility alias consumed by the existing local SPELL adapter.  Its
    # ``mcq_ready`` flag means only that options are materialized; the adapter
    # still owns the actual grounding/admission decision.
    atomic_json(run_root / "reports/MCQ_CANARY_SUMMARY.json", summary)
    atomic_text(run_root / "reports/MCQ_MATERIALIZER_SUMMARY.md", "\n".join([
        "# Single-root MCQ Materializer", "",
        f"- Root: `{root_id}`", f"- Status: `{summary['status']}`",
        f"- Question: {summary.get('question') or ''}",
        f"- Option-shape risk: `{bool((option_shape or {}).get('answer_shape_leakage_risk'))}`",
        f"- Evidence alignment: `{evidence_alignment['evidence_alignment_status']}`",
        "- Local SPELL filter: `pending`", "",
    ]))
    atomic_json(run_root / "run_status.json", {
        "status": summary["status"], "completed_at": _now(),
        "api_calls": len(completed), "mcq_ready": ready})
    return summary


__all__ = ["DISTRACTOR_PROMPT_CONTRACT_VERSION", "MATERIALIZER_VERSION",
           "build_shape_balanced_distractor_prompt", "preflight", "run",
           "evidence_alignment_audit", "validate_distractors_with_shape_gate",
           "decompose_cross_boundary_answer", "compile_factorial_distractors",
           "validate_factorial_components_with_shape_gate",
           "solver_guided_retry_eligible", "load_solver_guidance",
           "derive_component_selection_feedback",
           "build_solver_guided_full_distractor_prompt",
           "validate_solver_guided_distractors_with_shape_gate",
           "build_distractor_validator",
           "compile_solver_guided_variable_components"]
