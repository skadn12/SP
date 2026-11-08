"""Explicit prompt adapters for the next R03 debug branch.

Every function is a pure renderer.  The imported builders are existing active
builders; this module supplies the new versioned view without changing their
default historical behavior.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..h4_round1_r03_qa_output_contract_v1_1_revision_1.prompts import (
    multi_pass_prompt, targeted_prompt,
)
from ..simple_append_memorax_v1_memory_event_continuation_probe_v1.prompts import (
    build_planner_prompt, build_realizer_prompt,
)
from .compiler import (
    expansion_view_v3, qa_composition_view, qa_fresh_view_v3, qa_targeted_view,
)
from .templates import (
    EXPANSION_PLANNER_TEMPLATE_V4, QA_TEMPLATE_V3, SHARED_FEEDBACK_RULES_V2,
    render,
)


VERSIONED_RENDERER_CONFIG = {
    "version": "h4_round1_r03_feedback_view_v2",
    "branch_input": "Parent+R01+R02-feedback",
    "old_r03_input_allowed": False,
    "planner": "render_expansion_planner",
    "realizer": "render_expansion_realizer",
    "qa_targeted": "render_qa_targeted",
    "qa_composition": "render_qa_composition",
    "qa_fresh": "render_qa_fresh",
    "targeted_generation_scheduled": False,
    "remote_memory_operations": False,
    "template_version": "h4.r03.feedback_view.v4",
    "event_bucket_only": True,
    "realizer_modes": ["full_parent", "text_free", "bounded_continuation"],
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def normalize_parent(dialogue: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the frozen flat dialogue to the active expansion builder shape."""
    return {
        "root_id": str(dialogue["root_id"]),
        "dialogue": {"turns": [{
            "turn_id": str(row.get("turn_ref") or row.get("turn_id") or ""),
            "speaker": str(row.get("speaker") or ""),
            "text": str(row.get("text") or ""),
        } for row in dialogue.get("turns") or []]},
    }


def dialogue_text(dialogue: Mapping[str, Any]) -> str:
    return "\n".join(
        f"[{row.get('turn_ref')}] {row.get('speaker')}: {row.get('text')}"
        for row in dialogue.get("turns") or []
    )


def render_expansion_planner(
    *, record: Mapping[str, Any], dialogue: Mapping[str, Any],
    memory_overview: Mapping[str, Any] | None = None,
    memory_candidates: Sequence[Mapping[str, Any]] = (),
    verified_tracks: Sequence[Mapping[str, Any]] = (),
    memory_only_fast: bool = True,
    speaker_registry: Mapping[str, Any] | None = None,
    strict_speaker_refs: bool = True,
    expose_complete_memory: bool = True,
    role_contract: str = "Use the existing Planner JSON output contract.",
    planner_max_continuations: int = 3,
    planner_max_new_candidates: int = 4,
    planner_scene_quota: int = 3,
    planner_max_plan_variants: int = 1,
    planner_max_tokens: int = 4096,
    planner_output_contract_text: str | None = None,
) -> dict[str, Any]:
    parent = normalize_parent(dialogue)
    view = expansion_view_v3(record)
    suffix = render(EXPANSION_PLANNER_TEMPLATE_V4, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": _json(view),
        # build_planner_prompt owns the single authoritative transcript copy.
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": (
            "The authoritative cumulative transcript is included above by the "
            "active Planner builder; do not duplicate it here."),
        "EXISTING_EXPANSION_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": role_contract,
    })
    base = build_planner_prompt(
        parent=parent, memory_overview=dict(memory_overview or {}),
        candidates=list(memory_candidates), verified_tracks=list(verified_tracks),
        memory_only_fast=memory_only_fast,
        event_bucket_only=True, memory_source_mode="event_bucket_only",
        speaker_registry=speaker_registry, strict_speaker_refs=strict_speaker_refs,
        expose_complete_memory=expose_complete_memory,
        include_authoritative_transcript=True,
        max_continuations=planner_max_continuations,
        max_new_development_candidates=planner_max_new_candidates,
        max_plan_variants=planner_max_plan_variants,
        scene_quota=planner_scene_quota,
        output_contract_text=planner_output_contract_text,
        feedback_guidance=suffix,
    )
    return {
        "renderer": "active_build_planner_prompt+shared_feedback_view_v2",
        "prompt_version": "h4.r03.expansion.planner.shared_feedback.v4",
        "root_id": record["root_id"], "prompt": base["prompt"],
        "feedback_view": view,
        "max_plan_variants": int(planner_max_plan_variants),
        "max_continuations": int(planner_max_continuations),
        "planner_max_tokens": int(planner_max_tokens),
    }


def render_expansion_realizer(
    *, record: Mapping[str, Any], dialogue: Mapping[str, Any], plan: Mapping[str, Any],
    style_profile: Mapping[str, Any] | None = None,
    continuity_shape: Mapping[str, Any] | None = None,
    promoted_prefix: Sequence[Mapping[str, Any]] | None = None,
    include_full_parent: bool = False,
    input_mode: str | None = None,
    request_purpose: str = "normal_generation",
    role_contract: str = "Use the existing Realizer JSON output contract.",
) -> dict[str, Any]:
    mode = input_mode or ("full_parent" if include_full_parent else "text_free")
    if mode not in {"full_parent", "text_free"}:
        raise ValueError(f"unsupported_realizer_input_mode:{mode}")
    include_full_parent = mode == "full_parent"
    parent = normalize_parent(dialogue)
    base = build_realizer_prompt(
        parent=parent, plan=plan, style_profile=style_profile,
        continuity_shape=continuity_shape,
        promoted_prefix=promoted_prefix,
        include_full_parent=include_full_parent,
        request_purpose=request_purpose,
    )
    # Keep the historical Realizer builder as the sole owner of the role
    # contract, scene schema, turn budget, and transcript boundary.  The
    # feedback arm is a thin append-only adapter: it adds observations and
    # selection guidance after the frozen prompt, without adding another task
    # contract or repeating the transcript.  This makes the no-feedback call
    # exactly the historical ``build_realizer_prompt`` call.
    view = expansion_view_v3(record)
    suffix = "\n\n".join([
        SHARED_FEEDBACK_RULES_V2,
        "【当前角色可见的共享反馈】\n" + _json(view),
        "【反馈边界】\n"
        "反馈只用于选择和组织本场已经冻结的 Planner 场景，不能改变场景数量、"
        "输出 JSON 合同、required participants、人物绑定、turn 数、事实来源或"
        "历史。不要回答历史 QA，不要输出 QA、Gold、Solver 诊断或证据索引。"
        "如果反馈与冻结场景或权威历史不一致，以冻结场景和权威历史为准。",
    ])
    return {
        "renderer": "historical_build_realizer_prompt+feedback_appendix_v1",
        "prompt_version": f"h4.r03.expansion.realizer.feedback_appendix.v1.{mode}",
        "root_id": record["root_id"], "prompt": base["prompt"] + "\n\n" + suffix,
        "feedback_view": view,
        "input_mode": mode,
        "full_parent_included": bool(base.get("full_parent_included")),
    }


def render_qa_targeted(
    *, record: Mapping[str, Any], dialogue: Mapping[str, Any],
    schema: Mapping[str, Any], strategy_schema: Mapping[str, Any],
    role_contract: str = "Use the frozen revision-1 targeted output contract.",
) -> dict[str, Any]:
    view = qa_targeted_view(record)
    template = render(QA_TEMPLATE_V3, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": _json(view),
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": dialogue_text(dialogue),
        "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": role_contract,
    })
    prompt = targeted_prompt(view, dialogue, template=template, schema=schema,
                             strategy_schema=strategy_schema)
    return {"renderer": "frozen_revision_1_targeted_prompt+shared_view_v2",
            "prompt_version": "h4.r03.qa.targeted.shared_feedback.v3",
            "root_id": record["root_id"], "prompt": prompt, "feedback_view": view}


def render_qa_composition(
    *, records: Sequence[Mapping[str, Any]], dialogue: Mapping[str, Any],
    schema: Mapping[str, Any], role_contract: str = "Use the frozen revision-1 composition output contract.",
) -> dict[str, Any]:
    view = qa_composition_view(records)
    sample = {"mode": "composition", "offline_fixture": True,
              "shared_feedback_view": view}
    template = render(QA_TEMPLATE_V3, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": _json(view),
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": dialogue_text(dialogue),
        "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": role_contract,
    })
    prompt = multi_pass_prompt("composition", sample, dialogue, template=template, schema=schema)
    return {"renderer": "frozen_revision_1_multi_pass_prompt(composition)+shared_view_v2",
            "prompt_version": "h4.r03.qa.composition.shared_feedback.v3",
            "root_id": dialogue["root_id"], "prompt": prompt, "feedback_view": view}


def render_qa_fresh(
    *, record: Mapping[str, Any], dialogue: Mapping[str, Any],
    negative_signatures: Sequence[Mapping[str, Any]], schema: Mapping[str, Any],
    role_contract: str = "Use the frozen revision-1 fresh output contract.",
) -> dict[str, Any]:
    view = qa_fresh_view_v3(record, negative_signatures)
    sample = {"mode": "fresh", "offline_fixture": True,
              "shared_feedback_view": view}
    template = render(QA_TEMPLATE_V3, {
        "ROLE_SPECIFIC_COMMON_FEEDBACK_VIEW": _json(view),
        "AUTHORITATIVE_CUMULATIVE_DIALOGUE": dialogue_text(dialogue),
        "EXISTING_QA_ROLE_INSTRUCTIONS_AND_OUTPUT_CONTRACT": role_contract,
    })
    prompt = multi_pass_prompt("fresh", sample, dialogue, template=template, schema=schema)
    return {"renderer": "frozen_revision_1_multi_pass_prompt(fresh)+shared_view_v2",
            "prompt_version": "h4.r03.qa.fresh.shared_feedback.v3",
            "root_id": record["root_id"], "prompt": prompt, "feedback_view": view}
