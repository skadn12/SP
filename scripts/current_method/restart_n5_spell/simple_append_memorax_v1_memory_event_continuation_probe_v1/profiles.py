"""Explicit expansion profiles.

The legacy profile is intentionally equivalent to the historical three-scene
route.  The current expanded route is ``EXPANDED_SINGLE_PLAN_PROFILE``;
``EXPANDED_PROFILE`` remains a separately selectable historical capability
example so the former two-plan behavior stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ExpansionExecutionProfile:
    name: str
    event_bucket_only: bool = False
    memory_only_fast: bool = False
    realizer_input_mode: str = "full_parent"
    max_continuations: int = 3
    max_new_development_candidates: int = 4
    max_plan_variants: int = 1
    new_development_slots: int | None = None
    total_scene_limit: int = 3
    realizer_batch_size: int = 3
    # ``fixed_size`` reproduces the historical scheduler.  ``continuation_first``
    # keeps a contiguous continuation run in one Realizer request, then chunks
    # independent/new-development scenes using ``realizer_batch_size``.
    realizer_batch_policy: str = "fixed_size"
    sequential_batch_promotion: bool = False
    planner_max_tokens: int = 4096
    bounded_continuation_limit: int = 1
    replacement_limit: int = 1
    require_full_scene_quota: bool = True
    # Optional role-view integration.  ``None`` preserves the historical
    # builder path; the versioned R03 profile opts in explicitly.
    feedback_view_version: str | None = None
    development_policy: str | None = None
    gate_max_tokens: int = 2048
    outline_max_tokens: int = 4096
    planner_input_mode: str = "route_default"

    def __post_init__(self) -> None:
        if self.development_policy not in {None, "development_gain_v2"}:
            raise ValueError("unsupported_development_policy")
        if self.planner_input_mode not in {"route_default", "full_parent"}:
            raise ValueError("unsupported_planner_input_mode")
        if self.development_policy and (not self.memory_only_fast or
                self.realizer_input_mode != "text_free" or
                self.require_full_scene_quota or self.max_plan_variants != 1 or
                self.realizer_batch_size < self.total_scene_limit or
                self.sequential_batch_promotion or
                self.planner_input_mode != "full_parent"):
            raise ValueError("development_gain_requires_memory_text_free_partial_single_plan")
        if self.gate_max_tokens < 1 or self.outline_max_tokens < 1:
            raise ValueError("invalid_review_token_limit")
        if self.realizer_input_mode not in {"full_parent", "text_free"}:
            raise ValueError("invalid_realizer_input_mode")
        if self.realizer_batch_policy not in {"fixed_size", "continuation_first"}:
            raise ValueError("invalid_realizer_batch_policy")
        for field in ("max_continuations", "max_new_development_candidates",
                      "max_plan_variants",
                      "total_scene_limit", "realizer_batch_size",
                      "planner_max_tokens", "bounded_continuation_limit",
                      "replacement_limit"):
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field}_must_be_nonnegative")
        if int(self.total_scene_limit) < 1:
            raise ValueError("total_scene_limit_must_be_positive")
        if int(self.realizer_batch_size) < 1:
            raise ValueError("realizer_batch_size_must_be_positive")
        if int(self.max_plan_variants) < 1:
            raise ValueError("max_plan_variants_must_be_positive")
        if self.new_development_slots is not None and int(self.new_development_slots) < 0:
            raise ValueError("new_development_slots_must_be_nonnegative")
        if self.feedback_view_version not in {
            None, "h4_round1_r03_feedback_view_v2",
        }:
            raise ValueError("unsupported_feedback_view_version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def worst_case_requests(self, *, include_planner_repair: bool = True,
                            include_outline_audit: bool = True) -> dict[str, int]:
        """Derive bounded request counts without contacting any service."""
        if self.development_policy:
            include_planner_repair = False  # No automatic filling of a deficit.
        scenes = int(self.total_scene_limit)
        batches = (scenes + int(self.realizer_batch_size) - 1) // int(self.realizer_batch_size)
        return {
            "event_verifier": 0 if self.memory_only_fast else 1,
            "planner": 1,
            "planner_repair": 1 if include_planner_repair else 0,
            "semantic_gate": 2 if include_planner_repair else 1,
            "outline_audit": 1 if include_outline_audit else 0,
            "realizer_normal_batches": batches,
            "batch_adherence_audits": batches if self.sequential_batch_promotion else 0,
            "adherence": 1,
            "bounded_continuation": int(self.bounded_continuation_limit),
            "adherence_reaudit": int(self.bounded_continuation_limit),
            "replacement": int(self.replacement_limit),
            "replacement_adherence": int(self.replacement_limit),
            "total_model_requests": (
                (0 if self.memory_only_fast else 1) +
                1 + (1 if include_planner_repair else 0) +
                (2 if include_planner_repair else 1) +
                (1 if include_outline_audit else 0) + batches +
                (batches if self.sequential_batch_promotion else 0) + 1 +
                int(self.bounded_continuation_limit) * 2 +
                int(self.replacement_limit) * 2),
        }


LEGACY_PROFILE = ExpansionExecutionProfile(
    name="legacy_v1",
    # R01/R02 frozen routes use event buckets as the main candidate source;
    # person buckets remain auxiliary context in the prompt.
    event_bucket_only=True,
)

# Historical offline capability example.  This identity intentionally keeps
# the former two-plan behavior reproducible; it is not the current route.
EXPANDED_PROFILE = ExpansionExecutionProfile(
    name="expanded_multi_plan_v1",
    event_bucket_only=True,
    memory_only_fast=True,
    realizer_input_mode="text_free",
    max_continuations=3,
    max_new_development_candidates=8,
    max_plan_variants=2,
    # Six is the upper bound. With three continuations only the remaining
    # three slots can be used; with zero continuations all six independent
    # developments can be selected by the same explicit profile.
    new_development_slots=6,
    total_scene_limit=6,
    realizer_batch_size=2,
    sequential_batch_promotion=True,
    planner_max_tokens=8192,
    bounded_continuation_limit=1,
    replacement_limit=1,
    require_full_scene_quota=False,
    feedback_view_version="h4_round1_r03_feedback_view_v2",
)


# Current expanded route: one Planner response contains one complete plan with
# a larger candidate pool.  The old EXPANDED_PROFILE remains available solely
# for reproducing the pre-change two-plan experiment.
EXPANDED_SINGLE_PLAN_PROFILE = ExpansionExecutionProfile(
    name="expanded_single_plan_v1",
    event_bucket_only=True,
    memory_only_fast=True,
    realizer_input_mode="text_free",
    max_continuations=3,
    max_new_development_candidates=8,
    max_plan_variants=1,
    new_development_slots=6,
    total_scene_limit=6,
    realizer_batch_size=2,
    realizer_batch_policy="continuation_first",
    sequential_batch_promotion=True,
    planner_max_tokens=8192,
    bounded_continuation_limit=1,
    replacement_limit=1,
    require_full_scene_quota=False,
    feedback_view_version="h4_round1_r03_feedback_view_v2",
)

# Continuation pair starting from frozen R1-only Memory. Both arms preserve the
# historical R02 text-free Realizer boundary; full Parent text was R01-only.
# The feedback arm differs only by its prompt appendix.
R1_ALIGNED_BASELINE_PROFILE = ExpansionExecutionProfile(
    name="r1_aligned_baseline_v1", event_bucket_only=True,
    memory_only_fast=True, realizer_input_mode="text_free",
    max_continuations=3, max_new_development_candidates=4,
    max_plan_variants=1, total_scene_limit=3, realizer_batch_size=3,
    realizer_batch_policy="fixed_size", sequential_batch_promotion=False,
    planner_max_tokens=4096, bounded_continuation_limit=1,
    replacement_limit=1, require_full_scene_quota=True,
    feedback_view_version=None,
)

R1_ALIGNED_FEEDBACK_PROFILE = ExpansionExecutionProfile(
    name="r1_aligned_feedback_v1", event_bucket_only=True,
    memory_only_fast=True, realizer_input_mode="text_free",
    max_continuations=3, max_new_development_candidates=4,
    max_plan_variants=1, total_scene_limit=3, realizer_batch_size=3,
    realizer_batch_policy="fixed_size", sequential_batch_promotion=False,
    planner_max_tokens=4096, bounded_continuation_limit=1,
    replacement_limit=1, require_full_scene_quota=True,
    feedback_view_version="h4_round1_r03_feedback_view_v2",
)

# Opt-in quality repair on the same three-scene route. No extra model stage;
# the existing outline request also reviews expected information/dependencies.
DEVELOPMENT_GAIN_PROFILE = ExpansionExecutionProfile(
    name="development_gain_v2", event_bucket_only=True,
    memory_only_fast=True, realizer_input_mode="text_free",
    max_continuations=3, max_new_development_candidates=4,
    max_plan_variants=1, total_scene_limit=3, realizer_batch_size=3,
    realizer_batch_policy="fixed_size", sequential_batch_promotion=False,
    planner_max_tokens=4096, bounded_continuation_limit=1,
    replacement_limit=0, require_full_scene_quota=False,
    feedback_view_version="h4_round1_r03_feedback_view_v2", development_policy="development_gain_v2",
    gate_max_tokens=8192, outline_max_tokens=8192,
    planner_input_mode="full_parent",
)


__all__ = [
    "ExpansionExecutionProfile", "LEGACY_PROFILE", "EXPANDED_PROFILE",
    "EXPANDED_SINGLE_PLAN_PROFILE", "R1_ALIGNED_BASELINE_PROFILE",
    "R1_ALIGNED_FEEDBACK_PROFILE",
    "DEVELOPMENT_GAIN_PROFILE",
]
