"""Low-cost, tail-only dialogue expansion prototype.

This package is intentionally independent from ``world_memory_slim_v1``.  It
reuses only the dataset-neutral unified-root validator and keeps Memory,
placement, dependency-DAG, Binder, and transaction semantics out of the
generation path.
"""

from .assembly import (
    append_expansion_round,
    assemble_expanded_dialogue,
    expanded_dialogue_markdown,
    validate_expanded_dialogue,
    validate_appended_round,
)
from .context import build_tail_context, build_tail_context_from_expanded_dialogue
from .contracts import (
    compile_round_plan,
    normalize_round_realizer_output,
    shadow_project_flat_realizer_v1,
    validate_planner_output,
)
from .prompts import (
    build_mcq_batch_prompt,
    build_planner_prompt,
    build_round_realizer_prompt,
)
from .qa import (
    assigned_answer_letters,
    build_evidence_alias_registry,
    classify_balanced_no_context,
    classify_full_context_frontier,
    evaluate_focus_round_profile,
    evaluate_qa_quota_status,
    materialize_solver_candidate,
    normalize_focus_writer_batch,
    materialize_solver_batch,
    partition_mcq_candidates,
    resolve_evidence_turn_refs,
    validate_mcq_batch,
)

__all__ = [
    "assemble_expanded_dialogue",
    "append_expansion_round",
    "assigned_answer_letters",
    "build_evidence_alias_registry",
    "build_mcq_batch_prompt",
    "build_planner_prompt",
    "build_round_realizer_prompt",
    "build_tail_context",
    "build_tail_context_from_expanded_dialogue",
    "compile_round_plan",
    "classify_balanced_no_context",
    "classify_full_context_frontier",
    "evaluate_focus_round_profile",
    "evaluate_qa_quota_status",
    "expanded_dialogue_markdown",
    "normalize_round_realizer_output",
    "materialize_solver_batch",
    "materialize_solver_candidate",
    "normalize_focus_writer_batch",
    "partition_mcq_candidates",
    "resolve_evidence_turn_refs",
    "shadow_project_flat_realizer_v1",
    "validate_expanded_dialogue",
    "validate_appended_round",
    "validate_mcq_batch",
    "validate_planner_output",
]
