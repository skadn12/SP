"""Offline QA-level composition probe.

This package consumes frozen, materialized atomic QA only.  It never calls a
Writer, Solver, distractor service, or Memorax.
"""

from .contracts import AtomicQA, COMPOSITION_TYPES
from .input_adapter import build_atomic_ir, load_atomic_rows, load_evaluations
from .pair_compiler import compile_pairs
from .composer import build_composer_request
from .static_audit import audit_composite, is_mechanical_conjunction

__all__ = [
    "AtomicQA",
    "COMPOSITION_TYPES",
    "build_atomic_ir",
    "load_atomic_rows",
    "load_evaluations",
    "compile_pairs",
    "build_composer_request",
    "audit_composite",
    "is_mechanical_conjunction",
]
