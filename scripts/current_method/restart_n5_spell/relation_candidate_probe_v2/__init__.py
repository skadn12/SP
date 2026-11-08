"""Offline proposal-to-outcome relation candidate probe."""

from .contracts import Fact, RelationCandidate, RELATION_TYPE
from .input_adapter import load_transcript, load_structured_sources
from .fact_extractor import extract_facts
from .relation_miner import mine_relations
from .validator import validate_relation
from .paired_qa_planner import build_paired_qa_target

__all__ = [
    "Fact",
    "RelationCandidate",
    "RELATION_TYPE",
    "load_transcript",
    "load_structured_sources",
    "extract_facts",
    "mine_relations",
    "validate_relation",
    "build_paired_qa_target",
]
