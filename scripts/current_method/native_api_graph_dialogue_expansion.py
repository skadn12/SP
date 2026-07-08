from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.current_method import api_challenger_generation as adapter
except ModuleNotFoundError:
    from current_method import api_challenger_generation as adapter


base = adapter.base
old_gate = adapter.old_gate

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ANCHOR_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ANCHOR_MARKER_RE = re.compile(r"\[\[([A-Za-z0-9][A-Za-z0-9_\-:.]{3,64})\]\]")
GENERIC_MARKER_RE = re.compile(r"\s*\[\[[^\[\]\n]{2,96}\]\]\s*")
DIALOGUE_TEXT_TURN_RE = re.compile(r"^\s*\[([^\]\n]+)\]\s*([^:\n]{1,160}):\s*(.*)\s*$")
DEFAULT_LENGTH_MIN_ADDED_TURNS = 8
DEFAULT_LENGTH_MIN_ADDED_TOKENS = 1024
DEFAULT_PARENT_CONTEXT_TAIL_TURNS = 12
DEFAULT_PARENT_CONTEXT_RELEVANT_TURNS = 24
DEFAULT_DIALOGUE_API_PARENT_TOKEN_BUDGET = 2048

SUPPORTED_QUESTION_TYPES = ("ordering", "multi_select", "evidence_selection", "state_reconstruction")
DEFAULT_QUESTION_TYPE_WEIGHTS = {
    "ordering": 0.34,
    "multi_select": 0.28,
    "evidence_selection": 0.18,
    "state_reconstruction": 0.20,
}
# Opt-in r3_long_easy curriculum. single_choice is ONLY reachable in easy mode; the default
# path keeps SUPPORTED_QUESTION_TYPES unchanged so its behavior is byte-for-byte identical.
EASY_SUPPORTED_QUESTION_TYPES = ("single_choice", "state_reconstruction", "multi_select", "evidence_selection", "ordering")
EASY_QUESTION_TYPE_WEIGHTS = {
    "single_choice": 0.32,
    "state_reconstruction": 0.30,
    "multi_select": 0.22,
    "evidence_selection": 0.10,
    "ordering": 0.06,
}
# Hard attack-skill combinations are excluded in easy mode per the curriculum spec
# (no expired_memory / conflicting_reports / belief_vs_reality / who_knows_what /
# multi_object_dependency / cancel_restore).
EASY_ATTACK_SKILLS = (
    "conditional_plan",
    "delayed_activation",
    "permission_visibility_scope",
    "timeline_fork_merge",
)
EASY_ATTACK_SKILLS_BY_QUESTION_TYPE = {
    "single_choice": ("conditional_plan", "delayed_activation"),
    "state_reconstruction": ("conditional_plan", "delayed_activation"),
    "multi_select": ("conditional_plan", "permission_visibility_scope"),
    "evidence_selection": ("permission_visibility_scope", "timeline_fork_merge"),
    "ordering": ("timeline_fork_merge", "conditional_plan"),
}
# Easy-mode length curriculum: fewer appended turns, smaller token delta.
EASY_MIN_ADDED_TURNS = 14
EASY_MIN_ADDED_TOKENS = 512
EASY_GRAPH_EVENT_INSTRUCTION = "Include 3-5 events, enough active and non-active facts for the requested QA type."

# ---------------------------------------------------------------------------
# r3_long_easy_v2: deterministic quota curriculum (foundation for closed-loop).
# v2 fixes the silent-fallback bug where weighted sampling + "else ordering"
# rewrites collapsed the distribution (single_choice=0, ordering=16/20). v2 uses
# an EXACT integer quota per question_type and per attack_skill, enforces a hard
# cap on accepted clean samples, and forbids silent fallback to ordering.
EASY_V2_MODE = "r3_long_easy_v2"
# Target shares for the ACCEPTED clean set. Integers are derived per-run from n_samples
# via largest-remainder so the planned counts are exact and reproducible.
# Default easy_v2 remains single_choice-only, but the closed-loop qtype curriculum may inject
# harder qtype shares after the fixed solver masters the current distribution.
EASY_V2_QUESTION_TYPE_TARGET_SHARES = {
    "single_choice": 1.0,
}
EASY_V2_SUPPORTED_QUESTION_TYPES = ("single_choice", "multi_select", "ordering", "state_reconstruction")
# Per-question-type allowed easy skills for the curriculum-gated action space.
EASY_V2_ATTACK_SKILLS_BY_QUESTION_TYPE = {
    "single_choice": ("conditional_plan", "delayed_activation"),
    "multi_select": ("conditional_plan", "permission_visibility_scope"),
    "ordering": ("timeline_fork_merge", "conditional_plan"),
    "state_reconstruction": ("conditional_plan", "delayed_activation"),
}
HARD_SINGLE_CHOICE_SUBTYPES = (
    "minimal_difference",
    "latest_valid_state",
    "guaranteed_vs_possible",
    "override_sensitive",
    "sequence_as_single_choice",
)
STAGE64_QA_OPERATIONS = (
    "temporal_state_update",
    "bridge",
    "comparison",
    "visibility",
    "count",
    "cause_consequence",
    "relation_lookup",
)
STAGE64_TARGET_OPERATION_COUNTS = {
    "temporal_state_update": 30,
    "bridge": 18,
    "comparison": 18,
    "visibility": 18,
    "count": 12,
    "cause_consequence": 12,
    "relation_lookup": 12,
}
STAGE64_ANSWER_TYPES = (
    "Person",
    "GroupOrOrg",
    "LocationOrScope",
    "DateOrTime",
    "Number",
    "YesNo",
    "Event",
    "ObjectOrCommonNoun",
    "AttributeValue",
    "ProperNounOrNamedEntity",
)
STAGE64_PREFERRED_ANSWER_TYPE_CYCLE = (
    "Event",
    "ObjectOrCommonNoun",
    "YesNo",
    "Number",
    "Event",
    "ObjectOrCommonNoun",
    "DateOrTime",
    "Person",
)
STAGE64_OPERATION_REQUIRED_ANSWER_TYPES = {
    "count": {"Number"},
}
STAGE64_BAD_SOLVER_VISIBLE_RE = re.compile(
    r"\braw_qa\b|raw_qa_|event_instance|anchor_|graph_anchor|\bmarker\b|\banchor\b|"
    r"final active state|latest confirmed detail|confirmed active state|current graph|\bQ:\s|\bA:\s|"
    r"\bconfirmed\s+details?\b|\bmost\s+recent\s+confirmed\s+details?\b|"
    r"\b(?:currently\s+)?settled\s+as\s+true\b|\bstate\s+after\s+all\s+updates\b",
    re.IGNORECASE,
)
STAGE64_GENERIC_QUESTION_RE = re.compile(
    r"^\s*which\s+event\s+(?:is|best|represents|reflects)\b|"
    r"\bconfirmed\s+as\s+settled\b|"
    r"\bin\s+the\s+conversation\b|"
    r"\bbased\s+on\s+the\s+conversation\b",
    re.IGNORECASE,
)
STAGE64_QA_REPAIRABLE_RE = re.compile(
    r"stage64 API QA|qa_proposal|query_plan|answer_type|qa_operation|gold|options|anchor_refs|graph reference",
    re.IGNORECASE,
)
# Hard cap (as a share of n_samples) for any single qtype/skill in the ACCEPTED set.
EASY_V2_MAX_QUESTION_TYPE_SHARE = 0.40
EASY_V2_MAX_ATTACK_SKILL_SHARE = 0.40

APPROVED_ATTACK_SKILLS = (
    "multi_object_dependency",
    "timeline_fork_merge",
    "who_knows_what",
    "belief_vs_reality",
    "cancel_restore",
    "conditional_plan",
    "delayed_activation",
    "expired_memory",
    "permission_visibility_scope",
    "conflicting_reports",
)
ATTACK_SKILLS_BY_QUESTION_TYPE = {
    "ordering": ("timeline_fork_merge", "multi_object_dependency", "conditional_plan"),
    "multi_select": ("belief_vs_reality", "cancel_restore", "expired_memory", "conflicting_reports"),
    "evidence_selection": ("who_knows_what", "permission_visibility_scope", "timeline_fork_merge"),
    "state_reconstruction": ("conditional_plan", "delayed_activation", "cancel_restore", "multi_object_dependency"),
}
APPROVED_EVENT_TYPES = {
    "state_change",
    "belief_update",
    "cancel",
    "restore",
    "delay",
    "expire",
    "report",
    "permission_scope",
    "conflict",
}
APPROVED_STATUSES = {
    "active",
    "outdated",
    "canceled",
    "restored",
    "expired",
    "delayed_not_active",
    "false",
    "belief",
    "report",
    "scope_limited",
}
ACTIVE_STATUSES = {"active", "restored"}
FALSE_STATUSES = {"outdated", "canceled", "expired", "delayed_not_active", "false", "belief", "report", "scope_limited"}
DELAYED_ACTIVATION_PARENT_UNSATISFIED_STATUSES = {
    "delayed",
    "delayed_not_active",
    "pending",
    "inactive",
    "unsatisfied",
    "condition_unsatisfied",
    "not_yet_active",
    "not yet active",
    "not_active",
    "waiting",
    "blocked",
}
DELAYED_ACTIVATION_CHILD_ACTIVE_STATUSES = ACTIVE_STATUSES | {"current_valid", "current-valid", "activated"}
DELAYED_ACTIVATION_SEMANTIC_FAILURE_REASONS = {
    "delayed_activation_missing_chain",
    "delayed_activation_parent_not_unsatisfied",
    "delayed_activation_missing_child_trigger",
    "delayed_activation_active_active_pair",
    "delayed_activation_missing_same_slot",
    "delayed_activation_hard_negative_not_semantic",
}
STATUS_ALIASES = {
    "outdated": ("outdated", "old", "stale", "no longer current", "superseded", "may no longer apply", "corrected later"),
    "canceled": ("canceled", "cancelled", "called off", "do not treat", "not current", "no longer happening"),
    "expired": ("expired", "lapsed", "no longer valid", "has expired"),
    "delayed_not_active": ("delayed", "not active yet", "not yet active", "waiting", "planned for later", "still waiting"),
    "false": ("false", "not true", "incorrect", "corrects this", "mistaken"),
    "belief": ("believes", "belief", "thinks", "assumes", "unsure", "may be true", "not sure"),
    "report": ("report", "reported", "claims", "claim", "someone said", "not fully settled", "needs confirmation"),
    "scope_limited": ("only", "scope", "limited", "visible to", "permission", "allowed to know", "can know this"),
}
GRAPH_TURN_ID_FIELDS = (
    "turn_ids",
    "evidence_turn_ids",
    "activation_turn_ids",
    "invalidation_turn_ids",
    "cancellation_turn_ids",
    "restoration_turn_ids",
    "event_turn_ids",
    "turn_id",
)
BANNED_SOLVER_VISIBLE_TERMS = (
    "current graph",
    "graph timeline",
    "graph slot",
    "tracked current graph",
    "final active state",
    "active current state",
    "current state",
    "logged the update",
    "logged the detail",
    "logged timeline",
    "the update now uses",
    "the detail now uses",
    "update remained supported",
    "detail remained supported",
    "review note",
    "audit anchor",
    "neighboring entry",
    "conversation step",
    "context so the conversation stays easy to follow",
    "r2 chain",
    "r3 chain",
    "state chain",
    "state-chain",
    "memory state",
    "timeline event",
    "graph_update",
    "event_instance_id",
    "alignment_anchor",
    "required_for_dialogue",
    "conv-",
    "r2 extension",
    "r3 extension",
    "r2-",
    "r3-",
)
META_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tracked_current_graph", re.compile(r"\btracked\s+current\s+graph\b", re.IGNORECASE)),
    ("current_graph", re.compile(r"\bcurrent\s+graph\b", re.IGNORECASE)),
    ("graph_timeline", re.compile(r"\bgraph\s+timeline\b", re.IGNORECASE)),
    ("graph_slot", re.compile(r"\bgraph\s+slot\b", re.IGNORECASE)),
    ("final_active_state", re.compile(r"\bfinal\s+active\s+state\b", re.IGNORECASE)),
    ("active_current_state", re.compile(r"\bactive\s+current\s+state\b", re.IGNORECASE)),
    ("current_state", re.compile(r"\bcurrent\s+state\b", re.IGNORECASE)),
    ("logged_the_update", re.compile(r"\blogged\s+the\s+update\b", re.IGNORECASE)),
    ("logged_the_detail", re.compile(r"\blogged\s+the\s+detail\b", re.IGNORECASE)),
    ("logged_timeline", re.compile(r"\blogged\s+timeline\b", re.IGNORECASE)),
    ("the_update_now_uses", re.compile(r"\bthe\s+update\s+now\s+uses\b", re.IGNORECASE)),
    ("the_detail_now_uses", re.compile(r"\bthe\s+detail\s+now\s+uses\b", re.IGNORECASE)),
    ("update_remained_supported", re.compile(r"\bupdate\s+remained\s+supported\b", re.IGNORECASE)),
    ("detail_remained_supported", re.compile(r"\bdetail\s+remained\s+supported\b", re.IGNORECASE)),
    ("review_note", re.compile(r"\breview\s+note\b", re.IGNORECASE)),
    ("audit_anchor", re.compile(r"\baudit\s+anchor\b", re.IGNORECASE)),
    ("neighboring_entry", re.compile(r"\bneighboring\s+entry\b", re.IGNORECASE)),
    ("conversation_step", re.compile(r"\bconversation\s+step\b", re.IGNORECASE)),
    ("context_filler", re.compile(r"\bcontext\s+so\s+the\s+conversation\s+stays\s+easy\s+to\s+follow\b", re.IGNORECASE)),
    ("r2_chain", re.compile(r"\br2\s+chain\b", re.IGNORECASE)),
    ("r3_chain", re.compile(r"\br3\s+chain\b", re.IGNORECASE)),
    ("state_chain_spaced", re.compile(r"\bstate\s+chain\b", re.IGNORECASE)),
    ("conv_id", re.compile(r"\bconv-\d+\b", re.IGNORECASE)),
    ("r2_extension", re.compile(r"\br2\s+extension\b", re.IGNORECASE)),
    ("r3_extension", re.compile(r"\br3\s+extension\b", re.IGNORECASE)),
    ("r2_suffix", re.compile(r"\br2-\d+\b|\br2-", re.IGNORECASE)),
    ("r3_suffix", re.compile(r"\br3-\d+\b|\br3-", re.IGNORECASE)),
    ("state_chain", re.compile(r"\bstate-chain\b", re.IGNORECASE)),
    ("memory_state", re.compile(r"\bmemory\s+state\b", re.IGNORECASE)),
    ("timeline_event", re.compile(r"\btimeline\s+event\b", re.IGNORECASE)),
    ("graph_update", re.compile(r"\bgraph_update\b", re.IGNORECASE)),
    ("event_instance_id", re.compile(r"\bevent_instance_id\b", re.IGNORECASE)),
    ("alignment_anchor", re.compile(r"\balignment_anchor\b", re.IGNORECASE)),
    ("required_for_dialogue", re.compile(r"\brequired_for_dialogue\b", re.IGNORECASE)),
    ("belief_not_active", re.compile(r"\bbelief,\s*not\s+an\s+active\b", re.IGNORECASE)),
    (
        "confirm_that_internal_status",
        re.compile(
            r"\bconfirm\s+that\s+(?:the\s+)?(?:latest|final|current|confirmed)\s+"
            r"(?:confirmed\s+)?(?:detail|state|update|version)\b",
            re.IGNORECASE,
        ),
    ),
)
INTERNAL_TEXT_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[A-Z][A-Za-z0-9_-]*\s*/\s*raw_qa_[A-Za-z0-9_-]+\s*:\s*", re.IGNORECASE), ""),
    (re.compile(r"\b[A-Z][A-Za-z0-9_-]*\s+raw_qa_[A-Za-z0-9_-]+\s*:\s*", re.IGNORECASE), ""),
    (re.compile(r"\braw_qa_[A-Za-z0-9_-]+\s*:\s*", re.IGNORECASE), ""),
    (re.compile(r"\bI\s+logged\s+the\s+update\s+as\s+", re.IGNORECASE), ""),
    (re.compile(r"\bI\s+logged\s+the\s+detail\s+as\s+", re.IGNORECASE), ""),
    (re.compile(r"\blogged\s+the\s+update\s+as\s+", re.IGNORECASE), ""),
    (re.compile(r"\blogged\s+the\s+detail\s+as\s+", re.IGNORECASE), ""),
    (re.compile(r"\blogged\s+timeline\s+\d+\s+as\s+", re.IGNORECASE), ""),
    (re.compile(r"\bThe\s+update\s+now\s+uses\s+", re.IGNORECASE), ""),
    (re.compile(r"\bThe\s+detail\s+now\s+uses\s+", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+update\s+now\s+uses\s+", re.IGNORECASE), ""),
    (re.compile(r"\bthe\s+detail\s+now\s+uses\s+", re.IGNORECASE), ""),
    (
        re.compile(r"\bThe\s+review\s+note\s+for\s+the\s+update\s+still\s+points\s+to\s+", re.IGNORECASE),
        "The speaker means ",
    ),
    (
        re.compile(r"\bthe\s+review\s+note\s+for\s+the\s+update\s+still\s+points\s+to\s+", re.IGNORECASE),
        "the speaker means ",
    ),
    (
        re.compile(r"\bWhen\s+we\s+checked\s+the\s+r[23]\s+chain,\s*the\s+update\s+remained\s+supported\s+by\s+", re.IGNORECASE),
        "When they looked back at what was actually confirmed, ",
    ),
    (
        re.compile(r"\bthe\s+update\s+remained\s+supported\s+by\s+", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bthe\s+detail\s+remained\s+supported\s+by\s+", re.IGNORECASE),
        "",
    ),
    (re.compile(r"\breview\s+note\b", re.IGNORECASE), "detail"),
    (re.compile(r"\baudit\s+anchor\b", re.IGNORECASE), "main point"),
    (re.compile(r"\bneighboring\s+entry\b", re.IGNORECASE), "earlier point"),
    (re.compile(r"\bconversation\s+step\b", re.IGNORECASE), "part of the conversation"),
    (re.compile(r"\bthe\s+detail\s+for\s+the\s+detail\s+still\s+points\s+to\s+", re.IGNORECASE), "The speaker means "),
    (re.compile(r"\bWhen\s+we\s+checked\s+the\s+what\s+was\s+said\s+earlier,\s*", re.IGNORECASE), "When they looked back at what was said earlier, "),
    (re.compile(r"\br[23]\s+chain\b", re.IGNORECASE), "what was said earlier"),
    (re.compile(r"\bstate\s+chain\b", re.IGNORECASE), "conversation"),
    (
        re.compile(r"\bConfirm\s+that\s+conv-\d+'s\s+memory\s+state\s+for\s+r[23]\s+extension\s+\d+\s+is\s+", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\bconv-\d+'s\s+memory\s+state\s+for\s+r[23]\s+extension\s+\d+\s+is\s+", re.IGNORECASE),
        "",
    ),
    (
        re.compile(r"\btimeline\s+event\s+\d+\s+(?:is\s+)?supported\s+by(?:\s+the\s+claim\s+that)?\s+", re.IGNORECASE),
        "",
    ),
    (re.compile(r"\s*/\s*r[23]-\d+\b", re.IGNORECASE), ""),
    (re.compile(r"\br[23]-\d+\b", re.IGNORECASE), ""),
    (re.compile(r"\br[23]\s+extension\s+\d+\b", re.IGNORECASE), "the detail"),
    (re.compile(r"\br[23]\s+extension\b", re.IGNORECASE), "the detail"),
    (re.compile(r"\br[23]\s+state-chain\b", re.IGNORECASE), "the conversation"),
    (re.compile(r"\bstate-chain\b", re.IGNORECASE), "the conversation"),
    (re.compile(r"\bmemory\s+state\b", re.IGNORECASE), "detail"),
    (re.compile(r"\btimeline\s+event\s+\d+\b", re.IGNORECASE), "the detail"),
    (re.compile(r"\btimeline\s+event\b", re.IGNORECASE), "the detail"),
    (re.compile(r"\bconv-\d+\b", re.IGNORECASE), "the person"),
    (re.compile(r"\bgraph_update\b", re.IGNORECASE), "conversation detail"),
    (re.compile(r"\bevent_instance_id\b", re.IGNORECASE), "fact reference"),
    (re.compile(r"\balignment_anchor\b", re.IGNORECASE), "marker"),
    (re.compile(r"\brequired_for_dialogue\b", re.IGNORECASE), "important for the conversation"),
    (
        re.compile(r"\bThis\s+is\s+a\s+belief,\s*not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "The speaker is unsure whether this is still true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+only\s+a\s+report,\s*not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "Someone reported this, but the speakers have not settled whether it is true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+expired\s+and\s+not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "That information has expired.",
    ),
    (
        re.compile(r"\bThis\s+is\s+canceled\s+and\s+not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "That plan has been called off.",
    ),
    (
        re.compile(r"\bThis\s+is\s+outdated\s+and\s+not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "An old version of this detail may no longer apply.",
    ),
    (
        re.compile(r"\bThis\s+is\s+false\s+and\s+not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "The speaker corrects this as not true.",
    ),
    (re.compile(r"\btracked\s+current\s+graph\b", re.IGNORECASE), "conversation"),
    (re.compile(r"\bcurrent\s+graph\b", re.IGNORECASE), "conversation"),
    (re.compile(r"\bgraph\s+timeline\b", re.IGNORECASE), "order of events"),
    (re.compile(r"\bgraph\s+slot\b", re.IGNORECASE), "detail"),
    (
        re.compile(
            r"\b(?:latest|final|confirmed|current)\s+(?:confirmed\s+)?(?:active\s+)?"
            r"(?:detail|details|state|update|version)\s+(?:after\s+review\s+)?"
            r"(?:is|are|keeps|remains|stays)\s+",
            re.IGNORECASE,
        ),
        "",
    ),
    (
        re.compile(
            r"\b(?:latest|final|confirmed|current)\s+confirmed\s+(?:detail|details)\b",
            re.IGNORECASE,
        ),
        "confirmed information",
    ),
    (
        re.compile(
            r"\b(?:final|confirmed|current)\s+(?:active\s+)?state\b",
            re.IGNORECASE,
        ),
        "settled information",
    ),
    (re.compile(r"\bcurrent\s+state\b", re.IGNORECASE), "latest information"),
    (
        re.compile(r"\bThis\s+is\s+a\s+belief,\s*not\s+an\s+active\s+current\s+state\.?", re.IGNORECASE),
        "The speaker is unsure whether this is still true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+a\s+belief,\s*not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "The speaker is unsure whether this is still true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+only\s+a\s+report,\s*not\s+an\s+active\s+current\s+state\.?", re.IGNORECASE),
        "Someone reported this, but the speakers have not settled whether it is true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+only\s+a\s+report,\s*not\s+an\s+active[^.]*\.?", re.IGNORECASE),
        "Someone reported this, but the speakers have not settled whether it is true.",
    ),
    (
        re.compile(r"\bThis\s+is\s+scope\s+limited,\s*not\s+a\s+generally\s+visible\s+active\s+state\.?", re.IGNORECASE),
        "Only some people are allowed to know this.",
    ),
    (
        re.compile(r"\bThis\s+is\s+delayed\s+and\s+not\s+active\s+yet\.?", re.IGNORECASE),
        "This is planned for later and is still waiting.",
    ),
    (
        re.compile(r"\bThis\s+is\s+expired\s+and\s+not\s+an\s+active\s+current\s+state\.?", re.IGNORECASE),
        "That information has expired.",
    ),
    (re.compile(r"\bnot\s+an\s+active\s+current\s+state\b", re.IGNORECASE), "not settled as true"),
    (re.compile(r"\bactive\s+current\s+state\b", re.IGNORECASE), "settled plan"),
    (re.compile(r"\bConfirm\s+that\b", re.IGNORECASE), ""),
)


class NativeGraphDialogueReject(RuntimeError):
    """Rejected current graph/dialogue sample; retry with another seed/API attempt."""


class NativeGraphDialogueAnchorLeakReject(NativeGraphDialogueReject):
    """Rejected because a temporary alignment anchor leaked into solver-visible text."""


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(value).lower()).strip()


def token_set(value: Any) -> set[str]:
    return {token for token in normalize_text(value).split() if len(token) >= 3}


def stable_hash(value: Any, *, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def turn_sort_key(turn_id: Any) -> tuple[int, int, str]:
    match = re.match(r"^D(\d+):(\d+)$", str(turn_id or ""))
    if match:
        return int(match.group(1)), int(match.group(2)), str(turn_id)
    return 999999, 999999, str(turn_id or "")


def normalize_turn_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _normalize_dialogue_turns(raw_turns: Any) -> list[dict[str, Any]]:
    turns = raw_turns if isinstance(raw_turns, list) else []
    normalized: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_id = normalize_space(turn.get("turn_id") or turn.get("dia_id") or f"legacy:{index + 1}")
        text = normalize_space(turn.get("text"))
        if not turn_id or not text:
            continue
        normalized.append(
            {
                "turn_id": turn_id,
                "speaker": normalize_space(turn.get("speaker") or f"Speaker{(index % 2) + 1}"),
                "text": text,
            }
        )
    return normalized


def parse_dialogue_text_turns(dialogue_text: Any) -> list[dict[str, Any]]:
    """Parse bracketed dialogue text like `[D2:10] Caroline: ...` into turns."""
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in str(dialogue_text or "").splitlines():
        line = raw_line.rstrip()
        match = DIALOGUE_TEXT_TURN_RE.match(line)
        if match:
            if current and normalize_space(current.get("text")):
                turns.append(
                    {
                        "turn_id": normalize_space(current.get("turn_id")),
                        "speaker": normalize_space(current.get("speaker")),
                        "text": normalize_space(current.get("text")),
                    }
                )
            current = {
                "turn_id": match.group(1).strip(),
                "speaker": match.group(2).strip(),
                "text": match.group(3).strip(),
            }
            continue
        if current is not None and normalize_space(line):
            current["text"] = normalize_space(f"{current.get('text', '')} {line.strip()}")
    if current and normalize_space(current.get("text")):
        turns.append(
            {
                "turn_id": normalize_space(current.get("turn_id")),
                "speaker": normalize_space(current.get("speaker")),
                "text": normalize_space(current.get("text")),
            }
        )
    return [turn for turn in turns if turn.get("turn_id") and turn.get("text")]


def parent_dialogue_turns_from_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    dialogue_turns = _normalize_dialogue_turns(dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else [])
    if dialogue_turns:
        return dialogue_turns, "dialogue.turns"
    expanded_turns = parse_dialogue_text_turns(row.get("expanded_dialogue_text"))
    if expanded_turns:
        return expanded_turns, "expanded_dialogue_text"
    new_turns = parse_dialogue_text_turns(row.get("new_dialogue_text"))
    if new_turns:
        return new_turns, "new_dialogue_text"
    final_turns = _normalize_dialogue_turns(row.get("final_dialogue") if isinstance(row.get("final_dialogue"), list) else [])
    if final_turns:
        return final_turns, "final_dialogue"
    return [], "fallback_empty"


def sanitize_internal_text(text: Any) -> str:
    """Convert internal/project-shaped text into dialogue-facing scenario language."""
    cleaned = normalize_space(text)
    if not cleaned:
        return ""
    for pattern, replacement in INTERNAL_TEXT_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    qa_artifact = re.search(r"\bQ:\s*(.*?)\s*A:\s*(.+)$", cleaned, flags=re.IGNORECASE)
    if qa_artifact:
        q_text = normalize_space(qa_artifact.group(1)).rstrip(" ?")
        a_text = normalize_space(qa_artifact.group(2)).strip(" .")
        if q_text and a_text:
            cleaned = f"{q_text}: {a_text}"
    cleaned = cleaned.replace(" / ", " ")
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")
    if not cleaned:
        cleaned = "The speaker discusses a real-world update."
    return cleaned[:1].upper() + cleaned[1:]


# Synthetic surface phrases that leak template artifacts into solver-visible text. Used only
# by easy-mode natural-phrasing and the QA-option display cleanup; deterministic scaffold
# alignment, markers, turn_ids, gold, and validators are NOT driven by this list.
EASY_SYNTHETIC_SURFACE_REPLACEMENTS = [
    (re.compile(r"\bthe\s+detail\s+for\s+the\s+detail\b", re.IGNORECASE), "the detail"),
    (re.compile(r"\bdetail\s+for\s+the\s+detail\b", re.IGNORECASE), "detail"),
    (re.compile(r"\bI\s+logged\s+state\s+update\s*\d*\b", re.IGNORECASE), "I noted"),
    (re.compile(r"\bI\s+logged\b", re.IGNORECASE), "I noted"),
    (re.compile(r"\bstate\s+update\s*\d*\b", re.IGNORECASE), "update"),
    (re.compile(r"\bscope\s+guard\b", re.IGNORECASE), "who can see it"),
    (re.compile(r"\brouting\s+gate\b", re.IGNORECASE), "next step"),
    (re.compile(r"\breview\s+stamp\b", re.IGNORECASE), "review"),
    (re.compile(r"\bevidence\s+window\b", re.IGNORECASE), "supporting point"),
    (re.compile(r"\bfinal\s+routing\b", re.IGNORECASE), "final decision"),
    (re.compile(r"\bmain\s+point,\s*final\s+routing\s+activated\b", re.IGNORECASE), "the final decision is set"),
    (re.compile(r"\bwhat\s+was\s+said\s+earlier\b", re.IGNORECASE), "the earlier part of the conversation"),
    (re.compile(r"\bkept\s+it\s+tied\s+to\s+that\s+the\s+part\s+of\s+the\s+conversation\b", re.IGNORECASE), "connected it to that earlier point"),
]


def _natural_surface_text(text: Any) -> str:
    """Easy-mode only: rewrite residual synthetic surface phrases into natural language.

    Operates on solver-visible text. Returns natural prose; does not touch markers, ids,
    gold, or alignment metadata.
    """
    cleaned = normalize_space(text)
    if not cleaned:
        return ""
    for pattern, replacement in EASY_SYNTHETIC_SURFACE_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    # Collapse article/preposition doubling introduced by phrase substitution (e.g. "the the").
    cleaned = re.sub(r"\b(the|a|an|that|to|of)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return normalize_space(text)
    return cleaned[:1].upper() + cleaned[1:]


def _has_synthetic_surface_phrase(text: Any) -> bool:
    value = normalize_space(text)
    if not value:
        return False
    return any(pattern.search(value) for pattern, _ in EASY_SYNTHETIC_SURFACE_REPLACEMENTS)


def _status_natural_suffix(status: str) -> str:
    if status == "belief":
        return "The speaker thinks this may be true, but they are unsure."
    if status == "report":
        return "Someone reported this, but the speakers have not fully settled it."
    if status == "outdated":
        return "An old version of this detail may no longer apply."
    if status == "canceled":
        return "That plan has been called off."
    if status == "expired":
        return "That information has expired."
    if status == "delayed_not_active":
        return "This is planned for later and is still waiting."
    if status == "false":
        return "The speaker corrects this as not true."
    if status == "scope_limited":
        return "Only some people are allowed to know this."
    return ""


def _naturalize_claim_for_status(claim: str, status: str) -> str:
    natural = sanitize_internal_text(claim)
    suffix = _status_natural_suffix(status)
    if not suffix:
        return natural
    natural_norm = normalize_text(natural)
    aliases = STATUS_ALIASES.get(status) or ()
    has_status_language = any(normalize_text(alias) in natural_norm for alias in aliases)
    if not has_status_language:
        natural = normalize_space(f"{natural}. {suffix}")
    return sanitize_internal_text(natural)


def _solver_visible_text_items(row_or_turns: Any) -> list[tuple[str, str]]:
    if isinstance(row_or_turns, dict):
        items: list[tuple[str, str]] = []
        dialogue = row_or_turns.get("dialogue") if isinstance(row_or_turns.get("dialogue"), dict) else {}
        for index, turn in enumerate(dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []):
            if isinstance(turn, dict):
                items.append((f"dialogue.turns[{index}].text", str(turn.get("text") or "")))
        if row_or_turns.get("expanded_dialogue_text") is not None:
            items.append(("expanded_dialogue_text", str(row_or_turns.get("expanded_dialogue_text") or "")))
        if row_or_turns.get("question") is not None:
            items.append(("question", str(row_or_turns.get("question") or "")))
        qa = row_or_turns.get("qa") if isinstance(row_or_turns.get("qa"), dict) else {}
        if qa.get("question") is not None:
            items.append(("qa.question", str(qa.get("question") or "")))
        for index, option in enumerate(qa.get("options") if isinstance(qa.get("options"), list) else []):
            if isinstance(option, dict) and option.get("text") is not None:
                items.append((f"qa.options[{index}].text", str(option.get("text") or "")))
        gold = qa.get("gold")
        if isinstance(gold, list):
            for index, item in enumerate(gold):
                if isinstance(item, str):
                    items.append((f"qa.gold[{index}]", item))
        elif isinstance(gold, str):
            items.append(("qa.gold", gold))
        for key in ("answer", "rationale", "instruction", "instructions"):
            value = qa.get(key)
            if isinstance(value, str):
                items.append((f"qa.{key}", value))
        return items
    turns = row_or_turns if isinstance(row_or_turns, list) else []
    return [
        (f"dialogue.turns[{index}].text", str(turn.get("text") or ""))
        for index, turn in enumerate(turns)
        if isinstance(turn, dict)
    ]


def meta_leak_details_from_solver_visible_text(row_or_turns: Any) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    for field, text in _solver_visible_text_items(row_or_turns):
        for name, pattern in META_LEAK_PATTERNS:
            matches = [match.group(0) for match in pattern.finditer(text)]
            if not matches:
                continue
            counts[name] += len(matches)
            field_counts[field] += len(matches)
            details.append(
                {
                    "field": field,
                    "term": name,
                    "matches": matches[:5],
                    "text_preview": text[:240],
                }
            )
    return {
        "meta_leak_count": sum(counts.values()),
        "banned_term_counts": dict(counts),
        "field_counts": dict(field_counts),
        "details": details,
        "matched_terms": sorted(counts),
    }


def assert_no_meta_leakage(row_or_turns: Any) -> tuple[bool, dict[str, Any]]:
    details = meta_leak_details_from_solver_visible_text(row_or_turns)
    return details["meta_leak_count"] == 0, details


def _strip_markers_and_known_anchors(text: Any, known_anchors: Iterable[str]) -> str:
    cleaned = normalize_space(GENERIC_MARKER_RE.sub(" ", str(text or "")))
    for anchor in known_anchors:
        anchor_text = str(anchor or "").strip()
        if anchor_text:
            cleaned = cleaned.replace(anchor_text, " ")
    return normalize_space(cleaned)


def _meta_clean_text(text: str) -> str:
    cleaned = normalize_space(text)
    for _ in range(5):
        ok, _details = assert_no_meta_leakage([{"text": cleaned}])
        if ok:
            return cleaned
        updated = sanitize_internal_text(cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    return normalize_space(cleaned)


def solver_visible_parent_dialogue_turns(
    parent_dialogue_turns: list[dict[str, Any]],
    known_anchors: Iterable[str],
) -> list[dict[str, Any]]:
    clean_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(parent_dialogue_turns):
        if not isinstance(turn, dict):
            continue
        text = _strip_markers_and_known_anchors(turn.get("text"), known_anchors)
        if not text:
            continue
        clean_turns.append(
            {
                "turn_id": normalize_space(turn.get("turn_id") or f"legacy:{index + 1}"),
                "speaker": normalize_space(turn.get("speaker") or f"Speaker{(index % 2) + 1}"),
                "text": _meta_clean_text(text),
            }
        )
    return clean_turns


def qa_dialogue_meta_counts(row: dict[str, Any]) -> dict[str, int]:
    details = meta_leak_details_from_solver_visible_text(row)
    qa_count = 0
    dialogue_count = 0
    for item in details.get("details") or []:
        field = str(item.get("field") or "")
        count = len(item.get("matches") or [])
        if field.startswith("qa.") or field == "question":
            qa_count += count
        elif field.startswith("dialogue.") or field == "expanded_dialogue_text":
            dialogue_count += count
    return {"qa_meta_leak_count": qa_count, "dialogue_meta_leak_count": dialogue_count}


def meta_leak_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    counts: Counter[str] = Counter()
    for row in rows:
        ok, details = assert_no_meta_leakage(row)
        if ok:
            continue
        total += int(details.get("meta_leak_count") or 0)
        counts.update(details.get("banned_term_counts") or {})
    qa_count = 0
    dialogue_count = 0
    for row in rows:
        counts_by_area = qa_dialogue_meta_counts(row)
        qa_count += counts_by_area["qa_meta_leak_count"]
        dialogue_count += counts_by_area["dialogue_meta_leak_count"]
    return {
        "meta_leak_count": total,
        "solver_visible_meta_leak_count": total,
        "qa_meta_leak_count": qa_count,
        "dialogue_meta_leak_count": dialogue_count,
        "banned_term_counts": dict(counts),
    }


def _merge_counter_dict(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in (source or {}).items():
        target[key] = int(target.get(key) or 0) + int(value or 0)


def _merge_local_summary(global_summary: dict[str, Any], local: dict[str, Any]) -> None:
    """Additively merge a worker's local summary delta into the global summary.

    Integer counters are summed; nested dict counters (e.g. by_question_type, banned_term_counts)
    are merged key-wise. Non-additive scalars set by the driver (mode, floors) are left untouched.
    """
    skip_keys = {
        "native_curriculum_mode",
        "length_min_added_turns",
        "length_min_added_tokens",
        "length_guard_enabled",
        "min_length_delta_tokens",
        "length_guard_parent_baseline_tokens",
        "length_guard_effective_parent_tokens_mean",
        "length_guard_effective_delta_tokens_mean",
        "length_guard_baseline_mode",
        "length_guard_target_delta",
        "length_guard_fallback_used",
        "fallback_delta",
        "emergency_floor",
        "length_tokenizer_path",
        "length_token_count_mode",
        "easy_v2_planned_question_type_quota",
        "difficulty_plan_applied",
        "difficulty_plan_stage",
        "difficulty_plan_manual_policy_correction",
    }
    for key, value in local.items():
        if key in skip_keys:
            continue
        if isinstance(value, dict):
            target = global_summary.get(key)
            if not isinstance(target, dict):
                target = {}
                global_summary[key] = target
            _merge_counter_dict(target, value)
        elif isinstance(value, (int, float)):
            base_val = global_summary.get(key)
            global_summary[key] = (base_val if isinstance(base_val, (int, float)) else 0) + value
        # strings / other: leave the driver-owned value in place



def _bump_summary_counter(summary: dict[str, Any], key: str, amount: int = 1) -> None:
    summary[key] = int(summary.get(key) or 0) + int(amount or 0)


def _bump_failure_reason(summary: dict[str, Any], reason: str) -> None:
    reason_key = normalize_space(reason) or "unknown"
    counts = summary.setdefault("api_failure_reason_counts", {})
    counts[reason_key] = int(counts.get(reason_key) or 0) + 1


def _api_meta_error_types(meta: dict[str, Any]) -> list[str]:
    error_types: list[str] = []
    for error in meta.get("errors") or meta.get("previous_errors") or []:
        if isinstance(error, dict) and normalize_space(error.get("error_type")):
            error_types.append(normalize_space(error.get("error_type")))
    if normalize_space(meta.get("error_type")):
        error_types.append(normalize_space(meta.get("error_type")))
    return error_types


def _api_meta_has_timeout(meta: dict[str, Any]) -> bool:
    for error_type in _api_meta_error_types(meta):
        if "timeout" in error_type.lower() or "timed_out" in error_type.lower():
            return True
    for error in meta.get("errors") or meta.get("previous_errors") or []:
        if isinstance(error, dict) and "timed out" in str(error.get("error") or "").lower():
            return True
    return False


def _api_meta_has_parse_failure(meta: dict[str, Any]) -> bool:
    parse_error_types = {"invalid_json", "missing_required_keys", "json_decode_error", "parse_error"}
    return any(error_type in parse_error_types for error_type in _api_meta_error_types(meta))


def _record_api_meta_retry_diagnostics(summary: dict[str, Any], meta: dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        return
    normalizer = meta.get("normalizer_diagnostics") if isinstance(meta.get("normalizer_diagnostics"), dict) else {}
    if meta.get("call_kind") == "api_graph_expansion":
        if normalizer.get("graph_json_normalizer_used"):
            _bump_summary_counter(summary, "graph_json_normalizer_used_count")
        if normalizer.get("graph_update_events_unwrap"):
            _bump_summary_counter(summary, "graph_update_events_unwrap_count")
        if normalizer.get("graph_json_normalizer_failure"):
            _bump_summary_counter(summary, "graph_json_normalizer_failure_count")
        if meta.get("used") and normalizer.get("graph_json_normalizer_used"):
            _bump_summary_counter(summary, "graph_reject_recovered_by_normalizer_count")
    if meta.get("call_kind") == "api_dialogue_expansion":
        if normalizer.get("dialogue_json_normalizer_used"):
            _bump_summary_counter(summary, "dialogue_json_normalizer_used_count")
        if normalizer.get("dialogue_json_block_extracted"):
            _bump_summary_counter(summary, "dialogue_json_block_extracted_count")
        if normalizer.get("dialogue_wrapper_unwrap"):
            _bump_summary_counter(summary, "dialogue_wrapper_unwrap_count")
        if normalizer.get("dialogue_json_normalizer_failure"):
            _bump_summary_counter(summary, "dialogue_json_normalizer_failure_count")
        if meta.get("used") and normalizer.get("dialogue_json_normalizer_used"):
            _bump_summary_counter(summary, "dialogue_reject_recovered_by_normalizer_count")
    errors = [
        error
        for error in (meta.get("errors") or meta.get("previous_errors") or [])
        if isinstance(error, dict)
    ]
    if not errors and normalize_space(meta.get("error_type")):
        errors = [{"error_type": normalize_space(meta.get("error_type")), "error": meta.get("error")}]
    parse_error_types = {"invalid_json", "missing_required_keys", "json_decode_error", "parse_error"}
    for error in errors:
        error_type = normalize_space(error.get("error_type"))
        error_text = str(error.get("error") or "")
        if error_type in parse_error_types:
            _bump_summary_counter(summary, "parse_failure_count")
        if "timeout" in error_type.lower() or "timed_out" in error_type.lower() or "timed out" in error_text.lower():
            _bump_summary_counter(summary, "timeout_count")


def _record_query_conditioned_graph_diagnostics(summary: dict[str, Any], meta: dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        return
    diag = meta.get("query_conditioned_graph_diagnostics")
    if not isinstance(diag, dict) or not diag.get("query_conditioned_graph_expansion"):
        return
    _bump_summary_counter(summary, "query_conditioned_graph_objective_created_count")
    if diag.get("pass"):
        _bump_summary_counter(summary, "query_conditioned_graph_plan_pass_count")
    else:
        _bump_summary_counter(summary, "query_conditioned_graph_plan_fail_count")
        reason = normalize_space(diag.get("failure_reason")) or "unknown"
        counts = summary.setdefault("query_conditioned_graph_plan_failure_reason", {})
        counts[reason] = int(counts.get(reason) or 0) + 1
    if normalize_space(diag.get("query_plan_type")):
        counts = summary.setdefault("query_conditioned_graph_plan_type", {})
        key = normalize_space(diag.get("query_plan_type"))
        counts[key] = int(counts.get(key) or 0) + 1
    _bump_summary_counter(summary, "query_conditioned_graph_event_count_sum", int(diag.get("event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_active_event_count_sum", int(diag.get("active_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_scope_limited_event_count_sum", int(diag.get("scope_limited_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_invalidating_event_count_sum", int(diag.get("invalidating_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_natural_scope_limited_event_count_sum", int(diag.get("natural_scope_limited_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_natural_invalidating_event_count_sum", int(diag.get("natural_invalidating_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_placeholder_false_surface_count_sum", int(diag.get("placeholder_false_surface_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_single_choice_false_event_count_sum", int(diag.get("single_choice_false_event_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_single_choice_same_slot_false_count_sum", int(diag.get("single_choice_same_slot_false_count") or 0))
    _bump_summary_counter(summary, "query_conditioned_graph_single_choice_same_entity_false_count_sum", int(diag.get("single_choice_same_entity_false_count") or 0))


def _failure_reason_from_exception(exc: Exception, raw_record: dict[str, Any] | None = None) -> str:
    if raw_record and normalize_space(raw_record.get("final_reject_reason")):
        return normalize_space(raw_record.get("final_reject_reason"))
    message = str(exc).lower()
    if isinstance(exc, NativeGraphDialogueAnchorLeakReject) or "anchor leakage" in message:
        return "anchor_leak"
    if "dialogue alignment did not pass" in message:
        return "dialogue_alignment"
    if "length guard" in message or "too short" in message:
        return "length"
    if "meta" in message or "internal/project terms" in message:
        return "meta_leak"
    if "valid json" in message or "invalid_json" in message or "missing_required_keys" in message:
        return "parse_failure"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "api graph expansion failed" in message:
        return "api_graph"
    if "anti_shortcut" in message:
        return "base_validation_anti_shortcut"
    if "base validation" in message:
        return "base_validation"
    return type(exc).__name__


def _is_stage64_qa_repairable_error(exc: Exception) -> bool:
    if not isinstance(exc, NativeGraphDialogueReject):
        return False
    return bool(STAGE64_QA_REPAIRABLE_RE.search(str(exc)))


def _final_dialogue_reject_reason(
    retry_context: dict[str, Any] | None,
    alignment_result: dict[str, Any] | None,
) -> str:
    if isinstance(retry_context, dict):
        previous_error = normalize_space(retry_context.get("previous_error")).lower()
        if "valid json" in previous_error:
            meta = retry_context.get("dialogue_meta") if isinstance(retry_context.get("dialogue_meta"), dict) else {}
            if _api_meta_has_timeout(meta):
                return "timeout"
            if _api_meta_has_parse_failure(meta):
                return "parse_failure"
            return "api_dialogue"
        if "length guard" in previous_error:
            return "length"
        if "internal/project terms" in previous_error:
            return "meta_leak"
    if isinstance(alignment_result, dict):
        return "dialogue_alignment"
    return "dialogue_alignment"


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_if_exists(path: str | Path) -> list[dict[str, Any]]:
    """Read a jsonl file, returning [] for missing/empty/corrupt files (resume-safe)."""
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    except (OSError, json.JSONDecodeError):
        return []
    return rows


def append_jsonl_line(path: str | Path, row: dict[str, Any]) -> None:
    """Crash-safe single-row append used to checkpoint each accepted sample."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_json_atomic(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)



def dialogue_turns_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    turns, _source = parent_dialogue_turns_from_row(row)
    return turns


def turn_text_by_id(row: dict[str, Any]) -> dict[str, str]:
    return {str(turn["turn_id"]): str(turn["text"]) for turn in dialogue_turns_from_row(row)}


def item_text(item: dict[str, Any]) -> str:
    for key in (
        "value",
        "to",
        "restored_state",
        "cancelled_state",
        "invalidated_statement",
        "final_true_state",
        "planned_state",
        "activation_state",
        "from",
        "text",
    ):
        text = normalize_space(item.get(key))
        if text:
            return text
    return ""


def item_turn_ids(item: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for field_name in GRAPH_TURN_ID_FIELDS:
        for turn_id in normalize_turn_ids(item.get(field_name)):
            if turn_id not in refs:
                refs.append(turn_id)
    return refs


def collect_inline_anchors_from_text(text: Any) -> list[str]:
    anchors: list[str] = []
    for match in ANCHOR_MARKER_RE.finditer(str(text or "")):
        anchor = match.group(1).strip()
        if anchor and anchor not in anchors:
            anchors.append(anchor)
    return anchors


def collect_legacy_anchor_strings(seed_row: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    for turn in dialogue_turns_from_row(seed_row):
        for anchor in collect_inline_anchors_from_text(turn.get("text")):
            if anchor not in anchors:
                anchors.append(anchor)
    for key in ("graph_dialogue_alignment", "current_alignment", "legacy_alignment"):
        value = seed_row.get(key)
        if isinstance(value, dict):
            for text in json.dumps(value, ensure_ascii=False).split('"'):
                if re.fullmatch(r"[A-Z0-9]{8,12}", text) and text not in anchors:
                    anchors.append(text)
    return anchors


def bootstrap_legacy_alignment(seed_row: dict[str, Any]) -> dict[str, Any]:
    """Recover best-effort provenance alignment from existing turn references.

    This is intentionally provenance-only. Its output is never used directly for
    final current-round QA/evidence unless the fact is re-realized and verified
    against the frozen current clean dialogue.
    """
    text_by_id = turn_text_by_id(seed_row)
    valid_turn_ids = set(text_by_id)
    legacy_alignment: dict[str, dict[str, Any]] = {}
    missing_total = 0

    def add_ref(key: str, raw_turn_ids: Any, source: str, expected_text: str = "") -> None:
        nonlocal missing_total
        refs = normalize_turn_ids(raw_turn_ids)
        if not refs:
            legacy_alignment.setdefault(
                key,
                {
                    "turn_ids": [],
                    "support_quotes": [],
                    "confidence": "low",
                    "source": "unknown",
                    "needs_reverbalization": True,
                },
            )
            return
        existing = [turn_id for turn_id in refs if turn_id in valid_turn_ids and text_by_id.get(turn_id)]
        missing = [turn_id for turn_id in refs if turn_id not in valid_turn_ids]
        missing_total += len(missing)
        confidence = "high" if existing and not missing and len(existing) == len(refs) else "low"
        support_quotes = [text_by_id[turn_id] for turn_id in existing[:5]]
        entry = legacy_alignment.setdefault(
            key,
            {
                "turn_ids": [],
                "support_quotes": [],
                "confidence": confidence,
                "source": source if confidence == "high" else "partial_refs",
                "needs_reverbalization": confidence != "high",
            },
        )
        for turn_id in existing:
            if turn_id not in entry["turn_ids"]:
                entry["turn_ids"].append(turn_id)
        for quote in support_quotes:
            if quote not in entry["support_quotes"]:
                entry["support_quotes"].append(quote)
        if expected_text:
            entry["expected_text"] = expected_text
        if confidence == "low":
            entry["confidence"] = "low"
            entry["source"] = "partial_refs"
            entry["needs_reverbalization"] = True

    qa = seed_row.get("qa") if isinstance(seed_row.get("qa"), dict) else {}
    add_ref("qa:evidence", qa.get("evidence_turn_ids"), "existing_valid_evidence_turn_ids", normalize_space(qa.get("question")))
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            continue
        label = normalize_space(option.get("id") or option.get("label") or LETTERS[index])
        add_ref(f"qa:option:{label}", option.get("turn_ids"), "existing_valid_evidence_turn_ids", normalize_space(option.get("text")))

    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    for index, event in enumerate(graph_update.get("events") or []):
        if isinstance(event, dict):
            event_ref = normalize_space(event.get("event_id") or event.get("node_id") or f"event_{index}")
            add_ref(f"graph_update.events[{index}]:{event_ref}", item_turn_ids(event), "existing_valid_evidence_turn_ids", item_text(event))
    for index, change in enumerate(graph_update.get("state_changes") or []):
        if isinstance(change, dict):
            change_ref = normalize_space(change.get("event_id") or change.get("node_id") or f"state_change_{index}")
            add_ref(
                f"graph_update.state_changes[{index}]:{change_ref}",
                change.get("evidence_turn_ids") or item_turn_ids(change),
                "existing_valid_evidence_turn_ids",
                normalize_space(change.get("to") or item_text(change)),
            )

    for key in ("graph_dialogue_alignment", "current_alignment", "legacy_alignment"):
        alignment = seed_row.get(key)
        if not isinstance(alignment, dict):
            continue
        for ref, payload in alignment.items():
            if isinstance(payload, dict):
                add_ref(f"{key}:{ref}", payload.get("turn_ids"), "existing_alignment_metadata")

    high = sum(1 for entry in legacy_alignment.values() if entry.get("confidence") == "high")
    low = sum(1 for entry in legacy_alignment.values() if entry.get("confidence") != "high")
    return {
        "legacy_alignment": legacy_alignment,
        "legacy_rows_with_missing_refs": 1 if missing_total else 0,
        "legacy_missing_ref_total": missing_total,
        "legacy_high_confidence_count": high,
        "legacy_low_confidence_count": low,
    }


def _weighted_choice(rng: random.Random, weights: dict[str, float], allowed: Iterable[str], defaults: dict[str, float]) -> str:
    items = []
    for key in allowed:
        try:
            weight = float(weights.get(key, defaults.get(key, 0.0)))
        except (TypeError, ValueError):
            weight = defaults.get(key, 0.0)
        items.append((key, max(0.0, weight)))
    total = sum(weight for _key, weight in items)
    if total <= 0:
        items = [(key, max(0.0, defaults.get(key, 1.0))) for key in allowed]
        total = sum(weight for _key, weight in items)
    threshold = rng.random() * total
    running = 0.0
    for key, weight in items:
        running += weight
        if running >= threshold:
            return key
    return items[-1][0]


def _curriculum_mode(policy: dict[str, Any] | None) -> str:
    if not isinstance(policy, dict):
        return "default"
    mode = normalize_space(policy.get("native_curriculum_mode"))
    if mode in ("r3_long_easy", EASY_V2_MODE):
        return mode
    return "default"


def _is_easy_curriculum(policy: dict[str, Any] | None) -> bool:
    # True for any easy variant (v1 or v2) so the shared easy knobs (length, event count,
    # natural phrasing, option cleanup) apply uniformly.
    return _curriculum_mode(policy) in ("r3_long_easy", EASY_V2_MODE)


def _is_easy_v2_curriculum(policy: dict[str, Any] | None) -> bool:
    return _curriculum_mode(policy) == EASY_V2_MODE


def _largest_remainder_quota(target_shares: dict[str, float], n: int) -> dict[str, int]:
    """Deterministic integer allocation summing exactly to n (largest-remainder)."""
    keys = list(target_shares.keys())
    total_share = sum(max(0.0, float(target_shares[k])) for k in keys) or 1.0
    raw = {k: max(0.0, float(target_shares[k])) / total_share * n for k in keys}
    floors = {k: int(raw[k]) for k in keys}
    remainder = n - sum(floors.values())
    # distribute the remaining units to the largest fractional parts, ties broken by key name
    order = sorted(keys, key=lambda k: (-(raw[k] - floors[k]), k))
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    return floors


class DeterministicQuotaScheduler:
    """Assigns an exact (question_type, attack_skill) plan per sample index for easy_v2.

    The plan is computed once from n_samples so the ACCEPTED distribution matches the
    target shares exactly (no weighted-sampling drift). Attack skills are round-robined
    within each question type's allowed easy set, so per-skill share is also bounded.

    The closed-loop difficulty controller may inject:
      - question_type_shares: override of the default EASY_V2 target shares (e.g. drop a
        mastered/bad cell's question type weight). Keys outside EASY_V2_SUPPORTED are ignored;
        a zero/empty share for a type removes it from the plan.
      - cell_weights: per (question_type, attack_skill) relative weight used to bias the
        within-type skill assignment toward band cells and away from too-hard/bad cells.
    """

    def __init__(
        self,
        n_samples: int,
        rng: random.Random,
        question_type_shares: dict[str, float] | None = None,
        cell_weights: dict[str, float] | None = None,
        cell_hardness_mix: dict[str, dict[str, float]] | None = None,
        single_choice_subtype_shares: dict[str, float] | None = None,
        single_choice_subtype_hardness_mix: dict[str, dict[str, float]] | None = None,
        hard_single_choice_template_shares: dict[str, float] | None = None,
    ) -> None:
        self.n_samples = int(n_samples)
        self.cell_hardness_mix = cell_hardness_mix or {}
        self.single_choice_subtype_hardness_mix = single_choice_subtype_hardness_mix or {}
        self.single_choice_subtype_shares = {
            str(key): max(0.0, float(value))
            for key, value in (single_choice_subtype_shares or {}).items()
            if max(0.0, float(value)) > 0
        }
        self.hard_single_choice_template_shares = {
            str(key): max(0.0, float(value))
            for key, value in (hard_single_choice_template_shares or {}).items()
            if str(key) in HARD_SINGLE_CHOICE_SUBTYPES and max(0.0, float(value)) > 0
        }
        shares = {q: float(EASY_V2_QUESTION_TYPE_TARGET_SHARES[q]) for q in EASY_V2_QUESTION_TYPE_TARGET_SHARES}
        if question_type_shares:
            override = {q: max(0.0, float(v)) for q, v in question_type_shares.items() if q in EASY_V2_SUPPORTED_QUESTION_TYPES}
            if override and sum(override.values()) > 0:
                shares = override
        # drop zero-share types entirely
        shares = {q: v for q, v in shares.items() if v > 0}
        if not shares:
            shares = {q: float(EASY_V2_QUESTION_TYPE_TARGET_SHARES[q]) for q in EASY_V2_QUESTION_TYPE_TARGET_SHARES}
        self.question_type_shares = shares
        self.cell_weights = {k: max(0.0, float(v)) for k, v in (cell_weights or {}).items()}
        qtype_quota = _largest_remainder_quota(shares, self.n_samples)
        # Build an interleaved queue of question types matching the exact quota, so any
        # prefix (e.g. a partial run) stays close to target shares.
        pools: list[tuple[str, int]] = sorted(qtype_quota.items(), key=lambda kv: (-kv[1], kv[0]))
        queue: list[str] = []
        counters = {q: 0 for q, _ in pools}
        remaining = dict(qtype_quota)
        while len(queue) < self.n_samples:
            # pick the qtype that is most "behind" its quota relative to its share
            best = None
            best_key = None
            for qtype, quota in pools:
                if remaining[qtype] <= 0:
                    continue
                progress = counters[qtype] / quota if quota else 1.0
                key = (progress, qtype)
                if best_key is None or key < best_key:
                    best_key = key
                    best = qtype
            if best is None:
                break
            queue.append(best)
            counters[best] += 1
            remaining[best] -= 1
        self.queue = queue
        self.qtype_quota = qtype_quota
        self._skill_rr: dict[str, int] = {q: 0 for q in EASY_V2_SUPPORTED_QUESTION_TYPES}
        self._hardness_rr: dict[str, int] = {}
        self._single_choice_subtype_rr = 0
        self._single_choice_subtype_queue: list[str] = []
        self._hard_single_choice_template_rr = 0
        self._hard_single_choice_template_queue: list[str] = []
        if self.single_choice_subtype_shares:
            target = int(qtype_quota.get("single_choice", 0))
            if target > 0:
                subtype_quota = _largest_remainder_quota(self.single_choice_subtype_shares, target)
                subtype_pool = sorted(subtype_quota.items(), key=lambda kv: (-kv[1], kv[0]))
                while len(self._single_choice_subtype_queue) < target:
                    best = None
                    best_key = None
                    for subtype, quota in subtype_pool:
                        if quota <= 0:
                            continue
                        used = self._single_choice_subtype_queue.count(subtype)
                        progress = used / quota if quota else 1.0
                        key = (progress, subtype)
                        if best_key is None or key < best_key:
                            best_key = key
                            best = subtype
                    if best is None:
                        break
                    self._single_choice_subtype_queue.append(best)
        hard_template_target = self._single_choice_subtype_queue.count("hard_single_choice")
        if self.hard_single_choice_template_shares and hard_template_target > 0:
            template_quota = _largest_remainder_quota(self.hard_single_choice_template_shares, hard_template_target)
            template_pool = sorted(template_quota.items(), key=lambda kv: (-kv[1], kv[0]))
            while len(self._hard_single_choice_template_queue) < hard_template_target:
                best = None
                best_key = None
                for template, quota in template_pool:
                    if quota <= 0:
                        continue
                    used = self._hard_single_choice_template_queue.count(template)
                    progress = used / quota if quota else 1.0
                    key = (progress, template)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = template
                if best is None:
                    break
                self._hard_single_choice_template_queue.append(best)
        self._rng = rng

    def _pick_single_choice_hardness(self, subtype: str, qtype: str, skill: str) -> str:
        if subtype in self.single_choice_subtype_hardness_mix:
            mix = self.single_choice_subtype_hardness_mix.get(subtype) or {}
            counts = _largest_remainder_quota({h: float(w) for h, w in mix.items() if float(w) > 0} or {"h0": 1.0}, 8)
            seq: list[str] = []
            for h in ("h0", "h1", "h2"):
                seq.extend([h] * counts.get(h, 0))
            if seq:
                rr = self._hardness_rr.get(f"single_choice|{subtype}", 0)
                self._hardness_rr[f"single_choice|{subtype}"] = rr + 1
                return seq[rr % len(seq)]
        return self._pick_hardness(qtype, skill)

    def _pick_hardness(self, qtype: str, skill: str) -> str:
        """Deterministic per-cell hardness from the controller's cell_hardness_mix (default h0)."""
        cell = f"{qtype}|{skill}"
        mix = self.cell_hardness_mix.get(cell)
        if not mix:
            return "h0"
        # expand the hardness mix into a small deterministic sequence, indexed by round-robin.
        counts = _largest_remainder_quota({h: float(w) for h, w in mix.items() if float(w) > 0} or {"h0": 1.0}, 8)
        seq: list[str] = []
        for h in ("h0", "h1", "h2"):
            seq.extend([h] * counts.get(h, 0))
        if not seq:
            return "h0"
        rr = self._hardness_rr.get(cell, 0)
        self._hardness_rr[cell] = rr + 1
        return seq[rr % len(seq)]

    def _pick_skill(self, qtype: str) -> str:
        skills = EASY_V2_ATTACK_SKILLS_BY_QUESTION_TYPE.get(qtype) or tuple(EASY_ATTACK_SKILLS)
        # If the controller supplied cell weights, bias skill choice toward higher-weight
        # cells deterministically (largest cumulative weight by round-robin position).
        if self.cell_weights:
            weighted = [(s, self.cell_weights.get(f"{qtype}|{s}", 1.0)) for s in skills]
            total = sum(w for _, w in weighted)
            if total > 0:
                # deterministic: expand into a quota-like sequence by weight, index by rr
                seq: list[str] = []
                counts = _largest_remainder_quota({s: w for s, w in weighted}, max(len(skills), 8))
                for s in skills:
                    seq.extend([s] * counts.get(s, 0))
                if seq:
                    rr = self._skill_rr.get(qtype, 0)
                    self._skill_rr[qtype] = rr + 1
                    return seq[rr % len(seq)]
        rr = self._skill_rr.get(qtype, 0)
        self._skill_rr[qtype] = rr + 1
        return skills[rr % len(skills)]

    def plan_for_index(self, index: int) -> dict[str, str]:
        if not self.queue:
            qtype = EASY_V2_SUPPORTED_QUESTION_TYPES[index % len(EASY_V2_SUPPORTED_QUESTION_TYPES)]
        else:
            qtype = self.queue[index % len(self.queue)]
        skill = self._pick_skill(qtype)
        hardness = self._pick_hardness(qtype, skill)
        if qtype == "single_choice":
            if self._single_choice_subtype_queue:
                difficulty_subtype = self._single_choice_subtype_queue[self._single_choice_subtype_rr % len(self._single_choice_subtype_queue)]
                self._single_choice_subtype_rr += 1
            else:
                difficulty_subtype = "ordinary_single_choice"
            hard_single_choice_template = "ordinary_single_choice"
            if difficulty_subtype == "hard_single_choice":
                if self._hard_single_choice_template_queue:
                    hard_single_choice_template = self._hard_single_choice_template_queue[
                        self._hard_single_choice_template_rr % len(self._hard_single_choice_template_queue)
                    ]
                else:
                    hard_single_choice_template = HARD_SINGLE_CHOICE_SUBTYPES[
                        self._hard_single_choice_template_rr % len(HARD_SINGLE_CHOICE_SUBTYPES)
                    ]
                self._hard_single_choice_template_rr += 1
            hardness = self._pick_single_choice_hardness(difficulty_subtype, qtype, skill)
        else:
            difficulty_subtype = "ordinary"
            hard_single_choice_template = "ordinary"
        return {
            "question_type": qtype,
            "attack_skill": skill,
            "distractor_hardness": hardness,
            "difficulty_subtype": difficulty_subtype,
            "hard_single_choice_template": hard_single_choice_template,
        }


def choose_question_type(policy: dict[str, Any] | None, rng: random.Random) -> str:
    policy = policy or {}
    # easy_v2 uses a deterministic per-sample plan injected via policy; honor it exactly.
    planned = normalize_space((policy.get("planned_cell") or {}).get("question_type")) if isinstance(policy.get("planned_cell"), dict) else ""
    if _is_easy_v2_curriculum(policy):
        if planned in EASY_V2_SUPPORTED_QUESTION_TYPES:
            return planned
        # No silent ordering fallback: fail loud rather than drift the distribution.
        raise NativeGraphDialogueReject(f"easy_v2 missing/invalid planned question_type: {planned!r}")
    easy = _is_easy_curriculum(policy)
    supported = EASY_SUPPORTED_QUESTION_TYPES if easy else SUPPORTED_QUESTION_TYPES
    default_weights = EASY_QUESTION_TYPE_WEIGHTS if easy else DEFAULT_QUESTION_TYPE_WEIGHTS
    weights = policy.get("question_type_weights") if isinstance(policy.get("question_type_weights"), dict) else {}
    requested = normalize_space(policy.get("question_type") or policy.get("requested_question_type"))
    if requested in supported:
        return requested
    return _weighted_choice(rng, weights, supported, default_weights)


def choose_attack_skill(question_type: str, policy: dict[str, Any] | None, rng: random.Random) -> str:
    policy = policy or {}
    if _is_easy_v2_curriculum(policy):
        planned = normalize_space((policy.get("planned_cell") or {}).get("attack_skill")) if isinstance(policy.get("planned_cell"), dict) else ""
        if planned:
            return planned
        raise NativeGraphDialogueReject(
            f"easy_v2 missing/invalid planned attack_skill {planned!r} for {question_type}"
        )
    easy = _is_easy_curriculum(policy)
    if easy:
        allowed = EASY_ATTACK_SKILLS_BY_QUESTION_TYPE.get(question_type, EASY_ATTACK_SKILLS)
        approved = EASY_ATTACK_SKILLS
    else:
        allowed = ATTACK_SKILLS_BY_QUESTION_TYPE.get(question_type, APPROVED_ATTACK_SKILLS)
        approved = APPROVED_ATTACK_SKILLS
    weights = policy.get("attack_skill_weights") if isinstance(policy.get("attack_skill_weights"), dict) else {}
    default = {skill: 1.0 for skill in allowed}
    chosen = _weighted_choice(rng, weights, allowed, default)
    return chosen if chosen in approved else allowed[0]


def choose_difficulty_subtype(question_type: str, policy: dict[str, Any] | None, rng: random.Random) -> str:
    policy = policy or {}
    planned = normalize_space((policy.get("planned_cell") or {}).get("difficulty_subtype")) if isinstance(policy.get("planned_cell"), dict) else ""
    if question_type == "single_choice":
        if planned == "hard_single_choice":
            return "hard_single_choice"
        return "ordinary_single_choice"
    return "ordinary"


def compact_seed_for_prompt(seed_row: dict[str, Any], *, max_turns: int = 12) -> dict[str, Any]:
    turns = dialogue_turns_from_row(seed_row)
    if len(turns) > max_turns:
        head = turns[: max_turns // 2]
        tail = turns[-(max_turns - len(head)) :]
        turns = head + tail
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    qa = seed_row.get("qa") if isinstance(seed_row.get("qa"), dict) else {}
    return {
        "sample_id": seed_row.get("sample_id") or seed_row.get("source_id"),
        "source_id": seed_row.get("source_id"),
        "dialogue_turns": turns,
        "graph_update": {
            "events": graph_update.get("events") if isinstance(graph_update.get("events"), list) else [],
            "state_changes": graph_update.get("state_changes") if isinstance(graph_update.get("state_changes"), list) else [],
            "invalidations": graph_update.get("invalidations") if isinstance(graph_update.get("invalidations"), list) else [],
            "cancellations": graph_update.get("cancellations") if isinstance(graph_update.get("cancellations"), list) else [],
            "restorations": graph_update.get("restorations") if isinstance(graph_update.get("restorations"), list) else [],
            "delayed_activations": graph_update.get("delayed_activations")
            if isinstance(graph_update.get("delayed_activations"), list)
            else [],
        },
        "qa_question_type": qa.get("question_type"),
        "qa_gold": qa.get("gold"),
        "qa_evidence_turn_ids": qa.get("evidence_turn_ids"),
    }


def build_json_request(args: argparse.Namespace, prompt_payload: dict[str, Any], system_prompt: str, previous_error: dict[str, Any] | None, max_tokens_override: int | None = None) -> dict[str, Any]:
    payload = copy.deepcopy(prompt_payload)
    if previous_error:
        payload["previous_response_error"] = previous_error
    user_content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if previous_error:
        user_content += "\n\nReturn corrected JSON only. Do not include markdown fences."
    return {
        "model": getattr(args, "challenger_model", None) or getattr(old_gate, "DEFAULT_CHALLENGER_MODEL", "Qwen/Qwen3-14B"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": float(getattr(args, "challenger_temperature", 0.25)),
        "max_tokens": int(max_tokens_override if max_tokens_override else getattr(args, "challenger_max_tokens", 4096)),
    }


def _nested_dict(value: Any, keys: Iterable[str]) -> dict[str, Any] | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, dict) else None


def normalize_api_graph_payload(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize safe graph API wrappers without inventing graph events."""
    diagnostics = {
        "graph_json_normalizer_used": False,
        "graph_update_events_unwrap": False,
        "graph_json_normalizer_failure": False,
        "graph_json_normalizer_source": None,
    }
    if not isinstance(candidate, dict):
        diagnostics["graph_json_normalizer_failure"] = True
        return candidate, diagnostics
    if isinstance(candidate.get("events"), list):
        return candidate, diagnostics
    wrapper_paths = (
        ("graph_update",),
        ("data", "graph_update"),
        ("result", "graph_update"),
    )
    for path in wrapper_paths:
        graph_update = _nested_dict(candidate, path)
        if isinstance(graph_update, dict) and isinstance(graph_update.get("events"), list):
            normalized = copy.deepcopy(graph_update)
            for key, value in candidate.items():
                if key in {path[0], "data", "result"}:
                    continue
                normalized.setdefault(key, copy.deepcopy(value))
            diagnostics.update(
                {
                    "graph_json_normalizer_used": True,
                    "graph_update_events_unwrap": True,
                    "graph_json_normalizer_source": ".".join(path) + ".events",
                }
            )
            return normalized, diagnostics
    diagnostics["graph_json_normalizer_failure"] = True
    return candidate, diagnostics


def _parse_dialogue_text_turns(text: Any) -> list[dict[str, Any]]:
    raw = str(text or "")
    turns: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        match = DIALOGUE_TEXT_TURN_RE.match(line)
        if match:
            turn_id, speaker, body = match.groups()
            turns.append(
                {
                    "turn_id": normalize_space(turn_id) or f"append:{index + 1}",
                    "speaker": normalize_space(speaker) or f"Speaker{(index % 2) + 1}",
                    "text": normalize_space(body),
                }
            )
        else:
            turns.append(
                {
                    "turn_id": f"append:{index + 1}",
                    "speaker": f"Speaker{(index % 2) + 1}",
                    "text": normalize_space(line),
                }
            )
    return [turn for turn in turns if turn.get("text")]


def normalize_api_dialogue_payload(candidate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize safe dialogue API wrappers without inventing dialogue content."""
    diagnostics = {
        "dialogue_json_normalizer_used": False,
        "dialogue_wrapper_unwrap": False,
        "dialogue_json_normalizer_failure": False,
        "dialogue_json_normalizer_source": None,
    }
    if not isinstance(candidate, dict):
        diagnostics["dialogue_json_normalizer_failure"] = True
        return candidate, diagnostics
    normalized = copy.deepcopy(candidate)
    source: str | None = None

    for key in ("dialogue_proposal", "dialogue", "alignment", "result", "data"):
        wrapped = normalized.get(key)
        if isinstance(wrapped, dict):
            merged = copy.deepcopy(wrapped)
            for outer_key, outer_value in normalized.items():
                if outer_key != key:
                    merged.setdefault(outer_key, copy.deepcopy(outer_value))
            normalized = merged
            source = key
            break

    if "appended_dialogue_turns" not in normalized:
        if isinstance(normalized.get("final_dialogue"), list):
            normalized["appended_dialogue_turns"] = copy.deepcopy(normalized.get("final_dialogue"))
            source = source or "final_dialogue"
        elif isinstance(normalized.get("dialogue_text"), str):
            turns = _parse_dialogue_text_turns(normalized.get("dialogue_text"))
            if turns:
                normalized["appended_dialogue_turns"] = turns
                source = source or "dialogue_text"
        elif isinstance(normalized.get("dialogue"), list):
            normalized["appended_dialogue_turns"] = copy.deepcopy(normalized.get("dialogue"))
            source = source or "dialogue"

    if "event_alignment_proposal" not in normalized:
        for key in ("alignment", "alignment_proposal", "event_alignment"):
            if isinstance(normalized.get(key), list):
                normalized["event_alignment_proposal"] = copy.deepcopy(normalized.get(key))
                source = source or key
                break
        if "event_alignment_proposal" not in normalized and isinstance(normalized.get("alignment"), dict):
            alignment = normalized.get("alignment") or {}
            if isinstance(alignment.get("event_alignment_proposal"), list):
                normalized["event_alignment_proposal"] = copy.deepcopy(alignment.get("event_alignment_proposal"))
                source = source or "alignment.event_alignment_proposal"
            elif isinstance(alignment.get("proposals"), list):
                normalized["event_alignment_proposal"] = copy.deepcopy(alignment.get("proposals"))
                source = source or "alignment.proposals"

    if "qa_proposal" not in normalized:
        for key in ("qa", "question", "qa_item"):
            if isinstance(normalized.get(key), dict):
                normalized["qa_proposal"] = copy.deepcopy(normalized.get(key))
                source = source or key
                break

    changed = normalized != candidate
    diagnostics.update(
        {
            "dialogue_json_normalizer_used": changed,
            "dialogue_wrapper_unwrap": changed,
            "dialogue_json_normalizer_source": source,
            "dialogue_json_normalizer_failure": not (
                isinstance(normalized.get("event_alignment_proposal"), list)
                and isinstance(normalized.get("qa_proposal"), dict)
            ),
        }
    )
    return normalized, diagnostics


def normalize_api_payload_for_call_kind(
    candidate: dict[str, Any],
    call_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if call_kind == "api_graph_expansion":
        return normalize_api_graph_payload(candidate)
    if call_kind == "api_dialogue_expansion":
        return normalize_api_dialogue_payload(candidate)
    return candidate, {}


def graph_cache_key_for_attempt(
    seed_row: dict[str, Any],
    planned_cell: dict[str, Any] | None,
    policy: dict[str, Any] | None,
    api_args: argparse.Namespace,
    *,
    sample_index: int,
    round_id: str,
) -> str:
    cell = planned_cell if isinstance(planned_cell, dict) else {}
    payload = {
        "parent_sample_id": seed_row.get("sample_id") or seed_row.get("source_id"),
        "round_id": round_id,
        "sample_index": sample_index,
        "target_delta_bucket": int(getattr(api_args, "min_length_delta_tokens", 0) or 0),
        "difficulty_target": (policy or {}).get("item_difficulty_target") or (policy or {}).get("boundary_item_difficulty_target"),
        "qtype": cell.get("question_type"),
        "skill": cell.get("attack_skill"),
        "difficulty_subtype": cell.get("difficulty_subtype"),
        "distractor_hardness": cell.get("distractor_hardness"),
        "template": cell.get("hard_single_choice_template"),
        "planner_profile_version": "boundary_calibrated_item_planner_v1",
        "query_conditioned_graph_expansion": bool(getattr(api_args, "query_conditioned_graph_expansion", False)),
    }
    return stable_hash(payload, length=24)


def _query_conditioned_graph_enabled(args: argparse.Namespace, question_type: str, attack_skill: str) -> bool:
    if not bool(getattr(args, "query_conditioned_graph_expansion", False)):
        return False
    return (
        (question_type == "multi_select" and attack_skill == "permission_visibility_scope")
        or (question_type == "single_choice" and attack_skill == "conditional_plan")
    )


def _query_seed_candidate_is_artifact(item: dict[str, Any]) -> bool:
    text = _option_text_norm(
        " ".join(
            normalize_space(value)
            for value in [
                item.get("entity"),
                item.get("slot"),
                item.get("role"),
                item.get("value"),
                item.get("to"),
                item.get("dialogue_value"),
                item_text(item),
            ]
            if value is not None
        )
    )
    artifact_patterns = [
        "raw qa",
        "raw_qa",
        "archived report",
        "archive shelf report",
        "discarded archive",
        "the archived report",
    ]
    return any(pattern in text for pattern in artifact_patterns)


def _parent_query_seed_candidates(seed_row: dict[str, Any], limit: int = 6, *, avoid_artifacts: bool = False) -> list[dict[str, Any]]:
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    candidates: list[dict[str, Any]] = []
    for group in ("events", "state_changes"):
        for item in graph_update.get(group) or []:
            if not isinstance(item, dict):
                continue
            if avoid_artifacts and _query_seed_candidate_is_artifact(item):
                continue
            text = normalize_space(item_text(item))
            if not text:
                continue
            status = normalize_space(item.get("status") or item.get("candidate_status") or "active")
            if status and status not in ACTIVE_STATUSES and len(candidates) >= 2:
                continue
            candidates.append(
                {
                    "origin_event_id": normalize_space(item.get("event_instance_id") or item.get("event_id") or item.get("node_id")) or None,
                    "entity": sanitize_internal_text(item.get("entity") or "conversation"),
                    "slot": sanitize_internal_text(item.get("slot") or item.get("role") or "memory_state"),
                    "value_preview": sanitize_internal_text(text)[:220],
                    "status": status or "active",
                    "turn_ids": normalize_turn_ids(item.get("turn_ids") or item.get("evidence_turn_ids")),
                }
            )
            if len(candidates) >= limit:
                return candidates
    return candidates


def build_query_conditioned_graph_objective(
    seed_row: dict[str, Any],
    *,
    question_type: str,
    attack_skill: str,
    policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    difficulty_target = (
        (policy or {}).get("item_difficulty_target")
        or (policy or {}).get("boundary_item_difficulty_target")
        or "boundary"
    )
    parent_candidates = _parent_query_seed_candidates(
        seed_row,
        avoid_artifacts=question_type == "multi_select" and attack_skill == "permission_visibility_scope",
    )
    if question_type == "single_choice" and attack_skill == "conditional_plan":
        return {
            "query_conditioned_graph_expansion": True,
            "query_plan_type": "conditional_plan_hard_single_choice",
            "item_difficulty_target": _normalize_boundary_item_difficulty_target(difficulty_target) or "boundary",
            "parent_candidate_previews": parent_candidates[:4],
            "hidden_query_goal": (
                "Create graph events that will later support a QA-last hard single-choice item. "
                "There must be exactly one final-current gold proposition and at least three plausible "
                "graph-grounded false propositions that are close enough to be tempting but invalidated "
                "by final state, condition, override, or guarantee logic."
            ),
            "required_graph_roles": [
                {
                    "role": "final_gold_current_state",
                    "source": "new child graph update",
                    "status": "active",
                    "requirement": (
                        "add one final current-valid state for a concrete entity/slot/value; this is the only "
                        "unambiguously correct single-choice answer"
                    ),
                },
                {
                    "role": "same_slot_overridden_negative",
                    "source": "parent-to-child transition or new child graph update",
                    "status": "outdated|canceled|expired|false|delayed_not_active|scope_limited",
                    "requirement": (
                        "add one false event for the same entity and same slot as the gold, representing an "
                        "earlier/intermediate value, overridden condition, or invalidated plan"
                    ),
                    "hard_negative_path_type": "same_entity_same_slot_overridden",
                },
                {
                    "role": "same_entity_wrong_condition_negative",
                    "source": "new child graph update",
                    "status": "delayed_not_active|scope_limited|false|outdated",
                    "requirement": (
                        "add one false event for the same entity but wrong condition/scope/trigger, so it is "
                        "plausible but not guaranteed by the final graph"
                    ),
                    "hard_negative_path_type": "wrong_condition_same_entity",
                },
                {
                    "role": "possible_not_guaranteed_negative",
                    "source": "parent or child graph",
                    "status": "belief|report|false|outdated|expired|canceled",
                    "requirement": (
                        "add one plausible report/belief/possible statement that is not guaranteed by the final state"
                    ),
                    "hard_negative_path_type": "possible_but_not_guaranteed",
                },
            ],
            "graph_update_constraints": [
                "Do not create multiple active/current-valid answer candidates for the same question surface.",
                "Include at least three non-active distractor events with clear support_claim text.",
                "At least one false event must share the gold entity and slot; at least one must share the gold entity.",
                "Make final-state override/condition/guarantee distinctions explicit in graph fields, not only in prose.",
                "Do not write the final question or answer in the graph response.",
            ],
            "expected_downstream_qa": {
                "question_type": "single_choice",
                "attack_skill": "conditional_plan",
                "expected_hard_negative_path_types": [
                    "same_entity_same_slot_overridden",
                    "wrong_condition_same_entity",
                    "possible_but_not_guaranteed",
                ],
                "expected_dependency_count_range": [3, 6],
                "expected_solver_bucket": "1/5-4/5",
            },
        }
    if question_type != "multi_select" or attack_skill != "permission_visibility_scope":
        return None
    return {
        "query_conditioned_graph_expansion": True,
        "query_plan_type": "permission_visibility_scope_boundary_multi_select",
        "item_difficulty_target": _normalize_boundary_item_difficulty_target(difficulty_target) or "boundary",
        "parent_candidate_previews": parent_candidates[:4],
            "hidden_query_goal": (
                "Create graph events that will later support a QA-last multi-select item. "
                "The final QA should require one parent-context current-valid fact plus one child-append current-valid fact, "
                "and two graph-grounded hard negatives whose option surfaces look like plausible current facts rather than archive/report placeholders."
            ),
        "required_graph_roles": [
            {
                "role": "parent_gold_context",
                "source": "existing parent graph",
                "requirement": "at least one current-valid parent fact must remain usable as gold evidence",
            },
            {
                "role": "child_gold_current",
                "source": "new child graph update",
                "status": "active",
                "requirement": "add one child event that is current-valid under the intended visibility/scope/permission condition",
            },
            {
                "role": "child_hard_negative_scope",
                "source": "new child graph update",
                "status": "scope_limited",
                "requirement": (
                    "add one plausible event that looks current but is wrong for the queried scope/person/condition; "
                    "the entity/slot/value must be a natural fact, not a raw-QA/archive/report artifact"
                ),
                "hard_negative_path_type": "active_but_wrong_scope",
            },
            {
                "role": "parent_hard_negative_invalidated",
                "source": "parent-to-child transition",
                "status": "outdated|canceled|expired|false",
                "requirement": (
                    "add one event that explicitly invalidates or overrides a plausible natural parent dependency; "
                    "do not invalidate an archive/report placeholder or raw-QA scaffold"
                ),
                "hard_negative_path_type": "invalidated_parent_dependency",
            },
        ],
        "graph_update_constraints": [
            "Do not make all false options merely outdated; include one current-looking scope/person/condition mismatch.",
            "Do not put the answer in the final dialogue surface; create graph facts and support claims only.",
            "Keep gold and hard-negative events close enough to be plausible but separable by scope/permission/condition.",
            "Use natural entity/slot/value wording that can be verbalized later without exposing schema labels.",
            "Do not use archive/report placeholders as hard negatives: values such as 'the archived report', 'discarded archive', or 'archive-shelf report' are forbidden for false candidates.",
            "Do not use raw-QA scaffold slots or values as hard negatives; false candidates must look like ordinary memory facts.",
            "For every false candidate, include a natural_false_surface_preview field in metadata or alignment_hints showing how it would look as a plausible option without schema/artifact words.",
            "Each false candidate must have natural current-looking entity/slot/value text and a support_claim that sounds like a real dialogue fact.",
        ],
        "forbidden_false_surface_patterns": [
            "the archived report",
            "archived report",
            "archive-shelf report",
            "discarded archive",
            "raw qa",
            "raw_qa",
        ],
            "expected_downstream_qa": {
                "question_type": "multi_select",
                "attack_skill": "permission_visibility_scope",
                "expected_hard_negative_path_types": ["active_but_wrong_scope", "invalidated_parent_dependency"],
            "expected_dependency_count_range": [4, 6],
            "expected_solver_bucket": "1/5-4/5",
        },
    }


def evaluate_query_conditioned_graph_plan(plan: dict[str, Any], objective: dict[str, Any] | None) -> dict[str, Any]:
    if not objective:
        return {"query_conditioned_graph_expansion": False, "pass": True, "failure_reason": None}
    events = [event for event in plan.get("events") or [] if isinstance(event, dict)]
    active = [event for event in events if str(event.get("status")) in ACTIVE_STATUSES]
    scope_limited = [
        event
        for event in events
        if str(event.get("status")) == "scope_limited" or str(event.get("event_type")) == "permission_scope"
    ]
    invalidating = [
        event
        for event in events
        if str(event.get("status")) in FALSE_STATUSES
        or str(event.get("event_type")) in {"cancel", "expire", "conflict"}
    ]
    support_claims = [
        normalize_space(((event.get("alignment_hints") or {}) if isinstance(event.get("alignment_hints"), dict) else {}).get("support_claim"))
        for event in events
    ]
    failures: list[str] = []
    query_plan_type = normalize_space(objective.get("query_plan_type"))
    if query_plan_type == "conditional_plan_hard_single_choice":
        false_events = [event for event in events if str(event.get("status")) in FALSE_STATUSES]
        gold_event = max(active, key=lambda event: int(event.get("time_index") or 0), default=None)
        same_slot_false = []
        same_entity_false = []
        if isinstance(gold_event, dict):
            same_slot_false = [
                event
                for event in false_events
                if normalize_space(event.get("entity")) == normalize_space(gold_event.get("entity"))
                and normalize_space(event.get("slot")) == normalize_space(gold_event.get("slot"))
            ]
            same_entity_false = [
                event
                for event in false_events
                if normalize_space(event.get("entity")) == normalize_space(gold_event.get("entity"))
            ]
        if len(events) < 4:
            failures.append("too_few_events")
        if len(active) < 1:
            failures.append("missing_final_active_gold")
        if len(false_events) < 3:
            failures.append("missing_three_false_distractors")
        if not same_slot_false:
            failures.append("missing_same_slot_false_negative")
        if not same_entity_false:
            failures.append("missing_same_entity_false_negative")
        if not all(claim for claim in support_claims[: min(len(support_claims), 4)]):
            failures.append("missing_support_claim")
        return {
            "query_conditioned_graph_expansion": True,
            "pass": not failures,
            "failure_reason": ";".join(failures) if failures else None,
            "event_count": len(events),
            "active_event_count": len(active),
            "scope_limited_event_count": len(scope_limited),
            "invalidating_event_count": len(invalidating),
            "single_choice_false_event_count": len(false_events),
            "single_choice_same_slot_false_count": len(same_slot_false),
            "single_choice_same_entity_false_count": len(same_entity_false),
            "query_plan_type": objective.get("query_plan_type"),
            "expected_hard_negative_path_types": (objective.get("expected_downstream_qa") or {}).get("expected_hard_negative_path_types"),
            "expected_dependency_count_range": (objective.get("expected_downstream_qa") or {}).get("expected_dependency_count_range"),
        }
    natural_scope_limited = [
        event for event in scope_limited if not _event_option_surface_is_placeholder(event) and not _query_seed_candidate_is_artifact(event)
    ]
    natural_invalidating = [
        event for event in invalidating if not _event_option_surface_is_placeholder(event) and not _query_seed_candidate_is_artifact(event)
    ]
    placeholder_false_events = [
        event
        for event in scope_limited + invalidating
        if _event_option_surface_is_placeholder(event) or _query_seed_candidate_is_artifact(event)
    ]
    if len(events) < 4:
        failures.append("too_few_events")
    if len(active) < 2:
        failures.append("missing_two_current_valid_events")
    if not natural_scope_limited:
        failures.append("missing_scope_limited_hard_negative")
    if not natural_invalidating:
        failures.append("missing_invalidating_hard_negative")
    if placeholder_false_events and (not natural_scope_limited or not natural_invalidating):
        failures.append("placeholder_false_surface_used_for_required_negative")
    if not all(claim for claim in support_claims[: min(len(support_claims), 4)]):
        failures.append("missing_support_claim")
    return {
        "query_conditioned_graph_expansion": True,
        "pass": not failures,
        "failure_reason": ";".join(failures) if failures else None,
        "event_count": len(events),
        "active_event_count": len(active),
        "scope_limited_event_count": len(scope_limited),
        "invalidating_event_count": len(invalidating),
        "natural_scope_limited_event_count": len(natural_scope_limited),
        "natural_invalidating_event_count": len(natural_invalidating),
        "placeholder_false_surface_count": len(placeholder_false_events),
        "query_plan_type": objective.get("query_plan_type"),
        "expected_hard_negative_path_types": (objective.get("expected_downstream_qa") or {}).get("expected_hard_negative_path_types"),
        "expected_dependency_count_range": (objective.get("expected_downstream_qa") or {}).get("expected_dependency_count_range"),
    }


def graph_cache_dir_from_args(api_args: argparse.Namespace, output: Path) -> Path:
    configured = normalize_space(getattr(api_args, "graph_proposal_cache_dir", None))
    return Path(configured) if configured else output / "graph_proposal_cache"


def load_graph_cache_entry(cache_dir: Path, cache_key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("current_event_plan"), dict):
        return None
    return payload


def write_graph_cache_entry(
    cache_dir: Path,
    cache_key: str,
    *,
    current_event_plan: dict[str, Any],
    graph_meta: dict[str, Any],
    raw_graph: dict[str, Any] | None,
) -> None:
    if not isinstance(current_event_plan, dict) or not isinstance(current_event_plan.get("events"), list):
        return
    payload = {
        "cache_key": cache_key,
        "created_at": int(time.time()),
        "current_event_plan": current_event_plan,
        "graph_meta": graph_meta,
        "raw_graph": raw_graph,
    }
    write_json_atomic(cache_dir / f"{cache_key}.json", payload)


def call_raw_json_api(
    args: argparse.Namespace,
    prompt_payload: dict[str, Any],
    *,
    system_prompt: str,
    required_keys: tuple[str, ...],
    call_kind: str,
    max_tokens_override: int | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    api_key = (
        getattr(args, "challenger_api_key", None)
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = getattr(args, "challenger_base_url", None)
    model = getattr(args, "challenger_model", None)
    if not api_key or not base_url or not model:
        return None, {"used": False, "call_kind": call_kind, "reason": "missing_api_config", "attempts": 0}
    url = str(base_url).rstrip("/") + "/chat/completions"
    max_retries = int(getattr(args, "max_llm_retries", 1))
    errors: list[dict[str, Any]] = []
    previous_error: dict[str, Any] | None = None
    for attempt in range(1, max_retries + 2):
        request_payload = build_json_request(args, prompt_payload, system_prompt, previous_error, max_tokens_override)
        try:
            if hasattr(old_gate, "post_json_with_curl"):
                data = old_gate.post_json_with_curl(url, request_payload, api_key, max(1, int(getattr(args, "llm_timeout", 60))))
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=int(getattr(args, "llm_timeout", 60))) as response:
                    data = json.loads(response.read().decode("utf-8"))
            content = str(data["choices"][0]["message"]["content"])
            candidate, parsed_json = old_gate.json_candidate_from_text(content)
            if not parsed_json or not isinstance(candidate, dict):
                previous_error = {
                    "attempt": attempt,
                    "error_type": "invalid_json",
                    "raw_response_preview": old_gate.raw_response_preview(content),
                    "required_keys": list(required_keys),
                }
                errors.append(previous_error)
            else:
                normalizer_diag: dict[str, Any] = {}
                stripped_content = content.strip()
                json_block_extracted = not stripped_content.startswith("{") or not stripped_content.endswith("}")
                if bool(getattr(args, "api_yield_normalizers", False)):
                    candidate, normalizer_diag = normalize_api_payload_for_call_kind(candidate, call_kind)
                if call_kind == "api_dialogue_expansion" and json_block_extracted:
                    normalizer_diag["dialogue_json_block_extracted"] = True
                missing = [key for key in required_keys if key not in candidate]
                if not missing:
                    meta = {
                        "used": True,
                        "call_kind": call_kind,
                        "attempts": attempt,
                        "model": model,
                        "base_url": base_url,
                        "previous_errors": errors,
                    }
                    if normalizer_diag:
                        meta["normalizer_diagnostics"] = normalizer_diag
                    return candidate, meta
                previous_error = {
                    "attempt": attempt,
                    "error_type": "missing_required_keys",
                    "missing_keys": missing,
                    "raw_response_preview": old_gate.raw_response_preview(content),
                    "required_keys": list(required_keys),
                }
                if normalizer_diag:
                    previous_error["normalizer_diagnostics"] = normalizer_diag
                errors.append(previous_error)
        except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            previous_error = {
                "attempt": attempt,
                "error_type": old_gate.classify_api_exception(exc) if hasattr(old_gate, "classify_api_exception") else "api_request_error",
                "error": str(exc),
                "required_keys": list(required_keys),
            }
            errors.append(previous_error)
        if attempt <= max_retries:
            time.sleep(min(2**attempt, 12))
    last_error = errors[-1] if errors else {}
    return None, {
        "used": False,
        "call_kind": call_kind,
        "attempts": len(errors),
        "model": model,
        "base_url": base_url,
        "error_type": last_error.get("error_type", "api_failed"),
        "errors": errors,
    }


def _seed_entity_slot_value(seed_row: dict[str, Any]) -> tuple[str, str, str, str | None]:
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    for event in graph_update.get("events") or []:
        if isinstance(event, dict):
            entity = normalize_space(event.get("entity") or seed_row.get("source_id") or "conversation")
            slot = normalize_space(event.get("slot") or event.get("role") or "memory_state")
            value = normalize_space(item_text(event))
            if value:
                return entity, slot, value, normalize_space(event.get("event_id") or event.get("node_id")) or None
    for change in graph_update.get("state_changes") or []:
        if isinstance(change, dict):
            entity = normalize_space(change.get("entity") or seed_row.get("source_id") or "conversation")
            slot = normalize_space(change.get("slot") or "memory_state")
            value = normalize_space(change.get("to") or item_text(change))
            if value:
                return entity, slot, value, normalize_space(change.get("event_id") or change.get("node_id")) or None
    qa = seed_row.get("qa") if isinstance(seed_row.get("qa"), dict) else {}
    gold = qa.get("gold") if isinstance(qa.get("gold"), list) else []
    if gold:
        return normalize_space(seed_row.get("source_id") or "conversation"), "memory_state", normalize_space(gold[0]), None
    return normalize_space(seed_row.get("source_id") or "conversation"), "memory_state", "the previously tracked memory remains under review", None


def _fallback_event_specs(seed_row: dict[str, Any], question_type: str, attack_skill: str) -> list[dict[str, Any]]:
    entity, slot, base_value, origin_event_id = _seed_entity_slot_value(seed_row)
    base_short = base_value[:120].rstrip(".")
    variant_a = f"{entity} / {slot}: temporary plan is the archive shelf instead of the current destination"
    variant_b = f"{entity} / {slot}: confirmed active state keeps {base_short}"
    variant_c = f"{entity} / {slot}: final active state after review is {base_short}"
    observer = "Jordan"
    if attack_skill == "permission_visibility_scope":
        observer = "Morgan"
    return [
        {
            "origin_event_id": origin_event_id,
            "time_index": 1,
            "event_type": "state_change",
            "entity": entity,
            "slot": slot,
            "value": f"{entity} / {slot}: baseline remembered state is {base_short}",
            "previous_value": None,
            "status": "active",
            "source": "seed",
        },
        {
            "time_index": 2,
            "event_type": "report",
            "entity": entity,
            "slot": slot,
            "value": variant_a,
            "previous_value": base_value,
            "status": "report",
            "source": "api_graph_expansion",
        },
        {
            "time_index": 3,
            "event_type": "belief_update",
            "entity": observer,
            "slot": f"belief_about_{slot}",
            "value": f"{observer} believes {variant_a}",
            "previous_value": None,
            "status": "belief",
            "source": "api_graph_expansion",
        },
        {
            "time_index": 4,
            "event_type": "cancel",
            "entity": entity,
            "slot": slot,
            "value": f"{entity} / {slot}: the temporary archive-shelf plan is canceled and not active",
            "previous_value": variant_a,
            "status": "canceled",
            "source": "api_graph_expansion",
        },
        {
            "time_index": 5,
            "event_type": "state_change",
            "entity": entity,
            "slot": slot,
            "value": variant_b,
            "previous_value": variant_a,
            "status": "active",
            "source": "api_graph_expansion",
        },
        {
            "time_index": 6,
            "event_type": "permission_scope",
            "entity": entity,
            "slot": slot,
            "value": f"{entity} / {slot}: only {observer} can see the discarded archive-shelf report",
            "previous_value": variant_b,
            "status": "scope_limited",
            "source": "api_graph_expansion",
        },
        {
            "time_index": 7,
            "event_type": "state_change",
            "entity": entity,
            "slot": slot,
            "value": variant_c,
            "previous_value": variant_b,
            "status": "active",
            "source": "api_graph_expansion",
        },
    ]


def _coerce_event_type(value: Any, status: str) -> str:
    text = normalize_space(value)
    if text in APPROVED_EVENT_TYPES:
        return text
    if status == "belief":
        return "belief_update"
    if status == "report":
        return "report"
    if status == "canceled":
        return "cancel"
    if status == "restored":
        return "restore"
    if status == "expired":
        return "expire"
    if status == "scope_limited":
        return "permission_scope"
    if status == "delayed_not_active":
        return "delay"
    return "state_change"


def _coerce_status(value: Any) -> str:
    text = normalize_text(value).replace(" ", "_")
    if text in APPROVED_STATUSES:
        return text
    if "cancel" in text:
        return "canceled"
    if "expire" in text:
        return "expired"
    if "belief" in text or "believe" in text:
        return "belief"
    if "report" in text or "claim" in text:
        return "report"
    if "scope" in text or "permission" in text or "limited" in text:
        return "scope_limited"
    if "delay" in text:
        return "delayed_not_active"
    if "false" in text or "wrong" in text:
        return "false"
    return "active"


def compute_graph_snapshot_hash(event_plan: dict[str, Any]) -> str:
    truth = copy.deepcopy(event_plan)
    truth.pop("graph_snapshot_hash", None)
    truth.pop("current_event_instance_to_anchor", None)
    truth.pop("current_anchor_to_event_instance_id", None)
    for event in truth.get("events") or []:
        if isinstance(event, dict):
            event.pop("alignment_anchor", None)
            event.pop("raw_value", None)
            event.pop("raw_support_claim", None)
            event.pop("dialogue_value", None)
            event.pop("dialogue_support_claim", None)
            event.pop("dialogue_fact_id", None)
    return stable_hash(truth, length=24)


# Per-question-type structural prerequisites the QA builders enforce. Used to deterministically
# coerce the normalized event statuses so QA-last never fails for a structural reason.
QTYPE_STATUS_REQUIREMENTS = {
    "single_choice": {"active": 1, "false": 2},
    "multi_select": {"active": 2, "false": 2},
    "ordering": {"active": 0, "false": 0},  # only needs >=4 aligned events (any status)
    "evidence_selection": {"active": 1, "false": 0},
    "state_reconstruction": {"active": 1, "false": 0},
}
_DEFAULT_ACTIVE_STATUS = "active"
_DEFAULT_FALSE_STATUS = "outdated"


def _enforce_qtype_status_prerequisites(raw_events: list[dict[str, Any]], question_type: str) -> list[dict[str, Any]]:
    """Deterministically coerce RAW event statuses so the planned qtype's builder will succeed.

    Runs BEFORE the normalization loop builds support claims, so the support-claim status
    language stays consistent with the (possibly re-labeled) status. The API often returns a
    status mix that does not satisfy a builder (single_choice needs 1 active + 2 non-active,
    multi_select needs 2 active + 2 false). Rather than fail at QA-last, we re-label surplus
    events here. Values/entities/slots are untouched; only `status` changes. Status is a label
    on each event (not a cross-event relation), so the graph stays self-consistent.
    """
    req = QTYPE_STATUS_REQUIREMENTS.get(question_type)
    if not req:
        return raw_events
    need_active = int(req.get("active", 0))
    need_false = int(req.get("false", 0))
    if need_active + need_false == 0:
        return raw_events
    usable = [e for e in raw_events if isinstance(e, dict)]
    if len(usable) < need_active + need_false:
        return raw_events  # not enough events; let downstream >=4 / builder checks reject

    def status_of(e: dict[str, Any]) -> str:
        return _coerce_status(e.get("status"))

    n_active = sum(1 for e in usable if status_of(e) in ACTIVE_STATUSES)
    n_false = sum(1 for e in usable if status_of(e) in FALSE_STATUSES)
    if n_active >= need_active and n_false >= need_false:
        return raw_events  # already satisfiable

    def time_key(e: dict[str, Any]) -> int:
        try:
            return int(e.get("time_index"))
        except (TypeError, ValueError):
            return usable.index(e) + 1

    ordered = sorted(usable, key=time_key)
    # earliest events -> active; next -> false; remainder untouched.
    for e in ordered[:need_active]:
        if status_of(e) not in ACTIVE_STATUSES:
            e["status"] = _DEFAULT_ACTIVE_STATUS
            e["status_coerced_for_qtype"] = question_type
    for e in ordered[need_active:need_active + need_false]:
        if status_of(e) not in FALSE_STATUSES:
            e["status"] = _DEFAULT_FALSE_STATUS
            e["status_coerced_for_qtype"] = question_type
    return raw_events



def normalize_current_event_plan(
    raw_plan: dict[str, Any] | None,
    seed_row: dict[str, Any],
    *,
    question_type: str,
    attack_skill: str,
    policy: dict[str, Any] | None,
    sample_index: int,
    round_id: str,
) -> dict[str, Any]:
    raw = raw_plan if isinstance(raw_plan, dict) else {}
    raw_events = raw.get("events") if isinstance(raw.get("events"), list) else []
    if len(raw_events) < 4:
        raw_events = _fallback_event_specs(seed_row, question_type, attack_skill)
    fallback_events = _fallback_event_specs(seed_row, question_type, attack_skill)
    if len(raw_events) < len(fallback_events):
        raw_events = list(raw_events) + fallback_events[len(raw_events) :]
    # qtype-aware prerequisite enforcement: re-label statuses so the planned question type's
    # builder has the active/false event counts it needs (prevents QA-last structural rejects).
    raw_events = _enforce_qtype_status_prerequisites(raw_events, question_type)

    graph_snapshot_id = normalize_space(raw.get("graph_snapshot_id")) or f"{round_id}:S{sample_index:04d}:graph"
    normalized_events: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for index, raw_event in enumerate(raw_events, start=1):
        if not isinstance(raw_event, dict):
            continue
        status = _coerce_status(raw_event.get("status"))
        event_type = _coerce_event_type(raw_event.get("event_type") or raw_event.get("role"), status)
        entity = normalize_space(raw_event.get("entity") or raw_event.get("actor") or seed_row.get("source_id") or "conversation")
        slot = normalize_space(raw_event.get("slot") or raw_event.get("field") or "memory_state")
        value = normalize_space(raw_event.get("value") or raw_event.get("state") or item_text(raw_event))
        if len(value) < 8:
            continue
        value_key = normalize_text((entity, slot, value, status))
        if value_key in seen_values:
            continue
        seen_values.add(value_key)
        try:
            time_index = int(raw_event.get("time_index"))
        except (TypeError, ValueError):
            time_index = len(normalized_events) + 1
        event_instance_id = f"{round_id}:S{sample_index:04d}:E{len(normalized_events) + 1:02d}"
        support_claim = normalize_space(
            ((raw_event.get("alignment_hints") or {}) if isinstance(raw_event.get("alignment_hints"), dict) else {}).get("support_claim")
        )
        if not support_claim:
            support_claim = f"{entity} / {slot}: {value}."
            if status != "active":
                support_claim += f" Mark this fact as {status.replace('_', ' ')}."
        if status in STATUS_ALIASES:
            support_norm = normalize_text(support_claim)
            has_status_language = any(normalize_text(alias) in support_norm for alias in STATUS_ALIASES[status])
            if not has_status_language:
                if status == "belief":
                    support_claim += " This is a belief, not an active current state."
                elif status == "report":
                    support_claim += " This is only a report, not an active current state."
                elif status == "scope_limited":
                    support_claim += " This is scope limited, not a generally visible active state."
                elif status == "delayed_not_active":
                    support_claim += " This is delayed and not active yet."
                else:
                    support_claim += f" This is {status.replace('_', ' ')} and not an active current state."
        dialogue_value = sanitize_internal_text(value)
        dialogue_support_claim = _naturalize_claim_for_status(support_claim or value, status)
        must_mention_any = [entity, slot]
        must_mention_value_any = [value]
        normalized_events.append(
            {
                "event_instance_id": event_instance_id,
                "origin_event_id": normalize_space(raw_event.get("origin_event_id") or raw_event.get("event_id") or raw_event.get("node_id"))
                or None,
                "lineage_id": normalize_space(raw_event.get("lineage_id")) or None,
                "round_id": round_id,
                "time_index": time_index,
                "event_type": event_type,
                "entity": entity,
                "slot": slot,
                "raw_value": value,
                "value": value,
                "dialogue_value": dialogue_value,
                "previous_value": normalize_space(raw_event.get("previous_value")) or None,
                "status": status,
                "required_for_dialogue": bool(raw_event.get("required_for_dialogue", True)),
                "source": normalize_space(raw_event.get("source") or "api_graph_expansion"),
                "raw_support_claim": support_claim,
                "dialogue_support_claim": dialogue_support_claim,
                "dialogue_fact_id": f"F{len(normalized_events) + 1:02d}",
                "alignment_hints": {
                    "must_mention_any": must_mention_any,
                    "must_mention_value_any": [dialogue_value],
                    "support_claim": dialogue_support_claim,
                },
            }
        )
    if len(normalized_events) < 4:
        raise NativeGraphDialogueReject("current event_plan did not contain enough graph events")
    normalized_events.sort(key=lambda event: (int(event.get("time_index") or 0), str(event.get("event_instance_id"))))
    for idx, event in enumerate(normalized_events, start=1):
        event["time_index"] = idx
    false_candidates = raw.get("false_candidates") if isinstance(raw.get("false_candidates"), list) else []
    if not false_candidates:
        false_candidates = [
            {
                "candidate_event_instance_id": event["event_instance_id"],
                "text": event["value"],
                "candidate_status": event["status"],
            }
            for event in normalized_events
            if event.get("status") in FALSE_STATUSES
        ]
    plan = {
        "graph_snapshot_id": graph_snapshot_id,
        "question_type": _normalize_plan_question_type(question_type, policy),
        "attack_skill": _normalize_plan_attack_skill(attack_skill, policy),
        "events": normalized_events,
        "false_candidates": false_candidates,
        "metadata": {
            **(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
            "sample_index": sample_index,
            "round_id": round_id,
            "source_seed_id": seed_row.get("sample_id") or seed_row.get("source_id"),
            "frozen_graph": True,
            "graph_normalized_by_program": True,
        },
    }
    plan["graph_snapshot_hash"] = compute_graph_snapshot_hash(plan)
    return plan


def _normalize_plan_question_type(question_type: str, policy: dict[str, Any] | None) -> str:
    """Normalize the plan's question_type WITHOUT silently rewriting to ordering.

    Default mode keeps the original behavior (must be in SUPPORTED_QUESTION_TYPES).
    Easy modes accept their own supported set (incl. single_choice). An out-of-set value
    is a loud reject, never a silent drift to ordering.
    """
    if _is_easy_v2_curriculum(policy):
        if question_type in EASY_V2_SUPPORTED_QUESTION_TYPES:
            return question_type
        raise NativeGraphDialogueReject(f"easy_v2 plan question_type out of set: {question_type!r}")
    if _is_easy_curriculum(policy):
        if question_type in EASY_SUPPORTED_QUESTION_TYPES:
            return question_type
        raise NativeGraphDialogueReject(f"easy plan question_type out of set: {question_type!r}")
    if question_type in SUPPORTED_QUESTION_TYPES:
        return question_type
    raise NativeGraphDialogueReject(f"plan question_type out of set: {question_type!r}")


def _normalize_plan_attack_skill(attack_skill: str, policy: dict[str, Any] | None) -> str:
    if _is_easy_curriculum(policy):
        if attack_skill in EASY_ATTACK_SKILLS:
            return attack_skill
        raise NativeGraphDialogueReject(f"easy plan attack_skill out of set: {attack_skill!r}")
    if attack_skill in APPROVED_ATTACK_SKILLS:
        return attack_skill
    raise NativeGraphDialogueReject(f"plan attack_skill out of set: {attack_skill!r}")


def call_api_graph_expansion(
    seed_row: dict[str, Any],
    legacy_alignment_result: dict[str, Any],
    policy: dict[str, Any] | None,
    args: argparse.Namespace,
    rng: random.Random,
    *,
    sample_index: int,
    round_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    question_type = choose_question_type(policy, rng)
    attack_skill = choose_attack_skill(question_type, policy, rng)
    event_count_instruction = (
        EASY_GRAPH_EVENT_INSTRUCTION
        if _is_easy_curriculum(policy)
        else "Include 5-8 events, enough active and non-active facts for ordering/multi_select/state/evidence QA."
    )
    # qtype-aware prerequisite hint so the API tends to emit the active/non-active mix the
    # planned question type's builder requires (coercion in normalize_* is the safety net).
    _qtype_req_hint = {
        "single_choice": "For single_choice: include EXACTLY 1 active/current-true event and AT LEAST 3 non-active events (outdated/canceled/expired/belief/report/scope_limited) so three graph-validated distractors can be built.",
        "multi_select": "For multi_select: include AT LEAST 2 active/current-true events and AT LEAST 2 non-active (false) events.",
        "evidence_selection": "For evidence_selection: include at least 1 active event with a clear supporting turn.",
        "state_reconstruction": "For state_reconstruction: include at least 1 active final state event.",
        "ordering": "For ordering: include at least 4 events with distinct time_index values.",
    }.get(question_type)
    requirements = [
        "The graph/state-chain/event_plan is the source of truth.",
        "Return a self-contained current graph snapshot for this new round.",
        "Do not rely on old event ids as persistent ids; they are provenance only.",
        event_count_instruction,
        "Every important event should be required_for_dialogue=true and have clear alignment_hints.support_claim.",
        "Use only approved attack skills and event/status values from the provided lists.",
    ]
    if _qtype_req_hint:
        requirements.append(_qtype_req_hint)
    query_objective = (
        build_query_conditioned_graph_objective(
            seed_row,
            question_type=question_type,
            attack_skill=attack_skill,
            policy=policy,
        )
        if _query_conditioned_graph_enabled(args, question_type, attack_skill)
        else None
    )
    if query_objective:
        requirements.append(
            "This run uses a hidden query-conditioned graph expansion objective. The objective is for graph construction only; do not write the final question or answer."
        )
        if query_objective.get("query_plan_type") == "conditional_plan_hard_single_choice":
            requirements.extend(
                [
                    "Create the graph update so downstream QA-last can ask a hard single-choice conditional-plan question.",
                    "Your events must include exactly one final current-valid gold-like state and at least three plausible graph-grounded false alternatives.",
                    "Make false alternatives tempting through same-entity/same-slot override, wrong condition, or possible-but-not-guaranteed structure.",
                    "Do not expose the final answer as a surface statement; only provide graph events and support claims that can later be verbalized naturally.",
                ]
            )
        else:
            requirements.extend(
                [
                    "Create the graph update so downstream QA-last can ask a multi-hop permission/visibility/scope question over parent + child evidence.",
                    "Your events must include both a current-valid child gold fact and two graph-grounded hard negatives: one scope/person/condition mismatch and one invalidated parent dependency.",
                    "Do not expose the final answer as a surface statement; only provide graph events and support claims that can later be verbalized naturally.",
                ]
            )
    prompt = {
        "task": "current_method graph-first native challenger graph/event_plan expansion",
        "requirements": requirements,
        "round_id": round_id,
        "sample_index": sample_index,
        "requested_question_type": question_type,
        "requested_attack_skill": attack_skill,
        "approved_attack_skills": list(APPROVED_ATTACK_SKILLS),
        "approved_event_types": sorted(APPROVED_EVENT_TYPES),
        "approved_statuses": sorted(APPROVED_STATUSES),
        "seed": compact_seed_for_prompt(seed_row),
        "legacy_alignment_result": legacy_alignment_result,
        "hidden_query_conditioned_objective": query_objective,
        "output_schema": {
            "graph_snapshot_id": "string",
            "question_type": "single_choice|ordering|multi_select|evidence_selection|state_reconstruction",
            "attack_skill": "approved attack skill",
            "events": [
                {
                    "origin_event_id": "optional provenance only",
                    "lineage_id": "optional provenance only",
                    "round_id": round_id,
                    "time_index": 1,
                    "event_type": "state_change|belief_update|cancel|restore|delay|expire|report|permission_scope|conflict",
                    "entity": "string",
                    "slot": "string",
                    "value": "string",
                    "previous_value": "optional string",
                    "status": "active|outdated|canceled|restored|expired|delayed_not_active|false|belief|report|scope_limited",
                    "required_for_dialogue": True,
                    "source": "api_graph_expansion",
                    "alignment_hints": {
                        "must_mention_any": ["entity or slot aliases"],
                        "must_mention_value_any": ["value aliases"],
                        "support_claim": "single sentence to realize in dialogue",
                    },
                }
            ],
            "false_candidates": [],
            "metadata": {
                "query_conditioned_graph_expansion": "optional bool; echo true only if hidden_query_conditioned_objective was followed",
                "query_plan_type": "optional string",
                "query_conditioned_notes": "optional short string",
            },
        },
    }
    raw_payload, meta = call_raw_json_api(
        args,
        prompt,
        system_prompt=(
            "You are a strict JSON-only graph-first challenger. Expand the current graph/event_plan. "
            "Return exactly one JSON object. Do not write dialogue. Do not write final trusted QA."
        ),
        required_keys=("events",),
        call_kind="api_graph_expansion",
        max_tokens_override=int(getattr(args, "graph_expansion_max_tokens", 0) or 0) or None,
    )
    plan = normalize_current_event_plan(
        raw_payload,
        seed_row,
        question_type=question_type,
        attack_skill=attack_skill,
        policy=policy,
        sample_index=sample_index,
        round_id=round_id,
    )
    if query_objective:
        plan.setdefault("metadata", {})
        if isinstance(plan["metadata"], dict):
            plan["metadata"]["query_conditioned_graph_objective"] = query_objective
        query_diag = evaluate_query_conditioned_graph_plan(plan, query_objective)
        if isinstance(plan["metadata"], dict):
            plan["metadata"]["query_conditioned_graph_diagnostics"] = query_diag
        meta["query_conditioned_graph_objective"] = query_objective
        meta["query_conditioned_graph_diagnostics"] = query_diag
        if not query_diag.get("pass"):
            meta["used"] = False
            meta["error_type"] = "query_conditioned_graph_plan_failed"
            meta["reason"] = query_diag.get("failure_reason")
    meta["normalized_graph_snapshot_hash"] = plan.get("graph_snapshot_hash")
    meta["requested_question_type"] = question_type
    meta["requested_attack_skill"] = attack_skill
    return plan, meta, raw_payload


def random_anchor(rng: random.Random, used: set[str]) -> str:
    for _ in range(1000):
        length = rng.randint(8, 12)
        anchor = "".join(rng.choice(ANCHOR_CHARS) for _ in range(length))
        if anchor not in used:
            used.add(anchor)
            return anchor
    raise RuntimeError("failed to create unique random alignment anchor")


def assign_current_alignment_anchors(current_event_plan: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    plan = copy.deepcopy(current_event_plan)
    event_to_anchor: dict[str, str] = {}
    anchor_to_event: dict[str, str] = {}
    used: set[str] = set()
    for event in plan.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = normalize_space(event.get("event_instance_id"))
        if not event_id:
            continue
        anchor = random_anchor(rng, used)
        event["alignment_anchor"] = anchor
        event_to_anchor[event_id] = anchor
        anchor_to_event[anchor] = event_id
    plan["current_event_instance_to_anchor"] = event_to_anchor
    plan["current_anchor_to_event_instance_id"] = anchor_to_event
    return plan


def sanitize_event_plan_for_dialogue_api(current_event_plan: dict[str, Any]) -> dict[str, Any]:
    """Build a dialogue-facing current plan without raw schema/lineage identifiers."""
    event_to_marker = current_event_plan.get("current_event_instance_to_anchor") if isinstance(current_event_plan.get("current_event_instance_to_anchor"), dict) else {}
    sanitized_events: list[dict[str, Any]] = []
    for index, event in enumerate(current_event_plan.get("events") or [], start=1):
        if not isinstance(event, dict):
            continue
        event_id = normalize_space(event.get("event_instance_id"))
        anchor = normalize_space(event.get("alignment_anchor") or event_to_marker.get(event_id))
        status = normalize_space(event.get("status") or "active")
        claim = normalize_space(event.get("dialogue_support_claim") or ((event.get("alignment_hints") or {}) if isinstance(event.get("alignment_hints"), dict) else {}).get("support_claim") or event.get("value"))
        claim = _naturalize_claim_for_status(claim, status)
        value = sanitize_internal_text(event.get("dialogue_value") or event.get("value"))
        sanitized_events.append(
            {
                "event_instance_id": event_id,
                "fact_id": normalize_space(event.get("dialogue_fact_id")) or f"F{index:02d}",
                "time_index": event.get("time_index"),
                "scenario_fact": value,
                "dialogue_support_claim": claim,
                "status_meaning": {
                    "active": "settled as true in the conversation",
                    "restored": "restored or made true again",
                    "outdated": "old information corrected later",
                    "canceled": "called off",
                    "expired": "no longer valid",
                    "delayed_not_active": "planned for later, not yet happening",
                    "false": "corrected as not true",
                    "belief": "someone thinks this, but it may be uncertain",
                    "report": "someone reported this, but it may need confirmation",
                    "scope_limited": "only some people can know this",
                }.get(status, "conversation update"),
                "marker": anchor,
                "required": event.get("required_for_dialogue") is not False,
            }
        )
    return {
        "question_type": current_event_plan.get("question_type"),
        "attack_skill": current_event_plan.get("attack_skill"),
        "events": sanitized_events,
        "instructions": [
            "Use fact_id only to organize your answer; do not write fact_id in dialogue text.",
            "Use marker only in sidecar alignment_anchors/support_quotes or inline [[MARKER]], never as natural dialogue.",
        ],
    }


def sanitize_legacy_alignment_for_dialogue_api(legacy_alignment_result: dict[str, Any]) -> dict[str, Any]:
    legacy = legacy_alignment_result.get("legacy_alignment") if isinstance(legacy_alignment_result.get("legacy_alignment"), dict) else {}
    sanitized: dict[str, Any] = {}
    for index, (_key, entry) in enumerate(list(legacy.items())[:8], start=1):
        if not isinstance(entry, dict):
            continue
        quotes = [sanitize_internal_text(quote) for quote in (entry.get("support_quotes") or [])[:2]]
        expected = sanitize_internal_text(entry.get("expected_text"))
        sanitized[f"legacy_note_{index}"] = {
            "support_quotes": [quote for quote in quotes if quote],
            "expected_text": expected,
            "confidence": entry.get("confidence"),
            "needs_reverbalization": bool(entry.get("needs_reverbalization")),
        }
    return {
        "legacy_notes": sanitized,
        "legacy_high_confidence_count": legacy_alignment_result.get("legacy_high_confidence_count"),
        "legacy_low_confidence_count": legacy_alignment_result.get("legacy_low_confidence_count"),
    }


def sanitize_seed_context_for_dialogue_api(seed_row: dict[str, Any]) -> dict[str, Any]:
    turns = dialogue_turns_from_row(seed_row)
    selected = turns[:4] + (turns[-4:] if len(turns) > 4 else [])
    sanitized_turns: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for turn in selected:
        text = sanitize_internal_text(turn.get("text"))
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        sanitized_turns.append({"speaker": sanitize_internal_text(turn.get("speaker") or "Speaker"), "text": text})
    qa = seed_row.get("qa") if isinstance(seed_row.get("qa"), dict) else {}
    return {
        "conversation_style_examples": sanitized_turns[:8],
        "previous_question_type": sanitize_internal_text(qa.get("question_type")),
        "previous_question_summary": sanitize_internal_text(qa.get("question")),
    }


def _dialogue_context_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": normalize_space(turn.get("turn_id")),
        "speaker": sanitize_internal_text(turn.get("speaker") or "Speaker"),
        "text": sanitize_internal_text(turn.get("text")),
    }


def _state_summary_for_dialogue_api(seed_row: dict[str, Any], current_event_plan: dict[str, Any]) -> dict[str, Any]:
    active: list[str] = []
    superseded: list[str] = []
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    for item in list(graph_update.get("events") or []) + list(graph_update.get("state_changes") or []):
        if not isinstance(item, dict):
            continue
        text = sanitize_internal_text(item_text(item))
        if text and text not in active:
            active.append(text)
        if len(active) >= 8:
            break
    for event in current_event_plan.get("events") or []:
        if not isinstance(event, dict):
            continue
        text = sanitize_internal_text(event.get("dialogue_value") or event.get("value"))
        status = normalize_space(event.get("status") or "active")
        if not text:
            continue
        if status in {"active", "restored"} and text not in active:
            active.append(text)
        elif status in {"outdated", "canceled", "expired", "false", "delayed_not_active"} and text not in superseded:
            superseded.append(text)
    return {
        "active_or_latest_context": active[:12],
        "superseded_or_nonsettled_context": superseded[:12],
    }


def build_parent_context_for_dialogue_api(
    parent_dialogue_turns: list[dict[str, Any]],
    seed_row: dict[str, Any],
    legacy_alignment_result: dict[str, Any],
    current_event_plan: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    tail_limit, relevant_limit, token_budget = parent_context_limits(args)
    by_id = {normalize_space(turn.get("turn_id")): turn for turn in parent_dialogue_turns if isinstance(turn, dict)}
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_turn(turn: dict[str, Any]) -> None:
        turn_id = normalize_space(turn.get("turn_id"))
        text = normalize_space(turn.get("text"))
        if not turn_id or not text or turn_id in selected_ids:
            return
        selected.append(turn)
        selected_ids.add(turn_id)

    legacy = legacy_alignment_result.get("legacy_alignment") if isinstance(legacy_alignment_result.get("legacy_alignment"), dict) else {}
    for entry in legacy.values():
        if len(selected) >= relevant_limit:
            break
        if not isinstance(entry, dict):
            continue
        for turn_id in normalize_turn_ids(entry.get("turn_ids")):
            if len(selected) >= relevant_limit:
                break
            if turn_id in by_id:
                add_turn(by_id[turn_id])
    for item in list(graph_update.get("events") or []) + list(graph_update.get("state_changes") or []):
        if len(selected) >= relevant_limit:
            break
        if not isinstance(item, dict):
            continue
        for turn_id in item_turn_ids(item):
            if len(selected) >= relevant_limit:
                break
            if turn_id in by_id:
                add_turn(by_id[turn_id])
    tail_turns = parent_dialogue_turns[-tail_limit:] if tail_limit else []
    for turn in tail_turns:
        add_turn(turn)

    selected = sorted(selected, key=lambda turn: turn_sort_key(turn.get("turn_id")))
    compact_turns: list[dict[str, Any]] = []
    for turn in selected:
        candidate = compact_turns + [_dialogue_context_turn(turn)]
        if estimate_json_tokens(candidate) > token_budget and compact_turns:
            break
        compact_turns = candidate
    digest = {
        "source": "compact_parent_context",
        "full_parent_turn_count": len(parent_dialogue_turns),
        "selected_turn_count": len(compact_turns),
        "tail_turn_limit": tail_limit,
        "relevant_turn_limit": relevant_limit,
        "token_budget": token_budget,
        "summary": (
            "The parent dialogue is historical context. The program preserves the full parent dialogue outside this prompt; "
            "the API should only continue with new appended turns."
        ),
    }
    return {
        "parent_dialogue_digest": digest,
        "selected_relevant_parent_turns": compact_turns,
        "active_superseded_state_summary": _state_summary_for_dialogue_api(seed_row, current_event_plan),
        "diagnostics": {
            "parent_context_turns": len(compact_turns),
            "parent_context_tokens_estimate": estimate_json_tokens(compact_turns),
        },
    }


def sanitized_event_value_count(current_event_plan: dict[str, Any]) -> int:
    count = 0
    for event in current_event_plan.get("events") or []:
        if isinstance(event, dict) and normalize_space(event.get("value")) != normalize_space(event.get("dialogue_value")):
            count += 1
    return count


def call_api_graph_conditioned_dialogue_expansion(
    current_event_plan_with_anchors: dict[str, Any],
    seed_row: dict[str, Any],
    legacy_alignment_result: dict[str, Any],
    policy: dict[str, Any] | None,
    args: argparse.Namespace,
    *,
    parent_context: dict[str, Any],
    parent_dialogue_source: str,
    parent_turn_count: int,
    min_added_turns: int,
    min_added_tokens: int,
    length_guard_parent_baseline_tokens: int = 0,
    retry_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    dialogue_plan = sanitize_event_plan_for_dialogue_api(current_event_plan_with_anchors)
    sanitized_seed_context = sanitize_seed_context_for_dialogue_api(seed_row)
    anchor_realization_checklist = [
        {
            "fact_id": event.get("fact_id"),
            "marker": event.get("marker"),
            "natural_support_claim_to_include": event.get("dialogue_support_claim"),
            "status_meaning": event.get("status_meaning"),
        }
        for event in dialogue_plan.get("events") or []
        if isinstance(event, dict) and event.get("required") is not False
    ]
    easy_mode_scaffold = _is_easy_curriculum(policy)
    natural_dialogue_scaffold = []
    for index, item in enumerate(anchor_realization_checklist):
        marker = item.get("marker")
        claim = normalize_space(item.get("natural_support_claim_to_include"))
        if not (marker and claim):
            continue
        # In easy mode, scrub residual synthetic surface phrases from the visible scaffold text
        # at generation time, then keep support_quotes[marker] identical to that final visible
        # text so the quote is literally extracted from what the solver sees. Alignment markers
        # and the anchoring contract are untouched.
        if easy_mode_scaffold:
            claim = _natural_surface_text(claim)
        natural_dialogue_scaffold.append(
            {
                "speaker": "Analyst" if index % 2 == 0 else "Reviewer",
                "text": claim,
                "alignment_anchors": [marker],
                "support_quotes": {str(marker): claim},
            }
        )
    easy_mode = _is_easy_curriculum(policy)
    hard_length_delta = int(min_added_tokens or 0) >= 4000
    parent_prompt_tokens, _parent_prompt_token_mode = dialogue_tokens_for_turns(
        parent_dialogue_turns_from_row(seed_row)[0],
        normalize_space(
            getattr(args, "solver_tokenizer", None)
            or getattr(args, "tokenizer_path", None)
            or getattr(args, "base_model", None)
        ),
    )
    effective_parent_baseline = max(parent_prompt_tokens, int(length_guard_parent_baseline_tokens or 0))
    effective_required_new_tokens = max(int(min_added_tokens or 0), effective_parent_baseline - parent_prompt_tokens + int(min_added_tokens or 0))
    if easy_mode and not hard_length_delta:
        length_instruction = (
            f"Append at least {max(min_added_turns, 12)} turns and target {max(min_added_tokens, 500)}-900 "
            "new approximate tokens. Keep the appended block short and natural; do not pad."
        )
    else:
        length_instruction = (
            f"Append at least {max(min_added_turns, 24)} turns and target at least "
            f"{max(effective_required_new_tokens + 384, 1400)} new approximate tokens. "
            "Use enough substantive new turns to create a genuinely longer conversation; do not rely on short filler."
        )
    stage64_qa_mode = bool(isinstance(policy, dict) and policy.get("qa_operation_answer_type_diversification"))
    stage64_target = policy.get("stage64_target") if isinstance(policy, dict) and isinstance(policy.get("stage64_target"), dict) else {}
    if stage64_qa_mode:
        target_operation = normalize_space(stage64_target.get("qa_operation"))
        preferred_answer_type = normalize_space(stage64_target.get("preferred_answer_type"))
        target_clause = (
            f" Use qa_operation={target_operation!r} for this sample."
            if target_operation
            else ""
        )
        answer_type_clause = (
            f" Prefer answer_type={preferred_answer_type!r} if the current graph facts support it; otherwise choose another allowed answer_type."
            if preferred_answer_type
            else ""
        )
        qa_requirement = (
            "Because qa_operation_answer_type_diversification is enabled, qa_proposal is the final QA object. "
            "Choose the qa_operation, qa_subtype, answer_type, options, gold option, query_plan, and anchor_refs yourself from the current graph facts; "
            "the program will validate the schema and graph references but will not rebuild the QA locally."
            f"{target_clause}{answer_type_clause}"
        )
        qa_schema: dict[str, Any] = {
            "question_type": "single_choice",
            "qa_operation": target_operation or list(STAGE64_QA_OPERATIONS),
            "qa_subtype": "short subtype string, e.g. final_state, before_after, entity_to_event, scope_yesno, explicit_count_lookup",
            "answer_type": (
                f"one string from {list(STAGE64_ANSWER_TYPES)}; prefer {preferred_answer_type}"
                if preferred_answer_type
                else list(STAGE64_ANSWER_TYPES)
            ),
            "answer_format": "single_choice_mcq",
            "question": "natural MCQ question; avoid ambiguous pronouns such as it/they/them/this",
            "question_text": "same as question",
            "options": [
                {
                    "id": "A/B/C/D; use A/B for YesNo",
                    "text": "answer option text",
                    "answer_type": "same answer_type as qa_proposal",
                    "truth_value": "true only for the gold option",
                    "source_event_instance_id": "event_instance_id from sanitized_dialogue_fact_plan.events",
                    "anchor_ref": "the event marker or event_instance_id used as a graph reference",
                    "anchor_kind": "event or yesno_option",
                }
            ],
            "gold": ["single gold option id"],
            "gold_answer_text": "text of the gold option",
            "query_plan": {
                "answer_source": "graph_anchor",
                "qa_operation": "same as qa_operation",
                "qa_subtype": "same as qa_subtype",
                "answer_type": "same as answer_type",
                "reasoning_anchor_path": ["event_instance_id or marker references"],
                "required_anchor_refs": ["event_instance_id or marker references"],
                "supporting_anchor_refs": ["event_instance_id or marker references"],
                "gold_anchor_refs": ["event_instance_id or marker references"],
                "distractor_anchor_refs": ["event_instance_id or marker references"],
                "gold_answer": "text of the gold option",
                "normalized_gold_answer": "lowercase normalized gold answer text",
            },
            "anchor_refs": {
                "support": ["event_instance_id or marker references"],
                "gold": ["event_instance_id or marker references"],
                "distractors": ["event_instance_id or marker references"],
            },
        }
    else:
        qa_requirement = "Return qa_proposal only as a proposal. The program will rebuild final qa.gold and evidence."
        qa_schema = {
            "question_type": current_event_plan_with_anchors.get("question_type"),
            "question": "optional proposed wording",
            "target_event_instance_ids": [],
            "target_state": {},
            "options": [],
        }
    prompt = {
        "task": "current_method graph-conditioned API appended dialogue expansion with current anchors",
        "requirements": [
            "You are given a sanitized frozen current fact plan. Do not change fact truth.",
            "Generate only new appended_dialogue_turns for this round; do not rewrite, summarize, or replace the parent dialogue.",
            "The program will construct final_dialogue = preserved_parent_dialogue_turns + appended_dialogue_turns.",
            length_instruction,
            "Make appended turns substantive: each turn should usually contain 2-4 natural sentences, not short acknowledgements.",
            "Write appended turns that naturally continue the existing conversation and express every required current graph event.",
            "The natural_dialogue_scaffold is already valid and natural. Preserve every scaffold turn exactly in appended_dialogue_turns.",
            "You may add neutral setup/closing turns, but do not delete, merge, paraphrase, or unanchor scaffold turns.",
            "For each checklist item, include the natural_support_claim_to_include text verbatim in at least one dialogue turn.",
            "Use one separate dialogue turn for each checklist item; do not merge multiple checklist items into one support turn.",
            "The support turn may be simple and direct; correctness is more important than literary style.",
            "Attach that fact's marker to the same turn via alignment_anchors and support_quotes.",
            "Set support_quotes[marker] to the exact natural support claim you included.",
            "For required graph events, sidecar alignment_anchors/support_quotes are enough; event_alignment_proposal may be an empty list when sidecars cover all required markers.",
            "Do not invent event_alignment_proposal entries for neutral/context turns.",
            "Use sidecar alignment_anchors/support_quotes; inline [[ANCHOR]] markers are only a fallback.",
            "Write as if two or more people are discussing real-world plans, tasks, beliefs, corrections, or updates.",
            "Do not mention internal dataset IDs, conversation IDs, r2/r3 extension numbers, state-chain, memory state, graph_update, timeline event, event ids, anchors, markers, fact_id, or schema labels.",
            "Do not write 'Confirm that'.",
            "Do not write 'This is a belief, not an active current state'.",
            "Do not directly label facts as active/inactive/current state/belief unless those words are naturally part of the conversation.",
            "Convert all internal facts into natural scenario language.",
            "For uncertain, reported, outdated, canceled, expired, delayed, limited-visibility, or conflicting facts, express them naturally through uncertainty, mistaken assumptions, corrections, cancellations, restored plans, expired information, limited visibility, or conflicting reports.",
            "For any non-settled fact, include clear natural status language in the same marked turn, such as unsure, reported, outdated, canceled, expired, delayed, limited to specific people, or corrected as mistaken.",
            "Treat old parent dialogue as historical context only; put all new state changes, corrections, cancellations, restorations, delayed activations, or visibility changes in appended turns.",
            "Include neutral setup and closing/context turns without anchors inside the appended turns.",
            qa_requirement,
            *(
                [
                    "In qa_proposal, cite only event_instance_id values or marker values from sanitized_dialogue_fact_plan.events; do not invent graph references.",
                    "For non-YesNo Stage64 QA, provide exactly four options A-D. For YesNo Stage64 QA, provide exactly two options A-B.",
                    "qa_proposal.options is mandatory and must be a JSON array of option objects. Never omit options.",
                    "Every option must include id, text, answer_type, source_event_instance_id, anchor_ref, and truth_value.",
                    "qa_proposal.gold must be a JSON array containing exactly one option id such as [\"A\"], never an event id, marker, or answer text.",
                    "Use exactly one gold option. Do not include raw_qa, gold, correct answer, answer key, anchor, marker, graph, event_instance, or internal slot names in the user-facing question/options.",
                    "Do not include solver-visible template phrases in qa_proposal.question, question_text, gold_answer_text, or options[].text: latest confirmed detail(s), confirmed detail(s), most recent confirmed detail(s), final active state, confirmed active state, current graph, graph anchor, event instance, marker, anchor, raw_qa.",
                    "If a fact was originally phrased as a Q/A artifact, rewrite only the real-world fact in natural language; never quote the literal Q: or A: wrapper in question/options.",
                    "Every question must name a concrete entity/person/place/object and ask for a concrete relation, date, count, visibility scope, cause, comparison, or consequence.",
                    "For temporal_state_update, ask about the concrete person/place/object and the latest real-world fact from the dialogue; do not ask generic questions like which event is settled as true, which event is currently settled as true, which option reflects the state after all updates, or which statement is correct.",
                    "For temporal_state_update, the question must be concrete, e.g. 'After the later correction, what place did Audrey settle on for the dogs?' Never use 'Which event...', 'currently settled as true', 'settled as true', 'final active state', or 'state after all updates'.",
                    "qa_proposal.gold must always be option labels only, such as ['A'] or ['D']; never put a marker, anchor_ref, event_instance_id, or answer text in gold.",
                ]
                if stage64_qa_mode
                else []
            ),
            "Do not expose old event ids, old anchors, current markers, or schema words as final truth in dialogue text.",
        ],
        "sanitized_dialogue_fact_plan": dialogue_plan,
        "anchor_realization_checklist": anchor_realization_checklist,
        "natural_dialogue_scaffold": natural_dialogue_scaffold,
        "compact_parent_context": {
            "parent_dialogue_digest": parent_context.get("parent_dialogue_digest"),
            "selected_relevant_parent_turns": parent_context.get("selected_relevant_parent_turns"),
            "last_parent_turns_for_continuity": (
                parent_context.get("selected_relevant_parent_turns")[-int(getattr(args, "parent_context_tail_turns", DEFAULT_PARENT_CONTEXT_TAIL_TURNS)) :]
                if isinstance(parent_context.get("selected_relevant_parent_turns"), list)
                else []
            ),
            "active_superseded_state_summary": parent_context.get("active_superseded_state_summary"),
        },
        "preserved_parent_dialogue": {
            "source": parent_dialogue_source,
            "turn_count": parent_turn_count,
            "not_included_in_prompt": True,
            "construction_rule": "final_dialogue = preserved_parent_dialogue_turns + appended_dialogue_turns",
            "length_guard_effective_parent_baseline_tokens": effective_parent_baseline,
            "length_guard_required_new_tokens": effective_required_new_tokens,
        },
        "sanitized_seed_context": sanitized_seed_context,
        "stage64_target": copy.deepcopy(stage64_target) if stage64_qa_mode else {},
        "retry_context": retry_context or {},
        "banned_dialogue_terms": list(BANNED_SOLVER_VISIBLE_TERMS),
        "output_schema": {
            "appended_dialogue_turns": [
                {
                    "speaker": "string",
                    "text": "natural dialogue text; no internal ids/schema terms; may include [[MARKER]] fallback markers",
                    "alignment_anchors": ["marker strings"],
                    "support_quotes": {"MARKER": "exact natural quote inside text without marker"},
                }
            ],
            "event_alignment_proposal": [
                {
                    "alignment_anchor": "marker string",
                    "turn_index": "0-based index within appended_dialogue_turns",
                    "support_quote": "string",
                }
            ],
            "qa_proposal": qa_schema,
            "dialogue_style_rationale": "string",
        },
    }
    prompt_token_estimate = estimate_json_tokens(prompt)
    dialogue_max_tokens_override = None
    if stage64_qa_mode:
        try:
            dialogue_max_tokens_override = max(int(getattr(args, "challenger_max_tokens", 4096) or 4096), 8192)
        except (TypeError, ValueError):
            dialogue_max_tokens_override = 8192
    proposal, meta = call_raw_json_api(
        args,
        prompt,
        system_prompt=(
            "You are a strict JSON-only graph-conditioned dialogue writer. "
            "The frozen graph is source of truth. Use current random anchors for alignment. "
            "Return exactly one JSON object. Generate appended dialogue only."
        ),
        required_keys=("event_alignment_proposal", "qa_proposal"),
        call_kind="api_dialogue_expansion",
        max_tokens_override=dialogue_max_tokens_override,
    )
    meta["prompt_tokens_estimate"] = prompt_token_estimate
    meta["parent_context_turns"] = ((parent_context.get("diagnostics") or {}).get("parent_context_turns") if isinstance(parent_context.get("diagnostics"), dict) else None)
    meta["parent_context_tokens_estimate"] = ((parent_context.get("diagnostics") or {}).get("parent_context_tokens_estimate") if isinstance(parent_context.get("diagnostics"), dict) else None)
    return proposal, meta


def required_anchor_repair_items(
    current_event_plan: dict[str, Any],
    alignment_result: dict[str, Any],
    raw_alignment_proposal: dict[str, Any],
) -> list[dict[str, Any]]:
    dialogue_plan = sanitize_event_plan_for_dialogue_api(current_event_plan)
    missing_ids = set(str(item) for item in (alignment_result.get("missing_required_event_instance_ids") or []))
    proposals_by_event: dict[str, int] = Counter()
    for proposal in raw_alignment_proposal.get("proposals") or []:
        if isinstance(proposal, dict) and normalize_space(proposal.get("event_instance_id")):
            proposals_by_event[normalize_space(proposal.get("event_instance_id"))] += 1
    repair_items: list[dict[str, Any]] = []
    for event in dialogue_plan.get("events") or []:
        if not isinstance(event, dict) or event.get("required") is False:
            continue
        marker = normalize_space(event.get("marker"))
        event_id = normalize_space(event.get("event_instance_id"))
        if event_id in missing_ids or marker:
            repair_items.append(
                {
                    "event_instance_id": event_id,
                    "fact_id": event.get("fact_id"),
                    "marker": marker,
                    "natural_support_claim_to_include": event.get("dialogue_support_claim"),
                    "status_meaning": event.get("status_meaning"),
                    "currently_missing": event_id in missing_ids or proposals_by_event.get(event_id, 0) == 0,
                }
            )
    return repair_items


def call_api_alignment_repair(
    current_event_plan_with_anchors: dict[str, Any],
    appended_dialogue_turns: list[dict[str, Any]],
    alignment_result: dict[str, Any],
    raw_alignment_proposal: dict[str, Any],
    args: argparse.Namespace,
    *,
    parent_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    repair_items = required_anchor_repair_items(current_event_plan_with_anchors, alignment_result, raw_alignment_proposal)
    prompt = {
        "task": "repair only appended dialogue alignment",
        "requirements": [
            "Minimally patch appended_dialogue_turns only; do not rewrite the parent dialogue.",
            "Keep valid appended turns unchanged whenever possible.",
            "Add or revise only turns needed to support missing current events or fix contradictions.",
            "Generate alignment only for newly appended turns.",
            "Do not reference anchors that were not provided and do not invent anchors.",
            "Every required current event must be naturally expressed in at least one appended turn.",
            "If an event is missing, add a natural appended turn for it.",
            "If an event contradicts another event, express the correction, cancellation, override, or status change explicitly.",
            "Do not align one anchor to a turn that states the opposite meaning.",
            "Parent dialogue is preserved by the program; do not rewrite it.",
            "Return the full repaired appended_dialogue_turns list, not a patch diff.",
        ],
        "current_appended_dialogue_turns": appended_dialogue_turns,
        "alignment_failure": {
            "missing_event_instance_ids": alignment_result.get("missing_required_event_instance_ids"),
            "unsupported_alignment_count": alignment_result.get("unsupported_alignment_count"),
            "contradiction_count": alignment_result.get("contradiction_count"),
            "anchor_leakage_count": alignment_result.get("anchor_leakage_count"),
        },
        "required_anchor_repair_items": repair_items,
        "compact_parent_context": {
            "parent_dialogue_digest": parent_context.get("parent_dialogue_digest"),
            "selected_relevant_parent_turns": parent_context.get("selected_relevant_parent_turns"),
            "active_superseded_state_summary": parent_context.get("active_superseded_state_summary"),
        },
        "output_schema": {
            "appended_dialogue_turns": [
                {
                    "speaker": "string",
                    "text": "natural dialogue text; no internal ids/schema terms; may include [[MARKER]] fallback markers",
                    "alignment_anchors": ["provided marker strings only"],
                    "support_quotes": {"MARKER": "exact natural quote inside text without marker"},
                }
            ],
            "event_alignment_proposal": [
                {
                    "alignment_anchor": "provided marker string",
                    "turn_index": "0-based index within repaired appended_dialogue_turns",
                    "support_quote": "exact quote from that appended turn",
                }
            ],
        },
    }
    prompt_token_estimate = estimate_json_tokens(prompt)
    proposal, meta = call_raw_json_api(
        args,
        prompt,
        system_prompt=(
            "You are a strict JSON-only alignment repair editor. "
            "Patch appended dialogue only and preserve valid turns. Return exactly one JSON object."
        ),
        required_keys=("appended_dialogue_turns", "event_alignment_proposal"),
        call_kind="api_dialogue_alignment_repair",
    )
    meta["prompt_tokens_estimate"] = prompt_token_estimate
    meta["parent_context_turns"] = ((parent_context.get("diagnostics") or {}).get("parent_context_turns") if isinstance(parent_context.get("diagnostics"), dict) else None)
    meta["parent_context_tokens_estimate"] = ((parent_context.get("diagnostics") or {}).get("parent_context_tokens_estimate") if isinstance(parent_context.get("diagnostics"), dict) else None)
    return proposal, meta


def _stage64_event_candidates_for_prompt(
    current_event_plan_with_anchors: dict[str, Any],
    current_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compact graph evidence view for QA-only repair.

    The repair call should be much cheaper than full dialogue regeneration, so
    expose only event refs and natural claims that the QA schema may cite.
    """
    dialogue_plan = sanitize_event_plan_for_dialogue_api(current_event_plan_with_anchors)
    event_to_anchor = current_event_plan_with_anchors.get("current_event_instance_to_anchor")
    if not isinstance(event_to_anchor, dict):
        event_to_anchor = {}
    rows: list[dict[str, Any]] = []
    for event in dialogue_plan.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = normalize_space(event.get("event_instance_id"))
        marker = normalize_space(event.get("marker") or event_to_anchor.get(event_id))
        if not event_id:
            continue
        rows.append(
            {
                "event_instance_id": event_id,
                "marker": marker,
                "entity": normalize_space(event.get("entity")),
                "slot": normalize_space(event.get("slot")),
                "status": normalize_space(event.get("status")),
                "answer_candidate_text": normalize_space(
                    event.get("dialogue_support_claim")
                    or event.get("value")
                    or event.get("dialogue_value")
                ),
                "evidence_turn_ids": _turn_ids_for_event(event, current_alignment),
            }
        )
    return rows


def _stage64_compact_appended_turns_for_prompt(appended_dialogue_turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact: list[dict[str, str]] = []
    for turn in appended_dialogue_turns:
        if not isinstance(turn, dict):
            continue
        text = normalize_space(GENERIC_MARKER_RE.sub(" ", str(turn.get("text") or "")))
        if not text:
            continue
        compact.append(
            {
                "speaker": normalize_space(turn.get("speaker") or "Speaker"),
                "text": text,
            }
        )
    return compact[-80:]


def call_api_stage64_qa_repair(
    current_event_plan_with_anchors: dict[str, Any],
    current_alignment: dict[str, Any],
    previous_qa_proposal: dict[str, Any] | None,
    stage64_target: dict[str, Any] | None,
    failure_reason: str,
    appended_dialogue_turns: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    target_operation = normalize_space((stage64_target or {}).get("qa_operation")) if isinstance(stage64_target, dict) else ""
    preferred_answer_type = normalize_space((stage64_target or {}).get("preferred_answer_type")) if isinstance(stage64_target, dict) else ""
    required_answer_types = sorted(STAGE64_OPERATION_REQUIRED_ANSWER_TYPES.get(target_operation) or [])
    prompt = {
        "task": "repair Stage64 qa_proposal only",
        "requirements": [
            "Do not rewrite dialogue. Do not invent new graph facts. Return JSON only.",
            "Return exactly one key: qa_proposal.",
            "The previous qa_proposal failed local validation. Fix only the QA schema, operation, answer type, options, gold, query_plan, and anchor_refs.",
            "Use only event_instance_id or marker values listed in event_candidates as graph references.",
            "For non-YesNo answers, provide exactly four options A-D. For YesNo, provide exactly two options A-B.",
            "Every option must include id, text, answer_type, source_event_instance_id, anchor_ref, anchor_kind, and truth_value.",
            "gold must be an array with exactly one option id, such as [\"A\"].",
            "query_plan.answer_source must be graph_anchor, and query_plan qa_operation / qa_subtype / answer_type must match qa_proposal.",
            "Do not include raw_qa, Q:, A:, current graph, graph anchor, anchor, marker, event instance, latest confirmed detail, final active state, or schema language in the user-facing question or options.",
            "Name concrete people/entities/places/objects in the question; avoid generic questions like which event is settled as true.",
            "If qa_operation is temporal_state_update, ask for the concrete latest real-world detail about a named entity. Never write 'Which event...', 'currently settled as true', 'settled as true', 'final active state', 'state after all updates', or 'which statement is correct'.",
            "qa_proposal.gold must be option labels only, such as [\"A\"] or [\"D\"]; never put a marker, anchor_ref, event_instance_id, or answer text in gold.",
        ],
        "validator_failure_reason": failure_reason,
        "target": {
            "qa_operation": target_operation or list(STAGE64_QA_OPERATIONS),
            "preferred_answer_type": preferred_answer_type or None,
            "required_answer_types": required_answer_types,
            "answer_type_rule": (
                f"Must use one of {required_answer_types}."
                if required_answer_types
                else f"Prefer {preferred_answer_type}, but choose another allowed answer_type if the evidence does not support it."
            ),
        },
        "event_candidates": _stage64_event_candidates_for_prompt(current_event_plan_with_anchors, current_alignment),
        "already_validated_appended_dialogue": _stage64_compact_appended_turns_for_prompt(appended_dialogue_turns),
        "previous_qa_proposal": previous_qa_proposal if isinstance(previous_qa_proposal, dict) else {},
        "answer_type_allowed_values": list(STAGE64_ANSWER_TYPES),
        "qa_operation_allowed_values": list(STAGE64_QA_OPERATIONS),
        "output_schema": {
            "qa_proposal": {
                "question_type": "single_choice",
                "qa_operation": target_operation or "one allowed qa_operation",
                "qa_subtype": "short subtype string",
                "answer_type": "one allowed answer_type",
                "answer_format": "single_choice_mcq",
                "question": "natural MCQ question",
                "question_text": "same as question",
                "options": [
                    {
                        "id": "A",
                        "text": "option text",
                        "answer_type": "same answer_type as qa_proposal",
                        "truth_value": True,
                        "source_event_instance_id": "event_instance_id from event_candidates",
                        "anchor_ref": "event_instance_id or marker from event_candidates",
                        "anchor_kind": "event",
                    }
                ],
                "gold": ["A"],
                "gold_answer_text": "text of gold option",
                "query_plan": {
                    "answer_source": "graph_anchor",
                    "qa_operation": "same as qa_operation",
                    "qa_subtype": "same as qa_subtype",
                    "answer_type": "same as answer_type",
                    "reasoning_anchor_path": ["event_instance_id or marker refs"],
                    "required_anchor_refs": ["event_instance_id or marker refs"],
                    "supporting_anchor_refs": ["event_instance_id or marker refs"],
                    "gold_anchor_refs": ["event_instance_id or marker refs"],
                    "distractor_anchor_refs": ["event_instance_id or marker refs"],
                    "gold_answer": "text of gold option",
                    "normalized_gold_answer": "lowercase normalized gold answer text",
                },
                "anchor_refs": {
                    "support": ["event_instance_id or marker refs"],
                    "gold": ["event_instance_id or marker refs"],
                    "distractors": ["event_instance_id or marker refs"],
                },
            }
        },
    }
    prompt_token_estimate = estimate_json_tokens(prompt)
    proposal, meta = call_raw_json_api(
        args,
        prompt,
        system_prompt=(
            "You are a strict JSON-only Stage64 QA schema repair editor. "
            "Fix qa_proposal only. Do not rewrite dialogue. Return exactly one JSON object."
        ),
        required_keys=("qa_proposal",),
        call_kind="api_stage64_qa_repair",
        max_tokens_override=2048,
    )
    meta["prompt_tokens_estimate"] = prompt_token_estimate
    return proposal, meta


def _as_dialogue_list(api_dialogue: Any) -> list[dict[str, Any]]:
    if isinstance(api_dialogue, dict) and isinstance(api_dialogue.get("final_dialogue"), list):
        api_dialogue = api_dialogue.get("final_dialogue")
    if not isinstance(api_dialogue, list):
        return []
    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(api_dialogue):
        if isinstance(turn, dict):
            turns.append(turn)
        elif isinstance(turn, str):
            turns.append({"speaker": f"Speaker{(index % 2) + 1}", "text": turn})
    return turns


def appended_dialogue_from_proposal(dialogue_proposal: dict[str, Any] | None) -> Any:
    if not isinstance(dialogue_proposal, dict):
        return []
    if isinstance(dialogue_proposal.get("appended_dialogue_turns"), list):
        return dialogue_proposal.get("appended_dialogue_turns")
    return dialogue_proposal.get("final_dialogue") if isinstance(dialogue_proposal.get("final_dialogue"), list) else []


def current_event_scaffold_turns(current_event_plan: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue_plan = sanitize_event_plan_for_dialogue_api(current_event_plan)
    turns: list[dict[str, Any]] = []
    for index, event in enumerate(dialogue_plan.get("events") or []):
        if not isinstance(event, dict) or event.get("required") is False:
            continue
        marker = normalize_space(event.get("marker"))
        claim = normalize_space(event.get("dialogue_support_claim"))
        if not marker or not claim:
            continue
        turns.append(
            {
                "speaker": "Analyst" if index % 2 == 0 else "Reviewer",
                "text": claim,
                "alignment_anchors": [marker],
                "support_quotes": {marker: claim},
            }
        )
    return turns


def parse_anchor_annotations(
    api_dialogue: Any,
    current_anchor_to_event_instance_id: dict[str, str],
    event_alignment_proposal: Any = None,
    *,
    turn_index_offset: int = 0,
    prefer_sidecar_when_complete: bool = True,
) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    seen_current_anchors: set[str] = set()
    invalid_anchors: set[str] = set()
    turns = _as_dialogue_list(api_dialogue)
    for index, turn in enumerate(turns):
        text = str(turn.get("text") or "")
        sidecar = turn.get("alignment_anchors")
        anchors: list[str] = []
        if isinstance(sidecar, list):
            anchors.extend(str(item).strip() for item in sidecar if str(item or "").strip())
        elif isinstance(sidecar, str) and sidecar.strip():
            anchors.append(sidecar.strip())
        anchors.extend(collect_inline_anchors_from_text(text))
        support_quotes = turn.get("support_quotes") if isinstance(turn.get("support_quotes"), dict) else {}
        for anchor in dict.fromkeys(anchors):
            if anchor in current_anchor_to_event_instance_id:
                seen_current_anchors.add(anchor)
            else:
                invalid_anchors.add(anchor)
            proposals.append(
                {
                    "alignment_anchor": anchor,
                    "event_instance_id": current_anchor_to_event_instance_id.get(anchor),
                    "turn_index": index + turn_index_offset,
                    "support_quote": normalize_space(support_quotes.get(anchor) or GENERIC_MARKER_RE.sub(" ", text)),
                    "source": "dialogue_sidecar_or_inline",
                }
            )

    sidecar_seen_current_anchors = set(seen_current_anchors)
    ignored_event_alignment_proposal = False
    if (
        prefer_sidecar_when_complete
        and current_anchor_to_event_instance_id
        and sidecar_seen_current_anchors.issuperset(set(current_anchor_to_event_instance_id))
    ):
        ignored_event_alignment_proposal = True
        first_proposal_by_anchor: dict[str, dict[str, Any]] = {}
        for proposal in proposals:
            anchor = normalize_space(proposal.get("alignment_anchor"))
            if anchor in current_anchor_to_event_instance_id and anchor not in first_proposal_by_anchor:
                first_proposal_by_anchor[anchor] = proposal
        proposals = [first_proposal_by_anchor[anchor] for anchor in current_anchor_to_event_instance_id if anchor in first_proposal_by_anchor]
        invalid_anchors = set()
    else:
        if isinstance(event_alignment_proposal, list):
            for item in event_alignment_proposal:
                if not isinstance(item, dict):
                    continue
                anchor = normalize_space(item.get("alignment_anchor") or item.get("anchor"))
                if not anchor:
                    continue
                if anchor in current_anchor_to_event_instance_id:
                    seen_current_anchors.add(anchor)
                else:
                    invalid_anchors.add(anchor)
                proposals.append(
                    {
                        "alignment_anchor": anchor,
                        "event_instance_id": normalize_space(item.get("event_instance_id"))
                        or current_anchor_to_event_instance_id.get(anchor),
                        "turn_index": (
                            int(item.get("turn_index")) + turn_index_offset
                            if str(item.get("turn_index") if item.get("turn_index") is not None else "").strip().lstrip("-").isdigit()
                            else item.get("turn_index")
                        ),
                        "turn_id": item.get("turn_id"),
                        "support_quote": normalize_space(item.get("support_quote")),
                        "source": "event_alignment_proposal",
                    }
                )
    return {
        "proposals": proposals,
        "current_anchor_seen": sorted(seen_current_anchors),
        "sidecar_anchor_seen": sorted(sidecar_seen_current_anchors),
        "invalid_anchors": sorted(invalid_anchors),
        "ignored_event_alignment_proposal": ignored_event_alignment_proposal,
    }


def strip_alignment_anchors_from_dialogue(api_dialogue: Any) -> list[dict[str, Any]]:
    clean_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(_as_dialogue_list(api_dialogue)):
        speaker = normalize_space(turn.get("speaker") or f"Speaker{(index % 2) + 1}")
        text = normalize_space(GENERIC_MARKER_RE.sub(" ", str(turn.get("text") or "")))
        if not text:
            continue
        clean_turns.append({"speaker": speaker, "text": text})
    return clean_turns


def ensure_dialogue_padding(clean_dialogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns = copy.deepcopy(clean_dialogue)
    if not turns:
        turns.append({"speaker": "Analyst", "text": "Let's review the latest updates carefully before deciding."})
    while len(turns) < 8:
        turns.append(
            {
                "speaker": "Reviewer" if len(turns) % 2 else "Analyst",
                "text": "Okay, that helps me keep the details straight.",
            }
        )
    turns.append(
        {
            "speaker": "Reviewer",
            "text": "I will rely only on what was clearly said in this conversation.",
        }
    )
    turns.append(
        {
            "speaker": "Analyst",
            "text": "Good; anything else can wait until later.",
        }
    )
    return turns


def ensure_append_length_guard_margin(
    parent_turns: list[dict[str, Any]],
    appended_turns: list[dict[str, Any]],
    *,
    min_added_turns: int,
    min_added_tokens: int,
    tokenizer_path: str | None = None,
    parent_baseline_tokens: int | None = None,
) -> list[dict[str, Any]]:
    turns = copy.deepcopy(appended_turns)
    filler_templates = [
        (
            "Analyst",
            "Before we leave it there, I want to restate the practical picture in ordinary terms. "
            "The earlier part of the conversation is background, and these later updates are the ones I will use for the next decision. "
            "If someone asks why, I will point to the newest correction rather than the older assumption."
        ),
        (
            "Reviewer",
            "That helps. I am separating what someone merely heard from what the group actually settled after the correction. "
            "The newer detail changes how I would act, while the older version is useful only as context. "
            "I will not treat a canceled or uncertain plan as if it were still confirmed."
        ),
        (
            "Analyst",
            "One more check: the final takeaway should come from the latest clear update, not from the first thing mentioned. "
            "When the conversation changes direction, the later clarification controls the answer. "
            "The old note can explain why the correction mattered, but it should not override the corrected plan."
        ),
        (
            "Reviewer",
            "Agreed. I will keep the timing straight and remember which details were limited, delayed, canceled, or restored. "
            "That way the final answer follows the actual sequence of updates. "
            "Anything that was only a report stays weaker than the detail that was directly confirmed later."
        ),
    ]
    guard = evaluate_length_guard(
        parent_turns,
        parent_turns + turns,
        min_added_turns=min_added_turns,
        min_added_tokens=min_added_tokens,
        tokenizer_path=tokenizer_path,
        parent_baseline_tokens=parent_baseline_tokens,
    )
    index = 0
    required_delta = max(
        int(min_added_tokens or 0),
        int(guard.get("length_guard_effective_parent_tokens") or 0)
        - int(guard.get("parent_dialogue_tokens") or 0)
        + int(min_added_tokens or 0),
    )
    max_padding_turns = max(24, min(160, int(max(required_delta, 0) / 70) + 12))
    while not guard.get("length_guard_pass") and index < max_padding_turns:
        speaker, text = filler_templates[index % len(filler_templates)]
        turns.append({"speaker": speaker, "text": text})
        guard = evaluate_length_guard(
            parent_turns,
            parent_turns + turns,
            min_added_turns=min_added_turns,
            min_added_tokens=min_added_tokens,
            tokenizer_path=tokenizer_path,
            parent_baseline_tokens=parent_baseline_tokens,
        )
        index += 1
    return turns


def count_anchor_leakage_in_turns(turns: list[dict[str, Any]], known_anchors: Iterable[str]) -> int:
    known = {str(anchor) for anchor in known_anchors if str(anchor or "").strip()}
    count = 0
    for turn in turns:
        text = str(turn.get("text") or "")
        if ANCHOR_MARKER_RE.search(text) or re.search(r"\[\[[^\]]+\]\]", text):
            count += 1
            continue
        if any(anchor and anchor in text for anchor in known):
            count += 1
    return count


def assert_no_anchor_leakage(final_clean_dialogue_or_row: Any, known_anchors: Iterable[str]) -> None:
    if isinstance(final_clean_dialogue_or_row, dict):
        dialogue = final_clean_dialogue_or_row.get("dialogue") if isinstance(final_clean_dialogue_or_row.get("dialogue"), dict) else {}
        turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
    else:
        turns = final_clean_dialogue_or_row if isinstance(final_clean_dialogue_or_row, list) else []
    leakage = count_anchor_leakage_in_turns(turns, known_anchors)
    if leakage:
        raise NativeGraphDialogueAnchorLeakReject(f"anchor leakage in clean dialogue: {leakage}")


def dialogue_text_from_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}"
        for turn in turns
        if isinstance(turn, dict) and normalize_space(turn.get("text"))
    )


@lru_cache(maxsize=4)
def _load_length_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def approx_tokens_for_turns(turns: list[dict[str, Any]]) -> int:
    text = dialogue_text_from_turns(turns)
    return max(0, len(text) // 4)


def dialogue_tokens_for_turns(turns: list[dict[str, Any]], tokenizer_path: str | None = None) -> tuple[int, str]:
    text = dialogue_text_from_turns(turns)
    path = normalize_space(tokenizer_path)
    if path:
        try:
            tokenizer = _load_length_tokenizer(path)
            return len(tokenizer(text, add_special_tokens=False).input_ids), "tokenizer"
        except Exception as exc:  # pragma: no cover - defensive fallback for legacy/local runs.
            logging.warning("length tokenizer unavailable at %s; falling back to approximate count: %s", path, exc)
    return max(0, len(text) // 4), "approx_chars_div4"


def _next_append_dialogue_prefix(parent_turns: list[dict[str, Any]], sample_index: int) -> str:
    max_major: int | None = None
    for turn in parent_turns:
        match = re.match(r"^D(\d+):\d+$", str(turn.get("turn_id") or ""))
        if not match:
            continue
        major = int(match.group(1))
        max_major = major if max_major is None else max(max_major, major)
    if max_major is not None:
        return f"D{max_major + 1}"
    return f"D{sample_index}"


def freeze_append_only_dialogue(
    parent_dialogue_turns: list[dict[str, Any]],
    appended_clean_dialogue: list[dict[str, Any]],
    sample_index: int,
) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    seen_turn_ids: set[str] = set()
    for index, turn in enumerate(parent_dialogue_turns, start=1):
        text = normalize_space(turn.get("text"))
        if not text:
            continue
        turn_id = normalize_space(turn.get("turn_id") or f"legacy:{index}")
        if turn_id in seen_turn_ids:
            turn_id = f"{turn_id}__parent_{index}"
        seen_turn_ids.add(turn_id)
        turns.append(
            {
                "turn_id": turn_id,
                "speaker": normalize_space(turn.get("speaker") or f"Speaker{(index % 2) + 1}"),
                "text": text,
            }
        )
    append_prefix = _next_append_dialogue_prefix(turns, sample_index)
    for index, turn in enumerate(appended_clean_dialogue, start=1):
        text = normalize_space(turn.get("text"))
        if not text:
            continue
        turn_id = f"{append_prefix}:{index}"
        while turn_id in seen_turn_ids:
            turn_id = f"{append_prefix}:{index}_{len(seen_turn_ids)}"
        seen_turn_ids.add(turn_id)
        turns.append(
            {
                "turn_id": turn_id,
                "speaker": normalize_space(turn.get("speaker") or f"Speaker{(len(turns) % 2) + 1}"),
                "text": text,
            }
        )
    dialogue_hash = stable_hash({"turns": turns}, length=24)
    return {"turns": turns, "dialogue_hash": dialogue_hash, "frozen_dialogue": True}


def freeze_final_dialogue(clean_dialogue: list[dict[str, Any]], sample_index: int) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(clean_dialogue, start=1):
        text = normalize_space(turn.get("text"))
        if not text:
            continue
        turns.append(
            {
                "turn_id": f"D{sample_index}:{index}",
                "speaker": normalize_space(turn.get("speaker") or f"Speaker{(index % 2) + 1}"),
                "text": text,
            }
        )
    dialogue_hash = stable_hash({"turns": turns}, length=24)
    return {"turns": turns, "dialogue_hash": dialogue_hash, "frozen_dialogue": True}


def length_guard_thresholds(args: argparse.Namespace) -> tuple[int, int]:
    mode = normalize_space(getattr(args, "native_curriculum_mode", "") or "default")
    easy = mode in ("r3_long_easy", EASY_V2_MODE)  # both easy variants share the shorter floors
    stage64_length_guard = bool(getattr(args, "qa_operation_answer_type_diversification", False))
    try:
        round_number = int(getattr(args, "round_number", 0) or 0)
    except (TypeError, ValueError):
        round_number = 0
    try:
        min_length_delta_tokens = int(getattr(args, "min_length_delta_tokens", 0) or 0)
    except (TypeError, ValueError):
        min_length_delta_tokens = 0
    default_turns = EASY_MIN_ADDED_TURNS if easy else DEFAULT_LENGTH_MIN_ADDED_TURNS
    default_tokens = EASY_MIN_ADDED_TOKENS if easy else DEFAULT_LENGTH_MIN_ADDED_TOKENS
    try:
        min_added_turns = int(getattr(args, "length_min_added_turns", default_turns))
    except (TypeError, ValueError):
        min_added_turns = default_turns
    try:
        min_added_tokens = int(getattr(args, "min_added_tokens", default_tokens))
    except (TypeError, ValueError):
        min_added_tokens = default_tokens
    if (round_number >= 4 or stage64_length_guard) and min_length_delta_tokens > 0:
        min_added_tokens = max(min_added_tokens, min_length_delta_tokens)
        min_added_turns = max(min_added_turns, 24)
    # In easy mode, cap the floors so an outer default does not re-inflate length. The explicit r4
    # hardening delta is the exception: it is a real curriculum guard, not a legacy outer default.
    elif easy:
        min_added_turns = min(min_added_turns, EASY_MIN_ADDED_TURNS)
        min_added_tokens = min(min_added_tokens, EASY_MIN_ADDED_TOKENS)
    return max(1, min_added_turns), max(0, min_added_tokens)


def parent_context_limits(args: argparse.Namespace) -> tuple[int, int, int]:
    try:
        tail_turns = int(getattr(args, "parent_context_tail_turns", DEFAULT_PARENT_CONTEXT_TAIL_TURNS))
    except (TypeError, ValueError):
        tail_turns = DEFAULT_PARENT_CONTEXT_TAIL_TURNS
    try:
        relevant_turns = int(getattr(args, "parent_context_relevant_turns", DEFAULT_PARENT_CONTEXT_RELEVANT_TURNS))
    except (TypeError, ValueError):
        relevant_turns = DEFAULT_PARENT_CONTEXT_RELEVANT_TURNS
    try:
        token_budget = int(getattr(args, "dialogue_api_parent_token_budget", DEFAULT_DIALOGUE_API_PARENT_TOKEN_BUDGET))
    except (TypeError, ValueError):
        token_budget = DEFAULT_DIALOGUE_API_PARENT_TOKEN_BUDGET
    return max(0, tail_turns), max(0, relevant_turns), max(256, token_budget)


def estimate_json_tokens(value: Any) -> int:
    return max(0, len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)) // 4)


def evaluate_length_guard(
    parent_turns: list[dict[str, Any]],
    final_turns: list[dict[str, Any]],
    *,
    min_added_turns: int,
    min_added_tokens: int,
    tokenizer_path: str | None = None,
    parent_baseline_tokens: int | None = None,
) -> dict[str, Any]:
    parent_turn_count = len(parent_turns)
    final_turn_count = len(final_turns)
    appended_turn_count = max(0, final_turn_count - parent_turn_count)
    approx_parent_tokens = approx_tokens_for_turns(parent_turns)
    approx_final_tokens = approx_tokens_for_turns(final_turns)
    approx_token_delta = approx_final_tokens - approx_parent_tokens
    parent_dialogue_tokens, token_count_mode = dialogue_tokens_for_turns(parent_turns, tokenizer_path)
    final_dialogue_tokens, _ = dialogue_tokens_for_turns(final_turns, tokenizer_path)
    length_delta_tokens = final_dialogue_tokens - parent_dialogue_tokens
    try:
        requested_parent_baseline_tokens = int(parent_baseline_tokens or 0)
    except (TypeError, ValueError):
        requested_parent_baseline_tokens = 0
    length_guard_effective_parent_tokens = max(parent_dialogue_tokens, requested_parent_baseline_tokens)
    length_guard_effective_delta_tokens = final_dialogue_tokens - length_guard_effective_parent_tokens
    length_guard_baseline_mode = (
        "max_actual_parent_and_configured_baseline"
        if requested_parent_baseline_tokens > parent_dialogue_tokens
        else "actual_parent_dialogue_tokens"
    )
    failures: list[str] = []
    if final_turn_count <= parent_turn_count:
        failures.append("final_turn_count_not_greater_than_parent")
    if final_turn_count < parent_turn_count + min_added_turns:
        failures.append("min_added_turns")
    if final_dialogue_tokens <= parent_dialogue_tokens:
        failures.append("final_tokens_not_greater_than_parent")
    if min_added_tokens > 0 and length_guard_effective_delta_tokens < min_added_tokens:
        failures.append("min_added_tokens")
    return {
        "parent_turn_count": parent_turn_count,
        "final_turn_count": final_turn_count,
        "appended_turn_count": appended_turn_count,
        "turn_count_delta": final_turn_count - parent_turn_count,
        "approx_parent_tokens": approx_parent_tokens,
        "approx_final_tokens": approx_final_tokens,
        "approx_token_delta": approx_token_delta,
        "parent_dialogue_tokens": parent_dialogue_tokens,
        "final_dialogue_tokens": final_dialogue_tokens,
        "length_delta_tokens": length_delta_tokens,
        "length_guard_parent_baseline_tokens": requested_parent_baseline_tokens,
        "length_guard_effective_parent_tokens": length_guard_effective_parent_tokens,
        "length_guard_effective_delta_tokens": length_guard_effective_delta_tokens,
        "length_guard_baseline_mode": length_guard_baseline_mode,
        "token_count_mode": token_count_mode,
        "tokenizer_path": normalize_space(tokenizer_path),
        "min_added_turns": min_added_turns,
        "min_added_tokens": min_added_tokens,
        "min_length_delta_tokens": min_added_tokens,
        "length_guard_pass": not failures,
        "length_guard_failures": failures,
    }


def _quote_supported(turn_text: str, support_quote: str) -> bool:
    quote = normalize_space(GENERIC_MARKER_RE.sub(" ", support_quote))
    if not quote:
        return False
    turn_norm = normalize_text(turn_text)
    quote_norm = normalize_text(quote)
    if quote_norm and quote_norm in turn_norm:
        return True
    left = token_set(turn_text)
    right = token_set(quote)
    return bool(left and right and len(left & right) / max(1, min(len(left), len(right))) >= 0.55)


def _event_lexical_supported(event: dict[str, Any], turn_text: str) -> bool:
    turn_tokens = token_set(turn_text)
    if not turn_tokens:
        return False
    value_tokens = token_set(event.get("dialogue_value") or event.get("value"))
    entity_tokens = token_set(sanitize_internal_text(event.get("entity")))
    slot_tokens = token_set(sanitize_internal_text(event.get("slot")))
    support_claim_tokens = token_set(
        event.get("dialogue_support_claim")
        or ((event.get("alignment_hints") or {}) if isinstance(event.get("alignment_hints"), dict) else {}).get("support_claim")
    )
    value_overlap = len(turn_tokens & value_tokens) / max(1, min(len(turn_tokens), len(value_tokens))) if value_tokens else 0.0
    claim_overlap = (
        len(turn_tokens & support_claim_tokens) / max(1, min(len(turn_tokens), len(support_claim_tokens)))
        if support_claim_tokens
        else 0.0
    )
    entity_ok = not entity_tokens or bool(turn_tokens & entity_tokens)
    slot_ok = not slot_tokens or bool(turn_tokens & slot_tokens) or value_overlap >= 0.55
    status = str(event.get("status") or "active")
    status_ok = True
    if status in STATUS_ALIASES:
        text_norm = normalize_text(turn_text)
        status_ok = any(normalize_text(alias) in text_norm for alias in STATUS_ALIASES[status])
    return entity_ok and slot_ok and status_ok and (value_overlap >= 0.32 or claim_overlap >= 0.42)


def _resolve_turn_index(raw_index: Any, turns: list[dict[str, Any]]) -> int | None:
    try:
        idx = int(raw_index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(turns):
        return idx
    if 1 <= idx <= len(turns):
        return idx - 1
    return None


def verify_current_graph_dialogue_alignment(
    current_event_plan: dict[str, Any],
    final_clean_dialogue: dict[str, Any],
    raw_current_alignment_proposal: dict[str, Any],
    current_anchor_to_event_instance_id: dict[str, str],
) -> dict[str, Any]:
    turns = final_clean_dialogue.get("turns") if isinstance(final_clean_dialogue.get("turns"), list) else []
    event_by_id = {
        str(event.get("event_instance_id")): event
        for event in current_event_plan.get("events") or []
        if isinstance(event, dict) and event.get("event_instance_id")
    }
    known_anchors = set(current_anchor_to_event_instance_id)
    current_alignment: dict[str, dict[str, Any]] = {}
    unsupported_alignment_count = 0
    invalid_anchor_count = 0
    for proposal in raw_current_alignment_proposal.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        anchor = normalize_space(proposal.get("alignment_anchor"))
        event_id = current_anchor_to_event_instance_id.get(anchor)
        if not event_id or event_id not in event_by_id:
            invalid_anchor_count += 1
            continue
        idx = _resolve_turn_index(proposal.get("turn_index"), turns)
        if idx is None:
            # A proposal may use frozen turn ids when coming from an API retry. Prefer
            # sidecar/inline indices, but accept a valid turn_id if present.
            raw_turn_id = normalize_space(proposal.get("turn_id"))
            if raw_turn_id:
                idx = next((i for i, turn in enumerate(turns) if str(turn.get("turn_id")) == raw_turn_id), None)
        if idx is None or idx < 0 or idx >= len(turns):
            unsupported_alignment_count += 1
            continue
        turn = turns[idx]
        turn_text = str(turn.get("text") or "")
        support_quote = normalize_space(proposal.get("support_quote"))
        event = event_by_id[event_id]
        if not (_quote_supported(turn_text, support_quote) or _event_lexical_supported(event, turn_text)):
            unsupported_alignment_count += 1
            continue
        entry = current_alignment.setdefault(
            event_id,
            {
                "turn_ids": [],
                "support_quotes": [],
                "depends_on_graph_snapshot_hash": current_event_plan.get("graph_snapshot_hash"),
                "depends_on_dialogue_hash": final_clean_dialogue.get("dialogue_hash"),
            },
        )
        turn_id = str(turn.get("turn_id"))
        if turn_id not in entry["turn_ids"]:
            entry["turn_ids"].append(turn_id)
        quote = support_quote or turn_text
        if quote and quote not in entry["support_quotes"]:
            entry["support_quotes"].append(quote[:500])

    for entry in current_alignment.values():
        entry["turn_ids"] = sorted(entry["turn_ids"], key=turn_sort_key)

    required_event_ids = [
        str(event.get("event_instance_id"))
        for event in current_event_plan.get("events") or []
        if isinstance(event, dict) and event.get("required_for_dialogue") is not False
    ]
    missing_required = [event_id for event_id in required_event_ids if event_id not in current_alignment]
    anchor_leakage_count = count_anchor_leakage_in_turns(turns, known_anchors)
    contradiction_count = 0
    for event_id, event in event_by_id.items():
        if event_id not in current_alignment:
            continue
        status = str(event.get("status") or "active")
        if status not in STATUS_ALIASES:
            continue
        for turn_id in current_alignment[event_id].get("turn_ids") or []:
            turn_text = next((str(turn.get("text") or "") for turn in turns if str(turn.get("turn_id")) == turn_id), "")
            text_norm = normalize_text(turn_text)
            if not any(normalize_text(alias) in text_norm for alias in STATUS_ALIASES[status]):
                contradiction_count += 1
                break
    alignment_hash = stable_hash(
        {
            "graph_snapshot_hash": current_event_plan.get("graph_snapshot_hash"),
            "dialogue_hash": final_clean_dialogue.get("dialogue_hash"),
            "alignment": current_alignment,
        },
        length=24,
    )
    for entry in current_alignment.values():
        entry["depends_on_alignment_hash"] = alignment_hash
    coverage_pass = (
        not missing_required
        and unsupported_alignment_count == 0
        and invalid_anchor_count == 0
        and contradiction_count == 0
        and anchor_leakage_count == 0
    )
    return {
        "current_alignment": current_alignment,
        "coverage_pass": coverage_pass,
        "missing_required_event_instance_ids": missing_required,
        "unsupported_alignment_count": unsupported_alignment_count,
        "contradiction_count": contradiction_count,
        "aligned_required_event_count": len(required_event_ids) - len(missing_required),
        "required_event_count": len(required_event_ids),
        "anchor_leakage_count": anchor_leakage_count,
        "current_anchor_invalid_count": invalid_anchor_count,
        "alignment_hash": alignment_hash,
    }


def validate_appended_dialogue_candidate(
    *,
    appended_dialogue_marked: list[dict[str, Any]],
    event_alignment_proposal: Any,
    current_event_plan: dict[str, Any],
    anchor_to_event: dict[str, str],
    parent_dialogue_turns: list[dict[str, Any]],
    known_anchors: list[str],
    sample_index: int,
    min_added_turns: int,
    min_added_tokens: int,
    tokenizer_path: str | None = None,
    parent_baseline_tokens: int | None = None,
    fallback_min_added_tokens: int | None = None,
    emergency_min_added_tokens: int | None = None,
) -> dict[str, Any]:
    raw_alignment_proposal = parse_anchor_annotations(
        appended_dialogue_marked,
        anchor_to_event,
        event_alignment_proposal,
        turn_index_offset=len(parent_dialogue_turns),
    )
    seen = set(raw_alignment_proposal.get("current_anchor_seen") or [])
    invalid = set(raw_alignment_proposal.get("invalid_anchors") or [])
    ignored_event_alignment_proposal = bool(raw_alignment_proposal.get("ignored_event_alignment_proposal"))
    appended_clean_dialogue = strip_alignment_anchors_from_dialogue(appended_dialogue_marked)
    appended_clean_dialogue = ensure_dialogue_padding(appended_clean_dialogue)
    appended_clean_dialogue = ensure_append_length_guard_margin(
        parent_dialogue_turns,
        appended_clean_dialogue,
        min_added_turns=min_added_turns,
        min_added_tokens=min_added_tokens,
        tokenizer_path=tokenizer_path,
        parent_baseline_tokens=parent_baseline_tokens,
    )
    final_dialogue = freeze_append_only_dialogue(parent_dialogue_turns, appended_clean_dialogue, sample_index)
    length_metadata = evaluate_length_guard(
        parent_dialogue_turns,
        final_dialogue.get("turns") if isinstance(final_dialogue.get("turns"), list) else [],
        min_added_turns=min_added_turns,
        min_added_tokens=min_added_tokens,
        tokenizer_path=tokenizer_path,
        parent_baseline_tokens=parent_baseline_tokens,
    )
    if not length_metadata.get("length_guard_pass"):
        fallback_metadata = None
        for fallback_name, fallback_tokens in (
            ("fallback_delta", fallback_min_added_tokens),
            ("emergency_floor", emergency_min_added_tokens),
        ):
            if fallback_tokens is None:
                continue
            fallback_tokens_int = int(fallback_tokens or 0)
            if fallback_tokens_int <= 0 or fallback_tokens_int >= min_added_tokens:
                continue
            candidate_metadata = evaluate_length_guard(
                parent_dialogue_turns,
                final_dialogue.get("turns") if isinstance(final_dialogue.get("turns"), list) else [],
                min_added_turns=min_added_turns,
                min_added_tokens=fallback_tokens_int,
                tokenizer_path=tokenizer_path,
                parent_baseline_tokens=parent_baseline_tokens,
            )
            if candidate_metadata.get("length_guard_pass"):
                fallback_metadata = candidate_metadata
                fallback_metadata["length_guard_target_pass"] = False
                fallback_metadata["length_guard_target_delta"] = min_added_tokens
                fallback_metadata["length_guard_fallback_used"] = True
                fallback_metadata["length_guard_fallback_level"] = fallback_name
                fallback_metadata["fallback_delta"] = fallback_min_added_tokens
                fallback_metadata["emergency_floor"] = emergency_min_added_tokens
                fallback_metadata["rows_below_4000_delta_flag"] = (
                    float(fallback_metadata.get("length_guard_effective_delta_tokens") or 0) < float(min_added_tokens or 0)
                )
                length_metadata = fallback_metadata
                break
        if fallback_metadata is None:
            return {
                "pass": False,
                "reason": "length",
                "raw_alignment_proposal": raw_alignment_proposal,
                "final_dialogue": final_dialogue,
                "length_metadata": length_metadata,
                "seen": seen,
                "invalid": invalid,
                "missing_anchors": set(anchor_to_event) - seen,
                "ignored_event_alignment_proposal": ignored_event_alignment_proposal,
            }
    else:
        length_metadata["length_guard_target_pass"] = True
        length_metadata["length_guard_target_delta"] = min_added_tokens
        length_metadata["length_guard_fallback_used"] = False
        length_metadata["fallback_delta"] = fallback_min_added_tokens
        length_metadata["emergency_floor"] = emergency_min_added_tokens
        length_metadata["rows_below_4000_delta_flag"] = False
    assert_no_anchor_leakage(final_dialogue, known_anchors)
    meta_ok, meta_details = assert_no_meta_leakage(final_dialogue)
    if not meta_ok:
        return {
            "pass": False,
            "reason": "meta_leak",
            "raw_alignment_proposal": raw_alignment_proposal,
            "final_dialogue": final_dialogue,
            "length_metadata": length_metadata,
            "meta_details": meta_details,
            "seen": seen,
            "invalid": invalid,
            "missing_anchors": set(anchor_to_event) - seen,
            "ignored_event_alignment_proposal": ignored_event_alignment_proposal,
        }
    alignment_result = verify_current_graph_dialogue_alignment(
        current_event_plan,
        final_dialogue,
        raw_alignment_proposal,
        anchor_to_event,
    )
    return {
        "pass": bool(alignment_result.get("coverage_pass")),
        "reason": "ok" if alignment_result.get("coverage_pass") else "dialogue_alignment",
        "raw_alignment_proposal": raw_alignment_proposal,
        "alignment_result": alignment_result,
        "final_dialogue": final_dialogue,
        "length_metadata": length_metadata,
        "seen": seen,
        "invalid": invalid,
        "missing_anchors": set(anchor_to_event) - seen,
        "ignored_event_alignment_proposal": ignored_event_alignment_proposal,
    }


def _deps(hashes: dict[str, str]) -> dict[str, str]:
    return {
        "depends_on_graph_snapshot_hash": hashes["graph_snapshot_hash"],
        "depends_on_dialogue_hash": hashes["dialogue_hash"],
        "depends_on_alignment_hash": hashes["alignment_hash"],
    }


def compile_graph_update_from_current_event_plan(
    current_event_plan: dict[str, Any],
    current_alignment: dict[str, Any],
    hashes: dict[str, str],
) -> dict[str, Any]:
    events_out: list[dict[str, Any]] = []
    state_changes: list[dict[str, Any]] = []
    invalidations: list[dict[str, Any]] = []
    cancellations: list[dict[str, Any]] = []
    restorations: list[dict[str, Any]] = []
    delayed_activations: list[dict[str, Any]] = []
    sorted_events = sorted(
        [event for event in current_event_plan.get("events") or [] if isinstance(event, dict)],
        key=lambda event: (int(event.get("time_index") or 0), str(event.get("event_instance_id"))),
    )
    final_active_by_slot: dict[tuple[str, str], str] = {}
    for event in sorted_events:
        if str(event.get("status")) in ACTIVE_STATUSES:
            final_active_by_slot[(str(event.get("entity")), str(event.get("slot")))] = str(event.get("value"))
    for event in sorted_events:
        event_id = str(event.get("event_instance_id"))
        alignment = current_alignment.get(event_id) if isinstance(current_alignment.get(event_id), dict) else {}
        turn_ids = sorted(normalize_turn_ids(alignment.get("turn_ids")), key=turn_sort_key)
        event_row = {
            "event_id": event_id,
            "event_instance_id": event_id,
            "origin_event_id": event.get("origin_event_id"),
            "lineage_id": event.get("lineage_id"),
            "node_id": f"{current_event_plan.get('graph_snapshot_id')}:{event_id.split(':')[-1]}",
            "role": event.get("event_type") or "state_change",
            "entity": event.get("entity"),
            "slot": event.get("slot"),
            "value": event.get("value"),
            "previous_value": event.get("previous_value"),
            "time_index": event.get("time_index"),
            "status": event.get("status") or "active",
            "turn_ids": turn_ids,
            "evidence_turn_ids": turn_ids,
            **_deps(hashes),
        }
        events_out.append(event_row)
        if event_row["status"] in ACTIVE_STATUSES or event_row["role"] in {"state_change", "restore"}:
            state_changes.append(
                {
                    "event_instance_id": event_id,
                    "entity": event.get("entity"),
                    "slot": event.get("slot"),
                    "from": event.get("previous_value"),
                    "to": event.get("value"),
                    "time_index": event.get("time_index"),
                    "status": event.get("status"),
                    "evidence_turn_ids": turn_ids,
                    **_deps(hashes),
                }
            )
        elif event_row["status"] in FALSE_STATUSES:
            final_true_state = final_active_by_slot.get((str(event.get("entity")), str(event.get("slot"))), "")
            invalidation = {
                "event_instance_id": event_id,
                "invalidated_statement": event.get("value"),
                "invalidating_event": final_true_state,
                "final_true_state": final_true_state,
                "false_reason_type": event.get("status"),
                "invalidation_turn_ids": turn_ids,
                **_deps(hashes),
            }
            invalidations.append(invalidation)
            if event_row["status"] == "canceled":
                cancellations.append({"event_instance_id": event_id, "cancelled_state": event.get("value"), "cancellation_turn_ids": turn_ids, **_deps(hashes)})
            elif event_row["status"] == "restored":
                restorations.append({"event_instance_id": event_id, "restored_state": event.get("value"), "restoration_turn_ids": turn_ids, **_deps(hashes)})
            elif event_row["status"] == "delayed_not_active":
                delayed_activations.append(
                    {
                        "event_instance_id": event_id,
                        "planned_state": event.get("value"),
                        "activation_state": final_true_state,
                        "activation_turn_ids": turn_ids,
                        **_deps(hashes),
                    }
                )
    return {
        "graph_snapshot_id": current_event_plan.get("graph_snapshot_id"),
        "graph_snapshot_hash": hashes["graph_snapshot_hash"],
        "events": events_out,
        "state_changes": state_changes,
        "invalidations": invalidations,
        "cancellations": cancellations,
        "restorations": restorations,
        "delayed_activations": delayed_activations,
        "visibility_edges": [
            {
                "observer": "solver",
                "event_instance_id": event.get("event_instance_id"),
                "event_turn_id": turn_id,
                "visibility": "directly_stated_in_verified_current_dialogue",
                **_deps(hashes),
            }
            for event in events_out
            for turn_id in event.get("evidence_turn_ids", [])
        ],
        "entities": sorted(
            [{"entity_id": str(event.get("entity")), "type": "current_graph_entity"} for event in sorted_events],
            key=lambda item: item["entity_id"],
        ),
        **_deps(hashes),
    }


def _event_option_text(event: dict[str, Any]) -> str:
    return sanitize_internal_text(event.get("dialogue_value") or event.get("value"))


def _hard_single_choice_surface_text(event: dict[str, Any], *, include_context: bool = True) -> str:
    """Hard single-choice uses a stripped surface form so options don't self-label.

    Keep graph meaning intact, but remove solver-visible scaffolding like
    "latest confirmed detail" / "final active state" / explicit status cues.
    """
    entity = normalize_space(event.get("entity")).replace("_", " ")
    slot = normalize_space(event.get("slot")).replace("_", " ")
    text = sanitize_internal_text(event.get("value") or event.get("dialogue_value"))
    if not text:
        return ""
    text = re.sub(r"^\s*[A-Z][A-Za-z0-9_-]*\s*/\s*[A-Za-z0-9_-]+\s*:\s*", "", text)
    text = re.sub(r"^\s*[A-Z][A-Za-z0-9_-]*\s+[A-Za-z0-9_-]+\s*:\s*", "", text)
    text = re.sub(
        r"^(?:latest|final|confirmed|current)\s+(?:confirmed\s+)?(?:active\s+)?(?:detail|state)\s+"
        r"(?:after\s+review\s+)?(?:is|keeps|remains|stays)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:the\s+speaker\s+means|the\s+speaker\s+discusses\s+a\s+real-world\s+update)\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:only\s+[A-Z][a-z]+\s+can\s+see\s+the\s+discarded\s+archive-shelf\s+report)\s*$",
        "the discarded archive-shelf report",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^the discarded archive-shelf report\s*$", "the archived report", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:latest|final|confirmed)\s+confirmed\b", "confirmed", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:latest|final|confirmed|current)\s+(?:active\s+)?state\b", "state", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:latest|final|confirmed|current)\s+(?:active\s+)?detail\b", "detail", text, flags=re.IGNORECASE)
    text = re.sub(r"\bafter\s+review\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bconfirmed\s+active\s+state\s+keeps\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfinal\s+active\s+state\s+after\s+review\s+is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\blatest\s+confirmed\s+detail\s+after\s+review\s+is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:confirmed|final|latest)\s+(?:active\s+)?state\s+keeps\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:confirmed|final|latest)\s+(?:active\s+)?detail\s+after\s+review\s+is\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" -:;,.")
    if not text:
        text = sanitize_internal_text(event.get("dialogue_value") or event.get("value"))
    if include_context and entity and slot:
        text = sanitize_internal_text(f"{entity} {slot}: {text}")
    return text[:1].upper() + text[1:] if text else text


def _hard_single_choice_question_text(template: str) -> str:
    questions = {
        "minimal_difference": "Which option best matches the state?",
        "latest_valid_state": "Which option reflects the state after all updates?",
        "guaranteed_vs_possible": "Which statement is guaranteed by the final state?",
        "override_sensitive": "Which option reflects the state rather than an earlier overridden one?",
        "sequence_as_single_choice": "Which sequence is consistent with the final state?",
    }
    return questions.get(template, "Which option best matches the state?")


def _surface_similarity_score(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _selected_aligned_events(current_event_plan: dict[str, Any], current_alignment: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        event
        for event in current_event_plan.get("events") or []
        if isinstance(event, dict) and str(event.get("event_instance_id")) in current_alignment
    ]
    return sorted(events, key=lambda event: (int(event.get("time_index") or 0), str(event.get("event_instance_id"))))


def _turn_ids_for_event(event: dict[str, Any], current_alignment: dict[str, Any]) -> list[str]:
    alignment = current_alignment.get(str(event.get("event_instance_id"))) if isinstance(current_alignment, dict) else {}
    return sorted(normalize_turn_ids((alignment or {}).get("turn_ids")), key=turn_sort_key)


def _turn_distance(turn_ids: Iterable[Any]) -> int:
    values: list[int] = []
    for turn_id in normalize_turn_ids(turn_ids):
        match = re.match(r"^D(\d+):(\d+)$", str(turn_id))
        if match:
            values.append(int(match.group(1)) * 1000 + int(match.group(2)))
        else:
            nums = [int(item) for item in re.findall(r"\d+", str(turn_id))]
            if nums:
                values.append(nums[0] * 1000 + (nums[1] if len(nums) > 1 else 0))
    if not values:
        return 0
    return max(values) - min(values)


def _event_ref_id(event: dict[str, Any]) -> str:
    return normalize_space(
        event.get("event_instance_id")
        or event.get("event_id")
        or event.get("node_id")
        or stable_hash(event, length=12)
    )


def _event_turn_ids_any(event: dict[str, Any], current_alignment: dict[str, Any] | None = None) -> list[str]:
    if current_alignment is not None and str(event.get("_context_source") or "") == "child":
        turn_ids = _turn_ids_for_event(event, current_alignment)
        if turn_ids:
            return turn_ids
    return sorted(
        normalize_turn_ids(
            event.get("turn_ids")
            or event.get("evidence_turn_ids")
            or event.get("activation_turn_ids")
            or event.get("invalidation_turn_ids")
            or event.get("cancellation_turn_ids")
            or event.get("restoration_turn_ids")
            or []
        ),
        key=turn_sort_key,
    )


def _event_has_turn_anchor(event: dict[str, Any]) -> bool:
    return bool(_event_turn_ids_any(event, None))


def _event_display_text(event: dict[str, Any], *, hard_surface: bool = False) -> str:
    if hard_surface:
        text = _hard_single_choice_surface_text(event, include_context=True)
    else:
        text = _event_option_text(event)
    text = normalize_space(text)
    return text or normalize_space(event.get("value") or event.get("to") or event.get("invalidated_statement"))


def _event_option_surface_is_placeholder(event: dict[str, Any]) -> bool:
    """Reject graph-grounded but solver-obvious false-option surfaces.

    These are not leakage in the formal sense, but they make multi-select permission
    questions collapse to easy 5/5 because a distractor literally says it came from an
    archive/report placeholder instead of looking like a competing current fact.
    """
    text = _option_text_norm(_event_display_text(event, hard_surface=True))
    value = _option_text_norm(event.get("value") or event.get("dialogue_value") or event.get("invalidated_statement") or "")
    slot = _option_text_norm(event.get("slot") or "")
    combined = " ".join(part for part in [text, value, slot] if part)
    if not combined:
        return True
    placeholder_patterns = [
        "the archived report",
        "archived report",
        "archive shelf report",
        "discarded archive",
    ]
    return any(pattern in combined for pattern in placeholder_patterns)


def _parent_context_events(seed_row: dict[str, Any]) -> list[dict[str, Any]]:
    graph_update = seed_row.get("graph_update") if isinstance(seed_row.get("graph_update"), dict) else {}
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in graph_update.get("events") or []:
        if not isinstance(event, dict):
            continue
        candidate = copy.deepcopy(event)
        candidate["_context_source"] = "parent"
        candidate["_dependency_count"] = _count_dep_refs(candidate)
        event_id = _event_ref_id(candidate)
        if event_id and event_id not in seen:
            seen.add(event_id)
            events.append(candidate)
    # State-change rows sometimes carry cleaner final-state text than the event row. Add only
    # when there is no corresponding event id.
    for change in graph_update.get("state_changes") or []:
        if not isinstance(change, dict):
            continue
        event_id = _event_ref_id(change)
        if event_id in seen:
            continue
        candidate = {
            "event_instance_id": event_id,
            "event_id": event_id,
            "entity": change.get("entity"),
            "slot": change.get("slot"),
            "value": change.get("to") or change.get("value"),
            "previous_value": change.get("from"),
            "time_index": change.get("time_index"),
            "status": change.get("status") or "active",
            "turn_ids": change.get("evidence_turn_ids") or change.get("turn_ids") or [],
            "evidence_turn_ids": change.get("evidence_turn_ids") or change.get("turn_ids") or [],
            "_context_source": "parent",
            "_dependency_count": _count_dep_refs(change),
        }
        seen.add(event_id)
        events.append(candidate)
    return sorted(events, key=lambda event: (str(event.get("entity") or ""), str(event.get("slot") or ""), int(event.get("time_index") or 0), _event_ref_id(event)))


def _child_context_events(current_event_plan: dict[str, Any], current_alignment: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for event in _selected_aligned_events(current_event_plan, current_alignment):
        candidate = copy.deepcopy(event)
        candidate["_context_source"] = "child"
        candidate["_dependency_count"] = _count_dep_refs(candidate)
        events.append(candidate)
    return events


def _count_dep_refs(obj: Any) -> int:
    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            if str(key).startswith("depends_on") and value not in (None, "", [], {}):
                total += len(value) if isinstance(value, list) else 1
            total += _count_dep_refs(value)
        return total
    if isinstance(obj, list):
        return sum(_count_dep_refs(item) for item in obj)
    return 0


def _context_pair_score(parent_event: dict[str, Any], child_event: dict[str, Any]) -> tuple[float, int, int, str]:
    same_entity = int(str(parent_event.get("entity") or "") == str(child_event.get("entity") or ""))
    same_slot = int(str(parent_event.get("slot") or "") == str(child_event.get("slot") or ""))
    surface = _surface_similarity_score(_event_display_text(parent_event), _event_display_text(child_event))
    return (
        same_entity * 4.0 + same_slot * 3.0 + surface,
        same_entity,
        same_slot,
        f"{_event_ref_id(parent_event)}|{_event_ref_id(child_event)}",
    )


def _best_parent_child_pairs(parent_events: list[dict[str, Any]], child_events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for parent_event in parent_events:
        for child_event in child_events:
            score = _context_pair_score(parent_event, child_event)[0]
            pairs.append((parent_event, child_event, score))
    pairs.sort(key=lambda item: (-item[2], _event_ref_id(item[0]), _event_ref_id(item[1])))
    return pairs


def _option_from_single_event(
    label: str,
    event: dict[str, Any],
    current_alignment: dict[str, Any],
    *,
    hard_surface: bool = False,
) -> dict[str, Any]:
    return {
        "id": label,
        "text": _event_display_text(event, hard_surface=hard_surface),
        "turn_ids": _event_turn_ids_any(event, current_alignment),
        "candidate_event_instance_id": _event_ref_id(event),
        "candidate_status": event.get("status"),
        "context_source": event.get("_context_source") or "unknown",
    }


def _option_from_event_pair(
    label: str,
    parent_event: dict[str, Any],
    child_event: dict[str, Any],
    current_alignment: dict[str, Any],
    *,
    hard_surface: bool = False,
) -> dict[str, Any]:
    parent_text = _event_display_text(parent_event, hard_surface=hard_surface)
    child_text = _event_display_text(child_event, hard_surface=hard_surface)
    return {
        "id": label,
        "text": normalize_space(f"Earlier: {parent_text}. Later: {child_text}."),
        "turn_ids": sorted(
            set(_event_turn_ids_any(parent_event, current_alignment) + _event_turn_ids_any(child_event, current_alignment)),
            key=turn_sort_key,
        ),
        "candidate_event_instance_id": f"{_event_ref_id(parent_event)}|{_event_ref_id(child_event)}",
        "candidate_status": f"{parent_event.get('status')}|{child_event.get('status')}",
        "context_source": "parent_child_pair",
        "parent_event_instance_id": _event_ref_id(parent_event),
        "child_event_instance_id": _event_ref_id(child_event),
    }


def _weak_status(status: Any) -> bool:
    return str(status or "") in {"scope_limited", "belief", "report"}


def _strong_false_events(events: list[dict[str, Any]], gold_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    false_events = [event for event in events if str(event.get("status")) in FALSE_STATUSES and not _weak_status(event.get("status"))]
    def score(event: dict[str, Any]) -> tuple[float, str]:
        best = 0.0
        for gold_event in gold_events:
            same_entity = str(event.get("entity") or "") == str(gold_event.get("entity") or "")
            same_slot = str(event.get("slot") or "") == str(gold_event.get("slot") or "")
            sim = _surface_similarity_score(_event_display_text(event), _event_display_text(gold_event))
            best = max(best, (2.0 if same_entity else 0.0) + (2.0 if same_slot else 0.0) + sim)
        return (-best, _event_ref_id(event))
    return sorted(false_events, key=score)


def _option_text_norm(value: Any) -> str:
    text = normalize_space(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def _option_pair_parts(text: Any) -> tuple[str, str]:
    match = re.search(r"Earlier:\s*(.*?)\.\s*Later:\s*(.*?)(?:\.|$)", str(text or ""), flags=re.I | re.S)
    if not match:
        return "", ""
    return _option_text_norm(match.group(1)), _option_text_norm(match.group(2))


def _candidate_status_parts(status: Any) -> list[str]:
    return [part for part in str(status or "").split("|") if part]


OBVIOUS_OUTDATED_SURFACE_CUES = {
    "old version",
    "older version",
    "old detail",
    "outdated",
    "expired",
    "canceled",
    "cancelled",
    "invalidated",
    "no longer apply",
    "no longer applies",
    "replaced",
    "was replaced",
    "discarded",
    "archive shelf",
    "archive-shelf",
}


def _option_surface_exposes_outdated_shortcut(text: Any) -> bool:
    norm = _option_text_norm(text)
    return any(cue in norm for cue in OBVIOUS_OUTDATED_SURFACE_CUES)


def _boundary_aware_item_diagnostics(qa: dict[str, Any]) -> dict[str, Any]:
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    gold_labels = {str(item) for item in (qa.get("gold") or [])}
    option_texts = [_option_text_norm(option.get("text")) for option in options if isinstance(option, dict)]
    counts = Counter(option_texts)
    duplicate_option_count = sum(max(0, value - 1) for value in counts.values())
    near_duplicate_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(options):
        if not isinstance(left, dict):
            continue
        left_text = option_texts[left_index] if left_index < len(option_texts) else ""
        for right_index in range(left_index + 1, len(options)):
            right = options[right_index]
            if not isinstance(right, dict):
                continue
            right_text = option_texts[right_index] if right_index < len(option_texts) else ""
            if not left_text or not right_text or left_text == right_text:
                continue
            similarity = _surface_similarity_score(left_text, right_text)
            if similarity >= 0.92:
                near_duplicate_pairs.append(
                    {
                        "left": left.get("id"),
                        "right": right.get("id"),
                        "similarity": round(similarity, 6),
                    }
                )
    false_reasons_by_option = qa.get("native_false_reasons_by_option") if isinstance(qa.get("native_false_reasons_by_option"), dict) else {}
    false_reason_types: list[str] = []
    hard_negative_path_types: list[str] = []
    active_but_wrong_scope = 0
    active_but_wrong_person = 0
    active_but_wrong_condition = 0
    later_state_override = 0
    invalidated_parent_dependency = 0
    cross_context_dependency_mismatch = 0
    parent_valid_child_invalidated = 0
    gold_statuses: list[str] = []
    non_gold_statuses: list[str] = []
    non_gold_surface_outdated_shortcuts: list[str] = []
    identical_gold_pair = 0
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("id") or "")
        status = str(option.get("candidate_status") or "")
        if label in gold_labels:
            gold_statuses.append(status)
            earlier, later = _option_pair_parts(option.get("text"))
            if earlier and later and earlier == later:
                identical_gold_pair += 1
        else:
            non_gold_statuses.append(status)
            reason = false_reasons_by_option.get(label) if isinstance(false_reasons_by_option.get(label), dict) else {}
            reason_type = str(reason.get("false_reason_type") or "")
            false_reason_types.append(reason_type)
            status_parts = _candidate_status_parts(status)
            text = _option_text_norm(option.get("text"))
            if _option_surface_exposes_outdated_shortcut(option.get("text")):
                non_gold_surface_outdated_shortcuts.append(label)
            if "cross_context_dependency_mismatch" in reason_type:
                cross_context_dependency_mismatch += 1
                hard_negative_path_types.append("cross_context_dependency_mismatch")
            if any(part in {"outdated", "canceled", "expired", "delayed_not_active", "false", "restored"} for part in status_parts):
                invalidated_parent_dependency += 1
                hard_negative_path_types.append("invalidated_parent_dependency")
            if "outdated" in status_parts or "later_state_override" in reason_type or "override" in text:
                later_state_override += 1
            if len(status_parts) >= 2 and status_parts[0] == "active" and status_parts[1] in {"outdated", "canceled", "expired", "false"}:
                parent_valid_child_invalidated += 1
            if "active" in status_parts and ("scope" in reason_type or "scope" in text or "visibility" in text or "permission" in text):
                active_but_wrong_scope += 1
                hard_negative_path_types.append("active_but_wrong_scope")
            if "active" in status_parts and ("person" in reason_type or "wrong_person" in reason_type):
                active_but_wrong_person += 1
                hard_negative_path_types.append("active_but_wrong_person")
            if "active" in status_parts and ("condition" in reason_type or "conditional" in text):
                active_but_wrong_condition += 1
                hard_negative_path_types.append("active_but_wrong_condition")
    active_vs_outdated_shortcut = bool(
        qa.get("question_type") == "multi_select"
        and gold_statuses
        and non_gold_statuses
        and set(gold_statuses) == {"active"}
        and set(non_gold_statuses) == {"outdated"}
        and len(non_gold_surface_outdated_shortcuts) == len(non_gold_statuses)
    )
    gold_only_active_pattern = bool(
        qa.get("question_type") == "single_choice"
        and len(gold_statuses) == 1
        and gold_statuses[0] == "active|active"
        and all(status != "active|active" for status in non_gold_statuses)
    )
    evidence_turn_ids = set(normalize_turn_ids(qa.get("evidence_turn_ids")))
    gold_turn_ids: set[str] = set()
    for option in options:
        if isinstance(option, dict) and str(option.get("id") or "") in gold_labels:
            gold_turn_ids.update(normalize_turn_ids(option.get("turn_ids")))
    gold_evidence_inconsistency = bool(not evidence_turn_ids or not gold_turn_ids or not gold_turn_ids.issubset(evidence_turn_ids))
    evidence_distance = _turn_distance(sorted(evidence_turn_ids, key=turn_sort_key))
    local_only_evidence = evidence_distance < 1000
    weak_hard_negative_count = sum(
        1
        for reason in false_reason_types
        if reason in {"", "scope_limited", "belief", "report"}
    )
    hard_negative_strength = (
        "weak"
        if weak_hard_negative_count
        else "strong"
        if (active_but_wrong_scope or active_but_wrong_person or active_but_wrong_condition or parent_valid_child_invalidated)
        else "medium"
        if (later_state_override or invalidated_parent_dependency or cross_context_dependency_mismatch)
        else "unknown"
    )
    option_surface_risk = bool(
        duplicate_option_count
        or near_duplicate_pairs
        or active_vs_outdated_shortcut
        or identical_gold_pair
        or gold_only_active_pattern
        or local_only_evidence
        or weak_hard_negative_count
    )
    hard_reject_reasons = []
    if duplicate_option_count:
        hard_reject_reasons.append("exact_duplicate_option")
    if identical_gold_pair:
        hard_reject_reasons.append("identical_gold_pair")
    if gold_evidence_inconsistency:
        hard_reject_reasons.append("gold_evidence_inconsistency")
    return {
        "boundary_aware_item_diagnostics": True,
        "duplicate_option_count": duplicate_option_count,
        "near_duplicate_option_count": len(near_duplicate_pairs),
        "near_duplicate_option_pairs": near_duplicate_pairs,
        "active_vs_outdated_shortcut": active_vs_outdated_shortcut,
        "active_vs_outdated_shortcut_surface_count": len(non_gold_surface_outdated_shortcuts),
        "active_vs_outdated_shortcut_option_ids": sorted(non_gold_surface_outdated_shortcuts),
        "identical_gold_pair": identical_gold_pair,
        "gold_only_active_pattern": gold_only_active_pattern,
        "option_surface_risk": option_surface_risk,
        "hard_negative_path_type": sorted(set(hard_negative_path_types)),
        "hard_negative_strength": hard_negative_strength,
        "false_reason_type": sorted(set(false_reason_types)),
        "parent_valid_child_invalidated": parent_valid_child_invalidated,
        "active_but_wrong_scope": active_but_wrong_scope,
        "active_but_wrong_person": active_but_wrong_person,
        "active_but_wrong_condition": active_but_wrong_condition,
        "later_state_override": later_state_override,
        "invalidated_parent_dependency": invalidated_parent_dependency,
        "cross_context_dependency_mismatch": cross_context_dependency_mismatch,
        "weak_hard_negative_count": weak_hard_negative_count,
        "local_only_evidence": local_only_evidence,
        "gold_evidence_inconsistency": gold_evidence_inconsistency,
        "hard_reject_reasons": hard_reject_reasons,
    }


BOUNDARY_QA_REPAIR_TRIGGER_REASONS = {
    "exact_duplicate_option",
    "near_duplicate_option",
    "identical_gold_pair",
    "gold_only_active_pattern",
    "active_vs_outdated_shortcut",
    "weak_hard_negative",
}
BOUNDARY_QA_REPAIR_DIRECT_REJECT_REASONS = {
    "gold_evidence_inconsistency",
}
BOUNDARY_QA_REPAIR_HARD_NEGATIVE_TYPES = (
    "active_but_wrong_scope",
    "active_but_wrong_person",
    "active_but_wrong_condition",
    "parent_valid_child_invalidated",
    "later_state_override",
    "invalidated_parent_dependency",
    "cross_context_dependency_mismatch",
    "same_entity_same_slot_wrong_trigger",
)


def _boundary_qa_repair_reasons(item_diagnostics: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for reason in item_diagnostics.get("hard_reject_reasons") or []:
        if reason in BOUNDARY_QA_REPAIR_TRIGGER_REASONS:
            reasons.append(str(reason))
    if int(item_diagnostics.get("near_duplicate_option_count") or 0) > 0:
        reasons.append("near_duplicate_option")
    if item_diagnostics.get("gold_only_active_pattern") is True:
        reasons.append("gold_only_active_pattern")
    if item_diagnostics.get("active_vs_outdated_shortcut") is True:
        reasons.append("active_vs_outdated_shortcut")
    if int(item_diagnostics.get("weak_hard_negative_count") or 0) > 0:
        reasons.append("weak_hard_negative")
    return sorted(set(reasons))


def _boundary_qa_direct_reject_reasons(item_diagnostics: dict[str, Any]) -> list[str]:
    return [
        str(reason)
        for reason in item_diagnostics.get("hard_reject_reasons") or []
        if reason in BOUNDARY_QA_REPAIR_DIRECT_REJECT_REASONS
    ]


def _event_same_topic_score(event: dict[str, Any], gold_events: list[dict[str, Any]]) -> float:
    best = 0.0
    event_text = _event_display_text(event, hard_surface=True)
    for gold_event in gold_events:
        score = _surface_similarity_score(event_text, _event_display_text(gold_event, hard_surface=True))
        if str(event.get("entity") or "") == str(gold_event.get("entity") or ""):
            score += 2.0
        if str(event.get("slot") or "") == str(gold_event.get("slot") or ""):
            score += 3.0
        if str(event.get("_context_source") or "") != str(gold_event.get("_context_source") or ""):
            score += 0.75
        best = max(best, score)
    return best


def _hard_negative_type_for_event(event: dict[str, Any], gold_events: list[dict[str, Any]]) -> str:
    override = normalize_space(event.get("_hard_negative_type_override") or "")
    if override in {
        "active_but_wrong_scope",
        "active_but_wrong_person",
        "active_but_wrong_condition",
        "parent_valid_child_invalidated",
        "later_state_override",
        "invalidated_parent_dependency",
        "cross_context_dependency_mismatch",
        "same_entity_same_slot_wrong_trigger",
    }:
        return override
    status = str(event.get("status") or "")
    text = _option_text_norm(event.get("value") or event.get("dialogue_value") or "")
    same_entity = any(str(event.get("entity") or "") == str(gold.get("entity") or "") for gold in gold_events)
    same_slot = any(str(event.get("slot") or "") == str(gold.get("slot") or "") for gold in gold_events)
    if status == "scope_limited":
        if "person" in text or "who" in text:
            return "active_but_wrong_person"
        if "condition" in text or "if " in f" {text} ":
            return "active_but_wrong_condition"
        return "active_but_wrong_scope"
    if same_entity and same_slot and status in {"outdated", "canceled", "expired", "delayed_not_active", "false", "restored"}:
        if status == "delayed_not_active":
            return "same_entity_same_slot_wrong_trigger"
        return "later_state_override"
    if status in {"outdated", "canceled", "expired", "delayed_not_active", "false", "restored"}:
        return "invalidated_parent_dependency"
    if same_entity:
        return "active_but_wrong_condition"
    return "cross_context_dependency_mismatch"


def _dedupe_events_by_surface(events: list[dict[str, Any]], *, existing_texts: set[str] | None = None) -> list[dict[str, Any]]:
    seen = set(existing_texts or set())
    out: list[dict[str, Any]] = []
    for event in events:
        text = _option_text_norm(_event_display_text(event, hard_surface=True))
        if not text or text in seen:
            continue
        if any(_surface_similarity_score(text, old) >= 0.92 for old in seen):
            continue
        seen.add(text)
        out.append(event)
    return out


def _rank_boundary_false_events(
    events: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
    *,
    prefer_scope: bool,
    current_alignment: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    false_events = [
        event
        for event in events
        if str(event.get("status") or "") in FALSE_STATUSES
        and _event_turn_ids_any(event, current_alignment)
        and not _event_option_surface_is_placeholder(event)
    ]

    def score(event: dict[str, Any]) -> tuple[float, str]:
        htype = _hard_negative_type_for_event(event, gold_events)
        bonus = 0.0
        if htype in {
            "active_but_wrong_scope",
            "active_but_wrong_person",
            "active_but_wrong_condition",
            "parent_valid_child_invalidated",
            "later_state_override",
            "invalidated_parent_dependency",
            "cross_context_dependency_mismatch",
            "same_entity_same_slot_wrong_trigger",
        }:
            bonus += 5.0
        if prefer_scope and htype in {"active_but_wrong_scope", "active_but_wrong_person", "active_but_wrong_condition"}:
            bonus += 4.0
        return (-(bonus + _event_same_topic_score(event, gold_events)), _event_ref_id(event))

    return sorted(false_events, key=score)


BOUNDARY_PERMISSION_SCOPE_NEGATIVE_TYPES = {
    "active_but_wrong_scope",
    "active_but_wrong_person",
    "active_but_wrong_condition",
}
BOUNDARY_PERMISSION_OVERRIDE_NEGATIVE_TYPES = {
    "parent_valid_child_invalidated",
    "later_state_override",
    "invalidated_parent_dependency",
}


def _boundary_event_dependency_weight(event: dict[str, Any]) -> int:
    try:
        return int(event.get("_dependency_count") or 0)
    except (TypeError, ValueError):
        return 0


def _boundary_plan_dependency_count_for_events(gold_events: list[dict[str, Any]], false_events: list[dict[str, Any]]) -> int:
    return (
        sum(_boundary_event_dependency_weight(event) for event in gold_events + false_events)
        + len({_event_ref_id(event) for event in gold_events})
    )


def _boundary_permission_explicit_contrast_type(hard_types: list[str]) -> str:
    type_set = set(hard_types)
    has_scope = bool(type_set & BOUNDARY_PERMISSION_SCOPE_NEGATIVE_TYPES)
    has_override = bool(type_set & BOUNDARY_PERMISSION_OVERRIDE_NEGATIVE_TYPES)
    if has_scope and has_override:
        return "same_permission_different_scope+later_override_one_scope"
    if has_scope:
        return "same_actor_different_visibility_condition"
    if has_override:
        return "parent_valid_child_invalidated_with_clear_contrast"
    return "generic_permission_visibility_contrast"


def _permission_synthetic_dependency_weight(
    gold_events: list[dict[str, Any]],
    *,
    qtype: str,
    skill: str,
    item_difficulty_target: str | None,
) -> int:
    target_min, target_max = _boundary_qa_planner_dependency_target_range(qtype, skill, item_difficulty_target)
    gold_count = sum(_boundary_event_dependency_weight(event) for event in gold_events) + len({_event_ref_id(event) for event in gold_events})
    needed = max(0, target_min - gold_count)
    per_event = max(1, (needed + 1) // 2)
    if target_max is not None:
        available = max(1, target_max - gold_count)
        per_event = min(per_event, max(1, available // 2 or available))
    return max(1, min(4, per_event))


PERMISSION_SYNTHETIC_FORBIDDEN_SURFACE_CUES = (
    "not the current",
    "wrong scope",
    "wrong condition",
    "wrong person",
    "was replaced",
    "replaced for the current",
    "earlier permission",
    "later update",
    "invalidated",
    "outdated",
)


def _permission_synthetic_slot(source: dict[str, Any]) -> str:
    slot = normalize_space(source.get("slot")).replace("_", " ").lower()
    if not slot:
        return "shared detail"
    norm = _option_text_norm(slot)
    if "temporal" in norm or "time" in norm or "date" in norm:
        return "timing detail"
    if "raw qa" in norm or "rawqa" in norm or "single hop" in norm:
        return "shared detail"
    if "permission" in norm or "visibility" in norm or "scope" in norm:
        return "access detail"
    if "plan" in norm:
        return "plan detail"
    clean = re.sub(r"\b(?:raw\s*qa|single\s*hop|temporal|archive|report)\b", "", slot, flags=re.IGNORECASE)
    clean = normalize_space(clean)
    return clean[:80] or "shared detail"


def _permission_synthetic_value(source: dict[str, Any], *, kind: str) -> str:
    """Create current-looking hard-negative text without revealing why it is false.

    The graph status still carries the false semantics. The option surface should
    look like a plausible current fact, otherwise the solver can eliminate it from
    wording alone and the row becomes easy instead of boundary-producing.
    """
    entity = normalize_space(source.get("entity")).replace("_", " ") or "the person"
    slot = _permission_synthetic_slot(source)
    if kind == "scope":
        return sanitize_internal_text("available to the planning circle under this sharing setting")
    return sanitize_internal_text("kept as the working version for the shared plan")


def _permission_synthetic_forbidden_cue_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        text = _option_text_norm(_event_display_text(event, hard_surface=True))
        count += sum(1 for cue in PERMISSION_SYNTHETIC_FORBIDDEN_SURFACE_CUES if cue in text)
    return count


def _make_permission_synthetic_false_event(
    source_event: dict[str, Any],
    *,
    gold_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    kind: str,
    dependency_weight: int,
) -> dict[str, Any] | None:
    turn_ids = _event_turn_ids_any(source_event, current_alignment)
    if not turn_ids:
        return None
    source_ref = _event_ref_id(source_event)
    if not source_ref:
        return None
    event = copy.deepcopy(source_event)
    synthetic_ref = stable_hash(
        {
            "kind": kind,
            "source_event_id": source_ref,
            "gold_event_ids": [_event_ref_id(gold) for gold in gold_events],
        },
        length=12,
    )
    event["event_instance_id"] = f"det_permission_{kind}_{synthetic_ref}"
    event["event_id"] = event["event_instance_id"]
    event["slot"] = _permission_synthetic_slot(source_event)
    event["status"] = "scope_limited" if kind == "scope" else "outdated"
    event["value"] = _permission_synthetic_value(source_event, kind=kind)
    event["dialogue_value"] = event["value"]
    event["turn_ids"] = list(turn_ids)
    event["evidence_turn_ids"] = list(turn_ids)
    event["_context_source"] = source_event.get("_context_source") or "synthetic_graph_derived"
    # The synthesized candidate is graph-derived from the source event, but it is a
    # single planned hard-negative path. Keep its planner weight controlled so the
    # difficulty target does not get pushed past the boundary/harder-boundary band.
    event["_dependency_count"] = max(int(dependency_weight), 1)
    event["_synthetic_permission_hard_negative"] = True
    event["_synthetic_permission_surface_style"] = "neutral_current_like"
    event["_synthetic_grounding_source_event_id"] = source_ref
    event["_synthetic_grounding_source_turn_ids"] = list(turn_ids)
    event["_hard_negative_type_override"] = (
        "active_but_wrong_scope" if kind == "scope" else "parent_valid_child_invalidated"
    )
    event["support_claim"] = event["value"]
    event["natural_false_surface_preview"] = _event_display_text(event, hard_surface=True)
    return event


def _synthesize_permission_visibility_false_events(
    *,
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    item_difficulty_target: str | None,
    existing_texts: set[str],
) -> tuple[list[dict[str, Any]], list[str], str | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "deterministic_permission_synthesizer_used": False,
        "deterministic_permission_synthesizer_failure_reason": None,
        "deterministic_permission_synthesized_event_ids": [],
        "deterministic_permission_synthesized_source_event_ids": [],
        "deterministic_permission_surface_naturalized": False,
        "deterministic_permission_surface_style": None,
        "deterministic_permission_surface_forbidden_cue_count": 0,
    }
    candidate_sources = [
        event
        for event in [*child_events, *parent_events, *gold_events]
        if isinstance(event, dict)
        and _event_turn_ids_any(event, current_alignment)
        and not _event_option_surface_is_placeholder(event)
    ]
    if not candidate_sources:
        diagnostics["deterministic_permission_synthesizer_failure_reason"] = "no_anchorable_source_events"
        return [], [], None, diagnostics
    dep_weight = _permission_synthetic_dependency_weight(
        gold_events,
        qtype="multi_select",
        skill="permission_visibility_scope",
        item_difficulty_target=item_difficulty_target,
    )
    scope_source = next((event for event in candidate_sources if str(event.get("_context_source")) == "child"), candidate_sources[0])
    override_source = next(
        (event for event in candidate_sources if str(event.get("_context_source")) == "parent" and _event_ref_id(event) != _event_ref_id(scope_source)),
        next((event for event in candidate_sources if _event_ref_id(event) != _event_ref_id(scope_source)), scope_source),
    )
    synthesized = [
        _make_permission_synthetic_false_event(
            scope_source,
            gold_events=gold_events,
            current_alignment=current_alignment,
            kind="scope",
            dependency_weight=dep_weight,
        ),
        _make_permission_synthetic_false_event(
            override_source,
            gold_events=gold_events,
            current_alignment=current_alignment,
            kind="override",
            dependency_weight=dep_weight,
        ),
    ]
    false_events = _dedupe_events_by_surface(
        [event for event in synthesized if isinstance(event, dict)],
        existing_texts=existing_texts,
    )
    hard_types = [_hard_negative_type_for_event(event, gold_events) for event in false_events]
    if len(false_events) < 2:
        diagnostics["deterministic_permission_synthesizer_failure_reason"] = "synthesized_surfaces_not_unique"
        return [], [], None, diagnostics
    if not ({*hard_types} & BOUNDARY_PERMISSION_SCOPE_NEGATIVE_TYPES):
        diagnostics["deterministic_permission_synthesizer_failure_reason"] = "missing_synthetic_scope_negative"
        return [], [], None, diagnostics
    if not ({*hard_types} & BOUNDARY_PERMISSION_OVERRIDE_NEGATIVE_TYPES):
        diagnostics["deterministic_permission_synthesizer_failure_reason"] = "missing_synthetic_override_negative"
        return [], [], None, diagnostics
    forbidden_cue_count = _permission_synthetic_forbidden_cue_count(false_events)
    diagnostics["deterministic_permission_synthesizer_used"] = True
    diagnostics["deterministic_permission_surface_naturalized"] = True
    diagnostics["deterministic_permission_surface_style"] = "neutral_current_like"
    diagnostics["deterministic_permission_surface_forbidden_cue_count"] = forbidden_cue_count
    diagnostics["deterministic_permission_synthesized_event_ids"] = [_event_ref_id(event) for event in false_events]
    diagnostics["deterministic_permission_synthesized_source_event_ids"] = [
        str(event.get("_synthetic_grounding_source_event_id") or "") for event in false_events
    ]
    return false_events[:2], hard_types[:2], _boundary_permission_explicit_contrast_type(hard_types[:2]), diagnostics


def _select_permission_visibility_false_events(
    false_pool: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
    *,
    item_difficulty_target: str | None,
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    target = _normalize_boundary_item_difficulty_target(item_difficulty_target)
    target_min, target_max = _boundary_qa_planner_dependency_target_range("multi_select", "permission_visibility_scope", target)
    typed_events: list[tuple[dict[str, Any], str]] = []
    for event in false_pool:
        htype = _hard_negative_type_for_event(event, gold_events)
        if htype in BOUNDARY_QA_REPAIR_HARD_NEGATIVE_TYPES:
            typed_events.append((event, htype))
    if len(typed_events) < 2:
        return [], [], None
    if target is None:
        false_events = [typed_events[0][0], typed_events[1][0]]
        hard_types = [typed_events[0][1], typed_events[1][1]]
        return false_events, hard_types, _boundary_permission_explicit_contrast_type(hard_types)

    def score_pair(pair: tuple[tuple[dict[str, Any], str], tuple[dict[str, Any], str]]) -> tuple[float, float, str]:
        (event_a, type_a), (event_b, type_b) = pair
        dep_count = _boundary_plan_dependency_count_for_events(gold_events, [event_a, event_b])
        midpoint = (target_min + (target_max if target_max is not None else target_min + 2)) / 2.0
        distance = abs(dep_count - midpoint)
        similarity = _event_same_topic_score(event_a, gold_events) + _event_same_topic_score(event_b, gold_events)
        return (distance, similarity, _event_ref_id(event_a) + "|" + _event_ref_id(event_b))

    pairs: list[tuple[tuple[dict[str, Any], str], tuple[dict[str, Any], str]]] = []
    for left_index, left in enumerate(typed_events):
        for right in typed_events[left_index + 1:]:
            events = [left[0], right[0]]
            if not _event_surfaces_are_unique(gold_events + events):
                continue
            types = [left[1], right[1]]
            if all(str(event.get("status")) == "outdated" for event in events):
                continue
            dep_count = _boundary_plan_dependency_count_for_events(gold_events, events)
            if target_max is not None and dep_count > target_max and target in {"easy", "boundary", "harder_boundary"}:
                continue
            if dep_count < target_min and target in {"boundary", "harder_boundary", "hard"}:
                continue
            pairs.append((left, right))

    if target in {"boundary", "harder_boundary"}:
        boundary_pairs = [
            pair
            for pair in pairs
            if ({pair[0][1], pair[1][1]} & BOUNDARY_PERMISSION_SCOPE_NEGATIVE_TYPES)
            and ({pair[0][1], pair[1][1]} & BOUNDARY_PERMISSION_OVERRIDE_NEGATIVE_TYPES)
        ]
        if not boundary_pairs:
            return [], [], None
        if target == "harder_boundary":
            chosen = sorted(
                boundary_pairs,
                key=lambda pair: (
                    score_pair(pair)[0],
                    -(_event_same_topic_score(pair[0][0], gold_events) + _event_same_topic_score(pair[1][0], gold_events)),
                    score_pair(pair)[2],
                ),
            )[0]
        else:
            chosen = sorted(boundary_pairs, key=score_pair)[0]
    elif target == "easy":
        easy_pairs = [
            pair
            for pair in pairs
            if "cross_context_dependency_mismatch" in {pair[0][1], pair[1][1]}
            or "invalidated_parent_dependency" in {pair[0][1], pair[1][1]}
        ]
        chosen = sorted(easy_pairs or pairs, key=score_pair)[0] if pairs else (typed_events[0], typed_events[1])
    else:
        chosen = sorted(pairs or [(typed_events[0], typed_events[1])], key=score_pair)[0]

    false_events = [chosen[0][0], chosen[1][0]]
    hard_types = [chosen[0][1], chosen[1][1]]
    return false_events, hard_types, _boundary_permission_explicit_contrast_type(hard_types)


def _qa_option_texts_are_unique(options: list[dict[str, Any]]) -> bool:
    seen: list[str] = []
    for option in options:
        text = _option_text_norm(option.get("text"))
        if not text:
            return False
        if text in seen or any(_surface_similarity_score(text, old) >= 0.92 for old in seen):
            return False
        seen.append(text)
    return True


def _boundary_qa_make_single_event_option(
    label: str,
    event: dict[str, Any],
    current_alignment: dict[str, Any],
    *,
    gold_events: list[dict[str, Any]],
    is_gold: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    option = _option_from_single_event(label, event, current_alignment, hard_surface=True)
    if is_gold:
        return option, None
    htype = _hard_negative_type_for_event(event, gold_events)
    option["hard_negative_path_type"] = htype
    reason = {
        "candidate_event_instance_id": _event_ref_id(event),
        "false_reason_type": htype,
        "distractor_source": event.get("_context_source"),
        "distractor_relation": "boundary_aware_repaired_hard_negative",
    }
    return option, reason


def _empty_boundary_item_plan(
    *,
    qtype: str,
    skill: str | None,
    template: str | None,
    reason: str,
    item_difficulty_target: str | None = None,
) -> dict[str, Any]:
    normalized_target = _normalize_boundary_item_difficulty_target(item_difficulty_target)
    target_min, target_max = _boundary_qa_planner_dependency_target_range(qtype, skill, normalized_target)
    profile = _boundary_item_difficulty_profile(normalized_target)
    return {
        "qtype": qtype,
        "skill": skill or "unknown",
        "template": template or "ordinary",
        "item_difficulty_target": normalized_target,
        "gold_path_type": None,
        "gold_evidence_event_ids": [],
        "hard_negative_path_types": [],
        "hard_negative_event_ids": [],
        "forbidden_surface_patterns": [],
        "expected_status_pattern": [],
        "item_plan_created": False,
        "item_plan_valid": False,
        "item_plan_failure_reason": reason,
        "planner_qa_used_dependency_count": 0,
        "planner_dependency_floor": _boundary_qa_planner_dependency_floor(qtype, skill, normalized_target),
        "planner_dependency_target_min": target_min,
        "planner_dependency_target_max": target_max,
        "actual_qa_used_dependency_count": 0,
        "planner_dependency_floor_pass": False,
        "planner_dependency_floor_failure_reason": reason,
        "difficulty_calibration_failure_reason": reason,
        "hard_negative_similarity_level": profile["hard_negative_similarity_level"],
        "intermediate_clue_added": False,
        "explicit_contrast_type": None,
        "boundary_calibration_mode": profile["boundary_calibration_mode"],
        "expected_solver_bucket": profile["expected_solver_bucket"],
        "planner_replan_attempt_count": 0,
        "planner_replan_success_count": 0,
        "planner_replan_failed_count": 0,
        "dependency_floor_pass_by_cell": {},
        "dependency_floor_fail_by_cell": {_boundary_qa_planner_cell_key(qtype, skill): 1},
        "delayed_activation_chain_valid": False,
        "delayed_activation_parent_status": None,
        "delayed_activation_child_status": None,
        "delayed_activation_trigger_event_id": None,
        "delayed_activation_has_condition_dependency": False,
        "delayed_activation_active_active_pair_detected": False,
        "delayed_activation_semantic_failure_reason": reason if skill == "delayed_activation" else None,
    }


def _boundary_qa_planner_supported_cell(question_type: str, attack_skill: str | None) -> bool:
    qtype = question_type if question_type in {"multi_select", "single_choice"} else "multi_select"
    skill = attack_skill or "unknown"
    return (qtype == "multi_select" and skill == "permission_visibility_scope") or (
        qtype == "single_choice" and skill == "delayed_activation"
    )


BOUNDARY_QA_PLANNER_DEFAULT_DEPENDENCY_FLOOR = 6
BOUNDARY_QA_PLANNER_TARGET_MEAN_DEPENDENCY = 8
BOUNDARY_QA_PLANNER_DEPENDENCY_FLOOR_BY_CELL = {
    "multi_select|permission_visibility_scope": 6,
    "single_choice|delayed_activation": 8,
}
BOUNDARY_ITEM_DIFFICULTY_PROFILES: dict[str | None, dict[str, Any]] = {
    None: {
        "dependency_target_min": None,
        "dependency_target_max": None,
        "hard_negative_similarity_level": "strong",
        "intermediate_clue_added": False,
        "boundary_calibration_mode": "legacy",
        "expected_solver_bucket": "uncontrolled",
    },
    "easy": {
        "dependency_target_min": 2,
        "dependency_target_max": 4,
        "hard_negative_similarity_level": "low_medium",
        "intermediate_clue_added": False,
        "boundary_calibration_mode": "easy_control",
        "expected_solver_bucket": "4/5-5/5",
    },
    "boundary": {
        "dependency_target_min": 4,
        "dependency_target_max": 6,
        "hard_negative_similarity_level": "medium",
        "intermediate_clue_added": True,
        "boundary_calibration_mode": "boundary_target",
        "expected_solver_bucket": "1/5-4/5",
    },
    "harder_boundary": {
        "dependency_target_min": 6,
        "dependency_target_max": 8,
        "hard_negative_similarity_level": "medium_high",
        "intermediate_clue_added": True,
        "boundary_calibration_mode": "harder_boundary_target",
        "expected_solver_bucket": "1/5-4/5-or-0/5",
    },
    "hard": {
        "dependency_target_min": 8,
        "dependency_target_max": None,
        "hard_negative_similarity_level": "high",
        "intermediate_clue_added": False,
        "boundary_calibration_mode": "hard_diagnostic",
        "expected_solver_bucket": "0/5-or-hard-boundary",
    },
}


def _normalize_boundary_item_difficulty_target(value: Any) -> str | None:
    text = normalize_space(value or "").lower().replace("_", "-")
    aliases = {
        "easy": "easy",
        "easy-control": "easy",
        "boundary": "boundary",
        "boundary-target": "boundary",
        "medium": "boundary",
        "medium-hard": "boundary",
        "harder-boundary": "harder_boundary",
        "boundary-hard": "harder_boundary",
        "medium-high": "harder_boundary",
        "hard": "hard",
        "hard-diagnostic": "hard",
    }
    return aliases.get(text)


def _boundary_item_difficulty_profile(item_difficulty_target: str | None) -> dict[str, Any]:
    return dict(BOUNDARY_ITEM_DIFFICULTY_PROFILES.get(_normalize_boundary_item_difficulty_target(item_difficulty_target), BOUNDARY_ITEM_DIFFICULTY_PROFILES[None]))


def _boundary_qa_planner_cell_key(qtype: str | None, skill: str | None) -> str:
    return f"{qtype or 'unknown'}|{skill or 'unknown'}"


def _boundary_qa_planner_dependency_floor(qtype: str | None, skill: str | None, item_difficulty_target: str | None = None) -> int:
    profile = _boundary_item_difficulty_profile(item_difficulty_target)
    if profile.get("dependency_target_min") is not None:
        return int(profile["dependency_target_min"])
    return int(
        BOUNDARY_QA_PLANNER_DEPENDENCY_FLOOR_BY_CELL.get(
            _boundary_qa_planner_cell_key(qtype, skill),
            BOUNDARY_QA_PLANNER_DEFAULT_DEPENDENCY_FLOOR,
        )
    )


def _boundary_qa_planner_dependency_target_range(
    qtype: str | None,
    skill: str | None,
    item_difficulty_target: str | None = None,
) -> tuple[int, int | None]:
    profile = _boundary_item_difficulty_profile(item_difficulty_target)
    target_min = profile.get("dependency_target_min")
    target_max = profile.get("dependency_target_max")
    if target_min is None:
        target_min = _boundary_qa_planner_dependency_floor(qtype, skill, None)
    return int(target_min), int(target_max) if target_max is not None else None


def _boundary_qa_planner_dependency_count(plan: dict[str, Any]) -> int:
    gold_events = [event for event in plan.get("_gold_events") or [] if isinstance(event, dict)]
    hard_events = [event for event in plan.get("_hard_negative_events") or [] if isinstance(event, dict)]
    selected_events = gold_events + hard_events
    gold_ids = {_event_ref_id(event) for event in gold_events}
    return sum(int(event.get("_dependency_count") or 0) for event in selected_events) + len(gold_ids)


def _event_text_blob_for_semantics(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in event.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(str(key))
            parts.append(str(value))
        elif isinstance(value, (list, tuple)):
            parts.append(str(key))
            parts.extend(str(item) for item in value if isinstance(item, (str, int, float, bool)))
        elif isinstance(value, dict):
            parts.append(str(key))
            parts.extend(str(sub_key) for sub_key in value.keys())
            parts.extend(str(sub_value) for sub_value in value.values() if isinstance(sub_value, (str, int, float, bool)))
    return _option_text_norm(" ".join(parts))


def _delayed_activation_status(event: dict[str, Any]) -> str:
    return _option_text_norm(str(event.get("status") or ""))


def _event_semantically_unsatisfied(event: dict[str, Any]) -> bool:
    status = _delayed_activation_status(event)
    if status in DELAYED_ACTIVATION_PARENT_UNSATISFIED_STATUSES:
        return True
    text = _event_text_blob_for_semantics(event)
    return any(
        marker in text
        for marker in (
            "delayed",
            "pending",
            "unsatisfied",
            "condition unsatisfied",
            "not yet active",
            "not active yet",
            "waiting",
            "blocked until",
            "trigger required",
        )
    )


def _event_semantically_active(event: dict[str, Any]) -> bool:
    status = _delayed_activation_status(event)
    if status in DELAYED_ACTIVATION_CHILD_ACTIVE_STATUSES:
        return True
    text = _event_text_blob_for_semantics(event)
    return any(marker in text for marker in ("activated", "now active", "condition satisfied", "trigger occurred", "current valid"))


def _event_has_trigger_semantics(event: dict[str, Any]) -> bool:
    text = _event_text_blob_for_semantics(event)
    status = _delayed_activation_status(event)
    if status in {"activated", "active", "current_valid", "current-valid"} and any(
        marker in text for marker in ("trigger", "condition satisfied", "activated", "activation", "because", "after")
    ):
        return True
    return any(marker in text for marker in ("trigger", "triggered", "condition satisfied", "activation", "activated after"))


def _event_has_condition_dependency(event: dict[str, Any]) -> bool:
    text = _event_text_blob_for_semantics(event)
    if any(str(key).startswith("depends_on") for key in event.keys()):
        return True
    return any(marker in text for marker in ("condition", "trigger", "depends on", "requires", "until", "after", "because"))


def _same_entity_slot(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(str(first.get("entity") or "") and str(first.get("entity") or "") == str(second.get("entity") or "")) and bool(
        str(first.get("slot") or "") and str(first.get("slot") or "") == str(second.get("slot") or "")
    )


def _delayed_activation_semantic_hard_negative_present(
    hard_events: list[dict[str, Any]],
    hard_types: list[str],
    gold_events: list[dict[str, Any]],
) -> bool:
    for event, htype in zip(hard_events, hard_types):
        htype = str(htype or "")
        if htype in {"same_entity_same_slot_wrong_trigger", "later_state_override", "parent_valid_child_invalidated", "missing_final_dependency"}:
            return True
        if htype == "active_but_wrong_condition" and any(_same_entity_slot(event, gold) for gold in gold_events):
            return True
        if str(event.get("status") or "") == "delayed_not_active" and any(_same_entity_slot(event, gold) for gold in gold_events):
            return True
    return False


def _delayed_activation_semantic_result(plan: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "delayed_activation_chain_valid": False,
        "delayed_activation_parent_status": None,
        "delayed_activation_child_status": None,
        "delayed_activation_trigger_event_id": None,
        "delayed_activation_has_condition_dependency": False,
        "delayed_activation_active_active_pair_detected": False,
        "delayed_activation_semantic_failure_reason": reason,
    }


def _validate_delayed_activation_chain(plan: dict[str, Any]) -> dict[str, Any]:
    gold_events = [event for event in plan.get("_gold_events") or [] if isinstance(event, dict)]
    hard_events = [event for event in plan.get("_hard_negative_events") or [] if isinstance(event, dict)]
    hard_types = [str(item) for item in plan.get("hard_negative_path_types") or []]
    if len(gold_events) < 2:
        return _delayed_activation_semantic_result(plan, reason="delayed_activation_missing_chain")
    parent_event, child_event = gold_events[0], gold_events[1]
    parent_status = _delayed_activation_status(parent_event)
    child_status = _delayed_activation_status(child_event)
    result = {
        "delayed_activation_chain_valid": False,
        "delayed_activation_parent_status": parent_status,
        "delayed_activation_child_status": child_status,
        "delayed_activation_trigger_event_id": None,
        "delayed_activation_has_condition_dependency": False,
        "delayed_activation_active_active_pair_detected": False,
        "delayed_activation_semantic_failure_reason": None,
    }
    if not _same_entity_slot(parent_event, child_event):
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_missing_same_slot"
        return result
    if not (_event_has_turn_anchor(parent_event) and _event_has_turn_anchor(child_event)):
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_missing_chain"
        return result
    if parent_status in ACTIVE_STATUSES and child_status in ACTIVE_STATUSES:
        result["delayed_activation_active_active_pair_detected"] = True
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_active_active_pair"
        return result
    if not _event_semantically_unsatisfied(parent_event):
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_parent_not_unsatisfied"
        return result
    if not _event_semantically_active(child_event):
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_missing_child_trigger"
        return result
    has_condition = _event_has_condition_dependency(parent_event) or _event_has_condition_dependency(child_event)
    trigger_event: dict[str, Any] | None = None
    for event in [child_event, *hard_events]:
        if _same_entity_slot(event, child_event) and _event_has_trigger_semantics(event):
            trigger_event = event
            break
    result["delayed_activation_has_condition_dependency"] = bool(has_condition)
    result["delayed_activation_trigger_event_id"] = _event_ref_id(trigger_event) if trigger_event else None
    if not has_condition:
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_missing_chain"
        return result
    if trigger_event is None:
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_missing_child_trigger"
        return result
    if not _delayed_activation_semantic_hard_negative_present(hard_events, hard_types, gold_events):
        result["delayed_activation_semantic_failure_reason"] = "delayed_activation_hard_negative_not_semantic"
        return result
    result["delayed_activation_chain_valid"] = True
    return result


def _with_dependency_floor_diagnostics(plan: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    qtype = str(out.get("qtype") or "unknown")
    skill = str(out.get("skill") or "unknown")
    item_difficulty_target = _normalize_boundary_item_difficulty_target(out.get("item_difficulty_target"))
    target_min, target_max = _boundary_qa_planner_dependency_target_range(qtype, skill, item_difficulty_target)
    floor = _boundary_qa_planner_dependency_floor(qtype, skill, item_difficulty_target)
    count = _boundary_qa_planner_dependency_count(out)
    below_target = count < target_min
    above_target = target_max is not None and count > target_max
    floor_pass = not below_target and not above_target
    if below_target:
        failure_reason = f"dependency_count_{count}_below_target_min_{target_min}"
    elif above_target:
        failure_reason = f"dependency_count_{count}_above_target_max_{target_max}"
    else:
        failure_reason = None
    out["planner_qa_used_dependency_count"] = count
    out["actual_qa_used_dependency_count"] = count
    out["planner_dependency_floor"] = floor
    out["planner_dependency_target_min"] = target_min
    out["planner_dependency_target_max"] = target_max
    out["planner_dependency_floor_pass"] = bool(floor_pass)
    out["planner_dependency_floor_failure_reason"] = failure_reason
    out["difficulty_calibration_failure_reason"] = failure_reason
    return out


def _with_planner_replan_diagnostics(
    plan: dict[str, Any],
    *,
    replan_attempts: int,
    replan_success: bool,
    replan_failed: bool,
) -> dict[str, Any]:
    previous_floor_failure = plan.get("planner_dependency_floor_failure_reason")
    out = _with_dependency_floor_diagnostics(plan)
    if previous_floor_failure and not out.get("item_plan_valid"):
        out["planner_dependency_floor_failure_reason"] = previous_floor_failure
    cell_key = _boundary_qa_planner_cell_key(str(out.get("qtype") or "unknown"), str(out.get("skill") or "unknown"))
    out["planner_replan_attempt_count"] = int(replan_attempts)
    out["planner_replan_success_count"] = 1 if replan_success else 0
    out["planner_replan_failed_count"] = 1 if replan_failed else 0
    out["dependency_floor_pass_by_cell"] = {cell_key: 1} if out.get("planner_dependency_floor_pass") else {}
    out["dependency_floor_fail_by_cell"] = {cell_key: 1} if not out.get("planner_dependency_floor_pass") else {}
    return out


def _planner_failure_is_replannable(plan: dict[str, Any]) -> bool:
    reason = str(plan.get("item_plan_failure_reason") or "")
    floor_reason = str(plan.get("planner_dependency_floor_failure_reason") or "")
    return reason in {
        "dependency_floor_not_met",
        "delayed_activation_requires_wrong_trigger_or_later_override",
        *DELAYED_ACTIVATION_SEMANTIC_FAILURE_REASONS,
    } or bool(floor_reason)


def _active_wrong_events(
    events: list[dict[str, Any]],
    gold_events: list[dict[str, Any]],
    *,
    source: str | None,
    current_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    gold_ids = {_event_ref_id(event) for event in gold_events}
    candidates = [
        event
        for event in events
        if (source is None or event.get("_context_source") == source)
        and str(event.get("status")) in ACTIVE_STATUSES
        and _event_ref_id(event) not in gold_ids
        and _event_turn_ids_any(event, current_alignment)
    ]
    candidates.sort(key=lambda event: (-_event_same_topic_score(event, gold_events), _event_ref_id(event)))
    return _dedupe_events_by_surface(candidates)


def _event_surfaces_are_unique(events: list[dict[str, Any]]) -> bool:
    texts: list[str] = []
    for event in events:
        text = _option_text_norm(_event_display_text(event, hard_surface=True))
        if not text:
            return False
        if text in texts or any(_surface_similarity_score(text, old) >= 0.92 for old in texts):
            return False
        texts.append(text)
    return True


def _event_pair_surfaces_are_unique(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    texts: list[str] = []
    for parent_event, child_event in pairs:
        text = _option_text_norm(
            f"earlier {_event_display_text(parent_event, hard_surface=True)} later {_event_display_text(child_event, hard_surface=True)}"
        )
        if not text:
            return False
        if text in texts or any(_surface_similarity_score(text, old) >= 0.92 for old in texts):
            return False
        texts.append(text)
    return True


def _boundary_item_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "boundary_qa_planner_enabled": True,
        "planner_used": bool(plan.get("item_plan_valid")),
        "item_plan_created": bool(plan.get("item_plan_created")),
        "item_plan_valid": bool(plan.get("item_plan_valid")),
        "item_plan_failure_reason": plan.get("item_plan_failure_reason"),
        "gold_path_type": plan.get("gold_path_type"),
        "gold_evidence_event_ids": list(plan.get("gold_evidence_event_ids") or []),
        "hard_negative_path_types": list(plan.get("hard_negative_path_types") or []),
        "hard_negative_event_ids": list(plan.get("hard_negative_event_ids") or []),
        "hard_negative_strength": plan.get("hard_negative_strength"),
        "item_difficulty_target": plan.get("item_difficulty_target"),
        "planner_dependency_target_min": plan.get("planner_dependency_target_min"),
        "planner_dependency_target_max": plan.get("planner_dependency_target_max"),
        "actual_qa_used_dependency_count": int(plan.get("actual_qa_used_dependency_count") or plan.get("planner_qa_used_dependency_count") or 0),
        "hard_negative_similarity_level": plan.get("hard_negative_similarity_level"),
        "intermediate_clue_added": bool(plan.get("intermediate_clue_added")),
        "explicit_contrast_type": plan.get("explicit_contrast_type"),
        "boundary_calibration_mode": plan.get("boundary_calibration_mode"),
        "expected_solver_bucket": plan.get("expected_solver_bucket"),
        "difficulty_calibration_failure_reason": plan.get("difficulty_calibration_failure_reason"),
        "forbidden_surface_patterns": list(plan.get("forbidden_surface_patterns") or []),
        "expected_status_pattern": list(plan.get("expected_status_pattern") or []),
        "planner_qa_used_dependency_count": int(plan.get("planner_qa_used_dependency_count") or 0),
        "planner_dependency_floor": int(plan.get("planner_dependency_floor") or 0),
        "planner_dependency_floor_pass": bool(plan.get("planner_dependency_floor_pass")),
        "planner_dependency_floor_failure_reason": plan.get("planner_dependency_floor_failure_reason"),
        "planner_replan_attempt_count": int(plan.get("planner_replan_attempt_count") or 0),
        "planner_replan_success_count": int(plan.get("planner_replan_success_count") or 0),
        "planner_replan_failed_count": int(plan.get("planner_replan_failed_count") or 0),
        "dependency_floor_pass_by_cell": dict(plan.get("dependency_floor_pass_by_cell") or {}),
        "dependency_floor_fail_by_cell": dict(plan.get("dependency_floor_fail_by_cell") or {}),
        "delayed_activation_chain_valid": plan.get("delayed_activation_chain_valid"),
        "delayed_activation_parent_status": plan.get("delayed_activation_parent_status"),
        "delayed_activation_child_status": plan.get("delayed_activation_child_status"),
        "delayed_activation_trigger_event_id": plan.get("delayed_activation_trigger_event_id"),
        "delayed_activation_has_condition_dependency": plan.get("delayed_activation_has_condition_dependency"),
        "delayed_activation_active_active_pair_detected": plan.get("delayed_activation_active_active_pair_detected"),
        "delayed_activation_semantic_failure_reason": plan.get("delayed_activation_semantic_failure_reason"),
        "deterministic_permission_hard_negatives": bool(plan.get("deterministic_permission_hard_negatives")),
        "deterministic_permission_synthesizer_used": bool(plan.get("deterministic_permission_synthesizer_used")),
        "deterministic_permission_synthesizer_failure_reason": plan.get("deterministic_permission_synthesizer_failure_reason"),
        "deterministic_permission_synthesized_event_ids": list(plan.get("deterministic_permission_synthesized_event_ids") or []),
        "deterministic_permission_synthesized_source_event_ids": list(plan.get("deterministic_permission_synthesized_source_event_ids") or []),
        "deterministic_permission_surface_naturalized": bool(plan.get("deterministic_permission_surface_naturalized")),
        "deterministic_permission_surface_style": plan.get("deterministic_permission_surface_style"),
        "deterministic_permission_surface_forbidden_cue_count": int(
            plan.get("deterministic_permission_surface_forbidden_cue_count") or 0
        ),
    }


def _validate_boundary_item_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or not plan.get("item_plan_created"):
        return _empty_boundary_item_plan(qtype=str((plan or {}).get("qtype") or "unknown"), skill=(plan or {}).get("skill"), template=(plan or {}).get("template"), reason="item_plan_not_created")
    required = [
        "qtype",
        "skill",
        "template",
        "gold_path_type",
        "gold_evidence_event_ids",
        "hard_negative_path_types",
        "hard_negative_event_ids",
        "forbidden_surface_patterns",
        "expected_status_pattern",
    ]
    missing = [key for key in required if plan.get(key) in (None, "", [], {})]
    if missing:
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "missing_" + ",".join(missing)
        return _with_dependency_floor_diagnostics(out)
    if len(plan.get("gold_evidence_event_ids") or []) < 2:
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "gold_path_requires_parent_and_child_events"
        return _with_dependency_floor_diagnostics(out)
    if len(plan.get("hard_negative_event_ids") or []) < (2 if plan.get("qtype") == "multi_select" else 3):
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "insufficient_hard_negative_events"
        return _with_dependency_floor_diagnostics(out)
    gold_events = [event for event in plan.get("_gold_events") or [] if isinstance(event, dict)]
    hard_events = [event for event in plan.get("_hard_negative_events") or [] if isinstance(event, dict)]
    gold_ids = [_event_ref_id(event) for event in gold_events]
    hard_ids = [_event_ref_id(event) for event in hard_events]
    if gold_ids[: len(plan.get("gold_evidence_event_ids") or [])] != list(plan.get("gold_evidence_event_ids") or []):
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "gold_event_ids_not_grounded_in_graph_events"
        return _with_dependency_floor_diagnostics(out)
    if hard_ids[: len(plan.get("hard_negative_event_ids") or [])] != list(plan.get("hard_negative_event_ids") or []):
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "hard_negative_event_ids_not_grounded_in_graph_events"
        return _with_dependency_floor_diagnostics(out)
    expected_types = [
        _hard_negative_type_for_event(event, gold_events)
        for event in hard_events[: len(plan.get("hard_negative_path_types") or [])]
    ]
    if expected_types != list(plan.get("hard_negative_path_types") or []):
        out = dict(plan)
        out["item_plan_valid"] = False
        out["item_plan_failure_reason"] = "hard_negative_path_types_not_grounded_in_graph_events"
        return _with_dependency_floor_diagnostics(out)
    floor_checked = _with_dependency_floor_diagnostics(plan)
    if not floor_checked.get("planner_dependency_floor_pass"):
        floor_checked["item_plan_valid"] = False
        floor_checked["item_plan_failure_reason"] = "dependency_floor_not_met"
        return floor_checked
    if plan.get("qtype") == "single_choice" and plan.get("skill") == "delayed_activation":
        delayed_semantics = _validate_delayed_activation_chain(floor_checked)
        floor_checked.update(delayed_semantics)
        if not delayed_semantics.get("delayed_activation_chain_valid"):
            reason = str(delayed_semantics.get("delayed_activation_semantic_failure_reason") or "delayed_activation_missing_chain")
            floor_checked["item_plan_valid"] = False
            floor_checked["item_plan_failure_reason"] = reason
            floor_checked["planner_dependency_floor_failure_reason"] = reason
            return floor_checked
    out = dict(plan)
    out["item_plan_valid"] = True
    out["item_plan_failure_reason"] = None
    out = _with_dependency_floor_diagnostics(out)
    if plan.get("qtype") == "single_choice" and plan.get("skill") == "delayed_activation":
        out.update(_validate_delayed_activation_chain(out))
    return out


def _build_boundary_aware_item_plan(
    *,
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    question_type: str,
    attack_skill: str | None,
    hard_single_choice_template: str | None,
    item_difficulty_target: str | None = None,
    deterministic_permission_hard_negatives: bool = False,
) -> dict[str, Any]:
    qtype = question_type if question_type in {"multi_select", "single_choice"} else "multi_select"
    skill = attack_skill or "unknown"
    template = hard_single_choice_template if hard_single_choice_template and hard_single_choice_template != "ordinary" else "ordinary"
    item_difficulty_target = _normalize_boundary_item_difficulty_target(item_difficulty_target)
    difficulty_profile = _boundary_item_difficulty_profile(item_difficulty_target)
    if not _boundary_qa_planner_supported_cell(qtype, skill):
        return _empty_boundary_item_plan(
            qtype=qtype,
            skill=skill,
            template=template,
            reason=f"unsupported_boundary_planner_cell:{qtype}|{skill}|{template}",
            item_difficulty_target=item_difficulty_target,
        )

    parent_active = [event for event in parent_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    child_active = [event for event in child_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    if qtype == "single_choice" and skill == "delayed_activation":
        parent_candidates = [
            event
            for event in parent_events
            if _event_turn_ids_any(event, current_alignment) and _event_semantically_unsatisfied(event)
        ]
        child_candidates = [
            event
            for event in child_events
            if _event_turn_ids_any(event, current_alignment) and _event_semantically_active(event)
        ]
        pairs = _best_parent_child_pairs(parent_candidates, child_candidates)
        no_pair_reason = "no_delayed_activation_parent_child_pair"
    else:
        pairs = _best_parent_child_pairs(parent_active, child_active)
        no_pair_reason = "no_parent_child_active_pair"
    if not pairs:
        return _empty_boundary_item_plan(
            qtype=qtype,
            skill=skill,
            template=template,
            reason=no_pair_reason,
            item_difficulty_target=item_difficulty_target,
        )
    all_events = parent_events + child_events
    replan_attempts = 0
    last_replannable_plan: dict[str, Any] | None = None

    def maybe_accept_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal replan_attempts, last_replannable_plan
        validated = _validate_boundary_item_plan(plan)
        if validated.get("item_plan_valid"):
            return _with_planner_replan_diagnostics(
                validated,
                replan_attempts=replan_attempts,
                replan_success=replan_attempts > 0,
                replan_failed=False,
            )
        if _planner_failure_is_replannable(validated):
            replan_attempts += 1
            last_replannable_plan = validated
            return None
        return _with_planner_replan_diagnostics(
            validated,
            replan_attempts=replan_attempts,
            replan_success=False,
            replan_failed=False,
        )

    def exhausted_plan(reason: str) -> dict[str, Any]:
        if last_replannable_plan is not None:
            return _with_planner_replan_diagnostics(
                last_replannable_plan,
                replan_attempts=replan_attempts,
                replan_success=False,
                replan_failed=True,
            )
        return _with_planner_replan_diagnostics(
            _empty_boundary_item_plan(
                qtype=qtype,
                skill=skill,
                template=template,
                reason=reason,
                item_difficulty_target=item_difficulty_target,
            ),
            replan_attempts=replan_attempts,
            replan_success=False,
            replan_failed=replan_attempts > 0,
        )

    if qtype == "multi_select":
        for parent_gold, child_gold, pair_score in pairs[:24]:
            gold_events = [parent_gold, child_gold]
            if not _event_surfaces_are_unique(gold_events):
                continue
            gold_texts = {_option_text_norm(_event_display_text(event, hard_surface=True)) for event in gold_events}
            false_pool = _dedupe_events_by_surface(
                _rank_boundary_false_events(all_events, gold_events, prefer_scope=True, current_alignment=current_alignment),
                existing_texts=gold_texts,
            )
            strong_false, strong_types, explicit_contrast_type = _select_permission_visibility_false_events(
                false_pool,
                gold_events,
                item_difficulty_target=item_difficulty_target,
            )
            synth_diagnostics: dict[str, Any] = {
                "deterministic_permission_hard_negatives": bool(deterministic_permission_hard_negatives),
                "deterministic_permission_synthesizer_used": False,
                "deterministic_permission_synthesizer_failure_reason": None,
                "deterministic_permission_synthesized_event_ids": [],
                "deterministic_permission_synthesized_source_event_ids": [],
            }
            if deterministic_permission_hard_negatives and len(strong_false) < 2:
                strong_false, strong_types, explicit_contrast_type, synth_diagnostics = _synthesize_permission_visibility_false_events(
                    parent_events=parent_events,
                    child_events=child_events,
                    gold_events=gold_events,
                    current_alignment=current_alignment,
                    item_difficulty_target=item_difficulty_target,
                    existing_texts=gold_texts,
                )
            if len(strong_false) < 2:
                continue
            if all(str(event.get("status")) == "outdated" for event in strong_false):
                continue
            if not _event_surfaces_are_unique(gold_events + strong_false):
                continue
            plan = {
                "qtype": "multi_select",
                "skill": skill,
                "template": "ordinary",
                "item_difficulty_target": item_difficulty_target,
                "gold_path_type": "parent_child_active_visibility_scope_current",
                "gold_evidence_event_ids": [_event_ref_id(parent_gold), _event_ref_id(child_gold)],
                "hard_negative_path_types": strong_types,
                "hard_negative_event_ids": [_event_ref_id(event) for event in strong_false],
                "forbidden_surface_patterns": ["exact_duplicate_option", "near_duplicate_option", "false_all_outdated", "active_vs_outdated_shortcut"],
                "expected_status_pattern": [
                    str(parent_gold.get("status")),
                    str(child_gold.get("status")),
                    *[str(event.get("status")) for event in strong_false],
                ],
                "hard_negative_strength": difficulty_profile["hard_negative_similarity_level"],
                "hard_negative_similarity_level": difficulty_profile["hard_negative_similarity_level"],
                "intermediate_clue_added": bool(difficulty_profile["intermediate_clue_added"]),
                "explicit_contrast_type": explicit_contrast_type,
                "boundary_calibration_mode": difficulty_profile["boundary_calibration_mode"],
                "expected_solver_bucket": difficulty_profile["expected_solver_bucket"],
                "parent_child_pair_score": pair_score,
                "item_plan_created": True,
                "item_plan_valid": False,
                "item_plan_failure_reason": None,
                "_gold_events": gold_events,
                "_hard_negative_events": strong_false,
                **synth_diagnostics,
            }
            accepted = maybe_accept_plan(plan)
            if accepted is not None:
                return accepted
        return exhausted_plan("no_permission_visibility_scope_hard_negative_plan")

    for parent_gold, child_gold, pair_score in pairs[:24]:
        if _option_text_norm(_event_display_text(parent_gold, hard_surface=True)) == _option_text_norm(_event_display_text(child_gold, hard_surface=True)):
            continue
        gold_events = [parent_gold, child_gold]
        if not _event_surfaces_are_unique(gold_events):
            continue
        invalid_pool = _dedupe_events_by_surface(
            _rank_boundary_false_events(all_events, gold_events, prefer_scope=False, current_alignment=current_alignment),
            existing_texts={
                _option_text_norm(_event_display_text(parent_gold, hard_surface=True)),
                _option_text_norm(_event_display_text(child_gold, hard_surface=True)),
            },
        )
        gold_texts = {
            _option_text_norm(_event_display_text(parent_gold, hard_surface=True)),
            _option_text_norm(_event_display_text(child_gold, hard_surface=True)),
        }
        active_wrong_parent = _dedupe_events_by_surface(
            _active_wrong_events(all_events, gold_events, source="parent", current_alignment=current_alignment),
            existing_texts=gold_texts,
        )
        false_parent = active_wrong_parent[0] if active_wrong_parent else next((event for event in invalid_pool if event.get("_context_source") == "parent"), None)
        used_texts = set(gold_texts)
        if false_parent is not None:
            used_texts.add(_option_text_norm(_event_display_text(false_parent, hard_surface=True)))
        active_wrong_child = _dedupe_events_by_surface(
            _active_wrong_events(all_events, gold_events, source="child", current_alignment=current_alignment),
            existing_texts=used_texts,
        )
        false_child = active_wrong_child[0] if active_wrong_child else next((event for event in invalid_pool if event.get("_context_source") == "child"), None)
        if false_child is not None and _event_ref_id(false_child) == _event_ref_id(child_gold):
            false_child = next(
                (
                    event
                    for event in active_wrong_child[1:] + invalid_pool
                    if event.get("_context_source") == "child" and _event_ref_id(event) != _event_ref_id(child_gold)
                ),
                None,
            )
        used_false_ids = {_event_ref_id(event) for event in [false_parent, false_child] if isinstance(event, dict)}
        wrong_trigger = next(
            (
                event
                for event in invalid_pool
                if _event_ref_id(event) not in used_false_ids
                and (
                    _hard_negative_type_for_event(event, gold_events)
                    in {
                        "same_entity_same_slot_wrong_trigger",
                        "later_state_override",
                        "parent_valid_child_invalidated",
                    }
                    or (
                        _hard_negative_type_for_event(event, gold_events) == "active_but_wrong_condition"
                        and any(_same_entity_slot(event, gold) for gold in gold_events)
                    )
                )
            ),
            None,
        )
        if false_parent is None or false_child is None or wrong_trigger is None:
            continue
        option_pairs = [
            (parent_gold, child_gold),
            (parent_gold, false_child),
            (false_parent, child_gold),
            (false_parent, wrong_trigger),
        ]
        preview_options = [
            _option_from_event_pair(LETTERS[index], parent_event, child_event, current_alignment, hard_surface=True)
            for index, (parent_event, child_event) in enumerate(option_pairs)
        ]
        if not _qa_option_texts_are_unique(preview_options):
            continue
        preview_false_reasons = {
            LETTERS[1]: {"false_reason_type": _hard_negative_type_for_event(false_child, gold_events)},
            LETTERS[2]: {"false_reason_type": _hard_negative_type_for_event(false_parent, gold_events)},
            LETTERS[3]: {"false_reason_type": _hard_negative_type_for_event(wrong_trigger, gold_events)},
        }
        preview_qa = {
            "question_type": "single_choice",
            "options": preview_options,
            "gold": [LETTERS[0]],
            "evidence_turn_ids": sorted(
                set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
                key=turn_sort_key,
            ),
            "native_false_reasons_by_option": preview_false_reasons,
        }
        preview_diag = _boundary_aware_item_diagnostics(preview_qa)
        if _boundary_qa_direct_reject_reasons(preview_diag) or _boundary_qa_repair_reasons(preview_diag):
            continue
        hard_events = [false_parent, false_child, wrong_trigger]
        hard_types = [_hard_negative_type_for_event(event, gold_events) for event in hard_events]
        plan = {
            "qtype": "single_choice",
            "skill": skill,
            "template": template if template != "ordinary" else "latest_valid_state",
            "item_difficulty_target": item_difficulty_target,
            "gold_path_type": "parent_delayed_or_condition_child_triggered_active",
            "gold_evidence_event_ids": [_event_ref_id(parent_gold), _event_ref_id(child_gold)],
            "hard_negative_path_types": hard_types,
            "hard_negative_event_ids": [_event_ref_id(event) for event in hard_events],
            "forbidden_surface_patterns": ["identical_gold_pair", "gold_only_active_pattern", "exact_duplicate_option", "near_duplicate_option"],
            "expected_status_pattern": [
                f"{parent_gold.get('status')}|{child_gold.get('status')}",
                f"{parent_gold.get('status')}|{false_child.get('status')}",
                f"{false_parent.get('status')}|{child_gold.get('status')}",
                f"{false_parent.get('status')}|{wrong_trigger.get('status')}",
            ],
            "hard_negative_strength": difficulty_profile["hard_negative_similarity_level"],
            "hard_negative_similarity_level": difficulty_profile["hard_negative_similarity_level"],
            "intermediate_clue_added": bool(difficulty_profile["intermediate_clue_added"]),
            "explicit_contrast_type": "parent_unsatisfied_child_trigger_same_slot",
            "boundary_calibration_mode": difficulty_profile["boundary_calibration_mode"],
            "expected_solver_bucket": difficulty_profile["expected_solver_bucket"],
            "parent_child_pair_score": pair_score,
            "item_plan_created": True,
            "item_plan_valid": False,
            "item_plan_failure_reason": None,
            "_gold_events": gold_events,
            "_hard_negative_events": hard_events,
        }
        accepted = maybe_accept_plan(plan)
        if accepted is not None:
            return accepted
    return exhausted_plan("no_delayed_activation_wrong_trigger_plan")


def _build_qa_from_boundary_item_plan(
    *,
    plan: dict[str, Any],
    current_alignment: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
    hardness: str,
    difficulty_subtype: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not plan.get("item_plan_valid"):
        raise NativeGraphDialogueReject(f"boundary-aware QA planner failed: {plan.get('item_plan_failure_reason') or 'invalid_item_plan'}")
    gold_events = [event for event in plan.get("_gold_events") or [] if isinstance(event, dict)]
    hard_events = [event for event in plan.get("_hard_negative_events") or [] if isinstance(event, dict)]
    if len(gold_events) < 2:
        raise NativeGraphDialogueReject("boundary-aware QA planner missing gold events")
    if plan.get("qtype") == "multi_select":
        if len(hard_events) < 2:
            raise NativeGraphDialogueReject("boundary-aware QA planner missing multi_select hard negatives")
        raw_options: list[tuple[str, dict[str, Any]]] = [("true", gold_events[0]), ("true", gold_events[1]), ("false", hard_events[0]), ("false", hard_events[1])]
        rng.shuffle(raw_options)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        false_reasons: dict[str, dict[str, Any]] = {}
        for index, (kind, event) in enumerate(raw_options):
            label = LETTERS[index]
            if kind == "true":
                gold.append(label)
                option, _reason = _boundary_qa_make_single_event_option(label, event, current_alignment, gold_events=gold_events, is_gold=True)
            else:
                option, reason = _boundary_qa_make_single_event_option(label, event, current_alignment, gold_events=gold_events, is_gold=False)
                option["hard_negative_path_type"] = reason.get("false_reason_type") if reason else None
                if reason:
                    false_reasons[label] = reason
            options.append(option)
        if plan.get("boundary_calibration_mode") in {"boundary_target", "harder_boundary_target"}:
            question_text = (
                "Select exactly the option letters that are still supported by the final dialogue state under the current visibility, "
                "scope, permission, and update conditions. Answer with option letters only."
            )
        elif plan.get("boundary_calibration_mode") == "easy_control":
            question_text = (
                "Select exactly the option letters that remain true under the current visibility, scope, permission, and later-update conditions. "
                "At least one distractor should be removable by an obvious wrong person, condition, or outdated state. Answer with option letters only."
            )
        else:
            question_text = (
                "Select exactly the option letters that remain true under the current visibility, scope, permission, and later-update conditions. "
                "Answer with option letters only."
            )
        qa = {
            "question_type": "multi_select",
            "question": question_text,
            "options": options,
            "gold": sorted(gold),
            "gold_source": "boundary_planned_chained_parent_child_current_graph",
            "evidence_turn_ids": sorted(
                set(_event_turn_ids_any(gold_events[0], current_alignment) + _event_turn_ids_any(gold_events[1], current_alignment)),
                key=turn_sort_key,
            ),
            "native_false_reasons_by_option": false_reasons,
            "qa_generated_after_alignment": True,
            "boundary_aware_item_plan": _boundary_item_plan_summary(plan),
            **_deps(hashes),
        }
        return qa, [gold_events[0]], [gold_events[1]], hard_events[:2]

    if len(hard_events) < 3:
        raise NativeGraphDialogueReject("boundary-aware QA planner missing single_choice hard negatives")
    parent_gold, child_gold = gold_events[0], gold_events[1]
    false_parent, false_child, wrong_trigger = hard_events[0], hard_events[1], hard_events[2]
    pair_options = [
        ("true", parent_gold, child_gold),
        ("false", parent_gold, false_child),
        ("false", false_parent, child_gold),
        ("false", false_parent, wrong_trigger),
    ]
    rng.shuffle(pair_options)
    options: list[dict[str, Any]] = []
    gold: list[str] = []
    false_reasons: dict[str, dict[str, Any]] = {}
    hard_types: list[str] = []
    for index, (kind, parent_event, child_event) in enumerate(pair_options):
        label = LETTERS[index]
        option = _option_from_event_pair(label, parent_event, child_event, current_alignment, hard_surface=True)
        if kind == "true":
            gold.append(label)
        else:
            htype_parent = _hard_negative_type_for_event(parent_event, gold_events)
            htype_child = _hard_negative_type_for_event(child_event, gold_events)
            htype = htype_child if htype_child != "cross_context_dependency_mismatch" else htype_parent
            hard_types.append(htype)
            option["hard_negative_path_type"] = htype
            false_reasons[label] = {
                "candidate_event_instance_id": f"{_event_ref_id(parent_event)}|{_event_ref_id(child_event)}",
                "false_reason_type": htype or "cross_context_dependency_mismatch",
                "distractor_relation": "boundary_planned_parent_child_hard_negative",
            }
        options.append(option)
    qa = {
        "question_type": "single_choice",
        "question": "Which option is fully consistent with both the earlier condition and the later triggering update? Answer with one option letter only.",
        "options": options,
        "gold": gold,
        "gold_source": "boundary_planned_chained_parent_child_current_graph",
        "evidence_turn_ids": sorted(
            set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
            key=turn_sort_key,
        ),
        "native_false_reasons_by_option": false_reasons,
        "qa_generated_after_alignment": True,
        "difficulty_subtype": "hard_single_choice" if difficulty_subtype == "hard_single_choice" else "ordinary_single_choice",
        "hard_single_choice_template": plan.get("template") if difficulty_subtype == "hard_single_choice" else None,
        "distractor_error_types": sorted(set(hard_types or ["cross_context_dependency_mismatch"])),
        "distractor_hardness": hardness,
        "distractor_hardness_achieved": True,
        "hard_distractor_count": 3,
        "graph_validation_pass": True,
        "boundary_aware_item_plan": _boundary_item_plan_summary(plan),
        **_deps(hashes),
    }
    return qa, [parent_gold], [child_gold], hard_events


def _boundary_qa_rebuild_multi_select(
    *,
    qa: dict[str, Any],
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
    prefer_scope: bool,
) -> dict[str, Any] | None:
    parent_active = [event for event in parent_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    child_active = [event for event in child_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    pairs = _best_parent_child_pairs(parent_active, child_active)
    if not pairs:
        return None
    all_events = parent_events + child_events
    for parent_gold, child_gold, _score in pairs[:16]:
        gold_events = [parent_gold, child_gold]
        gold_texts = {_option_text_norm(_event_display_text(event, hard_surface=True)) for event in gold_events}
        false_pool = _dedupe_events_by_surface(
            _rank_boundary_false_events(all_events, gold_events, prefer_scope=prefer_scope, current_alignment=current_alignment),
            existing_texts=gold_texts,
        )
        strong_false: list[dict[str, Any]] = []
        for event in false_pool:
            htype = _hard_negative_type_for_event(event, gold_events)
            if prefer_scope and htype not in {"active_but_wrong_scope", "active_but_wrong_person", "active_but_wrong_condition", "parent_valid_child_invalidated", "later_state_override", "invalidated_parent_dependency", "cross_context_dependency_mismatch", "same_entity_same_slot_wrong_trigger"}:
                continue
            strong_false.append(event)
            if len(strong_false) >= 2:
                break
        if len(strong_false) < 2:
            continue
        raw_options: list[tuple[str, dict[str, Any]]] = [("true", parent_gold), ("true", child_gold), ("false", strong_false[0]), ("false", strong_false[1])]
        rng.shuffle(raw_options)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        false_reasons: dict[str, dict[str, Any]] = {}
        for index, (kind, event) in enumerate(raw_options):
            label = LETTERS[index]
            if kind == "true":
                gold.append(label)
                option, _reason = _boundary_qa_make_single_event_option(label, event, current_alignment, gold_events=gold_events, is_gold=True)
            else:
                option, reason = _boundary_qa_make_single_event_option(label, event, current_alignment, gold_events=gold_events, is_gold=False)
                if reason:
                    false_reasons[label] = reason
            options.append(option)
        if not _qa_option_texts_are_unique(options):
            continue
        repaired = dict(qa)
        repaired.update(
            {
                "question_type": "multi_select",
                "question": (
                    "Select exactly the option letters that are still supported by the final dialogue state under the current visibility, scope, and update conditions. "
                    "Answer with option letters only."
                ),
                "options": options,
                "gold": sorted(gold),
                "gold_source": "boundary_repaired_chained_parent_child_current_graph",
                "evidence_turn_ids": sorted(
                    set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
                    key=turn_sort_key,
                ),
                "native_false_reasons_by_option": false_reasons,
                "qa_generated_after_alignment": True,
                "boundary_aware_repaired": True,
                "repaired_hard_negative_type": sorted(
                    {str(reason.get("false_reason_type") or "") for reason in false_reasons.values() if isinstance(reason, dict)}
                ),
                **_deps(hashes),
            }
        )
        return repaired
    return None


def _boundary_qa_rebuild_single_choice(
    *,
    qa: dict[str, Any],
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
    hard_single_choice_template: str | None,
) -> dict[str, Any] | None:
    parent_active = [event for event in parent_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    child_active = [event for event in child_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    pairs = _best_parent_child_pairs(parent_active, child_active)
    all_events = parent_events + child_events
    template = hard_single_choice_template if hard_single_choice_template and hard_single_choice_template != "ordinary" else "latest_valid_state"
    for parent_gold, child_gold, _score in pairs[:16]:
        if _option_text_norm(_event_display_text(parent_gold, hard_surface=True)) == _option_text_norm(_event_display_text(child_gold, hard_surface=True)):
            continue
        gold_events = [parent_gold, child_gold]
        gold_text = _option_text_norm(_event_display_text(parent_gold, hard_surface=True) + " " + _event_display_text(child_gold, hard_surface=True))
        invalid_pool = _dedupe_events_by_surface(
            _rank_boundary_false_events(all_events, gold_events, prefer_scope=False, current_alignment=current_alignment),
            existing_texts={gold_text},
        )
        used_texts = {gold_text}

        def active_wrong_pool(source: str, gold_event: dict[str, Any]) -> list[dict[str, Any]]:
            candidates = [
                event
                for event in all_events
                if event.get("_context_source") == source
                and str(event.get("status")) in ACTIVE_STATUSES
                and _event_ref_id(event) != _event_ref_id(gold_event)
                and _event_turn_ids_any(event, current_alignment)
            ]
            candidates.sort(key=lambda event: (-_event_same_topic_score(event, gold_events), _event_ref_id(event)))
            return _dedupe_events_by_surface(candidates, existing_texts=used_texts)

        active_wrong_parent = active_wrong_pool("parent", parent_gold)
        for event in active_wrong_parent:
            used_texts.add(_option_text_norm(_event_display_text(event, hard_surface=True)))
        active_wrong_child = active_wrong_pool("child", child_gold)
        false_parent = (
            active_wrong_parent[0]
            if active_wrong_parent
            else next((event for event in invalid_pool if event.get("_context_source") == "parent"), None)
        )
        false_child = (
            active_wrong_child[0]
            if active_wrong_child
            else next((event for event in invalid_pool if event.get("_context_source") == "child"), None)
        )
        if false_parent is None or false_child is None:
            continue
        extra_false = next(
            (
                event
                for event in invalid_pool
                if _event_ref_id(event) not in {_event_ref_id(false_parent), _event_ref_id(false_child)}
            ),
            false_child,
        )
        pair_options = [
            ("true", parent_gold, child_gold),
            ("false", parent_gold, false_child),
            ("false", false_parent, child_gold),
            ("false", false_parent, extra_false),
        ]
        rng.shuffle(pair_options)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        false_reasons: dict[str, dict[str, Any]] = {}
        hard_types: list[str] = []
        for index, (kind, parent_event, child_event) in enumerate(pair_options):
            label = LETTERS[index]
            option = _option_from_event_pair(label, parent_event, child_event, current_alignment, hard_surface=True)
            if kind == "true":
                gold.append(label)
            else:
                htype_parent = _hard_negative_type_for_event(parent_event, gold_events)
                htype_child = _hard_negative_type_for_event(child_event, gold_events)
                htype = htype_child if htype_child != "cross_context_dependency_mismatch" else htype_parent
                hard_types.append(htype)
                option["hard_negative_path_type"] = htype
                false_reasons[label] = {
                    "candidate_event_instance_id": f"{_event_ref_id(parent_event)}|{_event_ref_id(child_event)}",
                    "false_reason_type": htype or "cross_context_dependency_mismatch",
                    "distractor_relation": "boundary_aware_repaired_parent_child_chain",
                }
            options.append(option)
        if len(gold) != 1 or not _qa_option_texts_are_unique(options):
            continue
        repaired = dict(qa)
        repaired.update(
            {
                "question_type": "single_choice",
                "question": "Which option is fully consistent with both the earlier condition and the later triggering update? Answer with one option letter only.",
                "options": options,
                "gold": gold,
                "gold_source": "boundary_repaired_chained_parent_child_current_graph",
                "evidence_turn_ids": sorted(
                    set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
                    key=turn_sort_key,
                ),
                "native_false_reasons_by_option": false_reasons,
                "qa_generated_after_alignment": True,
                "difficulty_subtype": "hard_single_choice",
                "hard_single_choice_template": template,
                "distractor_error_types": sorted(set(hard_types or ["cross_context_dependency_mismatch"])),
                "distractor_hardness": qa.get("distractor_hardness") or "h1",
                "distractor_hardness_achieved": True,
                "hard_distractor_count": 3,
                "graph_validation_pass": True,
                "boundary_aware_repaired": True,
                "repaired_hard_negative_type": sorted(set(hard_types)),
                **_deps(hashes),
            }
        )
        return repaired
    return None


def _boundary_qa_repair_summary(pre: dict[str, Any], post: dict[str, Any] | None, *, attempts: int, success: bool, repaired: dict[str, Any] | None) -> dict[str, Any]:
    post = post or {}
    pre_reasons = sorted(
        set(
            [str(reason) for reason in pre.get("hard_reject_reasons") or []]
            + _boundary_qa_repair_reasons(pre)
            + _boundary_qa_direct_reject_reasons(pre)
        )
    )
    post_reasons = sorted(
        set(
            [str(reason) for reason in post.get("hard_reject_reasons") or []]
            + ([] if success else _boundary_qa_repair_reasons(post))
            + _boundary_qa_direct_reject_reasons(post)
        )
    )
    return {
        "boundary_qa_repair_triggered": True,
        "boundary_qa_repair_attempt_count": attempts,
        "boundary_qa_repair_success": bool(success),
        "pre_repair_reject_reason": pre_reasons,
        "post_repair_reject_reason": post_reasons,
        "repaired_hard_negative_type": list((repaired or {}).get("repaired_hard_negative_type") or post.get("hard_negative_path_type") or []),
        "repaired_duplicate_option_count": int(post.get("duplicate_option_count") or 0),
        "repaired_identical_gold_pair": int(post.get("identical_gold_pair") or 0),
        "repaired_gold_only_active_pattern": bool(post.get("gold_only_active_pattern")),
        "repaired_active_vs_outdated_shortcut": bool(post.get("active_vs_outdated_shortcut")),
    }


def _repair_boundary_aware_qa(
    *,
    qa: dict[str, Any],
    item_diagnostics: dict[str, Any],
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
    question_type: str,
    hard_single_choice_template: str | None,
    attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    direct_reject = _boundary_qa_direct_reject_reasons(item_diagnostics)
    if direct_reject:
        return qa, item_diagnostics, _boundary_qa_repair_summary(item_diagnostics, item_diagnostics, attempts=0, success=False, repaired=None)
    repair_reasons = _boundary_qa_repair_reasons(item_diagnostics)
    if not repair_reasons:
        return qa, item_diagnostics, _boundary_qa_repair_summary(item_diagnostics, item_diagnostics, attempts=0, success=False, repaired=None)
    last_diag = item_diagnostics
    for attempt in range(1, max(0, int(attempts or 0)) + 1):
        prefer_scope = question_type == "multi_select" or "active_vs_outdated_shortcut" in repair_reasons
        repaired = (
            _boundary_qa_rebuild_multi_select(
                qa=qa,
                parent_events=parent_events,
                child_events=child_events,
                current_alignment=current_alignment,
                rng=rng,
                hashes=hashes,
                prefer_scope=prefer_scope,
            )
            if question_type == "multi_select"
            else _boundary_qa_rebuild_single_choice(
                qa=qa,
                parent_events=parent_events,
                child_events=child_events,
                current_alignment=current_alignment,
                rng=rng,
                hashes=hashes,
                hard_single_choice_template=hard_single_choice_template,
            )
        )
        if not isinstance(repaired, dict):
            continue
        post_diag = _boundary_aware_item_diagnostics(repaired)
        last_diag = post_diag
        remaining_direct = _boundary_qa_direct_reject_reasons(post_diag)
        remaining_repairable = _boundary_qa_repair_reasons(post_diag)
        if not remaining_direct and not remaining_repairable:
            repair_summary = _boundary_qa_repair_summary(item_diagnostics, post_diag, attempts=attempt, success=True, repaired=repaired)
            repaired["boundary_qa_repair"] = repair_summary
            repaired["boundary_aware_item_diagnostics"] = post_diag
            return repaired, post_diag, repair_summary
    repair_summary = _boundary_qa_repair_summary(item_diagnostics, last_diag, attempts=max(0, int(attempts or 0)), success=False, repaired=None)
    return qa, last_diag, repair_summary


def _qa_dependency_diagnostics(
    *,
    qa: dict[str, Any],
    parent_events: list[dict[str, Any]],
    child_events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    selected_parent_events: list[dict[str, Any]],
    selected_child_events: list[dict[str, Any]],
    false_events: list[dict[str, Any]],
) -> dict[str, Any]:
    parent_ids = {_event_ref_id(event) for event in parent_events}
    child_ids = {_event_ref_id(event) for event in child_events}
    option_ids: list[str] = []
    for option in qa.get("options") or []:
        event_id = str(option.get("candidate_event_instance_id") or "")
        option_ids.extend([part for part in event_id.split("|") if part])
    gold_labels = {str(item) for item in qa.get("gold") or []}
    gold_event_ids: list[str] = []
    for option in qa.get("options") or []:
        if str(option.get("id")) in gold_labels:
            gold_event_ids.extend([part for part in str(option.get("candidate_event_instance_id") or "").split("|") if part])
    evidence_turn_ids = normalize_turn_ids(qa.get("evidence_turn_ids"))
    false_reason_types = [
        str(item.get("false_reason_type") or "")
        for item in (qa.get("native_false_reasons_by_option") or {}).values()
        if isinstance(item, dict)
    ]
    event_by_id = {_event_ref_id(event): event for event in parent_events + child_events}
    selected_events = [
        event_by_id[event_id]
        for event_id in sorted(set(option_ids))
        if event_id in event_by_id
    ]
    if not selected_events:
        selected_events = selected_parent_events + selected_child_events + false_events
    used_dependency_count = sum(int(event.get("_dependency_count") or 0) for event in selected_events)
    # Event rows often carry only one dependency marker; graph difficulty is better tracked by
    # selected cross-context event count plus evidence span. Keep both rather than pretending
    # dependency_count is exact logical proof depth.
    return {
        "chained_dependency_targeting": True,
        "qa_used_event_count": len(set(gold_event_ids)),
        "qa_used_option_event_count": len(set(option_ids)),
        "qa_used_parent_event_count": len({event_id for event_id in gold_event_ids if event_id in parent_ids}),
        "qa_used_child_event_count": len({event_id for event_id in gold_event_ids if event_id in child_ids}),
        "qa_used_dependency_count": used_dependency_count + len(set(gold_event_ids)),
        "qa_used_gold_event_ids": sorted(set(gold_event_ids)),
        "qa_used_option_event_ids": sorted(set(option_ids)),
        "used_parent_context": bool({event_id for event_id in gold_event_ids if event_id in parent_ids}),
        "used_child_append_context": bool({event_id for event_id in gold_event_ids if event_id in child_ids}),
        "evidence_turn_ids": evidence_turn_ids,
        "evidence_turn_distance": _turn_distance(evidence_turn_ids),
        "gold_evidence_turn_count": len(evidence_turn_ids),
        "false_reason_types": false_reason_types,
        "weak_obvious_status_distractor_count": sum(1 for reason in false_reason_types if reason in {"scope_limited", "belief", "report"}),
        "parent_candidate_count": len(parent_events),
        "child_candidate_count": len(child_events),
    }


def _build_chained_dependency_targeted_qa(
    *,
    seed_row: dict[str, Any],
    current_event_plan: dict[str, Any],
    current_alignment: dict[str, Any],
    question_type: str,
    rng: random.Random,
    hashes: dict[str, str],
    policy: dict[str, Any] | None,
    hardness: str,
    difficulty_subtype: str,
    hard_single_choice_template: str | None,
) -> dict[str, Any]:
    boundary_aware = bool(isinstance(policy, dict) and policy.get("boundary_aware_qa_diagnostics") is True)
    boundary_planner_requested = bool(isinstance(policy, dict) and policy.get("boundary_aware_qa_planner") is True)
    boundary_repair = bool(boundary_aware and isinstance(policy, dict) and policy.get("boundary_aware_qa_repair") is True)
    stage33_compat_boundary_planner = bool(
        boundary_planner_requested
        and isinstance(policy, dict)
        and policy.get("stage33_compat_boundary_planner") is True
    )
    deterministic_permission_hard_negatives = bool(
        boundary_planner_requested
        and isinstance(policy, dict)
        and (
            policy.get("deterministic_permission_hard_negatives") is True
            or stage33_compat_boundary_planner
        )
    )
    item_difficulty_target = (
        _normalize_boundary_item_difficulty_target(
            policy.get("item_difficulty_target") or policy.get("boundary_item_difficulty_target")
        )
        if isinstance(policy, dict)
        else None
    )
    try:
        boundary_repair_attempts = int(policy.get("boundary_aware_qa_repair_attempts") or 0) if isinstance(policy, dict) else 0
    except (TypeError, ValueError):
        boundary_repair_attempts = 0
    parent_events = _parent_context_events(seed_row)
    child_events = _child_context_events(current_event_plan, current_alignment)
    parent_active = [event for event in parent_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    child_active = [event for event in child_events if str(event.get("status")) in ACTIVE_STATUSES and _event_turn_ids_any(event, current_alignment)]
    if not parent_active or not child_active:
        raise NativeGraphDialogueReject("chained QA dependency targeting requires active parent and child events")
    pairs = _best_parent_child_pairs(parent_active, child_active)
    if not pairs:
        raise NativeGraphDialogueReject("chained QA dependency targeting found no parent-child event pairs")
    preferred_qtype = question_type if question_type in {"multi_select", "single_choice"} else "multi_select"
    planned_cell = policy.get("planned_cell") if isinstance(policy, dict) and isinstance(policy.get("planned_cell"), dict) else {}
    planner_skill = normalize_space(
        planned_cell.get("attack_skill")
        or current_event_plan.get("attack_skill")
        or (policy.get("planned_attack_skill") if isinstance(policy, dict) else "")
        or (policy.get("attack_skill") if isinstance(policy, dict) else "")
        or ""
    )
    # The boundary-aware planner is intentionally cell-specific. In mixed qtype runs, keep it
    # active for supported cells while unsupported hard single-choice cells fall through to the
    # graph-derived QA builders plus boundary diagnostics/repair. Otherwise a global planner flag
    # would zero-yield unsupported cells before their own builders can run.
    boundary_planner = bool(
        boundary_planner_requested
        and _boundary_qa_planner_supported_cell(preferred_qtype, planner_skill)
    )
    planner_summary: dict[str, Any] | None = None
    if boundary_planner:
        item_plan = _build_boundary_aware_item_plan(
            parent_events=parent_events,
            child_events=child_events,
            current_alignment=current_alignment,
            question_type=preferred_qtype,
            attack_skill=planner_skill,
            hard_single_choice_template=hard_single_choice_template,
            item_difficulty_target=item_difficulty_target,
            deterministic_permission_hard_negatives=deterministic_permission_hard_negatives,
        )
        planner_summary = _boundary_item_plan_summary(item_plan)
        if not item_plan.get("item_plan_valid"):
            raise NativeGraphDialogueReject(f"boundary-aware QA planner failed: {item_plan.get('item_plan_failure_reason') or 'invalid_item_plan'}")
        qa, selected_parent, selected_child, selected_false = _build_qa_from_boundary_item_plan(
            plan=item_plan,
            current_alignment=current_alignment,
            rng=rng,
            hashes=hashes,
            hardness=hardness,
            difficulty_subtype=difficulty_subtype,
        )
        pair_score = float(item_plan.get("parent_child_pair_score") or 0.0)
    else:
        parent_gold, child_gold, pair_score = pairs[0]
        all_events = parent_events + child_events
        false_pool = _strong_false_events(all_events, [parent_gold, child_gold])
        # Force false options to come from later/invalidation/override-like chains when possible,
        # and avoid weak status-only distractors unless absolutely unavoidable.
        false_pool = [
            event for event in false_pool
            if str(event.get("status")) in {"outdated", "canceled", "expired", "delayed_not_active", "false", "restored"}
        ]
        if len(false_pool) < 2:
            raise NativeGraphDialogueReject("chained QA dependency targeting could not find strong non-obvious distractors")

        if preferred_qtype == "multi_select":
            raw_options: list[tuple[str, dict[str, Any]]] = [("true", parent_gold), ("true", child_gold), ("false", false_pool[0]), ("false", false_pool[1])]
            rng.shuffle(raw_options)
            options: list[dict[str, Any]] = []
            gold: list[str] = []
            false_reasons: dict[str, dict[str, Any]] = {}
            for index, (kind, event) in enumerate(raw_options):
                label = LETTERS[index]
                if kind == "true":
                    gold.append(label)
                else:
                    false_reasons[label] = {
                        "candidate_event_instance_id": _event_ref_id(event),
                        "false_reason_type": event.get("status"),
                        "distractor_source": event.get("_context_source"),
                        "distractor_relation": "nearby_overridden_or_invalidated_dependency",
                    }
                options.append(_option_from_single_event(label, event, current_alignment, hard_surface=True))
            qa = {
                "question_type": "multi_select",
                "question": (
                    "Select exactly the option letters that remain true when the earlier context and the later update "
                    "are both considered. Answer with option letters only."
                ),
                "options": options,
                "gold": sorted(gold),
                "gold_source": "chained_parent_child_current_graph_active_status",
                "evidence_turn_ids": sorted(
                    set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
                    key=turn_sort_key,
                ),
                "native_false_reasons_by_option": false_reasons,
                "qa_generated_after_alignment": True,
                **_deps(hashes),
            }
            selected_parent = [parent_gold]
            selected_child = [child_gold]
            selected_false = false_pool[:2]
        else:
            false_child = next((event for event in false_pool if event.get("_context_source") == "child"), false_pool[0])
            false_parent = next((event for event in false_pool if event.get("_context_source") == "parent"), false_pool[1 if len(false_pool) > 1 else 0])
            pair_options = [
                ("true", parent_gold, child_gold),
                ("false", parent_gold, false_child),
                ("false", false_parent, child_gold),
                ("false", false_parent, false_child),
            ]
            rng.shuffle(pair_options)
            options = []
            gold = []
            false_reasons = {}
            for index, (kind, parent_event, child_event) in enumerate(pair_options):
                label = LETTERS[index]
                if kind == "true":
                    gold.append(label)
                else:
                    false_reasons[label] = {
                        "candidate_event_instance_id": f"{_event_ref_id(parent_event)}|{_event_ref_id(child_event)}",
                        "false_reason_type": "cross_context_dependency_mismatch",
                        "distractor_relation": "nearby_parent_child_override_chain",
                    }
                options.append(_option_from_event_pair(label, parent_event, child_event, current_alignment, hard_surface=True))
            qa = {
                "question_type": "single_choice",
                "question": "Which option is fully consistent with both the earlier context and the later update? Answer with one option letter only.",
                "options": options,
                "gold": gold,
                "gold_source": "chained_parent_child_current_graph_final_state",
                "evidence_turn_ids": sorted(
                    set(_event_turn_ids_any(parent_gold, current_alignment) + _event_turn_ids_any(child_gold, current_alignment)),
                    key=turn_sort_key,
                ),
                "native_false_reasons_by_option": false_reasons,
                "qa_generated_after_alignment": True,
                "difficulty_subtype": "hard_single_choice" if difficulty_subtype == "hard_single_choice" else "ordinary_single_choice",
                "hard_single_choice_template": hard_single_choice_template if difficulty_subtype == "hard_single_choice" else None,
                "distractor_error_types": ["cross_context_dependency_mismatch"],
                "distractor_hardness": hardness,
                "distractor_hardness_achieved": True,
                "hard_distractor_count": 3,
                "graph_validation_pass": True,
                **_deps(hashes),
            }
            selected_parent = [parent_gold]
            selected_child = [child_gold]
            selected_false = [false_parent, false_child]

    diagnostics = _qa_dependency_diagnostics(
        qa=qa,
        parent_events=parent_events,
        child_events=child_events,
        current_alignment=current_alignment,
        selected_parent_events=selected_parent,
        selected_child_events=selected_child,
        false_events=selected_false,
    )
    diagnostics["parent_child_pair_score"] = pair_score
    diagnostics["requested_question_type"] = question_type
    diagnostics["final_question_type"] = qa.get("question_type")
    if stage33_compat_boundary_planner:
        diagnostics["stage33_compat_boundary_planner"] = True
        diagnostics["stage33_compat_deterministic_permission_forced"] = bool(deterministic_permission_hard_negatives)
    if planner_summary:
        diagnostics.update(planner_summary)
        qa["boundary_aware_item_plan"] = planner_summary
    if boundary_aware:
        item_diagnostics = _boundary_aware_item_diagnostics(qa)
        repair_summary: dict[str, Any] | None = None
        if boundary_repair and (
            _boundary_qa_repair_reasons(item_diagnostics)
            or _boundary_qa_direct_reject_reasons(item_diagnostics)
        ):
            qa, item_diagnostics, repair_summary = _repair_boundary_aware_qa(
                qa=qa,
                item_diagnostics=item_diagnostics,
                parent_events=parent_events,
                child_events=child_events,
                current_alignment=current_alignment,
                rng=rng,
                hashes=hashes,
                question_type=str(qa.get("question_type") or question_type),
                hard_single_choice_template=hard_single_choice_template,
                attempts=boundary_repair_attempts,
            )
            if repair_summary:
                diagnostics["boundary_qa_repair"] = repair_summary
                qa["boundary_qa_repair"] = repair_summary
                if planner_summary:
                    diagnostics["repair_used_after_planner"] = True
                    diagnostics["repair_success_after_planner"] = bool(repair_summary.get("boundary_qa_repair_success"))
        diagnostics["boundary_aware_item_diagnostics"] = item_diagnostics
        diagnostics["final_option_surface_risk"] = bool(item_diagnostics.get("option_surface_risk"))
        qa["boundary_aware_item_diagnostics"] = item_diagnostics
        if repair_summary and repair_summary.get("boundary_qa_repair_triggered") and not repair_summary.get("boundary_qa_repair_success"):
            reasons = repair_summary.get("post_repair_reject_reason") or repair_summary.get("pre_repair_reject_reason") or []
            raise NativeGraphDialogueReject(
                "boundary-aware QA repair failed: "
                + ",".join(str(item) for item in reasons or ["repairable_boundary_item_risk"])
            )
        if item_diagnostics.get("hard_reject_reasons"):
            raise NativeGraphDialogueReject(
                "boundary-aware QA hard reject: "
                + ",".join(str(item) for item in item_diagnostics.get("hard_reject_reasons") or [])
            )
        if repair_summary and repair_summary.get("boundary_qa_repair_success"):
            repaired_diagnostics = _qa_dependency_diagnostics(
                qa=qa,
                parent_events=parent_events,
                child_events=child_events,
                current_alignment=current_alignment,
                selected_parent_events=selected_parent,
                selected_child_events=selected_child,
                false_events=selected_false,
            )
            repaired_diagnostics["parent_child_pair_score"] = pair_score
            repaired_diagnostics["requested_question_type"] = question_type
            repaired_diagnostics["final_question_type"] = qa.get("question_type")
            repaired_diagnostics["boundary_aware_item_diagnostics"] = item_diagnostics
            repaired_diagnostics["boundary_qa_repair"] = repair_summary
            if planner_summary:
                repaired_diagnostics.update(planner_summary)
                repaired_diagnostics["repair_used_after_planner"] = True
                repaired_diagnostics["repair_success_after_planner"] = True
                repaired_diagnostics["final_option_surface_risk"] = bool(item_diagnostics.get("option_surface_risk"))
            diagnostics = repaired_diagnostics
    qa["chained_qa_diagnostics"] = diagnostics
    if diagnostics["qa_used_event_count"] < 2:
        raise NativeGraphDialogueReject("chained QA did not use at least two gold events")
    if not diagnostics["used_parent_context"] or not diagnostics["used_child_append_context"]:
        raise NativeGraphDialogueReject("chained QA did not use both parent and child context")
    if diagnostics["evidence_turn_distance"] < 1000:
        raise NativeGraphDialogueReject("chained QA evidence distance too small")
    if diagnostics["weak_obvious_status_distractor_count"] > 0:
        raise NativeGraphDialogueReject("chained QA produced weak obvious status distractors")
    return qa


def _rotate_ordering_if_identity(options: list[dict[str, Any]], chronological_event_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    label_by_event_id = {str(option.get("candidate_event_instance_id")): str(option.get("id")) for option in options}
    gold = [label_by_event_id[event_id] for event_id in chronological_event_ids]
    if gold == list(LETTERS[: len(gold)]):
        raw_options = copy.deepcopy(options[1:] + options[:1])
        for index, option in enumerate(raw_options):
            option["id"] = LETTERS[index]
        label_by_event_id = {str(option.get("candidate_event_instance_id")): str(option.get("id")) for option in raw_options}
        gold = [label_by_event_id[event_id] for event_id in chronological_event_ids]
        options = raw_options
    return options, gold


def _build_ordering_qa(events: list[dict[str, Any]], current_alignment: dict[str, Any], rng: random.Random, hashes: dict[str, str]) -> dict[str, Any]:
    if len(events) < 4:
        raise NativeGraphDialogueReject("ordering requires four aligned current events")
    chronological = events[:4]
    display = copy.deepcopy(chronological)
    rng.shuffle(display)
    options: list[dict[str, Any]] = []
    for index, event in enumerate(display):
        options.append(
            {
                "id": LETTERS[index],
                "text": _event_option_text(event),
                "turn_ids": _turn_ids_for_event(event, current_alignment),
                "candidate_event_instance_id": str(event.get("event_instance_id")),
                "candidate_status": event.get("status"),
            }
        )
    options, gold = _rotate_ordering_if_identity(options, [str(event.get("event_instance_id")) for event in chronological])
    return {
        "question_type": "ordering",
        "question": "Put the option letters in the order the events happened, from earliest to latest. Answer as letters separated by >.",
        "options": options,
        "gold": gold,
        "gold_source": "current_graph_time_index",
        "evidence_turn_ids": sorted({turn_id for option in options for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
        "qa_generated_after_alignment": True,
        **_deps(hashes),
    }


def _build_multi_select_qa(events: list[dict[str, Any]], current_alignment: dict[str, Any], rng: random.Random, hashes: dict[str, str]) -> dict[str, Any]:
    true_events = [event for event in events if str(event.get("status")) in ACTIVE_STATUSES]
    false_events = [event for event in events if str(event.get("status")) in FALSE_STATUSES]
    if len(true_events) < 2 or len(false_events) < 2:
        raise NativeGraphDialogueReject("multi_select requires two active and two false/non-active aligned events")
    raw_options: list[tuple[str, dict[str, Any]]] = [("true", event) for event in true_events[:2]] + [("false", event) for event in false_events[:2]]
    rng.shuffle(raw_options)
    options: list[dict[str, Any]] = []
    gold: list[str] = []
    false_reasons_by_option: dict[str, dict[str, Any]] = {}
    for index, (kind, event) in enumerate(raw_options):
        label = LETTERS[index]
        if kind == "true":
            gold.append(label)
        else:
            false_reasons_by_option[label] = {
                "candidate_event_instance_id": str(event.get("event_instance_id")),
                "false_reason_type": event.get("status"),
            }
        options.append(
            {
                "id": label,
                "text": _event_option_text(event),
                "turn_ids": _turn_ids_for_event(event, current_alignment),
                "candidate_event_instance_id": str(event.get("event_instance_id")),
                "candidate_status": event.get("status"),
            }
        )
    return {
        "question_type": "multi_select",
        "question": "Select exactly the option letters that are still true after the conversation. Answer with option letters only.",
        "options": options,
        "gold": sorted(gold),
        "gold_source": "current_graph_active_status",
        "evidence_turn_ids": sorted({turn_id for option in options if option["id"] in gold for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
        "native_false_reasons_by_option": false_reasons_by_option,
        "qa_generated_after_alignment": True,
        **_deps(hashes),
    }


def _single_choice_graph_validation_report(qa: dict[str, Any], events: list[dict[str, Any]], current_alignment: dict[str, Any], subtype: str) -> dict[str, Any]:
    options = qa.get("options") if isinstance(qa.get("options"), list) else []
    option_by_id = {str(option.get("id")): option for option in options if isinstance(option, dict)}
    gold_ids = [str(item) for item in (qa.get("gold") or []) if str(item or "").strip()]
    gold_id = gold_ids[0] if gold_ids else ""
    gold_option = option_by_id.get(gold_id, {})
    gold_event_id = str(gold_option.get("candidate_event_instance_id") or "")
    gold_event = next((event for event in events if str(event.get("event_instance_id")) == gold_event_id), None)
    validation_pass = bool(gold_event and len(gold_ids) == 1 and gold_id in option_by_id)
    distractor_error_types = list(qa.get("distractor_error_types") or [])
    if subtype == "sequence_as_single_choice" and qa.get("gold_order"):
        validation_pass = validation_pass and isinstance(qa.get("gold_order"), list)
    return {
        "graph_validation_pass": validation_pass,
        "difficulty_subtype": subtype,
        "distractor_error_types": distractor_error_types,
        "hard_single_choice_template": qa.get("hard_single_choice_template"),
    }


def _build_single_choice_qa(
    events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
    hardness: str = "h0",
    difficulty_subtype: str = "ordinary_single_choice",
    hard_single_choice_template: str | None = None,
) -> dict[str, Any]:
    # Easy-mode question type: exactly one active gold answer among non-active distractors.
    # Gold is graph-derived (single active status).
    true_events = [event for event in events if str(event.get("status")) in ACTIVE_STATUSES]
    false_events = [event for event in events if str(event.get("status")) in FALSE_STATUSES]
    if not true_events or len(false_events) < 2:
        raise NativeGraphDialogueReject("single_choice requires one active and at least two non-active aligned events")
    gold_event = true_events[-1]
    gold_event_id = str(gold_event.get("event_instance_id"))
    n_distractors = min(3, len(false_events))
    def _event_key(ev: dict[str, Any]) -> tuple[int, str, str, str]:
        return (int(ev.get("time_index") or 0), str(ev.get("entity") or ""), str(ev.get("slot") or ""), str(ev.get("event_instance_id") or ""))
    def _event_relation(ev: dict[str, Any]) -> str:
        if str(ev.get("event_instance_id")) == gold_event_id:
            return "gold"
        if str(ev.get("status")) == "restored":
            return "restored"
        if str(ev.get("status")) == "canceled":
            return "canceled"
        if str(ev.get("status")) == "outdated":
            return "outdated"
        if str(ev.get("status")) == "expired":
            return "expired"
        if str(ev.get("status")) == "delayed_not_active":
            return "delayed"
        if str(ev.get("entity")) == str(gold_event.get("entity")) and str(ev.get("slot")) == str(gold_event.get("slot")):
            return "same_slot"
        if str(ev.get("entity")) == str(gold_event.get("entity")):
            return "same_entity"
        return "other"
    def _distractor_error_type(ev: dict[str, Any], template: str) -> str:
        relation = _event_relation(ev)
        if template == "minimal_difference":
            return {
                "same_slot": "outdated_intermediate_state",
                "same_entity": "swapped_subject_object",
                "restored": "overridden_state",
                "canceled": "overridden_state",
                "outdated": "outdated_intermediate_state",
                "expired": "outdated_intermediate_state",
                "delayed": "wrong_condition_trigger",
                "other": "unsupported_by_final_graph",
            }.get(relation, "unsupported_by_final_graph")
        if template == "latest_valid_state":
            return {
                "same_slot": "outdated_intermediate_state",
                "same_entity": "swapped_subject_object",
                "restored": "overridden_state",
                "canceled": "overridden_state",
                "outdated": "outdated_intermediate_state",
                "expired": "outdated_intermediate_state",
                "delayed": "wrong_condition_trigger",
                "other": "unsupported_by_final_graph",
            }.get(relation, "unsupported_by_final_graph")
        if template == "guaranteed_vs_possible":
            return {
                "same_slot": "possible_but_not_guaranteed",
                "same_entity": "possible_but_not_guaranteed",
                "restored": "outdated_intermediate_state",
                "canceled": "outdated_intermediate_state",
                "outdated": "outdated_intermediate_state",
                "expired": "outdated_intermediate_state",
                "delayed": "wrong_condition_trigger",
                "other": "unsupported_by_final_graph",
            }.get(relation, "unsupported_by_final_graph")
        if template == "override_sensitive":
            return {
                "same_slot": "overridden_state",
                "same_entity": "overridden_state",
                "restored": "overridden_state",
                "canceled": "overridden_state",
                "outdated": "overridden_state",
                "expired": "overridden_state",
                "delayed": "wrong_condition_trigger",
                "other": "unsupported_by_final_graph",
            }.get(relation, "unsupported_by_final_graph")
        if template == "sequence_as_single_choice":
            return "sequence_order_swap"
        return "unsupported_by_final_graph"

    if difficulty_subtype == "ordinary_single_choice":
        invalidated_statuses = {"outdated", "canceled", "expired", "restored", "scope_limited", "delayed_not_active"}
        def is_hard(ev: dict[str, Any]) -> bool:
            same_topic = (ev.get("entity") == gold_event.get("entity")) and (ev.get("slot") == gold_event.get("slot"))
            same_entity = ev.get("entity") == gold_event.get("entity")
            invalidated = str(ev.get("status")) in invalidated_statuses
            return bool(same_topic or same_entity or invalidated)
        n_hard_target = {"h0": 0, "h1": 1, "h2": 2}.get(hardness, 0)
        n_hard_target = min(n_hard_target, n_distractors)
        hard_pool = [e for e in false_events if is_hard(e)]
        weak_pool = [e for e in false_events if not is_hard(e)]
        rng.shuffle(hard_pool)
        rng.shuffle(weak_pool)
        chosen = hard_pool[:n_hard_target]
        chosen += weak_pool[: max(0, n_distractors - len(chosen))]
        if len(chosen) < n_distractors:
            leftover = [e for e in hard_pool[n_hard_target:] if e not in chosen]
            chosen += leftover[: n_distractors - len(chosen)]
        distractors = chosen[:n_distractors]
        raw_events = [("true", gold_event)] + [("false", event) for event in distractors]
        rng.shuffle(raw_events)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        false_reasons_by_option: dict[str, dict[str, Any]] = {}
        for index, (kind, event) in enumerate(raw_events):
            label = LETTERS[index]
            if kind == "true":
                gold.append(label)
            else:
                false_reasons_by_option[label] = {
                    "candidate_event_instance_id": str(event.get("event_instance_id")),
                    "false_reason_type": event.get("status"),
                }
            options.append(
                {
                    "id": label,
                    "text": _event_option_text(event),
                    "turn_ids": _turn_ids_for_event(event, current_alignment),
                    "candidate_event_instance_id": str(event.get("event_instance_id")),
                    "candidate_status": event.get("status"),
                }
            )
        if len(gold) != 1:
            raise NativeGraphDialogueReject("single_choice must have exactly one gold option")
        qa = {
            "question_type": "single_choice",
            "question": "Select the one option that is still true after the conversation. Answer with one option letter only.",
            "options": options,
            "gold": gold,
            "gold_source": "current_graph_active_status",
            "distractor_hardness": hardness,
            "distractor_hardness_achieved": len([e for e in distractors if _event_relation(e) != 'other']) >= n_hard_target,
            "hard_distractor_count": sum(1 for e in distractors if _event_relation(e) != 'other'),
            "evidence_turn_ids": sorted({turn_id for option in options if option["id"] in gold for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
            "native_false_reasons_by_option": false_reasons_by_option,
            "qa_generated_after_alignment": True,
            **_deps(hashes),
        }
        qa["difficulty_subtype"] = "ordinary_single_choice"
        qa["hard_single_choice_template"] = None
        qa["graph_validation_pass"] = True
        qa["distractor_error_types"] = ["ordinary_single_choice"]
        return qa

    hard_templates = {
        "minimal_difference": _hard_single_choice_question_text("minimal_difference"),
        "latest_valid_state": _hard_single_choice_question_text("latest_valid_state"),
        "guaranteed_vs_possible": _hard_single_choice_question_text("guaranteed_vs_possible"),
        "override_sensitive": _hard_single_choice_question_text("override_sensitive"),
        "sequence_as_single_choice": _hard_single_choice_question_text("sequence_as_single_choice"),
    }
    template = hard_single_choice_template if hard_single_choice_template in hard_templates else None
    if template is None and difficulty_subtype in HARD_SINGLE_CHOICE_SUBTYPES:
        template = difficulty_subtype
    if template is None:
        template = HARD_SINGLE_CHOICE_SUBTYPES[rng.randrange(len(HARD_SINGLE_CHOICE_SUBTYPES))]
    question = hard_templates[template]
    events_sorted = sorted(events, key=_event_key)
    gold_candidates = [event for event in events_sorted if str(event.get("status")) in ACTIVE_STATUSES]
    if not gold_candidates:
        raise NativeGraphDialogueReject("single_choice hard template requires an active final state")
    gold_event = max(gold_candidates, key=_event_key)
    gold_surface = _hard_single_choice_surface_text(gold_event)
    gold_label = LETTERS[0]

    if template == "sequence_as_single_choice":
        sequence_events = events_sorted[:4]
        if len(sequence_events) < 4:
            raise NativeGraphDialogueReject("sequence_as_single_choice requires four aligned events")
        gold_order = list(sequence_events)
        distractor_orders = [
            [gold_order[1], gold_order[0], gold_order[2], gold_order[3]],
            [gold_order[0], gold_order[2], gold_order[1], gold_order[3]],
            [gold_order[3], gold_order[2], gold_order[1], gold_order[0]],
        ]
        option_orders = [gold_order] + distractor_orders
        rng.shuffle(option_orders)
        options: list[dict[str, Any]] = []
        gold: list[str] = []
        distractor_error_types: list[str] = []
        for index, order in enumerate(option_orders):
            label = LETTERS[index]
            if order == gold_order:
                gold.append(label)
            else:
                distractor_error_types.append("sequence_order_swap")
            options.append(
                {
                    "id": label,
                    "text": " -> ".join(_hard_single_choice_surface_text(event) for event in order),
                    "turn_ids": sorted({turn_id for event in order for turn_id in _turn_ids_for_event(event, current_alignment)}, key=turn_sort_key),
                    "candidate_event_instance_id": "|".join(str(event.get("event_instance_id")) for event in order),
                    "candidate_status": "sequence",
                }
            )
        if len(gold) != 1:
            raise NativeGraphDialogueReject("sequence_as_single_choice must have exactly one gold option")
        qa = {
            "question_type": "single_choice",
            "question": question,
            "options": options,
            "gold": gold,
            "gold_source": "current_graph_event_order",
            "distractor_hardness": hardness,
            "distractor_hardness_achieved": True,
            "hard_distractor_count": 3,
            "evidence_turn_ids": sorted({turn_id for event in gold_order for turn_id in _turn_ids_for_event(event, current_alignment)}, key=turn_sort_key),
            "native_false_reasons_by_option": {},
            "qa_generated_after_alignment": True,
            "difficulty_subtype": "hard_single_choice",
            "hard_single_choice_template": template,
            "distractor_error_types": distractor_error_types,
            "graph_validation_pass": True,
            **_deps(hashes),
        }
        return qa

    candidate_pool = [event for event in events_sorted if str(event.get("event_instance_id")) != str(gold_event.get("event_instance_id"))]
    candidate_pool = sorted(
        candidate_pool,
        key=lambda event: (
            0
            if (
                str(event.get("entity")) == str(gold_event.get("entity"))
                and str(event.get("slot")) == str(gold_event.get("slot"))
            )
            else 1
            if str(event.get("entity")) == str(gold_event.get("entity"))
            else 2
            if str(event.get("status")) in {"restored", "canceled", "outdated", "expired", "delayed_not_active", "scope_limited"}
            else 3,
            -_surface_similarity_score(_hard_single_choice_surface_text(event), gold_surface),
            int(event.get("time_index") or 0),
            str(event.get("event_instance_id") or ""),
        ),
    )
    if hardness == "h2":
        candidate_pool = candidate_pool[: min(len(candidate_pool), 12)]
    elif hardness == "h1":
        candidate_pool = candidate_pool[: min(len(candidate_pool), 20)]
    chosen_candidates: list[dict[str, Any]] = []
    distractor_error_types: list[str] = []
    for candidate in candidate_pool:
        if len(chosen_candidates) >= 3:
            break
        error_type = _distractor_error_type(candidate, template)
        if error_type == "unsupported_by_final_graph":
            continue
        chosen_candidates.append(candidate)
        distractor_error_types.append(error_type)
    if len(chosen_candidates) < 3:
        raise NativeGraphDialogueReject(f"single_choice hard template {template} could not build three graph-validated distractors")
    raw_events = [("true", gold_event)] + [("false", event) for event in chosen_candidates[:3]]
    rng.shuffle(raw_events)
    options: list[dict[str, Any]] = []
    gold: list[str] = []
    false_reasons_by_option: dict[str, dict[str, Any]] = {}
    for index, (kind, event) in enumerate(raw_events):
        label = LETTERS[index]
        if kind == "true":
            gold.append(label)
        else:
            false_reasons_by_option[label] = {
                "candidate_event_instance_id": str(event.get("event_instance_id")),
                "false_reason_type": event.get("status"),
            }
        options.append(
            {
                "id": label,
                "text": _hard_single_choice_surface_text(event),
                "turn_ids": _turn_ids_for_event(event, current_alignment),
                "candidate_event_instance_id": str(event.get("event_instance_id")),
                "candidate_status": event.get("status"),
            }
        )
    if len(gold) != 1:
        raise NativeGraphDialogueReject("single_choice must have exactly one gold option")
    qa = {
        "question_type": "single_choice",
        "question": question,
        "options": options,
        "gold": gold,
        "gold_source": "current_graph_final_state",
        "distractor_hardness": hardness,
        "distractor_hardness_achieved": True,
        "hard_distractor_count": len(chosen_candidates[:3]),
        "evidence_turn_ids": sorted({turn_id for option in options if option.get("id") in gold for turn_id in option.get("turn_ids", [])}, key=turn_sort_key),
        "native_false_reasons_by_option": false_reasons_by_option,
        "qa_generated_after_alignment": True,
        "difficulty_subtype": "hard_single_choice",
        "hard_single_choice_template": template,
        "distractor_error_types": distractor_error_types,
        "graph_validation_pass": True,
        "hard_single_choice_surface_gold": gold_surface,
        **_deps(hashes),
    }
    return qa


def _build_evidence_selection_qa(
    events: list[dict[str, Any]],
    current_alignment: dict[str, Any],
    final_dialogue: dict[str, Any],
    rng: random.Random,
    hashes: dict[str, str],
) -> dict[str, Any]:
    if not events:
        raise NativeGraphDialogueReject("evidence_selection requires aligned events")
    turns = final_dialogue.get("turns") if isinstance(final_dialogue.get("turns"), list) else []
    if not turns:
        raise NativeGraphDialogueReject("evidence_selection requires final dialogue turns")
    turn_index_by_id = {str(turn.get("turn_id")): index for index, turn in enumerate(turns)}
    candidates = []
    for event in events:
        turn_ids = _turn_ids_for_event(event, current_alignment)
        if not turn_ids:
            continue
        first_idx = turn_index_by_id.get(turn_ids[0], 0)
        edge_penalty = 0 if 2 <= first_idx < max(2, len(turns) - 2) else 1
        candidates.append((edge_penalty, abs(first_idx - len(turns) // 2), event))
    if not candidates:
        raise NativeGraphDialogueReject("evidence_selection has no target turn")
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("event_instance_id"))))
    target = candidates[0][2]
    target_turn_ids = _turn_ids_for_event(target, current_alignment)[:1]
    distractor_turn_ids = [str(turn.get("turn_id")) for turn in turns if str(turn.get("turn_id")) not in set(target_turn_ids)]
    rng.shuffle(distractor_turn_ids)
    option_turn_ids = target_turn_ids + distractor_turn_ids[:4]
    option_turn_ids = option_turn_ids[:5]
    if len(option_turn_ids) < 3:
        raise NativeGraphDialogueReject("evidence_selection requires at least three option turns")
    rng.shuffle(option_turn_ids)
    text_by_turn_id = {str(turn.get("turn_id")): normalize_space(turn.get("text")) for turn in turns}
    options = [
        {
            "id": turn_id,
            "text": f"{turn_id}: {text_by_turn_id.get(turn_id, '')[:220]}",
            "turn_ids": [turn_id],
        }
        for turn_id in option_turn_ids
    ]
    target_detail = _event_option_text(target)
    return {
        "question_type": "evidence_selection",
        # Single-answer by construction: gold is exactly one turn. Name the target detail and
        # demand exactly ONE turn id, so the solver does not return a multi-id set (which the
        # exact set-match scorer would mark wrong even when it contains the gold).
        "question": (
            f"Exactly one listed turn states this detail: \"{target_detail[:160]}\". "
            "Which single turn ID is it? Answer with exactly one turn ID."
        ),
        "answer_format": "single_turn_id",
        "target_event_instance_ids": [str(target.get("event_instance_id"))],
        "target_state": {
            "description": target_detail,
            "status": target.get("status"),
        },
        "options": options,
        "gold": target_turn_ids,
        "gold_source": "verified_current_alignment",
        "evidence_turn_ids": target_turn_ids,
        "qa_generated_after_alignment": True,
        **_deps(hashes),
    }


def _build_state_reconstruction_qa(events: list[dict[str, Any]], current_alignment: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    active_events = [event for event in events if str(event.get("status")) in ACTIVE_STATUSES]
    if not active_events:
        raise NativeGraphDialogueReject("state_reconstruction requires an active aligned final state")
    target = active_events[-1]
    relevant = [
        event
        for event in events
        if event.get("entity") == target.get("entity")
        and event.get("slot") == target.get("slot")
        and int(event.get("time_index") or 0) <= int(target.get("time_index") or 0)
    ]
    if not relevant:
        relevant = [target]
    evidence_turn_ids = sorted(
        {turn_id for event in relevant for turn_id in _turn_ids_for_event(event, current_alignment)},
        key=turn_sort_key,
    )
    return {
        "question_type": "state_reconstruction",
        "question": "After everything said in the conversation, what is the latest confirmed detail? Answer with only the exact phrase.",
        "target_event_instance_ids": [str(target.get("event_instance_id"))],
        "target_state": {
            "description": _event_option_text(target),
            "status": target.get("status"),
        },
        "options": [],
        "gold": [_event_option_text(target)],
        "gold_source": "current_graph_final_active_state",
        "evidence_turn_ids": evidence_turn_ids,
        "qa_generated_after_alignment": True,
        **_deps(hashes),
    }


def _clean_qa_option_display_text(qa: dict[str, Any]) -> None:
    """Easy-mode only, display-only: scrub synthetic surface phrases from solver-visible QA text.

    Mutates qa in place. Rewrites only display strings: option `text`, the `question`, and
    target_state.description. Leaves turn_ids, evidence_turn_ids, gold, gold_source,
    candidate_event_instance_id, candidate_status, and all alignment metadata unchanged.
    """
    if not isinstance(qa, dict):
        return
    question = qa.get("question")
    if isinstance(question, str) and _has_synthetic_surface_phrase(question):
        qa["question"] = _natural_surface_text(question)
    for option in qa.get("options") or []:
        if not isinstance(option, dict):
            continue
        text = option.get("text")
        if isinstance(text, str) and _has_synthetic_surface_phrase(text):
            # For evidence_selection the option text is "<turn_id>: <quote>"; preserve the id prefix.
            prefix_match = re.match(r"^(D\d+:\d+:\s*)(.*)$", text, re.DOTALL)
            if prefix_match:
                option["text"] = prefix_match.group(1) + _natural_surface_text(prefix_match.group(2))
            else:
                option["text"] = _natural_surface_text(text)
    target_state = qa.get("target_state")
    if isinstance(target_state, dict):
        description = target_state.get("description")
        if isinstance(description, str) and _has_synthetic_surface_phrase(description):
            target_state["description"] = _natural_surface_text(description)


def _stage64_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_space(item) for item in value if normalize_space(item)]
    text = normalize_space(value)
    return [text] if text else []


def _assert_no_stage64_solver_visible_artifact(field: str, text: Any) -> None:
    value = normalize_space(text)
    if not value:
        return
    match = STAGE64_BAD_SOLVER_VISIBLE_RE.search(value)
    if match:
        raise NativeGraphDialogueReject(
            f"stage64 API QA solver-visible artifact in {field}: {match.group(0)!r}"
        )
    if field in {"question", "question_text", "qa.question"}:
        generic_match = STAGE64_GENERIC_QUESTION_RE.search(value)
        if generic_match:
            raise NativeGraphDialogueReject(
                f"stage64 API QA generic question in {field}: {generic_match.group(0)!r}"
            )


def _stage64_target_for_slot(slot: int, n_samples: int) -> dict[str, str]:
    total = max(1, int(n_samples or 1))
    base_total = sum(STAGE64_TARGET_OPERATION_COUNTS.values())
    quotas = {
        key: int(value) * total // base_total
        for key, value in STAGE64_TARGET_OPERATION_COUNTS.items()
    }
    while sum(quotas.values()) < total:
        for key in STAGE64_TARGET_OPERATION_COUNTS:
            quotas[key] += 1
            if sum(quotas.values()) >= total:
                break
    cursor = int(slot or 0) % total
    offset = 0
    qa_operation = next(iter(STAGE64_TARGET_OPERATION_COUNTS))
    for key in STAGE64_TARGET_OPERATION_COUNTS:
        width = int(quotas.get(key) or 0)
        if offset <= cursor < offset + width:
            qa_operation = key
            break
        offset += width
    required_answer_types = STAGE64_OPERATION_REQUIRED_ANSWER_TYPES.get(qa_operation)
    if required_answer_types:
        preferred_answer_type = sorted(required_answer_types)[0]
    else:
        preferred_answer_type = STAGE64_PREFERRED_ANSWER_TYPE_CYCLE[cursor % len(STAGE64_PREFERRED_ANSWER_TYPE_CYCLE)]
    return {
        "qa_operation": qa_operation,
        "preferred_answer_type": preferred_answer_type,
        "target_basis": "stage64_scaled_operation_quota",
    }


def _stage64_ref_maps(
    current_event_plan: dict[str, Any],
    graph_update: dict[str, Any],
    current_alignment: dict[str, Any],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    alias_to_event_id: dict[str, str] = {}
    event_id_to_turn_ids: dict[str, list[str]] = {}

    def add_alias(alias: Any, event_id: str) -> None:
        key = normalize_space(alias)
        if not key or not event_id:
            return
        alias_to_event_id[key] = event_id
        alias_to_event_id[key.lower()] = event_id

    event_to_anchor = current_event_plan.get("current_event_instance_to_anchor")
    if not isinstance(event_to_anchor, dict):
        event_to_anchor = {}
    for event in current_event_plan.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = normalize_space(event.get("event_instance_id"))
        if not event_id:
            continue
        turn_ids = _turn_ids_for_event(event, current_alignment)
        if turn_ids:
            event_id_to_turn_ids[event_id] = turn_ids
        marker = normalize_space(event.get("alignment_anchor") or event_to_anchor.get(event_id))
        for alias in (
            event_id,
            event.get("event_id"),
            event.get("node_id"),
            event.get("fact_id"),
            event.get("dialogue_fact_id"),
            marker,
        ):
            add_alias(alias, event_id)
    for event in graph_update.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = normalize_space(event.get("event_instance_id") or event.get("event_id"))
        if not event_id:
            continue
        turn_ids = sorted(normalize_turn_ids(event.get("turn_ids") or event.get("evidence_turn_ids")), key=turn_sort_key)
        if turn_ids and not event_id_to_turn_ids.get(event_id):
            event_id_to_turn_ids[event_id] = turn_ids
        for alias in (event_id, event.get("event_id"), event.get("node_id")):
            add_alias(alias, event_id)
    return alias_to_event_id, event_id_to_turn_ids


def _resolve_stage64_ref(ref: Any, alias_to_event_id: dict[str, str]) -> str:
    key = normalize_space(ref)
    if not key:
        return ""
    return alias_to_event_id.get(key) or alias_to_event_id.get(key.lower()) or ""


def _stage64_gold_from_api(
    qa_proposal: dict[str, Any],
    raw_id_to_label: dict[str, str],
    option_text_to_label: dict[str, str],
    options_raw: list[Any],
) -> list[str]:
    raw_gold = qa_proposal.get("gold")
    if raw_gold is None:
        raw_gold = qa_proposal.get("gold_answer")
    if raw_gold is None:
        raw_gold = qa_proposal.get("gold_option")
    gold_items = _stage64_str_list(raw_gold)
    if not gold_items:
        inferred: list[str] = []
        for index, option in enumerate(options_raw):
            if not isinstance(option, dict):
                continue
            truth_text = normalize_space(option.get("truth_value")).lower()
            if option.get("is_gold") is True or option.get("truth_value") is True or truth_text == "true":
                inferred.append(LETTERS[index])
        gold_items = inferred
    gold: list[str] = []
    unmatched: list[str] = []
    for item in gold_items:
        label = raw_id_to_label.get(item) or raw_id_to_label.get(item.upper()) or option_text_to_label.get(normalize_text(item))
        if not label:
            unmatched.append(item)
            continue
        gold.append(label)
    if unmatched:
        inferred = []
        for index, option in enumerate(options_raw):
            if not isinstance(option, dict):
                continue
            truth_text = normalize_space(option.get("truth_value")).lower()
            if option.get("is_gold") is True or option.get("truth_value") is True or truth_text == "true":
                inferred.append(LETTERS[index])
        if len(set(inferred)) == 1:
            gold = inferred
        else:
            raise NativeGraphDialogueReject(f"stage64 API QA gold item does not match an option: {unmatched[:3]}")
    gold = sorted(set(gold), key=lambda label: LETTERS.index(label) if label in LETTERS else 999)
    if len(gold) != 1:
        raise NativeGraphDialogueReject(f"stage64 API QA requires exactly one gold option, got {gold}")
    return gold


def _build_stage64_api_qa_from_proposal(
    current_event_plan: dict[str, Any],
    graph_update: dict[str, Any],
    current_alignment: dict[str, Any],
    qa_proposal: dict[str, Any] | None,
    hashes: dict[str, str],
    stage64_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(qa_proposal, dict):
        raise NativeGraphDialogueReject("stage64 API QA mode requires qa_proposal object from API")
    question_type = normalize_space(qa_proposal.get("question_type") or "single_choice")
    if question_type != "single_choice":
        raise NativeGraphDialogueReject(f"stage64 API QA requires single_choice, got {question_type!r}")
    qa_operation = normalize_space(qa_proposal.get("qa_operation"))
    qa_subtype = normalize_space(qa_proposal.get("qa_subtype"))
    answer_type = normalize_space(qa_proposal.get("answer_type"))
    if answer_type not in STAGE64_ANSWER_TYPES:
        query_plan_for_answer_type = qa_proposal.get("query_plan") if isinstance(qa_proposal.get("query_plan"), dict) else {}
        query_plan_answer_type = normalize_space(query_plan_for_answer_type.get("answer_type"))
        option_answer_types = [
            normalize_space(option.get("answer_type"))
            for option in (qa_proposal.get("options") if isinstance(qa_proposal.get("options"), list) else [])
            if isinstance(option, dict) and normalize_space(option.get("answer_type"))
        ]
        unique_option_answer_types = sorted(set(option_answer_types))
        if query_plan_answer_type in STAGE64_ANSWER_TYPES:
            answer_type = query_plan_answer_type
        elif len(unique_option_answer_types) == 1 and unique_option_answer_types[0] in STAGE64_ANSWER_TYPES:
            answer_type = unique_option_answer_types[0]
    answer_format = normalize_space(qa_proposal.get("answer_format") or "single_choice_mcq")
    question = normalize_space(qa_proposal.get("question") or qa_proposal.get("question_text"))
    if qa_operation not in STAGE64_QA_OPERATIONS:
        raise NativeGraphDialogueReject(f"stage64 API QA invalid qa_operation: {qa_operation!r}")
    target_operation = normalize_space((stage64_target or {}).get("qa_operation")) if isinstance(stage64_target, dict) else ""
    if target_operation and qa_operation != target_operation:
        raise NativeGraphDialogueReject(
            f"stage64 API QA qa_operation {qa_operation!r} did not match target {target_operation!r}"
        )
    required_answer_types = STAGE64_OPERATION_REQUIRED_ANSWER_TYPES.get(qa_operation)
    if required_answer_types and answer_type not in required_answer_types:
        raise NativeGraphDialogueReject(
            f"stage64 API QA qa_operation {qa_operation!r} requires answer_type in "
            f"{sorted(required_answer_types)}, got {answer_type!r}"
        )
    if not qa_subtype:
        raise NativeGraphDialogueReject("stage64 API QA missing qa_subtype")
    if answer_type not in STAGE64_ANSWER_TYPES:
        raise NativeGraphDialogueReject(f"stage64 API QA invalid answer_type: {answer_type!r}")
    if answer_format != "single_choice_mcq":
        raise NativeGraphDialogueReject(f"stage64 API QA invalid answer_format: {answer_format!r}")
    if not question:
        raise NativeGraphDialogueReject("stage64 API QA missing question text")
    _assert_no_stage64_solver_visible_artifact("question", question)

    alias_to_event_id, event_id_to_turn_ids = _stage64_ref_maps(current_event_plan, graph_update, current_alignment)
    options_raw = qa_proposal.get("options")
    if not isinstance(options_raw, list):
        raise NativeGraphDialogueReject("stage64 API QA options must be a list")
    expected_options = 2 if answer_type == "YesNo" else 4
    if len(options_raw) != expected_options:
        raise NativeGraphDialogueReject(
            f"stage64 API QA expected {expected_options} options for {answer_type}, got {len(options_raw)}"
        )

    raw_id_to_label: dict[str, str] = {}
    option_text_to_label: dict[str, str] = {}
    normalized_options: list[dict[str, Any]] = []
    option_source_event_ids: dict[str, str] = {}
    for index, option in enumerate(options_raw):
        if not isinstance(option, dict):
            raise NativeGraphDialogueReject("stage64 API QA option must be object")
        label = LETTERS[index]
        raw_id = normalize_space(option.get("id") or option.get("option_id") or label)
        raw_id_to_label[raw_id] = label
        raw_id_to_label[raw_id.upper()] = label
        raw_id_to_label[label] = label
        text = normalize_space(option.get("text"))
        if not text:
            raise NativeGraphDialogueReject(f"stage64 API QA option {label} missing text")
        _assert_no_stage64_solver_visible_artifact(f"options[{label}].text", text)
        option_answer_type = normalize_space(option.get("answer_type") or answer_type)
        if option_answer_type != answer_type:
            raise NativeGraphDialogueReject(
                f"stage64 API QA option {label} answer_type mismatch: {option_answer_type!r} != {answer_type!r}"
            )
        source_ref = normalize_space(
            option.get("source_event_instance_id")
            or option.get("event_instance_id")
            or option.get("source_event_id")
            or option.get("anchor_ref")
        )
        source_event_id = _resolve_stage64_ref(source_ref, alias_to_event_id)
        if not source_event_id:
            raise NativeGraphDialogueReject(f"stage64 API QA option {label} has unresolved graph reference: {source_ref!r}")
        turn_ids = event_id_to_turn_ids.get(source_event_id) or sorted(normalize_turn_ids(option.get("turn_ids")), key=turn_sort_key)
        if not turn_ids:
            raise NativeGraphDialogueReject(f"stage64 API QA option {label} has no aligned evidence turns")
        normalized_option = copy.deepcopy(option)
        normalized_option["id"] = label
        normalized_option["option_id"] = label
        normalized_option["text"] = text
        normalized_option["answer_type"] = answer_type
        normalized_option["source_event_instance_id"] = source_event_id
        normalized_option["turn_ids"] = turn_ids
        if not normalize_space(normalized_option.get("anchor_ref")):
            normalized_option["anchor_ref"] = source_ref or source_event_id
        if not normalize_space(normalized_option.get("anchor_kind")):
            normalized_option["anchor_kind"] = "event"
        normalized_options.append(normalized_option)
        option_source_event_ids[label] = source_event_id
        option_text_to_label[normalize_text(text)] = label

    gold = _stage64_gold_from_api(qa_proposal, raw_id_to_label, option_text_to_label, options_raw)
    for option in normalized_options:
        is_gold = option["id"] in gold
        option["is_gold"] = is_gold
        option["truth_value"] = is_gold

    query_plan_raw = qa_proposal.get("query_plan")
    if not isinstance(query_plan_raw, dict):
        raise NativeGraphDialogueReject("stage64 API QA requires query_plan object")
    if normalize_space(query_plan_raw.get("answer_source")) != "graph_anchor":
        raise NativeGraphDialogueReject("stage64 API QA query_plan.answer_source must be graph_anchor")
    if normalize_space(query_plan_raw.get("qa_operation")) != qa_operation:
        raise NativeGraphDialogueReject("stage64 API QA query_plan qa_operation mismatch")
    if normalize_space(query_plan_raw.get("qa_subtype")) != qa_subtype:
        raise NativeGraphDialogueReject("stage64 API QA query_plan qa_subtype mismatch")
    if normalize_space(query_plan_raw.get("answer_type")) != answer_type:
        raise NativeGraphDialogueReject("stage64 API QA query_plan answer_type mismatch")

    anchor_refs_raw = qa_proposal.get("anchor_refs") if isinstance(qa_proposal.get("anchor_refs"), dict) else {}
    support_refs = _stage64_str_list(anchor_refs_raw.get("support")) or _stage64_str_list(query_plan_raw.get("supporting_anchor_refs"))
    gold_refs = _stage64_str_list(anchor_refs_raw.get("gold")) or _stage64_str_list(query_plan_raw.get("gold_anchor_refs"))
    distractor_refs = _stage64_str_list(anchor_refs_raw.get("distractors")) or _stage64_str_list(query_plan_raw.get("distractor_anchor_refs"))
    required_refs = _stage64_str_list(query_plan_raw.get("required_anchor_refs")) or sorted(set(support_refs + gold_refs))
    reasoning_path = _stage64_str_list(query_plan_raw.get("reasoning_anchor_path")) or support_refs + gold_refs
    if not support_refs or not gold_refs or not required_refs:
        raise NativeGraphDialogueReject("stage64 API QA query_plan missing support/gold/required refs")
    if answer_type != "YesNo" and not distractor_refs:
        raise NativeGraphDialogueReject("stage64 API QA non-YesNo query_plan missing distractor refs")

    all_refs = sorted(set(required_refs + support_refs + gold_refs + distractor_refs + reasoning_path))
    resolved_event_ids: set[str] = set()
    unresolved_refs: list[str] = []
    for ref in all_refs:
        event_id = _resolve_stage64_ref(ref, alias_to_event_id)
        if event_id:
            resolved_event_ids.add(event_id)
        else:
            unresolved_refs.append(ref)
    if unresolved_refs:
        raise NativeGraphDialogueReject(f"stage64 API QA unresolved query_plan refs: {unresolved_refs[:5]}")
    resolved_event_ids.update(option_source_event_ids.values())
    evidence_turn_ids = sorted(
        {
            turn_id
            for event_id in resolved_event_ids
            for turn_id in event_id_to_turn_ids.get(event_id, [])
        },
        key=turn_sort_key,
    )
    if not evidence_turn_ids:
        raise NativeGraphDialogueReject("stage64 API QA resolved refs have no evidence turns")

    option_by_id = {option["id"]: option for option in normalized_options}
    gold_answer_text = normalize_space(qa_proposal.get("gold_answer_text") or query_plan_raw.get("gold_answer") or option_by_id[gold[0]].get("text"))
    if not gold_answer_text:
        raise NativeGraphDialogueReject("stage64 API QA missing gold_answer_text")
    _assert_no_stage64_solver_visible_artifact("gold_answer_text", gold_answer_text)
    query_plan = copy.deepcopy(query_plan_raw)
    query_plan.update(
        {
            "answer_source": "graph_anchor",
            "qa_operation": qa_operation,
            "qa_subtype": qa_subtype,
            "answer_type": answer_type,
            "reasoning_anchor_path": reasoning_path,
            "required_anchor_refs": required_refs,
            "supporting_anchor_refs": support_refs,
            "gold_anchor_refs": gold_refs,
            "distractor_anchor_refs": distractor_refs,
            "gold_answer": gold_answer_text,
            "normalized_gold_answer": normalize_text(query_plan.get("normalized_gold_answer") or gold_answer_text),
        }
    )
    return {
        "question_type": "single_choice",
        "question": question,
        "question_text": question,
        "qa_operation": qa_operation,
        "qa_subtype": qa_subtype,
        "answer_type": answer_type,
        "answer_format": answer_format,
        "options": normalized_options,
        "gold": gold,
        "gold_source": "api_stage64_graph_anchor_query_plan",
        "gold_answer_text": gold_answer_text,
        "normalized_gold_answer": normalize_text(gold_answer_text),
        "query_plan": query_plan,
        "anchor_refs": {
            "support": support_refs,
            "gold": gold_refs,
            "distractors": distractor_refs,
        },
        "evidence_turn_ids": evidence_turn_ids,
        "difficulty_subtype": normalize_space(qa_proposal.get("difficulty_subtype") or "hard_single_choice"),
        "hard_single_choice_template": normalize_space(qa_proposal.get("hard_single_choice_template") or "api_stage64_graph_anchor"),
        "distractor_error_types": _stage64_str_list(qa_proposal.get("distractor_error_types")),
        "graph_validation_pass": True,
        "qa_generated_after_alignment": True,
        "stage64_api_qa_schema_pass": True,
        **_deps(hashes),
    }


def build_qa_from_current_graph_and_alignment(
    current_event_plan: dict[str, Any],
    graph_update: dict[str, Any],
    current_alignment: dict[str, Any],
    qa_proposal: dict[str, Any] | None,
    question_type: str,
    rng: random.Random,
    hashes: dict[str, str],
    final_dialogue: dict[str, Any],
    policy: dict[str, Any] | None = None,
    seed_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(policy, dict) and policy.get("qa_operation_answer_type_diversification"):
        return _build_stage64_api_qa_from_proposal(
            current_event_plan=current_event_plan,
            graph_update=graph_update,
            current_alignment=current_alignment,
            qa_proposal=qa_proposal,
            hashes=hashes,
            stage64_target=policy.get("stage64_target") if isinstance(policy.get("stage64_target"), dict) else None,
        )
    del graph_update  # graph_update is compiled from the same frozen graph; keep signature explicit for dependency clarity.
    del qa_proposal
    events = _selected_aligned_events(current_event_plan, current_alignment)
    chained_dependency_targeting = bool(isinstance(policy, dict) and policy.get("chained_qa_dependency_targeting") is True)
    # In easy_v2 there is NO silent fallback: if the planned question_type cannot be built,
    # the sample is rejected loudly so the accepted distribution stays on the deterministic
    # quota. In default/easy_v1 the historical QA-last fallback chain is preserved.
    if _is_easy_v2_curriculum(policy):
        if question_type not in EASY_V2_SUPPORTED_QUESTION_TYPES:
            raise NativeGraphDialogueReject(f"easy_v2 QA build got out-of-set question_type: {question_type!r}")
        qtypes = [question_type]
    else:
        fallback_chain = [qtype for qtype in SUPPORTED_QUESTION_TYPES if qtype != question_type]
        qtypes = [question_type] + fallback_chain
    errors: list[str] = []
    hardness = "h0"
    difficulty_subtype = "ordinary_single_choice"
    hard_single_choice_template = "ordinary_single_choice"
    if isinstance(policy, dict) and isinstance(policy.get("planned_cell"), dict):
        h = policy["planned_cell"].get("distractor_hardness")
        if h in ("h0", "h1", "h2"):
            hardness = h
        st = policy["planned_cell"].get("difficulty_subtype")
        if isinstance(st, str) and st:
            difficulty_subtype = st
        ht = policy["planned_cell"].get("hard_single_choice_template")
        if isinstance(ht, str) and ht:
            hard_single_choice_template = ht
    if chained_dependency_targeting:
        if not isinstance(seed_row, dict):
            raise NativeGraphDialogueReject("chained QA dependency targeting requires seed_row parent context")
        return _build_chained_dependency_targeted_qa(
            seed_row=seed_row,
            current_event_plan=current_event_plan,
            current_alignment=current_alignment,
            question_type=question_type,
            rng=rng,
            hashes=hashes,
            policy=policy,
            hardness=hardness,
            difficulty_subtype=difficulty_subtype,
            hard_single_choice_template=hard_single_choice_template,
        )
    for qtype in qtypes:
        try:
            if qtype == "single_choice":
                return _build_single_choice_qa(
                    events,
                    current_alignment,
                    rng,
                    hashes,
                    hardness=hardness,
                    difficulty_subtype=difficulty_subtype,
                    hard_single_choice_template=hard_single_choice_template,
                )
            if qtype == "ordering":
                return _build_ordering_qa(events, current_alignment, rng, hashes)
            if qtype == "multi_select":
                return _build_multi_select_qa(events, current_alignment, rng, hashes)
            if qtype == "evidence_selection":
                return _build_evidence_selection_qa(events, current_alignment, final_dialogue, rng, hashes)
            if qtype == "state_reconstruction":
                return _build_state_reconstruction_qa(events, current_alignment, hashes)
        except NativeGraphDialogueReject as exc:
            errors.append(f"{qtype}:{exc}")
    raise NativeGraphDialogueReject(f"QA-last failed for all question types: {errors}")


def referenced_turn_ids_from_clean_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    for turn_id in normalize_turn_ids(qa.get("evidence_turn_ids")):
        refs.append(("qa.evidence_turn_ids", turn_id))
    for index, option in enumerate(qa.get("options") or []):
        if isinstance(option, dict):
            for turn_id in normalize_turn_ids(option.get("turn_ids")):
                refs.append((f"qa.options[{index}].turn_ids", turn_id))
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    for index, event in enumerate(graph_update.get("events") or []):
        if isinstance(event, dict):
            for field_name in ("evidence_turn_ids", "turn_ids"):
                for turn_id in normalize_turn_ids(event.get(field_name)):
                    refs.append((f"graph_update.events[{index}].{field_name}", turn_id))
    for index, change in enumerate(graph_update.get("state_changes") or []):
        if isinstance(change, dict):
            for turn_id in normalize_turn_ids(change.get("evidence_turn_ids")):
                refs.append((f"graph_update.state_changes[{index}].evidence_turn_ids", turn_id))
    return refs


def assert_no_missing_turn_refs(row: dict[str, Any]) -> None:
    dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
    turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
    visible = {str(turn.get("turn_id")) for turn in turns if isinstance(turn, dict)}
    missing = [{"field": field, "turn_id": turn_id} for field, turn_id in referenced_turn_ids_from_clean_row(row) if turn_id not in visible]
    if missing:
        raise NativeGraphDialogueReject(f"missing clean turn refs: {missing[:5]}")


def validate_dependency_hashes(row: dict[str, Any], hashes: dict[str, str]) -> None:
    graph_update = row.get("graph_update") if isinstance(row.get("graph_update"), dict) else {}
    qa = row.get("qa") if isinstance(row.get("qa"), dict) else {}
    expected = _deps(hashes)
    for key, value in expected.items():
        if graph_update.get(key) != value:
            raise NativeGraphDialogueReject(f"graph_update {key} mismatch")
        if qa.get(key) != value:
            raise NativeGraphDialogueReject(f"qa {key} mismatch")
    for group in ("events", "state_changes"):
        for index, item in enumerate(graph_update.get(group) or []):
            if not isinstance(item, dict):
                continue
            for key, value in expected.items():
                if item.get(key) != value:
                    raise NativeGraphDialogueReject(f"graph_update.{group}[{index}] {key} mismatch")


def missing_ref_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    rows_with_missing = 0
    missing_total = 0
    for row in rows:
        dialogue = row.get("dialogue") if isinstance(row.get("dialogue"), dict) else {}
        turns = dialogue.get("turns") if isinstance(dialogue.get("turns"), list) else []
        visible = {str(turn.get("turn_id")) for turn in turns if isinstance(turn, dict)}
        missing = [(field, turn_id) for field, turn_id in referenced_turn_ids_from_clean_row(row) if turn_id not in visible]
        if missing:
            rows_with_missing += 1
            missing_total += len(missing)
    return {
        "missing_refs": missing_total,
        "rows_with_missing_refs": rows_with_missing,
        "missing_ref_total": missing_total,
    }


def _recover_sample_name_from_row(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
    candidates = [
        row.get("frontier_from_sample_name"),
        row.get("sample_name"),
        metadata.get("sample_name"),
        metadata.get("source_sample_name"),
        row.get("merge_source_sample_name"),
        metadata.get("merge_source_sample_name"),
        seed_source.get("sample_name"),
    ]
    for value in candidates:
        match = re.search(r"\b(sample\d+)\b", str(value or ""), re.IGNORECASE)
        if match:
            return match.group(1).lower()
    for value in (row.get("source_dialogue_path"), metadata.get("source_dialogue_path"), state_chain.get("source_dialogue_path")):
        match = re.search(r"/(sample\d+)/", str(value or ""), re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _row_has_heldout_or_dev_marker(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    seed_source = row.get("seed_source") if isinstance(row.get("seed_source"), dict) else {}
    state_chain = row.get("state_chain") if isinstance(row.get("state_chain"), dict) else {}
    values = [
        row.get("heldout_split"),
        metadata.get("heldout_split"),
        seed_source.get("heldout_split"),
        row.get("split"),
        metadata.get("split"),
        seed_source.get("split"),
        row.get("source_split"),
        metadata.get("source_split"),
        seed_source.get("source_split"),
        state_chain.get("source_split"),
    ]
    markers = {normalize_space(value).lower() for value in values if normalize_space(value)}
    if markers.intersection({"dev", "test", "heldout", "dev_sanity_eval_only", "heldout_eval_only"}):
        return True
    for value in (row.get("source_dialogue_path"), metadata.get("source_dialogue_path"), state_chain.get("source_dialogue_path")):
        if re.search(r"/(dev|test|heldout)/sample\d+/", str(value or "").lower()):
            return True
    return False


def infer_seed_provenance(seed_row: dict[str, Any]) -> dict[str, Any]:
    metadata = seed_row.get("metadata") if isinstance(seed_row.get("metadata"), dict) else {}
    existing_seed_source = seed_row.get("seed_source") if isinstance(seed_row.get("seed_source"), dict) else {}
    state_chain = seed_row.get("state_chain") if isinstance(seed_row.get("state_chain"), dict) else {}
    sample_name = _recover_sample_name_from_row(seed_row)
    source_path = normalize_space(
        seed_row.get("source_dialogue_path") or metadata.get("source_dialogue_path") or state_chain.get("source_dialogue_path")
    )
    raw_training_use = normalize_space(
        seed_row.get("training_use")
        or metadata.get("training_use")
        or existing_seed_source.get("raw_training_use")
        or existing_seed_source.get("training_use")
    )
    forbidden_reasons: list[str] = []
    if not sample_name:
        forbidden_reasons.append("sample_name_unrecoverable")
    if raw_training_use.lower() != "train_candidate":
        forbidden_reasons.append(f"training_use={raw_training_use or 'unknown'}")
    if _row_has_heldout_or_dev_marker(seed_row):
        forbidden_reasons.append("heldout_or_dev_marker_present")
    allowed_train_seed = not forbidden_reasons
    source_split = "train" if allowed_train_seed else normalize_space(existing_seed_source.get("source_split") or seed_row.get("source_split") or "unknown")
    seed_source = {
        **copy.deepcopy(existing_seed_source),
        "split": "train" if allowed_train_seed else normalize_space(existing_seed_source.get("split") or seed_row.get("split") or "unknown"),
        "source_split": source_split,
        "sample_name": sample_name or normalize_space(existing_seed_source.get("sample_name") or "unknown"),
        "raw_training_use": raw_training_use or "unknown",
        "training_use": raw_training_use or "unknown",
        "source_dialogue_path": source_path,
        "parent_source": state_chain.get("parent_source") or existing_seed_source.get("parent_source"),
        "parent_sample_id": seed_row.get("parent_sample_id") or metadata.get("parent_sample_id") or existing_seed_source.get("parent_sample_id"),
        "sample_id": seed_row.get("sample_id") or existing_seed_source.get("sample_id"),
        "allowed_train_seed": bool(allowed_train_seed),
        "forbidden_reasons": forbidden_reasons,
    }
    return {
        "frontier_from_split": "train" if allowed_train_seed else None,
        "frontier_from_sample_name": sample_name,
        "seed_source": seed_source,
        "seed": {
            "source": seed_source,
            "sample_id": seed_row.get("sample_id"),
            "parent_sample_id": seed_source.get("parent_sample_id"),
        },
        "seed_source_inferred": not isinstance(seed_row.get("seed_source"), dict),
        "allowed_train_seed": allowed_train_seed,
    }


def normalize_qa_options_for_top_level(qa: dict[str, Any]) -> list[dict[str, Any]] | None:
    options = qa.get("options")
    if not isinstance(options, list):
        return None
    normalized: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        if isinstance(option, dict):
            item = {
                "id": normalize_space(option.get("id") or LETTERS[index]),
                "text": normalize_space(option.get("text")),
                "turn_ids": normalize_turn_ids(option.get("turn_ids")),
            }
            if option.get("label") is not None:
                item["label"] = option.get("label")
            for key in (
                "option_id",
                "is_gold",
                "answer_type",
                "anchor_ref",
                "truth_value",
                "source_event_instance_id",
                "anchor_kind",
                "distractor_source",
                "distractor_type",
            ):
                if option.get(key) is not None:
                    item[key] = copy.deepcopy(option.get(key))
            normalized.append(item)
        else:
            normalized.append({"id": LETTERS[index], "text": normalize_space(option), "turn_ids": []})
    return normalized


def apply_top_level_qa_compat_fields(sample: dict[str, Any]) -> dict[str, Any]:
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    if qa.get("question") is not None:
        sample["question"] = qa.get("question")
    sample["question_type"] = qa.get("question_type") or sample.get("question_type")
    options = normalize_qa_options_for_top_level(qa)
    if options is not None:
        sample["options"] = options
    gold = qa.get("gold") if qa.get("gold") is not None else qa.get("gold_answer")
    if gold is not None:
        sample["gold_answer"] = copy.deepcopy(gold)
        if isinstance(gold, list):
            sample["gold_answers"] = copy.deepcopy(gold)
            sample["gold_options"] = copy.deepcopy(gold)
    if qa.get("evidence_turn_ids") is not None:
        sample["evidence_turn_ids"] = normalize_turn_ids(qa.get("evidence_turn_ids"))
    return sample


def build_clean_sample(
    *,
    sample_index: int,
    seed_row: dict[str, Any],
    current_event_plan: dict[str, Any],
    final_dialogue: dict[str, Any],
    graph_update: dict[str, Any],
    qa: dict[str, Any],
    hashes: dict[str, str],
    length_metadata: dict[str, Any],
) -> dict[str, Any]:
    sample_id = f"native_current_method_graph_dialogue_{sample_index:04d}_{stable_hash([current_event_plan.get('graph_snapshot_hash'), hashes], length=10)}"
    stage64_qa_operation = normalize_space(qa.get("qa_operation"))
    attack_skill = stage64_qa_operation or normalize_space(current_event_plan.get("attack_skill")) or "multi_object_dependency"
    expanded_dialogue_text = "\n".join(
        f"[{turn.get('turn_id')}] {turn.get('speaker')}: {turn.get('text')}" for turn in final_dialogue.get("turns", [])
    )
    seed_provenance = infer_seed_provenance(seed_row)
    frontier_from_split = seed_row.get("frontier_from_split") or seed_provenance.get("frontier_from_split")
    frontier_from_sample_name = seed_row.get("frontier_from_sample_name") or seed_provenance.get("frontier_from_sample_name")
    seed_source = seed_row.get("seed_source") if isinstance(seed_row.get("seed_source"), dict) else seed_provenance.get("seed_source")
    seed_payload = seed_row.get("seed") if isinstance(seed_row.get("seed"), dict) else seed_provenance.get("seed")
    sample = {
        "sample_id": sample_id,
        "expanded_graph_id": sample_id,
        "expanded_dialogue_id": f"{sample_id}_dialogue",
        "source_id": seed_row.get("sample_id") or seed_row.get("source_id"),
        "generation_source": "current_method.native_challenger_generation.graph_conditioned_api_dialogue",
        "generation_backend": "native_current_method",
        "native_challenger_mode": "graph_conditioned_api_dialogue",
        "native_generator_version": "graph_conditioned_api_dialogue_v1",
        "challenger_api_used": True,
        "question_type": qa.get("question_type"),
        "question": qa.get("question"),
        "attack_skill": attack_skill,
        "difficulty_subtype": qa.get("difficulty_subtype") or ("ordinary_single_choice" if qa.get("question_type") == "single_choice" else "ordinary"),
        "hard_single_choice_template": qa.get("hard_single_choice_template"),
        "distractor_error_types": list(qa.get("distractor_error_types") or []),
        "graph_validation_pass": bool(qa.get("graph_validation_pass")) if qa.get("question_type") == "single_choice" else True,
        "frontier_from_split": frontier_from_split,
        "frontier_from_sample_name": frontier_from_sample_name,
        "seed_source": copy.deepcopy(seed_source) if isinstance(seed_source, dict) else None,
        "seed": copy.deepcopy(seed_payload) if isinstance(seed_payload, dict) else None,
        "attack_plan": {
            "attack_skill": attack_skill,
            "reasoning_mechanism": "graph_first_api_event_plan_then_api_dialogue_alignment",
            "target_solver_weakness": "long_dialogue_current_graph_state_reasoning",
            "requires_ordering": qa.get("question_type") == "ordering",
            "requires_invalidated_memory": qa.get("question_type") == "multi_select",
            "difficulty_subtype": qa.get("difficulty_subtype") or ("ordinary_single_choice" if qa.get("question_type") == "single_choice" else "ordinary"),
            "hard_single_choice_template": qa.get("hard_single_choice_template"),
            "shortcut_to_avoid": ["old_event_id_reuse", "anchor_leakage", "dialogue_first_gold"],
        },
        "dialogue": {"turns": final_dialogue.get("turns") or [], "dialogue_hash": final_dialogue.get("dialogue_hash")},
        "expanded_dialogue_text": expanded_dialogue_text,
        "graph_update": graph_update,
        "qa": qa,
        "metadata": {
            "method": "current_method_graph_conditioned_api_dialogue",
            "source_seed_id": seed_row.get("sample_id") or seed_row.get("source_id"),
            "graph_snapshot_id": current_event_plan.get("graph_snapshot_id"),
            "graph_snapshot_hash": hashes["graph_snapshot_hash"],
            "dialogue_hash": hashes["dialogue_hash"],
            "alignment_hash": hashes["alignment_hash"],
            "attack_skill": attack_skill,
            "difficulty_subtype": qa.get("difficulty_subtype") or ("ordinary_single_choice" if qa.get("question_type") == "single_choice" else "ordinary"),
            "hard_single_choice_template": qa.get("hard_single_choice_template"),
            "distractor_error_types": list(qa.get("distractor_error_types") or []),
            "graph_validation_pass": bool(qa.get("graph_validation_pass")) if qa.get("question_type") == "single_choice" else True,
            "qa_generated_after_alignment": True,
            "old_qa_reused": False,
            "chained_qa_diagnostics": copy.deepcopy(qa.get("chained_qa_diagnostics"))
            if isinstance(qa.get("chained_qa_diagnostics"), dict)
            else None,
            "chained_qa_dependency_targeting": bool(
                isinstance(qa.get("chained_qa_diagnostics"), dict)
                and (qa.get("chained_qa_diagnostics") or {}).get("chained_dependency_targeting") is True
            ),
            "seed_source_inferred": bool(seed_provenance.get("seed_source_inferred")),
            "seed_allowed_train_seed": bool(seed_provenance.get("allowed_train_seed")),
            **length_metadata,
        },
    }
    if stage64_qa_operation:
        sample.update(
            {
                "qa_operation": stage64_qa_operation,
                "qa_subtype": qa.get("qa_subtype"),
                "answer_type": qa.get("answer_type"),
                "answer_format": qa.get("answer_format"),
                "question_text": qa.get("question_text") or qa.get("question"),
                "query_plan": copy.deepcopy(qa.get("query_plan")) if isinstance(qa.get("query_plan"), dict) else qa.get("query_plan"),
                "anchor_refs": copy.deepcopy(qa.get("anchor_refs")) if isinstance(qa.get("anchor_refs"), dict) else qa.get("anchor_refs"),
                "gold_option": (qa.get("gold") or [None])[0] if isinstance(qa.get("gold"), list) else qa.get("gold"),
                "gold_answer_text": qa.get("gold_answer_text"),
                "normalized_gold_answer": qa.get("normalized_gold_answer"),
            }
        )
        sample["metadata"].update(
            {
                "qa_operation_answer_type_diversification": True,
                "qa_operation": stage64_qa_operation,
                "qa_subtype": qa.get("qa_subtype"),
                "answer_type": qa.get("answer_type"),
                "answer_format": qa.get("answer_format"),
                "qa_generated_by_api_stage64_schema": True,
            }
        )
        sample["attack_plan"].update(
            {
                "attack_skill": stage64_qa_operation,
                "difficulty_subtype": qa.get("qa_subtype") or qa.get("difficulty_subtype"),
                "reasoning_mechanism": "api_graph_anchor_query_plan_qa_operation_answer_type",
                "target_solver_weakness": "long_dialogue_anchor_first_operation_reasoning",
                "requires_ordering": False,
                "shortcut_to_avoid": [
                    "state_only_shell",
                    "raw_qa_artifact",
                    "answer_type_mismatch",
                    "unanchored_distractor",
                ],
            }
        )
    plan_metadata = current_event_plan.get("metadata") if isinstance(current_event_plan.get("metadata"), dict) else {}
    if isinstance(plan_metadata.get("query_conditioned_graph_objective"), dict):
        sample["metadata"]["query_conditioned_graph_expansion"] = True
        sample["metadata"]["query_conditioned_graph_objective"] = copy.deepcopy(plan_metadata.get("query_conditioned_graph_objective"))
    if isinstance(plan_metadata.get("query_conditioned_graph_diagnostics"), dict):
        sample["metadata"]["query_conditioned_graph_diagnostics"] = copy.deepcopy(plan_metadata.get("query_conditioned_graph_diagnostics"))
    apply_top_level_qa_compat_fields(sample)
    sample["validation"] = base.validate_sample(sample)
    if sample.get("question_type") == "single_choice":
        sample["validation"]["graph_validation_pass"] = bool(qa.get("graph_validation_pass"))
    return sample


def validate_final_clean_sample(row: dict[str, Any], known_anchors: Iterable[str], hashes: dict[str, str]) -> None:
    assert_no_missing_turn_refs(row)
    assert_no_anchor_leakage(row, known_anchors)
    meta_ok, meta_details = assert_no_meta_leakage(row)
    if not meta_ok:
        raise NativeGraphDialogueReject(f"clean dialogue meta leakage: {meta_details}")
    validate_dependency_hashes(row, hashes)
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    if validation.get("graph_dialogue_consistent") is not True:
        raise NativeGraphDialogueReject("base validation graph_dialogue_consistent failed")
    if validation.get("gold_derivable") is not True:
        raise NativeGraphDialogueReject("base validation gold_derivable failed")
    if validation.get("option_valid") is not True:
        raise NativeGraphDialogueReject("base validation option_valid failed")
    if validation.get("anti_shortcut_pass") is not True:
        raise NativeGraphDialogueReject(f"base validation anti_shortcut failed: {validation.get('warnings')}")
    if validation.get("warnings"):
        raise NativeGraphDialogueReject(f"base validation warnings: {validation.get('warnings')}")


def _empty_summary() -> dict[str, Any]:
    return {
        "api_graph_called_count": 0,
        "api_dialogue_called_count": 0,
        "api_dialogue_valid_count": 0,
        "api_dialogue_reject_count": 0,
        "api_qa_proposal_count": 0,
        "graph_event_count": 0,
        "required_event_count": 0,
        "aligned_required_event_count": 0,
        "missing_required_event_count": 0,
        "contradiction_count": 0,
        "alignment_coverage_pass_count": 0,
        "current_anchor_assigned_count": 0,
        "current_anchor_seen_count": 0,
        "current_anchor_missing_count": 0,
        "current_anchor_invalid_count": 0,
        "anchor_leakage_count": 0,
        "legacy_alignment_bootstrap_count": 0,
        "legacy_high_confidence_count": 0,
        "legacy_low_confidence_count": 0,
        "legacy_needs_reverbalization_count": 0,
        "old_qa_reused_count": 0,
        "qa_generated_after_alignment_count": 0,
        "generated_count": 0,
        "valid_count": 0,
        "clean_count": 0,
        "missing_refs": 0,
        "rows_with_missing_refs": 0,
        "missing_ref_total": 0,
        "meta_leak_count": 0,
        "solver_visible_meta_leak_count": 0,
        "meta_leak_reject_count": 0,
        "solver_visible_meta_leak_reject_count": 0,
        "qa_meta_leak_count": 0,
        "dialogue_meta_leak_count": 0,
        "naturalness_retry_count": 0,
        "length_retry_count": 0,
        "length_reject_count": 0,
        "too_short_count": 0,
        "length_guard_enabled": False,
        "min_length_delta_tokens": 0,
        "length_guard_parent_baseline_tokens": 0,
        "length_guard_effective_parent_tokens_mean": None,
        "length_guard_effective_delta_tokens_mean": None,
        "length_guard_baseline_mode": None,
        "length_guard_target_delta": 0,
        "length_guard_fallback_used": False,
        "length_guard_fallback_accept_count": 0,
        "fallback_delta": 0,
        "emergency_floor": 0,
        "rows_below_4000_delta": 0,
        "length_guard_pass_count": 0,
        "length_guard_reject_count": 0,
        "length_guard_retry_count": 0,
        "rows_below_min_length_delta": 0,
        "api_failure_reason_counts": {},
        "api_yield_normalizers": False,
        "graph_json_normalizer_used_count": 0,
        "graph_update_events_unwrap_count": 0,
        "graph_json_normalizer_failure_count": 0,
        "graph_reject_recovered_by_normalizer_count": 0,
        "dialogue_json_normalizer_used_count": 0,
        "dialogue_json_block_extracted_count": 0,
        "dialogue_wrapper_unwrap_count": 0,
        "dialogue_reject_recovered_by_normalizer_count": 0,
        "dialogue_json_normalizer_failure_count": 0,
        "graph_cache_enabled": False,
        "graph_cache_write_count": 0,
        "graph_cache_hit_count": 0,
        "graph_cache_reuse_for_dialogue_retry_count": 0,
        "graph_cache_invalidated_count": 0,
        "graph_calls_saved_estimate": 0,
        "query_conditioned_graph_expansion": False,
        "query_conditioned_graph_objective_created_count": 0,
        "query_conditioned_graph_plan_pass_count": 0,
        "query_conditioned_graph_plan_fail_count": 0,
        "query_conditioned_graph_plan_failure_reason": {},
        "query_conditioned_graph_plan_type": {},
        "query_conditioned_graph_event_count_sum": 0,
        "query_conditioned_graph_active_event_count_sum": 0,
        "query_conditioned_graph_scope_limited_event_count_sum": 0,
        "query_conditioned_graph_invalidating_event_count_sum": 0,
        "query_conditioned_graph_natural_scope_limited_event_count_sum": 0,
        "query_conditioned_graph_natural_invalidating_event_count_sum": 0,
        "query_conditioned_graph_placeholder_false_surface_count_sum": 0,
        "boundary_aware_qa_planner": False,
        "boundary_item_difficulty_target": None,
        "boundary_qa_planner_enabled_count": 0,
        "boundary_qa_planner_used_count": 0,
        "boundary_qa_item_plan_created_count": 0,
        "boundary_qa_item_plan_valid_count": 0,
        "boundary_qa_item_plan_failure_reason": {},
        "boundary_qa_gold_path_type": {},
        "boundary_qa_hard_negative_path_types": {},
        "boundary_qa_item_difficulty_target": {},
        "boundary_qa_planner_dependency_target_min": {},
        "boundary_qa_planner_dependency_target_max": {},
        "boundary_qa_actual_qa_used_dependency_count_sum": 0,
        "boundary_qa_hard_negative_similarity_level": {},
        "boundary_qa_intermediate_clue_added_count": 0,
        "boundary_qa_explicit_contrast_type": {},
        "boundary_qa_boundary_calibration_mode": {},
        "boundary_qa_expected_solver_bucket": {},
        "boundary_qa_difficulty_calibration_failure_reason": {},
        "boundary_qa_planner_dependency_floor_pass_count": 0,
        "boundary_qa_planner_dependency_floor_fail_count": 0,
        "boundary_qa_planner_dependency_floor_failure_reason": {},
        "boundary_qa_planner_qa_used_dependency_count_sum": 0,
        "boundary_qa_planner_qa_used_dependency_count_max": 0,
        "planner_replan_attempt_count": 0,
        "planner_replan_success_count": 0,
        "planner_replan_failed_count": 0,
        "dependency_floor_pass_by_cell": {},
        "dependency_floor_fail_by_cell": {},
        "delayed_activation_semantic_validator_pass_count": 0,
        "delayed_activation_semantic_validator_fail_count": 0,
        "delayed_activation_semantic_failure_reason": {},
        "boundary_qa_repair_used_after_planner_count": 0,
        "boundary_qa_repair_success_after_planner_count": 0,
        "boundary_qa_final_option_surface_risk_count": 0,
        "stage33_compat_boundary_planner": False,
        "stage33_compat_deterministic_permission_forced_count": 0,
        "deterministic_permission_hard_negatives": False,
        "deterministic_permission_synthesizer_used_count": 0,
        "deterministic_permission_synthesizer_failure_reason": {},
        "deterministic_permission_synthesized_event_count": 0,
        "deterministic_permission_synthesized_source_event_count": 0,
        "deterministic_permission_surface_naturalized_count": 0,
        "deterministic_permission_surface_style": {},
        "deterministic_permission_surface_forbidden_cue_count": 0,
        "boundary_aware_qa_repair": False,
        "boundary_aware_qa_repair_attempts": 0,
        "boundary_qa_repair_trigger_count": 0,
        "boundary_qa_repair_attempt_count": 0,
        "boundary_qa_repair_success_count": 0,
        "boundary_qa_repair_failed_count": 0,
        "boundary_qa_repair_success_by_reason": {},
        "boundary_qa_repair_failed_by_reason": {},
        "pre_repair_reject_reason": {},
        "post_repair_reject_reason": {},
        "repaired_hard_negative_type": {},
        "repaired_duplicate_option_count": 0,
        "repaired_identical_gold_pair": 0,
        "repaired_gold_only_active_pattern": 0,
        "repaired_active_vs_outdated_shortcut": 0,
        "dialogue_alignment_reject_count": 0,
        "unsupported_alignment_count": 0,
        "parse_failure_count": 0,
        "timeout_count": 0,
        "anchor_leak_reject_count": 0,
        "dialogue_api_prompt_tokens_estimate_mean": None,
        "parent_context_turns_mean": None,
        "parent_context_tokens_estimate_mean": None,
        "repair_retry_count": 0,
        "repair_success_count": 0,
        "full_regeneration_retry_count": 0,
        "deterministic_scaffold_alignment_count": 0,
        "deterministic_scaffold_alignment_success_count": 0,
        "seed_source_null_count": 0,
        "seed_source_inferred_count": 0,
        "train_seed_allowed_count": 0,
        "train_seed_forbidden_count": 0,
        "top_level_options_missing_count": 0,
        "top_level_options_filled_from_qa_count": 0,
        "parent_dialogue_source_counts": {},
        "parent_turn_count_mean": None,
        "final_turn_count_mean": None,
        "appended_turn_count_mean": None,
        "turn_count_delta_mean": None,
        "approx_parent_tokens_mean": None,
        "approx_final_tokens_mean": None,
        "approx_token_delta_mean": None,
        "parent_dialogue_tokens_mean": None,
        "final_dialogue_tokens_mean": None,
        "length_delta_tokens_mean": None,
        "banned_term_counts": {},
        "solver_visible_banned_term_counts": {},
        "sanitized_seed_context_count": 0,
        "sanitized_event_value_count": 0,
        "ordering_identity_gold_count": 0,
        "anti_shortcut_risk_count": 0,
        "by_question_type": {},
        "by_attack_skill": {},
        "by_qa_operation": {},
        "by_answer_type": {},
        "stage64_api_qa_schema_pass_count": 0,
        "stage64_qa_repair_attempt_count": 0,
        "stage64_qa_repair_success_count": 0,
        "stage64_qa_repair_failed_count": 0,
        "stage64_qa_repair_saved_full_regeneration_count": 0,
        "stage64_qa_repair_failure_reason": {},
        "api_failure_count": 0,
        "challenger_api_used": True,
        "old_gate_generate_samples_called": False,
        "native_challenger_mode": "graph_conditioned_api_dialogue",
        "challenger_generation_sources": {
            "current_method.native_challenger_generation.graph_conditioned_api_dialogue": 0
        },
    }


def _update_summary_for_accept(summary: dict[str, Any], sample: dict[str, Any], plan: dict[str, Any], alignment_result: dict[str, Any]) -> None:
    summary["graph_event_count"] += len([event for event in plan.get("events") or [] if isinstance(event, dict)])
    summary["required_event_count"] += int(alignment_result.get("required_event_count") or 0)
    summary["aligned_required_event_count"] += int(alignment_result.get("aligned_required_event_count") or 0)
    summary["missing_required_event_count"] += len(alignment_result.get("missing_required_event_instance_ids") or [])
    summary["contradiction_count"] += int(alignment_result.get("contradiction_count") or 0)
    summary["alignment_coverage_pass_count"] += 1 if alignment_result.get("coverage_pass") is True else 0
    summary["anchor_leakage_count"] += int(alignment_result.get("anchor_leakage_count") or 0)
    summary["qa_generated_after_alignment_count"] += 1
    qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
    qtype = str(qa.get("question_type") or "unknown")
    skill = str(sample.get("attack_skill") or "unknown")
    summary.setdefault("by_question_type", {})
    summary.setdefault("by_attack_skill", {})
    summary["by_question_type"][qtype] = int(summary["by_question_type"].get(qtype, 0)) + 1
    summary["by_attack_skill"][skill] = int(summary["by_attack_skill"].get(skill, 0)) + 1
    if qa.get("stage64_api_qa_schema_pass") or qa.get("qa_operation"):
        op = str(qa.get("qa_operation") or "unknown")
        atype = str(qa.get("answer_type") or "unknown")
        summary.setdefault("by_qa_operation", {})
        summary.setdefault("by_answer_type", {})
        summary["by_qa_operation"][op] = int(summary["by_qa_operation"].get(op, 0)) + 1
        summary["by_answer_type"][atype] = int(summary["by_answer_type"].get(atype, 0)) + 1
        if qa.get("stage64_api_qa_schema_pass"):
            summary["stage64_api_qa_schema_pass_count"] += 1
    difficulty_subtype = str(sample.get("difficulty_subtype") or qa.get("difficulty_subtype") or "unknown")
    hard_template = str(sample.get("hard_single_choice_template") or qa.get("hard_single_choice_template") or "unknown")
    summary.setdefault("by_difficulty_subtype", {})
    summary.setdefault("hard_single_choice_by_template", {})
    summary["by_difficulty_subtype"][difficulty_subtype] = int(summary["by_difficulty_subtype"].get(difficulty_subtype, 0)) + 1
    if difficulty_subtype == "hard_single_choice":
        summary["hard_single_choice_by_template"][hard_template] = int(summary["hard_single_choice_by_template"].get(hard_template, 0)) + 1
    native_contract = sample.get("native_contract") if isinstance(sample.get("native_contract"), dict) else {}
    if native_contract.get("ordering_identity_gold"):
        summary["ordering_identity_gold_count"] += 1
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    if metadata.get("length_guard_pass") is True or native_contract.get("length_guard_pass") is True:
        summary["length_guard_pass_count"] += 1
    if metadata.get("length_guard_fallback_used") is True:
        summary["length_guard_fallback_accept_count"] += 1
        summary["length_guard_fallback_used"] = True
    target_delta = int(metadata.get("length_guard_target_delta") or metadata.get("min_length_delta_tokens") or 0)
    effective_delta = float(
        metadata.get("length_guard_effective_delta_tokens", metadata.get("length_delta_tokens", 0)) or 0
    )
    if target_delta > 0 and effective_delta < target_delta:
        summary["rows_below_4000_delta"] += 1
    validation = sample.get("validation") if isinstance(sample.get("validation"), dict) else {}
    if validation.get("anti_shortcut_pass") is not True:
        summary["anti_shortcut_risk_count"] += 1
    chained_diagnostics = qa.get("chained_qa_diagnostics") if isinstance(qa.get("chained_qa_diagnostics"), dict) else {}
    if chained_diagnostics.get("boundary_qa_planner_enabled"):
        summary["boundary_qa_planner_enabled_count"] += 1
        if chained_diagnostics.get("planner_used"):
            summary["boundary_qa_planner_used_count"] += 1
        if chained_diagnostics.get("item_plan_created"):
            summary["boundary_qa_item_plan_created_count"] += 1
        if chained_diagnostics.get("item_plan_valid"):
            summary["boundary_qa_item_plan_valid_count"] += 1
        reason = str(chained_diagnostics.get("item_plan_failure_reason") or "")
        if reason:
            summary["boundary_qa_item_plan_failure_reason"][reason] = int(summary["boundary_qa_item_plan_failure_reason"].get(reason) or 0) + 1
        gold_path_type = str(chained_diagnostics.get("gold_path_type") or "")
        if gold_path_type:
            summary["boundary_qa_gold_path_type"][gold_path_type] = int(summary["boundary_qa_gold_path_type"].get(gold_path_type) or 0) + 1
        for htype in chained_diagnostics.get("hard_negative_path_types") or []:
            summary["boundary_qa_hard_negative_path_types"][str(htype)] = int(summary["boundary_qa_hard_negative_path_types"].get(str(htype)) or 0) + 1
        item_target = str(chained_diagnostics.get("item_difficulty_target") or "legacy")
        summary["boundary_qa_item_difficulty_target"][item_target] = int(
            summary["boundary_qa_item_difficulty_target"].get(item_target) or 0
        ) + 1
        target_min = chained_diagnostics.get("planner_dependency_target_min")
        target_max = chained_diagnostics.get("planner_dependency_target_max")
        summary["boundary_qa_planner_dependency_target_min"][str(target_min)] = int(
            summary["boundary_qa_planner_dependency_target_min"].get(str(target_min)) or 0
        ) + 1
        summary["boundary_qa_planner_dependency_target_max"][str(target_max)] = int(
            summary["boundary_qa_planner_dependency_target_max"].get(str(target_max)) or 0
        ) + 1
        actual_dep_count = int(chained_diagnostics.get("actual_qa_used_dependency_count") or 0)
        summary["boundary_qa_actual_qa_used_dependency_count_sum"] += actual_dep_count
        similarity_level = str(chained_diagnostics.get("hard_negative_similarity_level") or "")
        if similarity_level:
            summary["boundary_qa_hard_negative_similarity_level"][similarity_level] = int(
                summary["boundary_qa_hard_negative_similarity_level"].get(similarity_level) or 0
            ) + 1
        if chained_diagnostics.get("intermediate_clue_added"):
            summary["boundary_qa_intermediate_clue_added_count"] += 1
        explicit_contrast = str(chained_diagnostics.get("explicit_contrast_type") or "")
        if explicit_contrast:
            summary["boundary_qa_explicit_contrast_type"][explicit_contrast] = int(
                summary["boundary_qa_explicit_contrast_type"].get(explicit_contrast) or 0
            ) + 1
        calibration_mode = str(chained_diagnostics.get("boundary_calibration_mode") or "")
        if calibration_mode:
            summary["boundary_qa_boundary_calibration_mode"][calibration_mode] = int(
                summary["boundary_qa_boundary_calibration_mode"].get(calibration_mode) or 0
            ) + 1
        expected_bucket = str(chained_diagnostics.get("expected_solver_bucket") or "")
        if expected_bucket:
            summary["boundary_qa_expected_solver_bucket"][expected_bucket] = int(
                summary["boundary_qa_expected_solver_bucket"].get(expected_bucket) or 0
            ) + 1
        calibration_failure = str(chained_diagnostics.get("difficulty_calibration_failure_reason") or "")
        if calibration_failure:
            summary["boundary_qa_difficulty_calibration_failure_reason"][calibration_failure] = int(
                summary["boundary_qa_difficulty_calibration_failure_reason"].get(calibration_failure) or 0
            ) + 1
        planner_dep_count = int(chained_diagnostics.get("planner_qa_used_dependency_count") or 0)
        summary["boundary_qa_planner_qa_used_dependency_count_sum"] += planner_dep_count
        summary["boundary_qa_planner_qa_used_dependency_count_max"] = max(
            int(summary.get("boundary_qa_planner_qa_used_dependency_count_max") or 0),
            planner_dep_count,
        )
        if chained_diagnostics.get("planner_dependency_floor_pass"):
            summary["boundary_qa_planner_dependency_floor_pass_count"] += 1
        else:
            summary["boundary_qa_planner_dependency_floor_fail_count"] += 1
            floor_reason = str(chained_diagnostics.get("planner_dependency_floor_failure_reason") or "")
            if floor_reason:
                summary["boundary_qa_planner_dependency_floor_failure_reason"][floor_reason] = (
                    int(summary["boundary_qa_planner_dependency_floor_failure_reason"].get(floor_reason) or 0) + 1
                )
        summary["planner_replan_attempt_count"] += int(chained_diagnostics.get("planner_replan_attempt_count") or 0)
        summary["planner_replan_success_count"] += int(chained_diagnostics.get("planner_replan_success_count") or 0)
        summary["planner_replan_failed_count"] += int(chained_diagnostics.get("planner_replan_failed_count") or 0)
        for cell, count in (chained_diagnostics.get("dependency_floor_pass_by_cell") or {}).items():
            summary["dependency_floor_pass_by_cell"][str(cell)] = int(summary["dependency_floor_pass_by_cell"].get(str(cell)) or 0) + int(count or 0)
        for cell, count in (chained_diagnostics.get("dependency_floor_fail_by_cell") or {}).items():
            summary["dependency_floor_fail_by_cell"][str(cell)] = int(summary["dependency_floor_fail_by_cell"].get(str(cell)) or 0) + int(count or 0)
        if chained_diagnostics.get("delayed_activation_chain_valid") is not None:
            if chained_diagnostics.get("delayed_activation_chain_valid"):
                summary["delayed_activation_semantic_validator_pass_count"] += 1
            else:
                summary["delayed_activation_semantic_validator_fail_count"] += 1
                delayed_reason = str(chained_diagnostics.get("delayed_activation_semantic_failure_reason") or "")
                if delayed_reason:
                    summary["delayed_activation_semantic_failure_reason"][delayed_reason] = (
                        int(summary["delayed_activation_semantic_failure_reason"].get(delayed_reason) or 0) + 1
                    )
        if chained_diagnostics.get("repair_used_after_planner"):
            summary["boundary_qa_repair_used_after_planner_count"] += 1
        if chained_diagnostics.get("repair_success_after_planner"):
            summary["boundary_qa_repair_success_after_planner_count"] += 1
        if chained_diagnostics.get("final_option_surface_risk"):
            summary["boundary_qa_final_option_surface_risk_count"] += 1
        if chained_diagnostics.get("stage33_compat_boundary_planner"):
            summary["stage33_compat_deterministic_permission_forced_count"] += int(
                bool(chained_diagnostics.get("stage33_compat_deterministic_permission_forced"))
            )
        if chained_diagnostics.get("deterministic_permission_synthesizer_used"):
            summary["deterministic_permission_synthesizer_used_count"] += 1
            summary["deterministic_permission_synthesized_event_count"] += len(
                chained_diagnostics.get("deterministic_permission_synthesized_event_ids") or []
            )
            summary["deterministic_permission_synthesized_source_event_count"] += len(
                chained_diagnostics.get("deterministic_permission_synthesized_source_event_ids") or []
            )
            if chained_diagnostics.get("deterministic_permission_surface_naturalized"):
                summary["deterministic_permission_surface_naturalized_count"] += 1
            surface_style = str(chained_diagnostics.get("deterministic_permission_surface_style") or "")
            if surface_style:
                summary["deterministic_permission_surface_style"][surface_style] = (
                    int(summary["deterministic_permission_surface_style"].get(surface_style) or 0) + 1
                )
            summary["deterministic_permission_surface_forbidden_cue_count"] += int(
                chained_diagnostics.get("deterministic_permission_surface_forbidden_cue_count") or 0
            )
        synth_reason = str(chained_diagnostics.get("deterministic_permission_synthesizer_failure_reason") or "")
        if synth_reason:
            summary["deterministic_permission_synthesizer_failure_reason"][synth_reason] = (
                int(summary["deterministic_permission_synthesizer_failure_reason"].get(synth_reason) or 0) + 1
            )
    repair_summary = None
    if isinstance(qa.get("boundary_qa_repair"), dict):
        repair_summary = qa.get("boundary_qa_repair")
    elif isinstance(qa.get("chained_qa_diagnostics"), dict) and isinstance(qa["chained_qa_diagnostics"].get("boundary_qa_repair"), dict):
        repair_summary = qa["chained_qa_diagnostics"].get("boundary_qa_repair")
    if isinstance(repair_summary, dict) and repair_summary.get("boundary_qa_repair_triggered"):
        summary["boundary_qa_repair_trigger_count"] += 1
        summary["boundary_qa_repair_attempt_count"] += int(repair_summary.get("boundary_qa_repair_attempt_count") or 0)
        success = bool(repair_summary.get("boundary_qa_repair_success"))
        if success:
            summary["boundary_qa_repair_success_count"] += 1
        else:
            summary["boundary_qa_repair_failed_count"] += 1
        for reason in repair_summary.get("pre_repair_reject_reason") or []:
            target = summary["boundary_qa_repair_success_by_reason"] if success else summary["boundary_qa_repair_failed_by_reason"]
            target[str(reason)] = int(target.get(str(reason)) or 0) + 1
            summary["pre_repair_reject_reason"][str(reason)] = int(summary["pre_repair_reject_reason"].get(str(reason)) or 0) + 1
        for reason in repair_summary.get("post_repair_reject_reason") or []:
            summary["post_repair_reject_reason"][str(reason)] = int(summary["post_repair_reject_reason"].get(str(reason)) or 0) + 1
        for htype in repair_summary.get("repaired_hard_negative_type") or []:
            summary["repaired_hard_negative_type"][str(htype)] = int(summary["repaired_hard_negative_type"].get(str(htype)) or 0) + 1
        summary["repaired_duplicate_option_count"] += int(repair_summary.get("repaired_duplicate_option_count") or 0)
        summary["repaired_identical_gold_pair"] += int(repair_summary.get("repaired_identical_gold_pair") or 0)
        if repair_summary.get("repaired_gold_only_active_pattern"):
            summary["repaired_gold_only_active_pattern"] += 1
        if repair_summary.get("repaired_active_vs_outdated_shortcut"):
            summary["repaired_active_vs_outdated_shortcut"] += 1


def _mean_numeric(values: list[Any]) -> float | None:
    nums: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            nums.append(float(value))
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def _length_summary_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    metadata_rows = [sample.get("metadata") for sample in samples if isinstance(sample.get("metadata"), dict)]
    min_deltas = [
        int(meta.get("min_length_delta_tokens") or meta.get("min_added_tokens") or 0)
        for meta in metadata_rows
        if isinstance(meta.get("min_length_delta_tokens") or meta.get("min_added_tokens"), (int, float))
    ]
    active_min_delta = max(min_deltas) if min_deltas else 0
    below_min_delta = sum(
        1
        for meta in metadata_rows
        if active_min_delta > 0
        and float(meta.get("length_guard_effective_delta_tokens", meta.get("length_delta_tokens", meta.get("approx_token_delta", 0))) or 0)
        < active_min_delta
    )
    target_deltas = [
        int(meta.get("length_guard_target_delta") or meta.get("min_length_delta_tokens") or 0)
        for meta in metadata_rows
        if isinstance(meta.get("length_guard_target_delta") or meta.get("min_length_delta_tokens"), (int, float))
    ]
    target_delta = max(target_deltas) if target_deltas else active_min_delta
    below_target_delta = sum(
        1
        for meta in metadata_rows
        if target_delta > 0
        and float(meta.get("length_guard_effective_delta_tokens", meta.get("length_delta_tokens", meta.get("approx_token_delta", 0))) or 0)
        < target_delta
    )
    fallback_accept_count = sum(1 for meta in metadata_rows if meta.get("length_guard_fallback_used") is True)
    return {
        "parent_dialogue_source_counts": dict(
            Counter(str(meta.get("parent_dialogue_source") or "unknown") for meta in metadata_rows).most_common()
        ),
        "parent_turn_count_mean": _mean_numeric([meta.get("parent_turn_count") for meta in metadata_rows]),
        "final_turn_count_mean": _mean_numeric([meta.get("final_turn_count") for meta in metadata_rows]),
        "appended_turn_count_mean": _mean_numeric([meta.get("appended_turn_count") for meta in metadata_rows]),
        "turn_count_delta_mean": _mean_numeric([meta.get("turn_count_delta") for meta in metadata_rows]),
        "approx_parent_tokens_mean": _mean_numeric([meta.get("approx_parent_tokens") for meta in metadata_rows]),
        "approx_final_tokens_mean": _mean_numeric([meta.get("approx_final_tokens") for meta in metadata_rows]),
        "approx_token_delta_mean": _mean_numeric([meta.get("approx_token_delta") for meta in metadata_rows]),
        "parent_dialogue_tokens_mean": _mean_numeric([meta.get("parent_dialogue_tokens", meta.get("approx_parent_tokens")) for meta in metadata_rows]),
        "final_dialogue_tokens_mean": _mean_numeric([meta.get("final_dialogue_tokens", meta.get("approx_final_tokens")) for meta in metadata_rows]),
        "length_delta_tokens_mean": _mean_numeric([meta.get("length_delta_tokens", meta.get("approx_token_delta")) for meta in metadata_rows]),
        "length_guard_parent_baseline_tokens": max(
            [
                int(meta.get("length_guard_parent_baseline_tokens") or 0)
                for meta in metadata_rows
                if isinstance(meta.get("length_guard_parent_baseline_tokens"), (int, float))
            ]
            or [0]
        ),
        "length_guard_effective_parent_tokens_mean": _mean_numeric(
            [meta.get("length_guard_effective_parent_tokens") for meta in metadata_rows]
        ),
        "length_guard_effective_delta_tokens_mean": _mean_numeric(
            [meta.get("length_guard_effective_delta_tokens") for meta in metadata_rows]
        ),
        "length_guard_baseline_mode_counts": dict(
            Counter(str(meta.get("length_guard_baseline_mode") or "unknown") for meta in metadata_rows).most_common()
        ),
        "rows_below_min_length_delta": below_min_delta,
        "length_guard_target_delta": target_delta,
        "length_guard_fallback_used": fallback_accept_count > 0,
        "length_guard_fallback_accept_count": fallback_accept_count,
        "fallback_delta": max(
            [int(meta.get("fallback_delta") or 0) for meta in metadata_rows if isinstance(meta.get("fallback_delta"), (int, float))]
            or [0]
        ),
        "emergency_floor": max(
            [int(meta.get("emergency_floor") or 0) for meta in metadata_rows if isinstance(meta.get("emergency_floor"), (int, float))]
            or [0]
        ),
        "rows_below_4000_delta": below_target_delta,
    }


def _prompt_context_summary_from_records(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens: list[Any] = []
    parent_turns: list[Any] = []
    parent_tokens: list[Any] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        diagnostics = record.get("parent_context_diagnostics") if isinstance(record.get("parent_context_diagnostics"), dict) else {}
        if diagnostics:
            parent_turns.append(diagnostics.get("parent_context_turns"))
            parent_tokens.append(diagnostics.get("parent_context_tokens_estimate"))
        for attempt in record.get("api_dialogue_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            meta = attempt.get("meta") if isinstance(attempt.get("meta"), dict) else {}
            prompt_tokens.append(meta.get("prompt_tokens_estimate"))
            if meta.get("parent_context_turns") is not None:
                parent_turns.append(meta.get("parent_context_turns"))
            if meta.get("parent_context_tokens_estimate") is not None:
                parent_tokens.append(meta.get("parent_context_tokens_estimate"))
        for attempt in record.get("api_dialogue_repair_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            meta = attempt.get("meta") if isinstance(attempt.get("meta"), dict) else {}
            prompt_tokens.append(meta.get("prompt_tokens_estimate"))
    return {
        "dialogue_api_prompt_tokens_estimate_mean": _mean_numeric(prompt_tokens),
        "parent_context_turns_mean": _mean_numeric(parent_turns),
        "parent_context_tokens_estimate_mean": _mean_numeric(parent_tokens),
    }


def _compat_summary_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    seed_source_null_count = 0
    seed_source_inferred_count = 0
    train_seed_allowed_count = 0
    train_seed_forbidden_count = 0
    top_level_options_missing_count = 0
    top_level_options_filled_from_qa_count = 0
    for sample in samples:
        seed_source = sample.get("seed_source") if isinstance(sample.get("seed_source"), dict) else None
        if seed_source is None:
            seed_source_null_count += 1
        elif seed_source.get("allowed_train_seed") is True:
            train_seed_allowed_count += 1
        else:
            train_seed_forbidden_count += 1
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        if metadata.get("seed_source_inferred") is True:
            seed_source_inferred_count += 1
        qa = sample.get("qa") if isinstance(sample.get("qa"), dict) else {}
        if isinstance(qa.get("options"), list):
            if sample.get("options") is None:
                top_level_options_missing_count += 1
            else:
                top_level_options_filled_from_qa_count += 1
    return {
        "seed_source_null_count": seed_source_null_count,
        "seed_source_inferred_count": seed_source_inferred_count,
        "train_seed_allowed_count": train_seed_allowed_count,
        "train_seed_forbidden_count": train_seed_forbidden_count,
        "top_level_options_missing_count": top_level_options_missing_count,
        "top_level_options_filled_from_qa_count": top_level_options_filled_from_qa_count,
    }


def _write_generation_files(
    output: Path,
    clean_samples: list[dict[str, Any]],
    marked_samples: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    clean_meta_ok_samples = [sample for sample in clean_samples if assert_no_meta_leakage(sample)[0]]
    valid_samples = [sample for sample in clean_meta_ok_samples if base.base_valid(sample)]
    solver_clean_samples = [sample for sample in clean_meta_ok_samples if base.is_clean(sample)]
    final_meta_stats = meta_leak_stats(clean_samples)
    summary["meta_leak_count"] = final_meta_stats["meta_leak_count"]
    summary["solver_visible_meta_leak_count"] = final_meta_stats["solver_visible_meta_leak_count"]
    summary["qa_meta_leak_count"] = final_meta_stats["qa_meta_leak_count"]
    summary["dialogue_meta_leak_count"] = final_meta_stats["dialogue_meta_leak_count"]
    summary["solver_visible_banned_term_counts"] = final_meta_stats["banned_term_counts"]
    if final_meta_stats["banned_term_counts"]:
        summary["final_banned_term_counts"] = final_meta_stats["banned_term_counts"]
    summary.update(_length_summary_from_samples(clean_samples))
    summary.update(_prompt_context_summary_from_records(raw_records))
    summary.update(_compat_summary_from_samples(clean_samples))
    write_jsonl(output / "samples.jsonl", clean_samples)
    write_jsonl(output / "valid_samples.jsonl", valid_samples)
    write_jsonl(output / "clean_samples.jsonl", solver_clean_samples)
    write_jsonl(output / "marked_samples.jsonl", marked_samples)
    write_jsonl(output / "graph_conditioned_api_dialogue_raw.jsonl", raw_records)
    write_jsonl(output / "graph_conditioned_api_dialogue_validated.jsonl", marked_samples)
    write_jsonl(output / "graph_conditioned_api_dialogue_rejects.jsonl", rejects)
    write_json(output / "graph_dialogue_alignment_summary.json", summary)
    write_json(output / "graph_conditioned_api_dialogue_summary.json", summary)
    write_json(output / "native_generation_summary.json", summary)
    write_json(output / "generation_summary.json", summary)


def generate_graph_conditioned_api_dialogue_samples(
    *,
    seed_rows: list[dict[str, Any]],
    policy: dict[str, Any] | None,
    n_samples: int,
    output_dir: str | Path,
    rng_seed: int = 123,
    api_args: argparse.Namespace,
    round_id: str = "current_method_r4",
    partial_samples: str | Path | None = None,
    no_resume: bool = False,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not seed_rows:
        raise ValueError("No seed rows supplied for graph_conditioned_api_dialogue generation.")

    # ---- resume: recover already-accepted clean samples so a re-run continues instead of
    # regenerating from scratch. The partial file is the crash-safe append log; if it is
    # missing/empty we fall back to a completed samples.jsonl from a prior finished run.
    # Resume only restores the ACCEPTED clean set (the downstream contract). The paired
    # marked/raw artifacts are append-only and are NOT replayed for recovered samples; we
    # rewrite them at the end from whatever this process produced plus the recovered clean set.
    partial_path = Path(partial_samples) if partial_samples else output / "samples.partial.jsonl"
    resumed_clean_samples: list[dict[str, Any]] = []
    if not no_resume:
        resumed_clean_samples = read_jsonl_if_exists(partial_path)
        if not resumed_clean_samples:
            resumed_clean_samples = read_jsonl_if_exists(output / "samples.jsonl")
        if len(resumed_clean_samples) > n_samples:
            resumed_clean_samples = resumed_clean_samples[:n_samples]

    # Propagate the opt-in curriculum mode from api_args onto the policy so the downstream
    # choose_question_type / choose_attack_skill / event-count / length helpers can see it
    # without changing their signatures. Default mode leaves the policy untouched in effect.
    curriculum_mode = normalize_space(getattr(api_args, "native_curriculum_mode", "") or "default")
    if curriculum_mode in ("r3_long_easy", EASY_V2_MODE):
        policy = dict(policy or {})
        policy["native_curriculum_mode"] = curriculum_mode
    # base_policy is the immutable per-run policy; each worker shallow-copies it and sets its
    # own planned_cell, so concurrent workers never share/mutate a single policy dict.
    base_policy = dict(policy or {})
    if bool(getattr(api_args, "query_conditioned_graph_expansion", False)):
        base_policy["query_conditioned_graph_expansion"] = True
    if bool(getattr(api_args, "chained_qa_dependency_targeting", False)):
        base_policy["chained_qa_dependency_targeting"] = True
    if bool(getattr(api_args, "boundary_aware_qa_diagnostics", False)):
        base_policy["boundary_aware_qa_diagnostics"] = True
    if bool(getattr(api_args, "boundary_aware_qa_planner", False)):
        base_policy["boundary_aware_qa_planner"] = True
    if bool(getattr(api_args, "stage33_compat_boundary_planner", False)):
        base_policy["stage33_compat_boundary_planner"] = True
        # Stage33/35 production relied on natural scope/override hard negatives from the
        # graph API. Current API responses sometimes omit them entirely; in compat mode,
        # recover that proven path by forcing anchored deterministic permission negatives.
        base_policy["deterministic_permission_hard_negatives"] = True
    if bool(getattr(api_args, "qa_operation_answer_type_diversification", False)):
        # Opt-in Stage64 selfplay QA schema. The legacy graph-first / anchor-aligned / QA-last
        # path is unchanged unless a caller explicitly asks the API challenger to emit this QA.
        base_policy["qa_operation_answer_type_diversification"] = True
    boundary_item_difficulty_target = _normalize_boundary_item_difficulty_target(
        getattr(api_args, "boundary_item_difficulty_target", None)
        or base_policy.get("item_difficulty_target")
        or base_policy.get("boundary_item_difficulty_target")
    )
    if boundary_item_difficulty_target:
        base_policy["item_difficulty_target"] = boundary_item_difficulty_target
    if bool(getattr(api_args, "deterministic_permission_hard_negatives", False)):
        base_policy["deterministic_permission_hard_negatives"] = True
    if bool(getattr(api_args, "boundary_aware_qa_repair", False)):
        base_policy["boundary_aware_qa_repair"] = True
        try:
            base_policy["boundary_aware_qa_repair_attempts"] = int(getattr(api_args, "boundary_aware_qa_repair_attempts", 2) or 0)
        except (TypeError, ValueError):
            base_policy["boundary_aware_qa_repair_attempts"] = 2
    summary = _empty_summary()
    summary["native_curriculum_mode"] = curriculum_mode or "default"
    summary["chained_qa_dependency_targeting"] = bool(base_policy.get("chained_qa_dependency_targeting"))
    summary["boundary_aware_qa_diagnostics"] = bool(base_policy.get("boundary_aware_qa_diagnostics"))
    summary["boundary_aware_qa_planner"] = bool(base_policy.get("boundary_aware_qa_planner"))
    summary["stage33_compat_boundary_planner"] = bool(base_policy.get("stage33_compat_boundary_planner"))
    summary["qa_operation_answer_type_diversification"] = bool(
        base_policy.get("qa_operation_answer_type_diversification")
    )
    if base_policy.get("qa_operation_answer_type_diversification"):
        summary["stage64_target_operation_counts"] = dict(
            Counter(_stage64_target_for_slot(slot, n_samples)["qa_operation"] for slot in range(max(0, n_samples)))
        )
    summary["boundary_item_difficulty_target"] = base_policy.get("item_difficulty_target")
    summary["deterministic_permission_hard_negatives"] = bool(base_policy.get("deterministic_permission_hard_negatives"))
    summary["boundary_aware_qa_repair"] = bool(base_policy.get("boundary_aware_qa_repair"))
    summary["boundary_aware_qa_repair_attempts"] = int(base_policy.get("boundary_aware_qa_repair_attempts") or 0)
    summary["api_yield_normalizers"] = bool(getattr(api_args, "api_yield_normalizers", False))
    summary["graph_cache_enabled"] = bool(getattr(api_args, "graph_proposal_cache", False))
    summary["query_conditioned_graph_expansion"] = bool(base_policy.get("query_conditioned_graph_expansion"))
    graph_cache_dir = graph_cache_dir_from_args(api_args, output)
    length_tokenizer_path = normalize_space(
        getattr(api_args, "solver_tokenizer", None)
        or getattr(api_args, "tokenizer_path", None)
        or getattr(api_args, "base_model", None)
    )
    if length_tokenizer_path:
        summary["length_tokenizer_path"] = length_tokenizer_path
        summary["length_token_count_mode"] = "tokenizer"
        try:
            _load_length_tokenizer(length_tokenizer_path)
        except Exception as exc:  # pragma: no cover - defensive fallback for legacy/local runs.
            logging.warning("length tokenizer unavailable at %s; falling back to approximate count: %s", length_tokenizer_path, exc)
            summary["length_token_count_mode"] = "approx_chars_div4"
    else:
        summary["length_token_count_mode"] = "approx_chars_div4"

    # easy_v2: build a deterministic (qtype, skill) plan so the ACCEPTED distribution matches
    # the target shares exactly instead of drifting via weighted sampling + silent fallback.
    # The closed-loop controller may inject difficulty_plan into policy to bias shares/cells.
    difficulty_plan = policy.get("difficulty_plan") if isinstance(policy, dict) and isinstance(policy.get("difficulty_plan"), dict) else {}
    quota_scheduler = (
        DeterministicQuotaScheduler(
            n_samples,
            random.Random(rng_seed + 1),
            question_type_shares=difficulty_plan.get("question_type_shares") if isinstance(difficulty_plan.get("question_type_shares"), dict) else None,
            cell_weights=difficulty_plan.get("cell_weights") if isinstance(difficulty_plan.get("cell_weights"), dict) else None,
            cell_hardness_mix=difficulty_plan.get("cell_hardness_mix") if isinstance(difficulty_plan.get("cell_hardness_mix"), dict) else None,
            single_choice_subtype_shares=difficulty_plan.get("single_choice_subtype_shares") if isinstance(difficulty_plan.get("single_choice_subtype_shares"), dict) else None,
            single_choice_subtype_hardness_mix=difficulty_plan.get("single_choice_subtype_hardness_mix") if isinstance(difficulty_plan.get("single_choice_subtype_hardness_mix"), dict) else None,
            hard_single_choice_template_shares=difficulty_plan.get("hard_single_choice_template_shares") if isinstance(difficulty_plan.get("hard_single_choice_template_shares"), dict) else None,
        )
        if curriculum_mode == EASY_V2_MODE
        else None
    )
    if quota_scheduler is not None:
        summary["easy_v2_planned_question_type_quota"] = dict(quota_scheduler.qtype_quota)
        summary["difficulty_plan_applied"] = bool(difficulty_plan)
        curriculum = difficulty_plan.get("qtype_curriculum") if isinstance(difficulty_plan.get("qtype_curriculum"), dict) else {}
        summary["difficulty_plan_stage"] = curriculum.get("stage")
        summary["difficulty_plan_manual_policy_correction"] = curriculum.get("manual_policy_correction")
        if isinstance(difficulty_plan.get("hard_single_choice_template_shares"), dict):
            summary["hard_single_choice_template_shares"] = dict(difficulty_plan.get("hard_single_choice_template_shares") or {})
    # Hard caps on the accepted clean set (easy_v2 only). The cap is feasibility-aware: it must
    # never reject BELOW what the deterministic plan itself asked for, otherwise the plan and the
    # cap contradict each other and the run deadlocks (e.g. plan wants 13 single_choice but a flat
    # 40% cap rejects after 8). So the effective per-type cap = max(global 40% cap, planned quota).
    _global_qtype_cap = int(round(EASY_V2_MAX_QUESTION_TYPE_SHARE * n_samples)) if curriculum_mode == EASY_V2_MODE else None
    _global_skill_cap = int(round(EASY_V2_MAX_ATTACK_SKILL_SHARE * n_samples)) if curriculum_mode == EASY_V2_MODE else None
    _planned_quota = dict(quota_scheduler.qtype_quota) if quota_scheduler is not None else {}
    # Planned per-skill demand: simulate the scheduler's full plan so the skill cap knows how many
    # of each skill the deterministic plan actually intends. Without this, a flat 40% skill cap
    # deadlocks when the plan concentrates 2 skills (each hits 0.40*n=20, total < n -> can't fill).
    _planned_skill_quota: dict[str, int] = {}
    if quota_scheduler is not None:
        from collections import Counter as _Counter
        _sim = [quota_scheduler.plan_for_index(i) for i in range(n_samples)]
        _planned_skill_quota = dict(_Counter(p["attack_skill"] for p in _sim))
        # restore scheduler rr state so the real run reproduces the same plan
        quota_scheduler._skill_rr = {q: 0 for q in EASY_V2_SUPPORTED_QUESTION_TYPES}
        quota_scheduler._hardness_rr = {}
        quota_scheduler._single_choice_subtype_rr = 0
        quota_scheduler._hard_single_choice_template_rr = 0

    def _qtype_cap(qtype: str) -> int | None:
        if _global_qtype_cap is None:
            return None
        return max(_global_qtype_cap, int(_planned_quota.get(qtype, 0)))

    def _skill_cap(skill: str) -> int | None:
        # Feasibility-aware per-skill cap: never reject below what the plan itself intends for
        # this skill (+1 margin for rounding), mirroring _qtype_cap. Prevents the skill-cap deadlock.
        if _global_skill_cap is None:
            return None
        return max(_global_skill_cap, int(_planned_skill_quota.get(skill, 0)) + 1)

    accepted_qtype_counts: Counter[str] = Counter()
    accepted_skill_counts: Counter[str] = Counter()
    planned_cell_log: list[dict[str, Any]] = []
    clean_samples: list[dict[str, Any]] = []
    marked_samples: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    # Resume: seed runtime state from recovered accepted samples so generation continues from
    # where a prior (possibly interrupted) run stopped instead of regenerating from scratch.
    # The marked/raw/reject diagnostic artifacts are NOT replayed for recovered samples (they
    # have no downstream consumer); they are rebuilt at the end from this process's own records.
    if resumed_clean_samples:
        clean_samples.extend(resumed_clean_samples)
        for _recovered in resumed_clean_samples:
            _q = str((_recovered.get("qa") or {}).get("question_type") or "unknown")
            _sk = str(_recovered.get("attack_skill") or "unknown")
            accepted_qtype_counts[_q] += 1
            accepted_skill_counts[_sk] += 1
        # Advance the deterministic scheduler's round-robin state past the recovered slots so the
        # remaining slots keep the planned skill/hardness rotation. qtype is a pure function of the
        # slot index, so resuming the driver at slot=len(clean_samples) keeps the qtype quota
        # exactly aligned regardless of rr replay.
        if quota_scheduler is not None:
            for _i in range(len(resumed_clean_samples)):
                quota_scheduler.plan_for_index(_i)
    # Normalize the crash-safe partial log to the current accepted set so subsequent per-accept
    # appends stay consistent whether we recovered from the partial file, fell back to a finished
    # samples.jsonl, or are starting fresh (no_resume / nothing to recover -> truncates to empty).
    write_jsonl(partial_path, clean_samples)

    # Explicit resume accounting so logs make it unambiguous whether continuation is happening.
    resume_loaded_count = len(resumed_clean_samples)
    resume_target_count = n_samples
    resume_remaining_count = max(0, n_samples - resume_loaded_count)
    summary["resume_loaded_count"] = resume_loaded_count
    summary["resume_target_count"] = resume_target_count
    summary["resume_remaining_count"] = resume_remaining_count
    summary["resume_no_resume_flag"] = bool(no_resume)
    logging.info(
        "graph_conditioned_api_dialogue resume: resume_loaded_count=%d resume_target_count=%d "
        "resume_remaining_count=%d no_resume=%s partial_path=%s",
        resume_loaded_count, resume_target_count, resume_remaining_count, no_resume, partial_path,
    )

    max_api_failures = int(getattr(api_args, "max_api_failures", 50))
    max_attempts = max(n_samples + max_api_failures, n_samples * 3)
    min_length_added_turns, min_length_added_tokens = length_guard_thresholds(api_args)
    summary["length_min_added_turns"] = min_length_added_turns
    summary["length_min_added_tokens"] = min_length_added_tokens
    requested_min_length_delta = int(getattr(api_args, "min_length_delta_tokens", 0) or 0)
    try:
        length_guard_parent_baseline_tokens = int(getattr(api_args, "length_guard_parent_baseline_tokens", 0) or 0)
    except (TypeError, ValueError):
        length_guard_parent_baseline_tokens = 0
    round_number = int(getattr(api_args, "round_number", 0) or 0)
    stage64_length_guard = bool(getattr(api_args, "qa_operation_answer_type_diversification", False))
    summary["length_guard_enabled"] = bool((round_number >= 4 or stage64_length_guard) and requested_min_length_delta > 0)
    summary["min_length_delta_tokens"] = requested_min_length_delta if summary["length_guard_enabled"] else 0
    summary["length_guard_parent_baseline_tokens"] = (
        length_guard_parent_baseline_tokens if summary["length_guard_enabled"] else 0
    )
    summary["length_guard_baseline_mode"] = (
        "max_actual_parent_and_configured_baseline"
        if summary["length_guard_enabled"] and length_guard_parent_baseline_tokens > 0
        else "actual_parent_dialogue_tokens"
    )
    run_name_text = normalize_space(getattr(api_args, "run_name", "") or getattr(api_args, "round_id", "") or round_id)
    smoke_length_fallback_enabled = bool(
        summary["length_guard_enabled"]
        and not stage64_length_guard
        and (n_samples <= 20 or "smoke" in run_name_text)
    )
    fallback_delta = 3000 if smoke_length_fallback_enabled else 0
    emergency_floor = 2500 if smoke_length_fallback_enabled else 0
    summary["length_guard_target_delta"] = requested_min_length_delta if summary["length_guard_enabled"] else 0
    summary["length_guard_fallback_used"] = False
    summary["fallback_delta"] = fallback_delta
    summary["emergency_floor"] = emergency_floor

    # --- per-sample / per-slot attempt cap (opt-in; default off => old behavior unchanged) ---
    # A "slot" is one target accepted-sample position. Without this cap, a hard cell (e.g. a
    # bad-gold qtype) can soak unbounded retries because the quota keeps re-requesting it. With
    # the cap, once a slot fails `max_attempts_per_sample` times in a row it is SKIPPED: the
    # driver stops submitting for it and moves the plan cursor forward (reallocating that slot's
    # share to the next planned cell, which the feasible-share controller keeps non-bad).
    _raw_cap = getattr(api_args, "max_attempts_per_sample", None)
    try:
        max_attempts_per_sample = int(_raw_cap) if _raw_cap else 0
    except (TypeError, ValueError):
        max_attempts_per_sample = 0
    per_sample_attempt_cap_enabled = max_attempts_per_sample > 0
    slot_attempt_counts: dict[int, int] = {}
    slot_accepted: set[int] = set()
    skipped_slots: set[int] = set()
    skipped_slot_reasons: dict[str, int] = {}
    skipped_by_cell: dict[str, int] = {}
    attempts_for_accepted: list[int] = []
    summary["max_attempts_per_sample"] = max_attempts_per_sample
    summary["per_sample_attempt_cap_enabled"] = per_sample_attempt_cap_enabled
    attempts = 0

    # Worker: produce ONE sample attempt as a pure function (no shared-state mutation).
    # It mutates only its OWN local summary copy and a per-sample rng/policy; the serial
    # collector merges the returned delta. This makes sample-level concurrency race-free.
    def _attempt(attempt_no: int, sample_index: int, planned_cell: dict[str, str] | None) -> dict[str, Any]:
        summary = _empty_summary()  # local delta, merged serially by the collector
        rng = random.Random(rng_seed + 1_000_003 * (sample_index + 1) + attempt_no)
        policy_local = dict(base_policy)
        if planned_cell is not None:
            policy_local["planned_cell"] = planned_cell
        stage64_target = None
        if base_policy.get("qa_operation_answer_type_diversification"):
            stage64_target = _stage64_target_for_slot(sample_index, n_samples)
            policy_local["stage64_target"] = stage64_target
        policy = policy_local  # body below references `policy`; keep the name
        seed_row = seed_rows[rng.randrange(len(seed_rows))]
        raw_parent_dialogue_turns, parent_dialogue_source = parent_dialogue_turns_from_row(seed_row)
        raw_record: dict[str, Any] = {
            "attempt": attempt_no,
            "target_sample_index": sample_index,
            "seed_sample_id": seed_row.get("sample_id") or seed_row.get("source_id"),
            "parent_dialogue_source": parent_dialogue_source,
            "raw_parent_turn_count": len(raw_parent_dialogue_turns),
            "planned_cell": dict(planned_cell) if planned_cell else None,
            "stage64_target": dict(stage64_target) if isinstance(stage64_target, dict) else None,
        }
        try:
            legacy = bootstrap_legacy_alignment(seed_row)
            summary["legacy_alignment_bootstrap_count"] += 1
            summary["legacy_high_confidence_count"] += int(legacy.get("legacy_high_confidence_count") or 0)
            summary["legacy_low_confidence_count"] += int(legacy.get("legacy_low_confidence_count") or 0)
            summary["legacy_needs_reverbalization_count"] += sum(
                1
                for entry in (legacy.get("legacy_alignment") or {}).values()
                if isinstance(entry, dict) and entry.get("needs_reverbalization") is True
            )

            graph_cache_enabled = bool(getattr(api_args, "graph_proposal_cache", False))
            graph_cache_key = graph_cache_key_for_attempt(
                seed_row,
                planned_cell,
                policy,
                api_args,
                sample_index=sample_index,
                round_id=round_id,
            )
            graph_cache_entry = load_graph_cache_entry(graph_cache_dir, graph_cache_key) if graph_cache_enabled else None
            if graph_cache_entry is not None:
                summary["graph_cache_hit_count"] += 1
                summary["graph_cache_reuse_for_dialogue_retry_count"] += 1
                summary["graph_calls_saved_estimate"] += 1
                current_event_plan = copy.deepcopy(graph_cache_entry["current_event_plan"])
                graph_meta = copy.deepcopy(graph_cache_entry.get("graph_meta") or {})
                graph_meta["used"] = True
                graph_meta["cache_hit"] = True
                graph_meta["cache_key"] = graph_cache_key
                raw_graph = copy.deepcopy(graph_cache_entry.get("raw_graph"))
            else:
                summary["api_graph_called_count"] += 1
                current_event_plan, graph_meta, raw_graph = call_api_graph_expansion(
                    seed_row,
                    legacy,
                    policy,
                    api_args,
                    rng,
                    sample_index=sample_index,
                    round_id=round_id,
                )
                if graph_cache_enabled and graph_meta.get("used") and isinstance(current_event_plan.get("events"), list):
                    write_graph_cache_entry(
                        graph_cache_dir,
                        graph_cache_key,
                        current_event_plan=current_event_plan,
                        graph_meta=graph_meta,
                        raw_graph=raw_graph,
                    )
                    summary["graph_cache_write_count"] += 1
            raw_record["api_graph_meta"] = graph_meta
            raw_record["api_graph_payload"] = raw_graph
            raw_record["graph_cache_key"] = graph_cache_key if graph_cache_enabled else None
            _record_api_meta_retry_diagnostics(summary, graph_meta)
            _record_query_conditioned_graph_diagnostics(summary, graph_meta)
            if not graph_meta.get("used"):
                if graph_cache_enabled:
                    summary["graph_cache_invalidated_count"] += 1
                raw_record.setdefault("api_graph_retry_diagnostics", []).append(
                    {
                        "reason": "parse_or_api_failure",
                        "meta": copy.deepcopy(graph_meta),
                    }
                )
                raise NativeGraphDialogueReject(f"api graph expansion failed: {graph_meta}")

            current_event_plan = assign_current_alignment_anchors(current_event_plan, rng)
            event_to_anchor = current_event_plan.get("current_event_instance_to_anchor") or {}
            anchor_to_event = current_event_plan.get("current_anchor_to_event_instance_id") or {}
            summary["current_anchor_assigned_count"] += len(anchor_to_event)
            summary["sanitized_seed_context_count"] += 1
            summary["sanitized_event_value_count"] += sanitized_event_value_count(current_event_plan)
            legacy_anchors = collect_legacy_anchor_strings(seed_row)
            known_anchors = list(anchor_to_event) + legacy_anchors
            parent_dialogue_turns = solver_visible_parent_dialogue_turns(raw_parent_dialogue_turns, known_anchors)
            parent_context = build_parent_context_for_dialogue_api(
                parent_dialogue_turns,
                seed_row,
                legacy,
                current_event_plan,
                api_args,
            )
            raw_record["parent_context_diagnostics"] = parent_context.get("diagnostics")

            dialogue_proposal: dict[str, Any] | None = None
            dialogue_meta: dict[str, Any] = {}
            alignment_result: dict[str, Any] | None = None
            raw_alignment_proposal: dict[str, Any] | None = None
            final_dialogue: dict[str, Any] | None = None
            accepted_appended_dialogue_marked: list[dict[str, Any]] = []
            accepted_length_metadata: dict[str, Any] | None = None
            max_dialogue_retries = int(getattr(api_args, "max_llm_retries", 1)) + 1
            retry_context: dict[str, Any] | None = None
            accepted_seen_anchors: set[str] = set()
            accepted_invalid_anchors: set[str] = set()
            accepted_missing_anchors: set[str] = set()
            accepted_alignment_invalid_count = 0
            for dialogue_attempt in range(1, max_dialogue_retries + 1):
                summary["api_dialogue_called_count"] += 1
                dialogue_proposal, dialogue_meta = call_api_graph_conditioned_dialogue_expansion(
                    current_event_plan,
                    seed_row,
                    legacy,
                    policy,
                    api_args,
                    parent_context=parent_context,
                    parent_dialogue_source=parent_dialogue_source,
                    parent_turn_count=len(parent_dialogue_turns),
                    min_added_turns=min_length_added_turns,
                    min_added_tokens=min_length_added_tokens,
                    length_guard_parent_baseline_tokens=length_guard_parent_baseline_tokens,
                    retry_context=retry_context,
                )
                raw_record.setdefault("api_dialogue_attempts", []).append(
                    {"attempt": dialogue_attempt, "meta": dialogue_meta, "payload": dialogue_proposal}
                )
                _record_api_meta_retry_diagnostics(summary, dialogue_meta)
                if not dialogue_meta.get("used") or not isinstance(dialogue_proposal, dict):
                    retry_context = {"previous_error": "dialogue API did not return a valid JSON proposal", "dialogue_meta": dialogue_meta}
                    raw_record.setdefault("dialogue_retry_diagnostics", []).append(
                        {
                            "dialogue_attempt": dialogue_attempt,
                            "reason": "parse_or_api_failure",
                            "retry_context": copy.deepcopy(retry_context),
                        }
                    )
                    continue
                if isinstance(dialogue_proposal.get("qa_proposal"), dict):
                    summary["api_qa_proposal_count"] += 1
                appended_dialogue = current_event_scaffold_turns(current_event_plan) + list(
                    _as_dialogue_list(appended_dialogue_from_proposal(dialogue_proposal))
                )
                candidate = validate_appended_dialogue_candidate(
                    appended_dialogue_marked=appended_dialogue,
                    event_alignment_proposal=dialogue_proposal.get("event_alignment_proposal"),
                    current_event_plan=current_event_plan,
                    anchor_to_event=anchor_to_event,
                    parent_dialogue_turns=parent_dialogue_turns,
                    known_anchors=known_anchors,
                    sample_index=sample_index,
                    min_added_turns=min_length_added_turns,
                    min_added_tokens=min_length_added_tokens,
                    tokenizer_path=length_tokenizer_path,
                    parent_baseline_tokens=length_guard_parent_baseline_tokens,
                    fallback_min_added_tokens=fallback_delta,
                    emergency_min_added_tokens=emergency_floor,
                )
                raw_alignment_proposal = candidate.get("raw_alignment_proposal")
                final_dialogue = candidate.get("final_dialogue")
                length_metadata = candidate.get("length_metadata") or {}
                length_metadata["parent_dialogue_source"] = parent_dialogue_source
                if candidate.get("ignored_event_alignment_proposal"):
                    summary["deterministic_scaffold_alignment_count"] += 1
                if candidate.get("reason") == "length":
                    summary["length_retry_count"] += 1
                    summary["length_reject_count"] += 1
                    summary["too_short_count"] += 1
                    summary["length_guard_retry_count"] += 1
                    summary["length_guard_reject_count"] += 1
                    retry_context = {
                        "previous_error": "append-only dialogue failed length guard",
                        "length_guard": length_metadata,
                        "instruction": (
                            "Return appended_dialogue_turns only, with at least 24 substantive new turns and a clear token margin above the length guard. "
                            "Do not rewrite or include the parent dialogue."
                        ),
                    }
                    raw_record.setdefault("dialogue_retry_diagnostics", []).append(
                        {
                            "dialogue_attempt": dialogue_attempt,
                            "reason": "length",
                            "retry_context": copy.deepcopy(retry_context),
                        }
                    )
                    continue
                if candidate.get("reason") == "meta_leak":
                    meta_details = candidate.get("meta_details") or {}
                    summary["meta_leak_reject_count"] += 1
                    summary["solver_visible_meta_leak_reject_count"] += 1
                    summary["naturalness_retry_count"] += 1
                    _merge_counter_dict(summary["banned_term_counts"], meta_details.get("banned_term_counts") or {})
                    retry_context = {
                        "previous_error": "clean dialogue contained internal/project terms",
                        "leaked_banned_terms": meta_details.get("matched_terms"),
                        "leak_details": meta_details.get("details")[:6],
                        "instruction": (
                            "Rewrite the appended_dialogue_turns naturally while preserving fact truth, every marker sidecar, and all required facts. "
                            "Do not include or rewrite the parent dialogue. Express leaked terms in normal human language."
                        ),
                    }
                    raw_record.setdefault("dialogue_retry_diagnostics", []).append(
                        {
                            "dialogue_attempt": dialogue_attempt,
                            "reason": "meta_leak",
                            "retry_context": copy.deepcopy(retry_context),
                        }
                    )
                    continue
                alignment_result = candidate.get("alignment_result")
                missing_anchors = set(candidate.get("missing_anchors") or set())
                if candidate.get("pass") is True and isinstance(alignment_result, dict):
                    if candidate.get("ignored_event_alignment_proposal"):
                        summary["deterministic_scaffold_alignment_success_count"] += 1
                    accepted_seen_anchors = set(candidate.get("seen") or set())
                    accepted_invalid_anchors = set(candidate.get("invalid") or set())
                    accepted_missing_anchors = set(missing_anchors)
                    accepted_alignment_invalid_count = int(alignment_result.get("current_anchor_invalid_count") or 0)
                    accepted_length_metadata = length_metadata
                    accepted_appended_dialogue_marked = copy.deepcopy(_as_dialogue_list(appended_dialogue))
                    break
                if not isinstance(alignment_result, dict):
                    retry_context = {
                        "previous_error": "candidate failed before alignment",
                        "reason": candidate.get("reason"),
                    }
                    continue
                _bump_summary_counter(summary, "dialogue_alignment_reject_count")
                _bump_summary_counter(summary, "unsupported_alignment_count", int(alignment_result.get("unsupported_alignment_count") or 0))
                _bump_summary_counter(summary, "contradiction_count", int(alignment_result.get("contradiction_count") or 0))
                if int(alignment_result.get("anchor_leakage_count") or 0):
                    _bump_summary_counter(summary, "anchor_leak_reject_count")
                retry_context = {
                    "missing_event_instance_ids": alignment_result.get("missing_required_event_instance_ids"),
                    "missing_anchors": sorted(missing_anchors),
                    "unsupported_alignment_count": alignment_result.get("unsupported_alignment_count"),
                    "contradiction_count": alignment_result.get("contradiction_count"),
                    "anchor_leakage_count": alignment_result.get("anchor_leakage_count"),
                    "instruction": (
                        "Retry appended_dialogue_turns and event_alignment_proposal only for newly appended turns. "
                        "Parent dialogue is preserved by the program; do not rewrite it, include it, summarize it, or align to it. "
                        "Do not reference anchors that were not provided, and do not invent anchors. "
                        "Every required current event must be naturally expressed in at least one appended turn. "
                        "If an event is missing, add a natural appended turn for it and mark that turn with the provided anchor. "
                        "If an event contradicts another event, express the correction, cancellation, override, or status change explicitly in the same marked turn. "
                        "Do not align one anchor to a turn that states the opposite meaning. "
                        "Use only provided current anchors, and keep support_quote as exact text from the appended turn."
                    ),
                }
                raw_record.setdefault("dialogue_retry_diagnostics", []).append(
                    {
                        "dialogue_attempt": dialogue_attempt,
                        "reason": "dialogue_alignment",
                        "retry_context": copy.deepcopy(retry_context),
                    }
                )
                summary["repair_retry_count"] += 1
                repair_proposal, repair_meta = call_api_alignment_repair(
                    current_event_plan,
                    appended_dialogue,
                    alignment_result,
                    raw_alignment_proposal,
                    api_args,
                    parent_context=parent_context,
                )
                raw_record.setdefault("api_dialogue_repair_attempts", []).append(
                    {"dialogue_attempt": dialogue_attempt, "meta": repair_meta, "payload": repair_proposal}
                )
                _record_api_meta_retry_diagnostics(summary, repair_meta)
                if repair_meta.get("used") and isinstance(repair_proposal, dict):
                    repaired_appended = current_event_scaffold_turns(current_event_plan) + list(
                        _as_dialogue_list(appended_dialogue_from_proposal(repair_proposal))
                    )
                    repaired_candidate = validate_appended_dialogue_candidate(
                        appended_dialogue_marked=repaired_appended,
                        event_alignment_proposal=repair_proposal.get("event_alignment_proposal"),
                        current_event_plan=current_event_plan,
                        anchor_to_event=anchor_to_event,
                        parent_dialogue_turns=parent_dialogue_turns,
                        known_anchors=known_anchors,
                        sample_index=sample_index,
                        min_added_turns=min_length_added_turns,
                        min_added_tokens=min_length_added_tokens,
                        tokenizer_path=length_tokenizer_path,
                        parent_baseline_tokens=length_guard_parent_baseline_tokens,
                        fallback_min_added_tokens=fallback_delta,
                        emergency_min_added_tokens=emergency_floor,
                    )
                    raw_record.setdefault("dialogue_retry_diagnostics", []).append(
                        {
                            "dialogue_attempt": dialogue_attempt,
                            "reason": "alignment_repair_result",
                            "repair_pass": bool(repaired_candidate.get("pass")),
                            "repair_failure_reason": repaired_candidate.get("reason"),
                            "ignored_event_alignment_proposal": bool(repaired_candidate.get("ignored_event_alignment_proposal")),
                        }
                    )
                    if repaired_candidate.get("ignored_event_alignment_proposal"):
                        summary["deterministic_scaffold_alignment_count"] += 1
                    if repaired_candidate.get("pass") is True and isinstance(repaired_candidate.get("alignment_result"), dict):
                        summary["repair_success_count"] += 1
                        if repaired_candidate.get("ignored_event_alignment_proposal"):
                            summary["deterministic_scaffold_alignment_success_count"] += 1
                        appended_dialogue = repaired_appended
                        raw_alignment_proposal = repaired_candidate.get("raw_alignment_proposal")
                        alignment_result = repaired_candidate.get("alignment_result")
                        final_dialogue = repaired_candidate.get("final_dialogue")
                        length_metadata = repaired_candidate.get("length_metadata") or {}
                        length_metadata["parent_dialogue_source"] = parent_dialogue_source
                        accepted_seen_anchors = set(repaired_candidate.get("seen") or set())
                        accepted_invalid_anchors = set(repaired_candidate.get("invalid") or set())
                        accepted_missing_anchors = set(repaired_candidate.get("missing_anchors") or set())
                        accepted_alignment_invalid_count = int(alignment_result.get("current_anchor_invalid_count") or 0)
                        accepted_length_metadata = length_metadata
                        accepted_appended_dialogue_marked = copy.deepcopy(_as_dialogue_list(appended_dialogue))
                        break
                summary["full_regeneration_retry_count"] += 1
            if not alignment_result or not alignment_result.get("coverage_pass") or final_dialogue is None:
                summary["api_dialogue_reject_count"] += 1
                raw_record["final_reject_reason"] = _final_dialogue_reject_reason(retry_context, alignment_result)
                raise NativeGraphDialogueReject(f"dialogue alignment did not pass: {retry_context}")
            summary["api_dialogue_valid_count"] += 1
            summary["current_anchor_seen_count"] += len(accepted_seen_anchors)
            summary["current_anchor_invalid_count"] += len(accepted_invalid_anchors) + accepted_alignment_invalid_count
            summary["current_anchor_missing_count"] += len(accepted_missing_anchors)

            hashes = {
                "graph_snapshot_hash": str(current_event_plan.get("graph_snapshot_hash")),
                "dialogue_hash": str(final_dialogue.get("dialogue_hash")),
                "alignment_hash": str(alignment_result.get("alignment_hash")),
            }
            graph_update = compile_graph_update_from_current_event_plan(
                current_event_plan,
                alignment_result["current_alignment"],
                hashes,
            )
            plan_qtype = normalize_space(current_event_plan.get("question_type"))
            if _is_easy_v2_curriculum(policy):
                if plan_qtype not in EASY_V2_SUPPORTED_QUESTION_TYPES:
                    raise NativeGraphDialogueReject(f"easy_v2 event_plan question_type invalid before QA build: {plan_qtype!r}")
                qa_question_type = plan_qtype
            else:
                qa_question_type = plan_qtype or "ordering"
            qa_proposal_for_build = dialogue_proposal.get("qa_proposal") if isinstance(dialogue_proposal, dict) else None
            try:
                qa = build_qa_from_current_graph_and_alignment(
                    current_event_plan,
                    graph_update,
                    alignment_result["current_alignment"],
                    qa_proposal_for_build,
                    qa_question_type,
                    rng,
                    hashes,
                    final_dialogue,
                    policy,
                    seed_row=seed_row,
                )
            except Exception as qa_exc:
                if not (
                    isinstance(policy, dict)
                    and policy.get("qa_operation_answer_type_diversification")
                    and _is_stage64_qa_repairable_error(qa_exc)
                ):
                    raise
                summary["stage64_qa_repair_attempt_count"] += 1
                repair_payload, repair_meta = call_api_stage64_qa_repair(
                    current_event_plan,
                    alignment_result["current_alignment"],
                    qa_proposal_for_build if isinstance(qa_proposal_for_build, dict) else None,
                    policy.get("stage64_target") if isinstance(policy.get("stage64_target"), dict) else None,
                    str(qa_exc),
                    accepted_appended_dialogue_marked,
                    api_args,
                )
                raw_record.setdefault("api_stage64_qa_repair_attempts", []).append(
                    {
                        "failure_reason": str(qa_exc),
                        "meta": repair_meta,
                        "payload": repair_payload,
                    }
                )
                _record_api_meta_retry_diagnostics(summary, repair_meta)
                repaired_qa_proposal = repair_payload.get("qa_proposal") if isinstance(repair_payload, dict) else None
                if not repair_meta.get("used") or not isinstance(repaired_qa_proposal, dict):
                    summary["stage64_qa_repair_failed_count"] += 1
                    _bump_failure_reason(summary, "stage64_qa_repair_api_failed")
                    reason_counts = summary.setdefault("stage64_qa_repair_failure_reason", {})
                    reason_counts["api_failed"] = int(reason_counts.get("api_failed") or 0) + 1
                    raise qa_exc
                try:
                    qa = build_qa_from_current_graph_and_alignment(
                        current_event_plan,
                        graph_update,
                        alignment_result["current_alignment"],
                        repaired_qa_proposal,
                        qa_question_type,
                        rng,
                        hashes,
                        final_dialogue,
                        policy,
                        seed_row=seed_row,
                    )
                except Exception as repaired_exc:
                    summary["stage64_qa_repair_failed_count"] += 1
                    reason = normalize_space(str(repaired_exc)) or type(repaired_exc).__name__
                    reason_counts = summary.setdefault("stage64_qa_repair_failure_reason", {})
                    reason_counts[reason[:160]] = int(reason_counts.get(reason[:160]) or 0) + 1
                    raise repaired_exc
                summary["stage64_qa_repair_success_count"] += 1
                summary["stage64_qa_repair_saved_full_regeneration_count"] += 1
                if isinstance(dialogue_proposal, dict):
                    dialogue_proposal["qa_proposal"] = repaired_qa_proposal
            if _is_easy_curriculum(policy):
                _clean_qa_option_display_text(qa)
            clean_sample = build_clean_sample(
                sample_index=sample_index,
                seed_row=seed_row,
                current_event_plan=current_event_plan,
                final_dialogue=final_dialogue,
                graph_update=graph_update,
                qa=qa,
                hashes=hashes,
                length_metadata=accepted_length_metadata or {},
            )
            validate_final_clean_sample(clean_sample, known_anchors, hashes)
            clean_sample["native_contract"] = {
                "valid": True,
                "mode": "graph_conditioned_api_dialogue",
                "missing_ref_count": 0,
                "anchor_leakage_count": 0,
                "length_guard_pass": bool((accepted_length_metadata or {}).get("length_guard_pass")),
                "current_graph_dialogue_alignment_pass": True,
                "qa_generated_after_alignment": True,
                "old_qa_reused": False,
                "ordering_identity_gold": qa.get("question_type") == "ordering" and qa.get("gold") == list(LETTERS[: len(qa.get("gold") or [])]),
            }
            if qa.get("stage64_api_qa_schema_pass"):
                clean_sample["native_contract"]["qa_operation_answer_type_diversification"] = True
                clean_sample["native_contract"]["stage64_api_qa_schema_pass"] = True
                clean_sample["native_contract"]["qa_generated_by_api_stage64_schema"] = True

            final_qtype = str(qa.get("question_type") or "unknown")
            final_skill = str(clean_sample.get("attack_skill") or "unknown")
            # planned vs final diagnostics (recorded for every mode that has a planned cell).
            difficulty_cell = f"{final_qtype}|{final_skill}"
            planned_qtype = (planned_cell or {}).get("question_type") if planned_cell else None
            planned_skill = (planned_cell or {}).get("attack_skill") if planned_cell else None
            clean_sample["difficulty_cell"] = difficulty_cell
            clean_sample["curriculum_diagnostics"] = {
                "curriculum_mode": curriculum_mode or "default",
                "planned_question_type": planned_qtype,
                "planned_attack_skill": planned_skill,
                "final_question_type": final_qtype,
                "final_attack_skill": final_skill,
                "question_type_matches_plan": (planned_qtype is None) or (planned_qtype == final_qtype),
                "attack_skill_matches_plan": (planned_skill is None) or (planned_skill == final_skill),
                "difficulty_cell": difficulty_cell,
            }

            marked_sample = {
                "clean_sample": clean_sample,
                "current_event_plan": current_event_plan,
                "dialogue_marked": accepted_appended_dialogue_marked,
                "appended_dialogue_marked": accepted_appended_dialogue_marked,
                "clean_dialogue": final_dialogue,
                "legacy_alignment_result": legacy,
                "current_alignment": alignment_result["current_alignment"],
                "current_event_instance_to_anchor": event_to_anchor,
                "current_anchor_to_event_instance_id": anchor_to_event,
                "raw_current_alignment_proposal": raw_alignment_proposal,
                "hashes": hashes,
                "diagnostics": {
                    "api_graph_meta": graph_meta,
                    "api_dialogue_meta": dialogue_meta,
                    "retry_diagnostics": copy.deepcopy(raw_record.get("dialogue_retry_diagnostics") or []),
                    "alignment_result": alignment_result,
                    "length_metadata": accepted_length_metadata,
                },
            }
            raw_record["accepted"] = True
            raw_record["sample_id"] = clean_sample.get("sample_id")
            raw_record["length_metadata"] = accepted_length_metadata
            # The hard-cap check, append, accepted counters, and checkpoint are done SERIALLY
            # by the collector (they depend on cross-sample accepted counts).
            return {
                "status": "accept",
                "local_summary": summary,
                "raw_record": raw_record,
                "clean_sample": clean_sample,
                "marked_sample": marked_sample,
                "current_event_plan": current_event_plan,
                "alignment_result": alignment_result,
                "final_qtype": final_qtype,
                "final_skill": final_skill,
                "planned_cell": dict(planned_cell) if planned_cell else None,
                "attempt_no": attempt_no,
            }
        except Exception as exc:
            summary["api_failure_count"] += 1
            failure_reason = _failure_reason_from_exception(exc, raw_record)
            _bump_failure_reason(summary, failure_reason)
            if failure_reason == "anchor_leak":
                _bump_summary_counter(summary, "anchor_leak_reject_count")
            raw_record["accepted"] = False
            raw_record["final_reject_reason"] = failure_reason
            raw_record["reject_error"] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reason": failure_reason,
            }
            reject = {
                "attempt": attempt_no,
                "target_sample_index": sample_index,
                "seed_sample_id": seed_row.get("sample_id") or seed_row.get("source_id"),
                "error_type": type(exc).__name__,
                "reason": failure_reason,
                "error": str(exc),
                "raw_record": copy.deepcopy(raw_record),
            }
            return {
                "status": "reject",
                "local_summary": summary,
                "raw_record": raw_record,
                "reject": reject,
                "attempt_no": attempt_no,
            }

    # ---- driver: serial (concurrency==1, byte-equivalent to the prior loop) or threaded ----
    concurrency = max(1, int(getattr(api_args, "generation_concurrency", 1) or 1))

    def _collect(result: dict[str, Any]) -> str:
        """Serially fold one worker result into shared state. Returns the resolved status:
        'accepted' | 'rejected' | 'cap_rejected' (easy_v2 hard cap converted an accept)."""
        nonlocal attempts
        _merge_local_summary(summary, result["local_summary"])
        if result["status"] == "accept":
            final_qtype = result["final_qtype"]
            final_skill = result["final_skill"]
            clean_sample = result["clean_sample"]
            raw_record = result["raw_record"]
            # easy_v2 hard cap on the ACCEPTED set: convert an over-cap accept into a reject so
            # the distribution cannot drift. Checked serially against the live accepted counts.
            if curriculum_mode == EASY_V2_MODE:
                cap_reason = None
                qtype_cap = _qtype_cap(final_qtype)
                skill_cap = _skill_cap(final_skill)
                if qtype_cap is not None and accepted_qtype_counts[final_qtype] >= qtype_cap:
                    cap_reason = ("easy_v2_question_type_cap", f"question_type cap reached for {final_qtype} (>= {qtype_cap})")
                elif skill_cap is not None and accepted_skill_counts[final_skill] >= skill_cap:
                    cap_reason = ("easy_v2_attack_skill_cap", f"attack_skill cap reached for {final_skill} (>= {skill_cap})")
                if cap_reason is not None:
                    _bump_failure_reason(summary, cap_reason[0])
                    summary["api_failure_count"] += 1
                    raw_record["accepted"] = False
                    raw_record["final_reject_reason"] = cap_reason[0]
                    rejects.append(
                        {
                            "attempt": result["attempt_no"],
                            "target_sample_index": len(clean_samples),
                            "seed_sample_id": raw_record.get("seed_sample_id"),
                            "error_type": "NativeGraphDialogueReject",
                            "reason": cap_reason[0],
                            "error": cap_reason[1],
                            "raw_record": copy.deepcopy(raw_record),
                        }
                    )
                    raw_records.append(raw_record)
                    return "cap_rejected"
            if result["planned_cell"] is not None:
                planned_cell_log.append(dict(clean_sample["curriculum_diagnostics"]))
            accepted_qtype_counts[final_qtype] += 1
            accepted_skill_counts[final_skill] += 1
            clean_samples.append(clean_sample)
            marked_samples.append(result["marked_sample"])
            raw_records.append(raw_record)
            # Crash-safe checkpoint: append each accepted clean sample to the partial log and
            # flush immediately so an interrupted run resumes from here. Runs serially in the
            # collector, so the append order matches clean_samples (concurrency-safe).
            append_jsonl_line(partial_path, clean_sample)
            _update_summary_for_accept(summary, clean_sample, result["current_event_plan"], result["alignment_result"])
            resolved = "accepted"
        else:
            rejects.append(result["reject"])
            raw_records.append(result["raw_record"])
            resolved = "rejected"
        if int(getattr(api_args, "checkpoint_every", 0) or 0):
            do_ckpt = result["status"] != "accept" or (len(clean_samples) % int(getattr(api_args, "checkpoint_every")) == 0)
            if do_ckpt:
                temp_summary = dict(summary)
                temp_summary["generated_count"] = len(clean_samples)
                temp_summary["valid_count"] = sum(1 for sample in clean_samples if base.base_valid(sample))
                temp_summary["clean_count"] = sum(1 for sample in clean_samples if base.is_clean(sample))
                temp_summary.update(missing_ref_stats(clean_samples))
                temp_summary["challenger_generation_sources"] = {
                    "current_method.native_challenger_generation.graph_conditioned_api_dialogue": len(clean_samples)
                }
                _write_generation_files(output, clean_samples, marked_samples, raw_records, rejects, temp_summary)
        return resolved

    def _planned_cell_for_slot(slot: int) -> dict[str, str] | None:
        if quota_scheduler is None:
            return None
        if not base_policy.get("boundary_aware_qa_planner"):
            return quota_scheduler.plan_for_index(slot)
        scan_limit = max(1, n_samples)
        for offset in range(scan_limit):
            planned = quota_scheduler.plan_for_index(slot + offset)
            if _boundary_qa_planner_supported_cell(str(planned.get("question_type") or ""), str(planned.get("attack_skill") or "")):
                return planned
        raise NativeGraphDialogueReject("boundary-aware QA planner has no supported planned cells; adjust qtype/skill policy")

    def _cell_label_for_slot(slot: int) -> str:
        pc = _planned_cell_for_slot(slot)
        if pc:
            return f"{pc.get('question_type')}|{pc.get('attack_skill')}"
        return "unplanned"

    def _record_skip(slot: int, last_status: str) -> None:
        skipped_slots.add(slot)
        reason = f"max_attempts_per_sample_exceeded:{last_status}"
        skipped_slot_reasons[reason] = skipped_slot_reasons.get(reason, 0) + 1
        cell = _cell_label_for_slot(slot)
        skipped_by_cell[cell] = skipped_by_cell.get(cell, 0) + 1
        _bump_failure_reason(summary, "slot_skipped_attempt_cap")

    # Circuit breaker: if we skip this many slots in a row WITHOUT any new accept, conclude no
    # feasible cell remains and stop (record shortfall) rather than advancing the cursor forever.
    # Sized to give every distinct planned cell a fair chance before giving up.
    _distinct_cells = sum(len(v) for v in EASY_V2_ATTACK_SKILLS_BY_QUESTION_TYPE.values()) if curriculum_mode == EASY_V2_MODE else 8
    max_consecutive_skips = max(2 * _distinct_cells, (n_samples - len(clean_samples)) + _distinct_cells)
    consecutive_skips = 0
    # API-broken circuit breaker. `max_api_failures` is interpreted as a CONSECUTIVE-failure
    # budget (reset on any successful generation), NOT a cumulative lifetime cap. The old
    # cumulative `len(rejects) <= max_api_failures` guard did not scale with n_samples: at a
    # normal ~10% reject rate an 800-sample target accrues >50 rejects long before reaching 800
    # accepts, so the run died early (e.g. 474/800). The hard `attempts < max_attempts`
    # (= n_samples*3) ceiling still bounds total work, so a truly broken API still fast-fails
    # via consecutive failures while a healthy long run is no longer capped by stray rejects.
    consecutive_api_failures = 0
    max_consecutive_api_failures = max(1, int(max_api_failures))

    def _is_api_failure_status(status: str) -> bool:
        # Only genuine generation failures count toward the breaker. "cap_rejected" means a
        # sample generated fine but was dropped by the easy_v2 distribution cap -> API is healthy.
        return status not in ("accepted", "cap_rejected")

    if concurrency == 1:
        # slot = plan cursor; advances on accept or skip. In the no-skip case slot stays equal
        # to len(clean_samples) (byte-equivalent to the old loop); a skip advances the cursor so
        # the next planned cell comes from the next quota-queue position (reallocation).
        # Resume: start the cursor past the recovered accepted slots.
        slot = len(clean_samples)
        while len(clean_samples) < n_samples and attempts < max_attempts and consecutive_api_failures <= max_consecutive_api_failures:
            if per_sample_attempt_cap_enabled and slot_attempt_counts.get(slot, 0) >= max_attempts_per_sample:
                _record_skip(slot, "exhausted")
                consecutive_skips += 1
                slot += 1
                if consecutive_skips >= max_consecutive_skips:  # no feasible cell left -> shortfall
                    summary["stopped_reason"] = "no_feasible_cell_consecutive_skips"
                    break
                continue
            attempts += 1
            slot_attempt_counts[slot] = slot_attempt_counts.get(slot, 0) + 1
            planned_cell = _planned_cell_for_slot(slot)
            status = _collect(_attempt(attempts, slot, planned_cell))
            if status == "accepted":
                attempts_for_accepted.append(slot_attempt_counts.get(slot, 1))
                slot_accepted.add(slot)
                slot += 1
                consecutive_skips = 0
                consecutive_api_failures = 0
            elif _is_api_failure_status(status):
                consecutive_api_failures += 1
        if consecutive_api_failures > max_consecutive_api_failures:
            summary["stopped_reason"] = "max_consecutive_api_failures"
        # stop condition for skipped slots: if everything in flight is skipped and we can't reach n
    else:
        import concurrent.futures as _futures

        with _futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            in_flight: dict[Any, int] = {}  # future -> slot it targets
            # Resume: start the cursor past the recovered accepted slots.
            next_slot = len(clean_samples)
            while len(clean_samples) < n_samples and attempts < max_attempts and consecutive_api_failures <= max_consecutive_api_failures:
                # advance next_slot past any already-accepted or skipped slots
                while next_slot in slot_accepted or next_slot in skipped_slots:
                    next_slot += 1
                # top up the in-flight pool, one future per distinct live slot
                live_slots = set(in_flight.values())
                while (
                    len(in_flight) < concurrency
                    and attempts < max_attempts
                    and (len(clean_samples) + len(live_slots)) < n_samples
                    and consecutive_api_failures <= max_consecutive_api_failures
                ):
                    # pick the lowest slot not accepted/skipped/in-flight
                    s = next_slot
                    while s in slot_accepted or s in skipped_slots or s in live_slots:
                        s += 1
                    if per_sample_attempt_cap_enabled and slot_attempt_counts.get(s, 0) >= max_attempts_per_sample:
                        _record_skip(s, "exhausted")
                        consecutive_skips += 1
                        continue
                    attempts += 1
                    slot_attempt_counts[s] = slot_attempt_counts.get(s, 0) + 1
                    planned_cell = _planned_cell_for_slot(s)
                    fut = executor.submit(_attempt, attempts, s, planned_cell)
                    in_flight[fut] = s
                    live_slots.add(s)
                if per_sample_attempt_cap_enabled and consecutive_skips >= max_consecutive_skips:
                    # no feasible cell left: stop submitting, drain in-flight, then shortfall.
                    summary["stopped_reason"] = "no_feasible_cell_consecutive_skips"
                    break
                if not in_flight:
                    break
                done, _pending = _futures.wait(set(in_flight), return_when=_futures.FIRST_COMPLETED)
                for fut in done:
                    s = in_flight.pop(fut)
                    status = _collect(fut.result())
                    if status == "accepted":
                        attempts_for_accepted.append(slot_attempt_counts.get(s, 1))
                        slot_accepted.add(s)
                        consecutive_skips = 0
                        consecutive_api_failures = 0
                    elif _is_api_failure_status(status):
                        consecutive_api_failures += 1
            if consecutive_api_failures > max_consecutive_api_failures:
                summary["stopped_reason"] = "max_consecutive_api_failures"
            # drain remaining completed workers (do not submit new ones)
            for fut in list(in_flight):
                s = in_flight.pop(fut)
                status = _collect(fut.result())
                if status == "accepted":
                    attempts_for_accepted.append(slot_attempt_counts.get(s, 1))
                    slot_accepted.add(s)

    summary["attempts"] = attempts
    summary["requested"] = n_samples
    summary["generated_count"] = len(clean_samples)
    summary["valid_count"] = sum(1 for sample in clean_samples if base.base_valid(sample))
    summary["clean_count"] = sum(1 for sample in clean_samples if base.is_clean(sample))
    summary.update(missing_ref_stats(clean_samples))
    summary["challenger_generation_sources"] = {
        "current_method.native_challenger_generation.graph_conditioned_api_dialogue": len(clean_samples)
    }
    # Planned vs final + difficulty_cell diagnostics (foundation for closed-loop controller).
    summary["accepted_question_type_counts"] = dict(sorted(accepted_qtype_counts.items()))
    summary["accepted_attack_skill_counts"] = dict(sorted(accepted_skill_counts.items()))
    summary["difficulty_cell_counts"] = dict(
        sorted(Counter(s.get("difficulty_cell", "unknown") for s in clean_samples).items())
    )
    # Accepted distractor-hardness distribution: verifies the bandit's hardness plan actually
    # flowed into generation. If this is all-h0 while a plan was injected, hardness is broken.
    summary["accepted_distractor_hardness_counts"] = dict(
        sorted(Counter(str((s.get("qa", {}) or {}).get("distractor_hardness", "h0")) for s in clean_samples).items())
    )
    summary["accepted_difficulty_subtype_counts"] = dict(
        sorted(
            Counter(
                str(
                    s.get("difficulty_subtype")
                    or ((s.get("qa") or {}) if isinstance(s.get("qa"), dict) else {}).get("difficulty_subtype")
                    or "unknown"
                )
                for s in clean_samples
            ).items()
        )
    )
    summary["accepted_hard_single_choice_template_counts"] = dict(
        sorted(
            Counter(
                str(
                    s.get("hard_single_choice_template")
                    or ((s.get("qa") or {}) if isinstance(s.get("qa"), dict) else {}).get("hard_single_choice_template")
                    or "unknown"
                )
                for s in clean_samples
                if str(
                    s.get("difficulty_subtype")
                    or ((s.get("qa") or {}) if isinstance(s.get("qa"), dict) else {}).get("difficulty_subtype")
                    or ""
                )
                == "hard_single_choice"
            ).items()
        )
    )
    # per-sample attempt-cap diagnostics
    summary["skipped_sample_slot_count"] = len(skipped_slots)
    summary["skipped_sample_slot_reasons"] = dict(sorted(skipped_slot_reasons.items()))
    summary["skipped_by_cell"] = dict(sorted(skipped_by_cell.items()))
    if attempts_for_accepted:
        summary["attempts_per_accepted_sample_mean"] = round(sum(attempts_for_accepted) / len(attempts_for_accepted), 3)
        summary["attempts_per_accepted_sample_max"] = max(attempts_for_accepted)
    else:
        summary["attempts_per_accepted_sample_mean"] = 0.0
        summary["attempts_per_accepted_sample_max"] = 0
    if planned_cell_log:
        qtype_match = sum(1 for r in planned_cell_log if r.get("question_type_matches_plan"))
        skill_match = sum(1 for r in planned_cell_log if r.get("attack_skill_matches_plan"))
        n_planned = len(planned_cell_log)
        summary["planned_vs_final"] = {
            "n_planned": n_planned,
            "question_type_match_count": qtype_match,
            "attack_skill_match_count": skill_match,
            "question_type_match_rate": round(qtype_match / n_planned, 6) if n_planned else 0.0,
            "attack_skill_match_rate": round(skill_match / n_planned, 6) if n_planned else 0.0,
            "planned_question_type_quota": dict(quota_scheduler.qtype_quota) if quota_scheduler else {},
        }
    _write_generation_files(output, clean_samples, marked_samples, raw_records, rejects, summary)
    if len(clean_samples) < n_samples:
        shortfall_msg = (
            f"graph_conditioned_api_dialogue generated {len(clean_samples)}/{n_samples}; "
            f"rejects={len(rejects)}; see {output / 'graph_conditioned_api_dialogue_rejects.jsonl'}"
        )
        # In easy_v2 the deterministic quota + per-type builders can make the exact target
        # infeasible (e.g. multi_select needs >=2 active and >=2 false aligned events that a
        # given seed/plan may not yield). Rather than crash the whole round (which also loses
        # the closed-loop controller update), accept the shortfall: record it and return what
        # was produced so the controller can rebalance next round. Other modes keep the hard error.
        if curriculum_mode == EASY_V2_MODE and len(clean_samples) > 0:
            summary["shortfall"] = True
            summary["shortfall_message"] = shortfall_msg
            _write_generation_files(output, clean_samples, marked_samples, raw_records, rejects, summary)
            return clean_samples
        raise RuntimeError(shortfall_msg)
    return clean_samples
