"""Internal contracts for the isolated atomic-to-composite QA probe."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping


COMPOSITION_TYPES = (
    "state_change",
    "proposal_vs_outcome",
    "actor_outcome_binding",
    "uncertain",
)
COMPOSER_TYPES = set(COMPOSITION_TYPES) - {"uncertain"}
OPTION_KEYS = ("A", "B", "C", "D")

_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as",
    "at", "be", "been", "being", "b", "by", "can", "did", "do", "does",
    "during", "for", "from", "had", "has", "have", "how", "in", "into",
    "is", "it", "its", "more", "of", "on", "or", "over", "that", "the",
    "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "which", "while", "who", "with", "would", "you", "your",
    "classmates", "event", "conversation", "process", "planning", "earlier",
    "following", "best", "describes", "occurred", "regarding", "during",
}

_ENTITY_PATTERNS = (
    ("classmates_a", (r"\bclassmates?\s+a\b", r"\bclassmate\s+a\b",
                       r"\bclassmates?\s+a\s+and\s+b\b")),
    ("classmates_b", (r"\bclassmates?\s+b\b", r"\bclassmate\s+b\b",
                       r"\bclassmates?\s+a\s+and\s+b\b")),
    ("elena", (r"\belena\b",)),
)

# These are deliberately small, canonical topic labels.  They are used as
# lexical evidence for pair generation, never as evidence or gold repair.
_TOPIC_PATTERNS = (
    ("music", (r"\bmusic\b", r"\bmusician", r"\bfestival")),
    ("environment", (r"\benvironment", r"\bclimate", r"\bsustainab")),
    ("charity", (r"\bcharity\b", r"\bdonation", r"\bfundrais")),
    ("outreach", (r"\boutreach\b", r"\bvolunteer", r"\bsponsor")),
    ("meeting", (r"\bmeeting\b", r"\bthursday\b", r"\b5\s*pm\b")),
    ("school_community", (r"\bschool", r"\bcommunity cent")),
    ("logistics", (r"\blogistics\b", r"\bvenue", r"\bbudget")),
    ("support", (r"\bsupport", r"\bencourag", r"\bgratitude")),
    ("schedule", (r"\bmonday\b", r"\btuesday\b", r"\bwednesday\b",
                  r"\bthursday\b", r"\bfriday\b", r"\bsaturday\b",
                  r"\bsunday\b", r"\btomorrow\b", r"\bnext week\b",
                  r"\bdate\b", r"\btime\b")),
    ("practice", (r"\bpractice\b", r"\btraining\b", r"\brehears")),
)

_DIMENSION_PATTERNS = (
    ("perspective", (r"\bapproach\b", r"\bperspective\b", r"\bfocus\b",
                      r"\bview", r"\battitude\b")),
    ("state", (r"\bstate change\b", r"\bshift\b", r"\bchange\b",
                r"\brevise", r"\bfrom .+ to ")),
    ("decision", (r"\bdecision\b", r"\bdecided\b", r"\bagreed\b",
                   r"\bchosen\b", r"\bintegrat")),
    ("role", (r"\brole", r"\bhandled\b", r"\btask\b", r"\bcommitted\b")),
    ("goal", (r"\bgoal", r"\bpurpose\b", r"\breason\b", r"\bcause\b")),
    ("involvement", (r"\binvolv", r"\bavailable\b", r"\baccess\b")),
    ("schedule", (r"\bday\b", r"\bdate\b", r"\btime\b", r"\bweek\b",
                  r"\btomorrow\b", r"\bschedule\b")),
)

_QUESTION_TYPE_DIMENSIONS = {
    "state_change": {"state"},
    "information_revision": {"state", "involvement"},
    "cross_turn_comparison": set(),
    "relationship_or_decision": {"decision"},
    "task_dependency_or_constraint": {"role"},
    "information_visibility": {"involvement"},
    "causal": {"goal"},
    "temporal_sequence": {"state"},
}


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def lexical_tokens(value: Any) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize_text(value))
    return {word for word in words if len(word) > 2 and word not in _STOPWORDS}


def extract_entities(*values: Any) -> list[str]:
    text = " ".join(normalize_text(value) for value in values)
    found = [name for name, patterns in _ENTITY_PATTERNS
             if any(re.search(pattern, text) for pattern in patterns)]
    return found


def extract_topic_terms(*values: Any) -> list[str]:
    text = " ".join(normalize_text(value) for value in values)
    found = [name for name, patterns in _TOPIC_PATTERNS
             if any(re.search(pattern, text) for pattern in patterns)]
    return found


def extract_state_dimensions(question_type: Any, *values: Any) -> list[str]:
    text = " ".join(normalize_text(value) for value in values)
    found = set(_QUESTION_TYPE_DIMENSIONS.get(str(question_type), set()))
    for name, patterns in _DIMENSION_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            found.add(name)
    return sorted(found)


def is_duplicate_fact(left: "AtomicQA", right: "AtomicQA") -> bool:
    """Detect exact/near-exact fact duplicates without semantic rewriting."""
    if normalize_text(left.gold_text) == normalize_text(right.gold_text):
        return True
    left_fact = lexical_tokens(left.gold_text) | lexical_tokens(left.question)
    right_fact = lexical_tokens(right.gold_text) | lexical_tokens(right.question)
    if not left_fact or not right_fact:
        return False
    overlap = len(left_fact & right_fact) / max(1, min(len(left_fact), len(right_fact)))
    question_overlap = len(lexical_tokens(left.question) & lexical_tokens(right.question))
    return overlap >= 0.88 and question_overlap >= 3


@dataclass(frozen=True)
class AtomicQA:
    qa_id: str
    root_id: str
    question: str
    gold_text: str
    correct_option: str
    options: dict[str, str]
    evidence_turn_refs: list[str]
    evidence_texts: list[str]
    source_scope: str
    entities: list[str] = field(default_factory=list)
    event_terms: list[str] = field(default_factory=list)
    turn_indices: list[int] = field(default_factory=list)
    question_type: str = ""
    grounding_decision: str = "pass_grounding"
    source_path: str = ""

    @property
    def state_dimensions(self) -> list[str]:
        return extract_state_dimensions(
            self.question_type, self.question, self.gold_text,
        )

    @property
    def gold_tokens(self) -> set[str]:
        return lexical_tokens(self.gold_text)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state_dimensions"] = self.state_dimensions
        return value


def shared_nontrivial_topics(left: AtomicQA, right: AtomicQA) -> list[str]:
    return sorted(set(left.event_terms) & set(right.event_terms))
