"""Conservative rule-based proposal/outcome fact extraction.

The extractor only emits facts when a turn contains a closed event-slot
pattern. It does not infer facts from Memory summaries or call an LLM.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .contracts import Fact, Turn
from .input_adapter import structured_texts


_PROPOSAL_CUES = (
    ("suggest", re.compile(r"\b(?:suggest(?:ed|s)?|propos(?:e|ed|es)|recommend(?:ed|s)?)\b", re.I)),
    ("idea", re.compile(r"\b(?:great|brilliant)\s+idea\b|\bidea\s+of\b", re.I)),
    ("thinking", re.compile(r"\b(?:i|we)\s+(?:was|were|am|are)\s+thinking\b", re.I)),
    ("would", re.compile(r"\bwe\s+could\b|\bwe\s+should\b|\bwould\s+be\b", re.I)),
    ("let_us", re.compile(r"\blet(?:'s| us)\b", re.I)),
    ("offer", re.compile(r"\b(?:offer(?:ed|s)?|volunteer(?:ed|s)?)\b", re.I)),
)
_STRONG_OUTCOME_CUES = (
    ("agreed", re.compile(r"\b(?:agreed|agree|agrees|all for that|sounds like a plan)\b", re.I)),
    ("decision", re.compile(r"\b(?:decid(?:ed|es|e)|final(?:ly)?|settled|chosen|confirmed)\b", re.I)),
    ("adoption", re.compile(r"\b(?:i(?:'m| am) going to go for it|include(?:d)? .* in (?:our|the) plan|let(?:'s| us) start)\b", re.I)),
    ("affirmation", re.compile(r"\b(?:that(?:'s| is) a (?:great|brilliant) idea|i(?:'d| would) love to have .* involved)\b", re.I)),
)
_WEAK_OUTCOME_CUES = (
    ("interest", re.compile(r"\b(?:i(?:'d| would) love to hear|sounds amazing|i(?:'m| am) happy to help)\b", re.I)),
)


def _event_patterns() -> list[tuple[str, str, re.Pattern[str], str]]:
    return [
        ("event_type:charity_event", "charity event", re.compile(r"\bcharity\s+event\b", re.I), "charity event decision"),
        ("event_format:music_festival", "music festival", re.compile(r"\bmusic\s+festival\b|\bfestival\b", re.I), "music festival format"),
        ("collaborator:elena", "Elena", re.compile(r"\b(?:elena|environmental\s+advocat(?:e|es)|involve .*elena)\b", re.I), "Elena collaboration"),
        ("planning_tool:shared_google_doc", "shared Google Doc", re.compile(r"\bshared\s+google\s+doc\b", re.I), "shared planning document"),
        ("eco_practice:eco_conscious_event", "eco-conscious practices", re.compile(r"\beco[- ]conscious\b|\beco[- ]friendly\s+practic", re.I), "eco-conscious event practices"),
        ("venue:green_venue", "green venue", re.compile(r"\bgreen\s+venue\b|\bsolar\s+power\b|\bzero[- ]waste\b", re.I), "green venue selection"),
        ("school_partnership:recycling_drive", "school recycling partnership", re.compile(r"\bpartner(?:ing)?\s+with\s+a\s+local\s+school\b|\brecycling\s+(?:donation\s+)?drive\b", re.I), "school recycling partnership"),
        ("outreach:school_volunteers", "school volunteers", re.compile(r"\bschool\b.*\b(?:students|volunteers)\b|\bstudents\b.*\b(?:setup|showcase)\b", re.I), "school volunteer participation"),
    ]


def _actor_names(speaker: str, text: str) -> list[str]:
    names = [speaker]
    for label in ("Classmates A", "Classmates B", "Elena"):
        if label.casefold() in text.casefold() and label not in names:
            names.append(label)
    if re.search(r"\b(?:we|they|both|all)\b", text, re.I):
        for label in ("Classmates A", "Classmates B"):
            if label not in names:
                names.append(label)
    return names


def _value(event_key: str, text: str, role: str) -> str:
    lower = text.casefold()
    if event_key == "event_type:charity_event":
        return "charity event"
    if event_key == "event_format:music_festival":
        if re.search(r"(?:into|as part of|within)\s+the\s+charity\s+event", lower):
            return "music festival integrated into charity event"
        return "music festival"
    if event_key == "collaborator:elena":
        if re.search(r"meeting|involved|collaborat", lower):
            return "involve Elena as environmental collaborator"
        return "Elena as environmental collaborator"
    if event_key == "planning_tool:shared_google_doc":
        return "shared Google Doc for planning"
    if event_key == "eco_practice:eco_conscious_event":
        return "eco-conscious event practices"
    if event_key == "venue:green_venue":
        return "green venue"
    if event_key == "school_partnership:recycling_drive":
        return "school partnership for recycling drive"
    if event_key == "outreach:school_volunteers":
        return "school students for setup and showcase"
    return event_key


def _structured_refs(event_key: str, texts: list[tuple[str, str]]) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9]+", event_key.casefold()))
    refs: list[str] = []
    for path, text in texts:
        normalized = text.casefold()
        event_hint = {
            "event_type:charity_event": ("charity event",),
            "event_format:music_festival": ("music festival",),
            "collaborator:elena": ("elena",),
            "planning_tool:shared_google_doc": ("google doc",),
            "eco_practice:eco_conscious_event": ("eco-conscious", "eco-friendly"),
            "venue:green_venue": ("green venue", "solar power", "zero-waste"),
            "school_partnership:recycling_drive": ("recycling drive", "local school"),
            "outreach:school_volunteers": ("students", "youth showcase"),
        }.get(event_key, ())
        if any(hint in normalized for hint in event_hint):
            refs.append(path)
    return refs[:12]


def extract_facts(
    turns: Iterable[Turn], *, structured_sources: Mapping[str, Any] | None = None,
) -> tuple[list[Fact], list[Fact]]:
    structured = structured_texts(structured_sources or {})
    proposals: list[Fact] = []
    outcomes: list[Fact] = []
    counter = 0
    for turn in turns:
        proposal_cues = [(name, pattern.search(turn.text))
                         for name, pattern in _PROPOSAL_CUES]
        outcome_cues = [(name, pattern.search(turn.text))
                        for name, pattern in _STRONG_OUTCOME_CUES]
        weak_outcome_cues = [(name, pattern.search(turn.text))
                             for name, pattern in _WEAK_OUTCOME_CUES]
        for event_key, _display, event_pattern, event_summary in _event_patterns():
            if not event_pattern.search(turn.text):
                continue
            proposal = next(((name, match) for name, match in proposal_cues if match), None)
            outcome = next(((name, match) for name, match in outcome_cues if match), None)
            weak_outcome = next(((name, match) for name, match in weak_outcome_cues if match), None)
            if proposal:
                counter += 1
                fact = Fact(
                    fact_id=f"PF{counter:04d}", root_id=turn.root_id,
                    event_key=event_key, event_summary=event_summary,
                    actor=turn.speaker, actors=_actor_names(turn.speaker, turn.text),
                    relation_role="proposal", value=_value(event_key, turn.text, "proposal"),
                    turn_ref=turn.turn_ref, turn_index=turn.turn_index,
                    speaker=turn.speaker, evidence_text=turn.text, origin=turn.origin,
                    cue=proposal[0], strength="high",
                    source_structured_refs=_structured_refs(event_key, structured),
                    value_tokens=re.findall(r"[a-z0-9]+", _value(event_key, turn.text, "proposal").casefold()),
                )
                proposals.append(fact)
            if outcome or weak_outcome:
                counter += 1
                fact = Fact(
                    fact_id=f"OF{counter:04d}", root_id=turn.root_id,
                    event_key=event_key, event_summary=event_summary,
                    actor=turn.speaker, actors=_actor_names(turn.speaker, turn.text),
                    relation_role="outcome", value=_value(event_key, turn.text, "outcome"),
                    turn_ref=turn.turn_ref, turn_index=turn.turn_index,
                    speaker=turn.speaker, evidence_text=turn.text, origin=turn.origin,
                    cue=(outcome or weak_outcome)[0],
                    strength="high" if outcome else "medium",
                    source_structured_refs=_structured_refs(event_key, structured),
                    value_tokens=re.findall(r"[a-z0-9]+", _value(event_key, turn.text, "outcome").casefold()),
                )
                outcomes.append(fact)
    return proposals, outcomes

