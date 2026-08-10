"""Pure local diagnostics for answer-shape leakage in MCQ option sets."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence


OPTION_DIAGNOSTIC_VERSION = (
    "mcq_option_shape_diagnostic_v5_shared_false_trajectory")
MEDOID_LEAKAGE_ADVANTAGE_RATIO = 0.05
QUESTION_CUE_MIN_EXCLUSIVE_TOKENS = 2
GOLD_LENGTH_ADVANTAGE_RATIO = 0.60
SMALL_MUTATION_LEAKAGE_COUNT = 2

_QUESTION_CUE_STOPWORDS = frozenset({
    "about", "after", "again", "also", "approach", "before", "being",
    "decide", "decided", "does", "doing", "during", "first", "from",
    "have", "having", "initially", "into", "later", "regarding", "said",
    "says", "their", "them", "then", "they", "this", "those", "what",
    "when", "where", "which", "while", "with", "would",
})


def _tokens(value: Any) -> list[str]:
    return re.findall(r"[\w']+", str(value or "").casefold())


def _content_token_set(value: Any) -> set[str]:
    return {
        token for token in _tokens(value)
        if len(token) >= 4 and token not in _QUESTION_CUE_STOPWORDS
    }


def _content_bigrams(value: Any) -> set[str]:
    tokens = [token for token in _tokens(value)
              if len(token) >= 4 and token not in _QUESTION_CUE_STOPWORDS]
    return {" ".join(tokens[index:index + 2])
            for index in range(len(tokens) - 1)}


def diagnose_question_option_cues(
    *, question: str, options: Mapping[str, str], reference_answer_letter: str,
) -> dict[str, Any]:
    """Detect question words that uniquely identify the gold option.

    This is deliberately comparative. A topic word may appear in the question,
    but it must not identify only the gold while all three alternatives silently
    switch to unrelated topics.
    """
    letters = tuple(options)
    if letters != ("A", "B", "C", "D") or reference_answer_letter not in letters:
        raise ValueError("mcq_question_cue_diagnostic_input_invalid")
    question_tokens = _content_token_set(question)
    overlap = {
        letter: sorted(question_tokens & _content_token_set(options[letter]))
        for letter in letters
    }
    gold_tokens = set(overlap[reference_answer_letter])
    distractor_tokens = set().union(*(
        set(overlap[letter]) for letter in letters
        if letter != reference_answer_letter
    ))
    exclusive_gold = sorted(gold_tokens - distractor_tokens)
    distractor_counts = [
        len(overlap[letter]) for letter in letters
        if letter != reference_answer_letter
    ]
    gold_overlap_advantage = (
        len(gold_tokens) - max(distractor_counts, default=0))
    leakage = (
        len(exclusive_gold) >= QUESTION_CUE_MIN_EXCLUSIVE_TOKENS and
        gold_overlap_advantage >= QUESTION_CUE_MIN_EXCLUSIVE_TOKENS
    )
    return {
        "diagnostic_version": "mcq_question_option_cue_diagnostic_v1",
        "question_content_tokens": sorted(question_tokens),
        "question_option_overlap": overlap,
        "exclusive_gold_question_tokens": exclusive_gold,
        "gold_overlap_advantage": gold_overlap_advantage,
        "minimum_exclusive_token_threshold":
            QUESTION_CUE_MIN_EXCLUSIVE_TOKENS,
        "question_cue_leakage_risk": leakage,
    }


def _edit_profile(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    matcher = SequenceMatcher(a=list(left), b=list(right), autojunk=False)
    changed_left = changed_right = blocks = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        blocks += 1
        changed_left += i2 - i1
        changed_right += j2 - j1
    return {
        "change_block_count": blocks,
        "changed_token_count": max(changed_left, changed_right),
        "similarity_ratio": matcher.ratio(),
    }


def diagnose_option_shape(
    *, options: Mapping[str, str], reference_answer_letter: str,
    question: str | None = None,
) -> dict[str, Any]:
    """Detect the gold-plus-three-small-mutations pattern without semantic judging."""
    letters = tuple(options)
    if letters != ("A", "B", "C", "D") or reference_answer_letter not in letters:
        raise ValueError("mcq_option_diagnostic_input_invalid")
    tokenized = {letter:_tokens(options[letter]) for letter in letters}
    pairwise: dict[str, dict[str, Any]] = {}
    total_distance = {letter:0.0 for letter in letters}
    for index, left in enumerate(letters):
        for right in letters[index + 1:]:
            profile = _edit_profile(tokenized[left], tokenized[right])
            pairwise[f"{left}:{right}"] = profile
            distance = 1.0 - float(profile["similarity_ratio"])
            total_distance[left] += distance
            total_distance[right] += distance
    minimum = min(total_distance.values())
    medoids = [letter for letter, value in total_distance.items()
               if abs(value - minimum) < 1e-12]
    gold_profiles = []
    for letter in letters:
        if letter == reference_answer_letter:
            continue
        key = (f"{reference_answer_letter}:{letter}"
               if letters.index(reference_answer_letter) < letters.index(letter)
               else f"{letter}:{reference_answer_letter}")
        profile = dict(pairwise[key])
        profile["distractor_letter"] = letter
        profile["small_local_mutation"] = (
            profile["change_block_count"] <= 2 and
            profile["changed_token_count"] <= 4 and
            profile["similarity_ratio"] >= 0.85)
        gold_profiles.append(profile)
    mutation_count = sum(row["small_local_mutation"] for row in gold_profiles)
    single_mutation_star = mutation_count >= SMALL_MUTATION_LEAKAGE_COUNT
    correct_unique_medoid = medoids == [reference_answer_letter]
    ordered_distances = sorted(total_distance.values())
    second_distance = ordered_distances[1]
    medoid_advantage_ratio = (
        (second_distance - minimum) / second_distance
        if correct_unique_medoid and second_distance > 0 else 0.0
    )
    correct_unique_medoid_leakage = (
        correct_unique_medoid and
        medoid_advantage_ratio >= MEDOID_LEAKAGE_ADVANTAGE_RATIO
    )
    question_cues = (diagnose_question_option_cues(
        question=question, options=options,
        reference_answer_letter=reference_answer_letter)
        if question is not None else None)
    question_cue_risk = bool(
        question_cues and question_cues["question_cue_leakage_risk"])
    gold_length = len(tokenized[reference_answer_letter])
    longest_distractor_length = max(
        len(tokens) for letter, tokens in tokenized.items()
        if letter != reference_answer_letter)
    gold_length_advantage_ratio = (
        (gold_length - longest_distractor_length) / longest_distractor_length
        if longest_distractor_length else 0.0)
    gold_length_leakage = (
        gold_length_advantage_ratio >= GOLD_LENGTH_ADVANTAGE_RATIO)
    distractor_letters = [letter for letter in letters
                          if letter != reference_answer_letter]
    shared_distractor_bigrams = set.intersection(*(
        _content_bigrams(options[letter]) for letter in distractor_letters))
    shared_distractor_only_bigrams = sorted(
        shared_distractor_bigrams - _content_bigrams(
            options[reference_answer_letter]))
    shared_false_trajectory_risk = bool(shared_distractor_only_bigrams)
    return {
        "diagnostic_version": OPTION_DIAGNOSTIC_VERSION,
        "reference_answer_letter": reference_answer_letter,
        "option_token_counts": {letter:len(tokens)
                                for letter, tokens in tokenized.items()},
        "pairwise_profiles": pairwise,
        "gold_to_distractor_profiles": gold_profiles,
        "small_local_mutation_distractor_count": mutation_count,
        "small_local_mutation_leakage_count_threshold":
            SMALL_MUTATION_LEAKAGE_COUNT,
        "single_mutation_star": single_mutation_star,
        "shared_distractor_only_bigrams": shared_distractor_only_bigrams,
        "shared_false_trajectory_risk": shared_false_trajectory_risk,
        "total_normalized_distance": total_distance,
        "medoid_letters": medoids,
        "correct_unique_medoid": correct_unique_medoid,
        "medoid_advantage_ratio": medoid_advantage_ratio,
        "medoid_leakage_advantage_ratio_threshold":
            MEDOID_LEAKAGE_ADVANTAGE_RATIO,
        "correct_unique_medoid_leakage": correct_unique_medoid_leakage,
        "question_option_cue_diagnostic": question_cues,
        "question_cue_leakage_risk": question_cue_risk,
        "gold_length": gold_length,
        "longest_distractor_length": longest_distractor_length,
        "gold_length_advantage_ratio": gold_length_advantage_ratio,
        "gold_length_advantage_ratio_threshold": GOLD_LENGTH_ADVANTAGE_RATIO,
        "gold_length_leakage": gold_length_leakage,
        "answer_shape_leakage_risk": (
            single_mutation_star or correct_unique_medoid_leakage or
            question_cue_risk or gold_length_leakage or
            shared_false_trajectory_risk),
    }


__all__ = ["GOLD_LENGTH_ADVANTAGE_RATIO", "MEDOID_LEAKAGE_ADVANTAGE_RATIO",
           "OPTION_DIAGNOSTIC_VERSION",
           "QUESTION_CUE_MIN_EXCLUSIVE_TOKENS", "SMALL_MUTATION_LEAKAGE_COUNT",
           "diagnose_option_shape",
           "diagnose_question_option_cues"]
