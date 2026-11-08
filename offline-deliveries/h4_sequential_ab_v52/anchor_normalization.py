"""Mechanical A citation normalization; no inferred events or semantic approval."""
from copy import deepcopy
import re


POLICY = "v52_A_exact_unique_quote_speaker_current_history"


def normalize(anchors, root, history, require, digest):
    require(isinstance(anchors, list), "plan_anchor_invalid")
    refs = {}
    for turn in history:
        refs.setdefault(turn["turn_ref"], []).append(turn)
    normalized, changes = [], []
    for index, original in enumerate(anchors):
        require(isinstance(original, dict), "plan_anchor_invalid")
        ref, quote = original.get("turn_ref"), original.get("quote")
        require(isinstance(ref, str) and isinstance(quote, str) and quote.strip(), "plan_anchor_invalid")
        candidate = ref
        operations = []
        if ref not in refs:
            # Only R01's known namespace variants are aliases. Never search for
            # another ref solely because a quote happens to occur there.
            if candidate.startswith(root + ":"):
                candidate = candidate[len(root) + 1:]
                operations.append("remove_exact_current_root_prefix")
            require(re.fullmatch(r"R01:S[0-9]{2}:T[0-9]{3}", candidate, re.I), "plan_anchor_invalid")
            if candidate != candidate.upper():
                operations.append("canonicalize_R01_ref_case")
            candidate = candidate.upper()
        require(candidate in refs and len(refs[candidate]) == 1, "plan_anchor_invalid")
        turn = refs[candidate][0]
        require(quote in turn["text"], "plan_anchor_invalid")
        supplied_speaker = original.get("speaker")
        if "speaker" in original:
            require(isinstance(supplied_speaker, str) and supplied_speaker == turn["speaker"],
                    "plan_anchor_speaker_mismatch")
        # A corrected legacy alias may lack speaker. Recover it only when the
        # verbatim quote identifies one turn in the complete current history.
        # Already-literal legacy refs need no repair or new rejection rule.
        matches = [t for t in history if quote in t["text"] and
                   (supplied_speaker is None or t["speaker"] == supplied_speaker)]
        if operations:
            require(len(matches) == 1 and matches[0]["turn_ref"] == candidate,
                    "plan_anchor_ambiguous_quote_speaker")
        if operations and "speaker" not in original:
            operations.append("derive_speaker_from_unique_exact_quote")
        anchor = {**deepcopy(original), "turn_ref": candidate}
        if operations:
            anchor["speaker"] = turn["speaker"]
        normalized.append(anchor)
        changes.append({"anchor_index": index, "original": deepcopy(original),
                        "normalized": deepcopy(anchor), "operations": operations,
                        "matched_history_speaker": turn["speaker"],
                        "quote_speaker_matching_refs": [t["turn_ref"] for t in matches],
                        "matched_turn_sha256": digest(turn)})
    return normalized, {"policy": POLICY, "history_sha256": digest(history),
                        "history_turns": len(history), "anchors": changes,
                        "semantic_certified": False}
