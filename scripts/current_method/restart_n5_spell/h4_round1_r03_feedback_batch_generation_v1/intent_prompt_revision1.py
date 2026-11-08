"""Compact rendering of the same frozen intent policy and schema.

This revision changes prompt presentation only. The full JSON Schema remains
the local validator; the model sees an explicit record shape and short rules.
"""

import json

from . import SLOTS
from .contract import SCHEMA
from .prompts import compact, replay_view


def render(dialogue, packet):
    shapes = []
    for i, slot in enumerate(SLOTS, 1):
        shapes.append({'intent_id': f'I{i:03}', 'intent_type': slot,
                       'status': 'READY or NO_SAFE_EVOLUTION',
                       'strategy': 'one allowed strategy, or NO_SAFE_EVOLUTION',
                       'target_relation': 'a relation to test, not a question; null if NO_SAFE',
                       'source_feedback_ids': [], 'coverage_gap_ids': [],
                       'required_r03_evidence': [],
                       'reasoning_chain': {'before_state': [], 'trigger': [], 'after_state': []},
                       'no_safe_reason': None})
    dtext = '\n'.join(f"{t['turn_ref']} | {t['speaker']} | {t['text']}" for t in dialogue['turns'])
    r03 = [t['turn_ref'] for t in dialogue['turns'] if t['turn_ref'].startswith('R03:')]
    return '\n'.join([
        'ROLE: QA intent miner. Plan exactly three slots. Do NOT write questions or QA batches.',
        'The cumulative dialogue below is the only factual source. Historical QA are replay/curriculum metadata.',
        'Solver numbers are behavioral observations, not diagnoses. Do not output weaknesses or invent strategies.',
        'SLOT I001 SUCCESSOR_FRONTIER: evolve a historical feedback seed into a distinct relation that NEEDS R03.',
        'SLOT I002 COVERAGE_GAP: use a supplied gap ID and at least one of that gap\'s uncited R02/R03 turns.',
        'SLOT I003 FRESH_R03: a new R03 relation independent of old QA. source_feedback_ids must be [].',
        'Every slot may return NO_SAFE_EVOLUTION. Never invent a relation merely to fill the slot.',
        'ALLOWED STRATEGIES BY SLOT:', compact(SCHEMA['x-human-defined-slot-strategies']),
        'HISTORICAL FEEDBACK (not factual authority):', compact(packet['eligible_feedback']),
        'PROGRAM COVERAGE CANDIDATES (uncited refs, semantic gap unverified):', compact(packet['coverage_gaps']),
        'HISTORICAL QUESTION STEMS TO AVOID REPLAYING:', compact(replay_view(packet['history'])),
        'BEGIN AUTHORITATIVE CUMULATIVE DIALOGUE', dtext, 'END AUTHORITATIVE CUMULATIVE DIALOGUE',
        'R03 TURN REF ALLOWLIST:', compact(r03),
        'FINAL OUTPUT REQUIREMENTS:',
        'Return exactly the following 3-record structure. Include EVERY field even when its value is [] or null.',
        'For READY: target_relation is a short distinction/relationship, NOT a question. no_safe_reason MUST be null.',
        'For READY: each reasoning_chain group has one or more real refs; no ref may appear in two groups.',
        'Use only refs actually printed in the dialogue. Do not extend turn numbering beyond the printed turns.',
        'required_r03_evidence may ONLY contain refs from the R03 allowlist. These may supplement reasoning_chain.',
        'I001 and I003 READY require at least one required_r03_evidence. I001 READY requires a real feedback_id.',
        'I002 READY requires a real gap_id whose uncited refs overlap reasoning_chain.',
        'Without an eligible feedback seed, I001 must be NO_SAFE_EVOLUTION.',
        'For NO_SAFE_EVOLUTION: status=strategy=NO_SAFE_EVOLUTION; target_relation=null; all arrays empty;',
        'reasoning_chain={"before_state":[],"trigger":[],"after_state":[]}; no_safe_reason is a nonempty reason.',
        'OUTPUT RECORD SHAPE (replace instructions with values; omit no fields):',
        json.dumps({'intents': shapes}, ensure_ascii=False, indent=2),
        'Check all three records contain no_safe_reason, source_feedback_ids and coverage_gap_ids. JSON only.',
    ])
