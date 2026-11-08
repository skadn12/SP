"""Human-defined templates. Solver observations enter the miner only."""

import json

from . import HERE
from .contract import SCHEMA, reasoning_contract
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt as guided
from ..h4_qa_mining_hybrid_generation_v1.free_exploration_generator import build_prompt as free
from ..h4_qa_mining_hybrid_generation_v1.hybrid_contract import expected_counts


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def replay_view(history):
    # Hashes alone are opaque to a language model. Supply readable stems.
    # Citation coverage is a separate miner input; repeating all historical
    # references here needlessly dominates the context. Full records stay local.
    seen = set()
    result = []
    for row in history:
        stem = ' '.join(str(row.get('question') or '').casefold().split())
        if stem not in seen:
            result.append(row['question'])
            seen.add(stem)
    return result


def intent_prompt(dialogue, packet):
    return '\n'.join([
        'ROLE: R03 QA Intent Miner. Apply the following frozen human policy.',
        (HERE / 'qa_feedback_policy_v2.md').read_text(),
        'OUTPUT_SCHEMA:', compact(SCHEMA),
        'Return intents only. Use canonical turn_refs, not 1-based numeric positions.',
        'FROZEN_FEEDBACK_PACKET (observations, not story facts):', compact({
            'eligible_feedback': packet['eligible_feedback'], 'coverage_gaps': packet['coverage_gaps']}),
        'HISTORICAL_QUESTIONS_FOR_REPLAY_CHECK:', compact(replay_view(packet['history'])),
        'AUTHORITATIVE_CUMULATIVE_DIALOGUE:',
        '\n'.join(f"{t['turn_ref']} | {t['speaker']} | {t['text']}" for t in dialogue['turns']),
        'Return JSON only with the three fixed intent slots. NO_SAFE_EVOLUTION is allowed in every slot.',
    ])


def generator_prompt(dialogue, intent, history, mode, previous=(), repair_type=None, repair_target_count=1):
    spec = (guided if mode == 'contract_guided' else free)(dialogue, reasoning_contract(intent))
    lines = spec['prompt'].splitlines()
    # Keep the first-round field examples, evidence layout and mode distinction;
    # replace quota/diagnosis language to implement the agreed policy exceptions.
    lines = [line for line in lines if not line.startswith((
        'Generate exactly', 'Explore any grounded question that may expose Solver weaknesses:',
        'Do not force the intent contract'))]
    scope = {k: intent[k] for k in ('intent_id', 'intent_type', 'strategy', 'target_relation',
                                   'reasoning_chain', 'required_r03_evidence')}
    counts = expected_counts(mode)
    target = ('Generate ONLY question_type ' + repair_type + '; target ' + str(repair_target_count)
              + ' distinct valid candidates. Do not substitute single_choice or another type.'
              if repair_type else 'Target type counts: ' + compact(counts) + '.')
    rules = [target, 'Targets are not quotas. Return fewer, including zero candidates, if necessary; never pad.',
             'All facts and Gold must follow the cumulative dialogue. Historical QA/Gold are not fact sources.',
             'Choose the actual supporting evidence first. evidence_refs must list exactly the union of the candidate evidence_groups, without duplicate refs. Do not copy irrelevant intent groups just to fill evidence.',
             'Keep actor ownership, event order, and intention versus completion distinct.',
             'Every candidate must implement the approved slot and target relation; free exploration may use alternative supporting evidence.',
             'For SUCCESSOR_FRONTIER and FRESH_R03 each candidate must use at least one required R03 anchor with a real contribution to its answer.',
            'For COVERAGE_GAP use the intent evidence groups to test the under-covered relation, not an unrelated old fact.',
            'When question_type is multi_select, require at least two independently stated facts or stages and make every gold option necessary; use plausible but contradicted distractors.',
            'When question_type is ordering, require at least four chronologically distinguishable events or state changes spanning multiple scenes; do not order items using generic plausibility.',
            'Before emitting a multi_select candidate, verify every option against the cited turns, including actor, time, and completed-versus-planned status. Every correct option needs support; every incorrect option needs a determinate reason for exclusion. Do not use unrelated personal topics as distractors for a work question.',
            'Before emitting ordering, verify each adjacent pair in the proposed Gold has an explicit temporal relation in the cited dialogue. Retrospective mentions are not onset times; two phrasings of the same action are not two events. Ready to submit is not submitted; sent to a friend is not sent to a boss. Return zero if a unique order cannot be established.',
            'Prefer a state contrast, cross-stage dependency, or explicit constraint interaction over a single surface fact. Difficulty must come from combining cited dialogue evidence, never from ambiguity.',
             'Do not replay or paraphrase historical or already generated questions. Return the original candidates JSON field layout only.',
             'Do not output policy decisions, diagnoses, admission claims, or Solver scores.',
             'APPROVED_INTENT:', compact(scope),
             'HISTORICAL_REPLAY_METADATA:', compact(replay_view(history)),
             'ALREADY_GENERATED_QUESTIONS:', compact(list(previous))]
    return '\n'.join(lines + ['R03 POLICY SCOPE (overrides count wording only):'] + rules)
