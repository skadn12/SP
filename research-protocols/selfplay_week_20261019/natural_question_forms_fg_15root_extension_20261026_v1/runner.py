#!/usr/bin/env python3
"""Frozen F/G question-form pilot; prepare and score on CPU, run once on B1.

The FULL condition reuses the original source renderer, prompt bridge, display
algorithm and root:slot sampling seeds. QUESTION_ONLY is a separate diagnostic.
Generation reads public artifacts only. Old responses are never regenerated.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
OLD = WEEK / 'feedback_closed_loop_20261024_v1'
OUT = HERE / 'execution'
SCORER_PATH = HERE.parents[2] / 'offline-deliveries/h4_multiselect_partial_admission_v1/scoring.py'
ROOTS = tuple('cc-train-episode-' + x for x in (
    '109164', '136202', '156136', '179186', '194901', '238359', '246750',
    '24719', '25652', '45849', '55788', '68549', '75241', '761', '77882'))
ARMS = ('F', 'G')
CONDITIONS = ('FULL', 'QUESTION_ONLY')
SLOTS = {
    'MC2': ('MC_QUANTITATIVE_RELATION', 2),
    'MC3': ('MC_CONSTRAINT_CONJUNCTION', 3),
    'MC4': ('MC_EVIDENCE_SELECTION', 4),
    'MC5': ('MC_CAUSAL_SUPPORT', 5),
    'MC6': ('MC_CONSTRAINT_CONJUNCTION', 6),
}
MAX_TOKENS, MODEL_LEN, BATCH = 6144, 16384, 8
SAMPLING = {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0}
PRIVATE_NAMES = ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'SOURCE_SELECTION_FINAL.json', 'CONTENT_FREEZE.json')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load('question_form_original_closed_loop', OLD / 'solver_runner.py')
engine = legacy.v2
orig = engine.orig
read, bind, check, need = legacy.read, legacy.bind, legacy.check, legacy.need
PUBLIC_KEYS = legacy.PUBLIC_KEYS
scorer = load('question_form_user_original_J', SCORER_PATH)


def save(path, value):
    legacy.save(path, value)


def sampling_seed(root, slot):
    return int.from_bytes(hashlib.sha256((root + ':' + slot).encode()).digest()[:4], 'big') % 2147483647


def validate_protocol(protocol):
    expected_slots = {
        slot: {'operation': operation, 'gold_count': gold_count}
        for slot, (operation, gold_count) in SLOTS.items()
    }
    expected_slots['MC6']['role'] = 'ALL_TRUE_CONTROL'
    need(protocol['experiment_id'] == HERE.name
         and protocol['roots'] == [r.split('-')[-1] for r in ROOTS]
         and protocol['arms'] == list(ARMS)
         and protocol['slots'] == expected_slots,
         'PROTOCOL_DESIGN_DRIFT')
    n_questions = len(ROOTS) * len(ARMS) * len(SLOTS)
    need(protocol['gold_by_slot'] == {slot: value[1] for slot, value in SLOTS.items()}
         and protocol['planned_questions'] == n_questions
         and protocol['planned_full_requests'] == n_questions * 2
         and protocol['planned_question_only_requests'] == n_questions * 2
         and protocol['planned_requests_total'] == n_questions * 2 * len(CONDITIONS)
         and protocol['display_variants'] == 2, 'PROTOCOL_COUNT_DRIFT')
    need(Path(protocol['model']).resolve() == orig.MODEL.resolve()
         and Path(protocol['tokenizer']).resolve() == orig.TOKENIZER.resolve(),
         'PROTOCOL_MODEL_DRIFT')
    need(protocol['thinking'] == 'OFF' and protocol['max_tokens'] == MAX_TOKENS
         and protocol['max_model_len'] == MODEL_LEN and protocol['batch_size'] == BATCH,
         'PROTOCOL_BUDGET_DRIFT')
    need(all(protocol['sampling'][key] == value for key, value in SAMPLING.items()),
         'PROTOCOL_SAMPLING_DRIFT')
    need(protocol['prefix_caching'] is False and protocol['automatic_retries'] == 0
         and protocol['training'] is False and protocol['source_extension'] is False,
         'PROTOCOL_EXECUTION_POLICY_DRIFT')



def messages_from_prompt(prompt):
    pattern = r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'
    match = re.fullmatch(pattern, prompt, re.S)
    need(match is not None, 'PROMPT_TEMPLATE_DRIFT')
    return [{'role': 'system', 'content': match[1]}, {'role': 'user', 'content': match[2]}]


def full_prompt(tokenizer, row, source):
    # This is the same neutral instruction bridge used by the original F/G run.
    original = orig.diagnostic.make_prompt(tokenizer, row, row['display'], source)
    messages = messages_from_prompt(original)
    before = 'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.'
    after = 'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.'
    need(messages[1]['content'].count(before) == 1, 'NEUTRAL_INSTRUCTION_DRIFT')
    messages[0]['content'] = messages[0]['content'].replace('Give concise, verifiable explanations', 'In the final response, give concise, verifiable explanations')
    messages[1]['content'] = messages[1]['content'].replace(before, after)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def score_one(row, presentation, raw, schema):
    valid, parsed, answer, final, errors = False, None, [], None, []
    try:
        need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
        need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
        thinking, final, _ = engine.split_output(raw['raw_output'])
        need(not thinking, 'UNEXPECTED_NONEMPTY_THINKING')
        engine.legal_answer(final, row)
        parsed, answer = orig.base.validate_output(final, row, schema)
        valid = True
    except Exception as error:
        errors.append(str(error))
    canonical_options = {presentation['display_to_canonical'][label]: text for label, text in row['display'].items()}
    question = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'], 'options': canonical_options, 'gold': presentation['canonical_gold']}
    request = {'question_type': 'multi_select', 'candidate_id': row['candidate_id'], 'request_id': row['request_id'],
               'variant': {'display_to_canonical': presentation['display_to_canonical'], 'options': row['display'], 'displayed_gold': presentation['display_gold']}}
    boxed = '<box>' + ','.join(answer) + '</box>' if valid else None
    official = scorer.score_response(question, request, {'request_id': row['request_id'], 'raw_output': boxed})
    strict = orig.base.score_record(row, presentation, answer, valid)
    if valid:
        need(official['scored'] and official['canonical_prediction'] == strict['prediction'] and official['exact_match'] == strict['exact'], 'ORIGINAL_J_EXACT_OR_MAPPING_DRIFT')
    return {k: row[k] for k in ('request_id', 'candidate_id', 'root_id', 'arm', 'slot_id', 'condition', 'variant_id', 'prompt_tokens')} | {
        'valid': valid, 'exact': strict['exact'], 'gold_count': len(presentation['canonical_gold']),
        'canonical_prediction': strict['prediction'], 'canonical_gold': presentation['canonical_gold'],
        'TP': official['tp'], 'FP': official['fp'], 'FN': official['fn'],
        'J': official['partial_score'], 'J_status': official['scoring_status'],
        'errors': errors, 'parsed': parsed, 'final_text': final,
        'finish_reason': raw.get('finish_reason'), 'output_tokens': len(raw.get('output_token_ids', [])),
    }


def baseline_records():
    raise RuntimeError('No same-item baseline exists for this 15-root extension.')



def prepare():
    need(not OUT.exists() or not (OUT / 'PREPARATION.json').exists(),
         'NO_PREPARATION_OVERWRITE')
    endpoint, config = orig.endpoint_receipt()
    del endpoint
    old_prep = read(OLD / 'execution/PREPARATION.json')
    need(config == old_prep['config_receipt']
         and orig.versions() == old_prep['runtime_versions'],
         'ORIGINAL_CHECKPOINT_CONFIG_OR_RUNTIME_DRIFT')
    frozen = read(HERE / 'CONTENT_FREEZE.json')
    for binding in frozen['bindings']:
        check(binding)
    for name in ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json',
                 'SOURCE_SELECTION_FINAL.json', 'PROTOCOL_FREEZE.json'):
        need(bind(HERE / name) in frozen['bindings'],
             'UNBOUND_CONTENT_INPUT:' + name)
    validate_protocol(read(HERE / 'PROTOCOL_FREEZE.json'))

    selection = read(HERE / 'SOURCE_SELECTION_FINAL.json')
    sources = {row['root_id']: row for row in selection['roots']}
    need(len(sources) == len(selection['roots']) == len(ROOTS)
         and set(sources) == set(ROOTS), 'FROZEN_15_ROOTS_REQUIRED')
    candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
    reviews = read(HERE / 'CONTENT_REVIEW_AI.json')['reviews']
    expected = {f'NQ15-{arm}-{root.split("-")[-1]}-{slot}'
                for root in ROOTS for arm in ARMS for slot in SLOTS}
    need(isinstance(candidates, list) and len(candidates) == len(expected)
         and {c['candidate_id'] for c in candidates} == expected,
         'EXACT_150_SLOT_COVERAGE_REQUIRED')
    review_by_id = {r['candidate_id']: r for r in reviews}
    question_ids = {c['candidate_id'] for c in candidates if 'question' in c}
    need(len(reviews) == len(review_by_id) == len(question_ids)
         and set(review_by_id) == question_ids, 'EXACT_AUTHORED_QUESTION_REVIEW_COVERAGE_REQUIRED')
    by_slot, decisions, ledger = {}, {}, []
    for candidate in candidates:
        cid, root, slot, arm = (candidate['candidate_id'], candidate['root_id'],
                                candidate['slot_id'], candidate['arm'])
        need(root in ROOTS and slot in SLOTS and arm in ARMS,
             'UNPLANNED_CANDIDATE:' + cid)
        need(cid == f'NQ15-{arm}-{root.split("-")[-1]}-{slot}',
             'CANDIDATE_ID_DRIFT:' + cid)
        key = (root, slot, arm)
        need(key not in by_slot, 'DUPLICATE_SLOT:' + cid)
        by_slot[key] = candidate
        if 'question' not in candidate:
            failure = candidate.get('failure_record')
            need(candidate.get('source_status') == 'SLOT_FAILURE'
                 and candidate.get('execution_status') == 'HOLD'
                 and isinstance(failure, dict)
                 and failure.get('candidate_id') == cid
                 and failure.get('root_id') == root
                 and failure.get('slot_id') == slot
                 and isinstance(failure.get('reason'), str)
                 and failure['reason'].strip(),
                 'INVALID_SLOT_FAILURE_RECORD:' + cid)
            decisions[key] = 'HOLD'
            ledger.append({'candidate_id': cid, 'root_id': root, 'arm': arm,
                           'slot_id': slot, 'review_decision': 'HOLD',
                           'execution_admitted': False,
                           'execution_hold_reason': 'RETAINED_SLOT_FAILURE_NO_REPLACEMENT'})
            continue
        review = review_by_id[cid]
        decision = str(review['decision']).upper()
        need(decision in ('PASS', 'HOLD', 'FAIL'), 'UNKNOWN_REVIEW_DECISION:' + cid)
        need(review['candidate_sha256'] == orig.fingerprint(candidate),
             'REVIEW_HASH_DRIFT:' + cid)
        need(candidate.get('author_id') and review.get('author_id') == candidate['author_id']
             and review.get('reviewer_id') and review['reviewer_id'] != candidate['author_id'],
             'INDEPENDENT_REVIEW_REQUIRED:' + cid)
        selected = {k: sources[root][k] for k in ('path', 'sha256', 'bytes')}
        need(candidate['source_binding'] == selected, 'SOURCE_SELECTION_DRIFT:' + cid)
        check(selected)
        gate = orig.production_gate.validate_review(
            candidate, review, HERE / 'SOURCE_SELECTION_FINAL.json')
        need(gate['review_record_complete'],
             'CONTENT_REVIEW_RECORD_INCOMPLETE:' + cid + ':'
             + json.dumps(gate['review_errors'], ensure_ascii=False))
        if decision == 'PASS':
            need(gate['mechanical_pass'] and gate['admitted'],
                 'CONTENT_GATE:' + cid + ':' + json.dumps(gate, ensure_ascii=False))
            orig.gate._validate(candidate)
            need(candidate['question_type'] == 'multi_select'
                 and candidate['task_operation'] == SLOTS[slot][0],
                 'SLOT_OPERATION_DRIFT:' + cid)
            need(list(candidate['options']) == list('ABCDEF')
                 and len(candidate['gold']) == SLOTS[slot][1],
                 'GOLD_QUOTA_OR_OPTION_DRIFT:' + cid)
            independent = review.get('independent_gold')
            need(isinstance(independent, list)
                 and all(isinstance(x, str) for x in independent)
                 and len(independent) == len(set(independent))
                 and set(independent) == set(candidate['gold']),
                 'INDEPENDENT_GOLD_MISMATCH:' + cid)
        decisions[key] = decision
        ledger.append({'candidate_id': cid, 'root_id': root, 'arm': arm,
                       'slot_id': slot, 'review_decision': decision,
                       'execution_admitted': False})
    matched_slots = {(root, slot) for root in ROOTS for slot in SLOTS
                     if all(decisions.get((root, slot, arm)) == 'PASS' for arm in ARMS)}
    need(matched_slots, 'NO_COMPLETE_FG_PASS_SLOT_PAIRS')
    admitted = [candidate for key, candidate in by_slot.items()
                if (key[0], key[1]) in matched_slots]
    admitted_ids = {candidate['candidate_id'] for candidate in admitted}
    for row in ledger:
        row['execution_admitted'] = row['candidate_id'] in admitted_ids
        if row['review_decision'] == 'PASS' and not row['execution_admitted']:
            row['execution_hold_reason'] = 'OTHER_ARM_NOT_PASS_OR_MISSING_NO_REPLACEMENT'

    from transformers import AutoTokenizer
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    tokenizer = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    reasoner = Qwen3ReasoningParser(tokenizer=tokenizer)
    api = engine.cpu_api_check()
    save(OUT / 'CPU_API_CHECK.json', api)
    rows, tokens, private_rows, bindings, prompt_audit = [], {}, [], [], []
    for candidate in sorted(admitted, key=lambda c: (c['root_id'], c['slot_id'], c['arm'])):
        source = read(candidate['source_binding']['path'])
        need(source['root_id'] == candidate['root_id'], 'SOURCE_ROOT_DRIFT')
        refs = [turn['turn_ref'] for turn in source['turns']]
        need(refs and len(refs) == len(set(refs)), 'SOURCE_REFS_INVALID')
        seed = sampling_seed(candidate['root_id'], candidate['slot_id'])
        for index, public_view, presentation in legacy.display_variants(candidate):
            variant = f'perm_{index:02}'
            for condition in CONDITIONS:
                rid = f'{candidate["candidate_id"]}__{variant}__{condition}__THINK_OFF__NQ15'
                row = {k: v for k, v in public_view.items() if k != 'options'}
                row.update(request_id=rid, base_request_id=rid, cohort='new',
                           condition=condition, mode='THINK_OFF',
                           enable_thinking=False, arm=candidate['arm'],
                           slot_id=candidate['slot_id'], variant_id=variant,
                           display=public_view['options'], labels=list(public_view['options']),
                           sampling_seed=seed)
                prompt = full_prompt(tokenizer, row, source) if condition == 'FULL' \
                    else orig.base.question_only_prompt(tokenizer, row)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                need(len(ids) + MAX_TOKENS <= MODEL_LEN,
                     'CONTEXT_OVERFLOW_NO_TRUNCATION:' + rid)
                need(reasoner.is_reasoning_end(ids) is True,
                     'THINK_OFF_REASONING_GATE_DRIFT:' + rid)
                messages = messages_from_prompt(prompt)
                schema = orig.diagnostic.repair.schema(row['labels'], refs, 'multi_select') \
                    if condition == 'FULL' else orig.base.question_only_schema(
                        row['labels'], 'multi_select')
                prompt_path = OUT / 'prompts' / (rid + '.txt')
                schema_path = OUT / 'schemas' / (rid + '.json')
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                with prompt_path.open('x', encoding='utf-8') as handle:
                    handle.write(prompt)
                save(schema_path, schema)
                row.update(prompt_path=str(prompt_path), schema_path=str(schema_path),
                           prompt_tokens=len(ids),
                           messages_sha256=hashlib.sha256(json.dumps(
                               messages, ensure_ascii=False, sort_keys=True
                           ).encode()).hexdigest())
                need(set(row) == PUBLIC_KEYS, 'PUBLIC_FIELD_ALLOWLIST_DRIFT')
                rows.append(row)
                tokens[rid] = ids
                private_rows.append({
                    'request_id': rid, 'candidate_id': candidate['candidate_id'],
                    'arm': candidate['arm'], 'slot_id': candidate['slot_id'],
                    'condition': condition, 'presentation': presentation,
                    'legacy_request_id': None,
                })
                bindings.extend((bind(prompt_path), bind(schema_path),
                                 candidate['source_binding']))
                prompt_audit.append({
                    'request_id': rid, 'candidate_id': candidate['candidate_id'],
                    'root_id': candidate['root_id'], 'arm': candidate['arm'],
                    'slot_id': candidate['slot_id'], 'condition': condition,
                    'variant_id': variant, 'same_condition_display_mapping': True,
                    'cross_arm_display_mapping_required': False,
                    'same_sampling_seed': True, 'full_uses_original_renderer': condition == 'FULL',
                    'source_supplied': condition == 'FULL', 'prompt_tokens': len(ids),
                    'reserved_output_tokens': MAX_TOKENS, 'context_limit': MODEL_LEN,
                    'no_truncation': True,
                })
    need(len(rows) == len(admitted) * 4 <= 600
         and len({row['request_id'] for row in rows}) == len(rows), 'REQUEST_COUNT')
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    private_path = OUT / 'PRIVATE_SCORING.json'
    save(private_path, {
        'requests': private_rows, 'admission_ledger': ledger,
        'content_bindings': [bind(HERE / name) for name in PRIVATE_NAMES],
        'matched_slots': [{'root_id': root, 'slot_id': slot}
                          for root, slot in sorted(matched_slots)],
        'historical_same_item_baseline': 'none for extension roots',
    })
    private_path.chmod(0o600)
    save(OUT / 'PRE_SOLVER_AUDIT.json', {
        'requests': prompt_audit, 'planned_semantic_questions': 150,
        'admitted_semantic_questions': len(admitted), 'planned_solver_requests': 600,
        'actual_solver_requests': len(rows), 'complete_fg_slot_pairs': len(matched_slots),
        'selection_before_new_solver': True, 'replacement_sampling': False,
        'max_prompt_tokens': max((row['prompt_tokens'] for row in rows), default=0),
        'max_prompt_plus_output_reserve': max(
            (row['prompt_tokens'] + MAX_TOKENS for row in rows), default=0),
        'all_prompts_fit_context': all(
            row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN for row in rows),
        'gpu_model_started': False,
    })
    required = [Path(__file__), OLD / 'solver_runner.py',
                WEEK / 'thinking_switch_20261024_structured_v2/run.py',
                SCORER_PATH, HERE / 'PROTOCOL_FREEZE.json',
                OUT / 'CPU_API_CHECK.json', OUT / 'PUBLIC_REQUESTS.json',
                OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'PRE_SOLVER_AUDIT.json',
                *orig.code_paths()]
    bindings.extend(bind(path) for path in required)
    bindings.extend(api['implementation_bindings'])
    save(OUT / 'PREPARATION.json', {
        'experiment_id': HERE.name, 'requests': len(rows), 'planned_requests': 600,
        'fixed_order': [row['request_id'] for row in rows],
        'bindings': list({item['path']: item for item in bindings}.values()),
        'private_scoring_binding': bind(private_path),
        'content_freeze_binding': bind(HERE / 'CONTENT_FREEZE.json'),
        'config_receipt': config, 'runtime_versions': orig.versions(),
        'structured_outputs_config': engine.STRUCTURED_CONFIG,
        'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH,
        'sampling': SAMPLING, 'prefix_cache': False, 'automatic_retry': False,
        'source_truncation': False, 'conditions': list(CONDITIONS),
        'planned_semantic_questions': 150, 'admitted_semantic_questions': len(admitted),
        'matched_slot_pairs': len(matched_slots),
        'max_prompt_tokens': max((row['prompt_tokens'] for row in rows), default=0),
        'generation_reads_gold': False,
    })
    print(json.dumps({'prepared': len(rows), 'admitted_questions': len(admitted),
                      'matched_slot_pairs': len(matched_slots),
                      'max_prompt_tokens': max((row['prompt_tokens'] for row in rows), default=0),
                      'gpu_model_started': False}, ensure_ascii=False), flush=True)



def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str((HERE / name).resolve()) for name in PRIVATE_NAMES}
    forbidden.add(str((OUT / 'PRIVATE_SCORING.json').resolve()))
    for binding in prep['bindings']:
        need(str(Path(binding['path']).resolve()) not in forbidden,
             'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(binding)
    validate_protocol(read(HERE / 'PROTOCOL_FREEZE.json'))
    rows, tokens = read(OUT / 'PUBLIC_REQUESTS.json')['requests'], read(OUT / 'PROMPT_TOKEN_IDS.json')
    ids = [row['request_id'] for row in rows]
    need(prep['experiment_id'] == HERE.name and ids == prep['fixed_order'],
         'PUBLIC_ORDER_OR_EXPERIMENT_DRIFT')
    need(0 < len(rows) == prep['requests'] == len(set(ids)) <= 600
         and set(tokens) == set(ids), 'PUBLIC_REQUEST_SET_DRIFT')
    need(prep['admitted_semantic_questions'] * 4 == len(rows)
         and prep['matched_slot_pairs'] * 8 == len(rows), 'PAIRED_DENOMINATOR_DRIFT')
    need(prep['max_tokens'] == MAX_TOKENS and prep['max_model_len'] == MODEL_LEN
         and prep['batch_size'] == BATCH and prep['sampling'] == SAMPLING,
         'PUBLIC_ENGINE_BUDGET_DRIFT')
    need(prep['structured_outputs_config'] == engine.STRUCTURED_CONFIG
         and prep['conditions'] == list(CONDITIONS), 'PUBLIC_ENGINE_OR_CONDITION_DRIFT')
    need(prep['prefix_cache'] is False and prep['automatic_retry'] is False
         and prep['source_truncation'] is False and prep['generation_reads_gold'] is False,
         'PUBLIC_EXECUTION_POLICY_DRIFT')
    grouped = collections.defaultdict(list)
    required = []
    for row in rows:
        need(set(row) == PUBLIC_KEYS and row['root_id'] in ROOTS
             and row['arm'] in ARMS and row['slot_id'] in SLOTS
             and row['condition'] in CONDITIONS, 'PUBLIC_FIELDS_OR_DESIGN_DRIFT')
        need(row['question_type'] == 'multi_select'
             and row['task_operation'] == SLOTS[row['slot_id']][0]
             and row['cohort'] == 'new', 'PUBLIC_OPERATION_DRIFT')
        need(row['mode'] == 'THINK_OFF' and row['enable_thinking'] is False
             and row['base_request_id'] == row['request_id'], 'PUBLIC_MODE_DRIFT')
        need(row['labels'] == list(row['display']) == list('ABCDEF')
             and row['sampling_seed'] == sampling_seed(row['root_id'], row['slot_id']),
             'PUBLIC_LABEL_OR_SEED_DRIFT')
        need(row['prompt_tokens'] == len(tokens[row['request_id']])
             and row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN,
             'PUBLIC_TOKEN_BUDGET_DRIFT')
        grouped[(row['root_id'], row['slot_id'], row['arm'], row['candidate_id'])].append(row)
        required.extend((Path(row['prompt_path']), Path(row['schema_path']),
                         Path(row['source_binding']['path'])))
    for group in grouped.values():
        need(len(group) == 4
             and {(r['condition'], r['variant_id']) for r in group}
             == {(condition, variant) for condition in CONDITIONS
                 for variant in ('perm_00', 'perm_01')},
             'PUBLIC_CONDITION_DISPLAY_PAIR_DRIFT')
        for variant in ('perm_00', 'perm_01'):
            pair = [row for row in group if row['variant_id'] == variant]
            need(pair[0]['display'] == pair[1]['display']
                 and pair[0]['sampling_seed'] == pair[1]['sampling_seed'],
                 'CONDITION_PAIR_DRIFT')
    slot_pairs = {(row['root_id'], row['slot_id']) for row in rows}
    need(all({row['arm'] for row in rows
              if (row['root_id'], row['slot_id']) == pair} == set(ARMS)
             for pair in slot_pairs), 'INCOMPLETE_FG_SLOT_PAIR')
    required.extend((Path(__file__), OLD / 'solver_runner.py',
                     WEEK / 'thinking_switch_20261024_structured_v2/run.py',
                     SCORER_PATH, HERE / 'PROTOCOL_FREEZE.json',
                     OUT / 'CPU_API_CHECK.json', OUT / 'PUBLIC_REQUESTS.json',
                     OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'PRE_SOLVER_AUDIT.json',
                     *orig.code_paths()))
    api = read(OUT / 'CPU_API_CHECK.json')
    need(api['gpu_model_started'] is False
         and api['structured_outputs_config'] == engine.STRUCTURED_CONFIG
         and api['runtime_versions'] == prep['runtime_versions'],
         'CPU_API_RECEIPT_DRIFT')
    required.extend(Path(item['path']) for item in api['implementation_bindings'])
    existing = [str(Path(item['path']).resolve()) for item in prep['bindings']]
    need(len(existing) == len(set(existing))
         and all(bind(path) in prep['bindings'] for path in required),
         'UNBOUND_PUBLIC_INPUT')
    return prep, rows, tokens



def check_raw(frozen, rows, tokens):
    """Original closed-loop raw receipt checks, with the declared 600-row cap."""
    ids = [r['request_id'] for r in rows]
    need(0 < len(ids) == len(set(ids)) <= 600 and frozen['request_ids'] == ids and frozen['requests'] == len(ids), 'RAW_SET_DRIFT')
    check(frozen['started_binding'])
    need(frozen['started_binding'] == bind(OUT / 'STARTED.json'), 'STARTED_PATH_DRIFT')
    started = read(OUT / 'STARTED.json')
    need(started['experiment_id'] == HERE.name and started['request_ids'] == ids and started['requests'] == len(ids), 'STARTED_REQUEST_SET_DRIFT')
    need(started['preparation_binding'] == bind(OUT / 'PREPARATION.json') and started['structured_outputs_config'] == engine.STRUCTURED_CONFIG and started['gold_read'] is False, 'STARTED_CONFIG_DRIFT')
    submitted = frozen['submitted_ids']
    need(len(submitted) == len(set(submitted)) and submitted == [rid for rid in ids if rid in set(submitted)], 'SUBMISSION_ID_DRIFT')
    for directory, bindings, expected in (
        ('raw', frozen['bindings'], [OUT / 'raw' / (rid + '.json') for rid in ids]),
        ('submitted', frozen['submission_bindings'], [OUT / 'submitted' / (rid + '.json') for rid in submitted])):
        legacy.paths_equal(bindings, expected, directory.upper() + '_BINDING_SET_DRIFT')
        need({str(p.resolve()) for p in (OUT / directory).glob('*.json')} == {str(p.resolve()) for p in expected}, directory.upper() + '_DIRECTORY_SET_DRIFT')
        for binding in bindings:
            check(binding)
    receipt = frozen['model_receipt_binding']
    if receipt is not None:
        check(receipt)
        need(receipt == bind(OUT / 'MODEL_LOAD_RECEIPT.json'), 'MODEL_RECEIPT_PATH_DRIFT')
        engine.HERE, engine.OUT = HERE, OUT
        engine.checked_model_receipt(read(receipt['path']))
    else:
        need(not submitted and frozen['returned'] == 0, 'MISSING_MODEL_RECEIPT_AFTER_SUBMISSION')
    legacy.paths_equal(frozen['batch_bindings'], list((OUT / 'batches').glob('*.json')), 'BATCH_DIRECTORY_SET_DRIFT')
    completed_batches, completed_ids = [], []
    for binding in frozen['batch_bindings']:
        check(binding)
        batch = read(binding['path'])
        index = batch['batch_index']
        need(type(index) is int and 0 <= index < (len(ids) + BATCH - 1) // BATCH, 'BATCH_INDEX_DRIFT')
        need(Path(binding['path']).resolve() == (OUT / 'batches' / (str(index).zfill(3) + '.json')).resolve(), 'BATCH_PATH_DRIFT')
        need(batch['request_ids'] == ids[index * BATCH:(index + 1) * BATCH], 'BATCH_REQUEST_SET_DRIFT')
        completed_batches.append(index)
        completed_ids.extend(batch['request_ids'])
    need(completed_batches == list(range(len(completed_batches))) and len(completed_ids) == len(set(completed_ids)) and set(completed_ids) <= set(submitted), 'BATCH_ORDER_OR_SUBMISSION_DRIFT')
    returned = 0
    for row in rows:
        rid = row['request_id']
        raw = read(OUT / 'raw' / (rid + '.json'))
        need(raw['request_id'] == rid and raw['mode'] == row['mode'] and raw['base_request_id'] == row['base_request_id'], 'RAW_ID_OR_MODE_DRIFT')
        need(raw['status'] in ('RETURNED', 'BACKEND_ERROR', 'NOT_SUBMITTED') and (raw['status'] != 'NOT_SUBMITTED') == (rid in set(submitted)), 'RAW_STATUS_DRIFT')
        if rid in set(submitted):
            submission = read(OUT / 'submitted' / (rid + '.json'))
            need(submission['request_id'] == rid and submission['experiment_id'] == HERE.name and submission['model_load_receipt_binding'] == receipt, 'SUBMISSION_RECEIPT_DRIFT')
        if raw['status'] == 'RETURNED':
            returned += 1
            need(raw['model_load_receipt_binding'] == receipt, 'RAW_MODEL_RECEIPT_DRIFT')
            need(raw.get('capture_errors') == orig.capture.response_errors(raw, Path(row['prompt_path']).read_text(), tokens[rid]), 'CAPTURE_CHECK_DRIFT')
    need(returned == frozen['returned'], 'RETURNED_COUNT_DRIFT')
    if frozen['failure'] is None:
        need(returned == len(rows) and submitted == ids and completed_batches == list(range((len(ids) + BATCH - 1) // BATCH)), 'INCOMPLETE_SUCCESS')


def run():
    # Keep model construction, sampling, GPU flock, capture, shutdown and no-retry
    # behavior byte-for-byte in the frozen structured-v2 engine.
    engine.HERE, engine.OUT = HERE, OUT
    engine.public, engine.checked_raw_manifest = public, check_raw
    engine.run()


def mean(values):
    return statistics.mean(values) if values else None


def summarize(records):
    return {'requests': len(records), 'semantic_questions': len({r['candidate_id'] for r in records}),
            'valid': sum(r['valid'] for r in records), 'invalid': sum(not r['valid'] for r in records),
            'exact_correct': sum(r['exact'] for r in records), 'exact_accuracy': mean([int(r['exact']) for r in records]),
            'J_score': mean([r['J'] * 100 for r in records if r['J'] is not None]),
            'J_scored_requests': sum(r['J'] is not None for r in records),
            'mean_prompt_tokens': mean([r['prompt_tokens'] for r in records]), 'mean_output_tokens': mean([r['output_tokens'] for r in records]),
            'length_finishes': sum(r['finish_reason'] == 'length' for r in records)}


def group_records(records, group):
    return [r for r in records if group == 'ALL' or (group == 'MC2_5_PRIMARY' and r['gold_count'] < 6) or (group == 'MC6_CONTROL' and r['gold_count'] == 6)]


def comparison(old, new, **labels):
    old_summary, new_summary = summarize(old), summarize(new)
    return labels | {'old': old_summary, 'new': new_summary,
                     'new_minus_old': {key: new_summary[key] - old_summary[key] if new_summary[key] is not None and old_summary[key] is not None else None for key in ('exact_accuracy', 'J_score')}}


def f_g_comparison(records, **labels):
    """F minus G within the same frozen root/slot cross, retaining denominators."""
    by_arm = {arm: [r for r in records if r['arm'] == arm] for arm in ARMS}
    keys = {arm: {(r['root_id'], r['slot_id'], r['variant_id']) for r in by_arm[arm]} for arm in ARMS}
    need(keys['F'] == keys['G'], 'FG_COMPARISON_REQUIRES_MATCHED_ROOT_SLOT_DISPLAYS')
    summaries = {arm: summarize(by_arm[arm]) for arm in ARMS}
    by_key = {arm: {(r['root_id'], r['slot_id'], r['variant_id']): r for r in by_arm[arm]} for arm in ARMS}
    paired_j = [(by_key['F'][key]['J'] - by_key['G'][key]['J']) * 100 for key in keys['F']
                if by_key['F'][key]['J'] is not None and by_key['G'][key]['J'] is not None]
    return labels | summaries | {
        'matched_root_slot_pairs': len({key[:2] for key in keys['F']}),
        'F_minus_G': {metric: summaries['F'][metric] - summaries['G'][metric]
                      if summaries['F'][metric] is not None and summaries['G'][metric] is not None else None
                      for metric in ('exact_accuracy', 'J_score')},
        'both_valid_display_pairs_J_score_difference': mean(paired_j),
        'both_valid_display_pairs_J_denominator': len(paired_j),
    }


def analyze():
    prep, rows, tokens = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    check_raw(freeze, rows, tokens)
    check(prep['content_freeze_binding'])
    for binding in read(HERE / 'CONTENT_FREEZE.json')['bindings']:
        check(binding)
    check(prep['private_scoring_binding'])
    private_doc = read(OUT / 'PRIVATE_SCORING.json')
    for binding in private_doc['content_bindings']:
        check(binding)
    private = {item['request_id']: item for item in private_doc['requests']}
    need(len(private) == len(private_doc['requests']) == len(rows)
         and list(private) == prep['fixed_order'], 'PRIVATE_REQUEST_SET_DRIFT')
    candidates = {item['candidate_id']: item
                  for item in read(HERE / 'CANDIDATES_PRIVATE.json')}
    records = []
    for row in rows:
        item = private[row['request_id']]
        need(all(item[key] == row[key]
                 for key in ('candidate_id', 'arm', 'slot_id', 'condition')),
             'PRIVATE_PUBLIC_ID_DRIFT')
        index = int(row['variant_id'][-2:])
        expected_public, expected_presentation = list(
            legacy.display_variants(candidates[row['candidate_id']])
        )[index][1:]
        need(item['presentation'] == expected_presentation
             and expected_public['options'] == row['display'],
             'PRIVATE_PRESENTATION_DRIFT')
        raw_path = OUT / 'raw' / (row['request_id'] + '.json')
        record = score_one(row, item['presentation'], read(raw_path),
                           read(row['schema_path']))
        record.update(phase='extension', raw_binding=bind(raw_path))
        records.append(record)

    summaries, strata, fg_comparisons = [], [], []
    for condition in CONDITIONS:
        selected = [row for row in records if row['condition'] == condition]
        for group in ('ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL'):
            subset = group_records(selected, group)
            summaries.append({'condition': condition, 'group': group, **summarize(subset)})
            for dimension in ('arm', 'root_id', 'slot_id'):
                for value in sorted({row[dimension] for row in subset}):
                    strata.append({'condition': condition, 'group': group,
                                   'dimension': dimension, 'value': value,
                                   **summarize([row for row in subset
                                                if row[dimension] == value])})
            fg_comparisons.append(f_g_comparison(
                subset, condition=condition, group=group,
                dimension='ALL', value='ALL'))
            for dimension in ('root_id', 'slot_id'):
                for value in sorted({row[dimension] for row in subset}):
                    fg_comparisons.append(f_g_comparison(
                        [row for row in subset if row[dimension] == value],
                        condition=condition, group=group,
                        dimension=dimension, value=value))
            if condition == 'FULL' and group == 'MC2_5_PRIMARY':
                for root, slot in sorted({(row['root_id'], row['slot_id'])
                                          for row in subset}):
                    fg_comparisons.append(f_g_comparison(
                        [row for row in subset
                         if (row['root_id'], row['slot_id']) == (root, slot)],
                        condition=condition, group=group, dimension='root_slot',
                        root_id=root, slot_id=slot))

    main = [row for row in records
            if row['condition'] == 'FULL' and row['gold_count'] < 6]
    root_effects = []
    for root in ROOTS:
        arm_summaries = {}
        for arm in ARMS:
            arm_summaries[arm] = summarize(
                [row for row in main if row['root_id'] == root and row['arm'] == arm])
        root_effects.append({
            'root_id': root, 'F': arm_summaries['F'], 'G': arm_summaries['G'],
            'F_minus_G': {
                metric: arm_summaries['F'][metric] - arm_summaries['G'][metric]
                if arm_summaries['F'][metric] is not None
                and arm_summaries['G'][metric] is not None else None
                for metric in ('exact_accuracy', 'J_score')
            },
        })
    pairs = collections.defaultdict(dict)
    for row in records:
        pairs[(row['candidate_id'], row['variant_id'])][row['condition']] = row
    diagnostics = []
    for (cid, variant), pair in pairs.items():
        full, only = pair['FULL'], pair['QUESTION_ONLY']
        diagnostics.append({
            'candidate_id': cid, 'variant_id': variant,
            'arm': full['arm'], 'root_id': full['root_id'], 'slot_id': full['slot_id'],
            'FULL_exact': full['exact'], 'QUESTION_ONLY_exact': only['exact'],
            'FULL_J': full['J'], 'QUESTION_ONLY_J': only['J'],
            'both_valid': full['valid'] and only['valid'],
            'same_canonical_prediction': full['valid'] and only['valid']
            and full['canonical_prediction'] == only['canonical_prediction'],
        })
    consistency = []
    for condition in CONDITIONS:
        selected = [row for row in records if row['condition'] == condition]
        for cid in sorted({row['candidate_id'] for row in selected}):
            subset = [row for row in selected if row['candidate_id'] == cid]
            consistency.append({
                'condition': condition, 'candidate_id': cid,
                'both_valid': all(row['valid'] for row in subset),
                'both_exact': all(row['exact'] for row in subset),
                'same_canonical_prediction': all(row['valid'] for row in subset)
                and len({tuple(row['canonical_prediction']) for row in subset}) == 1,
            })
    save(OUT / 'SCORES.json', {
        'records': records, 'raw_freeze_binding': bind(OUT / 'RAW_RESPONSE_FREEZE.json'),
        'private_scoring_binding': prep['private_scoring_binding'],
        'original_J_scorer_binding': bind(SCORER_PATH),
    })
    save(OUT / 'ANALYSIS_SUMMARY.json', {
        'cohort': '15 new roots; prior four-root pilot remains separate',
        'summaries': summaries, 'strata': strata,
        'F_minus_G_by_condition_root_slot': fg_comparisons,
        'primary_FULL_MC2_MC5_F_minus_G': [
            row for row in fg_comparisons
            if row['condition'] == 'FULL' and row['group'] == 'MC2_5_PRIMARY'],
        'root_level_PRIMARY_FULL_MC2_MC5': root_effects,
        'question_only_diagnostic_pairs': diagnostics,
        'permutation_consistency': consistency,
        'admission_ledger': private_doc['admission_ledger'],
        'denominators': {
            'planned_questions': 150,
            'admitted_questions': len({row['candidate_id'] for row in records}),
            'planned_new_requests': 600, 'frozen_new_requests': len(records),
            'new_submitted_requests': len(freeze['submitted_ids']),
            'new_returned_requests': freeze['returned'],
            'valid_new_responses': sum(row['valid'] for row in records),
            'dependent_displays_per_question': 2,
        },
        'metric_policy': 'Original J=TP/g-FP/(n-g), unrounded and unclipped; MC6 J is not applicable and is separately reported. Strict exact includes every frozen request.',
        'FG_metric_policy': 'F minus G exact uses all matched requests. F minus G J reports valid-response denominators. Display variants are repeated measures; root is the inferential unit.',
        'condition_policy': 'FULL is primary; QUESTION_ONLY is a separate no-dialogue diagnostic, not a pure form control.',
        'baseline_policy': 'No historical same-item baseline is available for these 15 roots. The completed four-root pilot is reported separately; pooled 19-root summaries are secondary and descriptive.',
        'limits': 'Fixed checkpoint; no training or unseen-root generalization claim. Sources and questions have the documented eligibility and review limits.',
        'raw_wall_seconds': freeze['wall_seconds'],
        'total_prompt_tokens': sum(row['prompt_tokens'] for row in records),
        'total_output_tokens': sum(row['output_tokens'] for row in records),
        'failure': freeze['failure'],
    })
    print(json.dumps({'summaries': summaries,
                      'new_model_calls': len(freeze['submitted_ids']),
                      'failure': freeze['failure']}, ensure_ascii=False), flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    globals()[parser.parse_args().action]()
