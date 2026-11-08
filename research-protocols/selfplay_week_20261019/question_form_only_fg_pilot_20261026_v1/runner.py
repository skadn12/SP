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
ROOTS = tuple('cc-train-episode-' + x for x in ('218519', '222937', '225575', '229686'))
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
    need(protocol['experiment_id'] == HERE.name and protocol['roots'] == [r.split('-')[-1] for r in ROOTS]
         and protocol['arms'] == list(ARMS) and protocol['slots'] == list(SLOTS), 'PROTOCOL_DESIGN_DRIFT')
    need(protocol['gold_by_slot'] == {slot: value[1] for slot, value in SLOTS.items()}
         and protocol['planned_questions'] == 40 and protocol['planned_full_requests'] == 80
         and protocol['planned_question_only_requests'] == 80 and protocol['display_variants'] == 2, 'PROTOCOL_COUNT_DRIFT')
    need(Path(protocol['model']).resolve() == orig.MODEL.resolve() and Path(protocol['tokenizer']).resolve() == orig.TOKENIZER.resolve(), 'PROTOCOL_MODEL_DRIFT')
    need(protocol['thinking'] == 'OFF' and protocol['max_tokens'] == MAX_TOKENS
         and protocol['max_model_len'] == MODEL_LEN and protocol['batch_size'] == BATCH, 'PROTOCOL_BUDGET_DRIFT')
    need(all(protocol['sampling'][key] == value for key, value in SAMPLING.items()), 'PROTOCOL_SAMPLING_DRIFT')
    need(protocol['prefix_caching'] is False and protocol['automatic_retries'] == 0
         and protocol['training'] is False and protocol['source_extension'] is False, 'PROTOCOL_EXECUTION_POLICY_DRIFT')


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
    """Validate the old freeze and rescore its 80 MC raw responses with original J."""
    prep, rows, tokens = legacy.public()
    legacy.check_raw(read(OLD / 'execution/RAW_RESPONSE_FREEZE.json'), rows, tokens)
    check(prep['private_scoring_binding'])
    check(prep['content_freeze_binding'])
    for binding in read(OLD / 'CONTENT_FREEZE.json')['bindings']:
        check(binding)
    private_doc = read(OLD / 'execution/PRIVATE_SCORING.json')
    for key in ('candidate_binding', 'review_binding', 'selection_binding'):
        check(private_doc[key])
    private = {r['request_id']: r for r in private_doc['requests']}
    selected = [r for r in rows if r['question_type'] == 'multi_select']
    need(len(selected) == 80 and len({r['candidate_id'] for r in selected}) == 40, 'OLD_40_QUESTION_80_REQUEST_BASELINE_REQUIRED')
    records = []
    for row in selected:
        need(row['root_id'] in ROOTS and row['arm'] in ARMS and row['slot_id'] in SLOTS, 'OLD_BASELINE_DESIGN_DRIFT')
        presentation = private[row['request_id']]['presentation']
        path = OLD / 'execution/raw' / (row['request_id'] + '.json')
        record = score_one(row, presentation, read(path), read(row['schema_path']))
        record.update(phase='old', raw_binding=bind(path))
        records.append(record)
    return records, selected, private, {
        'public_binding': bind(OLD / 'execution/PUBLIC_REQUESTS.json'),
        'private_binding': prep['private_scoring_binding'],
        'raw_freeze_binding': bind(OLD / 'execution/RAW_RESPONSE_FREEZE.json'),
        'preparation_binding': bind(OLD / 'execution/PREPARATION.json'),
        'scorer_binding': bind(SCORER_PATH), 'new_model_calls': 0,
        'requests': len(records), 'valid': sum(r['valid'] for r in records),
    }


def prepare():
    need(not OUT.exists(), 'NO_PREPARATION_OVERWRITE')
    endpoint, config = orig.endpoint_receipt()
    del endpoint
    old_prep = read(OLD / 'execution/PREPARATION.json')
    need(config == old_prep['config_receipt'] and orig.versions() == old_prep['runtime_versions'], 'ORIGINAL_CHECKPOINT_CONFIG_OR_RUNTIME_DRIFT')
    frozen = read(HERE / 'CONTENT_FREEZE.json')
    for binding in frozen['bindings']:
        check(binding)
    for name in ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'SOURCE_SELECTION_FINAL.json', 'PROTOCOL_FREEZE.json'):
        need(bind(HERE / name) in frozen['bindings'], 'UNBOUND_CONTENT_INPUT:' + name)
    validate_protocol(read(HERE / 'PROTOCOL_FREEZE.json'))
    baseline, old_rows, old_private, baseline_audit = baseline_records()
    old_by_key = {(r['candidate_id'], r['variant_id']): r for r in old_rows}
    source_baseline = {r['root_id']: r['source_binding'] for r in old_rows}
    selection = read(HERE / 'SOURCE_SELECTION_FINAL.json')
    sources = {r['root_id']: r for r in selection['roots']}
    need(len(sources) == len(selection['roots']) == 4 and set(sources) == set(ROOTS), 'FOUR_ORIGINAL_ROOTS_REQUIRED')
    need(all(r.get('split', selection.get('split')) == 'train' for r in sources.values()), 'TRAIN_ROOTS_REQUIRED')
    candidates = read(HERE / 'CANDIDATES_PRIVATE.json')
    review_rows = read(HERE / 'CONTENT_REVIEW_AI.json')['reviews']
    need(isinstance(candidates, list) and 0 < len(candidates) <= 40, 'CANDIDATE_COUNT')
    reviews = {r['candidate_id']: r for r in review_rows}
    need(len(reviews) == len(review_rows) == len(candidates) == len({c['candidate_id'] for c in candidates}), 'EXACT_UNIQUE_REVIEW_COVERAGE_REQUIRED')
    need(set(reviews) == {c['candidate_id'] for c in candidates}, 'REVIEW_SET_DRIFT')
    by_slot, decisions, ledger = {}, {}, []
    for candidate in candidates:
        cid = candidate['candidate_id']
        root, slot, arm = candidate['root_id'], candidate['slot_id'], candidate['arm']
        need(root in ROOTS and slot in SLOTS and arm in ARMS, 'UNPLANNED_CANDIDATE:' + cid)
        need(cid == f'CL1-{arm}-{root.split("-")[-1]}-{slot}', 'STABLE_ORIGINAL_CANDIDATE_ID_REQUIRED')
        key = (root, slot, arm)
        need(key not in by_slot, 'DUPLICATE_SLOT:' + cid)
        by_slot[key] = candidate
        review = reviews[cid]
        decision = str(review['decision']).upper()
        need(decision in ('PASS', 'HOLD', 'FAIL'), 'UNKNOWN_REVIEW_DECISION:' + cid)
        need(review['candidate_sha256'] == orig.fingerprint(candidate), 'REVIEW_HASH_DRIFT:' + cid)
        need(candidate.get('author_id') and review.get('author_id') == candidate['author_id'] and review.get('reviewer_id') and review['reviewer_id'] != candidate['author_id'], 'INDEPENDENT_REVIEW_REQUIRED:' + cid)
        selected_binding = {k: sources[root][k] for k in ('path', 'sha256', 'bytes')}
        need(candidate['source_binding'] == selected_binding, 'SOURCE_SELECTION_DRIFT:' + cid)
        need(all(selected_binding[k] == source_baseline[root][k] for k in ('sha256', 'bytes')), 'ORIGINAL_SOURCE_BYTES_REQUIRED:' + cid)
        check(selected_binding)
        if decision == 'PASS':
            orig.gate._validate(candidate)
            need(candidate['question_type'] == 'multi_select' and candidate['task_operation'] == SLOTS[slot][0], 'SLOT_OPERATION_DRIFT:' + cid)
            need(list(candidate['options']) == list('ABCDEF') and len(candidate['gold']) == SLOTS[slot][1], 'SLOT_LABEL_ORDER_OR_GOLD_COUNT_DRIFT:' + cid)
            independent = review.get('independent_gold')
            need(isinstance(independent, list) and all(isinstance(x, str) for x in independent) and len(independent) == len(set(independent)) and set(independent) == set(candidate['gold']), 'INDEPENDENT_GOLD_MISMATCH:' + cid)
        decisions[key] = decision
        ledger.append({'candidate_id': cid, 'root_id': root, 'arm': arm, 'slot_id': slot, 'review_decision': decision, 'execution_admitted': False})
    matched_slots = {(root, slot) for root in ROOTS for slot in SLOTS if all(decisions.get((root, slot, arm)) == 'PASS' for arm in ARMS)}
    need(matched_slots, 'NO_COMPLETE_FG_PASS_SLOT_PAIRS')
    admitted = [c for key, c in by_slot.items() if (key[0], key[1]) in matched_slots]
    admitted_ids = {c['candidate_id'] for c in admitted}
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
        for index, public_view, presentation in legacy.display_variants(candidate):
            variant = f'perm_{index:02}'
            previous = old_by_key[(candidate['candidate_id'], variant)]
            previous_map = old_private[previous['request_id']]['presentation']['display_to_canonical']
            need(presentation['display_to_canonical'] == previous_map, 'ORIGINAL_DISPLAY_MAPPING_DRIFT')
            need(sampling_seed(candidate['root_id'], candidate['slot_id']) == previous['sampling_seed'], 'ORIGINAL_SAMPLING_SEED_DRIFT')
            for condition in CONDITIONS:
                rid = f'{candidate["candidate_id"]}__{variant}__{condition}__THINK_OFF__QF1'
                row = {k: v for k, v in public_view.items() if k != 'options'}
                row.update(request_id=rid, base_request_id=rid, cohort='new', condition=condition, mode='THINK_OFF', enable_thinking=False,
                           arm=candidate['arm'], slot_id=candidate['slot_id'], variant_id=variant,
                           display=public_view['options'], labels=list(public_view['options']), sampling_seed=previous['sampling_seed'])
                prompt = full_prompt(tokenizer, row, source) if condition == 'FULL' else orig.base.question_only_prompt(tokenizer, row)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:' + rid)
                need(reasoner.is_reasoning_end(ids) is True, 'THINK_OFF_REASONING_GATE_DRIFT:' + rid)
                messages = messages_from_prompt(prompt)
                schema = orig.diagnostic.repair.schema(row['labels'], refs, 'multi_select') if condition == 'FULL' else orig.base.question_only_schema(row['labels'], 'multi_select')
                pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
                pp.parent.mkdir(parents=True, exist_ok=True)
                with pp.open('x', encoding='utf-8') as handle:
                    handle.write(prompt)
                save(sp, schema)
                row.update(prompt_path=str(pp), schema_path=str(sp), prompt_tokens=len(ids), messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
                need(set(row) == PUBLIC_KEYS, 'PUBLIC_FIELD_ALLOWLIST_DRIFT')
                rows.append(row)
                tokens[rid] = ids
                private_rows.append({'request_id': rid, 'candidate_id': candidate['candidate_id'], 'arm': candidate['arm'], 'slot_id': candidate['slot_id'], 'condition': condition, 'presentation': presentation, 'legacy_request_id': previous['request_id']})
                bindings.extend((bind(pp), bind(sp), candidate['source_binding']))
                prompt_audit.append({'request_id': rid, 'legacy_request_id': previous['request_id'], 'same_display_mapping': True, 'same_sampling_seed': True,
                                     'source_bytes_identical': True, 'full_uses_original_renderer_and_neutral_bridge': condition == 'FULL', 'source_supplied': condition == 'FULL'})
    need(len(rows) == len(admitted) * 4 <= 160 and len({r['request_id'] for r in rows}) == len(rows), 'REQUEST_COUNT')
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    save(OUT / 'PRIVATE_SCORING.json', {'requests': private_rows, 'admission_ledger': ledger,
         'content_bindings': [bind(HERE / name) for name in PRIVATE_NAMES], 'baseline_audit': baseline_audit,
         'matched_slots': [{'root_id': r, 'slot_id': s} for r, s in sorted(matched_slots)]})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    save(OUT / 'PRE_SOLVER_AUDIT.json', {'requests': prompt_audit, 'planned_semantic_questions': 40, 'admitted_semantic_questions': len(admitted),
         'planned_solver_requests': 160, 'actual_solver_requests': len(rows), 'complete_fg_slot_pairs': len(matched_slots),
         'baseline': baseline_audit, 'selection_before_new_solver': True, 'replacement_sampling': False,
         'question_only_role': 'Separate no-dialogue best-effort diagnostic, not a pure question-form control.'})
    required = [Path(__file__), OLD / 'solver_runner.py', WEEK / 'thinking_switch_20261024_structured_v2/run.py', SCORER_PATH,
                HERE / 'PROTOCOL_FREEZE.json', OUT / 'CPU_API_CHECK.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'PRE_SOLVER_AUDIT.json', *orig.code_paths()]
    bindings.extend(bind(path) for path in required)
    bindings.extend(api['implementation_bindings'])
    save(OUT / 'PREPARATION.json', {'experiment_id': HERE.name, 'requests': len(rows), 'planned_requests': 160,
         'fixed_order': [r['request_id'] for r in rows], 'bindings': list({b['path']: b for b in bindings}.values()),
         'private_scoring_binding': bind(OUT / 'PRIVATE_SCORING.json'), 'content_freeze_binding': bind(HERE / 'CONTENT_FREEZE.json'),
         'config_receipt': config, 'runtime_versions': orig.versions(), 'structured_outputs_config': engine.STRUCTURED_CONFIG,
         'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': BATCH, 'sampling': SAMPLING,
         'prefix_cache': False, 'automatic_retry': False, 'source_truncation': False, 'conditions': list(CONDITIONS),
         'planned_semantic_questions': 40, 'admitted_semantic_questions': len(admitted), 'matched_slot_pairs': len(matched_slots),
         'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'generation_reads_gold': False})
    print(json.dumps({'prepared': len(rows), 'admitted_questions': len(admitted), 'fg_slot_pairs': len(matched_slots),
                      'conditions': dict(collections.Counter(r['condition'] for r in rows)), 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'gpu_model_started': False}), flush=True)


def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str((HERE / name).resolve()) for name in PRIVATE_NAMES}
    forbidden.add(str((OUT / 'PRIVATE_SCORING.json').resolve()))
    for binding in prep['bindings']:
        need(str(Path(binding['path']).resolve()) not in forbidden, 'PRIVATE_INPUT_IN_GENERATION_BINDINGS')
        check(binding)
    validate_protocol(read(HERE / 'PROTOCOL_FREEZE.json'))
    rows, tokens = read(OUT / 'PUBLIC_REQUESTS.json')['requests'], read(OUT / 'PROMPT_TOKEN_IDS.json')
    ids = [r['request_id'] for r in rows]
    need(prep['experiment_id'] == HERE.name and ids == prep['fixed_order'], 'PUBLIC_ORDER_OR_EXPERIMENT_DRIFT')
    need(0 < len(rows) == prep['requests'] == len(set(ids)) <= 160 and set(tokens) == set(ids), 'PUBLIC_REQUEST_SET_DRIFT')
    need(prep['admitted_semantic_questions'] * 4 == len(rows) and prep['matched_slot_pairs'] * 8 == len(rows), 'PAIRED_DENOMINATOR_DRIFT')
    need(prep['max_tokens'] == MAX_TOKENS and prep['max_model_len'] == MODEL_LEN and prep['batch_size'] == BATCH and prep['sampling'] == SAMPLING, 'PUBLIC_ENGINE_BUDGET_DRIFT')
    need(prep['structured_outputs_config'] == engine.STRUCTURED_CONFIG and prep['conditions'] == list(CONDITIONS), 'PUBLIC_STRUCTURED_OR_CONDITION_DRIFT')
    need(prep['prefix_cache'] is False and prep['automatic_retry'] is False and prep['source_truncation'] is False and prep['generation_reads_gold'] is False, 'PUBLIC_EXECUTION_POLICY_DRIFT')
    groups, required = collections.defaultdict(list), []
    for row in rows:
        need(set(row) == PUBLIC_KEYS and row['root_id'] in ROOTS and row['arm'] in ARMS and row['slot_id'] in SLOTS and row['condition'] in CONDITIONS, 'PUBLIC_FIELDS_OR_DESIGN_DRIFT')
        need(row['question_type'] == 'multi_select' and row['task_operation'] == SLOTS[row['slot_id']][0] and row['cohort'] == 'new', 'PUBLIC_OPERATION_DRIFT')
        need(row['mode'] == 'THINK_OFF' and row['enable_thinking'] is False and row['base_request_id'] == row['request_id'], 'PUBLIC_MODE_DRIFT')
        need(row['labels'] == list(row['display']) == list('ABCDEF') and row['sampling_seed'] == sampling_seed(row['root_id'], row['slot_id']), 'PUBLIC_LABEL_OR_SEED_DRIFT')
        need(row['prompt_tokens'] == len(tokens[row['request_id']]) and row['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'PUBLIC_TOKEN_BUDGET_DRIFT')
        groups[(row['root_id'], row['slot_id'], row['arm'], row['candidate_id'])].append(row)
        required.extend((Path(row['prompt_path']), Path(row['schema_path']), Path(row['source_binding']['path'])))
    for group in groups.values():
        need(len(group) == 4 and {(r['condition'], r['variant_id']) for r in group} == {(c, v) for c in CONDITIONS for v in ('perm_00', 'perm_01')}, 'PUBLIC_CONDITION_DISPLAY_PAIR_DRIFT')
        for variant in ('perm_00', 'perm_01'):
            pair = [r for r in group if r['variant_id'] == variant]
            need(pair[0]['display'] == pair[1]['display'] and pair[0]['sampling_seed'] == pair[1]['sampling_seed'], 'CONDITION_PAIR_DRIFT')
    slots = {(r['root_id'], r['slot_id']) for r in rows}
    need(all({r['arm'] for r in rows if (r['root_id'], r['slot_id']) == slot} == set(ARMS) for slot in slots), 'INCOMPLETE_FG_SLOT_PAIR')
    required.extend((Path(__file__), OLD / 'solver_runner.py', WEEK / 'thinking_switch_20261024_structured_v2/run.py', SCORER_PATH,
                     HERE / 'PROTOCOL_FREEZE.json', OUT / 'CPU_API_CHECK.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json', OUT / 'PRE_SOLVER_AUDIT.json', *orig.code_paths()))
    api = read(OUT / 'CPU_API_CHECK.json')
    need(api['gpu_model_started'] is False and api['structured_outputs_config'] == engine.STRUCTURED_CONFIG and api['runtime_versions'] == prep['runtime_versions'], 'CPU_API_RECEIPT_DRIFT')
    required.extend(Path(b['path']) for b in api['implementation_bindings'])
    paths = [str(Path(b['path']).resolve()) for b in prep['bindings']]
    need(len(paths) == len(set(paths)) and all(bind(path) in prep['bindings'] for path in required), 'UNBOUND_PUBLIC_INPUT')
    return prep, rows, tokens


def check_raw(frozen, rows, tokens):
    """Original closed-loop raw receipt checks, with the declared 160-row cap."""
    ids = [r['request_id'] for r in rows]
    need(0 < len(ids) == len(set(ids)) <= 160 and frozen['request_ids'] == ids and frozen['requests'] == len(ids), 'RAW_SET_DRIFT')
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
    private = {r['request_id']: r for r in private_doc['requests']}
    need(len(private) == len(private_doc['requests']) == len(rows) and list(private) == prep['fixed_order'], 'PRIVATE_REQUEST_SET_DRIFT')
    candidates = {c['candidate_id']: c for c in read(HERE / 'CANDIDATES_PRIVATE.json')}
    records = []
    for row in rows:
        key = private[row['request_id']]
        need(all(key[k] == row[k] for k in ('candidate_id', 'arm', 'slot_id', 'condition')), 'PRIVATE_PUBLIC_ID_DRIFT')
        index = int(row['variant_id'][-2:])
        expected_public, expected_presentation = list(legacy.display_variants(candidates[row['candidate_id']]))[index][1:]
        need(key['presentation'] == expected_presentation and expected_public['options'] == row['display'], 'PRIVATE_PRESENTATION_DRIFT')
        path = OUT / 'raw' / (row['request_id'] + '.json')
        record = score_one(row, key['presentation'], read(path), read(row['schema_path']))
        record.update(phase='new', raw_binding=bind(path), legacy_request_id=key['legacy_request_id'])
        records.append(record)
    baseline, _, _, baseline_audit = baseline_records()
    need(baseline_audit == private_doc['baseline_audit'], 'OLD_BASELINE_AUDIT_DRIFT')
    admitted_ids = {r['candidate_id'] for r in records}
    matched_old = [r for r in baseline if r['candidate_id'] in admitted_ids]
    full_new = [r for r in records if r['condition'] == 'FULL']
    need(len(matched_old) == len(full_new), 'MATCHED_BASELINE_DENOMINATOR_DRIFT')
    all_records = baseline + records
    summaries, strata, comparisons, fg_comparisons = [], [], [], []
    for phase, condition in (('old', 'FULL'), ('new', 'FULL'), ('new', 'QUESTION_ONLY')):
        chosen = [r for r in all_records if r['phase'] == phase and r['condition'] == condition]
        for group in ('ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL'):
            subset = group_records(chosen, group)
            summaries.append({'phase': phase, 'condition': condition, 'group': group, **summarize(subset)})
            for dimension in ('arm', 'root_id', 'slot_id'):
                for value in sorted({r[dimension] for r in subset}):
                    strata.append({'phase': phase, 'condition': condition, 'group': group, 'dimension': dimension, 'value': value, **summarize([r for r in subset if r[dimension] == value])})
    # The historical complete baseline and its admitted-slot subset are explicit:
    # only the latter is used in old/new changes to the F minus G contrast.
    for scope, phase, condition, chosen in (
        ('complete_old_baseline', 'old', 'FULL', baseline),
        ('admitted_slot_matched', 'old', 'FULL', matched_old),
        ('admitted_slot_matched', 'new', 'FULL', full_new),
        ('admitted_slot_matched', 'new', 'QUESTION_ONLY', [r for r in records if r['condition'] == 'QUESTION_ONLY'])):
        for group in ('MC2_5_PRIMARY', 'MC6_CONTROL'):
            subset = group_records(chosen, group)
            labels = {'scope': scope, 'phase': phase, 'condition': condition, 'group': group}
            fg_comparisons.append(f_g_comparison(subset, **labels, dimension='ALL', value='ALL'))
            for dimension in ('root_id', 'slot_id'):
                for value in sorted({r[dimension] for r in subset}):
                    fg_comparisons.append(f_g_comparison([r for r in subset if r[dimension] == value], **labels, dimension=dimension, value=value))
            for root, slot in sorted({(r['root_id'], r['slot_id']) for r in subset}):
                fg_comparisons.append(f_g_comparison([r for r in subset if (r['root_id'], r['slot_id']) == (root, slot)],
                                                      **labels, dimension='root_slot', root_id=root, slot_id=slot))
    fg_full = {(r['phase'], r['group'], r['dimension'], r.get('value'), r.get('root_id'), r.get('slot_id')): r
               for r in fg_comparisons if r['scope'] == 'admitted_slot_matched' and r['condition'] == 'FULL'}
    fg_changes = []
    for key, new in fg_full.items():
        if key[0] != 'new':
            continue
        old = fg_full[('old', *key[1:])]
        fg_changes.append({k: new[k] for k in ('group', 'dimension', 'value', 'root_id', 'slot_id') if k in new} | {
            'old_F_minus_G': old['F_minus_G'], 'new_F_minus_G': new['F_minus_G'],
            'new_minus_old_F_minus_G': {metric: new['F_minus_G'][metric] - old['F_minus_G'][metric]
                                       if new['F_minus_G'][metric] is not None and old['F_minus_G'][metric] is not None else None
                                       for metric in ('exact_accuracy', 'J_score')}})
    for group in ('ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL'):
        old_group, new_group = group_records(matched_old, group), group_records(full_new, group)
        comparisons.append(comparison(old_group, new_group, group=group, dimension='ALL', value='ALL', condition='FULL'))
        for dimension in ('arm', 'root_id', 'slot_id'):
            for value in sorted({r[dimension] for r in new_group}):
                comparisons.append(comparison([r for r in old_group if r[dimension] == value], [r for r in new_group if r[dimension] == value], group=group, dimension=dimension, value=value, condition='FULL'))
    slot_comparisons = []
    for root in ROOTS:
        for arm in ARMS:
            for slot in SLOTS:
                new = [r for r in full_new if (r['root_id'], r['arm'], r['slot_id']) == (root, arm, slot)]
                old = [r for r in matched_old if (r['root_id'], r['arm'], r['slot_id']) == (root, arm, slot)]
                slot_comparisons.append(comparison(old, new, root_id=root, arm=arm, slot_id=slot, condition='FULL', content_admitted=bool(new)))
    pairs = collections.defaultdict(dict)
    for record in records:
        pairs[(record['candidate_id'], record['variant_id'])][record['condition']] = record
    diagnostic = [{'candidate_id': cid, 'variant_id': variant, 'arm': pair['FULL']['arm'], 'root_id': pair['FULL']['root_id'], 'slot_id': pair['FULL']['slot_id'],
                   'FULL_exact': pair['FULL']['exact'], 'QUESTION_ONLY_exact': pair['QUESTION_ONLY']['exact'],
                   'FULL_J': pair['FULL']['J'], 'QUESTION_ONLY_J': pair['QUESTION_ONLY']['J'],
                   'both_valid': pair['FULL']['valid'] and pair['QUESTION_ONLY']['valid'],
                   'same_canonical_prediction': pair['FULL']['valid'] and pair['QUESTION_ONLY']['valid'] and pair['FULL']['canonical_prediction'] == pair['QUESTION_ONLY']['canonical_prediction']} for (cid, variant), pair in pairs.items()]
    consistency = []
    for phase, condition in (('old', 'FULL'), ('new', 'FULL'), ('new', 'QUESTION_ONLY')):
        chosen = [r for r in all_records if r['phase'] == phase and r['condition'] == condition]
        for cid in sorted({r['candidate_id'] for r in chosen}):
            subset = [r for r in chosen if r['candidate_id'] == cid]
            consistency.append({'phase': phase, 'condition': condition, 'candidate_id': cid,
                                'both_valid': all(r['valid'] for r in subset), 'both_exact': all(r['exact'] for r in subset),
                                'same_canonical_prediction': all(r['valid'] for r in subset) and len({tuple(r['canonical_prediction']) for r in subset}) == 1})
    save(OUT / 'BASELINE_J_RECOMPUTED.json', {'records': baseline, 'audit': baseline_audit, 'historical_responses_reused': True, 'new_model_calls': 0})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze_binding': bind(OUT / 'RAW_RESPONSE_FREEZE.json'), 'private_scoring_binding': prep['private_scoring_binding'], 'original_J_scorer_binding': bind(SCORER_PATH)})
    save(OUT / 'ANALYSIS_SUMMARY.json', {'summaries': summaries, 'strata': strata, 'matched_old_vs_new_FULL': comparisons,
         'primary_comparison': [r for r in fg_comparisons if r['scope'] == 'admitted_slot_matched' and r['group'] == 'MC2_5_PRIMARY' and r['dimension'] == 'ALL'],
         'F_minus_G_by_condition_root_slot': fg_comparisons, 'FULL_change_in_F_minus_G': fg_changes,
         'root_arm_slot_old_vs_new_FULL': slot_comparisons, 'question_only_diagnostic_pairs': diagnostic, 'permutation_consistency': consistency,
         'admission_ledger': private_doc['admission_ledger'],
         'denominators': {'planned_questions': 40, 'admitted_questions': len(admitted_ids), 'planned_new_requests': 160, 'frozen_new_requests': len(records),
                          'new_submitted_requests': len(freeze['submitted_ids']), 'new_returned_requests': freeze['returned'],
                          'old_baseline_requests': len(baseline), 'matched_old_FULL_requests': len(matched_old),
                          'valid_new_responses': sum(r['valid'] for r in records), 'dependent_displays_per_question': 2},
         'metric_policy': 'Original J=TP/g-FP/(n-g), unrounded and unclipped; J_score=mean(J)*100 over valid applicable responses only. MC6 has J=None/PARTIAL_NOT_APPLICABLE and is a separate all-correct control. Strict exact uses every frozen request, including invalid/failed responses.',
         'FG_metric_policy': 'F_minus_G exact uses all matched frozen requests. F_minus_G J subtracts each arm valid-response J_score with its denominator shown. A separate both-valid display-pair J contrast uses only pairs for which both original J values apply. Root/slot pairing aligns design slots, not identical F/G question content or identical F/G permutations.',
         'condition_policy': 'FULL is the primary comparison. QUESTION_ONLY is a separate no-dialogue best-effort diagnostic, not a pure question-form control.',
         'comparison_policy': 'Historical FULL responses are rescored from their frozen raw outputs and private original mappings. Old/new deltas match admitted root/slot/arm only; rewritten questions differ semantically. No model reruns for the baseline.',
         'limits': 'Four previously exposed train roots, one frozen checkpoint, dependent display measurements, no training or unseen-root generalization claim.',
         'raw_wall_seconds': freeze['wall_seconds'], 'total_prompt_tokens': sum(r['prompt_tokens'] for r in records), 'total_output_tokens': sum(r['output_tokens'] for r in records), 'failure': freeze['failure']})
    print(json.dumps({'summaries': summaries, 'new_model_calls': len(freeze['submitted_ids']), 'old_baseline_model_calls': 0, 'failure': freeze['failure']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'analyze'))
    globals()[parser.parse_args().action]()
