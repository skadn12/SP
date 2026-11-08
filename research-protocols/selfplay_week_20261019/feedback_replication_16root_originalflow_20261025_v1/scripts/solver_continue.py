#!/usr/bin/env python3
"""Small continuation adapter for the frozen original F/G B1 evaluation.

The 146 sealed resume responses remain a read-only shard. Only slots that had
no prior PASS can enter this shard. Prompt, displays, engine and strict scoring
are imported from the original root-independent runner; no semantic retries.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[1]
W = P.parent
OUT = P / 'execution'
FROZEN_RUNNER = W / 'feedback_replication_20root_20261025_v1/solver_runner.py'
spec = importlib.util.spec_from_file_location('original_flow_continuation_frozen_runner', FROZEN_RUNNER)
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)
v2, orig = frozen.v2, frozen.orig
read, bind, check, need, save = frozen.read, frozen.bind, frozen.check, frozen.need, frozen.save
MAX_TOKENS, MODEL_LEN = frozen.MAX_TOKENS, frozen.MODEL_LEN
ARMS, SLOTS = frozen.ARMS, frozen.SLOTS
NAMESPACE = 'CL16C1'
frozen.P, frozen.OUT, frozen.REQUEST_NAMESPACE = P, OUT, NAMESPACE


def manifest():
    doc = read(P / 'CONTINUATION_MANIFEST.json')
    roots = doc['roots']
    need(len(roots) == len(set(roots)) == 16, 'FIXED_NEW16_ROOTS_REQUIRED')
    design = frozen.frozen_design({'roots': roots, 'arms': list(ARMS), 'slot_ids': list(SLOTS),
        'slots_per_root_per_arm': 8, 'display_variants': 2,
        'planned_semantic_questions': 256, 'planned_solver_requests': 512})
    return doc, design


def prior_slots():
    doc, _ = manifest()
    old = Path(doc['previous16'])
    inventory = read(P / 'REUSE_INVENTORY.json')
    need(inventory['existing_solver_outputs_reusable'] == 146
         and inventory['question_verified_reusable'] == 73 and not inventory['reuse_errors'], 'REUSE_AUDIT_NOT_PASS')
    for name, key in [('execution/RAW_RESPONSE_FREEZE.json', 'raw_freeze'),
                      ('execution/PUBLIC_REQUESTS.json', 'public_requests')]:
        expected = inventory[key]
        check(expected)
        need(Path(expected['path']).resolve() == (old / name).resolve(), 'REUSE_PATH_DRIFT')
    rows = read(old / 'execution/PUBLIC_REQUESTS.json')['requests']
    counts = collections.Counter((r['root_id'], r['arm'], r['slot_id']) for r in rows)
    need(len(rows) == 146 and len(counts) == 73 and set(counts.values()) == {2}, 'REUSE_PRESENTATION_PAIRS_DRIFT')
    return set(counts), [bind(old / 'execution/RAW_RESPONSE_FREEZE.json'),
                         bind(old / 'execution/PUBLIC_REQUESTS.json'), bind(P / 'REUSE_INVENTORY.json')]


def checked_receipt(path, candidate, expected):
    path = Path(path)
    doc = read(path)
    if doc.get('transport') == 'direct_collaboration':
        bindings = [bind(path)]
        for key in ('public_packet_binding', 'response_binding', 'private_mapping_binding'):
            check(doc[key])
            bindings.append(doc[key])
        packet = read(doc['public_packet_binding']['path'])
        response = read(doc['response_binding']['path'])
        mapping = read(doc['private_mapping_binding']['path'])
        need(doc['root_id'] == response['root_id'] == packet['root_id'] == candidate['root_id']
             and doc['arm'] == candidate['arm'], 'DIRECT_REVIEW_ROOT_OR_ARM_DRIFT')
        reviewer = doc.get('reviewer_id')
        need(isinstance(reviewer, str) and reviewer and response.get('reviewer_id') == reviewer
             and reviewer != candidate['author_id'], 'DIRECT_REVIEW_AUTHOR_OR_REVIEWER_DRIFT')
        need(packet['full_source'] == read(candidate['source_binding']['path']), 'DIRECT_REVIEW_SOURCE_DRIFT')
        entries = mapping['entries']
        questions = packet['questions']
        results = response['slots']
        ids = [entry['review_id'] for entry in entries]
        need(len(ids) == len(set(ids)) and set(ids) == {q['review_id'] for q in questions}
             and len(questions) == len(ids) and set(ids) == {r['review_id'] for r in results}
             and len(results) == len(ids), 'DIRECT_REVIEW_PUBLIC_ID_SET_DRIFT')
        matching_entries = [e for e in entries if e['candidate']['candidate_id'] == candidate['candidate_id']]
        need(len(matching_entries) == 1 and matching_entries[0]['candidate'] == candidate,
             'DIRECT_REVIEW_PRIVATE_MAPPING_DRIFT')
        review_id = matching_entries[0]['review_id']
        question = next(q for q in questions if q['review_id'] == review_id)
        public_question = {k: v for k, v in question.items() if k != 'review_id'}
        need(public_question == {k: candidate[k] for k in public_question}, 'DIRECT_REVIEW_QUESTION_DRIFT')
        result = next(r for r in results if r['review_id'] == review_id)
        matches = [r for r in doc['slots'] if r['slot_id'] == candidate['slot_id']]
        need(len(matches) == 1 and matches[0] == expected, 'DIRECT_REVIEW_RECEIPT_SLOT_DRIFT')
        mapped = matches[0]
        # Older direct reviews retained the private slot id in the response;
        # newer blind packets contain only review_id. Check either representation
        # before comparing every remaining content field exactly.
        need('slot_id' not in result or result['slot_id'] == candidate['slot_id'],
             'DIRECT_REVIEW_RESPONSE_SLOT_DRIFT')
        need(mapped.get('review_id', review_id) == review_id
             and {k: v for k, v in mapped.items() if k not in ('slot_id', 'review_id')}
                 == {k: v for k, v in result.items() if k not in ('slot_id', 'review_id')},
             'DIRECT_REVIEW_RESPONSE_MAPPING_DRIFT')
        return bindings
    need(doc.get('accepted') is True and not doc.get('timed_out') and doc.get('returncode') == 0,
         'INDEPENDENT_REVIEW_RECEIPT_NOT_ACCEPTED:' + str(path))
    bindings = [bind(path)]
    for b in doc.get('hashes', {}).values():
        check(b)
        bindings.append(b)
    response = doc['response']
    need(response['root_id'] == candidate['root_id'] and response['arm'] == candidate['arm'], 'REVIEW_RECEIPT_ROOT_OR_ARM_DRIFT')
    matches = [r for r in response['slots'] if r['slot_id'] == candidate['slot_id']]
    need(len(matches) == 1 and matches[0] == expected, 'REVIEW_RECEIPT_SLOT_DRIFT')
    return bindings


def checked_admission(candidate, slot, source):
    """Accept the existing independent Gold/content freeze, never invent a PASS."""
    need(slot['status'] == 'PASS' and slot['candidate'] == candidate, 'FROZEN_CANDIDATE_NOT_PASS')
    need(all(slot[k] == candidate[k] for k in ('root_id', 'arm', 'slot_id', 'candidate_id')), 'FROZEN_CANDIDATE_ID_DRIFT')
    review = slot['review']
    need(review['status'] == 'PASS' and not review.get('errors'), 'CONTENT_REVIEW_NOT_PASS')
    gold, content = review['gold'], review['review']
    need(gold['decision'] == content['decision'] == 'PASS', 'INDEPENDENT_GOLD_OR_CONTENT_NOT_PASS')
    need(gold['gold'] == candidate['gold'], 'INDEPENDENT_GOLD_DISAGREEMENT')
    need(review['gold_receipt'] != review['review_receipt'], 'GOLD_CONTENT_REVIEW_NOT_INDEPENDENT')
    refs = {t['turn_ref']: t for t in source['turns']}
    need(len(refs) == len(source['turns']), 'DUPLICATE_SOURCE_REFS')
    for audited in (gold, content):
        need(audited.get('source_quotes'), 'MISSING_REVIEW_SOURCE_QUOTES')
        for evidence in audited['source_quotes']:
            need(evidence['turn_ref'] in refs and evidence['quote'] in refs[evidence['turn_ref']]['text'], 'REVIEW_QUOTE_NOT_EXACT')
    bindings = checked_receipt(review['gold_receipt'], candidate, gold)
    bindings += checked_receipt(review['review_receipt'], candidate, content)
    reviewer_ids = [read(review[k]).get('reviewer_id') for k in ('gold_receipt', 'review_receipt')]
    if all(reviewer_ids):
        need(reviewer_ids[0] != reviewer_ids[1], 'GOLD_CONTENT_REVIEWER_NOT_INDEPENDENT')
    return bindings


def precheck():
    """Construct CPU API/config objects only; never create an LLM or use a GPU."""
    _, design = manifest()
    reused, _ = prior_slots()
    endpoint, config = orig.endpoint_receipt()
    need(Path(endpoint['tokenizer_directory']).resolve() == Path('/root/autodl-tmp/Qwen3-4B'), 'INDEPENDENT_TOKENIZER_REQUIRED')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    probe = tok.apply_chat_template([{'role': 'user', 'content': 'CPU interface check.'}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    need(reasoner.is_reasoning_end(tok.encode(probe, add_special_tokens=False)) is True, 'THINK_OFF_CPU_GATE_FAILED')
    api = v2.cpu_api_check()
    receipt = {'status': 'PASS_CPU_ONLY_NO_MODEL', 'gpu_model_started': False,
        'new16_roots': len(design['root_ids']), 'reused_semantic_slots': len(reused),
        'reused_responses': 146, 'remaining_semantic_slots': 256 - len(reused),
        'model_directory': str(orig.MODEL), 'tokenizer_directory': str(orig.TOKENIZER),
        'config_receipt': config, 'runtime_versions': orig.versions(), 'api': api,
        'adapter_binding': bind(Path(__file__)), 'frozen_runner_binding': bind(FROZEN_RUNNER),
        'scorer': str(orig.BASE_PATH), 'box_grammar': '<box>A B</box>',
        'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN, 'batch_size': 8}
    save(OUT / 'CPU_PREFLIGHT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'reused_responses', 'remaining_semantic_slots', 'tokenizer_directory')}), flush=True)


def prepare():
    need(not (OUT / 'PREPARATION.json').exists() and not (OUT / 'STARTED.json').exists(), 'NO_PREPARATION_OVERWRITE')
    _, design = manifest()
    reused, reuse_bindings = prior_slots()
    candidates = read(P / 'CANDIDATES_PRIVATE.json')['candidates']
    selection = read(P / 'SOURCE_SELECTION.json')
    roots = {r['root_id']: r for r in selection['roots']}
    need(len(roots) == len(selection['roots']) and 0 < len(roots) <= 16 and set(roots) <= set(design['root_ids']), 'SOURCE_ROOT_SET_DRIFT')
    question_freeze = read(P / 'QUESTION_FREEZE.json')
    for b in question_freeze.get('bindings', []):
        check(b)
    slots = {s['candidate_id']: s for s in question_freeze['slots'] if s['status'] == 'PASS'}
    need(isinstance(candidates, list) and len(candidates) == len({c['candidate_id'] for c in candidates}), 'CANDIDATE_SET_DRIFT')
    keys = [(c['root_id'], c['arm'], c['slot_id']) for c in candidates]
    need(len(keys) == len(set(keys)), 'DUPLICATE_NEW_SEMANTIC_SLOT')
    # Prior PASS slots are excluded before any prompt or request is prepared.
    skipped = [c['candidate_id'] for c in candidates if (c['root_id'], c['arm'], c['slot_id']) in reused]
    candidates = [c for c in candidates if (c['root_id'], c['arm'], c['slot_id']) not in reused]
    need(candidates and len(candidates) <= 183, 'NO_NEW_ADMITTED_CANDIDATES_OR_TOO_MANY')
    endpoint, config = orig.endpoint_receipt()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(orig.TOKENIZER), local_files_only=True)
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    api = v2.cpu_api_check()
    save(OUT / 'CPU_API_CHECK.json', api)
    rows, tokens, scoring, quality, private_bindings = [], {}, [], [], []
    bindings = [bind(Path(__file__)), bind(FROZEN_RUNNER), bind(W / 'thinking_switch_20261024_structured_v2/run.py'),
                bind(OUT / 'CPU_API_CHECK.json'), *api['implementation_bindings'], *[bind(x) for x in orig.code_paths()]]
    for c in sorted(candidates, key=lambda c: (c['root_id'], c['slot_id'], c['arm'])):
        cid, root, arm, slot = (c[k] for k in ('candidate_id', 'root_id', 'arm', 'slot_id'))
        need(root in roots and arm in ARMS and slot in SLOTS, 'UNPLANNED_CANDIDATE:' + cid)
        need(c['source_binding'] == {k: roots[root][k] for k in ('path', 'sha256', 'bytes')}, 'SOURCE_BINDING_DRIFT:' + cid)
        check(c['source_binding'])
        source = read(c['source_binding']['path'])
        need(source['root_id'] == root, 'SOURCE_ROOT_DRIFT:' + cid)
        orig.gate._validate(c)
        kind, operation, count = SLOTS[slot]
        field = 'options' if kind == 'multi_select' else 'items'
        need(c['question_type'] == kind and c['task_operation'] == operation and set(c[field]) == set('ABCDEF')
             and (count is None or len(c['gold']) == count), 'FROZEN_SLOT_DRIFT:' + cid)
        need(cid in slots, 'CANDIDATE_WITHOUT_FROZEN_PASS:' + cid)
        private_bindings += checked_admission(c, slots[cid], source)
        refs = [t['turn_ref'] for t in source['turns']]
        seed = int.from_bytes(hashlib.sha256((root + ':' + slot).encode()).digest()[:4], 'big') % 2147483647
        for index, public_row, presentation in frozen.display_variants(c):
            rid = f'{cid}__perm_{index:02}__FULL__THINK_OFF__{NAMESPACE}'
            row = {k: v for k, v in public_row.items() if k != field}
            row.update(request_id=rid, base_request_id=rid, cohort='new', condition='FULL', mode='THINK_OFF', enable_thinking=False,
                       arm=arm, slot_id=slot, variant_id=f'perm_{index:02}', display=public_row[field], labels=list(public_row[field]), sampling_seed=seed)
            old = orig.diagnostic.make_prompt(tok, row, row['display'], source)
            pattern = r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n'
            match = re.fullmatch(pattern, old, re.S)
            need(match, 'PROMPT_TEMPLATE_DRIFT')
            system, user = match.groups()
            oldtail = 'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.'
            newtail = 'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.'
            need(user.count(oldtail) == 1, 'NEUTRAL_INSTRUCTION_DRIFT')
            messages = [{'role': 'system', 'content': system.replace('Give concise, verifiable explanations', 'In the final response, give concise, verifiable explanations')},
                        {'role': 'user', 'content': user.replace(oldtail, newtail)}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(prompt, add_special_tokens=False)
            need(len(ids) + MAX_TOKENS <= MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION:' + rid)
            need(reasoner.is_reasoning_end(ids) is True, 'THINK_OFF_REASONING_GATE_DRIFT:' + rid)
            pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
            pp.parent.mkdir(parents=True, exist_ok=True)
            with pp.open('x') as handle:
                handle.write(prompt)
            save(sp, orig.diagnostic.repair.schema(row['labels'], refs, row['question_type']))
            row.update(prompt_path=str(pp), schema_path=str(sp), prompt_tokens=len(ids),
                       messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            need(set(row) == frozen.PUBLIC_KEYS, 'PUBLIC_FIELD_ALLOWLIST_DRIFT')
            rows.append(row)
            tokens[rid] = ids
            scoring.append({'request_id': rid, 'candidate_id': cid, 'arm': arm, 'slot_id': slot, 'presentation': presentation})
            bindings += [bind(pp), bind(sp), c['source_binding']]
        quality.append({'candidate_id': cid, 'root_id': root, 'arm': arm, 'slot_id': slot,
                        'question_words': len(c['question'].split()), 'option_words': {k: len(v.split()) for k, v in c[field].items()}})
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    private_doc = {'requests': scoring, 'candidate_binding': bind(P / 'CANDIDATES_PRIVATE.json'),
        'selection_binding': bind(P / 'SOURCE_SELECTION.json'), 'question_freeze_binding': bind(P / 'QUESTION_FREEZE.json'),
        'private_bindings': list({b['path']: b for b in private_bindings}.values()), 'reused_shard_bindings': reuse_bindings}
    save(OUT / 'PRIVATE_SCORING.json', private_doc)
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    save(OUT / 'PRE_SOLVER_QUALITY.json', {'selection_before_solver': True, 'records': quality,
        'skipped_prior_pass_candidate_ids': skipped, 'display_seed_bases': [2026102427, 2026102428],
        'matched_sampling_seed_key': ['root_id', 'slot_id'], 'same_cross_arm_permutation_claimed': False})
    bindings += [bind(OUT / n) for n in ('PUBLIC_REQUESTS.json', 'PROMPT_TOKEN_IDS.json', 'PRE_SOLVER_QUALITY.json')]
    save(OUT / 'PREPARATION.json', {'requests': len(rows), 'planned_requests': 512, 'planned_semantic_questions': 256,
        'design': design, 'planned_root_ids': design['root_ids'], 'selected_source_root_ids': sorted(roots),
        'bindings': list({b['path']: b for b in bindings}.values()), 'private_scoring_binding': bind(OUT / 'PRIVATE_SCORING.json'),
        'fixed_order': [r['request_id'] for r in rows], 'config_receipt': config, 'runtime_versions': orig.versions(),
        'structured_outputs_config': v2.STRUCTURED_CONFIG, 'max_tokens': MAX_TOKENS, 'max_model_len': MODEL_LEN,
        'sampling': {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0},
        'batch_size': 8, 'prefix_cache': False, 'automatic_retry': False, 'source_truncation': False,
        'reused_responses_separate_shard': 146, 'public_generation_reads_private_scoring': False})
    print(json.dumps({'prepared_new_responses': len(rows), 'new_questions': len(candidates), 'reused_responses_separate_shard': 146,
                      'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)}), flush=True)


def public():
    prep = read(OUT / 'PREPARATION.json')
    design = frozen.preparation_design(prep)
    for b in prep['bindings']:
        check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    need(0 < len(rows) == prep['requests'] <= 366 and len(rows) == len({r['request_id'] for r in rows}), 'NEW_REQUEST_SET_DRIFT')
    need(prep['fixed_order'] == [r['request_id'] for r in rows] and set(tokens) == set(prep['fixed_order']), 'PUBLIC_ORDER_OR_TOKENS_DRIFT')
    need(prep['max_tokens'] == MAX_TOKENS and prep['max_model_len'] == MODEL_LEN
         and prep['structured_outputs_config'] == v2.STRUCTURED_CONFIG, 'PUBLIC_BUDGET_DRIFT')
    need(prep['sampling'] == {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1, 'engine_seed': 0}
         and prep['batch_size'] == 8 and prep['prefix_cache'] is False and prep['automatic_retry'] is False
         and prep['source_truncation'] is False, 'PUBLIC_SAMPLING_DRIFT')
    groups = collections.defaultdict(list)
    bound_paths = {str(Path(b['path']).resolve()) for b in prep['bindings']}
    for r in rows:
        need(set(r) == frozen.PUBLIC_KEYS and r['root_id'] in design['root_ids'] and r['arm'] in ARMS and r['slot_id'] in SLOTS, 'PUBLIC_FIELDS_DRIFT')
        need(r['mode'] == 'THINK_OFF' and r['condition'] == 'FULL' and r['enable_thinking'] is False and r['cohort'] == 'new', 'PUBLIC_MODE_DRIFT')
        need(r['request_id'].endswith('__' + NAMESPACE) and r['base_request_id'] == r['request_id'], 'PUBLIC_REQUEST_NAMESPACE_DRIFT')
        need(r['labels'] == list(r['display']) == list('ABCDEF') and len(tokens[r['request_id']]) == r['prompt_tokens']
             and r['prompt_tokens'] + MAX_TOKENS <= MODEL_LEN, 'PUBLIC_CONTEXT_OR_LABEL_DRIFT')
        expected_seed = int.from_bytes(hashlib.sha256((r['root_id'] + ':' + r['slot_id']).encode()).digest()[:4], 'big') % 2147483647
        need(r['sampling_seed'] == expected_seed and r['question_type'] == SLOTS[r['slot_id']][0]
             and r['task_operation'] == SLOTS[r['slot_id']][1], 'PUBLIC_OPERATION_OR_SEED_DRIFT')
        for path in (r['prompt_path'], r['schema_path'], r['source_binding']['path']):
            need(str(Path(path).resolve()) in bound_paths, 'UNBOUND_PUBLIC_INPUT')
        groups[(r['root_id'], r['arm'], r['slot_id'], r['candidate_id'])].append(r)
    need(len({key[:3] for key in groups}) == len(groups), 'DUPLICATE_PUBLIC_SLOT')
    need(all(len(rs) == 2 and {r['variant_id'] for r in rs} == {'perm_00', 'perm_01'} for rs in groups.values()), 'DISPLAY_PAIR_DRIFT')
    return prep, rows, tokens


def run():
    # Keep the ZMQ IPC pathname below the OS limit without changing model settings.
    os.environ['TMPDIR'] = '/tmp/gxy16tmp'
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    frozen.public = public
    frozen.run()


def analyze():
    prep, rows, tokens = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    frozen.check_raw(freeze, rows, tokens)
    check(prep['private_scoring_binding'])
    private_doc = read(OUT / 'PRIVATE_SCORING.json')
    for b in [private_doc[k] for k in ('candidate_binding', 'selection_binding', 'question_freeze_binding')]:
        check(b)
    for b in private_doc['private_bindings'] + private_doc['reused_shard_bindings']:
        check(b)
    scoring = {r['request_id']: r for r in private_doc['requests']}
    candidates = {c['candidate_id']: c for c in read(private_doc['candidate_binding']['path'])['candidates']}
    need(list(scoring) == prep['fixed_order'] and len(scoring) == len(rows), 'PRIVATE_SCORING_REQUEST_SET_DRIFT')
    records = []
    for row in rows:
        path = OUT / 'raw' / (row['request_id'] + '.json')
        raw = read(path)
        final, parsed, answer, errors, valid = None, None, [], [], False
        try:
            need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
            need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
            thinking, final, boundary = v2.split_output(raw['raw_output'])
            need(not thinking, 'UNEXPECTED_NONEMPTY_THINKING')
            parsed, answer = v2.legal_answer(final, row)
            orig.base.validate_output(final, row, read(row['schema_path']))
            valid = True
        except Exception as exc:
            errors.append(str(exc))
        private = scoring[row['request_id']]
        need(all(private[k] == row[k] for k in ('candidate_id', 'arm', 'slot_id')), 'PRIVATE_PUBLIC_ID_DRIFT')
        _, expected_public, expected_presentation = list(frozen.display_variants(candidates[row['candidate_id']]))[int(row['variant_id'][-2:])]
        need(private['presentation'] == expected_presentation and expected_public['options' if row['question_type'] == 'multi_select' else 'items'] == row['display'], 'PRIVATE_PRESENTATION_DRIFT')
        strict = orig.base.score_record(row, private['presentation'], answer, valid)
        strict['variant_id'] = row['variant_id']
        if strict['question_type'] == 'multi_select':
            tp = len(set(strict['prediction']) & set(strict['gold']))
            fp, fn = strict['false_positive'], strict['false_negative']
            partial = {'precision': tp / (tp + fp) if tp + fp else 0, 'recall': tp / (tp + fn),
                       'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}
        else:
            partial = {'pairwise_accuracy': strict['pairwise_correct'] / strict['pairwise_total']}
        records.append({'request_id': row['request_id'], 'arm': row['arm'], 'slot_id': row['slot_id'],
            'strict': strict, 'partial': partial, 'errors': errors, 'parsed': parsed, 'final_text': final,
            'raw_binding': bind(path), 'prompt_tokens': row['prompt_tokens'], 'output_tokens': len(raw.get('output_token_ids', [])),
            'finish_reason': raw.get('finish_reason')})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze_binding': bind(OUT / 'RAW_RESPONSE_FREEZE.json'),
        'private_scoring_binding': prep['private_scoring_binding'], 'frozen_scorer_binding': bind(orig.BASE_PATH),
        'reused_responses_separate_shard': 146})
    summaries = [{'arm': arm, 'group': group, **frozen.summary_metrics([r for r in records if r['arm'] == arm and frozen.in_group(r, group)])}
                 for arm in ARMS for group in ('ALL', 'MC2_5_PRIMARY', 'MC6_CONTROL', 'ORDERING', 'NON_MC6_PRIMARY')]
    save(OUT / 'ANALYSIS_SUMMARY.json', {'summaries': summaries, 'new_shard_responses': len(records),
        'reused_responses_separate_shard': 146, 'planned_new16_responses': 512,
        'raw_wall_seconds': freeze['wall_seconds'], 'two_displays_are_dependent': True})
    print(json.dumps({'new_returned': freeze['returned'], 'strict_valid': sum(r['strict']['valid'] for r in records),
                      'strict_exact': sum(r['strict']['exact'] for r in records)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('precheck', 'prepare', 'run', 'analyze'))
    args = parser.parse_args()
    globals()[args.action]()
