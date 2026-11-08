"""Independent public-only preflight of the paired ordering construction probe.

Does not import prepare/execute implementations, open private QA JSON, read old
raw output or score JSON, instantiate an LLM, or write inside sealed Day 1/2.
Only immutable bytes of private inputs and old outputs are hashed for provenance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
DAY1 = WEEK / 'day1'
DAY2 = WEEK / 'day2'
sys.path.insert(0, str(DAY2 / 'solver'))
import adapter as a

DISCOVERY_ROOTS = {'cc-train-episode-' + s for s in ['112812', '148314', '130221']}
PINNED_SEALS = {
    'day1': '8918702665d1965e3dc54015239a871213df24c7d9d1db983d66f0bb013c30e8',
    'day2': '4411bdaedfbedc4c0b478f59ceda9e9a70b65f15c13e2a915b76c555a5dfafa7',
}
SYSTEM = 'Use only the supplied dialogue. Return brief auditable source citations and a final answer in the required JSON schema.'
TASK = ('For every displayed label provide a brief source citation: its status, turn_ref and short exact quote. '
        'Use empty turn_ref and quote when no citation exists. For ordering, cite the occurrence used to order each item. '
        'Return the final answer using displayed labels only. Return one JSON object with evidence and answer. Do not give extended reasoning.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def relaxed_schema(x):
    if isinstance(x, dict):
        return {k: relaxed_schema(v) for k, v in x.items() if k != 'uniqueItems'}
    if isinstance(x, list):
        return [relaxed_schema(v) for v in x]
    return x


def independent_prompt(tok, q, variant, source, intervention=None):
    rows = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
    user = f"Dialogue:\n{rows}\n\nQuestion (ordering):\n{q['question']}\n\nItems:\n"
    user += '\n'.join(f'{k}. {v}' for k, v in variant['items'].items()) + '\n\n' + TASK
    if intervention is not None:
        user += intervention
    return tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def walk_bindings(x):
    if isinstance(x, dict):
        if {'path', 'sha256', 'bytes'} <= set(x):
            yield x
        else:
            for child in x.values():
                yield from walk_bindings(child)
    elif isinstance(x, list):
        for child in x:
            yield from walk_bindings(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, outpath = args.run.resolve(), args.out.resolve()
    names = []
    verified_files = {}

    def check(value, name):
        if not value:
            raise AssertionError(name)
        names.append(name)

    def binding(row, name):
        p = Path(row['path']).resolve()
        expected = (row['sha256'], row['bytes'])
        if p in verified_files:
            check(verified_files[p] == expected, name + ':consistent_binding')
        else:
            check(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], name)
            verified_files[p] = expected

    check(run.is_relative_to(HERE) and outpath.is_relative_to(HERE), 'new_probe_directory_only')
    check(not outpath.exists(), 'exclusive_audit_output')
    check(not (run / 'STARTED.json').exists(), 'not_started_before_independent_audit')
    check(not any((run / x).exists() for x in ['SCORES.json', 'SCORING_STARTED.json', 'RAW_RESPONSE_FREEZE.json']),
          'no_current_scores_or_raw_freeze')

    # Read seal metadata only; historical output/private QA files are byte-hashed, not JSON-read.
    for label, pinned in PINNED_SEALS.items():
        directory = WEEK / label
        sealpath = directory / 'COMPLETE_MANIFEST.json'
        check(sha(sealpath) == pinned, label + ':pinned_manifest')
        seal = read(sealpath)
        check(seal['file_count'] == len(seal['files']), label + ':sealed_count')
        check(sum(x['bytes'] for x in seal['files']) == seal['total_bytes'], label + ':sealed_bytes')
        expected_files = set()
        for entry in seal['files']:
            rel = Path(entry['path'])
            check(not rel.is_absolute() and '..' not in rel.parts, label + ':safe_relative_path')
            p = (directory / rel).resolve()
            check(p.is_relative_to(directory), label + ':path_in_sealed_directory')
            expected_files.add(p)
            binding({'path': str(p), 'sha256': entry['sha256'], 'bytes': entry['bytes']}, label + ':sealed:' + entry['path'])
        actual_files = {p.resolve() for p in directory.rglob('*') if p.is_file() and p.name != 'COMPLETE_MANIFEST.json'
                        and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        check(actual_files == expected_files, label + ':no_old_files_added_removed_or_changed')

    protocol_path = HERE / 'PROTOCOL.json'
    protocol = read(protocol_path)
    intervention = protocol['intervention_text']
    check(isinstance(intervention, str) and intervention.startswith('\n\n') and bool(intervention.strip()), 'one_nonempty_global_intervention')
    freeze = read(run / 'PREFLIGHT_FREEZE.json')
    check(freeze.get('inference_started') is False, 'freeze_precedes_inference')
    frozen_paths = set()
    for row in freeze['files']:
        p = Path(row['path']).resolve()
        check(p.is_relative_to(run), 'frozen_input_in_new_run')
        check(p not in frozen_paths, 'no_duplicate_frozen_path')
        frozen_paths.add(p)
        binding(row, 'preflight:' + str(p.relative_to(run)))
    actual_prepared = {p.resolve() for p in run.rglob('*') if p.is_file() and p.name != 'PREFLIGHT_FREEZE.json'
                       and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    check(frozen_paths == actual_prepared, 'all_prepared_run_files_frozen')
    m = read(run / 'manifest.json')
    check(m['namespace'] == 'train.feedback' and m['use_as_final_eval'] is False, 'train_feedback_only_not_final_eval')
    check(m['solver_model'] == 'S0_M_clean' and m['branch'] == 'evidence', 'fixed_S0_evidence_branch')
    check(m['question_count'] == 9 and m['root_count'] == 9 and m['request_count'] == 108, 'fixed_9_roots_108_calls')
    check(m['batch_size'] == 8 and m['max_model_len'] == 16384, 'same_batch_and_context')
    check(m['prefix_caching'] is False and m['automatic_retry'] is False and m['source_truncation'] is False, 'unchanged_runtime_and_no_retry')
    check(m['gold_opened_by_prepare'] is False and m['training_started'] is False, 'no_prepare_gold_or_training')
    check(m['evidence_decoder'] == {'structured_json_schema': True, 'temperature': .8, 'top_p': .9,
                                   'max_tokens': 1024, 'enable_thinking': False}, 'same_evidence_decoder')
    check(m['versions'] == {k: importlib.metadata.version(k) for k in ['transformers', 'torch', 'vllm', 'tokenizers']}, 'runtime_library_versions')
    check(Path(m['python_prefix']).resolve() == Path(sys.prefix).resolve() == a.REPO / '.venv-local4b', 'proven_local4b_python')
    for group in ['input_bindings', 'code_bindings', 'model_bindings']:
        check(bool(m[group]), 'nonempty_' + group)
        for i, row in enumerate(m[group]):
            binding(row, group + ':' + str(i))
    bound_inputs = {Path(r['path']).resolve(): r for r in m['input_bindings']}
    check(protocol_path.resolve() in bound_inputs, 'protocol_bound_in_manifest')
    check(m['protocol_binding'] == bind(protocol_path), 'explicit_protocol_binding')
    policy_freeze = read(HERE / 'POLICY_FREEZE.json')
    check(policy_freeze['status'] == 'FROZEN_BEFORE_NEW_INFERENCE' and policy_freeze['new_solver_calls_at_freeze'] == 0,
          'policy_frozen_before_new_output')
    check(policy_freeze['prior_sealed_snapshot'] == bind(DAY2 / 'COMPLETE_MANIFEST.json'), 'policy_bound_to_original_day2')
    for row in policy_freeze['files']:
        binding(row, 'policy_freeze_bound_input')
    check({Path(r['path']).resolve() for r in policy_freeze['files']} == {protocol_path.resolve(), (HERE / 'SELECTION.json').resolve()},
          'policy_freeze_binds_rules_and_selection')
    check((HERE / 'POLICY_FREEZE.json').resolve() in bound_inputs and (HERE / 'SELECTION.json').resolve() in bound_inputs,
          'policy_freeze_and_selection_bound_in_manifest')
    check(protocol['namespace'] == 'train.feedback' and protocol['solver_model'] == 'S0_M_clean', 'protocol_fixed_namespace_and_model')
    check(protocol['planned'] == {'independent_questions': 9, 'roots': 9, 'matched_pairs': 54, 'requests': 108,
                                 'discovery_requests_per_arm': 18, 'replication_requests_per_arm': 36}, 'protocol_fixed_denominators')
    check(protocol['permutations']['seed'] == 2026101903 and protocol['permutations']['count_per_question'] == 6, 'protocol_display_seed_and_count')
    check(protocol['sampling']['seed_base'] == 2026101905 and protocol['sampling']['same_seed_within_pair'] is True,
          'protocol_paired_sampling_rule')
    sampling = protocol['sampling']
    check(all(sampling.get(k) == v for k, v in {'temperature': .8, 'top_p': .9, 'max_output_tokens': 1024,
          'enable_thinking': False, 'max_model_len': 16384, 'batch_size': 8, 'prefix_caching': False,
          'automatic_retry': False, 'source_truncation': False}.items()), 'protocol_runtime_constants')
    check(protocol['continue_criteria'] == {'role': 'Exploratory work threshold only; not statistical significance, generalization or automatic cause admission.',
          'minimum_positive_replication_roots': 3, 'replication_exact_net_positive': True,
          'replication_correct_pairs_not_decreased': True, 'replication_citation_answer_consistent_count_net_positive': True,
          'both_arms_all_strict_valid': True}, 'protocol_prefrozen_exploratory_criteria')
    for label in PINNED_SEALS:
        check((WEEK / label / 'COMPLETE_MANIFEST.json').resolve() in bound_inputs, label + ':seal_bound_in_manifest')
    model_spec = read(DAY1 / 'MODEL_BINDINGS.json')['models']
    expected_model_rows = model_spec['S0_M_clean_merged']['files'] + [r for r in model_spec['base']['files']
                            if Path(r['path']).name in {'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'}]
    check(m['model_bindings'] == expected_model_rows, 'original_day1_model_hash_bindings')
    check(m['model'] == model_spec['S0_M_clean_merged']['directory'], 'fixed_merged_S0_path')
    check(m['tokenizer'] == '/root/autodl-tmp/Qwen3-4B', 'fixed_original_base_tokenizer')
    check(m['private_answer_binding'] == bind(DAY1 / 'diagnosis_inputs/PRIVATE_QUESTIONS.json'), 'private_QA_bytes_bound_not_opened')
    binding(m['private_answer_binding'], 'private_QA_hash_only')
    admission_binding = m['content_admission_binding']
    check(Path(admission_binding['path']).resolve() == DAY2 / 'CONTENT_ADMISSION.json', 'same_prior_content_admission')
    binding(admission_binding, 'content_admission_binding')
    admission = read(admission_binding['path'])
    check(admission['status'] == 'PASS' and admission['namespace'] == 'train.feedback', 'existing_content_admission_PASS')
    check(admission['private_questions'] == m['private_answer_binding'], 'admission_same_private_QA_bytes')
    check(admission['public_questions'] == bind(DAY1 / 'diagnosis_inputs/PUBLIC_QUESTIONS.json'), 'admission_same_public_QA')
    for row in walk_bindings(admission):
        binding(row, 'admission_recursive_binding')

    original_questions = read(DAY1 / 'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    ordering = [q for q in original_questions if q['question_type'] == 'ordering']
    check(len(ordering) == 20 and len({q['root_id'] for q in ordering}) == 20, 'original_20_ordering_roots')
    discovery = [q for q in ordering if q['root_id'] in DISCOVERY_ROOTS]
    other = [q for q in ordering if q['root_id'] not in DISCOVERY_ROOTS]
    check(len(discovery) == 3 and len(other) == 17, 'discovery_vs_other_pool')
    replication = sorted(other, key=lambda q: hashlib.sha256(
        f"2026101900|order-construction-replication|{q['candidate_id']}".encode()).hexdigest())[:6]
    selected = sorted(discovery, key=lambda q: q['candidate_id']) + sorted(replication, key=lambda q: q['candidate_id'])
    qs = read(run / 'PUBLIC_QUESTIONS.json')['questions']
    check(qs == selected, 'independent_hash_selection_and_cohort_question_order')
    selection = read(HERE / 'SELECTION.json')
    expected_selection = [{'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                           'cohort': 'discovery' if q['root_id'] in DISCOVERY_ROOTS else 'replication'} for q in selected]
    check(selection['questions'] == expected_selection and selection['uses_new_scores'] is False,
          'public_selection_record_matches_independent_hash_selection')
    train_roots = set(read(DAY1 / 'evaluation/ROOT_SELECTION.json')['wave1']['root_ids'])
    check({q['root_id'] for q in qs} <= train_roots, 'all_roots_frozen_train_wave1')
    admitted = {q['candidate_id']: q for q in admission['questions']}
    maps = read(run / 'private/DISPLAY_MAPPINGS.json')
    ids = read(run / 'PROMPT_TOKEN_IDS.json')
    check(maps['public_to_canonical'] == {q['candidate_id']: a.content_ids(q) for q in qs}, 'canonical_mappings_public_content_only')
    rows = m['requests']
    rids = [r['request_id'] for r in rows]
    check(len(rows) == 108 and len(set(rids)) == 108, '108_unique_requests')
    check(set(maps['requests']) == set(ids) == set(rids), 'complete_display_maps_and_token_ids')
    check(Counter(r['probe_arm'] for r in rows) == {'control': 54, 'treatment': 54}, '54_calls_each_arm')
    check(Counter(r['cohort'] for r in rows) == {'discovery': 36, 'replication': 72}, 'fixed_cohort_denominators')
    pairs_file = read(run / 'MATCHED_PAIRS.json')['pairs']
    check(len(pairs_file) == 54 and len({p['pair_id'] for p in pairs_file}) == 54, '54_unique_pair_metadata_rows')
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['pair_id']].append(row)
    check(len(pairs) == 54 and all(len(v) == 2 for v in pairs.values()), '54_complete_matched_pairs')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    spec = importlib.util.spec_from_file_location('order_probe_independent_native_protocol', a.NATIVE / 'protocol.py')
    native_protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_protocol)
    _, solver = a.native()
    check(m['decoder_contract'] == solver.generation_config('multi_select'), 'same_native_decoder_contract')
    api = m['decoder_contract']['structured_output_api']
    expected_order = []
    params_to_verify = None
    for q in qs:
        qid = q['candidate_id']
        check(set(q) <= a.PUBLIC_KEYS, 'public_whitelist:' + qid)
        a.public_question(q)
        cohort = 'discovery' if q['root_id'] in DISCOVERY_ROOTS else 'replication'
        check(q['question_type'] == 'ordering' and len(q['items']) == 6, 'six_item_ordering:' + qid)
        check(admitted[qid]['decision'] == 'PASS_PRIOR_VERSION_BOUND', 'prior_question_content_PASS:' + qid)
        source_path = Path(q['source_path'])
        check(source_path.is_absolute(), 'absolute_source:' + qid)
        binding({'path': str(source_path), 'sha256': q['source_sha256'], 'bytes': source_path.stat().st_size}, 'source_original:' + qid)
        copied_source = run / 'sources' / (q['root_id'] + '.json')
        check(sha(copied_source) == q['source_sha256'], 'source_copy_exact_bytes:' + qid)
        source = read(source_path)
        check(source['root_id'] == q['root_id'] and len({t['turn_ref'] for t in source['turns']}) == len(source['turns']), 'source_refs_unique:' + qid)
        check(admitted[qid]['source_binding'] == bind(source_path), 'admission_same_source:' + qid)
        variants = a.variants(q, 2026101903, 6)
        check(variants == native_protocol.make_variants(q, seed=2026101903, count=6), 'native_v15_display_variants:' + qid)
        for v in variants:
            vid = v['variant_id']
            pair_id = f'{qid}__{vid}'
            pair = pairs[pair_id]
            arm_order = ['control', 'treatment'] if hashlib.sha256(('arm-order|' + pair_id).encode()).digest()[0] % 2 == 0 else ['treatment', 'control']
            check([r['probe_arm'] for r in pair] == arm_order, 'counterbalanced_arm_order:' + pair_id)
            seed = int.from_bytes(hashlib.sha256(f'2026101905|{pair_id}'.encode()).digest()[:8], 'big') % (2 ** 31 - 1)
            pair_rows = {}
            control_prompt = independent_prompt(tok, q, v, source)
            check(control_prompt == a.evidence_prompt(tok, q, v, source), 'independent_original_evidence_prompt:' + pair_id)
            for row in pair:
                rid = row['request_id']
                arm = row['probe_arm']
                expected_order.append(rid)
                check(row['candidate_id'] == qid and row['root_id'] == q['root_id'] and row['cohort'] == cohort, 'request_question_and_cohort:' + rid)
                check(row['question_type'] == 'ordering' and row['branch'] == 'evidence' and row['variant_id'] == vid, 'request_type_branch_variant:' + rid)
                check(row['slot'] == 0 and row['seed'] == seed, 'matched_pair_hashed_seed:' + rid)
                check(row['display'] == {'items': v['items']} and maps['requests'][rid] == v, 'unchanged_public_display_mapping:' + rid)
                check(row['source_sha256'] == q['source_sha256'] and row['source_turns'] == len(source['turns']), 'full_source_metadata:' + rid)
                check(row['max_output_tokens'] == 1024, 'fixed_1024_output_budget:' + rid)
                check(row['json_schema'] == a.evidence_schema(q), 'original_strict_evidence_schema:' + rid)
                check(row['json_schema']['properties']['answer']['uniqueItems'] is True, 'strict_unique_answer_preserved:' + rid)
                prompt_path = run / 'prompts' / (rid + '.txt')
                prompt = prompt_path.read_text()
                expected = independent_prompt(tok, q, v, source, intervention if arm == 'treatment' else None)
                check(prompt == expected, 'only_global_intervention_or_exact_control:' + rid)
                check(hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'], 'prompt_sha256:' + rid)
                check(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns']), 'all_full_source_turns:' + rid)
                token_ids = tok.encode(prompt, add_special_tokens=False)
                check(token_ids == ids[rid] and len(token_ids) == row['prompt_tokens'], 'exact_prompt_token_ids:' + rid)
                check(len(token_ids) + 1024 <= 16384, 'no_context_overflow_or_truncation:' + rid)
                check(not set(v['display_to_content'].values()) & set(prompt.split()), 'no_private_content_ids_in_prompt:' + rid)
                generation_row = copy.deepcopy(row)
                generation_row['json_schema'] = relaxed_schema(row['json_schema'])
                params = a.sampling_params(generation_row, api)
                check(params.temperature == .8 and params.top_p == .9 and params.max_tokens == 1024 and params.seed == seed, 'actual_SamplingParams_values:' + rid)
                constraint = getattr(params, 'structured_outputs', None) or getattr(params, 'guided_decoding', None)
                check(constraint is not None and constraint.choice is None and constraint.json == relaxed_schema(row['json_schema']), 'actual_JSON_only_backend_without_uniqueItems:' + rid)
                params_to_verify = params
                pair_rows[arm] = row
            left = {k: v for k, v in pair_rows['control'].items() if k not in {'probe_arm', 'request_id', 'prompt_sha256', 'prompt_tokens'}}
            right = {k: v for k, v in pair_rows['treatment'].items() if k not in {'probe_arm', 'request_id', 'prompt_sha256', 'prompt_tokens'}}
            check(left == right, 'all_other_pair_metadata_identical:' + pair_id)
    check(rids == expected_order, 'complete_prefrozen_cohort_question_variant_arm_order')
    expected_pairs = []
    for pair_id, pair_rows in pairs.items():
        first = pair_rows[0]
        expected_pair = {k: first[k] for k in ['pair_id', 'candidate_id', 'root_id', 'cohort', 'seed', 'variant_id']}
        for row in pair_rows:
            expected_pair[row['probe_arm'] + '_request_id'] = row['request_id']
        expected_pairs.append(expected_pair)
    check(pairs_file == expected_pairs, 'matched_pair_metadata_exact_and_complete')
    check(m['max_prompt_tokens'] == max(r['prompt_tokens'] for r in rows), 'maximum_prompt_tokens_correct')
    check(m['minimum_context_headroom'] == min(16384 - r['prompt_tokens'] - 1024 for r in rows), 'context_headroom_correct')

    # Use the real vLLM public validation and grammar compiler without loading model weights.
    from vllm.config import ModelConfig, StructuredOutputsConfig
    import xgrammar as xgr
    import jsonschema
    model_config = ModelConfig(model=m['model'], tokenizer=m['tokenizer'], max_model_len=16384, trust_remote_code=True)
    structured_config = StructuredOutputsConfig(backend='auto')
    params_to_verify.verify(model_config=model_config, speculative_config=None, structured_outputs_config=structured_config, tokenizer=tok)
    check(True, 'actual_vLLM_SamplingParams_verify_no_LLM')
    schema = a.evidence_schema(qs[0])
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok, vocab_size=model_config.get_vocab_size()), max_threads=4).compile_json_schema(relaxed_schema(schema))
    good = {'evidence': [{'label': k, 'status': 'supported', 'turn_ref': 'S01:T001', 'quote': 'Example quote.'} for k in qs[0]['items']], 'answer': list(qs[0]['items'])}
    check(xgr.GrammarMatcher(grammar).accept_string(json.dumps(good)), 'compiled_JSON_grammar_accepts_unique_answer')
    jsonschema.validate(good, schema)
    duplicate = copy.deepcopy(good)
    duplicate['answer'][-1] = duplicate['answer'][0]
    try:
        jsonschema.validate(duplicate, schema)
    except jsonschema.ValidationError:
        check(True, 'strict_schema_rejects_duplicate_answer')
    else:
        check(False, 'strict_schema_rejects_duplicate_answer')
    duplicate_evidence = copy.deepcopy(good)
    duplicate_evidence['evidence'][-1]['label'] = duplicate_evidence['evidence'][0]['label']
    check(len({e['label'] for e in duplicate_evidence['evidence']}) != len(good['evidence']), 'unique_complete_evidence_posthoc_gate_required')
    strict_score_source = (DAY2 / 'solver/run.py').read_text()
    check('jsonschema.validate(structured,row[\'json_schema\'])' in strict_score_source and "'DUPLICATE_EVIDENCE_LABEL'" in strict_score_source, 'frozen_runtime_score_still_strict')
    check(not (run / 'STARTED.json').exists(), 'still_not_started_at_audit_completion')
    output = {'status': 'PASS', 'auditor': '/root/order_probe_design',
              'review_scope': 'Independent public-input reconstruction and actual CPU grammar checks; shared filesystem/model, no model calls.',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': len(names), 'check_names': names,
              'new_solver_calls': 0, 'model_weights_loaded': False, 'private_gold_json_read_by_this_audit': False,
              'historical_raw_outputs_or_scores_json_read_by_this_audit': False,
              'historical_seal_files_byte_hashed': True, 'old_day1_day2_files_modified': False,
              'questions': 9, 'roots': 9, 'pairs': 54, 'requests': 108,
              'cohorts': {'discovery_roots': sorted(DISCOVERY_ROOTS), 'replication_roots': sorted(q['root_id'] for q in replication)},
              'audited_files': [bind(run / 'PREFLIGHT_FREEZE.json'), bind(Path(__file__)), bind(protocol_path)],
              'actual_vllm_verify': True, 'actual_grammar_compile': True, 'strict_schema_preserved': True}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open('x') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({k: output[k] for k in ['status', 'checks', 'questions', 'pairs', 'requests']}), flush=True)


if __name__ == '__main__':
    main()
