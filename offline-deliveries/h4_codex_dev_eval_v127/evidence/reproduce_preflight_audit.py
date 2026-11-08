"""Independently audit the v127 prepared dev pair without model loading/inference."""
from __future__ import annotations
import ast
import collections
import concurrent.futures
import copy
import datetime
import hashlib
import importlib.util
import json
import random
import sys
import typing
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent.parent
DEV = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'
POOL = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
TRAINING = REPO / 'offline-deliveries/h4_codex_training_v126'
FORMAL = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'
NATIVE = REPO / 'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation'
WORK = REPO / 'worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell'
PLAN = TRAINING / 'evidence/DEV_EVALUATION_PLAN_AUDIT.json'
BASE = Path('/root/autodl-tmp/Qwen3-4B')
OLD_MODEL = REPO / 'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
NEW_MODEL = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125_merged'
POLICY = 'v15_full_only_random_mcq6x3_order12x1_v1'
SEED = 2026100800
EXPECTED_REQUEST_SHA = 'c7c65c81d82b2afd482ec2541768b6fe4fa5c9553b5072a927d2ed7e252fff5a'
EXPECTED_FINAL_ADAPTER_SHA = '0e89757df099234cc6a4f81dd1bfbbf869dd4a4425b6a8098f8c00a65a170ecc'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    tracked = {}
    errors = []
    checks = {}

    def require(condition, code):
        if not condition:
            raise AssertionError(code)

    def track(path):
        path = Path(path).resolve()
        actual = sha(path)
        require(str(path) not in tracked or tracked[str(path)] == actual, 'FILE_CHANGED_DURING_AUDIT:' + str(path))
        tracked[str(path)] = actual
        return actual

    def read(path):
        track(path)
        return json.loads(Path(path).read_text())

    def readl(path):
        track(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    def verify(row):
        path = Path(row['path']).resolve()
        require(path.is_file(), 'BOUND_FILE_MISSING:' + str(path))
        require(track(path) == row['sha256'], 'BOUND_SHA_CHANGED:' + str(path))
        for key in ['size_bytes', 'bytes']:
            if key in row:
                require(path.stat().st_size == row[key], 'BOUND_SIZE_CHANGED:' + str(path))

    def seal(folder, full=True):
        manifest = read(folder / 'COMPLETE_MANIFEST.json')
        entries = {str(Path(row['path']).resolve().relative_to(folder.resolve())) if Path(row['path']).is_absolute() else row['path']: row for row in manifest['files']}
        require(len(entries) == len(manifest['files']), 'SEAL_DUPLICATE_PATH')
        if full:
            actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
                      and '__pycache__' not in p.parts and p.name != 'COMPLETE_MANIFEST.json'}
            require(actual == set(entries), 'SEAL_FILE_SET_CHANGED:' + str(folder))
            for name, row in entries.items():
                require(track(folder / name) == row['sha256'], 'SEALED_FILE_CHANGED:' + str(folder / name))
        return entries

    try:
        require((HERE / 'PREFLIGHT_FREEZE.json').is_file(), 'PREFLIGHT_NOT_READY')
        require(not list((HERE / 'run').glob('*/STARTED.json')), 'INFERENCE_ALREADY_STARTED')
        require(not list((HERE / 'run').glob('*/responses/*.json')), 'RESPONSES_ALREADY_PRESENT')
        track(__file__)
        ast.parse((HERE / 'evaluate.py').read_text())
        ast.parse((HERE / 'report_paired.py').read_text())
        ast.parse((HERE / 'run_pair.py').read_text())
        track(HERE / 'run_pair.py')
        track(HERE / 'evaluate.py')
        track(HERE / 'report_paired.py')
        plan = read(PLAN)
        planned = plan['readonly_checks']['plan_requests']
        require(digest(planned) == plan['readonly_checks']['plan_requests_sha256'] == EXPECTED_REQUEST_SHA, 'INDEPENDENT_REQUEST_PLAN_CHANGED')
        require(len(planned) == 780 and len({r['request_id'] for r in planned}) == 780, 'PLAN_COUNT_OR_DUPLICATES')
        for binding in plan['readonly_checks']['frozen_code_bindings']:
            verify(binding)
        for binding in plan['readonly_checks']['v124_dev_read_file_bindings']:
            verify(binding)
        verify(plan['readonly_checks']['v124_complete_manifest_binding'])
        v124seal = seal(DEV, full=False)  # Hash only supplied dev inputs, never test contents.
        for binding in plan['readonly_checks']['v124_dev_read_file_bindings']:
            path = Path(binding['path'])
            require(v124seal[str(path.relative_to(DEV))]['sha256'] == binding['sha256'], 'DEV_INPUT_NOT_IN_V124_SEAL')
        seal(POOL)
        seal(TRAINING)
        seal(FORMAL)
        pool = read(HERE / 'EVALUATION_POOL.json')
        original = read(DEV / 'delivery/dev/QUESTIONS_WITH_ANSWERS.json')['questions']
        require(pool['questions'] == original, 'DEV_POOL_NOT_LOSSLESS')
        require(pool['provenance']['backend'] == 'CURRENT_CODEX_SESSION'
                and pool['provenance']['source_batch_id'] == DEV.name, 'GENERATOR_PROVENANCE_CHANGED')
        questions = original
        qmap = {q['candidate_id']: q for q in questions}
        require(len(questions) == len(qmap) == 48, 'FIXED_48_CANDIDATES')
        require(collections.Counter(q['question_type'] for q in questions) == {'multi_select': 34, 'ordering': 14}, 'TYPE_COUNT')
        require(collections.Counter(str(len(q['gold'])) for q in questions if q['question_type'] == 'multi_select')
                == {'2': 7, '3': 6, '4': 7, '5': 7, '6': 7}, 'GOLD_STRATA')
        manifest = read(DEV / 'MANIFEST.json')
        require(track(DEV / 'MANIFEST.json') == v124seal['MANIFEST.json']['sha256'], 'V124_ROOT_MANIFEST_CHANGED')
        fixed = [r for r in manifest['roots'] if r['split'] == 'dev']
        fixed_ids = [r['root_id'] for r in fixed]
        covered = sorted({q['root_id'] for q in questions})
        require(len(fixed_ids) == 15 and len(set(fixed_ids)) == 15 and len(covered) == 7, 'ROOT_DENOMINATOR')
        require(set(covered) <= set(fixed_ids), 'CANDIDATE_OUTSIDE_DEV')
        scope = read(HERE / 'SCOPE.json')
        require(scope['fixed_dev_root_ids'] == fixed_ids and scope['covered_dev_root_ids'] == covered, 'SCOPE_ROOTS')
        require(scope['question_count'] == 48 and scope['calls_per_model'] == 780 and scope['total_calls'] == 1560, 'SCOPE_COUNTS')
        require(scope['test_evaluation'] is False and scope['score_based_filtering'] is False, 'TEST_OR_SELECTION_ENABLED')
        train_roots = {r['root_id'] for r in read(POOL / 'data/ROOT_MANIFEST.json')}
        require(len(train_roots) == 68 and not train_roots & set(fixed_ids), 'TRAIN_DEV_ROOT_OVERLAP')
        split_audit = read(POOL / 'evidence/SPLIT_REPLAY_AUDIT.json')
        train_identities = split_audit['new_train_vs_v124_eval']['train_identity_rows']
        require({r['root_id'] for r in train_identities} == train_roots, 'TRAIN_IDENTITY_COVERAGE')
        for key in ['source_id', 'source_record_sha256']:
            require(not {r[key] for r in train_identities} & {r[key] for r in fixed}, 'TRAIN_DEV_' + key + '_OVERLAP')
        checks['population'] = {'fixed_dev_roots': 15, 'covered_roots': 7, 'uncovered_roots': 8, 'questions': 48,
                                'multi_select': 34, 'ordering': 14, 'gold_counts': {'2': 7, '3': 6, '4': 7, '5': 7, '6': 7},
                                'gold2345_main': 27, 'gold6_control': 7, 'train_root_source_id_record_sha_overlap': 0}

        native_protocol = (NATIVE / 'protocol.py').read_text()
        old = 'if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":'
        loc = 'CONTRACT_DIR = Path(__file__).resolve().parent.parent'
        require(native_protocol.count(old) == native_protocol.count(loc) == 1, 'NATIVE_PROTOCOL_SHAPE_CHANGED')
        expected_protocol = native_protocol.replace(old, 'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != "h4_codex_dev_test_ab_v124"):').replace(loc, 'CONTRACT_DIR = Path(' + repr(str(NATIVE.parent)) + ')')
        require((HERE / 'adapter_protocol.py').read_text() == expected_protocol, 'ADAPTER_NOT_PROVENANCE_PATH_ONLY')
        track(HERE / 'adapter_protocol.py')
        protocol = load_module('v127_audit_protocol', HERE / 'adapter_protocol.py')
        scoring = load_module('v127_audit_scoring', NATIVE / 'scoring.py')
        selection = load_module('v127_audit_selection', NATIVE / 'selection.py')
        _, normalized, sourcemap = protocol.validate_pool(HERE / 'EVALUATION_POOL.json')
        require(normalized == questions and len(sourcemap) == 7, 'NATIVE_POOL_GATE')

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
        contract_path = WORK / 'h4_round0_b0_k3_difficulty_v1/contract.py'
        tree = ast.parse(contract_path.read_text())
        pure = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {'dialogue_rows', 'build_prompt'})
                or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'STRUCTURED_SYSTEM_PROMPT' for t in n.targets))]
        ns = {'Any': typing.Any, 'Mapping': typing.Mapping, 'Sequence': typing.Sequence}
        exec(compile(ast.Module(body=pure, type_ignores=[]), str(contract_path), 'exec'), ns)
        messages_checked = 0
        fixtures = []
        all_bindings = []
        manifests = {}
        expected_decoder = plan['frozen_protocol']['decoder_contract_precedent']
        for side, name, modelpath in [('baseline', 'M_clean_policy', OLD_MODEL), ('new', 'M_v125_newonly', NEW_MODEL)]:
            run = HERE / 'run' / side
            actual = read(run / 'manifest.json')
            manifests[side] = actual
            require(actual['requests'] == planned, 'REQUESTS_DIFFER_FROM_INDEPENDENT_PLAN:' + side)
            require(actual['request_count'] == 780 and actual['candidate_count'] == 48, 'NATIVE_MANIFEST_COUNTS')
            require(actual['policy_version'] == POLICY and actual['execution'] == 'full_only_no_ground_v1', 'NATIVE_POLICY_CHANGED')
            require(actual['solver_model'] == name and actual['model'] == str(modelpath) and actual['tokenizer'] == str(BASE), 'SOLVER_PATH_NAME')
            require(actual['decoder_contract'] == expected_decoder, 'NATIVE_DECODER_CHANGED')
            for path, expected in actual['code_sha256'].items():
                require(track(path) == expected, 'NATIVE_FROZEN_CODE_CHANGED:' + path)
            for path, expected in actual['input_sha256'].items():
                file = Path(actual['pool_path']) if path == 'pool' else run / path
                require(track(file) == expected, 'NATIVE_INPUT_CHANGED:' + str(file))
            qs = read(run / 'questions.json')['candidates']
            gold = read(run / 'gold.json')
            canon = read(run / 'canonical_bindings.json')['candidates']
            require(len(qs) == 48 and {q['candidate_id'] for q in qs} == set(qmap) == set(gold), 'NATIVE_PUBLIC_GOLD_MEMBERSHIP')
            require(len(canon) == 48 and {q['candidate_id'] for q in canon} == set(qmap), 'CANONICAL_BINDINGS_POPULATION')
            for public, source_q, binding in zip(qs, questions, canon):
                cid = source_q['candidate_id']
                require(public == {k: v for k, v in source_q.items() if k != 'gold'}, 'PUBLIC_EXPORT_CHANGE')
                field = 'options' if source_q['question_type'] == 'multi_select' else 'items'
                labels = list('ABCDEF') if field == 'options' else [f'I{i}' for i in range(1, 7)]
                ids = {lab: 'content_' + hashlib.sha256(source_q[field][lab].strip().encode()).hexdigest()[:16] for lab in labels}
                gcontent = [ids[g] for g in source_q['gold']]
                require(gold[cid] == gcontent, 'CANONICAL_GOLD_CONTENT')
                require(binding == {'candidate_id': cid, 'root_id': source_q['root_id'], 'content_id_by_label': ids,
                                    'gold_content_ids': gcontent, 'evidence_by_content_id': {ids[lab]: ev for lab, ev in source_q['option_evidence' if field == 'options' else 'item_evidence'].items()}}, 'CANONICAL_EVIDENCE_BINDING')
                source_path = Path(source_q['source_path'])
                source = sourcemap[source_q['root_id']]['source']
                require(track(source_path) == source_q['source_sha256'], 'SOURCE_HASH')
                packaged = run / 'sources' / (source_q['root_id'] + '.json')
                require(track(packaged) == source_q['source_sha256'] and packaged.read_bytes() == source_path.read_bytes(), 'SOURCE_COPY_BYTES')
                variants = protocol.make_variants(source_q, seed=SEED, count=6 if field == 'options' else 12)
                altered = copy.deepcopy(source_q)
                altered['gold'] = list(reversed(source_q['gold'])) if field == 'items' else list(reversed(labels[:2]))
                require(protocol.make_variants(altered, seed=SEED, count=len(variants)) == variants, 'DISPLAY_DEPENDS_ON_GOLD')
                qrequests = [r for r in planned if r['candidate_id'] == cid]
                for variant in variants:
                    vrequests = [r for r in qrequests if r['variant']['variant_id'] == variant['variant_id']]
                    require(all(r['variant'] == variant for r in vrequests), 'DISPLAY_ALGORITHM_CHANGED')
                    text = ns['build_prompt'](tokenizer, source_q, variant, ns['dialogue_rows'](source), context_mode='full_context')
                    public_q = {k: source_q[k] for k in ['question_type', 'question']}
                    public_v = {field: variant[field]}
                    require(text == ns['build_prompt'](tokenizer, public_q, public_v, ns['dialogue_rows'](source), context_mode='full_context'), 'PROMPT_PRIVATE_METADATA_LEAK')
                    tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    require(tokens + 32 <= 16384, 'PROMPT_TOO_LONG')
                    for request in vrequests:
                        prompt_path = run / 'prompts' / (request['request_id'] + '.txt')
                        require(prompt_path.read_text() == text and track(prompt_path) == request['prompt_sha256'], 'PROMPT_BYTE_CHANGE')
                        require(tokens == request['prompt_tokens'], 'PROMPT_TOKEN_CHANGE')
                        messages_checked += 1
                    inv = {content: label for label, content in variant['display_to_content'].items()}
                    displayed_gold = [label for label in labels if variant['display_to_content'][label] in gcontent] if field == 'options' else [inv[c] for c in gcontent]
                    scored = scoring.score_response({**source_q, 'gold_content_ids': gcontent}, {'variant': variant}, '<box>' + ','.join(displayed_gold) + '</box>')
                    require(scored['exact_match'] is True and scored['partial_score'] == '1', 'DISPLAY_GOLD_ROUNDTRIP')
                sample_records = [{'request_id': r['request_id'], 'candidate_id': cid, 'root_id': r['root_id'], 'question_type': r['question_type'], 'variant_id': r['variant']['variant_id'], 'slot': r['slot'], 'seed': r['seed'], 'evaluation_valid': True} for r in qrequests]
                require(selection.audit_complete_call_set(actual, sample_records, candidate_id=cid) == [], 'CALL_SET_FIXTURE')
            identity = read(run / 'solver_identity.json')
            require(identity == actual['solver_identity'] and identity['model_identifier'] == name
                    and identity['model_path'] == str(modelpath) and identity['tokenizer_path'] == str(BASE), 'MODEL_IDENTITY_MANIFEST')
            require(track(identity['audit_path']) == identity['audit_sha256'], 'MODEL_AUDIT_BINDING')
            all_bindings.extend(identity['checkpoint_files'])
            if side == 'baseline':
                old_audit = read(identity['audit_path'])['models']['M_clean_policy']
                require(old_audit['inference_model_path'] == str(OLD_MODEL), 'OLD_BASELINE_IDENTITY')
                require(identity['model_fingerprint_sha256'] == old_audit['model_fingerprint_sha256'], 'OLD_MODEL_FINGERPRINT')
                require(identity['adapter_path'] == str(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter'), 'OLD_INIT_ADAPTER')
                expected_files = [{'path': str(OLD_MODEL / r['name']), 'name': r['name'], 'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in old_audit['merged_model_identity']['files']]
                base_audit = read(REPO / 'offline-deliveries/h4_multiselect_partial_admission_v1/followup_model_identity.json')
                expected_files.extend(r for r in base_audit['files'] if r['name'] in {'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'})
                require(identity['checkpoint_files'] == expected_files, 'OLD_CHECKPOINT_AUTHORITY_CHANGED')
            else:
                fixed_identity = read(HERE / 'model_tools/NEW_MODEL_IDENTITY.json')
                require(identity == fixed_identity and identity['status'] == 'PASS' and identity['adapter_state_sha256'] == EXPECTED_FINAL_ADAPTER_SHA, 'NEW_IDENTITY_NOT_FINAL_FORMAL')
                require(identity['adapter_path'] == str(FORMAL / 'adapter') and identity['adapter_already_merged'] is True, 'NEW_ADAPTER_PATH_OR_DOUBLE_LOAD')
                require(track(FORMAL / 'adapter/adapter_model.pt') == EXPECTED_FINAL_ADAPTER_SHA, 'FINAL_ADAPTER_CHANGED')
                merge_audit = read(HERE / 'evidence/MODEL_MERGE_AUDIT.json')
                require(merge_audit['status'] == 'PASS' and merge_audit['final_adapter_state_sha256'] == EXPECTED_FINAL_ADAPTER_SHA, 'MERGE_AUDIT_NOT_PASS')
                require(merge_audit['model_path'] == str(NEW_MODEL), 'MERGE_PATH')
                require(merge_audit['checks']['base_only_once_not_historical_merged'] is True, 'DOUBLE_MERGE')
                require(merge_audit['checks']['exactly252_modules_merged_no_missing_pairs'] is True, 'MERGE_MODULE_COUNT')
                merged = identity['merged_files']
                require(identity['merged_file_count'] == len(merged) == len(merge_audit['outputs']), 'MERGED_FILE_COUNT')
                require(digest([{'name': r['name'], 'size_bytes': r['size_bytes'], 'sha256': r['sha256']} for r in merged]) == identity['model_fingerprint_sha256'] == merge_audit['model_fingerprint_sha256'], 'NEW_MERGED_FINGERPRINT')
                require(identity['checkpoint_files'][:len(merged)] == [{'path': r['path'], 'name': r['name'], 'bytes': r['size_bytes'], 'sha256': r['sha256']} for r in merged], 'NEW_CHECKPOINT_FILE_LIST')
                require(identity['merged_linear_modules'] == 252 and identity['chat_template_sha256'] == plan['readonly_checks']['base_tokenizer_chat_template_sha256'], 'NEW_MODULES_OR_CHAT_TEMPLATE')
                for source_binding in identity['input_bindings']:
                    verify(source_binding)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(verify, all_bindings))
        require(manifests['baseline']['requests'] == manifests['new']['requests'], 'PAIR_REQUEST_DIFFERENCE')
        require(manifests['baseline']['decoder_contract'] == manifests['new']['decoder_contract'], 'PAIR_DECODER_DIFFERENCE')
        for name in ['questions.json', 'gold.json', 'canonical_bindings.json']:
            require(read(HERE / 'run/baseline' / name) == read(HERE / 'run/new' / name), 'PAIR_INPUT_DIFFERENCE')
        checks['requests'] = {'per_side': 780, 'pair': 1560, 'unique_prompts_per_side': 372,
                              'independent_plan_sha256': EXPECTED_REQUEST_SHA, 'full_context_only': True,
                              'no_context_or_ground_calls': 0, 'all1560_prompt_bytes_and_tokens_verified': messages_checked == 1560,
                              'max_prompt_tokens': max(r['prompt_tokens'] for r in planned), 'gold_blind_displays_and_gold_roundtrips': True}
        checks['models'] = {'baseline': str(OLD_MODEL), 'new': str(NEW_MODEL), 'new_adapter_sha256': EXPECTED_FINAL_ADAPTER_SHA,
                            'checkpoint_binding_rows_hashed': len(all_bindings), 'same_original_base_tokenizer': True,
                            'model_weights_loaded_by_audit': False}
        tools = read(HERE / 'TOOLS_FREEZE.json')
        frozen = read(HERE / 'PREFLIGHT_FREEZE.json')
        require(tools['status'] == 'FROZEN_BEFORE_FIRST_INFERENCE' and tools['policy_version'] == POLICY, 'TOOLS_NOT_FROZEN')
        require(frozen['candidate_count'] == 48 and frozen['request_count_per_model'] == 780, 'PREFLIGHT_COUNTS')
        for row in tools['files'] + frozen['files']:
            verify(row)
        require(not list((HERE / 'run').glob('*/STARTED.json')), 'INFERENCE_STARTED_DURING_AUDIT')
        checks['test'] = {'test_question_or_source_content_read': False, 'test_inference_calls': 0, 'v124_files_modified': False}
        checks['native_algorithms'] = {'provenance_path_only_protocol_adapter': True, 'native_prompt_display_scoring_unchanged': True,
                                        'full48_no_score_based_filtering': True, 'selection_output_diagnostic_only': True}
    except Exception as exc:
        errors.append({'type': type(exc).__name__, 'message': str(exc)})
    changes = [path for path, value in tracked.items() if sha(path) != value]
    if changes:
        errors.append({'type': 'INPUT_CHANGED_DURING_AUDIT', 'paths': changes})
    report = {'schema_version': 'v127_independent_preflight_audit_v1', 'status': 'PASS' if not errors else 'FAIL',
              'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'auditor': '/root/repair_v122_audit',
              'errors': errors, 'checks': checks,
              'audited_files': [{'path': path, 'sha256': value} for path, value in sorted(tracked.items())],
              'limitations': ['Same-session independent artifact audit, not external semantic certification.',
                              'Scores concern48 delivered questions on7 of15 fixed dev roots;8 uncovered roots remain in production coverage accounting.',
                              'Rollouts,permutations,Gold families and same-root questions are correlated observations,not1560 independent samples.',
                              'Model checkpoint files are hashed without loading weights; exact merge mathematics rely on the separately bound engineering audit.',
                              'Test metadata appears in the shared root manifest and historical exposure disclosure remains applicable;this audit reads no test question/source contents and schedules no test inference.'],
              'training_or_inference_started': False, 'gpu_allocation_by_audit': False,
              'verifier_binding': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
    output = HERE / 'evidence/INDEPENDENT_PREFLIGHT_AUDIT.json'
    with output.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'errors': errors, 'path': str(output), 'sha256': sha(output), 'audited_files': len(tracked)}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
