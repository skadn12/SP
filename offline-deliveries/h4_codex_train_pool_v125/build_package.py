"""Reproduce the v125 training package without generation, scoring, or training."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SEED = 20261018
MAX_LENGTH = 12288
INPUT_NAMES = [
    'h4_codex_fullscale_ab_v118/delivery/all/SELECTED_CANDIDATES.json',
    'h4_codex_scene_retry_v120/delivery/TRAINING_CANDIDATES_V1.json',
    'h4_codex_all_failed_repair_v121/delivery/TRAINING_CANDIDATES_V1.json',
    'h4_codex_remaining_six_repair_v122/delivery/TRAINING_CANDIDATES_V1.json',
]
OLD = REPO / 'restart-n5-runs/h4_round0_b0_matched_training_v1_live_2'
SPLIT = REPO / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl'
HELDOUT = REPO / 'offline-deliveries/h4_codex_dev_test_ab_v124'
BASE = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter'
NATIVE = REPO / 'current_method_github'
TRAINER = NATIVE / 'train_qwen3_lora_sft.py'
DISPLAY = REPO / 'offline-deliveries/h4_codex_remaining_six_repair_v122/evaluations/paired_v1/adapter_protocol.py'
SYSTEM = ('You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. '
          'Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, reasoning, or any text outside the box.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def writel(name, rows):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(canonical(row) + '\n' for row in rows))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def distribution(values):
    s = sorted(values)
    return {'min': s[0], 'mean': round(statistics.mean(s), 3),
            'median': statistics.median(s), 'p95': s[math.ceil(len(s) * .95) - 1], 'max': s[-1]}


def main():
    if (HERE / 'COMPLETE_MANIFEST.json').exists():
        raise RuntimeError('PACKAGE_ALREADY_SEALED: reproduce into a separate directory')
    dep = {}

    def bind(path, expected=None):
        p = Path(path).resolve()
        # Audit reports retain their temporary creation paths. Resolve an exact,
        # hash-identical delivered copy instead of depending on /tmp at training time.
        if str(p) == '/tmp/v125_evidence_overlay.json':
            p = HERE / 'evidence/EVIDENCE_OVERLAY.json'
        elif p.is_relative_to('/tmp'):
            raise ValueError('TEMPORARY_DEPENDENCY_HAS_NO_PERSISTENT_BINDING:' + str(p))
        if str(p) not in dep:
            dep[str(p)] = {'path': str(p), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
        if expected and dep[str(p)]['sha256'] != expected:
            raise ValueError('DEPENDENCY_SHA_MISMATCH:' + str(p))
        return dep[str(p)]

    def bindings(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and (key == 'path' or key.endswith('_path')):
                    sha_key = 'sha256' if key == 'path' else key[:-5] + '_sha256'
                    expected = obj.get(sha_key)
                    p = Path(value)
                    if isinstance(expected, str) and len(expected) == 64 and p.is_absolute() and p.is_file():
                        bind(p, expected)
                bindings(value)
        elif isinstance(obj, list):
            for value in obj:
                bindings(value)

    for name in ['CANDIDATE_SOURCE_AUDIT.json', 'ENGINEERING_AUDIT.json', 'SPLIT_REPLAY_AUDIT.json']:
        bindings(read(HERE / 'evidence' / name))
    for row in read(HERE / 'evidence/SOURCE_DEPENDENCY_BINDINGS.json')['files']:
        bind(row['path'], row['sha256'])

    inputs = [REPO / 'offline-deliveries' / name for name in INPUT_NAMES]
    envelopes = [read(p) for p in inputs]
    for p, obj in zip(inputs, envelopes):
        bind(p)
        bindings(obj)
        batch = next(parent for parent in p.parents if (parent / 'COMPLETE_MANIFEST.json').is_file())
        bind(batch / 'COMPLETE_MANIFEST.json')
    raw = [copy.deepcopy(q) for envelope in envelopes for q in envelope['questions']]
    assert len(raw) == 363 and len({q['candidate_id'] for q in raw}) == 363
    write('data/RAW_SELECTED_QUESTIONS.json', {'questions': raw})
    questions = copy.deepcopy(raw)
    qmap = {q['candidate_id']: q for q in questions}
    overlay_path = HERE / 'evidence/EVIDENCE_OVERLAY.json'
    overlay = read(overlay_path)
    # Overlay schema is intentionally explicit; only evidence metadata may change.
    edits = overlay['corrections']
    assert len(edits) == 4
    for edit in edits:
        q = qmap[edit['candidate_id']]
        assert q['option_evidence'][edit['label']] == edit['original_evidence']
        q['option_evidence'][edit['label']] = copy.deepcopy(edit['effective_evidence'])
    bindings(overlay)
    for old, new in zip(raw, questions):
        assert {k: v for k, v in old.items() if k != 'option_evidence'} == {k: v for k, v in new.items() if k != 'option_evidence'}
    write('data/COMBINED_QUESTIONS.json', {'schema_version': 'v125_effective_evidence_v1', 'questions': questions,
          'evidence_overlay': 'evidence/EVIDENCE_OVERLAY.json', 'original_records': 'data/RAW_SELECTED_QUESTIONS.json'})

    split_rows = readl(SPLIT)
    splits = {r['root_id']: r['split'] for r in split_rows}
    bind(SPLIT)
    heldout = read(HELDOUT / 'MANIFEST.json')
    heldout_roots = {r['root_id'] for r in heldout['roots']}
    for name in ['MANIFEST.json', 'COMPLETE_MANIFEST.json', 'MASTER_FREEZE.json', 'evidence/INDEPENDENT_FINAL_AUDIT.json']:
        bind(HELDOUT / name)
    train_roots = {q['root_id'] for q in questions}
    assert len(train_roots) == 68 and train_roots.isdisjoint(heldout_roots)
    assert all(splits.get(r) == 'train' for r in train_roots)
    assert all(splits.get(r) == r0['split'] for r0 in heldout['roots'] for r in [r0['root_id']])
    old_manifest = readl(OLD / 'm_clean_policy_training_manifest.jsonl')
    assert train_roots == {r['root_id'] for r in old_manifest}
    for name in ['m_clean_policy_training_manifest.jsonl', 'm_clean_policy_train.jsonl', 'final_dataset_report.json']:
        bind(OLD / name)
    for name in ['READ_FIRST_ZH.md', 'RECOMMENDED_CANDIDATES.json', 'COMPLETE_MANIFEST.json']:
        bind(REPO / 'offline-deliveries/h4_codex_wording_minfix_v114' / name)
    replay_budget = REPO / 'restart-n5-runs/h4_round1_question_bank_feedback_compiler_v1_live_13/solver_replay_future_budget_plan.json'
    replay_review = REPO / 'restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/final_train_ordering_manual_review.tsv'
    bind(replay_budget)
    bind(replay_review)
    write('REPLAY_DISPOSITION.json', {
        'replay_count': 0, 'replay_ratio': 0, 'status': 'DEFERRED_REQUIRES_SOURCE_FIRST_REVIEW',
        'previous_budget': read(replay_budget),
        'decision': 'Do not sample old data merely because it was used by M_clean or ranked eligible. Known incorrect, ambiguous and unsupported ordering labels remain there.',
        'scope': 'This package combines the four new selected batches only. It does not delete or repair historical training data.',
        'next_replay_work': 'Source-first review and exclusion of known-invalid questions; freeze a separately reviewed replay subset before changing this package.',
        'budget_binding': bind(replay_budget), 'known_label_review_binding': bind(replay_review),
        'old_training_effect': 'Continuation from M_clean still inherits historical training and any noise learned from it.',
    })

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    trainer = module(TRAINER, 'v125_native_trainer')
    display_module = module(DISPLAY, 'v125_native_gold_blind_display')
    bind(TRAINER)
    bind(DISPLAY)
    # Match the native structured system contract exactly.
    assert all(r['messages'][0]['content'] == SYSTEM for r in readl(OLD / 'm_clean_policy_train.jsonl') if r['metadata']['question_type'] != 'single_choice')
    model_identity_path = REPO / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/model_identity_report.json'
    model_identity = read(model_identity_path)
    bind(model_identity_path)
    for info in model_identity['base_model_identity']['files']:
        bind(BASE / info['name'], info['sha256'])
    model_info = model_identity['models']['M_clean_policy']
    bind(ADAPTER / 'adapter_config.json', model_info['adapter_config_sha256'])
    bind(ADAPTER / 'adapter_model.pt', model_info['adapter_state_sha256'])
    bind(NATIVE / 'scripts/current_method/restart_n5_spell/h4_round0_b0_training_smoke_v1/train_smoke_runner.py')
    bind(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_training_report.json')
    bind(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/training_recipe.json')
    assert read(REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_training_report.json')['status'] == 'PASS'

    sources, root_manifest = {}, defaultdict(list)
    rows, lineage, lengths = [], [], []
    seen_prompts, seen_semantic = set(), set()
    for i, q in enumerate(questions):
        original = raw[i]
        root = q['root_id']
        src = Path(q['source_path'])
        bind(src, q['source_sha256'])
        source = read(src)
        assert source['root_id'] == root and len({t['turn_ref'] for t in source['turns']}) == len(source['turns'])
        if root in sources:
            assert sources[root]['sha256'] == q['source_sha256']
        sources[root] = {'path': str(src), 'sha256': q['source_sha256'], 'source': source}
        qtype = q['question_type']
        field = 'options' if qtype == 'multi_select' else 'items'
        labels = list('ABCDEF') if qtype == 'multi_select' else ['I' + str(n) for n in range(1, 7)]
        assert list(q[field]) == labels and len(set(q['gold'])) == len(q['gold'])
        assert set(q['gold']) <= set(labels)
        assert 2 <= len(q['gold']) <= 5 if qtype == 'multi_select' else set(q['gold']) == set(labels)
        assert set(q['option_evidence' if qtype == 'multi_select' else 'item_evidence']) == set(labels)
        variant = display_module.make_variants(q, seed=SEED, count=1)[0]
        # Content identifiers and permutation are computed without consulting Gold.
        _, original_ids = display_module._content_ids(q, field)
        gold_ids = [original_ids[label] for label in q['gold']]
        inverse = {content: label for label, content in variant['display_to_content'].items()}
        displayed_gold = [label for label in labels if variant['display_to_content'][label] in gold_ids] if qtype == 'multi_select' else [inverse[content] for content in gold_ids]
        original_from_content = {v: k for k, v in original_ids.items()}
        mapping = {label: original_from_content[cid] for label, cid in variant['display_to_content'].items()}
        assert set(mapping) == set(labels) and set(mapping.values()) == set(labels)
        rebound = [mapping[label] for label in displayed_gold]
        assert (set(rebound) == set(q['gold'])) if qtype == 'multi_select' else (rebound == q['gold'])
        user = display_module.prompt(q, variant, source)
        target = '<box>' + ','.join(displayed_gold) + '</box>'
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': target}]
        record = {'messages': messages, 'candidate_id': q['candidate_id'], 'root_id': root,
                  'question_type': qtype, 'split': 'train', 'selection_role': 'new_selected',
                  'dependence_group_id': root + ':' + qtype, 'gold_cardinality': len(q['gold'])}
        full = tokenizer(trainer.render_chat(tokenizer, messages, False), add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and (not full or full[-1] != tokenizer.eos_token_id):
            full.append(tokenizer.eos_token_id)
        prompt = tokenizer(trainer.render_chat(tokenizer, messages[:-1], True), add_special_tokens=False).input_ids
        assert full[:len(prompt)] == prompt, 'CHAT_PROMPT_IS_NOT_FULL_PREFIX'
        assert len(full) <= MAX_LENGTH, 'TRAINING_WOULD_TRUNCATE:' + q['candidate_id']
        features = trainer.tokenize_record(tokenizer, record, MAX_LENGTH)
        assert features['input_ids'] == full and features['labels'][:len(prompt)] == [-100] * len(prompt)
        assert features['labels'][len(prompt):] == full[len(prompt):] and len(full) > len(prompt)
        prompt_fp = digest(messages[:-1])
        assert prompt_fp not in seen_prompts, 'EXACT_PROMPT_DUPLICATE'
        seen_prompts.add(prompt_fp)
        semantic = {'question': ' '.join(q['question'].casefold().split()), 'type': qtype,
                    'contents': sorted(' '.join(x.casefold().split()) for x in q[field].values()),
                    'gold_contents': sorted(q[field][x] for x in q['gold']) if qtype == 'multi_select' else [q[field][x] for x in q['gold']]}
        semantic_fp = digest(semantic)
        assert semantic_fp not in seen_semantic, 'EXACT_CONTENT_DUPLICATE'
        seen_semantic.add(semantic_fp)
        batch_index = next(j for j, e in enumerate(envelopes) if any(t['candidate_id'] == q['candidate_id'] for t in e['questions']))
        root_provenance = envelopes[batch_index].get('root_source_provenance', {}).get(root)
        lin = {'candidate_id': q['candidate_id'], 'root_id': root, 'question_type': qtype,
               'batch_input': bind(inputs[batch_index]), 'original_record_sha256': digest(original),
               'effective_record_sha256': digest(q), 'source_binding': bind(src),
               'root_source_provenance': root_provenance,
               'dependence_group_id': record['dependence_group_id'], 'display_to_canonical': mapping,
               'canonical_gold': q['gold'], 'displayed_gold': displayed_gold,
               'display_variant': variant, 'prompt_sha256': prompt_fp, 'messages_sha256': digest(messages),
               'semantic_content_sha256': semantic_fp, 'input_token_count': len(full),
               'prompt_token_count': len(prompt), 'supervised_token_count': len(full) - len(prompt),
               'overlay_applied': any(e['candidate_id'] == q['candidate_id'] for e in edits)}
        lineage.append(lin)
        lengths.append(lin)
        rows.append(record)
        root_manifest[root].append(q['candidate_id'])

    # Deterministic ordering, one record per admitted question, no permutation multiplication.
    rows.sort(key=lambda row: (digest([SEED, 'training_order', row['candidate_id']]), row['candidate_id']))
    writel('data/TRAIN_NEW_ONLY.jsonl', rows)
    writel('data/QUESTION_LINEAGE.jsonl', lineage)
    for root, record in sources.items():
        dst = HERE / 'sources' / (root + '.json')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record['path'], dst)
        assert sha(dst) == record['sha256']
    write('data/ROOT_MANIFEST.json', [{'root_id': root, 'split': 'train', 'source_binding': bind(sources[root]['path']),
          'packaged_source': 'sources/' + root + '.json', 'candidate_ids': sorted(ids),
          'historical_parent_seen_by_M_clean': True} for root, ids in sorted(root_manifest.items())])
    groups = defaultdict(list)
    for row in rows:
        groups[row['dependence_group_id']].append(row['candidate_id'])
    write('data/DEPENDENCE_GROUPS.json', {'grouping': 'Conservative root x question_type dependence groups, not a claim of semantic independence.',
          'groups': [{'group_id': k, 'candidate_ids': sorted(v)} for k, v in sorted(groups.items())]})
    protocol = {
        'version': 'v125_training_pool_combination_v1', 'seed': SEED,
        'inputs': [bind(p) for p in inputs], 'new_count': 363, 'root_count': 68,
        'selection': 'Preserve frozen native selection including v118 mandatory v114 quarantine/overlay; no solver rerun, re-selection or score-guided rewrite.',
        'canonical_public_questions_and_gold_unchanged': True,
        'evidence_policy': 'Original records retained losslessly; four reviewed citation omissions supplemented in effective evidence only, with explicit overlay.',
        'replay': {'count': 0, 'status': 'DEFERRED_KNOWN_OLD_GOLD_PROBLEMS', 'default_budget_ratio': .1},
        'display': {'algorithm_binding': bind(DISPLAY), 'function': 'make_variants(seed=20261018,count=1)',
                    'gold_blind': True, 'per_candidate_training_rows': 1, 'mapping_logged': True},
        'rendering': {'system': SYSTEM, 'user': 'Frozen v122 adapter_protocol.prompt, whole source dialogue and public fields only',
                      'assistant': '<box>displayed comma-separated Gold labels</box>', 'evidence_or_review_metadata_in_prompt': False},
        'length': {'max_length': MAX_LENGTH, 'truncation_allowed': False, 'assistant_only_supervision': True,
                   'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()},
        'split': {'authority': bind(SPLIT), 'heldout_population_binding': bind(HELDOUT / 'MANIFEST.json'),
                  'heldout_root_count': 31, 'heldout_complete_manifest_binding': bind(HELDOUT / 'COMPLETE_MANIFEST.json'),
                  'dev_or_test_training_records': 0},
        'limitations': ['68 train parents were already present in first-round M_clean training; these are new expansions/questions, not unseen roots.',
                       'Shared-session auditors do not constitute external semantic certification.',
                       'Continuation inherits historical M_clean label noise; zero replay avoids adding it again but does not erase it.',
                       'No training or new solver evaluation is performed by this package build.'],
    }
    write('PROTOCOL.json', protocol)
    output_dir = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'
    command = [sys.executable, '-m', 'scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.train_smoke_runner',
        '--model', str(BASE), '--tokenizer', str(BASE), '--init-adapter-dir', str(ADAPTER),
        '--train-file', str(HERE / 'data/TRAIN_NEW_ONLY.jsonl'), '--output-dir', str(output_dir / 'adapter'),
        '--max-length', str(MAX_LENGTH), '--epochs', '2.0', '--batch-size', '1', '--grad-accum', '4', '--lr', '0.0001',
        '--min-lr-ratio', '0.1', '--warmup-ratio', '0.05', '--weight-decay', '0.0', '--max-grad-norm', '1.0',
        '--lora-rank', '16', '--lora-alpha', '32', '--lora-dropout', '0.05',
        '--target-modules', 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj', '--seed', str(SEED)]
    write('TRAINING_PLAN.json', {'status': 'PREPARED_NOT_RUN', 'training_started': False,
          'track': 'Continue M_clean custom LoRA from its actual clean_adapter on Qwen3-4B; reset optimizer/scheduler.',
          'cwd': str(NATIVE), 'argv': command, 'run_output_directory': str(output_dir),
          'environment': {'SMOKE_GRADIENT_AUDIT_PATH': str(output_dir / 'gradient_audit.json')},
          'configuration_note': 'Inherit first-round training hyperparameters except max_length=12288, initialization from M_clean and new seed/data. This new-only continuation is not a 10% replay experiment.',
          'validation_file': None, 'validation_policy': 'Keep heldout data sealed; perform standardized solver evaluation separately after training, with dev for model selection and test for final reporting.',
          'hardware_note': 'Tokenization and adapter compatibility verified without loading model weights. GPU memory and training stability are not yet tested at 12288.',
          'data_binding': {'path': str(HERE / 'data/TRAIN_NEW_ONLY.jsonl'), 'sha256': sha(HERE / 'data/TRAIN_NEW_ONLY.jsonl')}})
    summary = {'status': 'DATA_VALIDATED_AWAITING_INDEPENDENT_PACKAGE_AUDIT', 'training_started': False,
        'new_question_count': len(rows), 'replay_question_count': 0, 'training_record_count': len(rows),
        'root_count': len(train_roots), 'question_type_counts': dict(Counter(q['question_type'] for q in questions)),
        'multi_select_gold_counts': dict(sorted(Counter(str(len(q['gold'])) for q in questions if q['question_type'] == 'multi_select').items())),
        'source_batches': [{'path': str(p), 'count': len(e['questions']), 'roots': len({q['root_id'] for q in e['questions']})} for p, e in zip(inputs, envelopes)],
        'input_tokens': distribution([l['input_token_count'] for l in lengths]),
        'dialogue_tokens': distribution([len(tokenizer.encode('\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in s['source']['turns']), add_special_tokens=False)) for s in sources.values()]),
        'over_old_8192_count': sum(l['input_token_count'] > 8192 for l in lengths),
        'over_configured_limit_count': sum(l['input_token_count'] > MAX_LENGTH for l in lengths),
        'min_supervised_tokens': min(l['supervised_token_count'] for l in lengths),
        'max_length': MAX_LENGTH, 'heldout_roots_overlap': [], 'exact_duplicate_ids': 0,
        'exact_duplicate_public_content': 0, 'exact_duplicate_prompts': 0,
        'metadata_overlay_fields': len(edits), 'metadata_overlay_questions': len({e['candidate_id'] for e in edits}),
        'synthetic_source_repair_question_count_v121_v122': sum((l['root_source_provenance'] or {}).get('source_status') == 'EDITED_SYNTHETIC' for l in lineage),
    }
    write('SUMMARY.json', summary)
    # External bindings are frozen after every dependency has been registered.
    write('INPUT_FREEZE.json', {'version': 'v125_external_inputs_v1', 'files': sorted(dep.values(), key=lambda r: r['path']),
          'external_file_count': len(dep), 'immutable_input_policy': True})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
