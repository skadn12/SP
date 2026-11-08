"""Independent read-only v125 training-data audit; no weights, inference or training."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path('/root/autodl-tmp/gxy-selfplay')
PKG = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
BASE = Path('/root/autodl-tmp/Qwen3-4B')
TRAINER = REPO / 'current_method_github/train_qwen3_lora_sft.py'
DISPLAY = REPO / 'offline-deliveries/h4_codex_remaining_six_repair_v122/evaluations/paired_v1/adapter_protocol.py'
OLD = REPO / 'restart-n5-runs/h4_round0_b0_matched_training_v1_live_2'
ADAPTER = REPO / 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter'
SYSTEM = ('You are a strict structured dialogue QA solver. Use only the supplied dialogue when it is present. '
          'Return the final answer in exactly one <box>...</box> pair. Do not provide explanations, reasoning, or any text outside the box.')


def read(path):
    return json.loads(Path(path).read_text())


def readl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dist(values):
    s = sorted(values)
    return {'min': min(s), 'mean': round(statistics.mean(s), 3), 'median': statistics.median(s),
            'p95': s[math.ceil(.95 * len(s)) - 1], 'max': max(s)}


def main():
    required = ['build_package.py', 'preflight.py', 'INPUT_FREEZE.json', 'PROTOCOL.json', 'TRAINING_PLAN.json',
                'REPLAY_DISPOSITION.json', 'data/RAW_SELECTED_QUESTIONS.json', 'data/COMBINED_QUESTIONS.json',
                'data/TRAIN_NEW_ONLY.jsonl', 'data/QUESTION_LINEAGE.jsonl', 'data/ROOT_MANIFEST.json',
                'data/DEPENDENCE_GROUPS.json', 'evidence/EVIDENCE_OVERLAY.json', 'evidence/ENGINEERING_AUDIT.json',
                'evidence/CANDIDATE_SOURCE_AUDIT.json', 'evidence/SPLIT_REPLAY_AUDIT.json',
                'evidence/reproduce_engineering_audit.py', 'evidence/SOURCE_DEPENDENCY_BINDINGS.json']
    assert (PKG / 'SUMMARY.json').is_file(), 'PACKAGE_NOT_READY'
    roots_manifest = read(PKG / 'data/ROOT_MANIFEST.json')
    required += [r['packaged_source'] for r in roots_manifest]
    before = {name: sha(PKG / name) for name in required}
    freeze = read(PKG / 'INPUT_FREEZE.json')
    external = {r['path']: r for r in freeze['files']}
    assert len(external) == freeze['external_file_count']
    source_dependencies = read(PKG / 'evidence/SOURCE_DEPENDENCY_BINDINGS.json')
    assert source_dependencies['status'] == 'PASS' and len(source_dependencies['files']) == source_dependencies['persistent_dependency_file_count'] == 954
    for binding in source_dependencies['files']:
        assert external[binding['path']]['sha256'] == binding['sha256'] == sha(binding['path']), 'SOURCE_AUDIT_DEPENDENCY_NOT_FROZEN'
    protocol = read(PKG / 'PROTOCOL.json')
    plan = read(PKG / 'TRAINING_PLAN.json')
    summary = read(PKG / 'SUMMARY.json')
    raw = read(PKG / 'data/RAW_SELECTED_QUESTIONS.json')['questions']
    effective = read(PKG / 'data/COMBINED_QUESTIONS.json')['questions']
    rows = readl(PKG / 'data/TRAIN_NEW_ONLY.jsonl')
    lineage = readl(PKG / 'data/QUESTION_LINEAGE.jsonl')
    assert len(raw) == len(effective) == len(rows) == len(lineage) == 363
    rmap = {r['candidate_id']: r for r in rows}
    lmap = {r['candidate_id']: r for r in lineage}
    qmap = {q['candidate_id']: q for q in effective}
    assert len(rmap) == len(lmap) == len(qmap) == 363 and set(rmap) == set(lmap) == set(qmap)
    original = []
    for binding in protocol['inputs']:
        assert sha(binding['path']) == binding['sha256'] == external[binding['path']]['sha256']
        original.extend(read(binding['path'])['questions'])
    assert raw == original, 'RAW_NOT_LOSSLESS'
    patched = copy.deepcopy(raw)
    pmap = {q['candidate_id']: q for q in patched}
    overlay = read(PKG / 'evidence/EVIDENCE_OVERLAY.json')['corrections']
    assert len(overlay) == 4 and len({e['candidate_id'] for e in overlay}) == 2
    for edit in overlay:
        assert pmap[edit['candidate_id']]['option_evidence'][edit['label']] == edit['original_evidence']
        pmap[edit['candidate_id']]['option_evidence'][edit['label']] = edit['effective_evidence']
    assert patched == effective, 'UNRECORDED_CANONICAL_CHANGE'
    for old, new in zip(raw, effective):
        assert {k: v for k, v in old.items() if k != 'option_evidence'} == {k: v for k, v in new.items() if k != 'option_evidence'}

    roots = {r['root_id'] for r in rows}
    split_path = protocol['split']['authority']['path']
    split = {r['root_id']: r['split'] for r in readl(split_path)}
    assert sha(split_path) == protocol['split']['authority']['sha256']
    heldout_path = protocol['split']['heldout_population_binding']['path']
    heldout = read(heldout_path)
    heldout_roots = {r['root_id'] for r in heldout['roots']}
    assert sha(heldout_path) == protocol['split']['heldout_population_binding']['sha256']
    assert len(roots) == 68 and len(heldout_roots) == 31 and roots.isdisjoint(heldout_roots)
    assert all(split[r] == 'train' for r in roots)
    assert roots == {r['root_id'] for r in readl(OLD / 'm_clean_policy_training_manifest.jsonl')}
    assert protocol['replay']['count'] == summary['replay_question_count'] == read(PKG / 'REPLAY_DISPOSITION.json')['replay_count'] == 0
    assert len(roots_manifest) == 68 and {r['root_id'] for r in roots_manifest} == roots
    sources = {}
    for entry in roots_manifest:
        src_path = PKG / entry['packaged_source']
        original_path = entry['source_binding']['path']
        expected = entry['source_binding']['sha256']
        assert sha(src_path) == sha(original_path) == expected == external[original_path]['sha256']
        source = read(src_path)
        assert source['root_id'] == entry['root_id'] and source['turns']
        assert len(source['turns']) == len({t['turn_ref'] for t in source['turns']})
        assert sorted(entry['candidate_ids']) == sorted(q['candidate_id'] for q in effective if q['root_id'] == entry['root_id'])
        sources[entry['root_id']] = source

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    trainer = module(TRAINER, 'v125_independent_audit_trainer')
    display = module(DISPLAY, 'v125_independent_audit_display')
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == protocol['length']['chat_template_sha256'] == 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'
    assert sha(TRAINER) == external[str(TRAINER)]['sha256'] == '918a45232e4287a26f46a5b4eb48175476a82136f5c3e202c94bf0d8cecb87dc'
    assert sha(DISPLAY) == external[str(DISPLAY)]['sha256'] == protocol['display']['algorithm_binding']['sha256']
    lengths, supervised, prompt_fps, semantic_fps = [], [], set(), set()
    label_frequency = Counter()
    for i, q in enumerate(effective):
        row, lin = rmap[q['candidate_id']], lmap[q['candidate_id']]
        assert row['root_id'] == lin['root_id'] == q['root_id']
        assert row['question_type'] == lin['question_type'] == q['question_type']
        assert row['split'] == 'train' and row['selection_role'] == 'new_selected'
        assert row['dependence_group_id'] == lin['dependence_group_id'] == q['root_id'] + ':' + q['question_type']
        assert lin['original_record_sha256'] == digest(raw[i]) and lin['effective_record_sha256'] == digest(q)
        field = 'options' if q['question_type'] == 'multi_select' else 'items'
        labels = list('ABCDEF') if field == 'options' else [f'I{n}' for n in range(1, 7)]
        assert list(q[field]) == labels and len(set(q[field].values())) == 6
        variant = display.make_variants(q, seed=20261018, count=1)[0]
        assert variant == lin['display_variant']
        blind = copy.deepcopy(q)
        del blind['gold']
        assert display.make_variants(blind, seed=20261018, count=1)[0] == variant, 'DISPLAY_READS_GOLD'
        blind['gold'] = ['GOLD_SENTINEL_NOT_A_LABEL']
        blind['option_evidence'] = {'AUDIT_SENTINEL': 'private altered metadata'}
        blind['item_evidence'] = {'AUDIT_SENTINEL': 'private altered metadata'}
        assert display.make_variants(blind, seed=20261018, count=1)[0] == variant, 'DISPLAY_READS_GOLD_OR_EVIDENCE'
        mapping = lin['display_to_canonical']
        assert set(mapping) == set(mapping.values()) == set(labels)
        assert all(variant[field][label] == q[field][mapping[label]] for label in labels)
        expected_gold = ([label for label in labels if mapping[label] in q['gold']] if field == 'options'
                         else [next(label for label in labels if mapping[label] == old) for old in q['gold']])
        assert lin['canonical_gold'] == q['gold'] and lin['displayed_gold'] == expected_gold
        assert row['gold_cardinality'] == len(q['gold'])
        if field == 'options':
            assert 2 <= len(q['gold']) <= 5
            label_frequency.update(expected_gold)
        else:
            assert set(q['gold']) == set(labels) and len(q['gold']) == 6
        source = sources[q['root_id']]
        user = 'Dialogue:\n' + '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
        user += '\n\nQuestion:\n' + q['question']
        user += '\n\n' + ('Options' if field == 'options' else 'Items') + ':\n'
        user += '\n'.join(f'{label}. {variant[field][label]}' for label in labels)
        user += '\n\nOutput only one <box>... </box> answer. For multi-select use comma-separated letters; for ordering use comma-separated item IDs.'
        assert display.prompt(q, variant, source) == user
        assert display.prompt(blind, variant, source) == user, 'PROMPT_READS_GOLD_OR_EVIDENCE'
        expected_messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user},
                             {'role': 'assistant', 'content': '<box>' + ','.join(expected_gold) + '</box>'}]
        assert row['messages'] == expected_messages
        assert lin['messages_sha256'] == digest(expected_messages) and lin['prompt_sha256'] == digest(expected_messages[:-1])
        prompt_ids = tokenizer(trainer.render_chat(tokenizer, expected_messages[:-1], True), add_special_tokens=False).input_ids
        full_ids = tokenizer(trainer.render_chat(tokenizer, expected_messages, False), add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
            full_ids.append(tokenizer.eos_token_id)
        assert full_ids[:len(prompt_ids)] == prompt_ids and len(prompt_ids) < len(full_ids) <= 12288
        features = trainer.tokenize_record(tokenizer, row, 12288)
        assert features['input_ids'] == full_ids and features['attention_mask'] == [1] * len(full_ids)
        assert features['labels'] == [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        assert (lin['input_token_count'], lin['prompt_token_count'], lin['supervised_token_count']) == (len(full_ids), len(prompt_ids), len(full_ids) - len(prompt_ids))
        lengths.append(len(full_ids)); supervised.append(len(full_ids) - len(prompt_ids))
        semantic = {'question': ' '.join(q['question'].casefold().split()), 'type': q['question_type'],
                    'contents': sorted(' '.join(x.casefold().split()) for x in q[field].values()),
                    'gold_contents': sorted(q[field][x] for x in q['gold']) if field == 'options' else [q[field][x] for x in q['gold']]}
        assert lin['semantic_content_sha256'] == digest(semantic)
        assert lin['prompt_sha256'] not in prompt_fps and lin['semantic_content_sha256'] not in semantic_fps
        prompt_fps.add(lin['prompt_sha256']); semantic_fps.add(lin['semantic_content_sha256'])
    assert summary['input_tokens'] == dist(lengths)
    assert summary['dialogue_tokens'] == dist([len(tokenizer.encode('\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in s['turns']), add_special_tokens=False)) for s in sources.values()])
    assert summary['over_old_8192_count'] == sum(n > 8192 for n in lengths) == 11
    assert summary['over_configured_limit_count'] == 0
    assert summary['min_supervised_tokens'] == min(supervised)
    assert summary['question_type_counts'] == dict(Counter(q['question_type'] for q in effective)) == {'multi_select': 238, 'ordering': 125}
    assert summary['multi_select_gold_counts'] == dict(Counter(str(len(q['gold'])) for q in effective if q['question_type'] == 'multi_select')) == {'2': 65, '3': 59, '4': 57, '5': 57}
    groups = read(PKG / 'data/DEPENDENCE_GROUPS.json')['groups']
    assert len(groups) == len({r['dependence_group_id'] for r in rows}) == 134
    assert sum(len(g['candidate_ids']) for g in groups) == 363
    assert {c for g in groups for c in g['candidate_ids']} == set(rmap)
    for g in groups:
        assert all(rmap[c]['dependence_group_id'] == g['group_id'] for c in g['candidate_ids'])

    argv = plan['argv']
    params = dict(zip(argv[3::2], argv[4::2]))
    assert argv[1:3] == ['-m', 'scripts.current_method.restart_n5_spell.h4_round0_b0_training_smoke_v1.train_smoke_runner']
    assert params['--model'] == params['--tokenizer'] == str(BASE)
    assert params['--init-adapter-dir'] == str(ADAPTER) and params['--max-length'] == '12288'
    assert params['--train-file'] == str(PKG / 'data/TRAIN_NEW_ONLY.jsonl') and '--valid-file' not in params
    assert params['--seed'] == '20261018'
    adapter_config = read(ADAPTER / 'adapter_config.json')
    assert params['--lora-rank'] == str(adapter_config['lora_rank']) == '16'
    assert params['--lora-alpha'] == str(adapter_config['lora_alpha']) == '32'
    assert sorted(params['--target-modules'].split(',')) == sorted(adapter_config['target_modules'])
    recipe = read(ADAPTER.parent / 'training_recipe.json')
    for flag, key in [('epochs','epochs'),('batch-size','batch_size'),('grad-accum','grad_accum'),('lr','lr'),('min-lr-ratio','min_lr_ratio'),('warmup-ratio','warmup_ratio'),('weight-decay','weight_decay'),('max-grad-norm','max_grad_norm'),('lora-dropout','lora_dropout')]:
        assert float(params['--' + flag]) == float(recipe[key])
    assert plan['data_binding']['sha256'] == before['data/TRAIN_NEW_ONLY.jsonl']
    assert plan['training_started'] is summary['training_started'] is False and plan['status'] == 'PREPARED_NOT_RUN'
    assert plan['cwd'] == str(REPO / 'current_method_github')
    assert plan['environment']['SMOKE_GRADIENT_AUDIT_PATH'] == str(Path(plan['run_output_directory']) / 'gradient_audit.json')
    assert not Path(plan['run_output_directory']).exists(), 'TRAINING_ALREADY_STARTED'
    identity_path = REPO / 'restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/model_identity_report.json'
    identity = read(identity_path)
    model = identity['models']['M_clean_policy']
    assert sha(identity_path) == external[str(identity_path)]['sha256']
    assert sha(ADAPTER / 'adapter_config.json') == external[str(ADAPTER / 'adapter_config.json')]['sha256'] == model['adapter_config_sha256']
    assert external[str(ADAPTER / 'adapter_model.pt')]['sha256'] == model['adapter_state_sha256'] == '92b5706e2e2e4201e1aeff91bad8137e9662fc95ce96e9b71a2475e9097a31bf'
    for info in identity['base_model_identity']['files']:
        path = BASE / info['name']
        assert external[str(path)]['sha256'] == info['sha256'] and path.stat().st_size == info['size_bytes']
        if not info['name'].endswith('.safetensors'):
            assert sha(path) == info['sha256']
    assert before == {name: sha(PKG / name) for name in required}, 'AUDITED_FILES_CHANGED_DURING_CHECK'
    output = {
        'schema_version': 'v125_final_engineering_audit_v1', 'status': 'PASS', 'actual_executor': '/root/repair_v122_engineering',
        'audited_package_files': [{'path': name, 'sha256': before[name]} for name in sorted(required)],
        'record_count': 363, 'unique_train_root_count': 68, 'question_type_counts': summary['question_type_counts'],
        'multi_select_gold_counts': summary['multi_select_gold_counts'], 'replay_count': 0,
        'frozen_external_file_count': len(external), 'source_audit_dependency_files_rehashed': 954,
        'heldout_root_count': 31, 'heldout_overlap_count': 0,
        'input_tokens': dist(lengths), 'dialogue_tokens': summary['dialogue_tokens'],
        'supervised_tokens': dist(supervised), 'over_8192_count': 11, 'over_12288_count': 0,
        'checks': {'lossless_four_batch_selection': True, 'only_four_recorded_evidence_overlay_fields': True,
                   'canonical_questions_gold_and_sources_unchanged': True, '68_packaged_sources_byte_exact': True,
                   'all_source_turns_in_user_messages': True, 'public_only_prompt_exact_reconstruction': True,
                   'six_item_gold_blind_random_display_each_row': True, 'gold_removal_does_not_change_display': True,
                   'gold_evidence_mutation_does_not_change_display_or_prompt': True,
                   'all_display_maps_bijective_and_gold_rebound': True, 'one_training_record_per_canonical_question': True,
                   'chat_prompt_ids_are_full_input_prefix': True, 'all_prompt_labels_minus_100': True,
                   'all_assistant_labels_match_input_ids': True, 'all_supervision_nonempty': True,
                   'native_trainer_zero_truncation': True, 'base_plus_native_M_clean_custom_adapter_plan': True,
                   'all_954_source_audit_dependencies_frozen_and_hash_verified': True,
                   'LoRA_rank_alpha_target_modules_compatible': True, 'native_optimizer_scheduler_reset_verified_in_code': True,
                   'preflight_default_verifies_and_execute_flag_is_required_to_launch': True,
                   'read_package_files_unchanged_during_audit': True},
        'checkpoint_binding': {'base_model_path': str(BASE), 'init_adapter_path': str(ADAPTER),
                               'adapter_state_sha256': model['adapter_state_sha256'], 'binding_source': str(identity_path),
                               'large_weights_not_rehashed_in_this_final_audit': True,
                               'note': 'Adapter state was rehashed in the bound precursor ENGINEERING_AUDIT; final audit compares frozen model identity, input bindings and small configs without repeating 8GB model reads.'},
        'scope': 'Engineering audit of finalized data and execution plan only. SUMMARY statistics checked in memory but mutable final status intentionally excluded from audited_package_files.',
        'limitations': ['No model weights loaded, no inference, API calls or training performed.',
                        'GPU memory and training stability at max_length=12288 remain untested.',
                        'Train parents were previously seen by M_clean; no unseen-root claim.',
                        'Zero new replay does not remove label noise previously learned by M_clean.',
                        'Engineering validity does not constitute external semantic certification.'],
    }
    target = PKG / 'evidence/FINAL_ENGINEERING_AUDIT.json'
    with target.open('x') as handle:
        handle.write(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'path': str(target), 'sha256': sha(target), 'input_tokens': output['input_tokens'], 'supervised_tokens': output['supervised_tokens']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
