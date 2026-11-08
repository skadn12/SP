#!/usr/bin/env python3
"""Build the complete fixed common-DEV monitor using CPU tokenization only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from validation_callback import bind, canonical, checked_gold_feature, digest, load, sha, write_json

FINAL = Path(__file__).resolve().parents[1]
WEEK = FINAL.parent.parent
PROJECT = WEEK.parent.parent
RESULTS = PROJECT / 'restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_dev/candidate_results.jsonl'
DEV = PROJECT / 'restart-n5-runs/h4_round0_b0_matched_training_v1_live_2/dev_full_eval_manifest.jsonl'
OLD = WEEK / 'training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl'
TRAINER = PROJECT / 'current_method_github/train_qwen3_lora_sft.py'
ADAPTER = WEEK / 'training_set_structure_pilot_plan_20261106_v1/execution_20261106_v1/solver/adapter.py'
PACKAGE = PROJECT / 'offline-deliveries/h4_codex_train_pool_v125/build_package.py'
MODEL = Path('/root/autodl-tmp/Qwen3-4B')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as stream:
        stream.write(text)


def parse_historical_prompt(text, qtype):
    """Recover all original visible text; never consult a historical prediction."""
    match = re.fullmatch(r'<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n.*', text, re.S)
    assert match, 'HISTORICAL_CHAT_FORMAT_CHANGED'
    system, user = match.groups()
    assert user.startswith('Dialogue:\n'), 'FULL_DIALOGUE_REQUIRED'
    dialogue, rest = user[len('Dialogue:\n'):].split('\n\nQuestion (' + qtype + '):\n', 1)
    heading = 'Options' if qtype == 'multi_select' else 'Items'
    question, rest = rest.split('\n\n' + heading + ':\n', 1)
    options_text, instruction = rest.rsplit('\n\nOutput only ', 1)
    parts = re.split(r'^([A-Z]|I[0-9]+)\. ', options_text, flags=re.M)
    assert parts[0] == '' and len(parts) >= 5
    options = [{'id': parts[i], 'text': parts[i + 1][:-1] if parts[i + 1].endswith('\n') else parts[i + 1]}
               for i in range(1, len(parts), 2)]
    turns = []
    for line in dialogue.split('\n'):
        turn = re.fullmatch(r'\[([^\]]+)\] ([^:]+): (.*)', line)
        assert turn, 'HISTORICAL_TURN_PARSE_FAILED'
        ref, speaker, body = turn.groups()
        turns.append({'turn_ref': ref, 'speaker': speaker, 'text': body})
    assert len({t['turn_ref'] for t in turns}) == len(turns), 'DUPLICATE_SOURCE_REF'
    return {'system': system, 'user': user, 'dialogue_text': dialogue, 'question': question,
            'options': options, 'turns': turns, 'instruction': 'Output only ' + instruction}


def build():
    assert not (FINAL / 'VALIDATION_MANIFEST.json').exists(), 'VALIDATION_ALREADY_FROZEN'
    source_results = lines(RESULTS)
    frozen_dev = lines(DEV)
    assert len(source_results) == len(frozen_dev) == 485
    assert len({r['root_id'] for r in source_results}) == 15
    source_by_id = {r['candidate_id']: r for r in frozen_dev}
    assert len(source_by_id) == 485
    selected = sorted((r for r in source_results if r['question_type'] in ('multi_select', 'ordering')),
                      key=lambda row: row['candidate_id'])
    assert Counter(r['question_type'] for r in selected) == Counter(multi_select=123, ordering=119)
    original = lines(OLD)
    assert len(original) == 135
    old_roots = sorted({r['root_id'] for r in original})
    dev_roots = sorted({r['root_id'] for r in selected})
    assert not set(old_roots) & set(dev_roots), 'DEV_TRAIN_ROOT_LEAKAGE'
    trainer = load('fixed_dev_build_trainer', TRAINER)
    adapter = load('fixed_dev_build_adapter', ADAPTER)
    package = load('fixed_dev_build_training_package', PACKAGE)
    display = load('fixed_dev_build_training_display', package.DISPLAY)
    scorer_path = adapter.load_history().h.baseline.SCORER_PATH
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    dependencies = {RESULTS, DEV, OLD, TRAINER, ADAPTER, PACKAGE, Path(package.DISPLAY),
                    Path(scorer_path), Path(__file__), FINAL / 'tools/validation_callback.py'}
    dependencies.update(adapter.code_dependencies())
    dependencies.update(p for p in MODEL.iterdir() if p.is_file() and p.name in {
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'})
    records, gold_rows, preflight = [], [], []
    missing_gold_refs = []
    asset_dir = FINAL / 'validation_assets'
    asset_dir.mkdir(exist_ok=False)
    for index, result in enumerate(selected):
        qid, qtype = result['candidate_id'], result['question_type']
        source = source_by_id[qid]
        assert source['split'] == 'dev' and source['question_type'] == qtype
        candidates = [r for r in result['requests'] if r['shift'] == 0 and r['context_mode'] == 'full_context' and r['slot_id'] == 0]
        assert len(candidates) == 1, 'EXACTLY_ONE_SHIFT0_FULL_S00_REQUIRED'
        request = candidates[0]
        historical_prompt = RESULTS.parent / 'raw_prompts' / (request['prompt_sha256'] + '.txt')
        assert sha(historical_prompt) == request['prompt_sha256'], 'HISTORICAL_PROMPT_HASH_MISMATCH'
        dependencies.add(historical_prompt)
        parsed = parse_historical_prompt(historical_prompt.read_text(encoding='utf-8'), qtype)
        source_path, pool_path = Path(source['dialogue_path']), Path(source['candidate_pool_path'])
        assert sha(source_path) == source['dialogue_sha256'] == result['dialogue_sha256']
        assert sha(pool_path) == source['candidate_pool_sha256'] == result['candidate_pool_sha256']
        dependencies.update((source_path, pool_path))
        source_document = json.loads(source_path.read_text(encoding='utf-8'))
        full_source_turns = [{k: turn[k] for k in ('turn_ref', 'speaker', 'text')} for turn in source_document['turns']]
        assert parsed['turns'] == full_source_turns, 'COMPLETE_ORIGINAL_DIALOGUE_NOT_PRESERVED'
        candidate = source['candidate']
        field = 'options' if qtype == 'multi_select' else 'items'
        assert parsed['question'] == candidate['question'], 'QUESTION_TEXT_CHANGED'
        assert {o['id']: o['text'] for o in parsed['options']} == candidate[field], 'OPTION_TEXT_CHANGED'
        mapping = request['display_to_canonical']
        labels = [o['id'] for o in parsed['options']]
        assert list(mapping) == labels and all(k == v for k, v in mapping.items()), 'SHIFT0_IDENTITY_DISPLAY_REQUIRED'
        assert request['displayed_gold'] == request['canonical_gold'] == candidate['gold'], 'GOLD_REFERENCE_DRIFT'
        gold_labels = request['displayed_gold']
        gold_refs = list(candidate.get('evidence_refs', []))
        bad_refs = sorted(set(gold_refs) - {t['turn_ref'] for t in parsed['turns']})
        if bad_refs:
            missing_gold_refs.append({'candidate_id': qid, 'missing_original_gold_evidence_refs': bad_refs})
        # Gold loss uses the identical pure-box/comma prompt formatter as TRAIN135.
        loss_user = display.prompt({'question_type': qtype, 'question': parsed['question']},
                                  {field: {o['id']: o['text'] for o in parsed['options']}},
                                  {'turns': parsed['turns']})
        messages = [{'role': 'system', 'content': package.SYSTEM}, {'role': 'user', 'content': loss_user},
                    {'role': 'assistant', 'content': '<box>' + ','.join(gold_labels) + '</box>'}]
        target = {'candidate_id': qid, 'root_id': result['root_id'], 'question_type': qtype,
                  'split': 'dev', 'messages': messages, 'canonical_gold': request['canonical_gold'],
                  'displayed_gold': gold_labels, 'original_gold_evidence_refs': gold_refs,
                  'gold_refs_sha256': digest(gold_refs), 'gold_origin_record_sha256': digest(candidate),
                  'gold_origin': {'file': str(DEV), 'candidate_id': qid}}
        _, loss_counts = checked_gold_feature(tokenizer, trainer, target, 12288)
        # Only public fields are supplied to the frozen historical generation formatter.
        item = {'question_type': qtype, 'question': parsed['question'], 'options': parsed['options'], 'public_fields': {}}
        variant = {'display_to_option_id': mapping, 'options': parsed['options']}
        prompt = adapter.prompt_for(tokenizer, item, variant, parsed['turns'])
        assert '<box>' + ','.join(gold_labels) + '</box>' not in prompt, 'GOLD_TARGET_IN_GENERATION_PROMPT'
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        assert len(prompt_ids) + 6144 <= 16384, 'GENERATION_WINDOW_EXCEEDED_NO_TRUNCATION'
        name = f'{index:03d}_{digest(qid)[:16]}'
        prompt_path = asset_dir / (name + '.generation.txt')
        loss_path = asset_dir / (name + '.loss_prompt.txt')
        schema_path = asset_dir / (name + '.schema.json')
        write_text(prompt_path, prompt)
        write_text(loss_path, trainer.render_chat(tokenizer, messages[:-1], True))
        write_json(schema_path, adapter.schema_for(labels, [t['turn_ref'] for t in parsed['turns']], qtype))
        dependencies.update((prompt_path, loss_path, schema_path))
        record = {'index': index, 'candidate_id': qid, 'root_id': result['root_id'], 'question_type': qtype,
                  'context_mode': 'full_context', 'shift': 0, 'slot_id': 0,
                  'historical_request_id': request['request_id'], 'display_to_option_id': mapping,
                  'question': parsed['question'], 'options': parsed['options'],
                  'source': bind(source_path), 'candidate_pool': bind(pool_path),
                  'historical_prompt': bind(historical_prompt), 'source_turn_count': len(parsed['turns']),
                  'source_turn_refs_sha256': digest([t['turn_ref'] for t in parsed['turns']]),
                  'source_visible_turns_sha256': digest(parsed['turns']),
                  'gold_record_sha256': digest(target), 'gold_refs_sha256': digest(gold_refs),
                  'generation_prompt': bind(prompt_path), 'generation_prompt_tokens': len(prompt_ids),
                  'generation_input_ids_sha256': digest(prompt_ids), 'schema': bind(schema_path),
                  'loss_prompt': bind(loss_path), 'loss_tokenization': loss_counts}
        records.append(record)
        gold_rows.append(target)
        preflight.append({'index': index, 'candidate_id': qid, 'question_type': qtype,
                          'generation_prompt_tokens': len(prompt_ids), 'generation_total_budget': len(prompt_ids) + 6144,
                          'complete_source_turn_count': len(parsed['turns']), **loss_counts})
    for path, rows in ((FINAL / 'VALIDATION_RECORDS.jsonl', records), (FINAL / 'VALIDATION_GOLD.jsonl', gold_rows)):
        write_text(path, ''.join(canonical(row) + '\n' for row in rows))
        dependencies.add(path)
    report = {'schema_version': 'common_dev_fixed_greedy_cpu_preflight_v1', 'status': 'PASS',
              'record_count': 242, 'question_type_counts': dict(Counter(r['question_type'] for r in records)),
              'generation_max_prompt_tokens': max(r['generation_prompt_tokens'] for r in records),
              'generation_max_total_budget': max(r['generation_prompt_tokens'] + 6144 for r in records),
              'loss_max_full_tokens': max(r['loss_tokenization']['full_tokens'] for r in records),
              'loss_min_supervised_causal_targets': min(r['loss_tokenization']['supervised_causal_targets'] for r in records),
              'original_gold_reference_issues': missing_gold_refs, 'source_truncated': False,
              'historical_score_used_for_selection': False, 'model_weights_loaded': False, 'model_calls': 0,
              'rows': preflight}
    write_json(FINAL / 'VALIDATION_PREFLIGHT.json', report)
    dependencies.add(FINAL / 'VALIDATION_PREFLIGHT.json')
    manifest = {'schema_version': 'common_dev_fixed_greedy_monitor_v1', 'status': 'FROZEN_CPU_PREFLIGHT_PASS',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'record_count': 242,
                'question_type_counts': {'multi_select': 123, 'ordering': 119}, 'root_count': len(dev_roots),
                'root_ids': dev_roots, 'source_candidate_count': 485,
                'selection': 'All original common DEV multi_select and ordering records, candidate_id ascending; no historical score filtering.',
                'fixed_display': {'shift': 0, 'context_mode': 'full_context', 'slot_id': 0, 'displays_per_candidate': 1},
                'root_overlap_with_original_135': {'original_train_record_count': 135, 'original_train_unique_roots': len(old_roots),
                                                 'original_train_root_ids': old_roots, 'intersection': [], 'original_train': bind(OLD)},
                'generation': {'backend': 'in-process HuggingFace current model.generate', 'do_sample': False, 'num_beams': 1,
                               'max_new_tokens': 6144, 'max_model_len': 16384, 'enable_thinking': False,
                               'explicit_brief_reasoning': True, 'use_cache': True, 'automatic_retry': False,
                               'gold_supplied_to_model': False, 'truncate': False},
                'loss': {'max_length': 12288, 'assistant_only': True, 'target': 'pure original Gold box with comma-separated IDs',
                         'prompt': 'unchanged native TRAIN135 strict system and original training display.prompt formatter',
                         'batch_size': 1, 'truncate': False, 'native_mask': 'train_qwen3_lora_sft.tokenize_record'},
                'metric_definitions': {'valid_loss': 'Sample-weighted arithmetic mean of native CausalLM output.loss; one record per batch, all 242 records.',
                    'valid_loss_token_weighted': 'sum(record output.loss * count(labels[1:] != -100)) / sum(count(labels[1:] != -100)); actual causal Gold target tokens, including native suffix/EOS.',
                    'exact': 'Official final-answer exact_match mean over all 242 fixed records; invalid/incomplete final answer contributes zero.',
                    'mean_J': 'Official historical scorer signed J mean over fixed records: multi_select TP/g-FP/(n-g), g=n uses TP/g; ordering (correct_pairs-wrong_pairs)/total_pairs; invalid contributes zero.',
                    'final_answer_valid_rate': 'Fraction with complete response and an independently parseable final answer. Accept JSON final_answer or one final box in reasoned text; auxiliary format failure does not invalidate an otherwise valid answer.',
                    'auxiliary_protocol_valid_rate': 'Historical structure-pilot evidence/answer_reason/final_answer schema and citation-reference compliance, reported separately; does not verify citation entailment.',
                    'by_question_type': 'Same metrics restricted to the frozen 123 multi_select or 119 ordering records with their fixed denominators.'},
                'protocol_comparison_limit': 'New fixed single-display greedy training trajectory monitor. This is not the historical randomized display/repeat/schema-constrained formal screening protocol. Step0 requires new generation; historical box-only scores are not reused.',
                'records': bind(FINAL / 'VALIDATION_RECORDS.jsonl'), 'gold': bind(FINAL / 'VALIDATION_GOLD.jsonl'),
                'preflight': bind(FINAL / 'VALIDATION_PREFLIGHT.json'), 'native_trainer': bind(TRAINER),
                'generation_adapter': bind(ADAPTER), 'official_scorer': bind(scorer_path),
                'bindings': [bind(path) for path in sorted(dependencies)],
                'execution_status': {'model_weights_loaded': False, 'model_calls': 0, 'training_started': False}}
    write_json(FINAL / 'VALIDATION_MANIFEST.json', manifest)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    return manifest


if __name__ == '__main__':
    build()
