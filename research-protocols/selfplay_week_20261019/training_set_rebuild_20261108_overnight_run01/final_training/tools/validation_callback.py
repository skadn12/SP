"""Frozen common-DEV monitor, called with the trainer's current in-process model.

This module never constructs a model, starts a training run, retries a generation,
or uploads artifacts. The wrapper owns scheduling, RNG/state restoration and W&B.
"""
from __future__ import annotations

import hashlib
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_manifest(validation_manifest):
    """Verify the sealed files before any model call; support a path or dict."""
    if isinstance(validation_manifest, (str, Path)):
        manifest = json.loads(Path(validation_manifest).read_text(encoding='utf-8'))
    else:
        manifest = validation_manifest
    assert manifest['schema_version'] == 'common_dev_fixed_greedy_monitor_v1'
    assert manifest['status'] == 'FROZEN_CPU_PREFLIGHT_PASS'
    assert manifest['record_count'] == 242
    assert manifest['question_type_counts'] == {'multi_select': 123, 'ordering': 119}
    assert manifest['root_overlap_with_original_135']['intersection'] == []
    for item in manifest['bindings']:
        assert sha(item['path']) == item['sha256'], ('VALIDATION_BINDING_CHANGED', item['path'])
    records = read_jsonl(manifest['records']['path'])
    gold = read_jsonl(manifest['gold']['path'])
    assert len(records) == len(gold) == 242
    ids = [row['candidate_id'] for row in records]
    assert ids == sorted(set(ids)), 'FIXED_CANDIDATE_ORDER_REQUIRED'
    assert ids == [row['candidate_id'] for row in gold]
    for row, target in zip(records, gold):
        assert row['gold_record_sha256'] == digest(target), 'GOLD_RECORD_BINDING_CHANGED'
        assert row['question_type'] == target['question_type']
    return manifest, records, gold


def checked_gold_feature(tokenizer, trainer, record, max_length):
    """Use the native mask exactly, rejecting its otherwise silent truncation."""
    messages = record['messages']
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
    assert re.fullmatch(r'<box>[^<]*</box>', messages[-1]['content'])
    full_ids = tokenizer(trainer.render_chat(tokenizer, messages, False), add_special_tokens=False).input_ids
    prompt_ids = tokenizer(trainer.render_chat(tokenizer, messages[:-1], True), add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids.append(tokenizer.eos_token_id)
    assert full_ids[:len(prompt_ids)] == prompt_ids, 'LOSS_PROMPT_PREFIX_CHANGED'
    assert len(full_ids) <= max_length, 'LOSS_SEQUENCE_EXCEEDS_WINDOW_NO_TRUNCATION'
    feature = trainer.tokenize_record(tokenizer, record, max_length)
    assert feature['input_ids'] == full_ids, 'NATIVE_LOSS_WOULD_TRUNCATE'
    assert feature['labels'] == [-100] * len(prompt_ids) + full_ids[len(prompt_ids):], 'NATIVE_MASK_CHANGED'
    # CausalLM loss predicts labels[1:] from logits[:-1]. Count actual targets.
    targets = sum(label != -100 for label in feature['labels'][1:])
    assert targets > 0, 'NO_SUPERVISED_CAUSAL_TARGETS'
    return feature, {'full_tokens': len(full_ids), 'prompt_tokens': len(prompt_ids),
                     'supervised_causal_targets': targets, 'input_ids_sha256': digest(full_ids)}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('DUPLICATE_JSON_KEY')
        obj[key] = value
    return obj


def score_output(text, *, finish_reason, record, gold, scorer, adapter):
    """Score the final box independently of evidence/JSON protocol compliance.

    The transport accepts JSON.final_answer or exactly one box in reasoned text.
    Space/comma lists (and historical '>' ordering lists) normalize to the
    official scorer's comma interface without changing labels or order.
    """
    answer = {'final_answer_valid': False, 'exact': 0.0, 'J': 0.0,
              'auxiliary_protocol_valid': False, 'answer_errors': [],
              'auxiliary_protocol_errors': [], 'predicted_option_ids': None}
    obj, final = None, None
    try:
        obj = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        pass
    if isinstance(obj, dict) and isinstance(obj.get('final_answer'), str):
        final = obj['final_answer']
        answer['final_answer_transport'] = 'json_final_answer'
    else:
        boxes = re.findall(r'<box>(.*?)</box>', text, re.S)
        if len(boxes) == 1 and text.count('<box>') == text.count('</box>') == 1:
            final = '<box>' + boxes[0] + '</box>'
            answer['final_answer_transport'] = 'single_box_reasoned_text'
    try:
        assert finish_reason in ('eos', 'stop'), 'INCOMPLETE_RESPONSE'
        assert isinstance(final, str), 'UNAMBIGUOUS_FINAL_BOX_REQUIRED'
        match = re.fullmatch(r'\s*<box>(.*?)</box>\s*', final, re.S)
        assert match, 'FINAL_BOX_REQUIRED'
        body = match.group(1).strip()
        values = re.split(r'\s*(?:,|>)\s*|\s+', body) if body else []
        labels = list(record['display_to_option_id'])
        assert all(value in labels for value in values), 'UNKNOWN_OR_MALFORMED_LABEL'
        assert len(values) == len(set(values)), 'DUPLICATE_LABEL'
        if record['question_type'] == 'ordering':
            assert set(values) == set(labels), 'INCOMPLETE_ORDER'
        field = 'options' if record['question_type'] == 'multi_select' else 'items'
        result = scorer.score_response(
            {'question_type': record['question_type'], 'gold_content_ids': gold['canonical_gold']},
            {'variant': {field: {o['id']: o['text'] for o in record['options']},
                         'display_to_content': record['display_to_option_id']}},
            '<box>' + ','.join(values) + '</box>')
        assert result['scored'], result['scoring_status']
        answer.update(final_answer_valid=True, exact=float(result['exact_match']),
                      J=float(Fraction(result['partial_score_fraction'])),
                      predicted_option_ids=result['content_prediction'], official_score=result,
                      parsed_display_labels=values, final_answer=final)
    except (AssertionError, ValueError, TypeError) as exc:
        answer['answer_errors'].append(str(exc))
    try:
        assert isinstance(obj, dict), 'JSON_OBJECT_REQUIRED_FOR_AUXILIARY_PROTOCOL'
        from jsonschema import validate
        validate(obj, json.loads(Path(record['schema']['path']).read_text(encoding='utf-8')))
        assert list(obj) == ['item_evidence', 'answer_reason', 'final_answer'], 'REASONS_MUST_PRECEDE_ANSWER'
        assert obj['answer_reason'].strip(), 'EMPTY_ANSWER_REASON'
        for label, row in obj['item_evidence'].items():
            assert row['reason'].strip(), 'EMPTY_ITEM_REASON:' + label
            assert len(row['turn_refs']) == len(set(row['turn_refs'])), 'DUPLICATE_REFERENCE:' + label
            if row['judgment'] != 'not_supported':
                assert row['turn_refs'], 'CITATION_REQUIRED:' + label
        answer['auxiliary_protocol_valid'] = True
    except Exception as exc:
        answer['auxiliary_protocol_errors'].append(str(exc))
    return answer


def aggregate(rows):
    """All means use the fixed attempted-record denominator, including invalids."""
    metrics = {}
    groups = [('', rows)] + [(kind + '/', [r for r in rows if r['question_type'] == kind])
                            for kind in ('multi_select', 'ordering')]
    for prefix, group in groups:
        n = len(group)
        assert n, 'EMPTY_VALIDATION_GROUP'
        targets = sum(row['supervised_causal_targets'] for row in group)
        metrics.update({prefix + 'records': n,
                        prefix + 'valid_loss': sum(row['valid_loss'] for row in group) / n,
                        prefix + 'valid_loss_token_weighted': sum(row['valid_loss'] * row['supervised_causal_targets'] for row in group) / targets,
                        prefix + 'supervised_causal_targets': targets,
                        prefix + 'exact': sum(row['exact'] for row in group) / n,
                        prefix + 'mean_J': sum(row['J'] for row in group) / n,
                        prefix + 'final_answer_valid_rate': sum(row['final_answer_valid'] for row in group) / n,
                        prefix + 'auxiliary_protocol_valid_rate': sum(row['auxiliary_protocol_valid'] for row in group) / n,
                        prefix + 'truncation_rate': sum(row['truncated'] for row in group) / n})
    assert all(isinstance(v, (int, float)) and math.isfinite(v) for v in metrics.values())
    return metrics


def evaluate(*, model, tokenizer, step, output_dir, validation_manifest):
    """Evaluate all 242 records once using the already loaded current model.

    ``output_dir`` may exist and must be empty. Raw text, token IDs and each
    scored record are flushed to disk immediately. Infrastructure errors are
    persisted and raised without retries; a failed monitor cannot look complete.
    """
    import torch

    manifest, records, targets = verify_manifest(validation_manifest)
    trainer = load('fixed_dev_native_trainer', manifest['native_trainer']['path'])
    adapter = load('fixed_dev_structure_adapter', manifest['generation_adapter']['path'])
    scorer = load('fixed_dev_official_scorer', manifest['official_scorer']['path'])
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'VALIDATION_OUTPUT_MUST_BE_EMPTY_NO_RETRY'
    raw_dir = out / 'raw'
    raw_dir.mkdir()
    write_json(out / 'ATTEMPT.json', {'step': int(step), 'record_count': len(records),
               'validation_manifest_sha256': digest(manifest), 'automatic_retry': False,
               'generation': manifest['generation'], 'raw_cot_saved': True})
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    assert pad is not None, 'PAD_OR_EOS_TOKEN_REQUIRED'
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with (out / 'RECORDS.jsonl').open('x', encoding='utf-8') as journal, torch.no_grad():
        for index, (record, gold) in enumerate(zip(records, targets)):
            name = f'{index:03d}_{hashlib.sha256(record["candidate_id"].encode()).hexdigest()[:16]}'
            row = {'index': index, 'step': int(step), 'candidate_id': record['candidate_id'],
                   'root_id': record['root_id'], 'question_type': record['question_type'],
                   'automatic_retry': False}
            try:
                # Preflight every record again with the exact in-process tokenizer.
                feature, counts = checked_gold_feature(tokenizer, trainer, gold, manifest['loss']['max_length'])
                assert counts == record['loss_tokenization'], 'LOSS_TOKENIZATION_CHANGED'
                prompt = Path(record['generation_prompt']['path']).read_text(encoding='utf-8')
                input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
                assert len(input_ids) == record['generation_prompt_tokens'], 'GENERATION_TOKEN_COUNT_CHANGED'
                assert digest(input_ids) == record['generation_input_ids_sha256'], 'GENERATION_TOKENIZATION_CHANGED'
                assert len(input_ids) + manifest['generation']['max_new_tokens'] <= manifest['generation']['max_model_len'], 'GENERATION_WINDOW_EXCEEDED_NO_TRUNCATION'
                tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
                generated = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    do_sample=False, num_beams=1, max_new_tokens=manifest['generation']['max_new_tokens'],
                    use_cache=True, pad_token_id=pad, eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=False)
                sequence = generated.sequences[0]
                assert sequence[:len(input_ids)].tolist() == input_ids, 'GENERATION_INPUT_PREFIX_CHANGED'
                output_ids = sequence[len(input_ids):].detach().cpu().tolist()
                raw_text = tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_path, token_path = raw_dir / (name + '.txt'), raw_dir / (name + '.tokens.json')
                with raw_path.open('x', encoding='utf-8', newline='') as raw_stream:
                    raw_stream.write(raw_text)
                    raw_stream.flush()
                    os.fsync(raw_stream.fileno())
                write_json(token_path, {'token_ids': output_ids})
                eos = tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                stopped = bool(output_ids and output_ids[-1] in eos_ids)
                truncated = not stopped and len(output_ids) >= manifest['generation']['max_new_tokens']
                finish = 'eos' if stopped else 'length' if truncated else 'stop'
                row.update(raw_output=raw_text, scoring_text=text, raw_output_binding=bind(raw_path),
                           generated_tokens_binding=bind(token_path), completion_tokens=len(output_ids),
                           finish_reason=finish, truncated=truncated, raw_cot_saved=True,
                           generation_prompt=record['generation_prompt'], gold_record_sha256=record['gold_record_sha256'])
                # Persist returned raw output before either scoring or loss evaluation.
                write_json(raw_dir / (name + '.returned.json'), row)
                # HF may return a KV cache. Release it before teacher-forced loss.
                del generated, sequence, tokens
                row.update(score_output(text, finish_reason=finish, record=record, gold=gold, scorer=scorer, adapter=adapter))
                batch = trainer.collate([feature], pad, device)
                loss_output = model(**batch)
                loss = float(loss_output.loss.detach().float().cpu())
                assert math.isfinite(loss), 'NONFINITE_VALID_LOSS'
                row.update(valid_loss=loss, **counts)
                del loss_output, batch
            except Exception as exc:
                row.update(infrastructure_error={'type': type(exc).__name__, 'message': str(exc)})
                write_json(raw_dir / (name + '.error.json'), row)
                journal.write(canonical(row) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                write_json(out / 'INTERRUPTED.json', {'step': int(step), 'completed_records': len(rows),
                           'failed_candidate_id': record['candidate_id'], 'automatic_retry': False,
                           'infrastructure_error': row['infrastructure_error']})
                raise RuntimeError('VALIDATION_INTERRUPTED_NO_RETRY: ' + record['candidate_id']) from exc
            write_json(raw_dir / (name + '.scored.json'), row)
            journal.write(canonical(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            rows.append(row)
    metrics = aggregate(rows)
    write_json(out / 'METRICS.json', {'step': int(step), 'metrics': metrics,
               'definitions': manifest['metric_definitions'], 'evaluated_records': len(rows),
               'raw_cot_saved': True, 'protocol_comparison_limit': manifest['protocol_comparison_limit']})
    artifacts = [bind(path) for path in sorted(out.rglob('*')) if path.is_file()]
    return {'metrics': metrics, 'artifacts': artifacts, 'raw_cot_saved': True, 'evaluated_records': len(rows)}
