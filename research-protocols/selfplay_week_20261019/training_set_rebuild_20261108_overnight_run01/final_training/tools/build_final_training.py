#!/usr/bin/env python3
"""One-shot final export after explicit candidate admission; no model execution."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

from training_contract import (CANDIDATES, FINAL, FORMATTER, INIT, INIT_SHA, MODEL,
    OLD_SHA, OLD_TRAIN, OUTPUT, RECIPE, SCORES, TRAINER, TRAINER_SHA, bind,
    canonical, digest, load, now, read, sha, verify_binding, verify_fixed_inputs,
    write_once)


def normalize_candidate(row, decision):
    c = row['candidate']
    typ = row['question_type']
    options = c['options']
    labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(options)] if typ == 'multi_select' else [f'I{i+1}' for i in range(len(options))]
    ids = [o.get('option_id') or o.get('id') for o in options]
    assert len(ids) == len(set(ids)) and all(ids), 'NONBIJECTIVE_OPTION_IDS'
    mapping = dict(zip(labels, ids))
    item = dict(candidate_id=row['candidate_id'], root_id=row['root_id'], question_type=typ,
                question=c['question'], options=[dict(id=lab, text=opt['text']) for lab, opt in zip(labels, options)],
                label_to_option_id=mapping, gold_option_ids=c.get('gold_pairs') if typ == 'partial_order' else c['gold_option_ids'],
                source_path=row['source_path'], source_sha256=row['source_sha256'],
                source_turn_count=row['source_turn_count'], arm=row['arm'], split='train',
                origin='admitted_140_' + row['arm'], family_id=decision.get('semantic_family_id') or decision.get('family_id') or row['candidate_id'])
    if typ == 'partial_order':
        item['query_pairs'] = c.get('query_pairs', [])
    return item


def build(admission_path):
    verify_fixed_inputs()
    outputs = [FINAL / n for n in ('FINAL_TRAIN.jsonl', 'ADDED_TRAIN.jsonl', 'ADDED_QUESTION_LINEAGE.jsonl', 'FINAL_MANIFEST.json')]
    assert not any(p.exists() for p in outputs), 'FINAL_OUTPUT_ALREADY_EXISTS'
    admission = read(admission_path)
    assert admission['status'] == 'FINAL_ADMISSION_FROZEN', 'EXPLICIT_FINAL_ADMISSION_REQUIRED'
    ids = admission['accepted_candidate_ids']
    assert isinstance(ids, list) and len(ids) == len(set(ids)), 'DUPLICATE_ACCEPTED_ID'
    decisions = {r['candidate_id']: r for r in admission['decisions']}
    assert len(decisions) == len(admission['decisions']) == 140, 'ALL_140_DISPOSITIONS_REQUIRED'
    assert set(ids) == {cid for cid, d in decisions.items() if d['training_admission'] == 'ACCEPT'}
    sources = read(CANDIDATES)
    rows = {r['candidate_id']: r for r in sources['records']}
    assert set(decisions) == set(rows), 'ADMISSION_DENOMINATOR_MISMATCH'
    scores = read(SCORES)
    evidence = [bind(admission_path), bind(CANDIDATES), bind(SCORES)]
    for b in admission.get('bindings', []):
        verify_binding(b); evidence.append(bind(b['path']))
    old = OLD_TRAIN.read_bytes()
    assert old.endswith(b'\n'), 'OLD_PREFIX_MUST_END_IN_NEWLINE'
    old_rows = [json.loads(line) for line in old.splitlines() if line.strip()]
    assert len(old_rows) == 135 and len({r['candidate_id'] for r in old_rows}) == 135
    assert not set(ids) & {r['candidate_id'] for r in old_rows}, 'DUPLICATE_OLD_CANDIDATE_ID'
    pack = load('final_export_frozen_formatter', FORMATTER)
    deps = pack.dependencies()
    evidence += deps[-1]
    added, lineage = [], []
    seen_prompts = {digest(r['messages'][:-1]) for r in old_rows}
    for cid in ids:
        d, row = decisions[cid], rows[cid]
        assert d['quality_status'] == 'PASS', ('QUALITY_NOT_PASS', cid)
        assert d['semantic_status'] == 'PASS', ('SEMANTIC_NOT_PASS', cid)
        assert d['lineage_status'] == 'PASS', ('LINEAGE_NOT_PASS', cid)
        assert d.get('evidence_bindings'), ('NO_ADMISSION_EVIDENCE', cid)
        for b in d['evidence_bindings']:
            verify_binding(b); evidence.append(bind(b['path']))
        assert row['canonical_status'] == 'PASS' and row['evidence_complete']
        full = scores['by_item'][cid]['full_context']
        assert full['returned'] == full['planned'] and full['missing'] == 0, ('INCOMPLETE_FULL', cid)
        mean = full['mean_J_complete_plan']
        assert isinstance(mean, (int, float)) and 0 < mean < 1
        assert math.isclose(mean, d['solver_full_mean'], rel_tol=0., abs_tol=1e-12)
        source_binding = bind(row['source_path'])
        assert source_binding['sha256'] == row['source_sha256']
        source = read(row['source_path'])
        assert source['root_id'] == row['root_id'] and source['turns']
        refs = [t['turn_ref'] for t in source['turns']]
        assert len(refs) == len(set(refs)) == row['source_turn_count']
        item = normalize_candidate(row, d)
        messages, variant, gold, displayed, pairs = pack.public_and_gold(item, source, deps)
        counts, fits = pack.checked_features(messages, deps)
        assert fits, ('ACCEPTED_ITEM_OVER_MAX_LENGTH', cid, counts)
        # A second valid Gold tests the actual frozen formatter's blindness.
        other = copy.deepcopy(item)
        stable = list(item['label_to_option_id'].values())
        if item['question_type'] == 'multi_select':
            choices = [stable[:2], stable[-2:]]
            other['gold_option_ids'] = next(x for x in choices if set(x) != set(item['gold_option_ids']))
        elif item['question_type'] == 'ordering':
            other['gold_option_ids'] = list(reversed(item['gold_option_ids']))
        else:
            other['gold_option_ids'] = []
        changed, changed_variant, _, _, _ = pack.public_and_gold(other, source, deps)
        assert messages[:-1] == changed[:-1] and variant['display_to_option_id'] == changed_variant['display_to_option_id'], 'GOLD_DEPENDENT_DISPLAY'
        prompt_hash = digest(messages[:-1])
        assert prompt_hash not in seen_prompts, ('EXACT_PUBLIC_PROMPT_DUPLICATE', cid)
        seen_prompts.add(prompt_hash)
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
        assert messages[0]['content'] == old_rows[0]['messages'][0]['content']
        assert messages[-1]['content'].startswith('<box>') and messages[-1]['content'].endswith('</box>')
        added.append(dict(candidate_id=cid, root_id=row['root_id'], question_type=row['question_type'], split='train',
                          selection_role=item['origin'], dependence_group_id=item['family_id'],
                          gold_cardinality=len(gold['pairs' if item['question_type'] == 'partial_order' else 'option_ids']), messages=messages))
        lineage.append(dict(candidate_id=cid, source_binding=source_binding, arm=row['arm'],
                            source_turn_count=len(refs), display_variant=variant,
                            canonical_gold=gold, displayed_gold=displayed, public_query_pairs=pairs,
                            messages_sha256=digest(messages), prompt_sha256=prompt_hash,
                            admission_decision=d, gold_blind_display_verified=True, **counts))
        evidence.append(source_binding)
    additions = ''.join(canonical(r) + '\n' for r in added).encode('utf-8')
    lineage_raw = ''.join(canonical(r) + '\n' for r in lineage).encode('utf-8')
    final_raw = old + additions
    all_rows = old_rows + added
    manifest = dict(schema='final_merged_training_manifest_v2', status='FINAL_DATA_FROZEN_NO_TRAINING',
        created_at_utc=now(), old_record_count=135, added_record_count=len(added), record_count=len(all_rows),
        accepted_candidate_ids=ids, root_count=len({r['root_id'] for r in all_rows}),
        question_type_counts=dict(Counter(r['question_type'] for r in all_rows)),
        selection_role_counts=dict(Counter(r['selection_role'] for r in all_rows)),
        old_train_binding=bind(OLD_TRAIN), old_prefix_bytes=len(old), old_prefix_sha256=OLD_SHA,
        old_135_byte_prefix_preserved=True, base_model=str(MODEL), tokenizer=str(MODEL),
        init_adapter_dir=str(INIT), init_adapter_sha256=INIT_SHA, trainer=str(TRAINER), trainer_sha256=TRAINER_SHA,
        recipe=RECIPE, expected_backward_calls=2*len(all_rows), expected_optimizer_updates=math.ceil(2*len(all_rows)/4),
        scheduler_horizon=math.ceil(math.ceil(len(all_rows)/4)*2), output_dir=str(OUTPUT),
        assistant_only_gold_supervision=True, full_source=True, truncate=False, display_count_per_question=1,
        training_answer_format='old frozen export formatter: pure box target; comma letters/item IDs; no evidence JSON',
        wandb_mode='online_required', tensorboard_enabled=False, validation_every_optimizer_steps=20,
        validation_cot_policy='callback retains raw JSON item_evidence/answer_reason/final_answer unchanged',
        protocol_limitations=admission.get('protocol_limitations', []), training_started=False)
    raw_outputs = [(outputs[0], final_raw), (outputs[1], additions), (outputs[2], lineage_raw)]
    evidence += [dict(path=str(p), sha256=__import__('hashlib').sha256(raw).hexdigest(), size_bytes=len(raw)) for p, raw in raw_outputs]
    evidence += [bind(p) for p in (TRAINER, INIT/'adapter_model.pt', INIT/'adapter_config.json', OLD_TRAIN, Path(__file__))]
    unique = {}
    for b in evidence:
        if b['path'] in unique:
            assert b['sha256'] == unique[b['path']]['sha256'], 'BINDING_DRIFT'
        unique[b['path']] = b
    manifest['bindings'] = list(unique.values())
    for b in evidence:
        if Path(b['path']).exists(): verify_binding(b)
    for p, raw in raw_outputs:
        with p.open('xb') as stream: stream.write(raw)
    assert (FINAL/'FINAL_TRAIN.jsonl').read_bytes().startswith(old)
    write_once(outputs[3], manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.admission)
    print(json.dumps({k: result[k] for k in ('status', 'record_count', 'added_record_count', 'accepted_candidate_ids', 'question_type_counts')}, ensure_ascii=False, indent=2))
