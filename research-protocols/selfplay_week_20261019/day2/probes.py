"""Freeze matched evidence-reminder requests after train-side direct diagnosis."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / 'solver'))
import run as rt


def prepare():
    base = HERE / 'solver/direct'
    target = HERE / 'probes/evidence_hint'
    rt.require(not target.exists(), 'PROBE_RUN_EXISTS')
    m0 = rt.verify(base)
    scores = rt.read(base / 'SCORES.json')
    rt.require(scores['status'] == 'COMPLETE', 'DIRECT_NOT_COMPLETE')
    rt.require(not (target / 'STARTED.json').exists(), 'PROBE_ALREADY_STARTED')
    policy = rt.read(HERE / 'PROBE_SELECTION_POLICY.json')
    priv = {q['candidate_id']: q for q in rt.read(m0['private_answer_binding']['path'])['questions']}
    pub = {q['candidate_id']: q for q in rt.read(base / 'PUBLIC_QUESTIONS.json')['questions']}
    valid_wrong = {r['candidate_id'] for r in scores['requests'] if r['valid'] and not r['exact']}
    selected = []
    used = set()
    for typ in policy['selection']['type_order']:
        candidates = [q for q in pub.values() if q['question_type'] == typ and q['candidate_id'] in valid_wrong]
        candidates.sort(key=lambda q: hashlib.sha256(('2026101900|evidence-hint|' + q['candidate_id']).encode()).hexdigest())
        chosen = 0
        for q in candidates:
            if q['root_id'] in used:
                continue
            selected.append(q)
            used.add(q['root_id'])
            chosen += 1
            if chosen == policy['selection']['per_type_max']:
                break
    rt.require(bool(selected), 'NO_ELIGIBLE_VALID_CONTENT_FAILURES')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m0['tokenizer'], local_files_only=True)
    oldmaps = rt.read(base / 'private/DISPLAY_MAPPINGS.json')
    rows = []
    maps = {}
    promptids = {}
    hints = {}
    pairs = []
    prompts = {}
    target.mkdir(parents=True)
    for q in selected:
        qid = q['candidate_id']
        source = rt.read(q['source_path'])
        evidence = priv[qid]['option_evidence' if q['question_type'] == 'multi_select' else 'item_evidence']
        refs = set()
        for item in evidence.values():
            citations = item.get('citations') or [item.get('primary_citation') or item]
            for c in citations:
                rt.require(bool(c.get('turn_ref')), 'EVIDENCE_TURN_REQUIRED')
                refs.add(c['turn_ref'])
        relevant = [t for t in source['turns'] if t['turn_ref'] in refs]
        rt.require(len(relevant) == len(refs), 'HINT_TURN_NOT_IN_SOURCE')
        reminder = '\n\nRelevant-source reminder (use the complete dialogue and evaluate every item independently; this reminder does not identify correct labels):\n'
        reminder += '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in relevant)
        hints[qid] = {'turn_refs': [t['turn_ref'] for t in relevant], 'text': reminder,
                      'source_sha256': q['source_sha256'], 'all_item_evidence_included': True,
                      'gold_relations_or_label_mapping_included': False}
        for old in m0['requests']:
            if old['candidate_id'] != qid:
                continue
            row = copy.deepcopy(old)
            oldid = old['request_id']
            rid = oldid + '__evidence_hint'
            row['request_id'] = rid
            row['baseline_request_id'] = oldid
            row['intervention'] = 'all_item_reviewed_evidence_turns_reminder'
            text = (base / 'prompts' / (oldid + '.txt')).read_text()
            marker = '<|im_end|>\n<|im_start|>assistant'
            rt.require(text.count(marker) == 1, 'CHAT_INSERTION_BOUNDARY_INVALID')
            point = text.index(marker)
            prompt = text[:point] + reminder + text[point:]
            ids = tok.encode(prompt, add_special_tokens=False)
            row['prompt_sha256'] = rt.textsha(prompt)
            row['prompt_tokens'] = len(ids)
            rt.require(len(ids) + 32 <= m0['max_model_len'], 'PROBE_OVERFLOW_NO_TRUNCATION')
            rows.append(row)
            prompts[rid] = prompt
            promptids[rid] = ids
            maps[rid] = oldmaps['requests'][oldid]
            pairs.append({'baseline_request_id': oldid, 'probe_request_id': rid, 'candidate_id': qid,
                          'root_id': q['root_id'], 'same_seed': row['seed'],
                          'same_variant_id': row['variant_id'], 'slot': row['slot']})
        source_target = target / 'sources' / (q['root_id'] + '.json')
        source_target.parent.mkdir(exist_ok=True)
        shutil.copy2(q['source_path'], source_target)
    rt.require(len(rows) <= 90, 'PROBE_BUDGET_EXCEEDED')
    rt.save(target / 'PUBLIC_QUESTIONS.json', {'questions': selected})
    rt.save(target / 'private/DISPLAY_MAPPINGS.json', {'requests': maps,
            'public_to_canonical': {q['candidate_id']: oldmaps['public_to_canonical'][q['candidate_id']] for q in selected}})
    rt.save(target / 'PROMPT_TOKEN_IDS.json', promptids)
    rt.save(target / 'EVIDENCE_HINTS.json', hints)
    rt.save(target / 'MATCHED_PAIRS.json', {'status': 'FROZEN_BEFORE_PROBE', 'pairs': pairs})
    for rid, prompt in prompts.items():
        p = target / 'prompts' / (rid + '.txt')
        p.parent.mkdir(exist_ok=True)
        p.write_text(prompt)
    m = copy.deepcopy(m0)
    m.update({'status': 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT', 'prepared_at_utc': rt.now(),
              'purpose': 'controlled_evidence_hint_probe', 'question_count': len(selected),
              'root_count': len(used), 'request_count': len(rows), 'requests': rows,
              'baseline_manifest': rt.bind(base / 'manifest.json'), 'probe_policy': rt.bind(HERE / 'PROBE_SELECTION_POLICY.json'),
              'max_prompt_tokens': max(r['prompt_tokens'] for r in rows),
              'minimum_context_headroom': min(16384 - r['prompt_tokens'] - 32 for r in rows)})
    m['input_bindings'] += [rt.bind(p) for p in [base / 'SCORES.json', base / 'RAW_RESPONSE_FREEZE.json', HERE / 'PROBE_SELECTION_POLICY.json']]
    m['code_bindings'] += [rt.bind(Path(__file__))]
    rt.save(target / 'manifest.json', m)
    rt.save(target / 'PREFLIGHT_FREEZE.json', {'status': m['status'], 'created_at_utc': rt.now(),
            'files': [rt.bind(p) for p in sorted(target.rglob('*')) if p.is_file()],
            'questions': len(selected), 'requests': len(rows), 'inference_started': False})
    print(json.dumps({'status': m['status'], 'questions': len(selected), 'requests': len(rows),
                      'root_ids': [q['root_id'] for q in selected]}, ensure_ascii=False))


if __name__ == '__main__':
    prepare()
