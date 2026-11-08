"""Account for the fixed population without changing sources or admitting training data.

New and historical cohorts remain separate. This is delivery/reporting code,
not a source reviewer, question editor, Solver runner, or training-pool writer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
P = B.parent.parent
sys.path[:0] = [str(P / 'scripts'), str(B / 'tools')]
from codex_evidence_export import export_root_questions
from codex_sequential_ab import SCENES, inspect_root, load_config
from status import population


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(row): assert sha(row['path']) == row['sha256'], 'BOUND_FILE_CHANGED:' + row['path']
def save(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
def textsave(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: f.write(data)


def interval(start, end):
    return (datetime.fromisoformat(end.replace('Z', '+00:00')) - datetime.fromisoformat(start.replace('Z', '+00:00'))).total_seconds()


def receipt_timing(unit):
    rows = []
    for p in sorted(unit.glob('roots/*/scenes/*/AUTHOR_RECEIPT.json')):
        r = read(p)
        if r.get('started_at_utc') and r.get('finished_at_utc'):
            rows.append({'stage': p.parent.name, 'author_wall_seconds': interval(r['started_at_utc'], r['finished_at_utc']), 'receipt_binding': bind(p)})
    for p in unit.glob('roots/*/QA_AUTHOR_RECEIPT.json'):
        r = read(p)
        if r.get('started_at_utc') and r.get('finished_at_utc'):
            rows.append({'stage': 'QA', 'author_wall_seconds': interval(r['started_at_utc'], r['finished_at_utc']), 'receipt_binding': bind(p)})
    return {'author_receipt_intervals': rows,
            'scope': 'Observed wall-clock receipt intervals, including concurrent work and waits; not billed inference time or usage.'}


def require_audit(path):
    d = read(path)
    assert d['status'] == 'PASS' and not d.get('errors')
    for item in d.get('input_bindings', []): verify(item)
    return d


def paired_wave(wave):
    scope = B / 'evaluations' / wave
    adapter = read(scope / 'ADAPTER_BINDINGS.json')
    for item in adapter['files']: verify(item)
    if not adapter['candidate_count']:
        assert adapter['calls_per_model'] == 0
        return {}, {}, {'status': 'EMPTY_CONTENT_POOL', 'new_calls': 0}
    pre = read(scope / 'PAIRED_PREFLIGHT.json')
    assert pre['status'] == 'PASS'
    for item in [*pre['files'], pre['adapter_binding']]: verify(item)
    audit = require_audit(scope / 'INDEPENDENT_SOLVER_AUDIT.json')
    results, runs = {}, {}
    for side in ['base', 'trained']:
        run = scope / 'run' / side; runs[side] = run
        result = read(run / 'RESULTS.json'); results[side] = result
        complete = read(run / 'COMPLETE.json')
        wrapper = read(scope / (side.upper() + '_COMPLETE.json'))
        assert complete['status'] == 'COMPLETE'
        assert wrapper['actual_calls'] == complete['actual_call_count'] == pre['calls_per_model'] == adapter['calls_per_model']
        assert not result['call_set_errors']
        assert audit['results'][side]['calls'] == complete['actual_call_count']
    for name in ['questions.json', 'gold.json']:
        assert read(runs['base'] / name) == read(runs['trained'] / name)
    return results, runs, {'status': 'PAIRED_COMPLETED_AND_INDEPENDENTLY_RECOMPUTED',
                          'calls_per_model': adapter['calls_per_model'], 'preflight_binding': bind(scope / 'PAIRED_PREFLIGHT.json'),
                          'audit_binding': bind(scope / 'INDEPENDENT_SOLVER_AUDIT.json')}


def candidate_keys(q):
    keys = [q['question_type'], 'root:' + q['root_id']]
    if q['question_type'] == 'multi_select':
        k = len(q['gold']); keys.append('gold' + str(k))
        if 2 <= k <= 5: keys.append('multi_select_gold2_5')
        if q.get('role') == 'control6': keys.append('gold6_control')
    return keys


def summarize_calls(qmap, run):
    buckets = defaultdict(list)
    manifest = read(run / 'manifest.json')
    request_map = {r['request_id']: r for r in manifest['requests']}
    assert len(request_map) == manifest['request_count']
    seen = set()
    for p in sorted((run / 'responses').glob('*.json')):
        r = read(p); request = request_map[r['request_id']]; q = qmap[r['candidate_id']]
        assert r['request_id'] not in seen; seen.add(r['request_id'])
        valid = r.get('evaluation_valid') is True
        entry = {'valid': valid, 'exact': bool(r.get('score', {}).get('exact_match')) if valid else False}
        if valid:
            entry['partial'] = Fraction(r['score']['partial_score_fraction'])
            if q['question_type'] == 'multi_select':
                labels = [s.strip().upper() for s in r['raw_output'].strip()[5:-6].split(',') if s.strip()]
                predicted = {request['variant']['display_to_content'][label] for label in labels}
                expected = set(q['gold'])
                entry.update(false_positives=len(predicted - expected), false_negatives=len(expected - predicted),
                             predicted_size=len(predicted), gold_size=len(expected), all_six=len(predicted) == 6,
                             over_count=len(predicted) > len(expected), under_count=len(predicted) < len(expected))
        for key in candidate_keys(q): buckets[key].append(entry)
    assert seen == set(request_map)
    out = {}
    for key, rows in buckets.items():
        valid = [r for r in rows if r['valid']]
        item = {'calls': len(rows), 'valid_calls': len(valid), 'invalid_calls': len(rows) - len(valid),
                'correct': sum(r['exact'] for r in rows), 'exact_rate': sum(r['exact'] for r in rows) / len(rows),
                'partial_mean': str(sum((r['partial'] for r in valid), Fraction()) / len(valid)) if valid else None}
        mcq = [r for r in valid if 'predicted_size' in r]
        if mcq:
            item.update(false_positive_total=sum(r['false_positives'] for r in mcq),
                        false_negative_total=sum(r['false_negatives'] for r in mcq),
                        calls_with_false_positive=sum(r['false_positives'] > 0 for r in mcq),
                        calls_with_false_negative=sum(r['false_negatives'] > 0 for r in mcq),
                        selected_all_six_calls=sum(r['all_six'] for r in mcq),
                        over_count_calls=sum(r['over_count'] for r in mcq), under_count_calls=sum(r['under_count'] for r in mcq),
                        mean_predicted_size=statistics.mean(r['predicted_size'] for r in mcq))
        out[key] = item
    return out


def historical(row, overlay):
    old = Path(row['reuse_batch']); rid = row['root_id']
    for item in row['reuse_bindings']: verify(item)
    pool = read(old / 'evaluation/EVALUATION_POOL.json')['questions']
    qs = {q['candidate_id']: q for q in pool if q['root_id'] == rid}
    chosen = {q['candidate_id']: q for q in read(old / 'SELECTED_CANDIDATES.json')['questions'] if q['root_id'] == rid}
    quarantined = set(overlay['quarantined_original_ids'])
    if old.name.endswith('_v113'):
        chosen = {q['candidate_id']: q for q in overlay['questions'] if q['root_id'] == rid}
        qs = {cid: chosen.get(cid, q) for cid, q in qs.items() if cid not in quarantined}
        qs.update(chosen)
    assert set(qs).isdisjoint(quarantined) and set(chosen) <= set(qs)
    assert all(q.get('role') != 'control6' for q in chosen.values())
    return list(qs.values()), list(chosen.values()), old / 'roots' / ('root' + rid.rsplit('-', 1)[1]), {
        'original_batch': str(old), 'historical_solver_audit': bind(old / 'INDEPENDENT_SOLVER_AUDIT.json'),
        'selection_artifact': bind(B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json' if old.name.endswith('_v113') else old / 'SELECTED_CANDIDATES.json'),
        'legacy_evidence_export': not all(q.get('evidence_export', {}).get('all_reviewed_citations_preserved') for q in qs.values()),
        'rerun_solver': False}


def verify_candidates(questions):
    ids = [q['candidate_id'] for q in questions]
    assert len(ids) == len(set(ids)), 'DUPLICATE_CANDIDATE_ID'
    for q in questions:
        path = Path(q['source_path']); assert sha(path) == q['source_sha256']
        source = read(path); assert source['root_id'] == q['root_id']
        turns = {t['turn_ref']: t for t in source['turns']}
        field = 'option_evidence' if q['question_type'] == 'multi_select' else 'item_evidence'
        assert field in q, 'DELIVERY_EVIDENCE_MISSING:' + q['candidate_id']
        for ev in q[field].values():
            cites = [ev, *ev.get('citations', []), *ev.get('alternative_matching_anchors', [])]
            for c in cites:
                t = turns[c['turn_ref']]
                assert c['quote'] and c['quote'] in t['text'], q['candidate_id']
                if 'speaker' in c: assert c['speaker'] == t['speaker']


def summarize(wave, output):
    output = Path(output).resolve(); assert not output.exists(), 'DELIVERY_OUTPUT_ALREADY_EXISTS'
    m = read(B / 'MANIFEST.json'); states = {r['root_id']: r for r in population()['roots']}
    target = m['roots'] if wave == 'all' else [r for r in m['roots'] if r['wave'] == wave and r['action'] in ['SCREEN', 'INHERIT_SCREEN']]
    assert target and all(states[r['root_id']]['terminal'] for r in target), 'WAIT_FOR_FIXED_POPULATION_TERMINAL'
    audit_name = 'ALL_CONTENT_AUDIT.json' if wave == 'all' else wave.upper() + '_CONTENT_AUDIT.json'
    audit = require_audit(B / 'evidence' / audit_name)
    for item in read(B / 'MASTER_FREEZE.json')['files']: verify(item)
    protected = read(B / 'evidence/HISTORICAL_PROTECTION.json')['files']
    for item in protected: verify(item)
    waves = ['validation', 'production'] if wave == 'all' else [wave]
    pairs = {w: paired_wave(w) for w in waves}
    overlay = read(B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json')
    roots, all_questions, selected, provenance, excluded = [], [], [], {}, []
    for row in target:
        rid = row['root_id']; unit = Path(row['unit_path']); state = states[rid]
        item = {**state, 'baseline_binding': row['baseline_binding'], 'baseline_turn_count': row['baseline_turn_count']}
        questions, chosen = [], []; folder = None
        if row['action'] == 'REUSE':
            questions, chosen, folder, prov = historical(row, overlay); provenance[rid] = prov
        elif state['status'] == 'NEW_CONTENT_COMPLETED':
            folder = unit / 'roots' / ('root' + rid.rsplit('-', 1)[1])
            questions = export_root_questions(folder)
            scope_pool = read(B / 'evaluations' / row['wave'] / 'EVALUATION_POOL.json')['questions']
            frozen_by_id = {q['candidate_id']: q for q in scope_pool}
            for q in questions:
                q['source_path'] = str(Path(q['source_path']).resolve())
                assert frozen_by_id[q['candidate_id']] == q
            results = pairs[row['wave']][0]['trained']
            chosen_ids = {q['candidate_id'] for q in results['questions'] if q['selected_by_policy']}
            chosen = [q for q in questions if q['candidate_id'] in chosen_ids and q.get('role') != 'control6']
            for r in read(folder / 'FINAL_CROSSCHECK.json')['results']:
                if r['status'] != 'CONTENT_PASS': excluded.append({'root_id': rid, 'phase': 'content', **r})
            item['crosscheck_binding'] = bind(folder / 'FINAL_CROSSCHECK.json')
        elif state.get('ready_scenes') == 6:
            folder = unit / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        if row['action'] not in ['REUSE', 'HISTORICAL_HOLD']:
            item['timing'] = receipt_timing(unit)
            cp = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'; item['baseline_gate_binding'] = bind(cp)
            item['baseline_decision'] = read(cp)['decision']
            if (unit / 'RUN_CONFIG.json').exists():
                detailed = inspect_root(unit, load_config(unit), rid, record_reviews=False)
                item['ready_by_route'] = {route: sum(s['status'] == 'READY' and f':{route}:' in s['scene_ref'] for s in detailed['scenes']) for route in 'AB'}
                item['failed_scenes'] = [s for s in detailed['scenes'] if s['status'] in ['STRUCTURAL_FAILURE','INTEGRITY_FAILURE','REVIEW_INVALID','REVIEW_HOLD']]
                checks = Counter()
                for p in unit.glob('roots/*/scenes/*/SOURCE_REVIEW.json'):
                    review = read(p)
                    for name, value in review['checks'].items(): checks[name + ':' + value['status']] += 1
                item['scene_check_counts'] = dict(checks)
        item.update(content_questions=len(questions), selected_candidates=len(chosen),
                    content_type_counts=dict(Counter(q['question_type'] for q in questions)),
                    content_gold_counts=dict(Counter(str(len(q['gold'])) for q in questions if q['question_type'] == 'multi_select')),
                    selected_type_counts=dict(Counter(q['question_type'] for q in chosen)),
                    selected_gold_counts=dict(Counter(str(len(q['gold'])) for q in chosen if q['question_type'] == 'multi_select')))
        if folder is not None:
            source_path = folder / 'CUMULATIVE_SOURCE.json'
            source = read(source_path); turns = source['turns']; baseline = read(row['baseline_path'])['turns']
            assert turns[:len(baseline)] == baseline, 'BASELINE_PREFIX_CHANGED:' + rid
            counts = Counter(t.get('scene_ref') for t in turns if t.get('scene_ref') in SCENES)
            assert set(counts) == set(SCENES) and all(12 <= n <= 16 for n in counts.values())
            item.update(complete_dialogue=True, source_binding=bind(source_path),
                        total_turns=len(turns), new_turns=sum(counts.values()), scene_turn_counts=dict(counts))
            item['_folder'] = str(folder)
        else: item['complete_dialogue'] = False
        roots.append(item); all_questions.extend(questions); selected.extend(chosen)
    verify_candidates(all_questions); verify_candidates(selected)
    assert all(q.get('role') != 'control6' for q in selected)
    assert not set(overlay['quarantined_original_ids']) & {q['candidate_id'] for q in all_questions + selected}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    lengths = []
    for item in roots:
        if not item['complete_dialogue']: continue
        folder = Path(item.pop('_folder')); source = read(folder / 'CUMULATIVE_SOURCE.json'); turns = source['turns']
        new = [t for t in turns if t.get('scene_ref') in SCENES]
        formatted = '\n'.join(t['speaker'] + ': ' + t['text'] for t in turns)
        new_text = '\n'.join(t['speaker'] + ': ' + t['text'] for t in new)
        item['dialogue_tokens_speakers_and_body'] = len(tok.encode(formatted, add_special_tokens=False))
        item['new_dialogue_tokens_speakers_and_body'] = len(tok.encode(new_text, add_special_tokens=False))
        lengths.append({'root_id': item['root_id'], 'total_tokens': item['dialogue_tokens_speakers_and_body'], 'new_tokens': item['new_dialogue_tokens_speakers_and_body']})
        short = item['root_id'].rsplit('-', 1)[1]; out = output / 'roots' / ('root' + short)
        lines = [f'# {item["root_id"]}', '', 'Source: ' + str(folder / 'CUMULATIVE_SOURCE.json'), 'SHA256: ' + sha(folder / 'CUMULATIVE_SOURCE.json'), '']
        lines.extend('[' + t['turn_ref'] + '] ' + t['speaker'] + ': ' + t['text'] for t in turns)
        textsave(out / 'FULL_DIALOGUE.md', '\n\n'.join(lines) + '\n')
        qlines = ['# 题目、答案及来源', '', '内容通过题全部展示；入选状态见候选文件。', '']
        chosen_ids = {q['candidate_id'] for q in selected}
        for q in all_questions:
            if q['root_id'] != item['root_id']: continue
            qlines += ['## ' + q['candidate_id'], '', q['question'], '']
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            qlines += [label + ': ' + text for label, text in q[field].items()]
            qlines += ['', 'Gold: ' + ', '.join(q['gold']), '入选候选: ' + str(q['candidate_id'] in chosen_ids), '来源: ' + q['source_path'], '']
        textsave(out / 'QUESTIONS_AND_ANSWERS.md', '\n'.join(qlines))
        item['readable_dialogue_path'] = str(out / 'FULL_DIALOGUE.md'); item['readable_questions_path'] = str(out / 'QUESTIONS_AND_ANSWERS.md')
    metrics = {}
    for w, (results, runs, pair_info) in pairs.items():
        qmap = {q['candidate_id']: q for q in read(B / 'evaluations' / w / 'EVALUATION_POOL.json')['questions']}
        metrics[w] = {'pairing': pair_info, 'by_side': {side: summarize_calls(qmap, run) for side, run in runs.items()}}
        metrics[w]['wrapper_wall_seconds'] = {side: interval(read(B / 'evaluations' / w / (side.upper() + '_STARTED.json'))['started_at_utc'],
                                                            read(B / 'evaluations' / w / (side.upper() + '_COMPLETE.json'))['completed_at_utc']) for side in runs}
    new_roots = [r for r in roots if r['action'] in ['SCREEN', 'INHERIT_SCREEN']]
    reuse_roots = [r for r in roots if r['action'] == 'REUSE']
    summary = {'status': 'FULL_FIXED_POPULATION_PROCESSED' if wave == 'all' else 'FIXED_VALIDATION_WAVE_PROCESSED',
               'created_at_utc': now(), 'wave': wave, 'population_denominator': len(roots), 'terminal_roots': len(roots),
               'new_fixed_attempt_roots': len(new_roots), 'reused_complete_roots': len(reuse_roots),
               'complete_dialogues': sum(r['complete_dialogue'] for r in roots),
               'new_complete_dialogues': sum(r['complete_dialogue'] for r in new_roots),
               'statuses': dict(Counter(r['status'] for r in roots)),
               'cohort_results': {c: {'roots': len([r for r in new_roots if r['cohort'] == c]),
                                        'statuses': dict(Counter(r['status'] for r in new_roots if r['cohort'] == c)),
                                        'complete_dialogues': sum(r['complete_dialogue'] for r in new_roots if r['cohort'] == c)} for c in sorted({r['cohort'] for r in new_roots})},
               'content_questions': len(all_questions), 'selected_candidates': len(selected),
               'new_content_questions': sum(r['content_questions'] for r in new_roots),
               'new_selected_candidates': sum(r['selected_candidates'] for r in new_roots),
               'content_type_counts': dict(Counter(q['question_type'] for q in all_questions)),
               'content_gold_counts': dict(Counter(str(len(q['gold'])) for q in all_questions if q['question_type'] == 'multi_select')),
               'selected_type_counts': dict(Counter(q['question_type'] for q in selected)),
               'selected_gold_counts': dict(Counter(str(len(q['gold'])) for q in selected if q['question_type'] == 'multi_select')),
               'mean_dialogue_tokens_speakers_and_body': statistics.mean(r['total_tokens'] for r in lengths) if lengths else None,
               'mean_new_dialogue_tokens_speakers_and_body': statistics.mean(r['new_tokens'] for r in lengths) if lengths else None,
               'tokenizer': '/root/autodl-tmp/Qwen3-4B', 'tokens_are_billed_usage': False,
               'solver': metrics, 'reused_provenance': provenance,
               'historical_files_verified_unchanged': len(protected), 'content_audit_binding': bind(B / 'evidence' / audit_name),
               'population_manifest_binding': bind(B / 'MANIFEST.json'), 'accounting_tool_binding': bind(__file__),
               'training_started': False, 'training_pool_merged': False,
               'external_semantic_certification': False, 'unseen_holdout': False, 'exact_generator_model_id': None,
               'billed_generation_usage': None, 'generation_cost': None, 'roots': roots}
    summary['freeze_to_delivery_wall_seconds'] = interval(read(B / 'MASTER_FREEZE.json')['created_at_utc'], summary['created_at_utc'])
    save(output / 'CONTENT_QUESTIONS.json', {'questions': all_questions, 'historical_overlay_binding': bind(B.parent / 'h4_codex_wording_minfix_v114/RECOMMENDED_CANDIDATES.json')})
    save(output / 'SELECTED_CANDIDATES.json', {'questions': selected, 'status': 'STAGING_CANDIDATES_NOT_MERGED',
        'selection_rule': 'New: existing M_clean_policy complete-valid 0<equal-display mean partial<1; Gold6 excluded. Reuse: preserved historical selection with mandatory v114 quarantine/overlay.',
        'training_started': False, 'training_pool_merged': False})
    save(output / 'EXCLUDED_CANDIDATES.json', {'content_excluded': excluded, 'historical_quarantine': overlay['quarantined_original_ids'],
          'solver_exclusions_recorded_in': [str(B / 'evaluations' / w / 'run/trained/RESULTS.json') for w in waves]})
    save(output / 'FINAL_STATUS.json', summary)
    lines = ['# 固定全集扩充和联合出题', '',
             f"处理 {len(roots)} 个固定 root；完整对话 {summary['complete_dialogues']} 个，其中本批新增 {summary['new_complete_dialogues']} 个、复用 {len(reuse_roots)} 个。",
             '所有未通过 root 保留在固定分母中，未替换对象、修改原文或依据 Solver 分数重写。', '',
             f"内容题 {len(all_questions)} 道；入选暂存候选 {len(selected)} 道。候选 Gold 分布：{summary['selected_gold_counts']}；题型：{summary['selected_type_counts']}。", '',
             '|root|来源|状态|A/B READY|内容题|入选|', '|---|---|---|---|---:|---:|']
    for r in roots:
        route = r.get('ready_by_route', {'A': 3 if r['action'] == 'REUSE' else 0, 'B': 3 if r['action'] == 'REUSE' else 0})
        lines.append(f"|{r['root_id']}|{r['action']}|{r['status']}|{route['A']}/3，{route['B']}/3|{r['content_questions']}|{r['selected_candidates']}|")
    lines += ['', '## 本批配对 Solver', '', '|批次|题型|M0 正确/调用|既有训练后模型 正确/调用|', '|---|---|---:|---:|']
    for w, metric in metrics.items():
        if not metric['by_side']: continue
        for key in ['multi_select_gold2_5', 'gold2', 'gold3', 'gold4', 'gold5', 'ordering', 'gold6_control']:
            cells = [metric['by_side'][s].get(key, {'correct': 0, 'calls': 0}) for s in ['base', 'trained']]
            lines.append(f"|{w}|{key}|{cells[0]['correct']}/{cells[0]['calls']}|{cells[1]['correct']}/{cells[1]['calls']}|")
    lines += ['', f"完整对话平均 {summary['mean_dialogue_tokens_speakers_and_body']} tokens；新增六场平均 {summary['mean_new_dialogue_tokens_speakers_and_body']} tokens。使用本地 Qwen3-4B tokenizer，含 speaker 和正文，不是调用 usage。", '',
              '逐 root 的原始基线、匹配、首稿、审核、拦截和冻结证据仍在 units/。可读完整对话与题答在本交付目录 roots/root*/；答案和全部引用在 CONTENT_QUESTIONS.json。', '',
              '历史 13 个 root 保留旧审核流程及既有局限；v113 题应用 v114 隔离和证据覆盖，不将历史记录伪装成本批新生成或新认证。新批源先审流程由同一模型的不同会话代理执行，共享文件系统下身份与阅读属于可审计声明，不是密码学认证或外部语义认证。', '',
              '全部 root 有训练侧/会话曝光，不是未见过的 holdout；同一 family 的多选变体及多次展示相关，调用数不能当独立题目数。本轮未训练、未合并训练池。', '']
    textsave(output / 'REPORT_ZH.md', '\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ['status','population_denominator','complete_dialogues','new_complete_dialogues','content_questions','selected_candidates','selected_gold_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--wave', choices=['validation', 'all'], required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); summarize(a.wave, a.output)
