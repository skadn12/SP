"""Read-only accounting of immutable stages; create new reviewable delivery files."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'scripts'))
from codex_sequential_ab import SCENES, inspect_root, load_config

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    with Path(p).open('x') as f: json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def textsave(p, s):
    with Path(p).open('x') as f: f.write(s)

def verify_tooling():
    freeze = read(HERE / 'evidence/TOOLING_FREEZE.json')
    assert freeze['status'] == 'FROZEN_BEFORE_QUESTION_AUTHORING'
    expected = {str((HERE / name).resolve()) for name in
                ('evaluate.py', 'audit_solver.py', 'deliver.py')}
    assert expected.issubset({row['path'] for row in freeze['files']})
    for row in freeze['files']:
        assert sha(row['path']) == row['sha256'], 'TOOLING_HASH_CHANGED:' + row['path']
    frozen_at = datetime.fromisoformat(freeze['created_at_utc'].replace('Z', '+00:00'))
    for side in ('base', 'trained'):
        started = HERE / 'evaluation' / (side.upper() + '_STARTED.json')
        if started.exists():
            assert frozen_at <= datetime.fromisoformat(read(started)['started_at_utc'].replace('Z', '+00:00'))
    for qa in HERE.glob('roots/*/QA_DRAFT.json'):
        assert frozen_at.timestamp() <= qa.stat().st_mtime, 'TOOLING_FROZEN_AFTER_QUESTION_AUTHORING'
    return freeze

def require_source_audit():
    audit = read(HERE / 'SOURCE_PROVENANCE_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['errors'] == []
    assert audit['input_bindings'], 'SOURCE_AUDIT_BINDINGS_REQUIRED'
    for row in audit['input_bindings']:
        assert sha(row['path']) == row['sha256'], 'SOURCE_AUDITED_INPUT_CHANGED:' + row['path']
    required = {str((HERE / name).resolve()) for name in
                ('RUN_CONFIG.json', 'SOURCE_RESERVATIONS.json', 'EXECUTION_INPUTS_FROZEN.json')}
    required.update(str(p.resolve()) for p in HERE.glob('roots/*/scenes/**/*.json'))
    required.update(str(p.resolve()) for pattern in ('roots/*/CUMULATIVE_SOURCE.json', 'roots/*/SOURCE_REVIEW.json',
                                                     'roots/*/ROOT_STOP_RECEIPT.json') for p in HERE.glob(pattern))
    assert required.issubset({row['path'] for row in audit['input_bindings']}), 'SOURCE_AUDIT_COVERAGE_INCOMPLETE'
    cfg = load_config(HERE)
    assert {r['root_id'] for r in audit['roots']} == {r['root_id'] for r in cfg['roots']}
    return audit

def require_independent_audit():
    audit = read(HERE / 'INDEPENDENT_SOLVER_AUDIT.json')
    assert audit['status'] == 'PASS' and audit['errors'] == []
    assert set(audit['results']) == {'base', 'trained'}
    assert audit['input_bindings'], 'AUDIT_INPUT_BINDINGS_REQUIRED'
    for row in audit['input_bindings']:
        assert sha(row['path']) == row['sha256'], 'AUDITED_INPUT_CHANGED:' + row['path']
    for side in ('base', 'trained'):
        run = HERE / 'evaluation/run' / side
        manifest = read(run / 'manifest.json')
        assert audit['results'][side]['calls'] == manifest['request_count']
        assert audit['results'][side]['candidates'] == manifest['candidate_count']
        assert audit['results'][side]['errors'] == 0
    return audit

def require_completed_pair():
    """Do not publish a final result from a partial or stale paired run."""
    pre = read(HERE / 'evaluation/PAIRED_PREFLIGHT.json')
    binding = read(HERE / 'evaluation/ADAPTER_BINDINGS.json')
    assert pre['status'] == 'PASS'
    for row in [*pre['files'], pre['adapter_binding'], *binding['files']]:
        assert sha(row['path']) == row['sha256'], 'PAIRED_BOUND_FILE_CHANGED:' + row['path']
    assert pre['calls_per_model'] == binding['calls_per_model']
    for side in ('base', 'trained'):
        run = HERE / 'evaluation/run' / side
        complete = read(run / 'COMPLETE.json')
        wrapper = read(HERE / 'evaluation' / (side.upper() + '_COMPLETE.json'))
        assert complete['status'] == 'COMPLETE'
        assert complete['actual_call_count'] == wrapper['actual_calls'] == pre['calls_per_model']
        assert wrapper['side'] == side and wrapper['training_started'] is False
    for filename in ('questions.json', 'gold.json'):
        assert read(HERE / 'evaluation/run/base' / filename) == read(HERE / 'evaluation/run/trained' / filename)
    return pre

def root_accounting(folder, state):
    state['ready_by_route'] = {route: sum(s['status'] == 'READY' and f':{route}:' in s['scene_ref']
                                        for s in state['scenes']) for route in 'AB'}
    state['failed_scenes'] = [s for s in state['scenes'] if s['status'] in
                              {'STRUCTURAL_FAILURE', 'INTEGRITY_FAILURE', 'REVIEW_INVALID', 'REVIEW_HOLD'}]
    checks = [read(f)['checks']['actor_identity']['status'] for f in
              sorted((folder / 'scenes').glob('*/SOURCE_REVIEW.json'))]
    state['actor_identity_checked_scenes'] = len(checks)
    state['actor_identity_failed_scenes'] = sum(c == 'FAIL' for c in checks)
    state['actor_identity_warning_scenes'] = sum(c in {'WARNING', 'PASS_WITH_WARNING'} for c in checks)
    state['actor_identity_failure_rate'] = sum(c == 'FAIL' for c in checks) / len(checks) if checks else None
    state['actor_identity_rate_scope'] = 'Scene-level cumulative-source reviewer checks; warnings can include inherited baseline ambiguity, not turn-level independent accuracy.'
    state['content_pass_questions'] = 0
    state['content_pass_new_questions'] = 0
    state['content_excluded_questions'] = 0
    return state

def elapsed(started, finished):
    return (datetime.fromisoformat(finished.replace('Z', '+00:00')) -
            datetime.fromisoformat(started.replace('Z', '+00:00'))).total_seconds()

def timing_summary(summary):
    roots = []
    for root in summary['roots']:
        folder = HERE / 'roots' / ('root' + root['root_id'].rsplit('-', 1)[1])
        scenes = []
        for p in sorted((folder / 'scenes').glob('*/AUTHOR_RECEIPT.json')):
            r = read(p)
            if r.get('started_at_utc') and r.get('finished_at_utc'):
                scenes.append({'scene': p.parent.name, 'author_receipt_seconds': elapsed(r['started_at_utc'], r['finished_at_utc'])})
        row = {'root_id': root['root_id'], 'scene_author_receipt_intervals': scenes,
               'sum_author_receipt_seconds': sum(s['author_receipt_seconds'] for s in scenes)}
        q = folder / 'QA_AUTHOR_RECEIPT.json'
        if q.exists():
            receipt = read(q)
            if receipt.get('started_at_utc') and receipt.get('finished_at_utc'):
                row['qa_author_receipt_seconds'] = elapsed(receipt['started_at_utc'], receipt['finished_at_utc'])
        roots.append(row)
    solver = {side: elapsed(read(HERE / 'evaluation' / (side.upper() + '_STARTED.json'))['started_at_utc'],
                            read(HERE / 'evaluation' / (side.upper() + '_COMPLETE.json'))['completed_at_utc'])
              for side in ('base', 'trained')}
    return {'batch_setup_to_delivery_seconds': elapsed(read(HERE / 'PRECALL_PLAN.json')['created_at_utc'], now()),
            'root_receipt_intervals': roots, 'solver_wrapper_seconds': solver,
            'scope': 'Wall-clock receipt intervals; parallel intervals can overlap, author intervals need not include complete rereads; not billed inference time.'}

def content():
    verify_tooling()
    cfg = load_config(HERE); roots = []; public = []; gold = []; excluded = []; outcomes = []
    for r in cfg['roots']:
        rid = r['root_id']; folder = HERE / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        state = root_accounting(folder, inspect_root(HERE, cfg, rid))
        drafts = [read(f) for f in (folder / 'scenes').glob('*/EXPANSION_DRAFT.json')]
        state['attempted_new_turns'] = sum(len(d['turns']) for d in drafts)
        state['accepted_new_turns'] = sum(s.get('new_turn_count', 0) for s in state['scenes'] if s['status'] == 'READY')
        if state['full_expansion_complete']:
            assert (folder / 'CUMULATIVE_SOURCE.json').read_bytes() == (folder / 'scenes/B_S03/CUMULATIVE_SOURCE.json').read_bytes()
            assert (folder / 'FINAL_CROSSCHECK.json').exists(), 'COMPLETE_ROOT_QA_NOT_FINALIZED:' + rid
            src = read(folder / 'CUMULATIVE_SOURCE.json')
            state['cumulative_turn_count'] = len(src['turns'])
            lines = [f'# {rid} 完整累计对话', '', 'P／R01 为原有历史；R02:A 三场延续旧事件，R02:B 三场引入新事件。', '']
            last = None
            for t in src['turns']:
                scene = t.get('scene_ref') or t['session_ref']
                if scene != last: lines += ['', '## ' + scene, '']; last = scene
                lines.append(f"[{t['turn_ref']}] **{t['speaker']}**: {t['text']}")
            textsave(folder / 'FULL_DIALOGUE.md', '\n'.join(lines) + '\n')
        if (folder / 'FINAL_CROSSCHECK.json').exists():
            qs = read(folder / 'PUBLIC_QUESTIONS.json')['questions']
            gs = {q['candidate_id']: q for q in read(folder / 'PRIVATE_GOLD.json')['questions']}
            checks = {q['candidate_id']: q for q in read(folder / 'FINAL_CROSSCHECK.json')['results']}
            qlines = [f'# {rid} 题目及原始标签答案', '']
            for q in qs:
                g = gs[q['candidate_id']]; c = checks[q['candidate_id']]
                outcomes.append(c)
                qlines += ['## ' + q['candidate_id'], '', q['question'], '']
                field = 'options' if q['question_type'] == 'multi_select' else 'items'
                qlines += [f'{k}. {v}' for k, v in q[field].items()]
                qlines += ['', '**原始标签答案：** ' + ','.join(g['gold']), '**状态：** ' + c['status'], '']
                if c['status'] == 'CONTENT_PASS':
                    public.append(q); gold.append(g)
                    state['content_pass_questions'] += 1
                    state['content_pass_new_questions'] += g['role'] != 'control6'
                else:
                    excluded.append({'question': q, 'private_gold': g, 'crosscheck': c})
                    state['content_excluded_questions'] += 1
            textsave(folder / 'QUESTIONS_AND_ANSWERS.md', '\n'.join(qlines) + '\n')
        roots.append(state)
    save(HERE / 'CONTENT_PUBLIC_QUESTIONS.json', {'questions': public})
    save(HERE / 'CONTENT_PRIVATE_GOLD.json', {'questions': gold})
    save(HERE / 'EXCLUDED_CANDIDATES.json', {'questions': excluded})
    counts = Counter(q['question_type'] for q in public)
    gp = {q['candidate_id']: q for q in gold}
    cards = Counter(str(len(gp[q['candidate_id']]['gold'])) for q in public if q['question_type'] == 'multi_select')
    result = {'created_at_utc': now(), 'attempted_roots': len(roots),
              'complete_A3B3_roots': sum(r['full_expansion_complete'] for r in roots),
              'complete_A3B3_rate': sum(r['full_expansion_complete'] for r in roots) / len(roots),
              'actor_identity_checked_scenes': sum(r['actor_identity_checked_scenes'] for r in roots),
              'actor_identity_failed_scenes': sum(r['actor_identity_failed_scenes'] for r in roots),
              'actor_identity_warning_scenes': sum(r['actor_identity_warning_scenes'] for r in roots),
              'expected_scene_count': 6 * len(roots),
              'attempted_scene_count': sum(len(list((HERE / 'roots' / ('root' + r['root_id'].rsplit('-',1)[1]) / 'scenes').glob('*/EXPANSION_DRAFT.json'))) for r in roots),
              'ready_scene_count': sum(r['ready_scene_count'] for r in roots),
              'attempted_new_turns': sum(r['attempted_new_turns'] for r in roots),
              'accepted_new_turns': sum(r['accepted_new_turns'] for r in roots),
              'roots': roots, 'content_pass_type_counts': dict(counts), 'multi_gold_counts': dict(cards),
              'new_questions': sum(g['role'] != 'control6' for g in gold),
              'gold6_controls': sum(g['role'] == 'control6' for g in gold),
              'excluded_public_questions': len(excluded), 'all_public_outcomes': outcomes,
              'training_started': False, 'status': 'CONTENT_REVIEWED_NOT_SOLVER_VALIDATED'}
    denominator = result['actor_identity_checked_scenes']
    result['actor_identity_failure_rate'] = result['actor_identity_failed_scenes'] / denominator if denominator else None
    save(HERE / 'CONTENT_SUMMARY.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('roots','all_public_outcomes')}, ensure_ascii=False))

def final():
    verify_tooling()
    paired = require_completed_pair()
    require_independent_audit()
    source_audit = require_source_audit()
    summary = read(HERE / 'CONTENT_SUMMARY.json')
    pool = read(HERE / 'evaluation/EVALUATION_POOL.json')['questions']
    qmap = {q['candidate_id']: q for q in pool}; metrics = {}; selected = []
    for side in ('base', 'trained'):
        run = HERE / 'evaluation/run' / side
        result = read(run / 'RESULTS.json')
        assert result['call_set_errors'] == []
        records = [read(f) for f in (run / 'responses').glob('*.json')]
        assert len(records) == read(run / 'COMPLETE.json')['actual_call_count']
        assert all(r['evaluation_valid'] for r in records)
        bytype = {}
        groups = [('multi_select_gold2_5', lambda q: q['question_type'] == 'multi_select' and len(q['gold']) < 6),
                                ('gold6_control', lambda q: q['question_type'] == 'multi_select' and len(q['gold']) == 6),
                                ('ordering', lambda q: q['question_type'] == 'ordering')]
        groups += [(f'gold{card}', lambda q, card=card: q['question_type'] == 'multi_select' and len(q['gold']) == card)
                   for card in (2, 3, 4, 5)]
        for name, predicate in groups:
            sub = [r for r in records if predicate(qmap[r['candidate_id']])]
            bytype[name] = {'correct': sum(r['score']['exact_match'] for r in sub), 'calls': len(sub),
                            'accuracy': sum(r['score']['exact_match'] for r in sub) / len(sub) if sub else None,
                            'question_count': len({r['candidate_id'] for r in sub})}
        perroot = {}
        for rid in {q['root_id'] for q in pool}:
            perroot[rid] = {}
            for name, predicate in groups:
                sub = [r for r in records if r['root_id'] == rid and predicate(qmap[r['candidate_id']])]
                perroot[rid][name] = {'correct': sum(r['score']['exact_match'] for r in sub), 'calls': len(sub),
                                     'accuracy': sum(r['score']['exact_match'] for r in sub) / len(sub) if sub else None}
        metrics[side] = {'model': result['solver_model'], 'valid_calls': len(records), 'by_type': bytype, 'by_root': perroot,
                         'questions': result['questions'], 'call_set_errors': []}
        if side == 'trained':
            selected = [qmap[q['candidate_id']] for q in result['questions'] if q['selected_by_policy'] is True
                        and qmap[q['candidate_id']]['role'] != 'control6']
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    lengths = []
    for r in summary['roots']:
        if not r['full_expansion_complete']: continue
        f = HERE / 'roots' / ('root' + r['root_id'].rsplit('-',1)[1]) / 'CUMULATIVE_SOURCE.json'
        turns = read(f)['turns']
        new_turns = [t for t in turns if t.get('scene_ref') in SCENES]
        formatted = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        plain = '\n'.join(f"{t['speaker']}: {t['text']}" for t in turns)
        lengths.append({'root_id': r['root_id'], 'turns': len(turns),
                        'dialogue_tokens_with_refs_and_speakers': len(tok.encode(formatted, add_special_tokens=False)),
                        'dialogue_tokens_speakers_and_body': len(tok.encode(plain, add_special_tokens=False)),
                        'new_turns': len(new_turns),
                        'new_dialogue_tokens_speakers_and_body': len(tok.encode('\n'.join(f"{t['speaker']}: {t['text']}" for t in new_turns), add_special_tokens=False))})
    protected = read(HERE / 'evidence/HISTORICAL_PROTECTION.json')['files']
    assert all(sha(r['path']) == r['sha256'] for r in protected)
    frozen = read(HERE / 'EXECUTION_INPUTS_FROZEN.json')['files']
    assert all(sha(r['path']) == r['sha256'] for r in frozen)
    result = {'completed_at_utc': now(), 'status': 'COMPLETE_SERIALIZED_A3B3_VALIDATION_NO_TRAINING',
              'content': summary, 'solver': metrics, 'selected_new_question_count': len(selected),
              'selected_type_counts': dict(Counter(q['question_type'] for q in selected)),
              'dialogue_lengths': lengths, 'historical_files_verified_unchanged': len(protected),
              'mean_dialogue_tokens_speakers_and_body': sum(l['dialogue_tokens_speakers_and_body'] for l in lengths) / len(lengths) if lengths else None,
              'timing': timing_summary(summary),
              'paired_preflight_verified': True, 'source_provenance_audit_status': source_audit['status'],
              'training_started': False, 'external_semantic_certification': False}
    save(HERE / 'SELECTED_CANDIDATES.json', {'questions': selected, 'training_started': False,
         'selection_rule': 'M_clean_policy complete-valid 0<mean_partial<1; Gold6 controls excluded'})
    save(HERE / 'FINAL_STATUS.json', result)
    lines = ['# 自动编号 A3＋B3 流程复测 v113', '',
             f"尝试 {summary['attempted_roots']} 个已暴露训练 root；完整完成 {summary['complete_A3B3_roots']} 个，完成率 {summary['complete_A3B3_rate']:.1%}。",
             f"场景 READY {summary['ready_scene_count']}/{summary['expected_scene_count']}；新增有效发言 {summary['accepted_new_turns']} 条。失败 root 保留在分母，未替换 root／事件或修写失败稿。", '',
             '|root|A READY/3|B READY/3|完整|新增题内容通过|失败原因|', '|---|---:|---:|---|---:|---|']
    for root in summary['roots']:
        reasons = '; '.join(s.get('reason', s['status']) for s in root['failed_scenes']) or '无'
        lines.append(f"|{root['root_id']}|{root['ready_by_route']['A']}|{root['ready_by_route']['B']}|{root['full_expansion_complete']}|{root['content_pass_new_questions']}|{reasons}|")
    lines += ['', f"人物归属检查 FAIL {summary['actor_identity_failed_scenes']}/{summary['actor_identity_checked_scenes']} 场；WARNING {summary['actor_identity_warning_scenes']} 场。此为逐场累计对话审核口径，警告可能来自继承历史，不等于逐句人物混淆率。", '',
              f"内容通过 {summary['new_questions']} 道新增题及 {summary['gold6_controls']} 道 Gold6 对照；Gold 分布 {summary['multi_gold_counts']}。", '',
              '|题型|M0 正确／调用|M_clean_policy 正确／调用|', '|---|---:|---:|']
    for key, label in [('multi_select_gold2_5','多选 Gold2–5'), ('gold2','Gold2'), ('gold3','Gold3'), ('gold4','Gold4'), ('gold5','Gold5'), ('ordering','排序'), ('gold6_control','Gold6 对照')]:
        cells = [metrics[s]['by_type'][key] for s in ('base','trained')]
        lines.append(f"|{label}|{cells[0]['correct']}/{cells[0]['calls']}|{cells[1]['correct']}/{cells[1]['calls']}|")
    lines += ['', f"筛选保留 {len(selected)} 道新增候选（{dict(Counter(q['question_type'] for q in selected))}），未合并旧训练池，未启动训练。", '',
              '|root|累计发言|正文与speaker tokens|新增正文与speaker tokens|', '|---|---:|---:|---:|']
    for row in lengths:
        lines.append(f"|{row['root_id']}|{row['turns']}|{row['dialogue_tokens_speakers_and_body']}|{row['new_dialogue_tokens_speakers_and_body']}|")
    lines += ['', f"累计对话平均 tokens：{result['mean_dialogue_tokens_speakers_and_body']}（本地 Qwen3-4B tokenizer，含speaker，不含turn编号）。", '',
              f"从预注册到交付的墙钟时间：{result['timing']['batch_setup_to_delivery_seconds']:.1f} 秒；Solver 两侧包装器时间：{result['timing']['solver_wrapper_seconds']}。逐场作者与QA收据时间见 FINAL_STATUS.json；并行区间不能相加当批次耗时。", '',
              '多选每题 6 个置换×3 次回答，排序每题 12 个置换×1 次回答；调用不是独立题目。沿用原生解码、计分和 0<mean_partial<1 筛选规则。', '',
              '本轮固定既有人物，B 使用预留的本地 MultiWOZ 训练事件，服务交互由既有朋友转述。本轮验证自动编号后的全流程产出；root 均已有训练曝光，没有新增人物或随机 holdout，不用于宣称泛化收益。', '',
              '原历史逐字保留，作者与审核人员分开但继承同一会话模型；不构成异模型或外部语义认证。精确模型 ID、调用 usage 和费用不可得；文本 tokens 不是实际调用消耗。', '',
              '## 文件', '', '- 每root完整对话：roots/root*/FULL_DIALOGUE.md',
              '- 每root题目答案：roots/root*/QUESTIONS_AND_ANSWERS.md',
              '- 原稿、失败与审核：roots/root*/scenes/',
              '- 内容统计：CONTENT_SUMMARY.json', '- Solver原始回答：evaluation/run/{base,trained}/responses/',
              '- 独立复算：INDEPENDENT_SOLVER_AUDIT.json', '- 来源与流程审计：SOURCE_PROVENANCE_AUDIT.json',
              '- 筛选候选：SELECTED_CANDIDATES.json', '- 冻结工具：evidence/TOOLING_FREEZE.json', '']
    textsave(HERE / 'REPORT_ZH.md', '\n'.join(lines))
    files = [f for f in sorted(HERE.rglob('*')) if f.is_file() and f.name != 'COMPLETE_MANIFEST.json']
    save(HERE / 'COMPLETE_MANIFEST.json', {'created_at_utc': now(), 'status': 'SEALED_SERIALIZED_A3B3_VALIDATION',
         'files': [{'path': str(f.relative_to(HERE)), 'sha256': sha(f), 'bytes': f.stat().st_size} for f in files]})
    print(json.dumps({'status': result['status'], 'selected': len(selected), 'lengths': lengths}, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['content','final']); a = p.parse_args()
    content() if a.phase == 'content' else final()
