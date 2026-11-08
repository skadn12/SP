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
from codex_sequential_ab import inspect_root, load_config

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, x):
    with Path(p).open('x') as f: json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def textsave(p, s):
    with Path(p).open('x') as f: f.write(s)

def verify_tooling():
    freeze = read(HERE / 'evidence/TOOLING_CORRECTION_AND_FREEZE.json')
    assert freeze['status'] == 'FROZEN_BEFORE_SOLVER_LIVE'
    expected = {str((HERE / name).resolve()) for name in
                ('evaluate.py', 'evaluate_v2.py', 'audit_solver.py', 'deliver.py')}
    assert expected.issubset({row['path'] for row in freeze['files']})
    for row in freeze['files']:
        assert sha(row['path']) == row['sha256'], 'TOOLING_HASH_CHANGED:' + row['path']
    frozen_at = datetime.fromisoformat(freeze['created_at_utc'].replace('Z', '+00:00'))
    for side in ('base', 'trained'):
        started = HERE / 'evaluation' / (side.upper() + '_STARTED.json')
        if started.exists():
            assert frozen_at <= datetime.fromisoformat(read(started)['started_at_utc'].replace('Z', '+00:00'))
    return freeze

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

def content():
    verify_tooling()
    cfg = load_config(HERE); roots = []; public = []; gold = []; excluded = []; outcomes = []
    for r in cfg['roots']:
        rid = r['root_id']; folder = HERE / 'roots' / ('root' + rid.rsplit('-', 1)[1])
        state = inspect_root(HERE, cfg, rid)
        drafts = [read(f) for f in (folder / 'scenes').glob('*/EXPANSION_DRAFT.json')]
        state['attempted_new_turns'] = sum(len(d['turns']) for d in drafts)
        state['accepted_new_turns'] = sum(s.get('new_turn_count', 0) for s in state['scenes'] if s['status'] == 'READY')
        if state['full_expansion_complete']:
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
                if c['status'] == 'CONTENT_PASS': public.append(q); gold.append(g)
                else: excluded.append({'question': q, 'private_gold': g, 'crosscheck': c})
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
    save(HERE / 'CONTENT_SUMMARY.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('roots','all_public_outcomes')}, ensure_ascii=False))

def final():
    verify_tooling()
    require_independent_audit()
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
        for name, predicate in [('multi_select_gold2_5', lambda q: q['question_type'] == 'multi_select' and len(q['gold']) < 6),
                                ('gold6_control', lambda q: q['question_type'] == 'multi_select' and len(q['gold']) == 6),
                                ('ordering', lambda q: q['question_type'] == 'ordering')]:
            sub = [r for r in records if predicate(qmap[r['candidate_id']])]
            bytype[name] = {'correct': sum(r['score']['exact_match'] for r in sub), 'calls': len(sub),
                            'accuracy': sum(r['score']['exact_match'] for r in sub) / len(sub) if sub else None,
                            'question_count': len({r['candidate_id'] for r in sub})}
        metrics[side] = {'model': result['solver_model'], 'valid_calls': len(records), 'by_type': bytype,
                         'questions': result['questions'], 'call_set_errors': []}
        if side == 'trained':
            selected = [qmap[q['candidate_id']] for q in result['questions'] if q['selected_by_policy'] is True
                        and qmap[q['candidate_id']]['role'] != 'control6']
    save(HERE / 'SELECTED_CANDIDATES.json', {'questions': selected, 'training_started': False,
         'selection_rule': 'M_clean_policy complete-valid 0<mean_partial<1; Gold6 controls excluded'})
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    lengths = []
    for r in summary['roots']:
        if not r['full_expansion_complete']: continue
        f = HERE / 'roots' / ('root' + r['root_id'].rsplit('-',1)[1]) / 'CUMULATIVE_SOURCE.json'
        turns = read(f)['turns']
        formatted = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        plain = '\n'.join(f"{t['speaker']}: {t['text']}" for t in turns)
        lengths.append({'root_id': r['root_id'], 'turns': len(turns),
                        'dialogue_tokens_with_refs_and_speakers': len(tok.encode(formatted, add_special_tokens=False)),
                        'dialogue_tokens_speakers_and_body': len(tok.encode(plain, add_special_tokens=False))})
    protected = read(HERE / 'evidence/HISTORICAL_PROTECTION.json')['files']
    assert all(sha(r['path']) == r['sha256'] for r in protected)
    frozen = read(HERE / 'EXECUTION_INPUTS_FROZEN.json')['files']
    assert all(sha(r['path']) == r['sha256'] for r in frozen)
    result = {'completed_at_utc': now(), 'status': 'COMPLETE_A3B3_PILOT_WITH_RECORDED_FAILURE_NO_TRAINING',
              'content': summary, 'solver': metrics, 'selected_new_question_count': len(selected),
              'selected_type_counts': dict(Counter(q['question_type'] for q in selected)),
              'dialogue_lengths': lengths, 'historical_files_verified_unchanged': len(protected),
              'training_started': False, 'external_semantic_certification': False}
    save(HERE / 'FINAL_STATUS.json', result)
    lines = ['# A3＋B3 恢复验证结果 v112', '',
             f"尝试 {summary['attempted_roots']} 个已有暴露的训练 root，完整完成 A3＋B3 的为 {summary['complete_A3B3_roots']} 个。",
             f"共 {summary['ready_scene_count']}/{summary['expected_scene_count']} 个预定场景 READY；新增有效发言 {summary['accepted_new_turns']} 条。", '',
             '222937 按顺序逐场生成并审核 A1→A2→A3→B1→B2→B3。225575 在 A1 因 session_ref 多了 root 前缀发生结构失败，原稿保留，其后五场未执行。三个已预留 B 素材仍永久占用。失败计入分母。', '',
             f"内容通过 {summary['new_questions']} 道新增题及 {summary['gold6_controls']} 道 Gold6 对照；Gold 数量分布 {summary['multi_gold_counts']}。", '',
             '|题型|M0 正确／调用|M_clean_policy 正确／调用|', '|---|---:|---:|']
    for key, label in [('multi_select_gold2_5','多选 Gold2–5'), ('ordering','排序'), ('gold6_control','Gold6 对照')]:
        cells = [metrics[s]['by_type'][key] for s in ('base','trained')]
        lines.append(f"|{label}|{cells[0]['correct']}/{cells[0]['calls']}|{cells[1]['correct']}/{cells[1]['calls']}|")
    lines += ['', f"Solver 筛选保留 {len(selected)} 道新增候选（{dict(Counter(q['question_type'] for q in selected))}），未合并旧训练池、未启动新训练。",
              '', '每个多选题有 18 次回答，每个排序题有 12 次回答；调用不是独立题目。只有一个完整 root，不能证明批量稳定或训练收益。', '',
              '本轮 B 采用未消费的本地 MultiWOZ 训练事件；补充 SODA 下载的两次 503 均保留。服务角色通过既有朋友报告转述，审核记录保留改编限制。与此前 SODA 单场景批次不能当作单一因素对照。', '',
              '原历史逐字保留，作者与审核分开；均继承同一会话模型，不是异模型／外部认证。模型精确 ID、调用 usage 和账单费用不可得。文本 tokens 另列，不混充调用消耗。', '',
              '## 文件', '', '- 完整对话：roots/root222937/FULL_DIALOGUE.md',
              '- 题目和答案：roots/root222937/QUESTIONS_AND_ANSWERS.md',
              '- 逐场原稿和审核：roots/root222937/scenes/', '- 失败原稿：roots/root225575/scenes/A_S01/',
              '- 内容统计：CONTENT_SUMMARY.json', '- 求解器原始回答：evaluation/run/{base,trained}/responses/',
              '- 筛选候选：SELECTED_CANDIDATES.json', '- 配置与规则：RUN_CONFIG.json、evidence/CONTENT_CONTRACT.md', '']
    textsave(HERE / 'REPORT_ZH.md', '\n'.join(lines))
    files = [f for f in sorted(HERE.rglob('*')) if f.is_file() and f.name != 'COMPLETE_MANIFEST.json']
    save(HERE / 'COMPLETE_MANIFEST.json', {'created_at_utc': now(), 'status': 'SEALED_A3B3_PILOT',
         'files': [{'path': str(f.relative_to(HERE)), 'sha256': sha(f), 'bytes': f.stat().st_size} for f in files]})
    print(json.dumps({'status': result['status'], 'selected': len(selected), 'lengths': lengths}, ensure_ascii=False))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['content','final']); a = p.parse_args()
    content() if a.phase == 'content' else final()
