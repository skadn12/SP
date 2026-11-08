"""Parent final accounting verifier. Does not generate or change content."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, sys

B = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_dev_test_ab_v124')
sys.dont_write_bytecode = True
sys.path.insert(0, str(B / 'delivery_tools'))
sys.path.insert(0, str(B / 'tools'))
import state_and_export as delivery
from phase_gate_v124 import require_phase

def read(p):
    return json.loads(Path(p).read_text())

def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def verify(x):
    assert binding(x['path']) == x, x['path']

def write_new(p, value):
    with Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def close_dev():
    states = delivery.states('dev')
    assert len(states) == 15 and all(x['terminal_state'] for x in states)
    accounting_path = B / 'evidence/DEV_TERMINAL_ACCOUNTING.json'
    audit_path = B / 'evidence/DEV_PHASE_CLOSE_AUDIT.json'
    accounting = read(accounting_path); report = read(audit_path)
    assert accounting['roots'] == states
    assert report['status'] == 'PASS'
    assert report['terminal_accounting_binding'] == binding(accounting_path)
    for row in states:
        for item in row['terminal_evidence_bindings']:
            verify(item)
    result = {'schema_version':'v124_dev_phase_closed_v1',
        'status':'DEV_COMPLETE_BEFORE_TEST_READS','batch_id':B.name,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'solver_run':False,'score_based_selection':False,
        'master_freeze_binding':binding(B/'MASTER_FREEZE.json'),
        'manifest_binding':binding(B/'MANIFEST.json'),
        'terminal_accounting_binding':binding(accounting_path),
        'independent_audit_binding':binding(audit_path)}
    write_new(B/'evidence/DEV_PHASE_CLOSED.json', result)
    manifest = read(B/'MANIFEST.json')
    test_root = next(x['root_id'] for x in manifest['roots'] if x['split'] == 'test')
    require_phase(B, test_root)
    print(json.dumps({'closed':binding(B/'evidence/DEV_PHASE_CLOSED.json')}, ensure_ascii=False))

def seal():
    report_path = B / 'evidence/PARENT_DELIVERY_AUDIT.json'
    independent_path = B / 'evidence/INDEPENDENT_FINAL_AUDIT.json'
    report = read(report_path); independent = read(independent_path)
    assert report['status'] == independent['status'] == 'PASS'
    assert independent['parent_delivery_audit_binding'] == binding(report_path)
    split_reports = report['splits']
    lines = ['# v124 dev15/test16 扩充与出题交付', '',
        '固定31个root，全部保留在核算分母。逐root双人基线审、事件匹配、A3+B3逐场审、公开选项与题目审。',
        '本批不运行Solver、不按分数选题、不导入训练池。历史test曝光与同模型共享文件系统审核局限见evidence/EXPOSURE_DISCLOSURE.json。', '',
        '| split | root | 六场全审通过 | 已审新场 | 内容通过题 | 终态 |',
        '|---|---:|---:|---:|---:|---|']
    for split in ['dev','test']:
        t = split_reports[split]
        lines.append('| '+split+' | '+str(t['fixed_roots'])+' | '+str(t['roots_six_scenes_ready'])+' | '+str(t['ready_scenes'])+' | '+str(t['content_pass_questions'])+' | '+json.dumps(t['terminal_states'],ensure_ascii=False)+' |')
    lines += ['', '## Gold与题型', '']
    for split in ['dev','test']:
        t = split_reports[split]
        lines.append('- '+split+'：题型 '+json.dumps(t['question_types'],ensure_ascii=False)+'；多选Gold数量 '+json.dumps(t['multi_select_gold_counts'],ensure_ascii=False)+'；Gold6对照 '+str(t['gold6_control_questions'])+'。')
    lines += ['', '## 全部暂停与失败', '', '| split | root | 终态 | 证据 |', '|---|---|---|---|']
    for row in delivery.states():
        if row['terminal_state'] != 'QA_COMPLETE':
            evidence = '<br>'.join(x['path'] for x in row['terminal_evidence_bindings'])
            lines.append('| '+row['split']+' | '+row['root_id']+' | '+row['terminal_state']+' | '+evidence+' |')
    lines += ['', '## 文件入口', '',
        '- delivery/dev 与 delivery/test：逐root可读对话题目、公共题目、答案和完整证据。',
        '- delivery/<split>/TOTALS.json：固定分母、题型、Gold分布、对话长度。',
        '- delivery/<split>/QUESTION_DISPOSITION.json：题目终态；units/root*/：首稿、首判、失败与完整执行证据。',
        '- evidence/*TERMINAL_ACCOUNTING.json：所有root终态，暂停root不冒充生产成功。',
        '- evidence/PARENT_DELIVERY_AUDIT.json 与 evidence/INDEPENDENT_FINAL_AUDIT.json：最终核算审计。', '',
        '内容通过表示按本批审核政策通过；未验证难度或模型训练收益，不宣称严格未触及测试集。']
    with (B/'FINAL_REPORT.md').open('x') as f:
        f.write(chr(10).join(lines))
    paths = sorted(p for p in B.rglob('*') if p.is_file() and p.name != 'COMPLETE_MANIFEST.json')
    result = {'schema_version':'v124_complete_manifest_v1','batch_id':B.name,
        'status':'ALL_31_ROOTS_TERMINAL_ACCOUNTED_AND_DELIVERED',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'fixed_roots':31,'all_roots_success_claimed':False,
        'master_freeze_binding':binding(B/'MASTER_FREEZE.json'),
        'parent_delivery_audit_binding':binding(report_path),
        'independent_final_audit_binding':binding(independent_path),
        'final_report_binding':binding(B/'FINAL_REPORT.md'),
        'splits':split_reports,'solver_run':False,'score_based_selection':False,
        'training_export_or_merge':False,'files':[binding(p) for p in paths]}
    write_new(B/'COMPLETE_MANIFEST.json', result)
    for item in result['files']:
        verify(item)
    print(json.dumps({'complete_binding':binding(B/'COMPLETE_MANIFEST.json'),'file_count':len(paths),'splits':split_reports},ensure_ascii=False))

def audit():
    master = read(B / 'MASTER_FREEZE.json')
    for item in master['files']:
        verify(item)
    manifest = read(B / 'MANIFEST.json')
    states = delivery.states()
    assert len(states) == 31 and all(x['terminal_state'] for x in states)
    assert all(x['status'] == x['terminal_state'] for x in states)
    total_questions = []; reservations = []; generated = 0; ready = 0
    split_reports = {}
    for split, size in [('dev', 15), ('test', 16)]:
        rows = [x for x in states if x['split'] == split]
        assert len(rows) == size
        ledger = read(B / 'evidence' / (split.upper() + '_TERMINAL_ACCOUNTING.json'))
        assert ledger['roots'] == rows
        for row in rows:
            assert row['terminal_evidence_bindings']
            for item in row['terminal_evidence_bindings']:
                verify(item)
        totals = read(B / 'delivery' / split / 'TOTALS.json')
        qs = read(B / 'delivery' / split / 'QUESTIONS_WITH_ANSWERS.json')['questions']
        assert totals['fixed_roots'] == size and totals['content_pass_questions'] == len(qs)
        assert totals['ready_scenes'] == sum(x['ready_scene_count'] for x in rows)
        assert totals['generated_scenes'] == sum(x['generated_scene_count'] for x in rows)
        assert totals['terminal_states'] == dict(Counter(x['terminal_state'] for x in rows))
        assert totals['solver_run'] is False and totals['training_or_pool_merge'] is False
        total_questions.extend(qs); ready += totals['ready_scenes']; generated += totals['generated_scenes']
        split_reports[split] = {k: v for k, v in totals.items() if k != 'dialogue_lengths'}
        for row in rows:
            root = row['root_id']; unit = B / 'units' / ('root' + root.rsplit('-', 1)[1])
            public = read(B / 'delivery' / split / (unit.name + '_PUBLIC.json'))
            evidence = read(B / 'delivery' / split / (unit.name + '_ANSWERS_AND_EVIDENCE.json'))
            assert (B / 'delivery' / split / (unit.name + '_DIALOGUE_AND_QUESTIONS.md')).is_file()
            assert public['root_id'] == evidence['root_id'] == root
            assert public['status'] == row['terminal_state']
            verify(public['source_binding'])
            baseline = read(unit / 'inputs/BASELINE.json')['turns']
            assert public['turns'][:len(baseline)] == baseline
            assert len(evidence['questions']) == row['content_pass_questions']
            assert len(public['questions']) == len(evidence['questions'])
            assert all('gold' not in x and 'option_evidence' not in x and 'item_evidence' not in x for x in public['questions'])
            for q in evidence['questions']:
                assert q['evaluation_only'] is True and q['solver_evaluated'] is False
                assert q['root_id'] == root and q['split'] == split
                assert len(q['gold']) in range(2,7)
            c = unit / 'inputs/CONTINUATION_CONSTRAINTS.json'
            assert c.is_file()
            for review in read(c)['review_bindings']:
                verify(review)
            r = unit / 'SOURCE_RESERVATIONS.json'
            if r.exists():
                reservation_doc = read(r)
                def collect(x):
                    if isinstance(x, dict):
                        if isinstance(x.get('source_id'), str):
                            reservations.append(x['source_id'])
                        for key, value in x.items():
                            if key != 'source_id': collect(value)
                    elif isinstance(x, list):
                        for value in x: collect(value)
                collect(reservation_doc)
            for scene in sorted((unit / 'roots' / unit.name / 'scenes').glob('*')):
                p = scene / 'EXPANSION_DRAFT.json'
                if p.is_file():
                    turns = read(p)['turns']; assert 12 <= len(turns) <= 16
                    assert all(not any(0x3400 <= ord(ch) <= 0x9FFF for ch in t["text"]) for t in turns)
    test_root = next(x['root_id'] for x in manifest['roots'] if x['split'] == 'test')
    require_phase(B, test_root)
    assert len({x['candidate_id'] for x in total_questions}) == len(total_questions)
    assert len(reservations) == len(set(reservations)), 'duplicate reservation'
    consumed = set(read(B / 'evidence/SOURCE_POOL_FREEZE.json')['consumed_source_ids'])
    assert not consumed.intersection(reservations), 'historical source reuse'
    result = {'schema_version':'v124_parent_delivery_audit_v1','status':'PASS',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'fixed_roots':31,
        'master_freeze_binding':binding(B/'MASTER_FREEZE.json'),
        'audit_implementation_binding':binding(__file__),
        'all_frozen_bindings_unchanged':True,'all_roots_terminal':True,
        'baseline_prefix_preserved_all31':True,'dev_closed_before_test':True,
        'globally_unique_new_reserved_sources':len(reservations),
        'historical_source_overlap':0,'generated_scenes':generated,'ready_scenes':ready,
        'content_pass_questions':len(total_questions),'splits':split_reports,
        'solver_run':False,'score_based_selection':False,'training_export_or_merge':False,
        'limits':['Structural and evidence consistency verification does not establish external semantic certification or difficulty.','Same-model shared-filesystem review and historical test exposure remain disclosed.']}
    write_new(B/'evidence/PARENT_DELIVERY_AUDIT.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['audit','close-dev','seal']
    {'audit':audit,'close-dev':close_dev,'seal':seal}[sys.argv[1]]()