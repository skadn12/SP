"""Seal the versioned delivery with explicit current-vs-historical provenance.

Run after merge.py finalize and audit_v17.py. Never writes historical artifacts.
"""
import argparse
from collections import Counter
from pathlib import Path
import json

import merge


def seal(run):
    read, write, sha, require = merge.read, merge.write, merge.sha, merge.require
    target = run / 'training_pool'
    audit_path = run / 'FINAL_AUDIT_V2.json'
    audit = read(audit_path)
    require(audit.get('status') == 'PASS', 'independent_audit_not_passed')
    report = read(target / 'REPORT.json')
    rows = [json.loads(line) for line in (target / 'TRAINING_POOL.jsonl').read_text().splitlines()]
    results = read(run / 'M/solver/RESULTS.json')
    byid = {q['candidate_id']: q for q in results['questions']}
    provenance = []
    for row in rows:
        q = row['original_selected_question']
        require(q['training_selection'] == byid[row['candidate_id']], 'current_selection_binding_wrong')
        require(read(row['source_path']) == row['full_context'], 'full_context_binding_wrong')
        provenance.append({'candidate_id':row['candidate_id'], 'root_id':row['root_id'],
            'current_source':{'path':row['source_path'],'sha256':row['source_sha256']},
            'current_results':{'path':str(run/'M/solver/RESULTS.json'),'sha256':sha(run/'M/solver/RESULTS.json')},
            'current_selection':q['training_selection'],
            'row_provenance_chain_semantics':'historical generation and previous selection only; current selection is bound here',
            'row_historical_pool_row_semantics':'archived evidence, never current training source'})
    write(target/'CURRENT_PROVENANCE.json', {'rows':provenance,'training_started':False})
    # Keep old development-stage hashes honest; the completed merge stage binds
    # the effective implementation and source set separately.
    initial = read(run/'MANIFEST.json')
    initial_changes = [{'path':path,'prepared_sha256':expected,'effective_sha256':sha(path)}
                       for path,expected in initial['code_sha256'].items() if sha(path)!=expected]
    files = [target/'TRAINING_POOL.jsonl',target/'SELECTED_QUESTIONS.json',target/'REPORT.json',
        target/'CURRENT_PROVENANCE.json',audit_path,run/'MERGE_MANIFEST.json',run/'MIGRATION.json',
        run/'MANIFEST.json',run/'ORIGINAL_TRAINING_POOL.jsonl',
        run/'M/generation/SOLVER_POOL.json',run/'M/solver/manifest.json',run/'M/solver/RESULTS.json',
        run/'M/solver/COMPLETE.json',run/'M/solver/solver_identity.json',Path(__file__),
        merge.HERE/'merge.py',merge.HERE/'generate.py',merge.HERE/'audit_v17.py']
    files += sorted((run/'sources').glob('*.json'))
    files += sorted((run/'review/requests').glob('*/*.json'))
    files += sorted((run/'M/solver/responses').glob('*.json'))
    prepared = merge.HERE/'runs/merged_generation_prepared_01'
    files += [prepared/'MANIFEST.json',prepared/'SOURCE_FLOW_CHECK.json',merge.HERE/'audit_source_flow.py',merge.HERE/'README.md',merge.HERE/'POLICY.md']
    write(target/'MANIFEST.json', {'version':merge.VERSION,'status':'AUDITED_DELIVERY',
        'files':[{'path':str(path),'sha256':sha(path)} for path in files],
        'preparation_code_changes':initial_changes,
        'preparation_code_note':'Initial preparation predates local implementation fixes. Effective merge stage and final code are separately hashed; historical preparation is not rewritten.',
        'training_started':False,'new_question_generation_calls':0,'new_expansion_calls':0,
        'live_merged_question_generation_validated':False,
        'actual_both_routes_retained_in_one_root':any(len({k.split(':')[1] for k in r['accepted_scene_order']})==2
            for r in read(run/'MERGE_MANIFEST.json')['source_rows'])})
    lines=['# 单一 root 上下文迁移结果','',
        '流程：A、B 分别基于同一冻结上下文生成 → 检查重复与冲突 → 通过的场景合入同一 root → 完整上下文出题及评估。','',
        '本次修正来源组织并重评旧题，未生成新题、扩充新对话或开始训练。', '',
        f"旧池 {report['old_pool_count']} 道；迁移评估 {report['evaluated']} 道；本轮入选 {report['selected']} 道。",'',
        '| Root | 评估 | 多选入选 | 排序入选 | 总入选 |','|---|---:|---:|---:|---:|']
    for root,row in report['by_root'].items():
        lines.append(f"| {root.split('-')[-1]} | {row['evaluated']} | {row.get('multi_select',0)} | {row.get('ordering',0)} | {row.get('selected',0)} |")
    lines += ['',f"共 {report['full_solver_calls']} 次 M_clean_policy full 回答，ground=0；独立审计 PASS。",'',
        '138343 的 B 场景与火车出行的既有安排冲突，由 Qwen/Qwen3.6-35B-A3B 联合复核后暂存，保留 A。',
        '6 道来源暂存题：4 道依赖早期替代版本场景，2 道依赖本批冲突 B 场景；原池与原判定全部保留。',
        '保留题干、选项与完整 Gold。只使用本轮完整评估结果筛选，旧成绩只作历史记录。',
        '排列政策未变，解码 seed 随新池调用索引改变，因此结果差异不能归因于合并导致性能升降。',
        '本批通过的来源中没有同时保留 A+B 的 root；双分支拼接已做合成数据回归测试。',
        '正式合并出题入口已通过真实来源预检和离线请求检查，尚未运行新一批实际出题。']
    path=target/'REPORT.md'
    with path.open('x') as stream:stream.write('\n'.join(lines)+'\n')
    return {'status':'AUDITED_DELIVERY','selected':len(rows),'report':str(path)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(seal(args.run.resolve()),ensure_ascii=False,indent=2))
