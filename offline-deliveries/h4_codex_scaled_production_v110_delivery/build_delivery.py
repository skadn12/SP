"""Assemble reports and outer seals after all v110 reviews; never change data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
BATCH = HERE.parent / 'h4_codex_scaled_production_v110'
EVAL = HERE.parent / 'h4_codex_scaled_production_v110_solver_eval'
VIEW = HERE.parent / 'h4_codex_scaled_production_v110_training_view_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def bind(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')


def verify_manifest(p):
    p = Path(p)
    rows = read(p)['files']
    paths = []
    for r in rows:
        q = Path(r.get('path') or r['relative_path'])
        q = q if q.is_absolute() else p.parent / q
        assert q.is_file() and sha(q) == r['sha256'], str(q)
        paths.append(str(q.resolve()))
    assert len(paths) == len(set(paths)), str(p)
    return len(rows)


def seal(directory, name, status):
    # Enumerate BEFORE opening the new manifest. Nested manifests stay included.
    files = [p for p in sorted(directory.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts]
    target = directory / name
    assert not target.exists()
    write(target, {'status': status, 'sealed_at_utc': now(),
                   'self_hash_excluded': True,
                   'files': [{'path': str(p.relative_to(directory)),
                              'sha256': sha(p), 'bytes': p.stat().st_size}
                             for p in files]})
    verify_manifest(target)
    assert {str(p.relative_to(directory)) for p in directory.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p != target} == {
                r['path'] for r in read(target)['files']}
    return target


def main():
    report = read(EVAL / 'PRODUCTION_REPORT.json')
    a = read(BATCH / 'FINAL_BATCH_ACCOUNTING.json')
    package = read(EVAL / 'training_candidate_package/PACKAGE_REPORT.json')
    audit = read(EVAL / 'AUDIT_RESULTS.json')
    assert audit['status'] == 'VALID' and not audit['data_errors']
    reviews = {name: read(HERE / name) for name in (
        'INDEPENDENT_SOLVER_REVIEW.json', 'INDEPENDENT_PRESENTATION_REVIEW.json')}
    assert all(r['status'].startswith('PASS') for r in reviews.values())
    assert read(BATCH / 'evidence/INDEPENDENT_CONTENT_PROVENANCE_REVIEW.json')['status'].startswith('PASS')
    assert report['training_started'] is False
    manifest_paths = [BATCH/'COMPLETE_MANIFEST.json', EVAL/'EVALUATION_COMPLETE_MANIFEST.json',
        EVAL/'training_candidate_package/COMPLETE_MANIFEST.json', EVAL/'PRODUCTION_COMPLETE_MANIFEST.json',
        VIEW/'package/COMPLETE_MANIFEST.json']
    verified = {str(p): verify_manifest(p) for p in manifest_paths}
    history = read(BATCH/'evidence/HISTORICAL_BEFORE.json')['protected']
    history_count = 0
    for h in history:
        expected = {r['path']: r['sha256'] for r in h['files']}
        actual = {str(p): sha(p) for p in Path(h['directory']).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        assert expected == actual
        history_count += len(expected)

    exposure = read(BATCH/'EXPOSURE_AUDIT.json')
    train_binding = exposure['clean_training_manifest']
    train = Path(train_binding['path'])
    assert sha(train) == train_binding['sha256']
    train_roots = {json.loads(s)['root_id'] for s in train.read_text().splitlines()}
    attempted = set()
    for directory in [*(Path(h['directory']) for h in history), BATCH]:
        cfg = directory/'RUN_CONFIG.json'
        rows = read(cfg)['roots'] if cfg.exists() else read(directory/'FINAL_BATCH_ACCOUNTING.json')['roots']
        attempted.update(r if isinstance(r,str) else r['root_id'] for r in rows)
    attempted &= train_roots
    remaining = sorted(train_roots-attempted, key=lambda r:int(r.rsplit('-',1)[1]))
    inventory = {r['root_id']:r for r in read(PROJECT/'SP-820-publish/first_training_pipeline/data/qa_generation_only/manifest.json')['roots']}
    remaining_bindings = []
    selection_binding = read(BATCH/'evidence/SOURCE_IDENTITY_BINDINGS.json')['selection_manifest']
    assert sha(selection_binding['path']) == selection_binding['sha256']
    selection = read(selection_binding['path'])
    originals = {r['root_id']:r for r in selection['records']}
    holdouts = set()
    for binding in exposure['holdout_manifests']:
        assert sha(binding['path']) == binding['sha256']
        holdouts.update(json.loads(s)['root_id'] for s in Path(binding['path']).read_text().splitlines())
    for rid in holdouts | set(remaining):
        assert sha(originals[rid]['input_path']) == originals[rid]['input_sha256']
    holdout_sources = {r:read(originals[r]['input_path'])['source'] for r in holdouts}
    for rid in remaining:
        r = inventory[rid]
        assert sha(r['source_path']) == r['source_sha256']
        orig = read(originals[rid]['input_path'])['source']
        assert rid not in holdouts
        assert orig['source_id'] not in {s['source_id'] for s in holdout_sources.values()}
        assert orig['source_record_sha256'] not in {s['source_record_sha256'] for s in holdout_sources.values()}
        remaining_bindings.append({'root_id':rid, **bind(r['source_path']),
            'original_source_id':orig['source_id'], 'original_source_record_sha256':orig['source_record_sha256']})
    plan = read(BATCH/'PRECALL_PLAN.json')
    assert sha(plan['source_pool_path']) == plan['source_pool_sha256']
    pool = read(plan['source_pool_path'])['rows']
    consumed = set(plan['consumed_source_ids']) | {
        r['source_id'] for r in read(BATCH/'SOURCE_RESERVATIONS.json')['allocation']}
    available_events = sum(f"soda:train:{r['row']['original_index']}" not in consumed for r in pool)
    write(HERE/'REMAINING_INVENTORY.json', {'status':'VERIFIED_NOT_STARTED',
        'clean_training_roots':len(train_roots),'cumulative_attempted_roots':len(attempted),
        'remaining_roots':remaining_bindings, 'remaining_events_at_batch_reservation':available_events,
        'event_compatibility_or_success_guaranteed':False, 'holdout_identity_overlap':[],
        'source_reservation_or_generation_started':False})

    ready = report['source_ready_roots']
    passed = a['new_questions']+a['gold6_controls']
    fixed_slots = {'all':84, 'content_pass_including_controls':passed,
        'source_hold_slots':7*(12-ready), 'ready_missing_or_excluded_slots':7*ready-passed}
    assert sum(v for k,v in fixed_slots.items() if k!='all') == 84
    d = package['multi_gold_counts']
    candidate = VIEW/'package/frontier_candidates.jsonl'
    view_readme = f'''# v110 规范展示训练候选

推荐入口：`package/frontier_candidates.jsonl`。{package['eligible_candidates']}条候选：{package['type_counts']}；多选Gold分布{d}。

namespace `v110_training_presentation_v1` 已在正文生成前固定。与原候选包一题一条、顺序与入选集合一致；只重排选项/条目展示及映射答案标签。正确内容、来源、题干、系统提示、输出要求和历史Solver统计不变。一次固定seed，不搜索。逐题映射、内容不变量审核和绑定保留。外部目录保存同模型不同角色的独立只读复核；未混合replay、未执行新训练。
'''
    write(VIEW/'README.md',view_readme)
    view_manifest = seal(VIEW,'COMPLETE_MANIFEST.json','PRESENTATION_EXPORT_AND_REVIEW_BOUND_COMPLETE')
    verified[str(view_manifest)] = verify_manifest(view_manifest)
    eval_manifest = seal(EVAL,'DELIVERY_COMPLETE_MANIFEST.json','SOLVER_AND_CANDIDATE_DELIVERY_COMPLETE_NO_TRAINING')
    verified[str(eval_manifest)] = verify_manifest(eval_manifest)
    started = datetime.fromisoformat(read(BATCH/'BATCH_STARTED.json')['started_at_utc'])
    finished = datetime.now(timezone.utc)
    status = {'status':'COMPLETE_NEXT_TWELVE_ROOT_BATCH_NO_TRAINING',
        'finished_at_utc':finished.isoformat(), 'wall_seconds_total':(finished-started).total_seconds(),
        'attempted_roots':12, 'structural_pass_roots':12, 'source_ready_roots':ready,
        'source_hold_roots':12-ready,'new_turns_all':report['new_turns_all_attempts'],
        'new_turns_ready':report['new_turns_ready_sources'], 'new_questions':a['new_questions'],
        'new_multi_select':a['new_multiselect'],'new_ordering':a['new_ordering'],
        'content_gold2_3_4_5':[a['gold2_to5_distribution'].get(str(g),0) for g in range(2,6)],
        'gold6_controls':a['gold6_controls'], 'fixed_slots':fixed_slots,
        'solver_calls':report['paired_solver_calls_total'],'solver_audit':'VALID','solver_errors':0,
        'new_training_candidates':package['eligible_candidates'],
        'candidate_multi_select':package['type_counts'].get('multi_select',0),
        'candidate_ordering':package['type_counts'].get('ordering',0),
        'candidate_gold2_3_4_5':[d.get(str(g),0) for g in range(2,6)],
        'candidate_entry':bind(candidate), 'independent_presentation_review':reviews['INDEPENDENT_PRESENTATION_REVIEW.json']['status'],
        'total_clean_training_roots':len(train_roots),'cumulative_codex_attempted_roots':len(attempted),
        'remaining_roots':len(remaining),'available_soda_events':available_events,
        'remaining_roots_started':False,'new_training_started':False,
        'historical_files_unchanged':history_count,'manifest_verification_counts':verified,
        'deliverable_manifest_bindings':[bind(p) for p in [*manifest_paths, view_manifest, eval_manifest]],
        'limitations':['Same inherited model and shared project context across author/reviewer roles.',
            'Training-side first-round exposure; not untouched evaluation or causal training-benefit evidence.',
            'Generator model identifier, sampling, usage and billing unavailable.']}
    write(HERE/'FINAL_STATUS.json',status)
    lines = ['# v110：下一批12个训练侧root交付','',
        '已完成固定选源、一次扩充、来源全文审核、多选与排序联合出题、两层公开来源判断、原设计Gold交叉核对、去重封存、两模型配对Solver评估及固定seed展示导出。未启动新训练。','',
        f"12/12结构通过，共新增{report['new_turns_all_attempts']}轮。来源READY {ready}/12、HOLD {12-ready}/12。所有首稿、失败和审核记录保留，不改旧史、不替换源、不改Gold救题。",'',
        '| 项目 | 全部尝试分母 | READY条件分母 |','|---|---:|---:|',
        f"| Gold2–5多选内容通过 |{a['new_multiselect']}/48|{a['new_multiselect']}/{4*ready}|",
        f"| 排序内容通过 |{a['new_ordering']}/24|{a['new_ordering']}/{2*ready}|",
        f"| 新增主任务题合计 |{a['new_questions']}/72|{a['new_questions']}/{6*ready}|",
        f"| Gold6对照 |{a['gold6_controls']}/12|{a['gold6_controls']}/{ready}|",'',
        f"完整84槽＝{passed}道内容通过＋{fixed_slots['source_hold_slots']}个来源HOLD槽＋{fixed_slots['ready_missing_or_excluded_slots']}个READY后缺额/排除槽。原accounting的unfilled_slots仅统计组合阶段缺额，不代表全部失败。多选Gold2/3/4/5内容分布为{status['content_gold2_3_4_5']}。",'',
        '| Root | 来源 | 多选 | 排序 | 对照 |','|---|---|---:|---:|---:|']
    for r in a['roots']:
        lines.append(f"|{r['root_id'].rsplit('-',1)[1]}|{r['source_status']}|{r['new_multiselect']}|{r['ordering']}|{r['controls']}|")
    lines += ['','## 失败与质量边界','']
    for failure in report['source_failures']:
        reasons = [f['reason'] for f in failure['findings'] if f.get('severity')=='BLOCKING']
        lines.append('- '+failure['root_id']+'：'+'；'.join(reasons))
    for r in a['roots']:
        if r['source_status']=='READY':
            f=BATCH/'roots'/('root'+r['root_id'].rsplit('-',1)[1])
            for q in read(f/'COMPOSITION_RECEIPT.json')['unfilled_slots']:
                lines.append('- '+r['root_id']+'：'+json.dumps(q,ensure_ascii=False))
            for q in read(f/'FINAL_CROSSCHECK.json')['results']:
                if q['status']!='CONTENT_PASS':
                    lines.append('- '+q['candidate_id']+'：'+'；'.join(q['issues']))
    lines += ['','READY仍可有历史重复、称呼、转场或自然度WARNING；多选同family子题和重叠排序槽相关。题目主要围绕新增scene，不能据此声称验证了长历史记忆。语义审核由同一继承模型的不同角色完成，共享背景；机械审核不能认证语义零错误。','',
        '## 配对Solver与候选','',
        f"M0与首轮已有M_clean_policy共{report['paired_solver_calls_total']}次调用，全量独立解析重评分/配对检查通过，0错误；多选每题18调用、排序12调用，无自动重试。",'',
        '| 分层 | M0完全正确 | 首轮M_clean_policy完全正确 |','|---|---:|---:|']
    for key in ('gold2','gold3','gold4','gold5','gold2_5','ordering','gold6'):
        b,t=(report['solver_groups'][key][s] for s in ('base','trained'))
        def cell(v):
            return f"{v['exact_correct']}/{v['valid']} ({100*v['exact_rate']:.2f}%)" if v['valid'] else 'NA'
        lines.append(f'|{key}|{cell(b)}|{cell(t)}|')
    lines += ['',
        '这是已暴露训练侧root的新题诊断；没有新训练，不能证明新增数据的训练收益。不同批次root不同，跨批正确率不直接代表难度升降。每次展示相关，不能当独立样本。','',
        f"训练候选{package['eligible_candidates']}条，类型{package['type_counts']}，多选Gold2/3/4/5={status['candidate_gold2_3_4_5']}。Gold6全部排除。推荐入口：`../{VIEW.name}/package/frontier_candidates.jsonl`；一次预先固定namespace=v110_training_presentation_v1重排展示，同步Gold；候选、正确内容、来源和历史评分不变。独立只读复核见本目录两个INDEPENDENT记录。原replay仅绑定，未混合；六选项/六项排序与旧replay格式不同，后续训练需冻结配比和family权重。",'',
        '## 耗时与剩余进度','',
        f"作者记录并行共有墙钟窗口{report['expansion_authoring_parallel_window_seconds']:.1f}秒，含读源、首稿、机械检查和receipt收尾，不是孤立模型生成延迟；共享时段不能相加。完整流程含最终复核封存约{status['wall_seconds_total']/60:.1f}分钟。",
        f"clean训练侧共{len(train_roots)}个root，累计尝试{len(attempted)}个，剩{len(remaining)}个；剩余baseline哈希和与holdout原来源身份隔离已核验，详见REMAINING_INVENTORY.json。冻结SODA池余{available_events}个未消耗素材，数量足够剩余root但不保证兼容或通过率。剩余root尚未开始匹配或生成。",
        f"历史{history_count}个受保护文件及文件集合未变；生成精确模型ID、采样、usage、费用不可得，均不猜测。匹配训练历史反馈root为{report['matching_historical_feedback_roots']}，未用当前Solver分数救题。",
        '执行仍需要当前会话的语义读者，尚非无人值守的一条命令生产后端。','']
    write(HERE/'DELIVERY_REPORT_ZH.md','\n'.join(lines))
    write(HERE/'REPRODUCE.md', f'''# v110 展示复现

从项目根目录运行，输出须为不存在的新目录：

```bash
.venv-local4b/bin/python -B {VIEW.relative_to(PROJECT)}/export_presentation.py --source-package {EVAL.relative_to(PROJECT)}/training_candidate_package --pool {EVAL.relative_to(PROJECT)}/EVALUATION_POOL.json --output /tmp/v110_reproduced_presentation_NEW --seed v110_training_presentation_v1
```

实际本批只导出一次。复现应得到相同训练JSONL和逐题映射；时间、输出路径等元数据可不同。勿修改已封存包。
''')
    outer = seal(HERE,'DELIVERY_COMPLETE_MANIFEST.json','CONTROLLED_BATCH_ALL_DELIVERABLES_COMPLETE_NO_TRAINING')
    print(json.dumps({'status':status['status'], 'candidates':package['eligible_candidates'],
        'remaining_roots':len(remaining),'outer_manifest':bind(outer)},ensure_ascii=False))


if __name__ == '__main__':
    main()
