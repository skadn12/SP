"""Build closeout reports AFTER model exit, scoring, combined analysis and final audit.

Only the five named closeout outputs are written. No generation, retries, training,
question editing or frozen-artifact changes occur. Use --manifest-only after a late
read-only report audit; the parent owns independent final-manifest verification.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
SLOTS = ('MC2', 'MC3', 'MC4', 'MC5', 'MC6', 'ORDER1', 'ORDER2', 'ORDER3')
OUTPUTS = ('CLOSEOUT_VALIDATION.json', 'FINAL_REPORT.json', 'FINAL_REPORT.md',
           'STATUS.json', 'FINAL_MANIFEST.json')
REQUIRED = ('CONTINUATION_MANIFEST.json', 'SOURCE_FREEZE.json', 'QUESTION_FREEZE.json',
            'REUSED_SLOTS.json', 'GENERATION_LEDGER.json', 'F_TARGET_REALIZATION_PRE_SOLVER.json',
            'COMBINED_ANALYSIS.json', 'NORMALIZED_SCORES.json',
            'execution/PREPARATION.json', 'execution/PUBLIC_REQUESTS.json',
            'execution/PRIVATE_SCORING.json', 'execution/RAW_RESPONSE_FREEZE.json',
            'execution/SCORES.json', 'execution/COMBINED_ANALYSIS_CPU_VERIFICATION.json',
            'audits/INDEPENDENT_PROCESS_AUDIT.json', 'audits/SCHEDULING_REUSE_AUDIT.json',
            'audits/ANALYSIS_METHOD_REVIEW.json',
            'audits/ANALYSIS_METHOD_REVIEW_COMPATIBILITY_ADDENDUM.json',
            'audits/FINAL_RESULT_AUDIT.json')
METHOD_SNAPSHOTS = ('audits/ANALYSIS_METHOD_REVIEW.json',
                    'audits/ANALYSIS_METHOD_REVIEW_COMPATIBILITY_ADDENDUM.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    # Atomic replacement is confined to explicitly allowed closeout files.
    if path.parent != BASE or path.name not in OUTPUTS:
        raise ValueError('Unapproved output: ' + str(path))
    temp = path.with_name('.' + path.name + '.closeout-tmp')
    temp.write_text(text, encoding='utf-8')
    os.replace(temp, path)


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('File changed while binding: ' + str(path))
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size}


def bindings_in(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and isinstance(obj.get('sha256'), str):
            yield obj
        for value in obj.values():
            yield from bindings_in(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from bindings_in(value)


def active_model_processes():
    """Read /proc only; never start a model or query a model endpoint."""
    found = []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            joined = ' '.join(args)
            own_run = any(Path(a).name == 'solver_continue.py' for a in args if a) and 'run' in args
            # vLLM workers commonly replace their argv with the engine process title.
            worker = any(a.startswith(('VLLM::EngineCore', 'VLLM::Worker')) for a in args)
            if own_run or worker:
                found.append({'pid': int(folder.name), 'command': joined[:600]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def fraction(n, d):
    return n / d if d else None


def short_id(root):
    return root.removeprefix('cc-train-episode-')


def pct(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def pp(value):
    return '—' if value is None else f'{value * 100:+.2f} pp'


def confidence(result):
    interval = (result.get('bootstrap') or {}).get('percentile95')
    return '—' if interval is None else '[' + ', '.join(pp(v) for v in interval) + ']'


def bootstrap(values):
    if not values:
        return None
    rng = random.Random(20261025)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    def quantile(p):
        i = (len(draws) - 1) * p
        low = int(i)
        return draws[low] + (draws[min(low + 1, len(draws) - 1)] - draws[low]) * (i - low)
    return {'resamples': 10000, 'seed': 20261025, 'unit': 'root',
            'percentile95': [quantile(.025), quantile(.975)]}


def complete_primary(result):
    roots = [r for r in result['roots'] if r['common_slot_count'] == 7]
    return {'definition': 'Both arms and both returned displays for all seven non-MC6 slots',
            'eligible_roots': len(roots), 'root_ids': [r['root_id'] for r in roots],
            'F': statistics.mean(r['F'] for r in roots) if roots else None,
            'G': statistics.mean(r['G'] for r in roots) if roots else None,
            'difference': statistics.mean(r['difference'] for r in roots) if roots else None,
            'bootstrap': bootstrap([r['difference'] for r in roots])}


def old_artifacts(manifest):
    specifications = {
        'original4': ('original4', ('FINAL_STATUS.json', 'FINAL_MANIFEST.json', 'REPORT_ZH.md',
                      'RESULT_INTEGRITY_AUDIT.json', 'FINAL_REPORT_REVIEW.json')),
        'initial16': ('initial16', ('FINAL_REPORT.json', 'FINAL_ARTIFACT_MANIFEST.json',
                      'INDEPENDENT_CLOSEOUT_AUDIT.json', 'CONTENT_GATE_FINAL.json', 'SOURCE_FREEZE.json')),
        'prior_recovery16': ('previous16', ('FINAL_REPORT.json', 'ARTIFACT_MANIFEST.json',
                      'INTEGRITY_AUDIT.json', 'QUESTION_FREEZE.json', 'SOURCE_FREEZE.json'))}
    result = {}
    for cohort, (key, closing_names) in specifications.items():
        folder = Path(manifest[key])
        required = ('execution/RAW_RESPONSE_FREEZE.json', 'execution/SCORES.json', 'execution/PRIVATE_SCORING.json')
        paths = [folder / name for name in required]
        paths += [folder / name for name in closing_names if (folder / name).is_file()]
        # Bind actual raw bytes in addition to the frozen index and score shard.
        raw = read(folder / 'execution/RAW_RESPONSE_FREEZE.json')
        scores = read(folder / 'execution/SCORES.json')
        for obj in (raw, scores):
            for item in bindings_in(obj):
                paths.append(Path(item['path']))
        unique = {str(path.resolve()): path for path in paths}
        result[cohort] = [binding(path) for _, path in sorted(unique.items())]
    return result


def validate(docs):
    issues, checked = [], {}
    def need(ok, message):
        if not ok:
            issues.append(message)
    def check_binding(item):
        key = (item['path'], item['sha256'], item.get('bytes'))
        if key in checked:
            return
        try:
            current = binding(item['path'])
            ok = current['sha256'] == item['sha256'] and ('bytes' not in item or current['bytes'] == item['bytes'])
            checked[key] = ok
            need(ok, 'Binding drift: ' + item['path'])
        except (OSError, RuntimeError) as exc:
            checked[key] = False
            issues.append(str(exc))
    # Base methods review is an explicitly superseded historical snapshot. The
    # addendum retains one labelled old live-path binding and binds the archived
    # old script as well as the final script. No final CPU/result binding is exempt.
    for name in REQUIRED:
        if name in METHOD_SNAPSHOTS:
            continue
        for item in bindings_in(docs[name]):
            check_binding(item)
    methods = docs[METHOD_SNAPSHOTS[0]]
    addendum = docs[METHOD_SNAPSHOTS[1]]
    need(methods.get('status') == 'PASS_METHODS_SNAPSHOT', 'Methods snapshot not passed')
    need(addendum.get('status') == 'PASS_COMPATIBILITY_PATCH'
         and addendum.get('unresolved_patch_review_blockers') == [], 'Compatibility addendum not passed')
    check_binding(addendum['base_review_binding'])
    patch_bindings = addendum['bindings']
    old = patch_bindings['superseded_solver_in_base_review']
    archived = patch_bindings['archived_solver_before_compatibility_fix']
    need(old['sha256'] == archived['sha256'] and old.get('bytes') == archived.get('bytes'),
         'Superseded solver is not represented by matching archived bytes')
    for key, item in patch_bindings.items():
        if key != 'superseded_solver_in_base_review':
            for row in bindings_in(item):
                check_binding(row)
    manifest, sf, qf = (docs[n] for n in ('CONTINUATION_MANIFEST.json', 'SOURCE_FREEZE.json', 'QUESTION_FREEZE.json'))
    analysis = docs['COMBINED_ANALYSIS.json']
    roots = manifest['roots']
    need(len(roots) == len(set(roots)) == 16, 'Fixed new16 root universe differs')
    slots = qf['slots']
    expected = {(r, a, s) for r in roots for a in ('F', 'G') for s in SLOTS}
    keys = [(s['root_id'], s['arm'], s['slot_id']) for s in slots]
    need(len(keys) == len(set(keys)) and set(keys) == expected, 'Frozen 256-slot universe differs')
    need(all(s['status'] in ('PASS', 'FAIL', 'SOURCE_BLOCKED') for s in slots), 'Pending or unknown frozen slot status')
    need(qf.get('adaptive_recovery') is True and analysis.get('adaptive_recovery') is True, 'Adaptive recovery disclosure absent')
    statuses = collections.Counter(s['status'] for s in slots)
    need(statuses == {'PASS': 222, 'FAIL': 18, 'SOURCE_BLOCKED': 16}, 'Final frozen coverage differs from audited 222/256')
    failed = [s for s in slots if s['status'] == 'FAIL']
    need(all(s['slot_id'] == 'ORDER2' for s in failed), 'Non-ORDER2 final failure needs revised report wording')
    blocked_roots = sorted({s['root_id'] for s in slots if s['status'] == 'SOURCE_BLOCKED'})
    need(blocked_roots == ['cc-train-episode-14304'], 'Source-blocked root differs')
    source_outcomes = {s['root_id']: s['status'] for s in sf['outcomes']}
    need(set(source_outcomes) == set(roots), 'Source root denominator differs')
    need(source_outcomes == {root: 'FAIL' if root == 'cc-train-episode-14304' else 'PASS'
                             for root in roots}, 'Final source statuses differ from audited 15/16')
    passes = [s for s in slots if s['status'] == 'PASS']
    admitted = {s['candidate_id'] for s in passes}
    reused = {s['candidate_id'] for s in docs['REUSED_SLOTS.json']['slots']}
    need(len(admitted) == len(passes) and reused <= admitted, 'Candidate reuse identities differ')
    prep = docs['execution/PREPARATION.json']
    public = docs['execution/PUBLIC_REQUESTS.json']['requests']
    scores = docs['execution/SCORES.json']['records']
    freeze = docs['execution/RAW_RESPONSE_FREEZE.json']
    public_ids = [r['request_id'] for r in public]
    score_ids = [r['request_id'] for r in scores]
    need(len(public_ids) == len(set(public_ids)) and set(public_ids) == set(score_ids)
         and len(score_ids) == len(set(score_ids)), 'Current prepared/scored request identity mismatch')
    need(prep['requests'] == len(public) == 2 * len(admitted - reused), 'New prepared count differs from new admitted questions')
    need(set(r['candidate_id'] for r in public) == admitted - reused, 'Reused questions rerun or new questions omitted')
    need(freeze.get('failure') is None and freeze.get('returned') == len(public), 'Current run not fully returned or backend failure present')
    need(freeze.get('requests') == len(public), 'Current raw-freeze prepared count differs')
    need(set(freeze.get('request_ids', [])) == set(public_ids)
         and set(freeze.get('submitted_ids', [])) == set(public_ids), 'Current frozen/submitted request identities differ')
    new_returned = 0
    for row in scores:
        raw_binding = row.get('raw_binding', row.get('raw'))
        if not raw_binding:
            issues.append('Current score row lacks raw binding: ' + row['request_id'])
            continue
        raw = read(raw_binding['path'])
        need(raw.get('request_id') == row['request_id'], 'Current raw request identity differs: ' + row['request_id'])
        if raw.get('status') == 'RETURNED':
            new_returned += 1
        need(raw.get('status') == 'RETURNED' and raw.get('finish_reason') == 'stop'
             and not raw.get('capture_errors'), 'Current raw failure or non-stop: ' + row['request_id'])
    need(new_returned == len(public), 'Current returned records differ')
    normalized = docs['NORMALIZED_SCORES.json']['records']
    need(len({r['request_id'] for r in normalized}) == len(normalized), 'Duplicate normalized request IDs')
    need(all(r['returned'] for r in normalized), 'A normalized row is not returned')
    need(all(not r['exact'] for r in normalized if not r['valid']), 'Invalid returned output has nonzero strict exact score')
    actual_shards = dict(collections.Counter(r['cohort'] for r in normalized))
    reported_shards = analysis['shard_counts']
    need(actual_shards == reported_shards, 'Combined normalized/shard counts disagree')
    new16 = [r for r in normalized if r['cohort'] != 'original4']
    need({r['candidate_id'] for r in new16} == admitted and len(new16) == 2 * len(admitted), 'New16 scores do not cover admitted questions exactly twice')
    need(reported_shards.get('current_continuation') == len(scores), 'Current shard count mismatch')
    need(reported_shards.get('prior_recovery') == 2 * len(reused), 'Prior shard count mismatch')
    primary = analysis['new16_primary']
    need(primary['planned_roots'] == 16 and set(r['root_id'] for r in primary['roots']) == set(roots), 'Paired root denominator differs')
    need(primary['slots'] == [s for s in SLOTS if s != 'MC6'], 'Primary includes control or omits a prescribed slot')
    need((primary.get('bootstrap') or {}).get('unit') == 'root', 'Primary CI is not clustered at root')
    need(analysis.get('two_displays_are_dependent') is True, 'Dependent repeated-draw disclosure absent')
    process = docs['audits/INDEPENDENT_PROCESS_AUDIT.json']
    final = docs['audits/FINAL_RESULT_AUDIT.json']
    scheduling = docs['audits/SCHEDULING_REUSE_AUDIT.json']
    need(process.get('status') in ('PASS', 'PASS_WITH_LIMITATIONS') and process.get('mechanical_findings') == [], 'Process audit not passed')
    need(scheduling.get('status') == 'PASS' and scheduling.get('findings') == []
         and scheduling.get('mechanical_findings') == [], 'Scheduling/reuse audit not passed')
    need(final.get('status') == 'PASS' and final.get('findings') == [], 'Final result audit not strictly passed')
    cpu = docs['execution/COMBINED_ANALYSIS_CPU_VERIFICATION.json']
    need(cpu.get('status') == 'PASS' and cpu.get('findings') == [], 'Final CPU verification not passed')
    need(cpu.get('rows_verified') == len(normalized), 'Final CPU verification does not cover every normalized row')
    cpu_bindings = list(bindings_in(cpu))
    need(any(Path(b['path']).resolve() == (BASE / 'scripts/combined_analysis.py').resolve() for b in cpu_bindings), 'CPU verification does not bind final combined script')
    final_bindings = list(bindings_in(final))
    need(any(Path(b['path']).resolve() == (BASE / 'COMBINED_ANALYSIS.json').resolve() for b in final_bindings), 'Final audit does not bind combined analysis')
    target = docs['F_TARGET_REALIZATION_PRE_SOLVER.json']
    need(target.get('status') == 'COMPLETE_PRE_SOLVER' and not target.get('pending_roots'), 'Pre-Solver target review incomplete')
    need(target.get('summary', {}).get('pending_roots') == 0, 'Pre-Solver target summary has pending roots')
    for arm in ('F', 'G'):
        actual = dict(collections.Counter(s['status'] for s in slots if s['arm'] == arm))
        need(docs['GENERATION_LEDGER.json']['summary'][arm]['final_statuses'] == actual, 'Generation ledger final statuses drift: ' + arm)
    return issues, {'bindings_checked': len(checked), 'bindings_matched': sum(checked.values()),
                    'question_statuses': dict(statuses), 'source_statuses': dict(collections.Counter(source_outcomes.values())),
                    'prepared_current': len(public), 'returned_current': new_returned,
                    'score_rows_current': len(scores), 'normalized_total': len(normalized),
                    'shard_counts': actual_shards, 'pending_slots': 0,
                    'reused_questions': len(reused), 'current_questions': len(admitted - reused)}


def build_report(docs, validation, historical):
    qf, analysis = docs['QUESTION_FREEZE.json'], docs['COMBINED_ANALYSIS.json']
    slots = qf['slots']
    passed = [s for s in slots if s['status'] == 'PASS']
    c = validation['counts']
    complete = complete_primary(analysis['new16_primary'])
    initial = analysis['initial16_reported_run']
    expected_initial = {
        'status': 'PARTIAL', 'source_pass': 5, 'content_candidates': 63,
        'content_admitted': 49, 'prepared_requests': 98, 'returned': 0}
    if {k: initial[k] for k in expected_initial} != expected_initial:
        raise ValueError('Initial16 historical counts differ; update report before closeout')
    report = {'schema_version': 'original_workflow_closeout_v1', 'at': now(),
        'status': 'completed_with_coverage_shortfall', 'objective_completed': True,
        'objective_completion_scope': 'bounded original-flow continuation and reporting; planned full coverage remains false',
        'operational_closeout_completed': True, 'full_coverage_achieved': False,
        'validation_binding': binding(BASE / 'CLOSEOUT_VALIDATION.json'),
        'counts': {'planned_roots': 16, 'source_pass_roots': c['source_statuses'].get('PASS', 0),
                   'source_fail_roots': c['source_statuses'].get('FAIL', 0),
                   'planned_question_slots': len(slots), 'admitted_questions': len(passed),
                   'coverage': fraction(len(passed), len(slots)), 'pending_question_slots': 0,
                   'source_blocked_slots': c['question_statuses'].get('SOURCE_BLOCKED', 0),
                   'failed_ORDER2_slots': c['question_statuses'].get('FAIL', 0),
                   'reused_questions': c['reused_questions'], 'new_questions': c['current_questions'],
                   'reused_returned_responses': c['shard_counts']['prior_recovery'],
                   'new_prepared_responses': c['prepared_current'], 'new_returned_responses': c['returned_current'],
                   'new16_returned_responses': c['shard_counts']['prior_recovery'] + c['shard_counts']['current_continuation'],
                   'original4_returned_responses': c['shard_counts']['original4'],
                   'combined20_returned_responses': c['normalized_total'],
                   'admitted_by_arm': dict(collections.Counter(s['arm'] for s in passed))},
        'run_history': {'initial16_recorded_closeout': initial,
                        'prior_recovery16': {'semantic_questions': c['reused_questions'],
                            'returned': c['shard_counts']['prior_recovery'], 'status': 'PARTIAL_HISTORICAL'},
                        'current_continuation': {'semantic_questions': c['current_questions'],
                            'prepared': c['prepared_current'], 'returned': c['returned_current'],
                            'rerun_prior_successes': False}},
        'analysis': {k: analysis[k] for k in ('positive_difference_means', 'new16_primary',
            'new16_MC2_5_secondary', 'new16_ordering_secondary', 'new16_MC6_control',
            'new16_per_operation', 'original4_frozen', 'combined20_secondary',
            'descriptive_new16', 'descriptive_original4')},
        'complete_seven_non_MC6_slot_roots_sensitivity': complete,
        'generation_ledger_summary': docs['GENERATION_LEDGER.json']['summary'],
        'target_review': {'summary': docs['F_TARGET_REALIZATION_PRE_SOLVER.json']['summary'],
                         'completion_checks': docs['F_TARGET_REALIZATION_PRE_SOLVER.json']['completion_checks'],
                         'REALIZED_is_solver_failure': False, 'semantic_reason_and_citation_review_complete': False},
        'failed_source_roots': [s['root_id'] for s in docs['SOURCE_FREEZE.json']['outcomes'] if s['status'] != 'PASS'],
        'coverage_shortfall_reasons_zh': [
            '14304来源B1:T008把needs-check状态归给James，而该请求者实际为Linda；授权恢复预算耗尽后独立复审仍未通过，因此该root的16槽全部阻断。',
            '18个ORDER2槽缺乏必要的R03程序边、唯一全序或其他审核所需证据；保留失败和尝试记录，不补造程序关系。'],
        'strict_endpoint_zh': '已实际返回但不符合严格输出合同的响应计strict=0；无响应或内容未通过/缺失的槽不补零，不能进入要求F/G各两次返回的共同槽。',
        'execution_notes_zh': [
            'CPU准备阶段曾因旧直接审核回执的slot_id元数据兼容问题失败，模型尚未启动；失败产物及旧脚本已归档，兼容修复经过独立addendum审核，未修改题目或审核判断。',
            '旧CPU验证仅覆盖历史分片和旧脚本，已归档并由最终572行独立CPU验证取代。',
            '运行日志的Triton JIT提示只涉及延迟；退出时NCCL清理警告在全部298响应已返回后出现。最终原始响应均RETURNED、finish_reason=stop且无capture_errors。'],
        'missing_slots': [{k: s[k] for k in ('root_id', 'arm', 'slot_id', 'status')} for s in slots if s['status'] != 'PASS'],
        'identity_disclosure': {
            'content_review_new_label_actual_agent': '/root',
            'label_is_additional_independent_agent': False,
            '179186_F_actual_author': '/root', '179186_F_actual_content_reviewer': '/root/artifact_audit',
            'basis': binding(BASE / 'audits/INDEPENDENT_PROCESS_AUDIT.json')},
        'limitations_zh': [
            '这是含定向恢复的适应性续跑；初始、恢复与本次结果分列，不能声称沿用未改变的初始预算。',
            '16个计划root均保留在分母说明中；难度估计只使用双方均有两个已返回显示的共同槽，未通过与缺失槽不按零分补齐。',
            '每题两次显示和采样彼此相关；统计推断以配对root为单位，不把响应、标签或排列当作独立样本。',
            'MC2–5、排序与MC6全真控制分别报告；MC6不能证明部分选择能力、干扰项质量或全选回避。',
            '原4root仅作冻结参照；合并20root仅为次要描述性结果，不能作为新的独立复现。',
            'REALIZED只表示冻结目标关系在该题中具备，不表示Solver已失败；HOLD有效题仍保留在主难度分析。',
            '最终答案及严格输出合同有机械评分；理由蕴含、人物/数量/状态绑定与精确引文覆盖的逐项语义审核尚未完成。',
            '未训练、未改变dev/test；本结果不支持内部机制、训练收益、修复效果或稳定迁移的结论。',
            '静态回执支持作者/Gold/内容审核的实际角色区分，不能证明提供方独立性或不存在未记录的信息暴露。'],
        'training': False, 'dev_test_changed': False, 'adaptive_recovery': True,
        'dependent_repeated_draws': True, 'mechanism_claim': False,
        'process_audit_limitations': docs['audits/INDEPENDENT_PROCESS_AUDIT.json'].get('metadata_limitations', []),
        'evidence': {name: binding(BASE / name) for name in REQUIRED},
        'historical_frozen_artifacts': historical}
    return report


def markdown(report):
    a, c = report['analysis'], report['counts']
    p = a['new16_primary']
    ci = (p.get('bootstrap') or {}).get('percentile95')
    uncertainty = '该区间包含零，方向仍不确定。' if ci and ci[0] <= 0 <= ci[1] else '区间仅描述此次适应性样本中的配对差异。'
    lines = ['# 自然作者流程续跑收尾报告', '',
        f"流程已收尾，状态为 `completed_with_coverage_shortfall`。来源通过 {c['source_pass_roots']}/16；题目通过 {c['admitted_questions']}/{c['planned_question_slots']}（{pct(c['coverage'])}），无待处理槽。14304 来源失败阻断16槽，另有18个 ORDER2 槽失败；这不等于覆盖完整。", '',
        f"新16 root主分析有 {p['eligible_roots']}/16 个可配对root、{p['common_root_slots']} 个共同槽。F={pct(p['F'])}，G={pct(p['G'])}，F−G={pp(p['difference'])}，root bootstrap 95% CI {confidence(p)}。正值表示F更容易，负值表示F更难。{uncertainty}", '',
        '## 分层结果', '', '| 分析 | 可配对root | F | G | F−G | 95% CI |', '|---|---:|---:|---:|---:|---|']
    groups = [('新16主分析：排除MC6', p), ('新16 MC2–5', a['new16_MC2_5_secondary']),
              ('新16排序', a['new16_ordering_secondary']), ('新16 MC6全真控制', a['new16_MC6_control']),
              ('七个非MC6槽完整root敏感性', report['complete_seven_non_MC6_slot_roots_sensitivity']),
              ('原4冻结参照', a['original4_frozen']), ('合并20次要描述', a['combined20_secondary'])]
    for title, result in groups:
        lines.append(f"| {title} | {result['eligible_roots']} | {pct(result['F'])} | {pct(result['G'])} | {pp(result['difference'])} | {confidence(result)} |")
    lines += ['', '主分析先在每个root内平均共同槽，再等权平均root。两个显示相关，bootstrap重采样单位为root。完整槽敏感性要求双方在全部七个非MC6槽均有两个返回显示；它不会补齐缺失来源或题目。', '',
              '| 单操作 | 可配对root | 共同槽 | F | G | F−G |', '|---|---:|---:|---:|---:|---:|']
    for slot, r in a['new16_per_operation'].items():
        lines.append(f"| {slot} | {r['eligible_roots']} | {r['common_root_slots']} | {pct(r['F'])} | {pct(r['G'])} | {pp(r['difference'])} |")
    lines += ['', '## 各次运行与覆盖', '',
              '| 阶段 | 题目 | 已准备响应 | 实际返回 |', '|---|---:|---:|---:|',
              '| 初始16记录：PARTIAL；5来源、63候选、49通过 | 49 | 98 | 0 |',
              f"| 历史恢复R，成功产物复用 | {c['reused_questions']} | {c['reused_returned_responses']} | {c['reused_returned_responses']} |",
              f"| 本次新增 | {c['new_questions']} | {c['new_prepared_responses']} | {c['new_returned_responses']} |",
              f"| 新16累计（R＋本次） | {c['admitted_questions']} | {c['new16_returned_responses']} | {c['new16_returned_responses']} |",
              f"| 原4冻结参照 | {c['original4_returned_responses']//2} | {c['original4_returned_responses']} | {c['original4_returned_responses']} |",
              f"| 合并20，仅次要分析 | {c['combined20_returned_responses']//2} | {c['combined20_returned_responses']} | {c['combined20_returned_responses']} |", '',
              '本次只运行新增题目，历史成功题及原4响应未重跑。初始记录的0返回不能混写为本次返回，也不能用本次恢复成果覆盖历史PARTIAL。', '']
    lines += report['coverage_shortfall_reasons_zh'] + ['', report['strict_endpoint_zh'], '']
    lines += report['execution_notes_zh'] + ['',
              '## 解释边界与实际角色', '']
    lines += ['- ' + text for text in report['limitations_zh']]
    t = report['target_review']['summary']
    lines += ['', f"出题前目标审核：新增 {t['new_roots_audited']} root、{t['new_slot_records']} 槽，REALIZED {t['new_REALIZED']}、HOLD {t['new_HOLD']}，pending {t['pending_roots']}。目标实现与Solver是否出错是不同判断。", '',
              '`content_review_new` 是实际 `/root` 的标签，不是另一独立代理；179186F实际由 `/root` 出题、`/root/artifact_audit` 做内容审核。具体作者、Gold与审核身份及元数据缺口见过程审计。', '',
              '## 证据', '']
    for title, name in [('主分析与共同槽明细', 'COMBINED_ANALYSIS.json'), ('冻结题目和失败槽', 'QUESTION_FREEZE.json'),
                        ('出题过程计数', 'GENERATION_LEDGER.json'), ('出题前目标审核', 'F_TARGET_REALIZATION_PRE_SOLVER.json'),
                        ('独立过程审计', 'audits/INDEPENDENT_PROCESS_AUDIT.json'), ('最终结果独立审计', 'audits/FINAL_RESULT_AUDIT.json'),
                        ('复用与启动调度审计', 'audits/SCHEDULING_REUSE_AUDIT.json'),
                        ('方法快照审核', 'audits/ANALYSIS_METHOD_REVIEW.json'),
                        ('兼容补丁审核', 'audits/ANALYSIS_METHOD_REVIEW_COMPATIBILITY_ADDENDUM.json'),
                        ('收尾验证', 'CLOSEOUT_VALIDATION.json'), ('完整机读报告', 'FINAL_REPORT.json')]:
        lines.append(f'- [{title}]({name})')
    lines += ['', '旧E/P/R冻结收尾、原始响应及评分文件的SHA-256绑定见 FINAL_REPORT.json；当前目录全部文件（除manifest本身与pycache）见 FINAL_MANIFEST.json。父代理另做最终清单只读核验。', '']
    return '\n'.join(lines)


def build_manifest(historical):
    active = active_model_processes()
    if active:
        raise RuntimeError('Model process still active; logs must not be bound: ' + json.dumps(active))
    paths = sorted(p for p in BASE.rglob('*') if p.is_file()
                   and p.name != 'FINAL_MANIFEST.json' and '__pycache__' not in p.parts
                   and p.suffix != '.pyc' and not p.name.endswith('.closeout-tmp'))
    stamps = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    files = [binding(p) for p in paths]
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    current_paths = sorted(p for p in BASE.rglob('*') if p.is_file()
                           and p.name != 'FINAL_MANIFEST.json' and '__pycache__' not in p.parts
                           and p.suffix != '.pyc' and not p.name.endswith('.closeout-tmp'))
    if stamps != after or paths != current_paths or active_model_processes():
        raise RuntimeError('Directory or model state changed during manifest assembly')
    doc = {'schema_version': 'original_workflow_final_manifest_v1', 'at': now(),
           'base_directory': str(BASE), 'status': 'completed_with_coverage_shortfall',
           'files': files, 'file_count': len(files), 'total_bytes': sum(b['bytes'] for b in files),
           'excluded': ['FINAL_MANIFEST.json itself', '__pycache__/**', '*.pyc', 'atomic closeout temporary files'],
           'historical_frozen_artifacts': historical, 'model_processes_active': [],
           'training': False, 'solver_automatic_retry': False,
           'question_revisions': 'bounded adaptive recovery, see ledger',
           'independent_final_manifest_verification_owner': '/root'}
    write(BASE / 'FINAL_MANIFEST.json', doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-only', action='store_true',
                        help='Refresh manifest after a late read-only audit; preserve all other closeout files.')
    args = parser.parse_args()
    active = active_model_processes()
    if active:
        raise SystemExit('REFUSED: model process still active; no reports/log bindings written: ' + json.dumps(active))
    missing = [name for name in REQUIRED if not (BASE / name).is_file()]
    if missing:
        raise SystemExit('REFUSED: final inputs absent: ' + ', '.join(missing))
    docs = {name: read(BASE / name) for name in REQUIRED}
    if args.manifest_only:
        validation = read(BASE / 'CLOSEOUT_VALIDATION.json')
        status = read(BASE / 'STATUS.json')
        if validation.get('status') != 'PASS' or status.get('objective_completed') is not True:
            raise SystemExit('REFUSED: existing closeout is not validated')
        for name, saved in validation.get('input_bindings', {}).items():
            current = binding(BASE / name)
            if current != saved:
                raise SystemExit('REFUSED: closeout input changed; rebuild reports before manifest: ' + name)
        if set(validation.get('input_bindings', {})) != set(REQUIRED):
            raise SystemExit('REFUSED: prior closeout input set differs; rebuild reports before manifest')
        # Late audit may add files; still verify final input bindings and audit gates.
        issues, _ = validate(docs)
        if issues:
            raise SystemExit('REFUSED: ' + '\n'.join(issues))
        result = build_manifest(old_artifacts(docs['CONTINUATION_MANIFEST.json']))
        print(json.dumps({'manifest_file_count': result['file_count'], 'mode': 'manifest-only'}))
        return
    issues, counts = validate(docs)
    historical = {}
    try:
        historical = old_artifacts(docs['CONTINUATION_MANIFEST.json'])
    except (OSError, RuntimeError, ValueError) as exc:
        issues.append('Historical artifact binding failure: ' + str(exc))
    validation = {'schema_version': 'closeout_validation_v1', 'at': now(),
                  'status': 'PASS' if not issues else 'FAIL', 'findings': issues, 'counts': counts,
                  'model_processes_active': [], 'validation_scope': 'Final frozen bindings, counts, returned raw rows, paired denominator and mandatory audit gates; no semantic reason/citation certification.',
                  'input_bindings': {name: binding(BASE / name) for name in REQUIRED},
                  'historical_frozen_artifact_count': sum(len(v) for v in historical.values()),
                  'final_manifest_independent_verification_pending_with_owner': '/root'}
    write(BASE / 'CLOSEOUT_VALIDATION.json', validation)
    if issues:
        raise SystemExit('REFUSED: closeout validation failed; objective not marked complete: ' + '\n'.join(issues))
    report = build_report(docs, validation, historical)
    write(BASE / 'FINAL_REPORT.json', report)
    write(BASE / 'FINAL_REPORT.md', markdown(report))
    write(BASE / 'STATUS.json', {'at': now(), 'status': report['status'], 'objective_completed': True,
            'objective_completion_scope': report['objective_completion_scope'],
            'operational_closeout_completed': True, 'full_coverage_achieved': False,
            'pending_slots': 0, 'counts': report['counts'], 'adaptive_recovery': True,
            'semantic_reason_and_citation_review_complete': False,
            'validation': binding(BASE / 'CLOSEOUT_VALIDATION.json'),
            'report': binding(BASE / 'FINAL_REPORT.json'),
            'final_result_audit': binding(BASE / 'audits/FINAL_RESULT_AUDIT.json'),
            'final_manifest_independent_verification_owner': '/root'})
    result = build_manifest(historical)
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
                      'manifest_file_count': result['file_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
