#!/usr/bin/env python3
"""Read-only Train/DEV evidence audit. No model calls; no TEST inputs.

Run from any directory: python this_file.py --workspace /root/autodl-tmp/gxy-selfplay
Only the two named audit outputs in this script's directory are written.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--workspace', type=Path, default=Path('/root/autodl-tmp/gxy-selfplay'))
P.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
A = P.parse_args()
W = A.workspace.resolve()
bindings = {}


def bind(path, expected=None):
    p = Path(path)
    if not p.is_absolute():
        p = W / p
    p = p.resolve()
    if str(p) not in bindings:
        data = p.read_bytes()
        bindings[str(p)] = {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    row = bindings[str(p)]
    if expected:
        assert row['sha256'] == expected, ('input SHA mismatch', str(p), expected, row['sha256'])
    return p


def j(path, expected=None):
    return json.loads(bind(path, expected).read_text())


def jl(path, expected=None):
    return [json.loads(x) for x in bind(path, expected).read_text().splitlines() if x.strip()]


def ratio(n, d):
    return {'correct': n, 'calls': d, 'fraction': str(F(n, d)), 'accuracy': n / d}


def native_stats(rows):
    out = {}
    for typ in sorted({r['question_type'] for r in rows}):
        rr = [r for r in rows if r['question_type'] == typ]
        n = sum(r['expected_full_calls'] for r in rr)
        c = sum(F(r['full_exact_rate']) * r['expected_full_calls'] for r in rr)
        assert c.denominator == 1
        partial = sum(F(r['full_partial_mean']) * r['expected_full_calls'] for r in rr) / n
        out[typ] = dict(questions=len(rr), roots=len({r['root_id'] for r in rr}), **ratio(int(c), n),
                        signed_partial_fraction=str(partial), signed_partial=float(partial))
        if typ == 'ordering':
            out[typ]['pair_accuracy_fraction'] = str((partial + 1) / 2)
            out[typ]['pair_accuracy'] = float((partial + 1) / 2)
    return out


def matrix_stats(rows):
    out = {}
    for typ in sorted({r['question_type'] for r in rows}):
        rr = [r for r in rows if r['question_type'] == typ]
        out[typ] = {'questions': len(rr), 'roots': len({r['root_id'] for r in rr}), 'models': {}}
        for model in ('M0', 'M_clean_policy'):
            out[typ]['models'][model] = {}
            for ctx in ('full_context', 'no_context'):
                c = sum(r['models'][model][ctx + '_correct_count'] for r in rr)
                n = sum(r['models'][model][ctx + '_total_count'] for r in rr)
                out[typ]['models'][model][ctx] = ratio(c, n)
    return out


# Actual first formal cohort: frozen training manifest, not a hand-picked frontier.
old_data = 'restart-n5-runs/h4_round0_b0_matched_training_v1_live_2'
old_formal = 'restart-n5-runs/h4_round0_b0_formal_training_v1_live_2'
preflight = j(old_formal + '/training_preflight.json')
old_manifest = jl(old_data + '/m_clean_policy_training_manifest.jsonl',
                  preflight['data_source_sha256']['m_clean_policy_training_manifest.jsonl'])
bind(old_data + '/m_clean_policy_train.jsonl', preflight['data_source_sha256']['m_clean_policy_train.jsonl'])
old_report = j(old_formal + '/final_formal_training_report.json')
matrix = jl('restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/candidate_transition_matrix.jsonl')
matrix_by_id = {r['candidate_id']: r for r in matrix}
old_ids = {r['candidate_id'] for r in old_manifest}
assert len(old_manifest) == len(old_ids) == 386
assert len(matrix_by_id) == len(matrix) == 2294
assert old_ids <= matrix_by_id.keys()
old_rows = [matrix_by_id[r['candidate_id']] for r in old_manifest]
assert all(r['split'] == 'train' for r in old_rows)
assert all(r['flags']['directly_trained_by_clean'] for r in old_rows)
assert len({r['root_id'] for r in old_rows}) == 68
old_shapes = defaultdict(Counter)
pool_cache = {}
for row in old_manifest:
    p = bind(row['candidate_pool_path'], row['candidate_pool_sha256'])
    if p not in pool_cache:
        pool_cache[p] = {q['candidate_id']: q for q in json.loads(p.read_text())}
    q = pool_cache[p][row['candidate_id']]
    assert q['question_type'] == row['question_type']
    field = 'items' if row['question_type'] == 'ordering' else 'options'
    old_shapes[row['question_type']][len(q[field])] += 1
old_gold = Counter(r['gold_cardinality'] for r in old_manifest if r['question_type'] == 'multi_select')
assert dict(old_gold) == {2: 28, 3: 94, 4: 47, 5: 3}
old_quality = j('offline-deliveries/h4_codex_dev_ceiling_review_v128/evidence/v127_ceiling_training_findings.json')
tsv = bind('restart-n5-runs/h4_round0_train_ordering_full_manual_review_v1_live_1/final_train_ordering_manual_review.tsv')
review_rows = list(csv.DictReader(io.StringIO(tsv.read_text()), delimiter='\t'))
actual_review = [r for r in review_rows if r['candidate_id'] in old_ids]
assert {r['candidate_id'] for r in actual_review} == {r['candidate_id'] for r in old_rows if r['question_type'] == 'ordering'}
review_status = Counter(r['gold_status'] for r in actual_review)
assert len(actual_review) == 159 and sum(v for k, v in review_status.items() if k != 'valid') == 77

# Actual second SFT cohort, joined to every existing native M0/M_clean result.
pool = j('offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json')['questions']
new_by_id = {r['candidate_id']: r for r in pool}
assert len(pool) == len(new_by_id) == 363
assert len({r['root_id'] for r in pool}) == 68
assert Counter(r['question_type'] for r in pool) == {'multi_select': 238, 'ordering': 125}
new_gold = Counter(len(r['gold']) for r in pool if r['question_type'] == 'multi_select')
assert dict(new_gold) == {2: 65, 3: 59, 4: 57, 5: 57}
assert all(len(q['items'] if q['question_type'] == 'ordering' else q['options']) == 6 for q in pool)
native_roots = [
 'h4_codex_sequential_ab_pilot_v112/evaluation/run',
 'h4_codex_serialized_ab_validation_v113/evaluation/run',
 'h4_codex_wording_minfix_v114/evaluation/run',
 'h4_codex_newroots_ab_validation_v115/evaluation/run',
 'h4_codex_screened_ab_validation_v117/evaluation/run',
 'h4_codex_fullscale_ab_v118/evaluations/validation/run',
 'h4_codex_fullscale_ab_v118/evaluations/production/run',
 'h4_codex_scene_retry_v120/evaluations/paired_v1/run',
 'h4_codex_all_failed_repair_v121/evaluations/paired_v1/run',
 'h4_codex_remaining_six_repair_v122/evaluations/paired_v1/run']
joined = {'M0': {}, 'M_clean_policy': {}}
raw_calls_verified = Counter()
public_fields_verified = Counter()
per_question = []
for root in native_roots:
    for sub, model in [('base', 'M0'), ('trained', 'M_clean_policy')]:
        run = W / 'offline-deliveries' / root / sub
        results = j(run / 'RESULTS.json')
        assert results['solver_model'] == model and not results['call_set_errors']
        manifest = j(run / 'manifest.json')
        for name in ('questions.json', 'gold.json', 'canonical_bindings.json'):
            bind(run / name, manifest['input_sha256'][name])
        for name, sha in manifest['input_sha256'].items():
            if name.startswith('sources/'):
                bind(run / name, sha)
        qs = {q['candidate_id']: q for q in j(run / 'questions.json')['candidates']}
        cb = {q['candidate_id']: q for q in j(run / 'canonical_bindings.json')['candidates']}
        gold = j(run / 'gold.json')
        requests = defaultdict(list)
        for req in manifest['requests']:
            if req['candidate_id'] in new_by_id:
                requests[req['candidate_id']].append(req)
        for r in results['questions']:
            cid = r['candidate_id']
            if cid not in new_by_id:
                continue
            assert cid not in joined[model], ('duplicate native result', model, cid)
            q, nq = qs[cid], new_by_id[cid]
            for key in ('candidate_id', 'root_id', 'question_type', 'question', 'options', 'items'):
                assert q.get(key, {}) == nq.get(key, {}), ('public input mismatch', cid, key)
            canonical_gold = [cb[cid]['content_id_by_label'][label] for label in nq['gold']]
            assert canonical_gold == gold[cid] == cb[cid]['gold_content_ids']
            public_fields_verified[model] += 1
            assert not r['data_errors'] and r['actual_full_calls'] == r['expected_full_calls']
            assert len(requests[cid]) == r['expected_full_calls']
            exact, partial = 0, F(0)
            seen = set()
            for req in requests[cid]:
                assert req['mode'] == 'full_context' and req['request_id'] not in seen
                seen.add(req['request_id'])
                raw = j(run / 'responses' / (req['request_id'] + '.json'))
                assert raw['request_id'] == req['request_id'] and raw['candidate_id'] == cid
                assert raw['evaluation_valid'] and not raw['data_errors']
                s = raw['score']
                assert s['parse_status'] == 'success' and s['scored']
                box = re.fullmatch(r'\s*<box>([^<]+)</box>\s*', raw['raw_output'])
                assert box, ('unrecognized finite-choice response', req['request_id'])
                labels = [x.strip() for x in box.group(1).split(',')]
                assert labels == s['display_prediction'] and len(set(labels)) == len(labels)
                pred = [req['variant']['display_to_content'][label] for label in labels]
                if q['question_type'] == 'ordering':
                    assert pred == s['content_prediction'] and gold[cid] == s['content_gold']
                else:
                    assert set(pred) == set(s['content_prediction']) and set(gold[cid]) == set(s['content_gold'])
                if q['question_type'] == 'ordering':
                    assert set(pred) == set(gold[cid])
                    ranks = {x: i for i, x in enumerate(pred)}
                    pairs = [(a, b) for i, a in enumerate(gold[cid]) for b in gold[cid][i+1:]]
                    tp = sum(ranks[a] < ranks[b] for a, b in pairs)
                    score = F(2 * tp - len(pairs), len(pairs))
                    is_exact = pred == gold[cid]
                else:
                    gs, ps = set(gold[cid]), set(pred)
                    tp, fp = len(gs & ps), len(ps - gs)
                    non_gold = len(req['variant']['display_to_content']) - len(gs)
                    score = F(tp, len(gs)) - (F(fp, non_gold) if non_gold else F(0))
                    is_exact = gs == ps
                assert is_exact == s['exact_match'] and score == F(s['partial_score_fraction'])
                exact += is_exact
                partial += score
                raw_calls_verified[model] += 1
            assert F(exact, len(seen)) == F(r['full_exact_rate'])
            assert partial / len(seen) == F(r['full_partial_mean'])
            joined[model][cid] = r
            per_question.append({'model': model, 'candidate_id': cid, 'root_id': r['root_id'],
                                 'question_type': r['question_type'], 'correct': exact, 'calls': len(seen),
                                 'signed_partial_fraction': r['full_partial_mean'],
                                 'native_results_path': str(run / 'RESULTS.json')})
assert all(set(rows) == set(new_by_id) for rows in joined.values())
new_stats = {model: native_stats(list(rows.values())) for model, rows in joined.items()}
assert new_stats['M0']['multi_select']['correct'] == 1665
assert new_stats['M_clean_policy']['multi_select']['correct'] == 2005
assert new_stats['M0']['ordering']['correct'] == 41
assert new_stats['M_clean_policy']['ordering']['correct'] == 38

# DEV is a separate held-out descriptive control; no TEST content is read.
dev_summary = j('offline-deliveries/h4_codex_dev_eval_v127/PAIRED_SUMMARY.json')
dev_calls = j('offline-deliveries/h4_codex_dev_eval_v127/PER_CALL.json')
dev_scope = j('offline-deliveries/h4_codex_dev_eval_v127/SCOPE.json')
dev_control = j('offline-deliveries/h4_codex_base_order_control_v130/SUMMARY.json')
control_calls = j('offline-deliveries/h4_codex_base_order_control_v130/PER_CALL.json')
assert dev_summary['status'] == 'COMPLETE' and not dev_summary['errors']
dev_stats = {}
for model in ('baseline', 'new'):
    rr = [r[model] for r in dev_calls]
    assert len(rr) == 780 and all(r['valid'] and not r['errors'] for r in rr)
    dev_stats[model] = {}
    for label, subset in (
        ('multi_select', [r for r in rr if r['question_type'] == 'multi_select']),
        ('multi_select_gold2345', [r for r in rr if r['question_type'] == 'multi_select' and r['gold_count'] < 6]),
        ('ordering', [r for r in rr if r['question_type'] == 'ordering'])):
        dev_stats[model][label] = dict(questions=len({r['candidate_id'] for r in subset}),
                                    **ratio(sum(r['exact_match'] for r in subset), len(subset)))
        if label == 'ordering':
            pair = sum(F(r['pair_accuracy']) for r in subset) / len(subset)
            dev_stats[model][label]['pair_accuracy'] = float(pair)
            dev_stats[model][label]['pair_accuracy_fraction'] = str(pair)
base_rr = [r['models']['base'] for r in control_calls]
assert len(base_rr) == 168 and all(r['valid'] and not r['errors'] for r in base_rr)
base_pair = sum(F(r['pair_accuracy']) for r in base_rr) / len(base_rr)
dev_stats['M0_ordering_control'] = dict(questions=14, **ratio(sum(r['exact_match'] for r in base_rr), 168),
                                      pair_accuracy=float(base_pair), pair_accuracy_fraction=str(base_pair))
assert dev_stats['baseline']['ordering']['correct'] == 4
assert dev_stats['new']['ordering']['correct'] == 168
assert dev_stats['M0_ordering_control']['correct'] == 3

# Existing controlled same-input QA feedback on/off experiment, independently recomputed.
ab = 'restart-n5-runs/h4_r02_feedback_ablation_p0_p1_v1_live_1'
ab_preflight = j(ab + '/p0_preflight_report.json')
assert ab_preflight['pass'] and ab_preflight['root_count'] == 6
j(ab + '/execution_contract.json')
pairing = jl(ab + '/pairing_manifest.jsonl')
assert len(pairing) == 6
comparison = j(ab + '/reports/p1_comparison.json')
cm = jl(ab + '/reports/candidate_context_metrics.jsonl')
quality = j(ab + '/reports/blind_review_quality_summary.json')
for ref in quality['review_sources']:
    bind(ref['review_source']['path'], ref['review_source']['sha256'])
bind(quality['mapping_source']['path'], quality['mapping_source']['sha256'])


historical_mutable_code_differences = []


def bind_refs(obj):
    if isinstance(obj, dict):
        if 'path' in obj and 'sha256' in obj and isinstance(obj['path'], str):
            if Path(obj['path']).suffix == '.py':
                p = bind(obj['path'])
                actual = bindings[str(p)]['sha256']
                if actual != obj['sha256']:
                    detail = {'path': str(p), 'historical_expected_sha256': obj['sha256'], 'current_sha256': actual,
                              'scope': 'mutable historical analysis code; not executed, not an input to this recomputation'}
                    if detail not in historical_mutable_code_differences:
                        historical_mutable_code_differences.append(detail)
            else:
                bind(obj['path'], obj['sha256'])
        for val in obj.values():
            if isinstance(val, (list, dict)):
                bind_refs(val)
    elif isinstance(obj, list):
        for val in obj:
            bind_refs(val)


bind_refs(pairing)
feedback_metrics = []
for model in ('M0', 'M_clean_policy', 'M_inclusive_policy'):
    for arm in ('feedback_off', 'feedback_on'):
        out = {'model': model, 'arm': arm}
        for context in ('full_context', 'no_context'):
            rr = [r for r in cm if r['model'] == model and r['arm'] == arm and
                  r['context_mode'] == context and r['question_type'] != 'ordering']
            assert all(r['technical_complete'] and r['scored'] == r['requests'] for r in rr)
            by_root = defaultdict(list)
            for r in rr:
                by_root[r['root_id']].append(F(r['correct'], r['requests']))
            assert len(by_root) == 6
            root_means = {root: sum(vals) / len(vals) for root, vals in by_root.items()}
            root_mean = sum(root_means.values()) / 6
            source = next(r for r in comparison['all_models_main'] if r['model'] == model and
                          r['arm'] == arm and r['context_mode'] == context)
            assert abs(float(root_mean) - source['root_macro_candidate_accuracy']) < 1e-12
            out[context] = dict(questions=len(rr), **ratio(sum(r['correct'] for r in rr), sum(r['requests'] for r in rr)),
                               root_macro_candidate_accuracy=float(root_mean), root_macro_fraction=str(root_mean),
                               root_candidate_means={k: str(v) for k, v in sorted(root_means.items())})
        out['root_macro_context_gain'] = float(F(out['full_context']['root_macro_fraction']) - F(out['no_context']['root_macro_fraction']))
        feedback_metrics.append(out)
quality_stats = []
for arm in ('feedback_off', 'feedback_on'):
    rr = [r for r in quality['reviewed_candidate_rows'] if r['arm'] == arm and r['question_type'] != 'ordering']
    count = Counter(r['gold_assessment'] for r in rr)
    assert len(rr) == 24
    source = next(r for r in quality['groups'] if r['arm'] == arm and r['root_id'] == 'all' and r['scope'] == 'main')
    assert all(count[k] == v for k, v in source['gold_assessment_counts'].items())
    quality_stats.append({'arm': arm, 'reviewed': len(rr), 'gold_assessment_counts': dict(count),
                          'gold_supported_rate': count['supported'] / len(rr),
                          'historical_feedback_relation_counts': dict(Counter(r['feedback_relation_present'] for r in rr))})

eight = j('restart-n5-runs/h4_round1_paired_blind_solver_compare_v1_live_1/final_paired_solver_comparison.json')
eight_qa = jl('restart-n5-runs/h4_round1_paired_blind_fresh_qa_v1_live_8/paired_dialogue_manifest.jsonl')
r03 = j('restart-n5-runs/h4_round1_r03_first_round_no_feedback_6roots_live_1/comparison_to_feedback_r03.json')
p22 = j('restart-n5-runs/h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1/reports/first_acceptance_report.json')
bind('offline-deliveries/h4_codex_qa_feedback_filter_v1/README.md')
bind('offline-deliveries/h4_codex_qa_feedback_filter_v1/reports/FEEDBACK_REWRITE.md')
for p in ('offline-deliveries/h4_codex_train_pool_v125/PROTOCOL.json',
          'offline-deliveries/h4_codex_train_pool_v125/SUMMARY.json',
          'offline-deliveries/h4_codex_fullscale_ab_v118/evidence/CONTENT_CONTRACT.md',
          'offline-deliveries/h4_codex_fullscale_ab_v118/evidence/BASELINE_POLICY.md',
          'current_method_github/scripts/current_method/restart_n5_spell/h4_round0_b0_k3_difficulty_v1/contract.py'):
    bind(p)

protocol_table = [
 {'dimension': '实际训练池', 'round1': '386题/68roots；SC55+MS172+排序159', 'round2': '363题/68roots；MS238+排序125；new-only，无旧题回放'},
 {'dimension': '题型规模', 'round1': 'MS5选项，Gold2/3/4/5=28/94/47/3；排序4项', 'round2': 'MS6选项，Gold2/3/4/5=65/59/57/57；排序6项；Gold6控制不训练'},
 {'dimension': '排序目标', 'round1': '历史多种事件先后/发生/出现表述；非统一六场景规则', 'round2': '六个A3+B3场景各取一项，按供给对话中的出现顺序；场景结构高度一致'},
 {'dimension': '同一排序任务的展示', 'round1': '4个cyclic展示只轮换条目行，I标签身份保留', 'round2': '12个gold-blind随机重排，同时重新绑定显示标签'},
 {'dimension': '评估调用', 'round1': 'MS5展示×3=15；排序4展示×3=12；各有full和no-context', 'round2': 'MS6随机展示×3=18；排序12随机展示×1=12；full-only'},
 {'dimension': '筛选模型', 'round1': '基于M0难度与旧EVIDENCE_CLEAN规则的实际已训练集合', 'round2': '已有M_clean完整有效且0<mean signed partial<1；排除Gold6'},
 {'dimension': '训练暴露', 'round1': '表内M_clean已直接训练过这些题，M0未训练', 'round2': '表内M0与第一轮M_clean均未训练这些新题；roots曾在第一轮Train出现'},
 {'dimension': 'Gold可靠性', 'round1': '既有语义审核对实际排序159题报告82valid、77nonvalid；本次只连接核对', 'round2': '来源、引用、说话者、SHA、Gold与公开字段冻结检查；同源模型审核，不等同独立人工认证'},
 {'dimension': '运行环境', 'round1': '旧vLLM0.10.2及旧展示/解析协议', 'round2': 'vLLM0.21、finite-choice约束输出及新版随机显示协议'},
 {'dimension': '可得结论', 'round1': '描述这批实际训练题上的原生表现', 'round2': '描述新题在现有原生协议下更低的整题正确率；不能量化纯难度提升或归因反馈'}]

inventory = [
 {'experiment': 'R02 P0/P1 六root同输入QA反馈开关', 'source': ab, 'design': '相同冻结Parent+R01+R02全文；off为空包、on为非空旧QA反馈加使用说明；同Qwen3-14B程序；单次生成；SC+MS主分析；不训练',
  'root_count': 6, 'metrics': feedback_metrics, 'blind_sample_quality': quality_stats,
  'supports': '已实施同输入QA反馈开关对照，不能说历史完全没有消融',
  'limits': ['扩充对话已经生成，不能隔离对话扩充反馈作用', 'off145/on153题，不是等题量', '六roots、单次生成且无显式生成seed', '排序被排除主效果，盲审每arm主样本24题', '不是当前Codex流程对照，也未测试后续训练增益'],
  'conclusion': '准确率下降伴随Gold支持率下降，平均上下文收益未增强；不支持可靠且可回答的难度提升'},
 {'experiment': '八root control/feedback扩充分支后盲生成新QA', 'source': 'restart-n5-runs/h4_round1_paired_blind_solver_compare_v1_live_1',
  'root_count': len(eight_qa), 'candidate_alignment': eight['candidate_alignment'],
  'metrics': eight['models']['M_clean_policy'],
  'conclusion': '相同祖先root、不同R02全文和新QA；276对254题；无完全相同题对。探索性分支比较，不能隔离当前Codex反馈或等量训练效果'},
 {'experiment': 'R03 no-feedback生成比较', 'source': 'restart-n5-runs/h4_round1_r03_first_round_no_feedback_6roots_live_1',
  'reported': r03, 'conclusion': '171对23候选，root数和证据/意图门控不同；语义覆盖未对齐，不能据产量推断反馈优劣'},
 {'experiment': 'P2-2反馈迁移验收', 'source': 'restart-n5-runs/h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1',
  'engineering_acceptance': p22['engineering_acceptance'], 'content_feasibility': p22['content_feasibility'],
  'expansion_readiness': p22['expansion_readiness'], 'solver_calls': p22['solver_calls'],
  'conclusion': '工程接通PASS，内容FAIL/NOT_READY；接入成功不能当作反馈有效'},
 {'experiment': '当前Codex反馈改写/过滤与A3+B3批量流程', 'source': 'offline-deliveries/h4_codex_qa_feedback_filter_v1',
  'conclusion': '已有离线诊断、提示修订、审核和修复记录；本次检查的最终交付未发现当前Codex反馈开/关且来源、预算、题量匹配的训练消融。A/B指两条扩充路线，不能当作反馈treatment/control',
  'limits': '这是所列文件范围内的证据判断，不声称遍历了整个项目所有历史实验'}]

report = {
 'schema_version': 'v132_feedback_and_difficulty_audit_v1', 'status': 'PASS',
 'scope': {'method': 'CPU offline deterministic recomputation and source-hash verification', 'new_inference_calls': 0,
           'new_training': False, 'new_semantic_judgments': False, 'test_answer_or_content_reads': False,
           'historical_files_modified': False, 'independent_questions_are_not_rollout_calls': True},
 'findings': [
  '实际训练池比较必须使用第一轮386题与第二轮363题，不能用123道历史frontier代替第一轮总体',
  '第二轮实际363题原生结果完整连接，M0/MS38.87%、排序2.73%；旧M_clean/MS46.80%、排序2.53%',
  '相同冻结模型下第二轮整题正确率较低，但题型项数、随机标签、Gold审核、筛选与训练暴露同时变化，不能作纯难度因果结论',
  '历史存在六root同输入QA反馈开关消融，结果不支持可靠可回答的难度提升；不能写成已有正向反馈有效证据',
  '当前Codex审核/修复和第二轮训练收益不能单独归因于反馈；尚需匹配当前流程的控制实验'],
 'round1_actual_training': {'questions': 386, 'roots': 68, 'question_type_counts': dict(Counter(r['question_type'] for r in old_rows)),
     'cardinality_histograms': {k: dict(sorted(v.items())) for k, v in old_shapes.items()},
     'multi_select_gold_counts': dict(sorted(old_gold.items())), 'accuracy': matrix_stats(old_rows),
     'historical_ordering_semantic_review': {'actual_training_questions_matched': 159, 'gold_status_counts': dict(review_status),
        'nonvalid': 77, 'nonvalid_fraction': '77/159', 'provenance': '既有final_train_ordering_manual_review.tsv语义审核；本次只做ID连接和重算，未重审'}} ,
 'round1_full_train_bank_secondary_descriptive': {'questions': len(matrix), 'roots': len({r['root_id'] for r in matrix}), 'accuracy': matrix_stats(matrix)},
 'round2_actual_training': {'questions': 363, 'roots': 68, 'question_type_counts': dict(Counter(q['question_type'] for q in pool)),
     'multi_select_gold_counts': dict(sorted(new_gold.items())), 'option_or_item_count': 6,
     'native_accuracy': new_stats, 'join_audit': {'M0_unique_matches': len(joined['M0']), 'M_clean_unique_matches': len(joined['M_clean_policy']),
       'missing': [], 'duplicates': [], 'public_fields_and_content_gold_verified': dict(public_fields_verified),
       'raw_output_reparse_score_and_result_verifications': dict(raw_calls_verified),
       'partial_score_formula': 'MS: TP/|Gold| - FP/(N-|Gold|), Gold=N时只取TP/|Gold|；排序:2*pair_accuracy-1'},
     'per_question_native_results': sorted(per_question, key=lambda r: (r['candidate_id'], r['model']))},
 'second_round_dev_separate': {'population': dev_summary['population'], 'results': dev_stats,
    'scope': '7个有交付题DEV roots/固定15roots；48题全部保留，未按分数过滤；M0只有排序对照，无DEV MS测量',
    'post_second_training_use': 'new仅说明训练后行为；不得用于判定训练输入的固有难度'},
 'difficulty_comparability_table': protocol_table,
 'feedback_evidence_inventory': inventory,
 'historical_mutable_code_differences': historical_mutable_code_differences,
 'causal_claim_matrix': [
    {'claim': '当前流程能产出工程检查通过的扩充/题目并完成训练', 'judgment': '有工程和交付记录支持；不等于全面语义正确或反馈净增益'},
    {'claim': '反馈本身让问题更难而且答案可靠', 'judgment': '当前证据不足；历史六root对照还有Gold支持下降反证'},
    {'claim': '第二轮排序100%说明广泛排序能力解决', 'judgment': '不支持；仅14题7个覆盖DEV roots，六场景模板高度一致'},
    {'claim': '第一轮训练把原本会答的第二轮排序教坏', 'judgment': 'DEV M0仅3/168、旧M_clean4/168，不支持此断言'},
    {'claim': '整题低分代表所有局部理解都差', 'judgment': '不支持；第二轮Train排序pair约70.91%/72.14%，DEV M0/旧M_clean约70.63%/70.00%'}],
 'recommended_next_experiment': '保持当前数据、提示、来源和预算冻结，在Train预指定匹配roots做当前流程的反馈开/关生成及独立盲审；先报告可答率、有效产量、固定模型难度，再做等题量/等tokens训练对照。DEV用于协议固定后的验证，TEST继续封存。',
 'reproduction': {'script': str(Path(__file__).resolve()), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'command': 'python evidence/reproduce_feedback_and_difficulty_audit.py --workspace /root/autodl-tmp/gxy-selfplay'},
 'source_bindings': sorted(bindings.values(), key=lambda r: r['path'])}


def pct(x):
    return f'{float(x) * 100:.2f}%'


md = [
 '# 反馈证据与两轮难度可比性审计', '',
 '本报告只读重算 Train/DEV 既有记录，没有新增模型调用、训练或语义裁决，没有读取 TEST 题目/答案/对话。所有引用文件绑定 SHA-256；JSON 保存逐题连接、分子分母和原始来源。历史 pairing manifest 的可变 analysis.py 与其旧SHA不一致，已单列披露；本次不执行该旧脚本，使用本报告独立脚本重算保存的指标。', '',
 '**已有证据能说明第二轮题目在现有协议下整题正确率更低，不能把差值解释为纯难度提升，也不能把训练增益单独归因于反馈。历史确实做过同输入 QA 反馈开关对照，结果未支持可靠且可回答的难度提升。**', '',
 '## 实际训练池上的原生表现', '',
 '第一轮取正式训练 manifest 的386题，第二轮取最终363题。第二轮363题对两模型均完整连接，无缺失和重复；公开题干/选项/条目及内容 Gold 与冻结 native 输入一致，逐调用原始输出重新解析和评分后与 RESULTS 完全一致。', '',
 '| 实际训练题 | 第一轮题数 | 第二轮题数 | M0 第一轮 | M0 第二轮 | 第一轮 M_clean 第一轮题 | 同一 M_clean 第二轮新题 |',
 '|---|---:|---:|---:|---:|---:|---:|']
for typ, label in [('multi_select', '多选'), ('ordering', '排序')]:
    old = report['round1_actual_training']['accuracy'][typ]
    new = new_stats
    md.append(f"| {label} | {old['questions']} | {new['M0'][typ]['questions']} | {pct(old['models']['M0']['full_context']['accuracy'])} | {pct(new['M0'][typ]['accuracy'])} | {pct(old['models']['M_clean_policy']['full_context']['accuracy'])} | {pct(new['M_clean_policy'][typ]['accuracy'])} |")
md += ['',
 '第一轮另有55道单选。第一轮表内 M_clean 已直接训练过该批题；第二轮表内用的是第一轮 M_clean，在这些新题上尚未二轮训练。两轮都经过难度筛选且筛选模型不同，不能当作随机总体样本。第一轮完整2294题 Train bank 另存 JSON，属于次要描述，不能与实际386题训练集合混用。', '',
 '第二轮排序虽整序只有2.73%/2.53%，M0/旧M_clean 的成对正确率分别为70.91%/72.14%。六项排序的全对比四项严格，完整随机重标标签也改变任务；不能只看整序值推断局部理解。', '',
 '## 协议与质量可比性', '', '| 维度 | 第一轮 | 第二轮 |', '|---|---|---|']
for row in protocol_table:
    md.append(f"| {row['dimension']} | {row['round1']} | {row['round2']} |")
md += ['',
 '历史排序质量数来自已存在的逐题语义审核：对实际第一轮 M_clean 排序159题，82题 valid，77题 nonvalid。本次只核验 ID 连接并复算统计，未重新语义审核；不能将旧 EVIDENCE_CLEAN 标签当作今天已重新确认答案可靠。', '',
 '## DEV 单独呈现', '',
 '| 模型/阶段 | 多选34题 | Gold2–5多选27题 | 排序14题 | 排序成对正确率 |',
 '|---|---:|---:|---:|---:|']
for model, name in [('baseline', '第一轮M_clean'), ('new', '第二轮训练后')]:
    d = dev_stats[model]
    md.append(f"| {name} | {pct(d['multi_select']['accuracy'])} | {pct(d['multi_select_gold2345']['accuracy'])} | {pct(d['ordering']['accuracy'])} | {pct(d['ordering']['pair_accuracy'])} |")
md.append(f"| M0额外排序对照 | 未测 | 未测 | {pct(dev_stats['M0_ordering_control']['accuracy'])} | {pct(base_pair)} |")
md += ['',
 'DEV 固定15个roots，仅7个roots产出48题；所有48题均保留，无按分数择题。168个排序调用是14题×12展示，不能算168道独立题。M0为3/168、旧M_clean为4/168，不能据此说第一轮把原本能做的排序教坏。训练后100%只代表这批同构六场景题表现，不是普遍排序能力的结论。', '',
 '## 反馈有效性：已有与缺少的证据', '',
 '最接近受控实验的是 R02 P0/P1 六root对照：同一冻结 Parent+R01+R02 对话、同一 Qwen3-14B QA程序，off为空反馈包、on为旧QA反馈加使用说明。主分析只含单选和多选；off145题、on153题，单次生成，无后续训练。扩充对话预先已生成，因此它检验新增 QA 反馈机制，不检验扩充反馈。', '',
 '| 同输入QA反馈开关（root等权） | off | on |', '|---|---:|---:|']
for model in ('M0', 'M_clean_policy', 'M_inclusive_policy'):
    off = next(r for r in feedback_metrics if r['model'] == model and r['arm'] == 'feedback_off')
    on = next(r for r in feedback_metrics if r['model'] == model and r['arm'] == 'feedback_on')
    md.append(f"| {model} full-context正确率 | {pct(off['full_context']['root_macro_candidate_accuracy'])} | {pct(on['full_context']['root_macro_candidate_accuracy'])} |")
md += [
 '| M0 full−no-context增益 | 18.36pp | 17.62pp |',
 '| 每arm固定24题盲审样本Gold支持率 | 15/24（62.50%） | 11/24（45.83%） |', '',
 'M0下降2.64个百分点，但Gold支持率也下降，平均上下文收益未增强，不能把更低正确率直接记为成功增难。盲审比例只描述固定样本，不能推广为全池缺陷率。六roots、单次生成、题数不同，也不能推出任何反馈都无效。', '',
 '另有八root同祖先 control/feedback 扩充分支比较：随后盲生成的题为276对254，完全相同题对为0；M_clean整题74.17%→72.22%，root等权上下文增益20.51pp→21.72pp。它提供探索性分支证据，但对话和题内容都变了，不能隔离当前Codex反馈净效应。R03的171对23候选还混入门控与root数差异；P2-2工程验收PASS而内容FAIL/NOT_READY，也不构成正向效果。', '',
 '当前Codex有标准化扩充、出题、审核、修复和冻结流程，也有离线反馈改写记录；本次所列最终交付中未发现当前流程的匹配反馈开关与等量训练消融。A/B是两条扩充路线，不是反馈开/关实验。二轮训练增益同时混合新增数据、Gold修复、模板适应和训练作用，不能单独归因反馈。', '',
 '下一项能支持论文因果结论的实验应在预指定 Train roots 上冻结来源、提示、预算和采样，做当前反馈开/关生成及独立盲审，报告可答率、有效产量和固定模型难度，再做等题量/等tokens训练对照。DEV用于冻结协议的验证，TEST继续封存。', '',
 '## 重算', '',
 '```bash',
 'python offline-deliveries/h4_codex_round2_data_bundle_v132/evidence/reproduce_feedback_and_difficulty_audit.py --workspace /root/autodl-tmp/gxy-selfplay',
 '```', '',
 f"输入绑定：{len(bindings)}个文件。第二轮原始逐调用验证：M0 {raw_calls_verified['M0']}次，M_clean {raw_calls_verified['M_clean_policy']}次。明细见同目录 FEEDBACK_AND_DIFFICULTY_AUDIT.json。", '']
A.output.mkdir(parents=True, exist_ok=True)
(A.output / 'FEEDBACK_AND_DIFFICULTY_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(A.output / 'FEEDBACK_AND_DIFFICULTY_AUDIT.md').write_text('\n'.join(md))
print(json.dumps({'status': 'PASS', 'source_bindings': len(bindings), 'second_round_raw_calls_verified': dict(raw_calls_verified), 'output': str(A.output)}, ensure_ascii=False))
