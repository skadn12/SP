"""Independently replay frozen v57 paired responses and export one final report.

Read-only with respect to all original inputs and runner artifacts. No model
calls, retraining, question selection, or edits to question/Gold are performed.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DELIVERIES = HERE.parent
OLD_REPORT = DELIVERIES / 'h4_new_root_collection_v57'
OLD_RUNS = DELIVERIES / 'h4_sequential_ab_v57_qa/runs'
SCORER = DELIVERIES / 'h4_feedback_training_pool_v15_qa/evaluation/scoring.py'
TRAINED_PATH = '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'
BASE_PATH = '/root/autodl-tmp/Qwen3-4B'
OUTPUTS = ['REPORT.md', 'COMPARISON.json', 'PER_QUESTION.md', 'PAIRED_CALLS.json', 'VERIFICATION.json', 'RESULT_BINDINGS.json']
MANIFEST_ALLOWED_DIFFERENCES = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def bind(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


def fvalue(value):
    value = Fraction(value)
    return {'fraction': str(value), 'value': float(value), 'percent': 100 * float(value)}


def fmean(values):
    return sum(values, Fraction()) / len(values)


def audit_old_bindings():
    path = OLD_REPORT / 'RESULT_BINDINGS.json'
    rows = read(path)['files']
    for row in rows:
        require(sha(row['path']) == row['sha256'], 'OLD_FILE_CHANGED:' + row['path'])
    return {'status': 'PASS', 'binding': bind(path), 'verified_files': len(rows)}, rows


def load_scorer():
    spec = importlib.util.spec_from_file_location('v57_paired_original_scorer', SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def replay(directory, manifest, questions, gold, canonical, scorer, model):
    require((directory / 'COMPLETE.json').is_file(), 'SOLVER_NOT_COMPLETE:' + str(directory))
    requests = {r['request_id']: r for r in manifest['requests']}
    require(len(requests) == manifest['request_count'], 'DUPLICATE_REQUESTS')
    paths = sorted((directory / 'responses').glob('*.json'))
    require(len(paths) == len(requests), 'RESPONSE_COUNT_MISMATCH')
    records = {}
    for path in paths:
        raw = read(path)
        rid, qid = raw['request_id'], raw['candidate_id']
        require(rid in requests and rid not in records, 'UNEXPECTED_OR_DUPLICATE_RESPONSE:' + rid)
        request = requests[rid]
        question = {**questions[qid], 'gold_content_ids': gold[qid]}
        require(raw['solver_model'] == manifest['solver_model'] == model, 'RESPONSE_MODEL_MISMATCH')
        require(raw['model_path'] == manifest['model'], 'RESPONSE_WEIGHT_PATH_MISMATCH')
        for key in ('candidate_id', 'root_id', 'question_type', 'mode', 'slot', 'seed'):
            require(raw[key] == request[key], 'RESPONSE_REQUEST_MISMATCH:' + key)
        require(raw['mode'] == 'full_context', 'NON_FULL_CONTEXT_RESPONSE')
        require(raw['variant_id'] == request['variant']['variant_id'], 'VARIANT_MISMATCH')
        require(raw['evaluation_valid'] is True and raw['data_errors'] == [], 'INVALID_RESPONSE')
        require(raw['metadata']['finish_reason'] in ('stop', 'eos'), 'TRUNCATED_RESPONSE')
        require(raw['metadata']['sampling_seed'] == request['seed'], 'SAMPLING_SEED_MISMATCH')
        require(raw['metadata']['prompt_token_count'] == request['prompt_tokens'], 'PROMPT_TOKEN_COUNT_MISMATCH')
        for key in ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version'):
            require(raw['metadata'][key] == manifest['decoder_contract'][key], 'DECODER_MISMATCH:' + key)
        score = scorer.score_response(question, request, raw['raw_output'])
        require(score == raw['score'], 'ORIGINAL_SCORER_REPLAY_MISMATCH:' + rid)
        require(score['scored'] is True, 'UNSCORED_RESPONSE')
        # A separate parser and direct set calculation check the shared scorer.
        match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw['raw_output'], re.S)
        require(match is not None, 'RAW_BOX_PARSE_FAILURE')
        text = match[1].strip()
        labels = [s.strip().upper() for s in text.split(',')] if text else []
        require(len(labels) == len(set(labels)) and set(labels) <= set('ABCDEF'), 'INVALID_DISPLAY_LABELS')
        prediction = {request['variant']['display_to_content'][label] for label in labels}
        gold_ids = set(gold[qid])
        require(gold_ids == set(canonical[qid]['gold_content_ids']), 'GOLD_BINDING_MISMATCH')
        supported = {canonical[qid]['content_id_by_label'][label] for label, evidence in questions[qid]['option_evidence'].items() if evidence['relation'] == 'supports'}
        require(gold_ids == supported, 'GOLD_EVIDENCE_RELATION_MISMATCH')
        tp, fp, fn = len(prediction & gold_ids), len(prediction - gold_ids), len(gold_ids - prediction)
        partial = Fraction(tp, len(gold_ids)) - (Fraction(fp, 6 - len(gold_ids)) if len(gold_ids) < 6 else 0)
        exact = prediction == gold_ids
        require(partial == Fraction(score['partial_score_fraction']) and exact is score['exact_match'], 'INDEPENDENT_SCORE_MISMATCH')
        require((tp, fp, fn) == (score['tp'], score['fp'], score['fn']), 'TP_FP_FN_MISMATCH')
        records[rid] = {'request_id': rid, 'candidate_id': qid, 'model': model,
                        'exact': exact, 'partial': str(partial), 'tp': tp, 'fp': fp, 'fn': fn,
                        'gold_count': len(gold_ids), 'selected_count': len(prediction), 'all_six': len(prediction) == 6,
                        'prediction_content_ids': sorted(prediction), 'gold_content_ids': sorted(gold_ids),
                        'raw_output': raw['raw_output'], 'response_binding': bind(path)}
    require(set(records) == set(requests), 'INCOMPLETE_CALL_SET')
    require(read(directory / 'COMPLETE.json')['actual_call_count'] == len(records), 'COMPLETE_COUNT_MISMATCH')
    original_results = read(directory / 'RESULTS.json')
    result_by_id = {q['candidate_id']: q for q in original_results['questions']}
    require(set(result_by_id) == set(questions), 'RESULT_QUESTION_SET_MISMATCH')
    for qid in questions:
        rows = [r for r in records.values() if r['candidate_id'] == qid]
        require(len(rows) == 18, 'WRONG_CALLS_PER_QUESTION')
        expected = {(f'perm_{i:02d}', j) for i in range(6) for j in range(3)}
        observed = {(requests[r['request_id']]['variant']['variant_id'], requests[r['request_id']]['slot']) for r in rows}
        require(observed == expected, 'WRONG_PERMUTATION_ROLLOUT_SET')
        require(Fraction(sum(r['exact'] for r in rows), 18) == Fraction(result_by_id[qid]['full_exact_rate']), 'RESULT_EXACT_MISMATCH')
        require(fmean([Fraction(r['partial']) for r in rows]) == Fraction(result_by_id[qid]['full_partial_mean']), 'RESULT_PARTIAL_MISMATCH')
    return records


def aggregate(pairs, side):
    records = [p[side] for p in pairs]
    ids = {p['candidate_id'] for p in pairs}
    exact = sum(r['exact'] for r in records)
    n = len(records)
    histogram = Counter(r['selected_count'] for r in records)
    return {'questions': len(ids), 'calls': n, 'exact_correct': exact,
            'exact_rate': fvalue(Fraction(exact, n)),
            'partial_mean': fvalue(fmean([Fraction(r['partial']) for r in records])),
            'prediction_count_histogram': dict(sorted(histogram.items())),
            'mean_selected_count': fvalue(Fraction(sum(r['selected_count'] for r in records), n)),
            'all_six_prediction_count': histogram[6], 'all_six_prediction_rate': fvalue(Fraction(histogram[6], n)),
            'tp': sum(r['tp'] for r in records), 'fp': sum(r['fp'] for r in records), 'fn': sum(r['fn'] for r in records),
            'positive_option_recall': fvalue(Fraction(sum(r['tp'] for r in records), sum(r['tp'] + r['fn'] for r in records))),
            'negative_option_false_positive_rate': fvalue(Fraction(sum(r['fp'] for r in records), sum(6 - r['gold_count'] for r in records)))}


def comparison(pairs):
    base, trained = aggregate(pairs, 'base'), aggregate(pairs, 'trained')
    by_question = {}
    for qid in sorted({p['candidate_id'] for p in pairs}):
        rows = [p for p in pairs if p['candidate_id'] == qid]
        exact_delta = sum(int(r['trained']['exact']) - int(r['base']['exact']) for r in rows)
        partial_delta = sum((Fraction(r['trained']['partial']) - Fraction(r['base']['partial']) for r in rows), Fraction())
        by_question[qid] = (exact_delta, partial_delta)
    def counts(index):
        values = [v[index] for v in by_question.values()]
        return {'trained_higher': sum(v > 0 for v in values), 'tied': sum(v == 0 for v in values), 'base_higher': sum(v < 0 for v in values)}
    return {'base': base, 'trained': trained,
            'trained_minus_base_exact_percentage_points': fvalue(Fraction(trained['exact_rate']['fraction']) - Fraction(base['exact_rate']['fraction']))['percent'],
            'trained_minus_base_partial_percentage_points': fvalue(Fraction(trained['partial_mean']['fraction']) - Fraction(base['partial_mean']['fraction']))['percent'],
            'question_wins_exact': counts(0), 'question_wins_partial': counts(1),
            'paired_call_exact': {'both_correct': sum(p['base']['exact'] and p['trained']['exact'] for p in pairs),
                                  'trained_only_correct': sum(not p['base']['exact'] and p['trained']['exact'] for p in pairs),
                                  'base_only_correct': sum(p['base']['exact'] and not p['trained']['exact'] for p in pairs),
                                  'both_wrong': sum(not p['base']['exact'] and not p['trained']['exact'] for p in pairs)}}


def main():
    for name in OUTPUTS:
        require(not (HERE / name).exists(), 'REPORT_ALREADY_EXISTS:' + name)
    complete = read(HERE / 'EVALUATION.COMPLETE.json')
    require(complete['status'] == 'COMPLETE' and complete['base_calls'] == 180 and complete['new_trained_calls'] == 0, 'EVALUATION_NOT_COMPLETE')
    require(complete['training_started'] is False, 'UNEXPECTED_TRAINING')
    require(not (HERE / 'EVALUATION.ERROR.json').exists(), 'EVALUATION_ERROR_REQUIRES_EXPLICIT_RECOVERY_REPORT')
    preflight = read(HERE / 'PAIRED_PREFLIGHT.json')
    require(preflight['status'] == 'PASS', 'PAIRED_PREFLIGHT_FAILED')
    require(preflight['script_sha256'] == sha(HERE / 'run_paired_base.py'), 'RUNNER_CHANGED_AFTER_PREFLIGHT')
    require(read(HERE / 'EVALUATION.STARTED.json')['paired_preflight_sha256'] == sha(HERE / 'PAIRED_PREFLIGHT.json'), 'PREFLIGHT_CHANGED_AFTER_START')
    original_verification, old_files = audit_old_bindings()
    require(preflight['original_bindings'] == complete['original_bindings'], 'OLD_BINDING_RECEIPT_CHANGED_DURING_BASE')
    require(preflight['original_bindings']['binding_sha256'] == original_verification['binding']['sha256'], 'OLD_BINDING_FILE_CHANGED_AFTER_PREFLIGHT')
    require(preflight['original_bindings']['verified_files'] == original_verification['verified_files'], 'OLD_BINDING_COUNT_CHANGED')
    identity = read(HERE / 'MODEL_IDENTITY_AUDIT.json')
    require(identity['status'] == 'PASS', 'MODEL_IDENTITY_FAILED')
    require(read(HERE / 'EVALUATION.STARTED.json')['identity_audit_sha256'] == sha(HERE / 'MODEL_IDENTITY_AUDIT.json'), 'IDENTITY_AUDIT_CHANGED_AFTER_START')
    for row in identity['provenance_bindings']:
        require(sha(row['path']) == row['sha256'], 'IDENTITY_PROVENANCE_CHANGED:' + row['path'])
    content = read(OLD_REPORT / 'INDEPENDENT_CONTENT_AUDIT.json')
    require(content['solver_scores_used'] is False, 'CONTENT_AUDIT_NOT_SOLVER_BLIND')
    statuses = {q['candidate_id']: q['core_status'] for q in content['questions']}
    fixed_usable = {q['candidate_id'] for q in content['questions'] if q['core_status'] == 'CORE_USABLE'}
    exported_usable = {q['candidate_id'] for q in read(OLD_REPORT / 'USABLE_SELECTED_CANDIDATES.json')['questions']}
    require(fixed_usable == exported_usable and len(fixed_usable) == 7 and len(statuses) == 10, 'FROZEN_CONTENT_SET_MISMATCH')
    scorer = load_scorer()
    paired, question_info, chunk_audits = [], {}, []
    all_trained_selected = True
    for chunk in range(1, 7):
        short = f'c{chunk:02d}'
        old = OLD_RUNS / f'root194259_diversity12_01_{short}_evaluation/solver'
        new = HERE / 'runs' / short / 'solver'
        old_manifest, new_manifest = read(old / 'manifest.json'), read(new / 'manifest.json')
        keep = lambda obj: {k: v for k, v in obj.items() if k not in MANIFEST_ALLOWED_DIFFERENCES}
        require(keep(old_manifest) == keep(new_manifest), 'PAIRED_MANIFEST_MISMATCH:' + short)
        require(old_manifest['model'] == TRAINED_PATH and new_manifest['model'] == BASE_PATH, 'MODEL_PATH_MISMATCH')
        require(old_manifest['tokenizer'] == new_manifest['tokenizer'] == BASE_PATH, 'TOKENIZER_MISMATCH')
        for path, expected in old_manifest['code_sha256'].items():
            require(sha(path) == expected, 'SCORER_OR_HARNESS_CHANGED:' + path)
        for relative, expected in old_manifest['input_sha256'].items():
            if relative == 'pool':
                require(sha(old_manifest['pool_path']) == expected, 'SOURCE_POOL_CHANGED')
            else:
                require(sha(old / relative) == sha(new / relative) == expected, 'PAIRED_INPUT_MISMATCH:' + relative)
        for request in old_manifest['requests']:
            relative = Path('prompts') / (request['request_id'] + '.txt')
            require(sha(old / relative) == sha(new / relative) == request['prompt_sha256'], 'PAIRED_PROMPT_MISMATCH')
        questions = {q['candidate_id']: q for q in read(old / 'questions.json')['candidates']}
        gold = read(old / 'gold.json')
        canonical = {q['candidate_id']: q for q in read(old / 'canonical_bindings.json')['candidates']}
        trained = replay(old, old_manifest, questions, gold, canonical, scorer, 'M_clean_policy')
        base = replay(new, new_manifest, questions, gold, canonical, scorer, 'M0')
        require(set(trained) == set(base), 'PAIRED_RESPONSE_SET_MISMATCH')
        trained_results = read(old / 'RESULTS.json')
        all_trained_selected &= all(q['selected_by_policy'] for q in trained_results['questions'])
        for request in old_manifest['requests']:
            rid, qid = request['request_id'], request['candidate_id']
            paired.append({'request_id': rid, 'candidate_id': qid, 'root_id': request['root_id'],
                           'chunk': short, 'content_status': statuses[qid], 'included_in_primary': qid in fixed_usable,
                           'variant_id': request['variant']['variant_id'], 'slot': request['slot'], 'seed': request['seed'],
                           'prompt_sha256': request['prompt_sha256'], 'base': base[rid], 'trained': trained[rid]})
        for qid, question in questions.items():
            require(qid not in question_info, 'REPEATED_CANDIDATE_ACROSS_CHUNKS')
            question_info[qid] = {'candidate_id': qid, 'short_id': short + 'MC' + qid[-2:], 'topic': question['topic'],
                                  'core_status': statuses[qid], 'primary': qid in fixed_usable, 'gold_count': len(gold[qid])}
        chunk_audits.append({'chunk': short, 'questions': len(questions), 'paired_calls': len(base),
                             'base_manifest': bind(new / 'manifest.json'), 'trained_manifest': bind(old / 'manifest.json'),
                             'prompts_inputs_gold_sources_decoder_requests_seeds_permutations_identical': True,
                             'both_scorer_replay_and_independent_raw_replay_passed': True})
    require(len(paired) == 180 and len(question_info) == 10, 'WRONG_TOTAL_BUDGET')
    require(set(question_info) == set(statuses), 'READY_CONTENT_SET_CHANGED')
    require(all_trained_selected, 'V57_MACHINE_SELECTION_FACT_CHANGED')
    primary_pairs = [p for p in paired if p['included_in_primary']]
    require(len(primary_pairs) == 126, 'PRIMARY_RESPONSE_COUNT_WRONG')
    primary, all_ready = comparison(primary_pairs), comparison(paired)
    issue = comparison([p for p in paired if not p['included_in_primary']])
    per_question = []
    for qid, info in question_info.items():
        per_question.append({**info, **comparison([p for p in paired if p['candidate_id'] == qid])})
    per_question.sort(key=lambda x: x['short_id'])
    # Frozen pre-base manual baseline provides a second aggregate cross-check.
    require(primary['trained']['exact_correct'] == 77 and primary['trained']['calls'] == 126, 'TRAINED_BASELINE_EXACT_CHANGED')
    require(Fraction(primary['trained']['partial_mean']['fraction']) == Fraction(463, 504), 'TRAINED_BASELINE_PARTIAL_CHANGED')
    require(all_ready['trained']['exact_correct'] == 107 and all_ready['trained']['calls'] == 180, 'TRAINED_ALL_BASELINE_CHANGED')
    now = datetime.now(timezone.utc).isoformat()
    all_six = {'questions': 7, 'calls': 126, 'exact_rate': fvalue(0), 'partial_mean': fvalue(0),
               'reason': 'Every usable question has four or five correct options; selecting all six makes TP/G = 1 and FP/(6-G) = 1.'}
    result = {'version': 'v57_base_vs_historical_clean_policy_paired_comparison_v1', 'created_at_utc': now,
              'root_id': 'cc-train-episode-194259', 'models': {'base': {k:v for k,v in identity['base'].items() if k != 'files'},
                                                           'trained': {k:v for k,v in identity['trained'].items() if k != 'files'}},
              'scope': 'Latest v57 single-root comparison; not an evaluation of the 32-question merged collection.',
              'primary_selection': 'Seven pre-existing Solver-blind CORE_USABLE questions; no reselection by either current model score.',
              'primary': primary, 'all_ready_diagnostic': all_ready, 'known_content_issue_diagnostic': issue,
              'per_question': per_question, 'all_six_prediction_baseline_primary': all_six,
              'scoring': {'primary': 'exact match of selected content set and Gold content set',
                          'secondary_formula': 'TP/G - FP/(6-G), or TP/G when G=6',
                          'partial_is_not_option_accuracy': True, 'scorer': bind(SCORER)},
              'protocol': {'permutations_per_question': 6, 'rollouts_per_permutation': 3, 'calls_per_question': 18,
                           'decoder_contract': old_manifest['decoder_contract'], 'full_context': True,
                           'inputs_prompt_bytes_gold_sources_seeds_permutations_identical': True,
                           'new_base_calls': 180, 'reused_trained_calls': 180, 'new_trained_calls': 0,
                           'new_training_calls': 0, 'training_pool_admission': False, 'new_generation_calls': 0},
              'limitations': ['Seven distinct usable questions from one root; 126 responses are correlated repeated displays and samples, not 126 independent questions.',
                              'Generation uses historical trained-model feedback and a project root population; this is not an independent benchmark.',
                              'Identity audit found zero root194259 rows in the audited 386-row checkpoint training source; no claim of this root being used in that training.',
                              'The trained checkpoint is historical round0/B0 clean-policy step193, not a second-round-trained model.']}
    verification = {'version': 'v57_paired_final_verification_v1', 'status': 'PASS', 'created_at_utc': now,
                    'old_v57_files_unchanged': original_verification, 'chunks': chunk_audits,
                    'base_responses_independently_replayed': 180, 'trained_responses_independently_replayed': 180,
                    'frozen_content_primary_ids': sorted(fixed_usable), 'content_selection_unchanged': True,
                    'all_ten_trained_machine_selected': True, 'no_new_score_based_question_selection': True,
                    'identity_audit': bind(HERE / 'MODEL_IDENTITY_AUDIT.json'),
                    'content_audit': bind(OLD_REPORT / 'INDEPENDENT_CONTENT_AUDIT.json'),
                    'completed_runner': bind(HERE / 'EVALUATION.COMPLETE.json'),
                    'scorer': bind(SCORER), 'exporter': bind(__file__)}
    rows = ['# v57 Base 与训练后模型逐题结果', '', '每题 6 种随机展示 × 3 次作答，共 18 次。完全正确要求选中集合与 Gold 完全一致；部分分数按冻结惩罚公式计算。', '',
            '| 题目 | 内容状态 | 正确项数 | Base 完全正确 | 训练后完全正确 | 正确率差（百分点） | Base 部分得分 | 训练后部分得分 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for q in per_question:
        status = '主比较可用' if q['primary'] else '内容问题，仅诊断'
        rows.append(f"| {q['short_id']} | {status} | {q['gold_count']} | {q['base']['exact_correct']}/18 ({q['base']['exact_rate']['percent']:.2f}%) | {q['trained']['exact_correct']}/18 ({q['trained']['exact_rate']['percent']:.2f}%) | {q['trained_minus_base_exact_percentage_points']:+.2f} | {q['base']['partial_mean']['value']:.6f} | {q['trained']['partial_mean']['value']:.6f} |")
    rows += ['', '内容问题的三题在 Base 运行前已固定排除，原因是把未被明确反证的选项判假；题目、Gold 均未修改。', '',
             'c04MC01、c06MC01：提议过东侧 Pizza Hut，不能单独反证另有一次西侧法国餐馆建议。c05MC02：用演讲排练场景反证不相干的餐馆预订。', '']
    delta = primary['trained_minus_base_exact_percentage_points']
    verb = '高于' if delta > 0 else '低于' if delta < 0 else '等于'
    b, t = primary['base'], primary['trained']
    wins = primary['question_wins_exact']
    report = ['# v57 Base 与训练后模型比较', '',
              f"在最新 root **194259** 的 **7 道内容可用题**上，训练后模型正确率 **{t['exact_rate']['percent']:.2f}%**，Base 为 **{b['exact_rate']['percent']:.2f}%**；训练后{verb} Base **{abs(delta):.2f} 个百分点**。", '',
              '| 固定口径 | 模型 | 完全正确 / 作答次数 | 完全正确率 | 部分得分均值 |',
              '|---|---|---:|---:|---:|',
              f"| 7 道内容可用题（主比较） | Base | {b['exact_correct']}/126 | {b['exact_rate']['percent']:.2f}% | {b['partial_mean']['value']:.6f} |",
              f"| 7 道内容可用题（主比较） | 训练后 | {t['exact_correct']}/126 | {t['exact_rate']['percent']:.2f}% | {t['partial_mean']['value']:.6f} |",
              f"| 全部 10 道 ready（含 3 道内容问题，仅诊断） | Base | {all_ready['base']['exact_correct']}/180 | {all_ready['base']['exact_rate']['percent']:.2f}% | {all_ready['base']['partial_mean']['value']:.6f} |",
              f"| 全部 10 道 ready（含 3 道内容问题，仅诊断） | 训练后 | {all_ready['trained']['exact_correct']}/180 | {all_ready['trained']['exact_rate']['percent']:.2f}% | {all_ready['trained']['partial_mean']['value']:.6f} |", '',
              f"主比较部分得分变化为 **{primary['trained_minus_base_partial_percentage_points']:+.2f} 个百分点**。按每题的完全正确率，训练后更好 **{wins['trained_higher']} 题**，相同 **{wins['tied']} 题**，Base 更好 **{wins['base_higher']} 题**。逐题数值见 [PER_QUESTION.md](PER_QUESTION.md)。", '',
              'Base 是 `Qwen/Qwen3-4B`（`M0`，未启用 adapter）；训练后是既有 `M_clean_policy`，历史 round0/B0 clean-policy **step193、386 条记录、2 epochs** 的合并权重，沿用 v57 原 Solver。本次未训练第二轮，也没有重跑训练后模型：新跑 Base 180 次，复用训练后原 180 次。模型文件身份审计见 [MODEL_IDENTITY_AUDIT.md](MODEL_IDENTITY_AUDIT.md)。', '',
              '两模型使用相同完整对话、题干、Gold、6 种显示排列、每排列 3 次采样、原随机种子和解码参数（temperature 0.8、top_p 0.9、max_tokens 32、thinking 关闭及相同结构化输出）。逐份 prompt 与来源字节一致；两边全部 360 份原始输出按原评分器及独立集合计算重放。7 道主比较题由此前 Solver 盲审内容结论固定，没有依据 Base 分数再次筛选。原 v57 的 10 道恰好全部机器入选，因此此次 7 道只去掉了内容有问题的 3 道。', '',
              '完全正确率统计整道多选题是否全对。部分分数是 `TP/G − FP/(6−G)`（G=6 时为 `TP/G`），包含错选惩罚，不能当作逐选项正确率。本次 7 道可用题为选四 3 道、选五 4 道；一律输出 ABCDEF 的完全正确率和部分分数均为 **0**。', '',
              '| 预测行为诊断（7 道可用题） | Base | 训练后 |', '|---|---:|---:|',
              f"| 每次平均选择项数 | {b['mean_selected_count']['value']:.3f} | {t['mean_selected_count']['value']:.3f} |",
              f"| ABCDEF 全选次数 / 比例 | {b['all_six_prediction_count']}/126 ({b['all_six_prediction_rate']['percent']:.2f}%) | {t['all_six_prediction_count']}/126 ({t['all_six_prediction_rate']['percent']:.2f}%) |",
              f"| 应选项召回率 | {b['positive_option_recall']['percent']:.2f}% | {t['positive_option_recall']['percent']:.2f}% |",
              f"| 不应选项的错选率 | {b['negative_option_false_positive_rate']['percent']:.2f}% | {t['negative_option_false_positive_rate']['percent']:.2f}% |", '',
              '这些行为统计帮助观察漏选和错选，不能单独证明模型收益来自理解提升或选择数量倾向。', '',
              '本次仅覆盖 **1 个 root、7 道不同可用题**；126 次是重复排列与采样，并非 126 道独立题，不能直接外推到全体训练池。题目生成使用既有训练后模型的历史反馈，因此也不是独立留出 benchmark。身份审计在该 checkpoint 的 386 条训练来源中查得 root194259 **0 条**；不能说该 root 已进入这次 checkpoint 训练。累计 **32 道题合集未在本次全部评估**。', '',
              f"旧 v57 绑定的 **{original_verification['verified_files']} 份文件**重新验哈希通过。检查明细见 [VERIFICATION.json](VERIFICATION.json)，机器可读结果见 [COMPARISON.json](COMPARISON.json)，逐次对应见 [PAIRED_CALLS.json](PAIRED_CALLS.json)。", '']
    # All validations finish before the first write; outputs are write-once.
    write(HERE / 'COMPARISON.json', result)
    write(HERE / 'PAIRED_CALLS.json', {'version': 'v57_base_trained_paired_raw_calls_v1', 'pairs': paired})
    write(HERE / 'VERIFICATION.json', verification)
    write(HERE / 'PER_QUESTION.md', '\n'.join(rows))
    write(HERE / 'REPORT.md', '\n'.join(report))
    # Old provenance, model audit sources, frozen scorer, runner artifacts and all
    # final reports are covered. Only the binding file itself and bytecode are excluded.
    file_paths = {Path(r['path']) for r in old_files}
    file_paths.add(OLD_REPORT / 'RESULT_BINDINGS.json')
    file_paths.update(Path(r['path']) for r in identity['provenance_bindings'])
    file_paths.add(SCORER)
    file_paths.update(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'RESULT_BINDINGS.json')
    bindings = [bind(p) for p in sorted(file_paths)]
    write(HERE / 'RESULT_BINDINGS.json', {'scope': 'complete_v57_base_trained_paired_report_inputs_originals_runner_and_audits',
                                        'created_at_utc': now, 'original_v57_file_count': original_verification['verified_files'],
                                        'file_count': len(bindings), 'files': bindings})
    print(json.dumps({'status': 'COMPLETE', 'report': str(HERE / 'REPORT.md'), 'primary': primary,
                      'all_ready_diagnostic': all_ready, 'bound_files': len(bindings)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
