"""Report actual training telemetry; never interpret teacher-forced loss as accuracy."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN = REPO / 'restart-n5-runs/h4_codex_combined_sft_v125'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    status = read(HERE / 'RUN_STATUS.json')
    if status['status'] != 'TRAINING_COMPLETE_AWAITING_EVALUATION':
        raise RuntimeError('TRAINING_NOT_COMPLETE')
    result = read(RUN / 'TRAINING_RESULT.json')
    smoke = read(HERE / 'smoke/RESULT.json')
    audit = read(RUN / 'gradient_audit.json')
    events = [json.loads(x) for x in (RUN / 'telemetry.jsonl').read_text().splitlines()]
    back = [x for x in events if x['event'] == 'backward_complete']
    clips = [x for x in events if x['event'] == 'gradient_clip_complete']
    assert len(back) == len(audit['backward_records']) == 726
    assert len(clips) == len(audit['gradient_records']) == 182
    for i, (left, right) in enumerate(zip(back, audit['backward_records']), 1):
        assert left['micro_step'] == right['micro_step'] == i
        assert math.isclose(left['unscaled_loss'], right['scaled_loss'] * 4, rel_tol=1e-7, abs_tol=1e-9)
    for left, right in zip(clips, audit['gradient_records']):
        assert math.isclose(left['gradient_norm_before_clip'], right['clipped_norm'], rel_tol=1e-7, abs_tol=1e-9)
    losses = [e['unscaled_loss'] for e in back]
    for field, computed in {
        'mean_training_loss': statistics.mean(losses),
        'first_25_microbatch_mean_loss': statistics.mean(losses[:25]),
        'last_25_microbatch_mean_loss': statistics.mean(losses[-25:]),
    }.items():
        assert math.isclose(result[field], computed, rel_tol=1e-10, abs_tol=1e-12), field
    assert result['peak_allocated_bytes'] == max(e['peak_allocated_bytes'] for e in back)
    assert result['peak_reserved_bytes'] == max(e['peak_reserved_bytes'] for e in back)
    for field, path in {
        'adapter_config_sha256': RUN / 'adapter/adapter_config.json',
        'adapter_state_sha256': RUN / 'adapter/adapter_model.pt',
        'gradient_audit_sha256': RUN / 'gradient_audit.json',
        'telemetry_sha256': RUN / 'telemetry.jsonl',
    }.items():
        assert result[field] == sha(path), field
    first_epoch = [e['unscaled_loss'] for e in back[:363]]
    second_epoch = [e['unscaled_loss'] for e in back[363:]]
    timing = [e['since_previous_backward_seconds'] for e in back[1:]]
    summary = {
        'status': 'TRAINED_NOT_YET_SOLVER_EVALUATED', 'records': 363, 'roots': 68,
        'multi_select': 238, 'ordering': 125, 'replay': 0, 'epochs': 2,
        'microsteps': 726, 'optimizer_steps': 182, 'max_length': 12288,
        'smoke_elapsed_seconds': smoke['elapsed_seconds'], 'formal_elapsed_seconds': result['elapsed_seconds'],
        'formal_mean_microstep_seconds_excluding_initialization': statistics.mean(timing),
        'formal_peak_allocated_gib': result['peak_allocated_bytes'] / 2**30,
        'formal_peak_reserved_gib': result['peak_reserved_bytes'] / 2**30,
        'mean_teacher_forced_training_loss': result['mean_training_loss'],
        'epoch1_mean_training_loss': statistics.mean(first_epoch),
        'epoch2_mean_training_loss': statistics.mean(second_epoch),
        'first25_microbatch_mean_training_loss': result['first_25_microbatch_mean_loss'],
        'last25_microbatch_mean_training_loss': result['last_25_microbatch_mean_loss'],
        'solver_accuracy': None, 'dev_evaluation_started': False, 'test_evaluation_started': False,
        'adapter_state_sha256': result['adapter_state_sha256'],
        'notes': [
            'Training loss uses teacher-forced answer tokens; it is not generated-answer accuracy.',
            'Microbatch mean losses are unweighted by target length; they describe the native batch-size-1 training stream.',
            'Epoch boundary does not flush accumulated gradients in the inherited trainer.',
            'The final two microbatches retain native divisor4; no training algorithm change was made.',
            'Native first-step printed loss has a biased denominator; use raw-loss telemetry.',
            'Last clip telemetry mean spans four trailing microbatches, not only the two in the final update; overall and epoch loss values use raw per-backward telemetry.',
            'The new-only continuation still inherits historical M_clean learning and any label noise in it.',
            'v125 remains an immutable preparation-time snapshot. v126 records actual execution status.',
        ],
    }
    (HERE / 'TRAINING_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    text = f'''本轮正式续训已完成；尚未运行 Solver 验证，不能据训练 loss 判断正确率是否提高。

封存数据为 363 道新题 / 68 个 train root，238 多选 + 125 排序，回放 0。完整输入最长 8533 tokens，上限 12288，零截断。先用各题型最长的两道题做了 4 次反传、1 次更新的试训，耗时 {smoke['elapsed_seconds']:.1f} 秒；正式训练另从原始 M_clean adapter 加载，未使用试训后的权重。

正式训练执行 2 轮、726 次反传、182 次更新，耗时 {result['elapsed_seconds'] / 60:.2f} 分钟。全部损失有限，梯度有限且非零。显存峰值实际分配 {summary['formal_peak_allocated_gib']:.2f} GiB，分配器保留峰值 {summary['formal_peak_reserved_gib']:.2f} GiB。

第一轮平均训练 loss 为 {summary['epoch1_mean_training_loss']:.6f}，第二轮为 {summary['epoch2_mean_training_loss']:.6f}。这是已知答案 token 的拟合损失，不是多选/排序正确率。完整原始遥测、逐步梯度审计和独立训练审核均保留，v125 封存输入保持不变。

实际模型位于 `restart-n5-runs/h4_codex_combined_sft_v125/adapter/`，是项目原生 LoRA（`adapter_model.pt`、`adapter_config.json`），须配合原始 Qwen3-4B 使用。它已包含续训后的 M_clean 参数，合并时只能叠加到原始 base 一次，不能再叠加到旧 M_clean merged 模型。

沿用原训练器行为：跨 epoch 累积梯度，末次两个 microbatch 仍除以 4；原日志首步 loss 有分母偏差，本报告使用每次反传的原始 loss。末次 clip 遥测的末四条均值不是末两条 update 均值，此字段不用于本报告统计。以上细节未改变原生训练算法。

下一步在 v124 dev 的 48 道题上，对旧 M_clean 和本轮模型做相同提示、展示与采样的配对评估，先分别报告 Gold2345 多选、排序和 Gold6 控制；test 保留用于最终评估。固定 15 个 dev root 中仅 7 个产出这 48 题（34 多选、14 排序），另 8 个仍计入覆盖分母。配对得分只反映这 7 个有题 root，不能外推全部 15 个，也不能把 1560 次调用当作独立样本。27 道 Gold2345 为多选主指标，7 道 Gold6 另列。

本目录记录实际执行结果；封存的 v125 中“尚未训练”是当时的数据交付状态，不作回写。
'''
    (HERE / 'REPORT_ZH.md').write_text(text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
