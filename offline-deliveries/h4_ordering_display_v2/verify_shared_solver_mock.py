"""Offline prepared-ordering plumbing check. Uses a fake Gold-returning backend.

This script never loads model weights, creates a vLLM engine, or sends a request.
Reproduce with a new --output-dir; an existing mock directory is never resumed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORKTREE = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-ordering-display-v2')
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.contract import RELABEL_POLICY, permutation_version
from scripts.current_method.restart_n5_spell.h4_ordering_display_v2.dated_prompt import dated_prompt_builder
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1 import contract, gate, solver
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.io_utils import stable_seed, write_once_json
from scripts.current_method.restart_n5_spell.h4_round0_b0_k3_difficulty_v1.request_ledger import Round0RequestLedger


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class FakeGoldBackend:
    """Return supplied displayed Gold, solely to test parsing/mapping/ledgers."""
    model_identifier = 'FAKE_GOLD_BACKEND_NO_MODEL'
    tokenizer_path = TOKENIZER
    max_model_len = 16384

    def __init__(self):
        self.calls = []

    def generate(self, request):
        raise AssertionError('ordering_only_no_unstructured_backend_calls')

    def generate_structured(self, request, *, question_type):
        require(question_type == 'ordering', 'unexpected_question_type')
        self.calls.append({
            'request_id': request.request_id,
            'candidate_id': request.qa_id,
            'context_mode': request.evaluation_mode,
            'variant_id': request.order_id,
            'slot_id': request.target_slot,
            'seed': request.sampling_seed,
            'prompt_sha256': request.prompt_sha256,
        })
        return BackendResponse(
            '<box>' + request.canonical_answer_text + '</box>',
            self.model_identifier,
            {'mock': True, 'finish_reason': 'stop', 'sampling_seed': request.sampling_seed,
             'model_inference_performed': False, 'usage': 'NOT_APPLICABLE_FAKE_BACKEND'},
        )


def execute(prepared: Path, output: Path):
    require(not output.exists(), 'mock_output_exists_use_a_new_output_dir')
    sources = {}

    def read(path):
        path = Path(path)
        sources[str(path.resolve())] = sha(path)
        return json.loads(path.read_text())

    candidates = read(prepared / 'candidate_manifest.json')
    saved_gates = read(prepared / 'program_gate_results.json')
    prompt_rows = read(prepared / 'prompt_manifest.json')
    preparation = read(prepared / 'OFFLINE_VERIFICATION.json')
    require(len(candidates) == 30, 'unexpected_prepared_candidate_count')
    selected = [row for row in candidates if row['question_type'] == 'ordering']
    require(len(selected) == 5, 'unexpected_ordering_count')
    require(len({row['candidate_id'] for row in candidates}) == 30, 'candidate_identity_collision')
    gates = {row['candidate_id']: row for row in saved_gates}
    dialogues = {root: read(prepared / 'dialogues' / (root + '.json'))
                 for root in sorted({q['root_id'] for q in selected})}
    prompt_index = {(r['candidate_id'], r['variant_id'], r['context_mode']): r for r in prompt_rows}
    require(len(prompt_rows) == len(prompt_index) == 40, 'expected_40_distinct_ordering_prompts')
    for row in prompt_rows:
        path = Path(row['path'])
        require(sha(path) == row['sha256'], 'frozen_prompt_hash_drift')
        sources[str(path.resolve())] = row['sha256']
    require(preparation['local_tokenizer'] == str(TOKENIZER), 'prepared_tokenizer_identity_mismatch')
    require(preparation['ordering_display_policy'] == RELABEL_POLICY, 'prepared_policy_mismatch')

    for relative in (
        'h4_ordering_display_v2/contract.py', 'h4_ordering_display_v2/dated_prompt.py',
        'h4_round0_b0_k3_difficulty_v1/contract.py', 'h4_round0_b0_k3_difficulty_v1/gate.py',
        'h4_round0_b0_k3_difficulty_v1/solver.py', 'h4_round0_b0_k3_difficulty_v1/request_ledger.py',
    ):
        path = WORKTREE / 'scripts/current_method/restart_n5_spell' / relative
        sources[str(path)] = sha(path)
    sources[str(Path(__file__).resolve())] = sha(__file__)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    renderer = dated_prompt_builder(dialogues)
    mock_contract = {
        'version': 'ordering_display_v2_shared_solver_mock_v1', 'mock': True,
        'backend': FakeGoldBackend.model_identifier,
        'ordering_display_policy': RELABEL_POLICY,
        'permutation_contract_version': permutation_version(RELABEL_POLICY),
        'prepared_candidate_manifest_sha256': sources[str((prepared / 'candidate_manifest.json').resolve())],
        'source_files': sources,
        'model_weights_loaded': False, 'real_solver_accuracy': 'NOT_EVALUATED',
    }
    write_once_json(output / 'MOCK_ONLY.json', mock_contract)
    contract_sha = sha(output / 'MOCK_ONLY.json')
    backend = FakeGoldBackend()
    ledger = Round0RequestLedger(output, solver_contract_sha256=contract_sha)
    solver.validate_gate_display_policy(mock_contract, saved_gates)
    results, checks = [], []
    with patch.object(gate, 'build_prompt', renderer), patch.object(solver, 'build_prompt', renderer):
        for candidate in selected:
            cid, root = candidate['candidate_id'], candidate['root_id']
            stored = gates[cid]
            regenerated = gate.gate_candidate(
                candidate, dialogues[root], tokenizer, root_id=root,
                ordering_display_policy=RELABEL_POLICY,
            )
            require(regenerated == stored, 'regenerated_gate_not_identical:' + cid)
            variants = {v['variant_id']: v for v in contract.position_variants(
                candidate, ordering_display_policy=RELABEL_POLICY)}
            for variant in variants.values():
                for mode in ('no_context', 'full_context'):
                    saved = prompt_index[(cid, variant['variant_id'], mode)]
                    rendered = renderer(tokenizer, candidate, variant,
                                        contract.dialogue_rows(dialogues[root]), context_mode=mode)
                    require(rendered == Path(saved['path']).read_text(), 'saved_prompt_text_mismatch')
            scoped_gate = {**stored,
                'candidate_pool_sha256': sources[str((prepared / 'candidate_manifest.json').resolve())],
                'dialogue_sha256': sources[str((prepared / 'dialogues' / (root + '.json')).resolve())]}
            kwargs = dict(candidate=candidate, expanded=dialogues[root], gate_row=scoped_gate,
                          tokenizer=tokenizer, backend=backend, ledger=ledger,
                          solver_contract_sha256=contract_sha)
            result = solver.evaluate_candidate(**kwargs)
            require(result['request_count'] == 24, 'unexpected_per_candidate_mock_count')
            require(result['technical_failure_count'] == 0, 'unexpected_mock_technical_failure')
            require(result['ordering_display_policy'] == RELABEL_POLICY, 'result_policy_drift')
            for request in result['requests']:
                v = variants[request['variant_id']]
                saved = prompt_index[(cid, request['variant_id'], request['context_mode'])]
                require(request['correct'] is True, 'gold_fake_failed_scoring')
                require(request['canonical_prediction'] == candidate['gold'], 'canonical_rebinding_failed')
                require(request['display_prediction'] == v['displayed_gold'], 'display_prediction_mismatch')
                require(request['display_to_canonical'] == v['display_to_canonical'], 'mapping_drift')
                require(request['prompt_sha256'] == saved['sha256'], 'gate_execution_prompt_sha_drift')
                require(request['prompt_token_count'] == saved['input_tokens'], 'prompt_token_count_drift')
                require(request['seed'] == stable_seed(root, cid, request['slot_id']), 'paired_seed_drift')
                require(request['backend_metadata']['mock'] is True, 'not_explicitly_fake')
            before_replay = len(backend.calls)
            require(solver.evaluate_candidate(**kwargs) == result, 'cached_result_changed')
            require(len(backend.calls) == before_replay, 'cache_replay_called_fake_backend')
            write_once_json(solver.candidate_result_path(output, cid), result)
            results.append(result)
            checks.append({'candidate_id': cid, 'source_key': candidate['display_identity'],
                           'regenerated_gate_identical': True, 'saved_prompt_checks': 8,
                           'mock_requests': 24, 'canonical_roundtrips': 24,
                           'paired_seed_checks': 24, 'cached_replay_additional_backend_calls': 0,
                           'displayed_gold_sequences': [v['displayed_gold'] for v in variants.values()]})
    require(len(backend.calls) == 120, 'total_mock_count_not_120')
    summary = ledger.summary()
    require(summary['request_count'] == summary['completed_request_count'] == 120, 'ledger_count_mismatch')
    require(summary['failed_request_count'] == summary['ambiguous_request_count'] == 0, 'ledger_failure')
    for state in ledger.states():
        require(state['request_contract']['backend_identity']['model'] == backend.model_identifier,
                'ledger_backend_not_fake')
        require(state['request_contract']['ordering_display_policy'] == RELABEL_POLICY,
                'ledger_policy_drift')
        require(len(state['attempts']) == 1, 'unexpected_mock_retry')
    require(all(Path(p).is_file() and sha(p) == digest for p, digest in sources.items()), 'source_modified')
    write_once_json(output / 'candidate_results.json', results)
    report = {
        'version': 'ordering_display_v2_shared_solver_mock_v1', 'status': 'PASS',
        'label': 'FAKE_GOLD_PLUMBING_ONLY_NOT_MODEL_ACCURACY',
        'prepared_root': str(prepared), 'mock_output': str(output),
        'candidate_count': 5, 'mock_backend_calls': 120, 'mock_ledger_requests': 120,
        'gate_regenerations_identical': 5, 'distinct_saved_prompts_matched': 40,
        'canonical_roundtrips_passed': 120, 'paired_seed_checks_passed': 120,
        'cached_replay_additional_backend_calls': 0, 'retries': 0,
        'model_inference_calls': 0, 'model_weights_loaded': False, 'http_calls': 0,
        'training_calls': 0, 'production_or_frozen_source_changes': 0,
        'actual_solver_accuracy': 'NOT_EVALUATED', 'verified_new_hard_count': 'UNKNOWN_NOT_EVALUATED',
        'actual_model_tokens_and_cost': 'NOT_APPLICABLE_FAKE_BACKEND',
        'other_25_choices': 'NOT_REEVALUATED_THIS_CHECK; existing preparation byte-identity check retained',
        'ordering_display_policy': RELABEL_POLICY,
        'tokenizer': str(TOKENIZER), 'tokenizer_load_mode': 'local_files_only',
        'ledger': summary, 'checks': checks, 'source_sha256': sources,
        'reproduction_command': 'python ' + str(Path(__file__).resolve()) + ' --output-dir NEW_EMPTY_MOCK_DIRECTORY',
    }
    write_once_json(output / 'MOCK_VERIFICATION.json', report)
    markdown = '''# 共享 Solver 模拟贯通检查

**PASS：5道准备好的排序题，经共享 `evaluate_candidate` 完成120次假后端调用。**

这只验证程序接线。假后端直接返回请求中的展示Gold，不能解释为模型准确率、排序语义认证或难度提升。

- 使用本地 Qwen3-4B tokenizer，`local_files_only=True`；未加载模型权重。
- 5份 gate 重新生成后与准备文件完全一致；40份有原文/无原文提示逐字及 SHA 一致。
- 120条回答正确映射回 canonical Gold；展示映射、提示token计数和配对 seed 全部一致。
- 原 `Round0RequestLedger` 留下120条明确标注 FAKE 的完成记录；缓存重放新增后端调用0，重试0。
- 候选、原文、准备提示及读取的代码 SHA 前后不变。其余25道选择题未在本检查中重新评估。
- 真实模型推理、HTTP、训练调用均为 **0**；真实Solver准确率和新增难题量仍未评估。

结果与假响应位于 `''' + str(output) + '''`。复现使用新空目录：

```bash
python ''' + str(Path(__file__).resolve()) + ''' --output-dir NEW_EMPTY_MOCK_DIRECTORY
```

脚本禁止连接网络，并拒绝覆盖已有mock目录。完整源文件SHA、逐题核对、ledger统计见 `MOCK_VERIFICATION.json`。
'''
    (output / 'MOCK_VERIFICATION.md').write_text(markdown)
    return report, markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=HERE / 'prepared')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'mock_solver')
    args = parser.parse_args()

    def no_network(*_args, **_kwargs):
        raise AssertionError('NETWORK_DISABLED_OFFLINE_MOCK')

    with patch.object(socket.socket, 'connect', no_network), \
            patch.object(socket, 'create_connection', no_network):
        report, markdown = execute(args.prepared.resolve(), args.output_dir.resolve())
    if args.output_dir.resolve() == (HERE / 'mock_solver').resolve():
        write_once_json(HERE / 'MOCK_VERIFICATION.json', report)
        (HERE / 'MOCK_VERIFICATION.md').write_text(markdown)
    print(json.dumps({key: report[key] for key in
                      ('status', 'candidate_count', 'mock_backend_calls', 'http_calls',
                       'model_inference_calls', 'actual_solver_accuracy')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
