#!/usr/bin/env python3
"""Exercise real frozen DEV plumbing with a CPU test double, never a real model."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys

sys.dont_write_bytecode = True
from validation_callback import aggregate, bind, evaluate, load, score_output, verify_manifest, write_json

FINAL = Path(__file__).resolve().parents[1]


def main():
    import torch
    import transformers
    manifest, records, golds = verify_manifest(FINAL / 'VALIDATION_MANIFEST.json')
    tokenizer = transformers.AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    scorer = load('selfcheck_official_scorer', manifest['official_scorer']['path'])
    def score(text, index=0, finish='eos'):
        return score_output(text, finish_reason=finish, record=records[index], gold=golds[index], scorer=scorer, adapter=None)
    idx = next(i for i, row in enumerate(records) if row['question_type'] == 'multi_select' and len(golds[i]['displayed_gold']) < len(row['options']))
    right = ','.join(golds[idx]['displayed_gold'])
    reasoned = '<think>Keep this reasoning exactly.\n</think>\nEvidence considered. <box>' + right + '</box>'
    scored = score(reasoned, idx)
    assert scored['exact'] == scored['J'] == 1.0 and scored['final_answer_valid'] and not scored['auxiliary_protocol_valid']
    spaced = score(json.dumps({'final_answer': '<box>' + right.replace(',', ' ') + '</box>'}), idx)
    assert spaced['exact'] == 1.0 and not spaced['auxiliary_protocol_valid']
    assert score(reasoned, idx, 'length')['J'] == 0.0
    assert not score(reasoned + '<box>A</box>', idx)['final_answer_valid']
    assert not score('<box>A,A</box>', idx)['final_answer_valid']
    wrong = next(label for label in records[idx]['display_to_option_id'] if label not in golds[idx]['displayed_gold'])
    assert score('<box>' + wrong + '</box>', idx)['J'] < 0.0
    order_idx = next(i for i, row in enumerate(records) if row['question_type'] == 'ordering')
    order = golds[order_idx]['displayed_gold']
    assert score('<box>' + '>'.join(order) + '</box>', order_idx)['exact'] == 1.0
    assert score('<box>' + ' '.join(reversed(order)) + '</box>', order_idx)['J'] == -1.0
    # All actual Gold rows, including the three original all-selected 5/5 cases.
    all_selected = 0
    for i, (record, gold) in enumerate(zip(records, golds)):
        good = score('<box>' + ','.join(gold['displayed_gold']) + '</box>', i)
        assert good['final_answer_valid'] and good['exact'] == good['J'] == 1.0
        bad = score('<box>UNKNOWN_LABEL</box>', i)
        assert not bad['final_answer_valid'] and bad['exact'] == bad['J'] == 0.0
        duplicate = score('<box>' + ','.join([gold['displayed_gold'][0]] * 2) + '</box>', i)
        assert not duplicate['final_answer_valid'] and duplicate['exact'] == duplicate['J'] == 0.0
        if record['question_type'] == 'multi_select' and len(gold['displayed_gold']) == len(record['options']):
            all_selected += 1
    assert all_selected == 3

    def fail_real_loader(*args, **kwargs):
        raise AssertionError('REAL_MODEL_LOADING_IS_FORBIDDEN_IN_CPU_SELFCHECK')
    transformers.AutoModelForCausalLM.from_pretrained = fail_real_loader

    class CPUModelDouble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.generated_count = 0
            self.loss_count = 0

        def generate(self, *, input_ids, attention_mask, **kwargs):
            assert input_ids.device.type == 'cpu' and not torch.is_grad_enabled() and not self.training
            assert kwargs['do_sample'] is False and kwargs['use_cache'] is True
            assert kwargs['max_new_tokens'] == 6144 and kwargs['num_beams'] == 1
            row, target = records[self.generated_count], golds[self.generated_count]
            obj = {'item_evidence': {label: {'judgment': 'not_supported', 'turn_refs': [], 'reason': 'CPU fixture only.'}
                                     for label in row['display_to_option_id']},
                   'answer_reason': 'Preserve this explicit CPU fixture reasoning verbatim.',
                   'final_answer': '<box>' + ' '.join(target['displayed_gold']) + '</box>'}
            completion = tokenizer(json.dumps(obj), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            self.generated_count += 1
            return SimpleNamespace(sequences=torch.cat((input_ids, torch.tensor([completion], dtype=torch.long)), dim=1))

        def forward(self, input_ids, attention_mask, labels):
            assert input_ids.device.type == 'cpu' and not torch.is_grad_enabled() and not self.training
            assert self.generated_count == self.loss_count + 1
            self.loss_count += 1
            # Nonconstant per-record loss makes weighting errors observable.
            return SimpleNamespace(loss=torch.tensor(1.0 + self.loss_count / 242.0))

    fake = CPUModelDouble()
    with tempfile.TemporaryDirectory(prefix='fixed_dev_cpu_selfcheck_') as tmp:
        output = Path(tmp) / 'evaluation'
        output.mkdir()
        result = evaluate(model=fake, tokenizer=tokenizer, step=0, output_dir=output,
                          validation_manifest=FINAL / 'VALIDATION_MANIFEST.json')
        assert result['evaluated_records'] == 242 and result['raw_cot_saved'] is True
        assert fake.generated_count == fake.loss_count == 242
        assert result['metrics']['exact'] == result['metrics']['mean_J'] == 1.0
        assert result['metrics']['auxiliary_protocol_valid_rate'] == 1.0
        assert not math.isclose(result['metrics']['valid_loss'], result['metrics']['valid_loss_token_weighted'])
        assert all(Path(item['path']).is_relative_to(output) and bind(item['path'])['sha256'] == item['sha256'] for item in result['artifacts'])
        actual = [json.loads(line) for line in (output / 'RECORDS.jsonl').read_text().splitlines()]
        assert all('explicit CPU fixture reasoning verbatim.' in row['raw_output'] for row in actual)
        assert all(row['finish_reason'] == 'eos' and row['truncated'] is False for row in actual)
        assert aggregate(actual) == result['metrics']
        artifact_count = len(result['artifacts'])
        values = result['metrics']
    report = {'schema_version': 'fixed_dev_callback_cpu_selfcheck_v1', 'status': 'PASS',
              'real_model_weights_loaded': False, 'gpu_execution': False, 'real_model_calls': 0,
              'cpu_test_double_generations': 242, 'cpu_test_double_loss_calls': 242,
              'actual_gold_oracle_records': 242, 'per_record_unknown_label_zero_fixtures': 242,
              'per_record_duplicate_label_zero_fixtures': 242, 'actual_multi_select_all_selected_gold_records': all_selected,
              'artifact_bindings_verified': artifact_count,
              'checks': ['complete real-DEV242 callback with native tokenizer/mask and CPU test double',
                         'JSON/evidence format failures preserve correct final-answer semantic scores',
                         'space/comma/order-delimiter final-box transport', 'invalid/truncated answers zero',
                         'signed official J boundaries', 'full raw reasoning persisted',
                         'distinct sample/token weighted loss', 'all artifacts confined to output_dir'],
              'fixture_metrics': values, 'validation_manifest': bind(FINAL / 'VALIDATION_MANIFEST.json'),
              'callback': bind(FINAL / 'tools/validation_callback.py')}
    write_json(FINAL / 'VALIDATION_CALLBACK_CPU_SELFCHECK_V2.json', report)
    print(json.dumps({key: value for key, value in report.items() if key not in ('fixture_metrics',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
