#!/usr/bin/env python3
"""Read-only finite-weight/identity validation of native adapter checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_contract import INIT, INIT_SHA, MODEL, RECIPE, bind, now, read, sha


def validate_checkpoint(path, expected_step, initial=INIT):
    import torch

    path = Path(path)
    initial = Path(initial)
    assert sha(initial/'adapter_model.pt') == INIT_SHA, 'MCLEAN_ADAPTER_CHANGED'
    config = read(path/'adapter_config.json')
    assert config['step'] == expected_step, 'CHECKPOINT_STEP_MISMATCH'
    assert config['model'] == str(MODEL), 'CHECKPOINT_BASE_MISMATCH'
    assert config['tokenizer'] == str(MODEL), 'CHECKPOINT_TOKENIZER_MISMATCH'
    assert config['init_adapter_dir'] == str(initial), 'CHECKPOINT_START_ADAPTER_MISMATCH'
    for field in ('lora_rank', 'lora_alpha', 'lora_dropout', 'max_length'):
        assert config[field] == RECIPE[field], ('CHECKPOINT_RECIPE_MISMATCH', field)
    assert config['target_modules'] == RECIPE['target_modules'].split(',')
    state = torch.load(path/'adapter_model.pt', map_location='cpu', weights_only=True)
    start = torch.load(initial/'adapter_model.pt', map_location='cpu', weights_only=True)
    assert state and set(state) == set(start), 'CHECKPOINT_LORA_KEY_MISMATCH'
    finite = True
    nonzero = changed = count = 0
    for key, value in state.items():
        assert '.lora_' in key and value.shape == start[key].shape, ('CHECKPOINT_SHAPE_MISMATCH', key)
        finite = finite and bool(torch.isfinite(value).all().item())
        count += value.numel()
        nonzero += int(torch.count_nonzero(value).item())
        changed += int(torch.count_nonzero(value != start[key]).item())
    assert finite and nonzero > 0 and changed > 0, 'CHECKPOINT_NONFINITE_ZERO_OR_UNCHANGED'
    return dict(status='PASS', checked_at_utc=now(), step=expected_step,
                config_binding=bind(path/'adapter_config.json'), weight_binding=bind(path/'adapter_model.pt'),
                tensor_count=len(state), parameter_count=count, all_weights_finite=finite,
                nonzero_weight_elements=nonzero, changed_elements_from_mclean=changed,
                init_adapter_sha256=INIT_SHA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.checkpoint, args.expected_step), ensure_ascii=False, indent=2))
