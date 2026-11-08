#!/usr/bin/env python3
"""Versioned engineering repair: verify non-JSON artifacts without parsing them.

The v1 producer's initial execution succeeded. Its resume branch called the JSON
reader on every artifact, including Python source. No frozen input is rewritten.
"""
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('oet1_preparation_v1',BASE/'scripts/prepare_production.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

def verify(binding):
    assert module.bind(binding['path'])==binding,('BINDING_CHANGED',binding['path'])
    if Path(binding['path']).suffix=='.json': return module.read(binding['path'])
    return None

module.verify=verify
if __name__=='__main__': module.main()
