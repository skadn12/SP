#!/usr/bin/env python3
"""Frozen v4 finite scheduler with the byte-only non-JSON binding repair.

Only the scoped question adapter is upgraded to frozen questions_v2. Existing
author output and semantic FAIL/HOLD remain immutable; cached original author
responses are consumed without another semantic request. Import launches no
driver, data CLI, or GPU process.
"""
import importlib.util
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


version4 = module('oet1_frozen_driver_v4', BASE / 'scripts/production_driver_v4.py')
driver = version4.driver
questions = module('oet1_frozen_questions_v2', BASE / 'scripts/questions_v2.py')
guard = version4.version3.guard
original_guard_module = guard.module


def question_unit(root, arm):
    return questions.run_cli_unit(root, arm, driver.call, driver.prompt_prefix)


def guarded_module(name, path):
    # Keep any explicit second-wave gate on the same repaired mechanical
    # validator. It must still pass the original frozen independent gate.
    if Path(path).resolve() == BASE / 'scripts/questions.py':
        return questions
    return original_guard_module(name, path)


def main():
    expected = {
        'production_driver_v4.py': '3b70cbdffe28d57aa1d33a4fe73e75cb21ff3dcf0a01ea4a1afd3ca984f9228a',
        'questions_v2.py': 'cad3f33d96e33fa2ffe9653ec7ee9dcd278564320bd8ee4d167739d147a13eb1',
    }
    for name, digest in expected.items():
        if driver.bind(BASE / 'scripts' / name)['sha256'] != digest:
            raise ValueError('FROZEN_V5_DEPENDENCY_CHANGED:' + name)
    driver.question_unit = question_unit
    guard.module = guarded_module
    version4.main()


if __name__ == '__main__':
    main()
