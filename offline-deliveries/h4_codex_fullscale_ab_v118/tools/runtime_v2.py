"""Versioned entry points with a strictly source-bound read-only path adapter.

Frozen original tools and all source/question/Gold files remain unchanged.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_export_v2 import export_root_questions


def module(name):
    spec = importlib.util.spec_from_file_location('v118_preserved_' + name, HERE / (name + '.py'))
    obj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = obj; spec.loader.exec_module(obj)
    obj.export_root_questions = export_root_questions
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tool', choices=['audit', 'evaluate', 'deliver'])
    p.add_argument('action', choices=['prepare', 'preflight', 'live', 'run'])
    p.add_argument('--wave', choices=['validation', 'production', 'all'], required=True)
    p.add_argument('--side', choices=['base', 'trained'])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.tool == 'evaluate':
        assert a.wave != 'all' and a.action in ['prepare', 'preflight', 'live']
        obj = module('evaluate')
        if a.action == 'live':
            assert a.side; obj.live(a.wave, a.side)
        else: getattr(obj, a.action)(a.wave)
    else:
        assert a.action == 'run' and a.output
        obj = module('audit_content' if a.tool == 'audit' else 'deliver')
        obj.main(a.wave, a.output.resolve()) if a.tool == 'audit' else obj.summarize(a.wave, a.output.resolve())


if __name__ == '__main__': main()
