"""Prospectively normalize evaluation source_path before native validation.

The source text, questions, gold, citations and authoring files stay byte-for-byte
unchanged. This batch does not claim that a failed preparation happened first.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_questions(questions, project=PROJECT):
    """Return a deep copy and truthful per-candidate representation receipts."""
    normalized = copy.deepcopy(questions)
    changes = []
    for old, new in zip(questions, normalized):
        source = Path(old['source_path'])
        resolved = (source if source.is_absolute() else Path(project) / source).resolve()
        assert resolved.is_file(), 'SOURCE_PATH_MISSING:' + str(resolved)
        assert sha(resolved) == old['source_sha256'], 'SOURCE_HASH_MISMATCH:' + str(resolved)
        new['source_path'] = str(resolved)
        assert {k: v for k, v in old.items() if k != 'source_path'} == {
            k: v for k, v in new.items() if k != 'source_path'}
        changes.append({'candidate_id': old['candidate_id'],
                        'before': old['source_path'], 'after': new['source_path'],
                        'representation_changed': old['source_path'] != new['source_path'],
                        'source_sha256': old['source_sha256']})
    return normalized, changes


def main():
    """Keep the historical script entry point, using the integrated preparation."""
    sys.path.insert(0, str(HERE))
    import evaluate
    assert not (HERE / 'evaluation/EVALUATION_POOL.json').exists(), 'PREPARATION_ALREADY_SAVED'
    assert not (HERE / 'evaluation/ADAPTER_BINDINGS.json').exists(), 'BINDINGS_ALREADY_SAVED'
    assert not list((HERE / 'evaluation').glob('*_STARTED.json')), 'SOLVER_ALREADY_STARTED'
    assert not (HERE / 'evaluation/run').exists(), 'PREFLIGHT_ALREADY_SAVED'
    evaluate.prepare()


if __name__ == '__main__':
    main()
