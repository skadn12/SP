"""Resolve recorded source path spellings without editing frozen question files.

Only paths which identify the exact root's cumulative source are accepted.
The unchanged lossless exporter performs all evidence/hash/Gold checks.
"""
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
P = B.parent.parent
sys.path.insert(0, str(P / 'scripts'))
from codex_evidence_export import export_root_questions as original_export


def resolve_public_source(batch, unit, value):
    p = Path(value)
    if p.is_absolute():
        assert p.is_file(), 'RECORDED_ABSOLUTE_SOURCE_MISSING'
        return p.resolve()
    candidates = {(base / p).resolve() for base in [Path(unit), Path(batch), P] if (base / p).is_file()}
    assert len(candidates) == 1, 'RECORDED_RELATIVE_SOURCE_MISSING_OR_AMBIGUOUS'
    return candidates.pop()


@contextmanager
def source_working_directory(folder):
    import json
    folder = Path(folder).resolve(); expected = folder / 'CUMULATIVE_SOURCE.json'
    questions = json.loads((folder / 'PUBLIC_QUESTIONS.json').read_text())['questions']
    assert questions, 'PUBLIC_QUESTIONS_EMPTY'
    unit = folder.parents[1]
    for q in questions:
        assert resolve_public_source(B, unit, q['source_path']) == expected, 'PUBLIC_SOURCE_IS_NOT_THIS_ROOT'
    bases = [unit, B, P]
    valid = [base for base in bases if all((Path(q['source_path']) if Path(q['source_path']).is_absolute()
                                          else base / q['source_path']).resolve() == expected for q in questions)]
    assert valid, 'NO_SINGLE_CANONICAL_SOURCE_DIRECTORY'
    previous = Path.cwd()
    try:
        os.chdir(valid[0]); yield
    finally:
        os.chdir(previous)


def export_root_questions(root_folder):
    folder = Path(root_folder).resolve()
    with source_working_directory(folder):
        questions = original_export(folder)
    out = deepcopy(questions)
    for q in out:
        # Path metadata only: question text, option order, Gold and all evidence
        # remain byte-for-byte equal as JSON values to the unchanged exporter.
        q['source_path'] = str(folder / 'CUMULATIVE_SOURCE.json')
    return out
