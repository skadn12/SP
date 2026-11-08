"""Fresh A3+B3 evaluation controller with frozen input and provenance gates.

The native serialization and scene mechanics are reused unchanged. This
adapter never imports a historical prefix or invokes a model or Solver.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MASTER = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(MASTER / 'scripts'))
spec = importlib.util.spec_from_file_location('v124_native_scene_controller', MASTER / 'scripts/codex_sequential_ab.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
_native_inspect = native.inspect_root
SCENES, CHECKS, TERMINAL_RECEIPTS = native.SCENES, native.CHECKS, native.TERMINAL_RECEIPTS
receipts, load_config, root_config = native.receipts, native.load_config, native.root_config
root_folder, scene_folder, load_baseline = native.root_folder, native.scene_folder, native.load_baseline
reservation_for, append_receipt, digest = native.reservation_for, native.append_receipt, native.digest


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def require(ok, why):
    if not ok: raise ValueError(why)
def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
def require_loaded_function(function, expected):
    code = getattr(function, '__code__', None)
    require(code is not None and Path(code.co_filename).resolve() == Path(expected).resolve(), 'FOREIGN_LOADED_RUNTIME:' + str(expected))
def require_tool_origin(name):
    expected = (HERE / (name + '.py')).resolve()
    found = importlib.util.find_spec(name)
    require(found is not None and found.origin is not None and Path(found.origin).resolve() == expected, 'FOREIGN_LOADED_RUNTIME:' + name)
    return expected
def frozen_tool(name):
    expected = require_tool_origin(name)
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == expected, 'FOREIGN_LOADED_RUNTIME:' + name)
    return module


def execution_input_paths(unit):
    unit = Path(unit).resolve()
    fixed = [unit / name for name in ('RUN_CONFIG.json', 'SOURCE_RESERVATIONS.json', 'selection.json')]
    fixed += [unit / 'inputs' / name for name in ('BASELINE.json', 'CONTINUATION_CONSTRAINTS.json', 'EVENT_POOL.json', 'B_S01_EVENT.json', 'B_S02_EVENT.json', 'B_S03_EVENT.json')]
    fixed += [unit / 'evidence/MATCHING' / name for name in ('FULL_INPUT.txt', 'READ_CONTEXT.json', 'MATCHING_PAYLOAD.json')]
    for path in fixed: require(path.is_file(), 'EXECUTION_REQUIRED_INPUT_MISSING:' + str(path))
    return sorted(set(fixed))


def check_execution_freeze(unit, master):
    unit, master = Path(unit).resolve(), Path(master).resolve()
    path = unit / 'EXECUTION_INPUTS_FROZEN.json'
    doc = read(path)
    require(doc.get('status') == 'READY_FOR_A3_THEN_B3' and doc.get('generation_started') is False, 'EXECUTION_FREEZE_NOT_READY')
    require(doc.get('master_freeze_binding') == binding(master / 'MASTER_FREEZE.json'), 'EXECUTION_MASTER_CHANGED')
    paths = execution_input_paths(unit)
    rows = doc.get('files', [])
    require(len(rows) == len(paths) and {r.get('path') for r in rows} == {str(p) for p in paths}, 'EXECUTION_FREEZE_FILE_SET')
    for row in rows: require(binding(row['path']) == row, 'EXECUTION_INPUT_CHANGED:' + row['path'])
    cfg = read(unit / 'RUN_CONFIG.json')
    require(cfg.get('usage') == 'EVALUATION_ONLY' and cfg.get('solver_run') is False and cfg.get('training_run') is False
            and cfg.get('score_based_selection') is False and cfg.get('training_export') is False, 'EVAL_ONLY_CONFIGURATION_REQUIRED')
    root = doc.get('root_id')
    from phase_gate_v124 import require_phase, check_candidate_pool
    require_phase(master, root)
    check_candidate_pool(master, root, read(unit / 'inputs/EVENT_POOL.json'))
    require([r['root_id'] for r in cfg['roots']] == [root], 'EXECUTION_FREEZE_ROOT')
    roles = [r for r in read(master / 'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id'] == root]
    require(len(roles) == 1 and doc.get('author') == roles[0]['author'] and doc.get('reviewer') == roles[0]['reviewer'], 'EXECUTION_FREEZE_ROLES')
    require(cfg.get('split') in ('dev', 'test'), 'EVAL_SPLIT_REQUIRED')
    from public_anchor_v1 import prefix_input_paths
    prefix_input_paths(unit)
    return doc


def check_freeze(unit):
    unit = Path(unit).resolve()
    require(unit.parent.name == 'units', 'EXPECTED_UNIT_LAYOUT')
    master = unit.parents[1]
    require(master == MASTER, 'FOREIGN_BATCH_RUNTIME')
    from public_anchor_v1 import validate_runtime
    validate_runtime(master, __file__)
    freeze = read(master / 'MASTER_FREEZE.json')
    require(freeze.get('status') == 'FROZEN_BEFORE_NEW_BASELINE_FULL_TEXT_READS', 'MASTER_FREEZE_NOT_READY')
    for row in freeze['files']:
        require(binding(row['path']) == row, 'MASTER_INPUT_CHANGED:' + row['path'])
    check_execution_freeze(unit, master)
    return master


def new_scene_english_guard(draft):
    for turn in draft.get('turns', []) if isinstance(draft, dict) else []:
        text = turn.get('text', '') if isinstance(turn, dict) else ''
        require(not any('\u3400' <= c <= '\u9fff' or '\uf900' <= c <= '\ufaff' for c in text), 'NEW_SCENE_CJK_TEXT_BLOCKED_ENGLISH_PROTOCOL')


@contextmanager
def guarded_review_runtime():
    original = native.validate_review
    def validate_review(folder, root, scene, source_path, source):
        gate = frozen_tool('review_provenance_gate_v2').check_scene(MASTER, root, scene)
        require(gate['pass'], 'SCENE_REVIEW_PROVENANCE_BLOCKED:' + json.dumps(gate.get('errors', []), ensure_ascii=False))
        return original(folder, root, scene, source_path, source)
    native.validate_review = validate_review
    try: yield
    finally: native.validate_review = original


def inspect_root(batch, cfg, root_id, record_reviews=True):
    check_freeze(batch)
    with guarded_review_runtime():
        return _native_inspect(batch, cfg, root_id, record_reviews=record_reviews)


def materialize_scene(batch, root_id, scene_ref):
    master = check_freeze(batch)
    draft_path = scene_folder(batch, root_id, scene_ref) / 'EXPANSION_DRAFT.json'
    if draft_path.exists():
        draft = read(draft_path)
        new_scene_english_guard(draft)
        role = next(r for r in read(master / 'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id'] == root_id)
        require(draft.get('author') == role['author'], 'DRAFT_AUTHOR_NOT_ASSIGNED')
    old = native.inspect_root
    native.inspect_root = inspect_root
    try: return native.materialize_scene(batch, root_id, scene_ref)
    finally: native.inspect_root = old


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['materialize-scene', 'progress'])
    p.add_argument('--batch-dir', type=Path, required=True); p.add_argument('--root-id', required=True)
    p.add_argument('--scene-ref', choices=SCENES)
    a = p.parse_args()
    if a.action == 'materialize-scene':
        require(a.scene_ref, 'SCENE_REQUIRED'); out = materialize_scene(a.batch_dir, a.root_id, a.scene_ref)
    else: out = inspect_root(a.batch_dir, load_config(a.batch_dir), a.root_id)
    print(json.dumps(out, ensure_ascii=False, indent=2))
