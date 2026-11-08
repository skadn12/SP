"""Shared append-only helpers and read-only access to frozen engines."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DELIVERIES = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries')
V75 = DELIVERIES / 'h4_output_contract_multiselect_v79'
ORDERING_CODE = DELIVERIES / 'h4_sequential_ab_v58_ordering_qa'
NATIVE = DELIVERIES / 'h4_feedback_training_pool_v15_qa/evaluation'
PYTHON = Path('/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python')


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def require_shard(home=HERE):
    """Run the installed copy only; shared engine code never owns shard state."""
    home = Path(home).resolve()
    need(home == HERE, 'shard_home_must_equal_installed_code_directory')
    record = read(home / 'SHARD_ENGINE_INSTALLATION.json')
    need(record['version'] == 'v98_shard_engine_installation_v1'
         and Path(record['shard_home']).resolve() == home
         and record['training_allowed'] is False, 'shard_installation_identity')
    require_bindings(record['files'], list(home.glob('*.py')))
    return home


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verify_bindings(records):
    for row in records:
        need(sha(row['path']) == row['sha256'], 'frozen_file_changed:' + row['path'])


def require_bindings(records, required):
    need(isinstance(records, list), 'frozen_file_list_required')
    paths = []
    for row in records:
        need(isinstance(row, dict) and set(row) == {'path', 'sha256'}
             and isinstance(row['path'], str) and Path(row['path']).is_absolute()
             and isinstance(row['sha256'], str) and len(row['sha256']) == 64
             and all(c in '0123456789abcdef' for c in row['sha256']), 'frozen_binding_schema')
        paths.append(Path(row['path']).resolve())
    need(len(paths) == len(set(paths)) and {Path(p).resolve() for p in required} <= set(paths),
         'frozen_binding_coverage')
    verify_bindings(records)


def now():
    return datetime.now(timezone.utc).isoformat()


def aware(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        raise ValueError('aware_timestamp_required')
    need(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'timezone_required')
    return parsed


def load_multi(name):
    need(name in ('asset_generation', 'generation_driver', 'compose', 'evaluate', 'review_schema'),
         'unknown_multiselect_engine')
    import importlib
    if str(V75) not in sys.path:
        sys.path.insert(0, str(V75))
    result = importlib.import_module(name)
    need(Path(result.__file__).resolve() == V75 / (name + '.py'), 'wrong_multiselect_module')
    import prior_rendering
    assets = importlib.import_module('asset_generation')
    need(Path(assets.__file__).resolve() == V75 / 'asset_generation.py', 'wrong_multiselect_asset_module')
    prior_rendering.install(assets)
    return result


def load_ordering_gate():
    key = '_v76_frozen_v58_ordering_gate'
    if key not in sys.modules:
        original = sys.path[:]
        saved = {name: sys.modules.get(name) for name in ('prompt_layout', 'reference_compat')}
        try:
            sys.path.insert(0, str(ORDERING_CODE))
            for name in saved:
                sys.modules.pop(name, None)
            spec = importlib.util.spec_from_file_location(key, ORDERING_CODE / 'ordering_gate_v45.py')
            result = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(result)
            sys.modules[key] = result
        finally:
            sys.path[:] = original
            for name, module in saved.items():
                sys.modules.pop(name, None)
                if module is not None:
                    sys.modules[name] = module
    return sys.modules[key]
