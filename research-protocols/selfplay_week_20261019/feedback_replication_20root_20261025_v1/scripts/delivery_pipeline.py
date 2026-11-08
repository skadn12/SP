#!/usr/bin/env python3
"""Finite delivery of settled, authorized replication artifacts.

Only ``run`` launches children, after an independently bound engineering READY.
No source/question/Gold authoring is reachable. The only possible fresh text
call is the existing closeout audit of an already frozen report. This script
never changes a frozen implementation, question, source, review or score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
PYTHON = '/root/miniconda3/bin/python'
READY = BASE / 'DELIVERY_PIPELINE_READY.json'
EVENTS = BASE / 'worker_events/delivery_pipeline'
DISPOSITIONS = BASE / 'DELIVERY_TERMINAL_DISPOSITIONS.json'
F179_READY = BASE / 'question_saved_revision_verification/179186/F/CONTINUATION_FINALIZER_READY.json'
F136_READY = BASE / 'question_saved_mc5_first_verification/136202/F/CONTINUATION_FINALIZER_READY.json'
SERIAL_STEPS = ('shape_77882_G', 'shape_136202_F', 'saved_179186_G',
                'saved_136202_G', 'saved_109164_G', 'unused_77882_F', 'unused_failed_109164_F')
STAGES = ('content', 'cpu-length', 'prepare', 'run', 'analyze', 'replication',
          'report', 'closeout-prepare', 'closeout-run', 'manifest', 'validate')
GPU_STAGES = {'prepare', 'run'}
NEW_ENGINEERING_STAGES = {'cpu-length', 'prepare', 'run'}
CLOSEOUT_CALL = 'q20_closeout_frozen_report_01'
HEARTBEAT_SECONDS = 35
MUTABLE_ANALYSIS_SNAPSHOTS = frozenset(BASE / name for name in
    ('TASK_STATUS.json', 'SUPERVISOR_STATUS.json', 'PROGRESS.jsonl', 'supervisor.log'))
STOPPING = False


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def verify(row: dict[str, Any]) -> None:
    if not isinstance(row, dict) or bind(row['path']) != {k: row.get(k) for k in ('path', 'sha256', 'bytes')}:
        raise RuntimeError('BINDING_DRIFT:' + str(row))


def verify_nested(value: Any) -> None:
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= set(value): verify(value)
        for child in value.values(): verify_nested(child)
    elif isinstance(value, list):
        for child in value: verify_nested(child)


def verify_comparison(value: Any, base: Path = BASE) -> list[dict[str, Any]]:
    """Only direct exact-root mutable analysis inputs are historical snapshots.

    Similarly named files elsewhere and all other nested bindings must match.
    Renamed disclosure keys prevent later receipt walks from mistaking an old
    status snapshot for a current scientific input binding.
    """
    if not isinstance(value, dict): raise RuntimeError('COMPARISON_NOT_OBJECT')
    permitted = {base / path.name for path in MUTABLE_ANALYSIS_SNAPSHOTS}
    snapshots = []
    for key, child in value.items():
        if key != 'input_bindings':
            verify_nested(child); continue
        if not isinstance(child, list): raise RuntimeError('COMPARISON_INPUT_BINDINGS_NOT_LIST')
        for row in child:
            if (isinstance(row, dict) and set(row) == {'path', 'sha256', 'bytes'}
                    and Path(row['path']) in permitted and Path(row['path']).is_absolute()):
                if (not isinstance(row['sha256'], str) or len(row['sha256']) != 64
                        or any(char not in '0123456789abcdef' for char in row['sha256'])
                        or not isinstance(row['bytes'], int) or isinstance(row['bytes'], bool) or row['bytes'] < 0):
                    raise RuntimeError('MALFORMED_HISTORICAL_ANALYSIS_SNAPSHOT')
                snapshots.append({'snapshot_path': row['path'], 'snapshot_sha256': row['sha256'],
                    'snapshot_bytes': row['bytes'], 'asserted_current': False,
                    'policy': 'Historical mutable analysis input snapshot; no scientific input exemption.'})
            else: verify_nested(row)
    return snapshots


def immutable(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw: raise RuntimeError('IMMUTABLE_OUTPUT_DRIFT:' + str(path))
    else:
        with path.open('xb') as handle: handle.write(raw)
    return bind(path)


def heartbeat(phase: str, **fields: Any) -> None:
    row = {'at_utc': utc(), 'kind': 'DELIVERY_PIPELINE_HEARTBEAT', 'phase': phase,
           'delivery_pid': os.getpid(), 'training': False, 'rerun_original_four': False, **fields}
    with (BASE / 'PROGRESS.jsonl').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX); handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    state_path = BASE / 'TASK_STATUS.json'; state = read(state_path)
    state['updated_at_utc'] = row['at_utc']; state['delivery_pipeline'] = row
    temporary = state_path.with_name(state_path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n'); os.replace(temporary, state_path)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def process_info(pid: int) -> tuple[int, str] | None:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else (int(fields[1]), fields[19])
    except (OSError, IndexError, ValueError): return None


def live_calls() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((BASE / 'process_registry').glob('*.json')):
        row = read(path); info = process_info(row.get('pid', -1))
        if row.get('active') is True and info and info[1] == str(row.get('start_token')):
            rows.append({**row, 'registry_path': str(path)})
    return rows


def scoped_script(argv: list[str], cwd: Path) -> Path | None:
    if not argv or 'python' not in Path(argv[0]).name: return None
    script = next((arg for arg in argv[1:] if arg.endswith('.py')), None)
    if script is None: return None
    actual = (Path(script) if Path(script).is_absolute() else cwd / script).resolve()
    if actual in {BASE / 'supervisor.py', BASE / 'launch.py', BASE / 'scripts/status_heartbeat.py'}: return None
    if actual.parent == BASE / 'scripts' or actual in {BASE / 'solver_runner.py', BASE / 'analysis_replication.py', BASE / 'validate_final.py'}:
        return actual
    return None


def active_parents() -> list[dict[str, Any]]:
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid(): continue
        try:
            argv = [s.decode(errors='replace') for s in (proc / 'cmdline').read_bytes().split(b'\0') if s]
            actual = scoped_script(argv, (proc / 'cwd').resolve()); info = process_info(int(proc.name))
            if actual and info: rows.append({'pid': int(proc.name), 'start_token': info[1], 'script': str(actual), 'argv': argv})
        except OSError: continue
    return rows


def cutoff(base: Path = BASE, now: dt.datetime | None = None) -> str | None:
    if STOPPING or (base / 'STOP_REQUEST').exists(): return 'STOP_REQUEST'
    deadline = dt.datetime.fromisoformat(read(base / 'SUPERVISOR_STATUS.json')['deadline_at_utc'])
    current = now or dt.datetime.now(dt.timezone.utc)
    if current >= deadline: return 'DEADLINE'
    if current >= deadline - dt.timedelta(minutes=20): return 'CLOSEOUT_ONLY'
    if (base / 'FINALIZE_REQUEST').exists() or current >= deadline - dt.timedelta(minutes=45): return 'NO_NEW_CONTENT_GENERATION'
    return None


def orphan_starts() -> list[str]:
    return [str(path) for path in sorted((BASE / 'cli_calls').glob('*/attempt_*/START.json'))
            if not any((path.parent / name).exists() for name in ('TERMINAL.json', 'TRANSPORT_FAILURE.json'))
            and not (path.parent.parent / 'RESULT.json').exists()
            and not (path.parent.parent / 'SEMANTIC_FAILURE.json').exists()]


def wait_idle() -> None:
    last = 0.0
    while True:
        reason = cutoff()
        if reason == 'STOP_REQUEST': raise RuntimeError(reason + ': no further wait or new work')
        calls, parents = live_calls(), active_parents()
        if not calls and not parents: return
        if reason == 'DEADLINE': raise RuntimeError('DEADLINE: no further wait for existing parents')
        if time.monotonic() - last >= HEARTBEAT_SECONDS:
            heartbeat('waiting_all_task_parents_and_registry_idle', calls=calls, parents=parents, cutoff=reason); last = time.monotonic()
        time.sleep(1)


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value; spec.loader.exec_module(value); return value


def terminal_dispositions() -> dict[str, dict[str, Any]]:
    if not DISPOSITIONS.exists(): return {}
    doc = read(DISPOSITIONS)
    if (doc.get('schema_version') != 'fgr20_delivery_terminal_dispositions_v1'
            or doc.get('issuer_agent') != '/root' or doc.get('authorizes_no_new_semantic_calls') is not True
            or not isinstance(doc.get('dispositions'), list)):
        raise RuntimeError('EXPLICIT_ROOT_TERMINAL_DISPOSITION_INVALID')
    result = {}
    for row in doc['dispositions']:
        component = row.get('component')
        if (component not in {'serial7', 'f179186_final_verifier', 'f136202_mc5_first_verifier'} or component in result
                or row.get('status') not in {'FAILED_TERMINAL', 'NOT_RUN_TERMINAL'}
                or row.get('allow_delivery_with_missingness') is not True
                or not isinstance(row.get('reason'), str) or not row['reason'].strip()
                or not isinstance(row.get('evidence_bindings'), list) or not row['evidence_bindings']):
            raise RuntimeError('EXPLICIT_ROOT_TERMINAL_DISPOSITION_ROW_INVALID')
        for evidence in row['evidence_bindings']:
            Path(evidence['path']).resolve().relative_to(BASE); verify(evidence)
        result[component] = row
    return result


def checked_continuation(path: Path) -> dict[str, Any]:
    import content_continuation_adapter as adapter
    checker = adapter.Continuations.__new__(adapter.Continuations); checker.inputs = []
    row = checker.validate_ready(path)
    return {'ready_binding': bind(path), 'row': row, 'verified_input_bindings': len(checker.inputs)}


def prerequisites() -> dict[str, Any]:
    dispositions = terminal_dispositions(); result = {}
    matches = []
    for path in sorted((BASE / 'worker_events/authorized_pipeline_resume').glob('HANDOFF_*.json')):
        row = read(path)
        if (row.get('status') == 'SEMANTIC_SEQUENCE_TERMINAL_HANDOFF' and not row.get('cutoff')
                and tuple(x.get('step') for x in row.get('steps', [])) == SERIAL_STEPS
                and all(x.get('status') in {'REUSED_TERMINAL_OUTPUT', 'TERMINAL_OUTPUT_OBSERVED'} for x in row['steps'])):
            verify_nested(row); matches.append((path, row))
    if len(matches) == 1:
        path, handoff = matches[0]
        import resume_authorized_pipeline as serial
        if tuple(step[0] for step in serial.STEPS) != SERIAL_STEPS: raise RuntimeError('SERIAL7_CODE_ORDER_DRIFT')
        for step, observed in zip(serial.STEPS, handoff['steps']):
            actual = serial.validated_output(step)
            recorded = observed.get('ready_binding') or (observed.get('output_validation') or {}).get('ready_binding')
            if not actual or actual['ready_binding'] != recorded: raise RuntimeError('SERIAL7_TERMINAL_OUTPUT_DRIFT:' + step[0])
        result['serial7'] = {'status': 'ACTUAL_SEVEN_TERMINAL_OUTPUTS_VERIFIED', 'handoff_binding': bind(path)}
    elif 'serial7' in dispositions:
        result['serial7'] = {'status': 'EXPLICIT_ROOT_MISSINGNESS_DISPOSITION', 'disposition': dispositions['serial7']}
    else: raise RuntimeError('REQUIRED_SERIAL7_TERMINAL_OR_EXPLICIT_ROOT_DISPOSITION_MISSING')
    if F179_READY.is_file():
        checked = checked_continuation(F179_READY); row = checked['row']
        if ((row.get('root_id'), row.get('arm'), row.get('serialization_variant'))
                != ('cc-train-episode-179186', 'F', 'f179186_shape_final_revision_one_r03_alias_no_global_copy')):
            raise RuntimeError('F179_ACTUAL_FINAL_VERIFIER_OUTPUT_IDENTITY_DRIFT')
        result['f179186_final_verifier'] = {'status': 'ACTUAL_TERMINAL_OUTPUT_VERIFIED', **checked}
    elif 'f179186_final_verifier' in dispositions:
        result['f179186_final_verifier'] = {'status': 'EXPLICIT_ROOT_MISSINGNESS_DISPOSITION', 'disposition': dispositions['f179186_final_verifier']}
    else: raise RuntimeError('REQUIRED_F179_ACTUAL_TERMINAL_OR_EXPLICIT_ROOT_DISPOSITION_MISSING')
    if F136_READY.is_file():
        import content_assembly_delivery as delivery_content
        checked = delivery_content.validate_f136_terminal_ready(F136_READY); row = checked['record']
        if ((row.get('root_id'), row.get('arm'), row.get('serialization_variant'), row.get('continuation_kind'))
                != ('cc-train-episode-136202', 'F', 'f136202_final_revision_exact_five_MC_first_verifier_list_keyed',
                    'saved_final_MC_five_first_verifier')):
            raise RuntimeError('F136_EXACT_FIVE_FIRST_VERIFIER_TERMINAL_IDENTITY_DRIFT')
        result['f136202_mc5_first_verifier'] = {'status': 'ACTUAL_TERMINAL_OUTPUT_VERIFIED',
            'ready_binding':bind(F136_READY), 'terminal_validation':checked}
    elif 'f136202_mc5_first_verifier' in dispositions:
        result['f136202_mc5_first_verifier'] = {'status': 'EXPLICIT_ROOT_MISSINGNESS_DISPOSITION',
            'disposition': dispositions['f136202_mc5_first_verifier']}
    else: raise RuntimeError('REQUIRED_F136_EXACT_FIVE_TERMINAL_OR_EXPLICIT_ROOT_DISPOSITION_MISSING')
    if dispositions: result['terminal_dispositions_binding'] = bind(DISPOSITIONS)
    if orphan_starts():
        # A root must have classified every interrupted semantic attempt before
        # delivery. This coordinator neither reissues it nor calls it successful.
        raise RuntimeError('UNCLASSIFIED_SEMANTIC_START_NO_REPLAY:' + json.dumps(orphan_starts()))
    return result


def command(stage: str) -> list[str]:
    if stage not in STAGES: raise ValueError('Unapproved delivery stage')
    if stage == 'content': return [PYTHON, str(BASE / 'scripts/content_assembly_delivery.py')]
    if stage == 'report': return [PYTHON, str(BASE / 'scripts/finalize_report_delivery.py'), '--write-report']
    if stage.startswith('closeout-'):
        return [PYTHON, str(BASE / 'scripts/closeout_audit_driver.py'), stage.removeprefix('closeout-')]
    return [PYTHON, str(BASE / 'scripts/production_driver.py'), stage, '--invocation-id', 'delivery_v1_' + stage.replace('-', '_')]


def products(stage: str) -> list[Path]:
    names = {
        'content': ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'CONTENT_GATE_FINAL.json', 'CONTENT_FREEZE.json',
                    'CONTENT_ASSEMBLY_DELIVERY_ADAPTATION.json'),
        'cpu-length': ('CPU_LENGTH_PRECHECK.json',),
        'prepare': ('execution/PREPARATION.json', 'execution/PUBLIC_REQUESTS.json', 'execution/PROMPT_TOKEN_IDS.json', 'execution/PRIVATE_SCORING.json'),
        'run': ('execution/STARTED.json', 'execution/RAW_RESPONSE_FREEZE.json'),
        'analyze': ('execution/SCORES.json', 'execution/ANALYSIS_SUMMARY.json'),
        'replication': ('REPLICATION_COMPARISON.json', 'REPLICATION_TABLES_ZH.md', 'REPLICATION_ROOT_DIFFERENCES.csv'),
        'report': ('FINAL_REPORT.json', 'FINAL_REPORT_ZH.md', 'ORIGINAL_INTEGRITY_AFTER.json'),
        'closeout-prepare': ('CLOSEOUT_AUDIT_INPUT.json',),
        'closeout-run': ('INDEPENDENT_CLOSEOUT_AUDIT.json', 'INDEPENDENT_CLOSEOUT_AUDIT_CALL.json', 'INDEPENDENT_CLOSEOUT_AUDIT_METADATA.json'),
        'manifest': ('FINAL_ARTIFACT_MANIFEST.json',), 'validate': ('FINAL_ARTIFACT_MANIFEST.json',),
    }
    return [BASE / name for name in names[stage]]


def checked_products(stage: str) -> dict[str, Any] | None:
    paths = products(stage)
    if not all(path.is_file() for path in paths): return None
    snapshots = []
    for path in paths:
        if path.suffix == '.json':
            doc = read(path)
            # FINAL_REPORT carries historical, separately resolved code identities.
            if stage == 'replication': snapshots.extend(verify_comparison(doc))
            elif stage not in {'report', 'manifest', 'validate', 'closeout-prepare', 'closeout-run'}: verify_nested(doc)
    if stage == 'content':
        frozen = read(BASE / 'CONTENT_FREEZE.json')
        if frozen.get('status') != 'FROZEN_BEFORE_SOLVER': raise RuntimeError('CONTENT_FREEZE_STATUS_DRIFT')
        for name in ('CANDIDATES_PRIVATE.json', 'CONTENT_REVIEW_AI.json', 'SOURCE_SELECTION_FINAL.json', 'SOURCE_FREEZE.json'):
            if bind(BASE / name) not in frozen['bindings']: raise RuntimeError('UNBOUND_FINAL_CONTENT_INPUT:' + name)
    if stage in {'prepare', 'run'}:
        import solver_runtime_adapter as runtime
        runner = runtime.get_runner(); _, rows, tokens = runner.public()
        if stage == 'run': runner.check_raw(read(BASE / 'execution/RAW_RESPONSE_FREEZE.json'), rows, tokens)
    if stage == 'analyze':
        scores = read(BASE / 'execution/SCORES.json')
        if scores.get('raw_freeze_binding') != bind(BASE / 'execution/RAW_RESPONSE_FREEZE.json'): raise RuntimeError('SCORE_RAW_FREEZE_DRIFT')
    if stage.startswith('closeout-'):
        import closeout_audit_driver as audit
        packet = read(BASE / 'CLOSEOUT_AUDIT_INPUT.json')
        if packet.get('report_binding') != bind(BASE / 'FINAL_REPORT.json'): raise RuntimeError('CLOSEOUT_PACKET_REPORT_DRIFT')
        if stage == 'closeout-run':
            verify_closeout_products(audit, packet)
    if stage in {'manifest', 'validate'}:
        validator = module('delivery_existing_manifest_validator', BASE / 'validate_final.py')
        result = validator.validate(BASE)
        if result.get('valid') is not True: raise RuntimeError('FINAL_VALIDATOR_REJECTED_EXISTING_MANIFEST:' + json.dumps(result))
    status = read(paths[0]).get('status') if stage == 'cpu-length' else None
    return {'product_bindings': [bind(path) for path in paths], 'observed_CPU_status': status,
        'historical_mutable_analysis_snapshots': snapshots, 'semantic_acceptance_authored': False}


def closeout_replay_guard(base: Path | None = None) -> bool:
    """No implicit replay or reconstruction after any consumed audit attempt."""
    base = base or BASE
    directory = base / 'cli_calls' / CLOSEOUT_CALL
    if any((directory / name).exists() for name in ('SEMANTIC_FAILURE.json', 'TRANSPORT_EXHAUSTED.json')):
        raise RuntimeError('TERMINAL_CLOSEOUT_FAILURE_NO_REPLAY')
    completed = (directory / 'RESULT.json').is_file()
    if (any(directory.glob('attempt_*/REQUEST.txt')) or any(directory.glob('attempt_*/START.json'))) and not completed:
        raise RuntimeError('CONSUMED_CLOSEOUT_WITHOUT_RESULT_NO_REPLAY')
    return completed


def verify_closeout_products(audit: Any, packet: dict[str, Any]) -> None:
    """Validate saved author-independent audit lineage without calling its helper."""
    closeout_replay_guard()
    actual = read(BASE / 'INDEPENDENT_CLOSEOUT_AUDIT.json')
    meta = read(BASE / 'INDEPENDENT_CLOSEOUT_AUDIT_METADATA.json')
    receipt = read(BASE / 'INDEPENDENT_CLOSEOUT_AUDIT_CALL.json')
    saved_path = BASE / 'cli_calls' / CLOSEOUT_CALL / 'RESULT.json'
    if not saved_path.is_file() or receipt != read(saved_path): raise RuntimeError('CLOSEOUT_ACTUAL_CALL_RECEIPT_MISSING_OR_DRIFT')
    verify_nested(meta); verify_nested(receipt)
    if (meta.get('schema_version') != 'fgr20_closeout_audit_call_metadata_v1'
            or meta.get('call_id') != CLOSEOUT_CALL or meta.get('schema_and_lineage_errors') != []
            or meta.get('report_binding') != packet['report_binding']
            or meta.get('audit_binding') != bind(BASE / 'INDEPENDENT_CLOSEOUT_AUDIT.json')
            or meta.get('audit_input_binding') != bind(BASE / 'CLOSEOUT_AUDIT_INPUT.json')
            or meta.get('call_receipt_binding') != bind(BASE / 'INDEPENDENT_CLOSEOUT_AUDIT_CALL.json')
            or meta.get('reviewer_response_preserved_without_semantic_edits') is not True
            or meta.get('closeout_existing_artifacts_only') is not True
            or meta.get('semantic_verdict_authored_by_manager') is not False):
        raise RuntimeError('CLOSEOUT_SAVED_METADATA_INVALID')
    errors = audit.audit_errors(actual, packet['report_binding']['sha256'])
    if errors: raise RuntimeError('CLOSEOUT_SAVED_AUDIT_SCHEMA_INVALID:' + json.dumps(errors))
    if read(receipt['parsed_response']['path']) != actual: raise RuntimeError('CLOSEOUT_PARSED_RESPONSE_DRIFT')
    response_path = BASE / 'INDEPENDENT_CLOSEOUT_AUDIT_RESPONSE.json'
    if not response_path.is_file() or read(response_path) != actual: raise RuntimeError('CLOSEOUT_RESPONSE_PROJECTION_MISSING_OR_DRIFT')
    backend = receipt.get('backend', {})
    if (not all(backend.get(key) for key in ('thread_id', 'model', 'model_provider', 'effort', 'cli_version'))
            or backend.get('model_override') is not None or meta.get('backend') != backend
            or receipt.get('response_received') is not True):
        raise RuntimeError('CLOSEOUT_ACTUAL_BACKEND_LINEAGE_INVALID')
    terminal_path = Path(receipt['terminal_binding']['path']); terminal = read(terminal_path)
    start = read(terminal_path.with_name('START.json'))
    if (terminal.get('backend') != backend or terminal.get('returncode') != 0
            or terminal.get('stopped') is not False or terminal.get('timed_out') is not False
            or start.get('fresh_session') is not True or start.get('tool_calls_permitted') is not False
            or start.get('request') != receipt.get('request')):
        raise RuntimeError('CLOSEOUT_FRESH_START_TERMINAL_INVALID')
    classification = audit.q.classify_cli_events(Path(receipt['events']['path']), terminal)
    saved_classification = receipt.get('event_classification')
    if not isinstance(saved_classification, dict) or any(saved_classification.get(k) != v for k, v in classification.items()):
        raise RuntimeError('CLOSEOUT_ACTUAL_EVENTS_CLASSIFICATION_DRIFT')
    raw_text = Path(receipt['raw_response']['path']).read_text().strip()
    messages = [row.get('item', {}).get('text') for row in
        (json.loads(line) for line in Path(receipt['events']['path']).read_text().splitlines())
        if row.get('type') == 'item.completed' and row.get('item', {}).get('type') == 'agent_message']
    if not messages or not isinstance(messages[-1], str) or messages[-1].strip() != raw_text:
        raise RuntimeError('CLOSEOUT_RAW_DIFFERS_FROM_FINAL_COMPLETED_MESSAGE')
    if raw_text.startswith(chr(96)*3) and raw_text.endswith(chr(96)*3): raw_text = '\n'.join(raw_text.splitlines()[1:-1])
    if json.loads(raw_text) != actual: raise RuntimeError('CLOSEOUT_RAW_RESPONSE_PARSE_DRIFT')
    prompt = audit.prompt_for(packet)
    if receipt['request']['sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
        raise RuntimeError('CLOSEOUT_SAVED_PROMPT_DIFFERS_FROM_EXACT_FROZEN_INPUT')
    for path in sorted((BASE / 'cli_calls').glob('*/RESULT.json')) + sorted((BASE / 'cli_calls').glob('*/attempt_*/TERMINAL.json')):
        if CLOSEOUT_CALL not in path.parts and read(path).get('backend', {}).get('thread_id') == backend['thread_id']:
            raise RuntimeError('CLOSEOUT_THREAD_REUSED_FROM_PRIOR_CALL')


def production_receipts(stage: str) -> list[dict[str, Any]]:
    if stage.startswith('closeout-'): return []
    rows = []
    for path in sorted((BASE / 'worker_events/production_driver').glob('*/START.json')):
        doc = read(path)
        if doc.get('identity', {}).get('stage') != stage: continue
        if doc.get('command') != command(stage)[:2] + [stage]:
            # production START records the actual fixed inner command, rather
            # than this coordinator's invocation of the production wrapper.
            import production_driver as production
            from argparse import Namespace
            args = Namespace(stage=stage, root=None, arm=None, selection=BASE/'SOURCE_SELECTION_FINAL.json', source_freeze=BASE/'SOURCE_FREEZE.json')
            if doc.get('command') != production.command_for(args): raise RuntimeError('PRODUCTION_STAGE_COMMAND_DRIFT:' + str(path))
        verify(doc['driver_binding']); rows.append({'start_binding': bind(path),
            'result_binding': bind(path.with_name('RESULT.json')) if path.with_name('RESULT.json').exists() else None})
    for row in rows:
        if row['result_binding']: verify_nested(read(row['result_binding']['path']))
    return rows


def descendants(pid: int) -> list[tuple[int, str]]:
    all_info = {int(p.name): process_info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    result, queue = [], [pid]
    while queue:
        parent = queue.pop(0)
        children = [(child, info[1]) for child, info in all_info.items() if info and info[0] == parent]
        result.extend(children); queue.extend(child for child, _ in children)
    return result


def stop_owned(child: subprocess.Popen, token: str | None, registry_before: set[str]) -> list[dict[str, Any]]:
    owned = descendants(child.pid)
    if token is not None: owned.append((child.pid, token))
    for row in live_calls():
        if (row['registry_path'] not in registry_before and row.get('task_directory') == str(BASE)
                and row.get('call_id') == CLOSEOUT_CALL): owned.append((row['pid'], str(row['start_token'])))
    owned = list(dict.fromkeys(owned)); actions = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid, start in reversed(owned):
            info = process_info(pid)
            if info and info[1] == start:
                try: os.kill(pid, sig); actions.append({'pid': pid, 'start_token': start, 'signal': sig.name})
                except ProcessLookupError: pass
        if sig == signal.SIGTERM:
            until = time.monotonic() + 3
            while child.poll() is None and time.monotonic() < until: time.sleep(.1)
    try: child.wait(timeout=5)
    except subprocess.TimeoutExpired: raise RuntimeError('OWNED_CHILD_FAILED_TO_TERMINATE')
    return actions


def recorded_stage_result(target: Path) -> dict[str, Any] | None:
    path = target / 'RESULT.json'
    if not path.exists(): return None
    value = read(path); verify_nested(value); return value


def verify_launch_inputs() -> dict[str, Any]:
    """Recheck immutable engineering and experimental inputs at every launch."""
    ready = read(READY)
    if (ready.get('status') != 'READY_ENGINEERING_ONLY' or ready.get('script_binding') != bind(__file__)
            or ready.get('scope') != 'SETTLED_REPLICATION_DELIVERY_ONLY'
            or ready.get('semantic_calls_started_by_engineering') != 0):
        raise RuntimeError('DELIVERY_ENGINEERING_READY_DRIFT')
    for row in ready['dependency_bindings']: verify(row)
    review_binding = ready['independent_engineering_review_binding']; verify(review_binding)
    review = read(review_binding['path'])
    if (review.get('status') not in {'PASS', 'PASS_ENGINEERING_ONLY'}
            or review.get('script_binding') != bind(__file__)
            or review.get('reviewer_agent') in {None, '/root/solver_preflight_audit'}):
        raise RuntimeError('DELIVERY_INDEPENDENT_ENGINEERING_REVIEW_INVALID')
    verify_nested(read(BASE/'INPUT_LOCK.json'))
    if (EVENTS/'LAUNCH.json').exists(): verify_nested(read(EVENTS/'LAUNCH.json'))
    return ready


def execute_stage(stage: str) -> dict[str, Any]:
    target = EVENTS / 'stages' / stage
    saved = recorded_stage_result(target)
    if saved is not None: return {**saved, 'idempotent_result_reuse': True}
    wait_idle()
    completed_closeout = closeout_replay_guard() if stage == 'closeout-run' else False
    prior_production = production_receipts(stage)
    existing = checked_products(stage)
    if existing is not None:
        result = {'stage': stage, 'status': 'REUSED_VERIFIED_PRODUCTS', 'subprocess_started': False,
                  'exit_code': 2 if existing.get('observed_CPU_status') != 'PASS' and stage == 'cpu-length' else 0,
                  'production_receipts': prior_production, **existing}
        immutable(target / 'RESULT.json', result); return result
    if (target / 'START.json').exists() or any(not row['result_binding'] for row in prior_production):
        raise RuntimeError('INTERRUPTED_STAGE_NO_TERMINAL_NO_REPLAY:' + stage)
    if prior_production: raise RuntimeError('CONSUMED_PRODUCTION_STAGE_WITHOUT_PRODUCTS_NO_REPLAY:' + stage)
    if stage == 'prepare' and any(path.exists() for path in [BASE/'execution/CPU_API_CHECK.json', *products('prepare')]):
        raise RuntimeError('PARTIAL_PREPARATION_NO_REPLAY')
    if stage == 'run' and (BASE / 'execution/STARTED.json').exists(): raise RuntimeError('SOLVER_STARTED_NO_RAW_NO_REPLAY')
    if completed_closeout: raise RuntimeError('SAVED_CLOSEOUT_RESULT_WITHOUT_PRODUCTS_REQUIRES_EXPLICIT_NONCALL_RECOVERY')
    reason = cutoff()
    if reason == 'STOP_REQUEST': raise RuntimeError(reason)
    if reason == 'DEADLINE' and stage == 'closeout-run': raise RuntimeError('NO_FRESH_CLOSEOUT_CALL_AFTER_DEADLINE')
    if reason in {'CLOSEOUT_ONLY', 'DEADLINE'} and stage in NEW_ENGINEERING_STAGES: raise RuntimeError('FINAL20_NO_NEW_SOLVER_OR_PREFLIGHT')
    verify_launch_inputs()
    argv = command(stage); target.mkdir(parents=True, exist_ok=True)
    immutable(target/'START.json', {'at_utc': utc(), 'stage': stage, 'argv': argv, 'script_binding': bind(__file__),
               'automatic_retry': False, 'existing_production_receipts': prior_production})
    before = {str(path) for path in (BASE/'process_registry').glob('*.json')}
    with (target/'STDOUT.log').open('xb') as out, (target/'STDERR.log').open('xb') as err:
        child = subprocess.Popen(argv, cwd=BASE, stdout=out, stderr=err, start_new_session=True)
        info = process_info(child.pid); token = info[1] if info else None
        immutable(target/'PID.json', {'pid': child.pid, 'start_token': token, 'argv': argv, 'task_directory': str(BASE)})
        last = 0.0; termination = []
        try:
            while child.poll() is None:
                reason = cutoff()
                if reason == 'STOP_REQUEST' or (stage in NEW_ENGINEERING_STAGES and reason in {'CLOSEOUT_ONLY', 'DEADLINE'}) or (stage == 'closeout-run' and reason == 'DEADLINE'):
                    termination = stop_owned(child, token, before); break
                if time.monotonic() - last >= HEARTBEAT_SECONDS:
                    heartbeat('running_delivery_stage', stage=stage, child_pid=child.pid, child_start_token=token, cutoff=reason); last = time.monotonic()
                time.sleep(1)
        finally:
            if child.poll() is None: termination.extend(stop_owned(child, token, before))
        code = child.wait()
    observed = checked_products(stage)
    result = {'at_utc': utc(), 'stage': stage, 'status': 'TERMINAL_PRODUCTS_OBSERVED' if observed else 'FAILED_NO_RETRY',
              'exit_code': code, 'subprocess_started': True, 'argv': argv, 'termination_actions': termination,
              'stdout_binding': bind(target/'STDOUT.log'), 'stderr_binding': bind(target/'STDERR.log'),
              'products': observed, 'production_receipts': production_receipts(stage), 'automatic_retry': False}
    immutable(target/'RESULT.json', result); return result


def may_launch(stage: str, solver_blocked: bool, failures: int, base: Path = BASE) -> tuple[bool, str | None]:
    if stage in NEW_ENGINEERING_STAGES and failures >= 3: return False, 'THREE_ENGINEERING_FAILURES_CLOSEOUT_ONLY'
    if stage in {'prepare', 'run'} and (solver_blocked or failures >= 3): return False, 'SOLVER_DISABLED_BY_PRECHECK_OR_ENGINEERING_FAILURES'
    if stage == 'analyze' and not (base/'execution/RAW_RESPONSE_FREEZE.json').exists(): return False, 'NO_FROZEN_RAW_NO_IMPUTATION'
    if stage in {'closeout-prepare', 'closeout-run'} and not (base/'FINAL_REPORT.json').exists(): return False, 'NO_FROZEN_REPORT'
    if stage in {'manifest', 'validate'} and not (base/'INDEPENDENT_CLOSEOUT_AUDIT.json').exists(): return False, 'NO_ACTUAL_CLOSEOUT_AUDIT'
    return True, None


def run() -> int:
    EVENTS.mkdir(parents=True, exist_ok=True)
    with (EVENTS/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_launch_inputs()
        wait_idle(); prereq = prerequisites()
        launch = {'schema_version':'fgr20_delivery_pipeline_launch_v1','script_binding':bind(__file__),
                  'ready_binding':bind(READY),'prerequisites':prereq,'scope':'SETTLED_REPLICATION_DELIVERY_ONLY',
                  'training':False,'rerun_original_four':False,'no_new_source_question_Gold_or_content_verifier':True}
        immutable(EVENTS/'LAUNCH.json',launch)
        outputs=[]; failures=0; solver_blocked=False; new_work_disabled=False
        for stage in STAGES:
            reason=cutoff()
            if reason=='STOP_REQUEST': break
            allowed,why=may_launch(stage,solver_blocked,3 if new_work_disabled else failures)
            if reason in {'CLOSEOUT_ONLY','DEADLINE'} and stage in NEW_ENGINEERING_STAGES:
                allowed,why=False,'FINAL20_NO_NEW_SOLVER_OR_PREFLIGHT'
            if reason=='DEADLINE' and stage=='closeout-run': allowed,why=False,'DEADLINE_NO_NEW_CLOSEOUT_CALL'
            if not allowed:
                outputs.append({'stage':stage,'status':'NOT_STARTED_MISSINGNESS_PRESERVED','reason':why})
                if stage in {'cpu-length','prepare','run'}: solver_blocked=True
                heartbeat('delivery_stage_not_started',stage=stage,reason=why); continue
            try:
                value=execute_stage(stage);outputs.append(value)
                valid=value['status'] in {'REUSED_VERIFIED_PRODUCTS','TERMINAL_PRODUCTS_OBSERVED'}
                engineering_failure=not valid or (value.get('exit_code',0)!=0 and not (stage=='cpu-length' and value.get('exit_code')==2))
                failures=failures+1 if engineering_failure else 0
                if stage=='cpu-length':
                    observed=checked_products(stage);solver_blocked=not observed or observed.get('observed_CPU_status')!='PASS'
                if stage=='prepare' and not checked_products(stage):solver_blocked=True
                if stage=='run' and not checked_products(stage):solver_blocked=True
            except Exception as exc:
                failures+=1
                value={'stage':stage,'status':'FAILED_WITHOUT_RETRY','error':f'{type(exc).__name__}: {exc}'}
                outputs.append(value)
                immutable(EVENTS/'stages'/stage/'FAILURE_CLASSIFICATION.json',value)
                if stage in {'content','cpu-length','prepare','run'}:solver_blocked=True
            if failures >= 3: new_work_disabled=True
            heartbeat('delivery_stage_terminal',stage=stage,consecutive_engineering_failures=failures,solver_blocked=solver_blocked,new_generation_solver_and_engineering_disabled=new_work_disabled)
        summary={'schema_version':'fgr20_delivery_pipeline_handoff_v1','steps':outputs,'cutoff':cutoff(),
                 'status':'VALIDATED_DELIVERY' if outputs and outputs[-1]['stage']=='validate' and outputs[-1]['status'] in {'REUSED_VERIFIED_PRODUCTS','TERMINAL_PRODUCTS_OBSERVED'} else 'PARTIAL_DELIVERY_HANDOFF',
                  'consecutive_engineering_failures':failures,'new_generation_solver_and_engineering_disabled':new_work_disabled,
                  'only_once_existing_frozen_report_closeout_remains_authorized':True,'training':False,'rerun_original_four':False,
                 'semantic_acceptance_authored':False,'response_imputation_performed':False}
        run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        immutable(EVENTS/('HANDOFF_'+run_id+'.json'),summary);heartbeat('delivery_handoff',summary=summary)
        return 0 if summary['status']=='VALIDATED_DELIVERY' else 2


def self_check() -> dict[str, Any]:
    assert len(STAGES)==11 and len(set(STAGES))==11
    assert all('question_driver.py' not in command(s) and 'f179186_saved_shape_gold_verification.py' not in command(s) for s in STAGES)
    assert [command(s)[-1] for s in ('closeout-prepare','closeout-run')]==['prepare','run']
    assert scoped_script([PYTHON,'scripts/review_schema_adapter_v2.py','launch'],BASE)==BASE/'scripts/review_schema_adapter_v2.py'
    assert scoped_script([PYTHON,'scripts/resume_authorized_pipeline.py','run'],BASE)==BASE/'scripts/resume_authorized_pipeline.py'
    assert scoped_script([PYTHON,'/tmp/review_schema_adapter_v2.py'],BASE) is None
    assert scoped_script([PYTHON,str(BASE/'supervisor.py')],BASE) is None
    assert scoped_script([PYTHON,str(BASE/'scripts/status_heartbeat.py')],BASE) is None
    assert scoped_script([PYTHON,'scripts/status_heartbeat.py'],BASE)==None
    cache=BASE/'runtime_cache/delivery_pipeline_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='delivery_selfcheck_',dir=cache) as directory:
        base=Path(directory);end=dt.datetime(2030,1,1,tzinfo=dt.timezone.utc)
        (base/'SUPERVISOR_STATUS.json').write_text(json.dumps({'deadline_at_utc':end.isoformat()}))
        assert cutoff(base,end-dt.timedelta(minutes=46)) is None
        assert cutoff(base,end-dt.timedelta(minutes=45))=='NO_NEW_CONTENT_GENERATION'
        (base/'FINALIZE_REQUEST').touch()
        assert cutoff(base,end-dt.timedelta(minutes=20))=='CLOSEOUT_ONLY'
        assert cutoff(base,end)=='DEADLINE'
        (base/'STOP_REQUEST').touch();assert cutoff(base)=='STOP_REQUEST'
        assert may_launch('run',True,0,base)[0] is False
        assert may_launch('cpu-length',False,3,base)[0] is False
        assert may_launch('prepare',False,3,base)[0] is False
        assert may_launch('closeout-run',False,3,base)[0] is False
        (base/'FINAL_REPORT.json').write_text('{}')
        assert may_launch('closeout-run',False,3,base)[0] is True
        assert may_launch('analyze',False,0,base)==(False,'NO_FROZEN_RAW_NO_IMPUTATION')
        (base/'execution').mkdir();(base/'execution/RAW_RESPONSE_FREEZE.json').write_text('{}')
        assert may_launch('analyze',True,3,base)[0] is True
        immutable(base/'receipt.json',{'mock':True});verify(bind(base/'receipt.json'))
        try:immutable(base/'receipt.json',{'mock':False})
        except RuntimeError:pass
        else:raise AssertionError('receipt overwrite')
        # Mutable progress snapshots are recognized only in the exact direct
        # input field; scientific inputs and same-name foreign paths stay strict.
        status=base/'TASK_STATUS.json';status.write_text('before');historical=bind(status);status.write_text('after')
        stable=base/'stable.json';stable.write_text('{}');stable_binding=bind(stable)
        snapshots=verify_comparison({'input_bindings':[historical,stable_binding]},base)
        assert len(snapshots)==1 and snapshots[0]['snapshot_sha256']==historical['sha256']
        verify_nested(snapshots)
        for malformed in ({'other':[historical]}, {'input_bindings':[dict(stable_binding,sha256='0'*64)]}):
            try:verify_comparison(malformed,base)
            except RuntimeError:pass
            else:raise AssertionError('Scientific/nested binding drift was exempted')
        call=base/'cli_calls'/CLOSEOUT_CALL;attempt=call/'attempt_01';attempt.mkdir(parents=True)
        start=attempt/'START.json';start.write_text('{}')
        try:closeout_replay_guard(base)
        except RuntimeError as exc:assert 'NO_REPLAY' in str(exc)
        else:raise AssertionError('START-only closeout replayed')
        start.unlink();request=attempt/'REQUEST.txt';request.write_text('fixture')
        try:closeout_replay_guard(base)
        except RuntimeError as exc:assert 'NO_REPLAY' in str(exc)
        else:raise AssertionError('REQUEST-only closeout replayed')
        (call/'RESULT.json').write_text('{}');assert closeout_replay_guard(base) is True
        for name in ('SEMANTIC_FAILURE.json','TRANSPORT_EXHAUSTED.json'):
            marker=call/name;marker.write_text('{}')
            try:closeout_replay_guard(base)
            except RuntimeError as exc:assert 'TERMINAL_CLOSEOUT_FAILURE' in str(exc)
            else:raise AssertionError('Terminal closeout failure replayed')
            marker.unlink()
        # Exercise the actual existing audit schema/event validators using only
        # synthetic in-memory/file fixtures. No helper.call or child is reachable.
        import closeout_audit_driver as audit
        audit_base=base/'audit_fixture';attempt=audit_base/'cli_calls'/CLOSEOUT_CALL/'attempt_01';attempt.mkdir(parents=True)
        report=audit_base/'FINAL_REPORT.json';report.write_text('{"fixture":true}')
        packet={'report_binding':bind(report)};packet_path=audit_base/'CLOSEOUT_AUDIT_INPUT.json'
        packet_path.write_text(json.dumps(packet));prompt=audit.prompt_for(packet)
        response={'verdict':'FAIL','report_sha256':bind(report)['sha256'],'independent_of_report_author':True,
            'checks':[{'fixture':'Engineering schema only; no real review.'}],'issues':['fixture']}
        parsed=audit_base/'cli_calls'/CLOSEOUT_CALL/'PARSED_RESPONSE.json';parsed.write_text(json.dumps(response))
        raw=attempt/'LAST_MESSAGE.txt';raw.write_text(json.dumps(response))
        request=attempt/'REQUEST.txt';request.write_text(prompt)
        stderr=attempt/'STDERR.log';stderr.write_text('')
        backend={'thread_id':'engineering-fixture-thread','model':'fixture','model_provider':'fixture',
            'effort':'fixture','cli_version':'fixture','model_override':None}
        usage={'input_tokens':1,'output_tokens':1}
        event_rows=[{'type':'thread.started','thread_id':backend['thread_id']},
            {'type':'item.completed','item':{'type':'agent_message','text':raw.read_text()}},
            {'type':'turn.completed','usage':usage}]
        events=attempt/'EVENTS.jsonl';events.write_text('\n'.join(json.dumps(row) for row in event_rows)+'\n')
        terminal={'backend':backend,'returncode':0,'stopped':False,'timed_out':False,'request':bind(request),
            'raw_response':bind(raw),'events':bind(events),'stderr':bind(stderr),'usage':usage,'forbidden_tool_items':[]}
        terminal_path=attempt/'TERMINAL.json';terminal_path.write_text(json.dumps(terminal))
        (attempt/'START.json').write_text(json.dumps({'fresh_session':True,'tool_calls_permitted':False,'request':bind(request)}))
        receipt={**terminal,'terminal_binding':bind(terminal_path),'parsed_response':bind(parsed),'response_received':True,
            'event_classification':audit.q.classify_cli_events(events,terminal)}
        (attempt.parent/'RESULT.json').write_text(json.dumps(receipt))
        call_path=audit_base/'INDEPENDENT_CLOSEOUT_AUDIT_CALL.json';call_path.write_text(json.dumps(receipt))
        actual=audit_base/'INDEPENDENT_CLOSEOUT_AUDIT.json';actual.write_text(json.dumps(response))
        (audit_base/'INDEPENDENT_CLOSEOUT_AUDIT_RESPONSE.json').write_text(json.dumps(response))
        meta={'schema_version':'fgr20_closeout_audit_call_metadata_v1','call_id':CLOSEOUT_CALL,
            'audit_binding':bind(actual),'report_binding':bind(report),'audit_input_binding':bind(packet_path),
            'call_receipt_binding':bind(call_path),'backend':backend,'schema_and_lineage_errors':[],
            'reviewer_response_preserved_without_semantic_edits':True,'closeout_existing_artifacts_only':True,
            'semantic_verdict_authored_by_manager':False}
        metadata=audit_base/'INDEPENDENT_CLOSEOUT_AUDIT_METADATA.json';metadata.write_text(json.dumps(meta))
        actual_base=globals()['BASE']
        try:
            globals()['BASE']=audit_base
            verify_closeout_products(audit,packet)
            bad={**meta,'schema_and_lineage_errors':['fixture_invalid']};metadata.write_text(json.dumps(bad))
            try:verify_closeout_products(audit,packet)
            except RuntimeError as exc:assert 'METADATA_INVALID' in str(exc)
            else:raise AssertionError('Malformed saved audit lineage admitted')
        finally:globals()['BASE']=actual_base
        # Exercise no-replay and STOP ownership without starting a process or
        # delivering a signal. All patched globals are restored before return.
        names=('EVENTS','wait_idle','production_receipts','checked_products','descendants','live_calls','process_info')
        original={name:globals()[name] for name in names};kill=os.kill
        try:
            globals()['EVENTS']=base/'events'
            globals()['wait_idle']=lambda:None
            globals()['production_receipts']=lambda stage:[]
            globals()['checked_products']=lambda stage:{'product_bindings':[],'observed_CPU_status':None}
            reused=execute_stage('content');assert reused['status']=='REUSED_VERIFIED_PRODUCTS'
            assert execute_stage('content')['idempotent_result_reuse'] is True
            globals()['checked_products']=lambda stage:None
            globals()['production_receipts']=lambda stage:[{'start_binding':{'fixture':True},'result_binding':None}]
            try:execute_stage('replication')
            except RuntimeError as exc:assert 'NO_REPLAY' in str(exc)
            else:raise AssertionError('Production START without RESULT replayed')
            globals()['production_receipts']=lambda stage:[]
            target=globals()['EVENTS']/'stages'/'report';target.mkdir(parents=True);(target/'START.json').write_text('{}')
            try:execute_stage('report')
            except RuntimeError as exc:assert 'NO_REPLAY' in str(exc)
            else:raise AssertionError('Delivery START without RESULT replayed')
            globals()['descendants']=lambda pid:[(901,'t901')]
            globals()['live_calls']=lambda:[{'registry_path':'new','task_directory':str(BASE),'call_id':CLOSEOUT_CALL,'pid':902,'start_token':'t902'},
                {'registry_path':'old','task_directory':str(BASE),'call_id':CLOSEOUT_CALL,'pid':903,'start_token':'t903'},
                {'registry_path':'foreign','task_directory':'/foreign','call_id':CLOSEOUT_CALL,'pid':904,'start_token':'t904'},
                {'registry_path':'othercall','task_directory':str(BASE),'call_id':'unrelated','pid':905,'start_token':'t905'}]
            globals()['process_info']=lambda pid:(1,'t'+str(pid))
            signals=[];os.kill=lambda pid,sig:signals.append((pid,sig.name))
            class Child:
                pid=900
                def poll(self):return 0
                def wait(self,**kwargs):return 0
            stop_owned(Child(),'t900',{'old'})
            assert {pid for pid,_ in signals}=={900,901,902}
            assert len(signals)==6 and all(pid not in {903,904,905} for pid,_ in signals)
        finally:
            for name,value in original.items():globals()[name]=value
            os.kill=kill
    return {'status':'PASS','semantic_calls_started':0,'solver_calls_started':0,'subprocesses_started':0,
            'mock_only':True,'stages':list(STAGES),'commands':[command(s) for s in STAGES],
            'checks':['finite exact command allowlist','no source question Gold or F179 launch','v2 and serial7 cwd-scoped parent recognition',
                       'exact supervisor launch and status heartbeat management excluded','STOP 45min 20min deadline precedence','no raw no zero-imputation analyze',
                       'CPU failure Solver gating','three failures prevent new CPU preflight and Solver; unique existing-report closeout remains allowed','immutable receipt no overwrite',
                       'exact-root direct mutable analysis snapshots disclosed without scientific exemptions',
                       'closeout REQUEST or START without RESULT and terminal failures no replay',
                       'actual saved audit schema exact report prompt backend fresh events and response lineage',
                      'existing products/results reused without subprocess','delivery and production START without RESULT no replay',
                      'STOP exact child descendants and new closeout call only; prior foreign unrelated processes excluded']}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['describe','self-check','run']);args=parser.parse_args()
    if args.command=='self-check':print(json.dumps(self_check(),ensure_ascii=False,indent=2));return 0
    if args.command=='describe':
        print(json.dumps({'scope':'SETTLED_REPLICATION_DELIVERY_ONLY','stages':[{'stage':s,'argv':command(s)} for s in STAGES],
            'requires_actual_serial7_and_F179_terminal_or_explicit_root_dispositions':True,'root_terminal_disposition_path':str(DISPOSITIONS),
            'automatic_semantic_or_engineering_retry':False,'heartbeat_seconds':HEARTBEAT_SECONDS},ensure_ascii=False,indent=2));return 0
    def interrupted(signum,frame):
        global STOPPING
        STOPPING=True
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:return run()
    except Exception as exc:
        heartbeat('delivery_stopped_without_replay',error=f'{type(exc).__name__}: {exc}');return 2


if __name__=='__main__':raise SystemExit(main())
