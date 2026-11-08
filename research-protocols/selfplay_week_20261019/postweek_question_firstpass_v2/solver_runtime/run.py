"""QFP2 prospective authoring trial on the frozen FBC1 B1 runtime.

The imported runtime retains its historical internal cohort string FBC1. New
paths, public opaque IDs and per-request seeds bind this experiment separately.
There is no dependency on old solver responses or old sampling outcomes.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PARENT = BASE.parent / 'postweek_feedback_control_4root_v1'
spec = importlib.util.spec_from_file_location('qfp2_frozen_b1_runtime', PARENT / 'solver_runtime/run.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
original_code_paths = runtime.code_paths
runtime.HERE = HERE
runtime.EXPERIMENT = BASE


def sampling_seed(question, branch):
    material = 'qfp2:firstpass:20261024:' + question['candidate_id'] + ':' + branch
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], 'big') % 2147483647


def code_paths():
    return original_code_paths() + [Path(__file__), BASE / 'PROTOCOL_FREEZE.json',
                                  BASE.parents[2] / 'scripts/current_method/question_presentation_gate_v2.py']


runtime.sampling_seed = sampling_seed
runtime.code_paths = code_paths


def prepare(args):
    runtime.prepare(args)
    checks = []
    for row in runtime.read(args.output / 'PUBLIC_REQUESTS.json')['requests']:
        prompt = Path(row['prompt_path']).read_text()
        assert row['candidate_id'] not in prompt and row['request_id'] not in prompt
        assert row['task_operation'] != 'MC_RECORD_AUDIT_CONTROL'
        assert 'MC_RECORD_AUDIT_CONTROL' not in prompt and 'all-correct' not in prompt.lower()
        checks.append({'request_id': row['request_id'], 'prompt_binding': runtime.bind(row['prompt_path']),
                       'opaque_id_absent': True, 'control_label_absent': True})
    runtime.save(args.output / 'PUBLIC_PRESENTATION_CHECK.json', {
        'status': 'PASS_RENDERED_PROMPTS_ONLY', 'checks': checks,
        'runtime_binding': runtime.bind(__file__), 'historical_internal_cohort': 'FBC1'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'verify', 'run', 'analyze'])
    ap.add_argument('--export-freeze', type=Path, default=BASE / 'inputs/EXPORT_FREEZE.json')
    ap.add_argument('--source-root', type=Path, default=PARENT / 'sources')
    ap.add_argument('--output', type=Path, default=HERE / 'execution')
    ap.add_argument('--endpoint', type=Path, default=runtime.ENDPOINT)
    ap.add_argument('--preflight-review', type=Path, default=HERE / 'PREFLIGHT_REVIEW.json')
    ap.add_argument('--execute-gpu', action='store_true')
    args = ap.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'verify':
        prep, rows, tokens = runtime.verify(args)
        print({'status': 'PUBLIC_INPUTS_VERIFIED', 'requests': len(rows)})
    else:
        getattr(runtime, args.action)(args)
