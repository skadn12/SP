#!/usr/bin/env python3
"""Read-only preflight. A PASS is impossible until history scoping is reviewed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from build_v7 import readj, validate_contract_review, V6


def check_execution(base: Path):
    issues = ['HISTORY_BASELINE_INCOMPLETE: five-dimensional semantic privacy and same-root scoping need independent review']

    def fail(message): issues.append(message)
    def load(path):
        try: return readj(base / path)
        except Exception as exc:
            fail(f'{path}: {exc}')
            return {}
    def check_bytes(path, expected):
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            fail(f'invalid relative path: {path!r}'); return
        target = (base / path).resolve()
        if not target.is_relative_to(base.resolve()):
            fail(f'path outside execution: {path}'); return
        try: actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc: fail(f'{path}: {exc}'); return
        if actual != expected: fail(f'hash mismatch: {path}')

    try: validate_contract_review()
    except Exception as exc: fail(f'contract review: {exc}')
    manifest = load('BUILD_MANIFEST.json')
    for row in manifest.get('artifacts', []): check_bytes(row.get('path'), row.get('sha256'))
    for name in ('SCHEDULE_RESTRICTED.json','SCHEDULE_PUBLIC.json','PRIVATE_BINDINGS.json','PRIVATE_OPPORTUNITY_MAPPING.json','SOURCE_ADMISSION_AUDIT.json'):
        try:
            if (base/name).read_bytes() != (V6/name).read_bytes(): fail(f'{name}: parent bytes changed')
        except OSError as exc: fail(f'{name}: {exc}')
    schedule = load('SCHEDULE_RESTRICTED.json').get('opportunities', [])
    public = load('SCHEDULE_PUBLIC.json').get('opportunities', [])
    packets = load('PACKET_MANIFEST.json').get('packets', [])
    inputs = load('INPUT_MANIFEST.json')
    sources = inputs.get('sources', [])
    if len(schedule) != 32 or len(public) != 32 or len(packets) != 32 or len(sources) != 8:
        fail('schedule, packet or source count mismatch')
    codes = [r.get('opportunity_code') for r in schedule]
    if len(set(codes)) != 32 or set(codes) != {r.get('opportunity_code') for r in packets} or set(codes) != {r.get('opportunity_code') for r in public}:
        fail('schedule opportunity identities mismatch')
    if Counter(r.get('group') for r in schedule) != Counter({'OLD_F':8,'OLD_G':8,'NEW_F':8,'NEW_G':8}): fail('group balance mismatch')
    if len({r.get('source_code') for r in sources}) != 8: fail('source identities mismatch')
    if Counter((r.get('source_code'),r.get('group')) for r in schedule) != Counter({(s.get('source_code'),g):1 for s in sources for g in ('OLD_F','OLD_G','NEW_F','NEW_G')}): fail('source/group assignment mismatch')
    for src in sources:
        check_bytes(src.get('source_path'), src.get('source_sha256'))
        check_bytes(src.get('neutral_path'), src.get('neutral_sha256'))
    context = inputs.get('context_card', {})
    check_bytes(context.get('path'), context.get('sha256'))
    history = inputs.get('historical_projection', {})
    check_bytes(history.get('path'), history.get('sha256'))
    hist = load(history.get('path', 'missing_history'))
    if len(hist.get('coverage_set', [])) != 135 or len(hist.get('overlap_set', [])) != 77 or 'records' in hist:
        fail('history sets or counts do not match v3 contract')
    source_by_id = {r.get('source_code'):r for r in sources}
    schedule_by_id = {r.get('opportunity_code'):r for r in schedule}
    selector_count = 0
    for row in packets:
        code = row.get('opportunity_code'); sched = schedule_by_id.get(code, {})
        source = source_by_id.get(sched.get('source_code'), {})
        check_bytes(row.get('role_input_path'), row.get('role_input_sha256'))
        role = load(row.get('role_input_path', 'missing_role'))
        allowed_role={'schema_version','role','opportunity_code','workflow','source','neutral','contract','schema','output','opportunity_ceiling','slots_ceiling','generation_authorization','context_card'}
        if set(role)-allowed_role or role.get('role') != 'author' or role.get('opportunity_ceiling') != 1 or role.get('slots_ceiling') != (3 if sched.get('workflow') == 'NEW' else 0): fail(f'{code}: author packet boundary')
        if role.get('generation_authorization') != 'NONE' or role.get('workflow') != sched.get('workflow') or role.get('opportunity_code') != code: fail(f'{code}: role identity or authorization')
        expected_group = f"{sched.get('workflow')}_{'F' if sched.get('feedback') == 'CONTEXT_CARD' else 'G'}"
        if sched.get('group') != expected_group or sched.get('feedback') not in ('CONTEXT_CARD','NONE'): fail(f'{code}: feedback schedule mismatch')
        if ('context_card' in role) != (sched.get('feedback') == 'CONTEXT_CARD'): fail(f'{code}: context card assignment')
        if 'context_card' in role and role['context_card'] != {'path':context.get('path'),'sha256':context.get('sha256')}: fail(f'{code}: context card binding')
        for label in ('source','neutral'):
            if role.get(label, {}).get('path') != source.get(label+'_path') or role.get(label, {}).get('sha256') != source.get(label+'_sha256'): fail(f'{code}: {label} binding')
        for label in ('contract','schema'):
            binding = role.get(label, {}); check_bytes(binding.get('path'), binding.get('sha256'))
        if sched.get('workflow') == 'NEW':
            selector_count += 1
            check_bytes(row.get('selector_input_path'), row.get('selector_input_sha256'))
            selector = load(row.get('selector_input_path', 'missing_selector'))
            if selector.get('generation_authorization') != 'NONE' or selector.get('history') != {'path':history.get('path'),'sha256':history.get('sha256')}: fail(f'{code}: selector history or authorization')
            if set(selector) != {'schema_version','role','source','neutral','history','slots','contract','schema','generation_authorization','expected_result_schema'}: fail(f'{code}: selector packet boundary')
            for label in ('source','neutral','contract','schema'):
                binding=selector.get(label, {}); check_bytes(binding.get('path'),binding.get('sha256'))
            for label in ('source','neutral'):
                if selector.get(label) != {'path':source.get(label+'_path'),'sha256':source.get(label+'_sha256')}: fail(f'{code}: selector {label} binding')
            if selector.get('slots', {}).get('sha256') is not None: fail(f'{code}: slots already bound')
        elif 'selector_input_path' in row: fail(f'{code}: unexpected selector')
    if selector_count != 16: fail('selector count mismatch')
    for name in ('AUTHORIZATION.json','AUTHOR_LAUNCH_AUTHORIZATION.json'):
        auth=load(name)
        if auth.get('generation_authorization') != 'NONE' or any(v is not False for v in auth.get('scope', {}).values()) or set(auth.get('scope', {})) != {'author','selector','solver','training','export','retry','fallback'} or auth.get('author_launch_authorized',False) is not False or auth.get('selector_launch_authorized',False) is not False: fail(f'{name}: authorization is not NONE')
    access=load('ACCESS_RULES.json')
    private={'INPUT_MANIFEST.json','PACKET_MANIFEST.json','BUILD_MANIFEST.json','PROTOCOL_FREEZE.json','AUTHORIZATION.json','AUTHOR_LAUNCH_AUTHORIZATION.json','CONTROLLER_HISTORY_BINDINGS.json','PRIVATE_BINDINGS.json','PRIVATE_OPPORTUNITY_MAPPING.json','SCHEDULE_RESTRICTED.json','SOURCE_ADMISSION_AUDIT.json','SELECTOR_OPERATION_SUMMARY.json'}
    if not private.issubset(set(access.get('controller_only_roots', []))): fail('controller-only boundaries incomplete')
    if any((base / name).exists() for name in ('author_output','selector_output','solver')): fail('output directory already exists')
    return {'schema_version':'stage_b_v7_strict_preflight_v1','all_pass':not issues,'issues':issues}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('execution', type=Path); args=parser.parse_args()
    result=check_execution(args.execution)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['all_pass'] else 1


if __name__ == '__main__': sys.exit(main())
