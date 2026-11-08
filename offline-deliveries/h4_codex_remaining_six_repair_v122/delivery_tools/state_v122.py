"""Read-only common terminal classification for the fixed six-root cohort."""
from pathlib import Path
import hashlib
import json
import sys

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / 'tools'))
import scene_controller_v120 as scene

TERMINAL = {'SOURCE_HOLD', 'PREFIX_HOLD', 'SCENE_HOLD', 'QA_FIRST_ATTEMPT_FAILED',
            'QA_REVIEW_SCHEMA_HOLD', 'QA_PIPELINE_HOLD', 'FINALIZED'}


def read(path): return json.loads(Path(path).read_text())
def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def classify(row, require_terminal=False):
    unit = Path(row['unit_path']); folder = unit / 'roots' / unit.name
    out = {'root_id': row['root_id'], 'source_status': row.get('effective_source_status', row.get('source_status', 'PENDING')),
           'effective_source_status': row.get('effective_source_status', row.get('source_status', 'PENDING')),
           'local_patch_status': row.get('local_patch_status', row.get('source_status', 'PENDING'))}
    source = unit / 'source_repair/FINALIZE_RECEIPT.json'
    if not source.exists():
        out['status'] = 'SOURCE_PENDING'
    elif read(source)['status'] == 'SOURCE_HOLD' or row.get('source_ready') is False:
        out['status'] = 'SOURCE_HOLD'
    elif not row.get('prefix_reuse_approved'):
        out['status'] = 'PREFIX_HOLD'
    elif not (unit / 'RUN_CONFIG.json').exists():
        out['status'] = 'MATCHING_PENDING'
    else:
        state = scene.inspect_root(unit, scene.load_config(unit), row['root_id'], record_reviews=False)
        out['scene_state'] = state
        if state['root_stopped']:
            out['status'] = 'SCENE_HOLD'
        elif (folder / 'QA_AUTHOR_FAILURE.json').exists():
            out.update(status='QA_FIRST_ATTEMPT_FAILED', failure_binding=bind(folder / 'QA_AUTHOR_FAILURE.json'))
        elif (folder / 'QA_REVIEW_FAILURE.json').exists():
            failure = read(folder / 'QA_REVIEW_FAILURE.json')
            if failure.get('terminal') is not True or (folder / 'PRIVATE_GOLD.json').exists():
                raise ValueError('INVALID_TERMINAL_QA_REVIEW_FAILURE:' + row['root_id'])
            for key in ('original_raw_binding', 'save_failure_binding', 'corrected_payload_binding'):
                if failure.get(key) and bind(failure[key]['path']) != failure[key]:
                    raise ValueError('QA_REVIEW_FAILURE_BINDING_CHANGED:' + key)
            out.update(status='QA_REVIEW_SCHEMA_HOLD', failure_binding=bind(folder / 'QA_REVIEW_FAILURE.json'))
        elif (folder / 'ROOT_TERMINAL.json').exists():
            failure = read(folder / 'ROOT_TERMINAL.json')
            status = failure.get('status')
            if failure.get('root_id') != row['root_id'] or failure.get('terminal') is not True or status not in TERMINAL - {'FINALIZED', 'SOURCE_HOLD', 'PREFIX_HOLD'}:
                raise ValueError('INVALID_ROOT_TERMINAL:' + row['root_id'])
            if (folder / 'PRIVATE_GOLD.json').exists():
                raise ValueError('GOLD_AFTER_ROOT_TERMINAL:' + row['root_id'])
            out.update(status=status, failure_binding=bind(folder / 'ROOT_TERMINAL.json'))
        elif not state['full_expansion_complete']:
            out['status'] = 'SCENES_PENDING'
        elif (folder / 'FINAL_CROSSCHECK.json').exists():
            cross = read(folder / 'FINAL_CROSSCHECK.json')
            out.update(status='FINALIZED', content_pass=sum(q['status'] == 'CONTENT_PASS' for q in cross['results']),
                       final_crosscheck_binding=bind(folder / 'FINAL_CROSSCHECK.json'))
        else:
            out['status'] = 'QA_PENDING'
    if require_terminal and out['status'] not in TERMINAL:
        raise ValueError('ROOT_NOT_TERMINAL:' + row['root_id'] + ':' + out['status'])
    return out
