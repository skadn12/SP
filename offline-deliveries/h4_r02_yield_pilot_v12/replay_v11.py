"""Offline compatibility diagnostic; never promotes or rewrites historical data."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path

import event_state
import realization_review


def read(path):
    return json.loads(path.read_text())


def fingerprints(run):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.rglob('*')) if p.is_file()}


def context(payload, marker):
    text = payload['messages'][0]['content'].split(marker, 1)[1]
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def replay(run, output):
    if output.exists() or run.resolve() in output.resolve().parents:
        raise ValueError('output_must_be_new_and_outside_historical_run')
    before = fingerprints(run)
    records, stages = [], defaultdict(Counter)
    for folder in sorted((run / 'requests').iterdir()):
        proof = read(folder / 'PROOF.json')
        role = proof.get('role', 'dataset_extractor')
        response = read(folder / 'RESPONSE.json')
        metadata = response['metadata']
        stages[role].update(calls=1, input_tokens=metadata.get('input_tokens', 0),
                            output_tokens=metadata.get('output_tokens', 0))
        payload = read(folder / 'REQUEST.json')
        row = {'request': str(folder), 'role': role,
               'raw_response_sha256': before[str(folder / 'RESPONSE.json')]}
        if role == 'event_state_extractor':
            ctx = context(payload, '\nEVENT_STATE_INPUT_JSON\n')
            parsed = event_state._parse(response['text'])
            projected = deepcopy(parsed)
            # Explicit schema projection only. Deleted ref declarations remain
            # available in the original artifact and below for diagnostic use.
            removed = []
            for segment in projected['segments']:
                if 'relevant_turn_refs' in segment:
                    removed.append({'segment_id': segment['segment_id'],
                                    'relevant_turn_refs': segment.pop('relevant_turn_refs')})
            row.update(original_validation=read(folder / 'VALIDATION.json'),
                       schema_projection='remove_duplicate_segments.relevant_turn_refs_only',
                       removed_duplicate_declarations=removed)
            try:
                checked = event_state.validate(projected, ctx['root_id'], ctx['route'],
                                               ctx['scene_plan'], ctx['cumulative_history'])
            except (ValueError, TypeError) as error:
                row.update(v12_schema_status='REJECT', v12_errors=getattr(error, 'errors', [str(error)]))
            else:
                assert checked == projected
                row.update(v12_schema_status='PASS', requires_replan=event_state.requires_replan(checked),
                           derived_coverage=event_state.derive_coverage(checked, ctx['cumulative_history']))
        elif role == 'scene_reviewer':
            ctx = context(payload, '\nREVIEW INPUT DATA:\n')
            verdict = realization_review.validate_review(response['text'], ctx['root_id'], ctx['route'],
                ctx['scene_plan'], ctx['new_scene'], dataset_card=ctx.get('dataset_card_guidance'),
                latest_state=ctx.get('latest_state'), event_state=ctx.get('event_state'),
                history=ctx['branch_history'])
            row.update(original_review=read(folder / 'REVIEW.json'), v12_review=verdict)
        else:
            continue
        records.append(row)
    assert fingerprints(run) == before, 'historical_files_changed'
    scans = [r for r in records if r['role'] == 'event_state_extractor']
    result = {
        'version': 'v12_offline_v11_schema_compatibility_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'historical_run': str(run), 'historical_files_unchanged': True, 'http_calls': 0,
        'promoted_scenes': 0, 'training_calls': 0, 'records': records,
        'original_stage_cost': dict(stages),
        'original_event_schema_pass': sum(r['original_validation']['status'] == 'PASS' for r in scans),
        'projected_v12_event_schema_pass': sum(r['v12_schema_status'] == 'PASS' for r in scans),
        'projected_v12_replan': sum(r.get('requires_replan') is True for r in scans),
        'projected_v12_compatible': sum(r.get('requires_replan') is False for r in scans),
        'limitation': 'Offline schema compatibility of already-generated answers only. Changed prompts were not tested here. No old rejection is replaced and no scene or question is accepted by this diagnostic.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {k: v for k, v in result.items() if k != 'records'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.run.resolve(), args.out.resolve()), ensure_ascii=False, indent=2))
