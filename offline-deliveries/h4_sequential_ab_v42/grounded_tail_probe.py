"""One grounded generation diagnosis using an already consumed v41 source."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
V41 = HERE.parent / 'h4_sequential_ab_v41'
sys.path.insert(0, str(V41))
import candidate_runner as c
from candidate_transport import Caller

SOURCE = V41 / 'runs/fresh_b2_candidates_01'
PREVIOUS = V41 / 'diagnostics/prompt_tail_b2_s01_01'
ROOT, ROUTE, SCENE = c.ROOT, c.r.ROUTES[1], 'S01'


def make_request(run):
    inputs = c.r.base.read(run / 'INPUT.json')
    history, card = inputs['history'], inputs['source_card']
    guidance = c.r.base.dataset_cards.build_realization_guidance(card)
    request, proof = c.request(run, ROOT, ROUTE, SCENE, 'realizer', history, history, guidance)
    before_tail_hash = c.r.base.digest(request)
    target = {**c.identity(ROOT, ROUTE, SCENE), 'dataset_card_id': card['card_id'],
              'actor_mapping': {'actor_1': 'Emily', 'actor_2': 'James'},
              'allowed_speakers': sorted({t['speaker'] for t in history}),
              'source_card_with_original_evidence': card}
    tail = '''

CURRENT TASK AFTER ALL HISTORICAL DATA AND FROZEN FEEDBACK
Write a NEW dataset_new_event conversation after the complete history above.
Only the following target identity applies to your output. Earlier scene numbers
are historical data. Use the exact target scene and source card IDs below.
The full history is authoritative. Introduce this new occasion now; do not claim
that these people previously discussed a college search unless the history says
so. Do not replay or paraphrase the completed historical workshop/Friday farewell
scene. Preserve other commitments and distinguish plans from completed actions.

Use the original quoted evidence in this card to understand its interaction.
The subject is information about VISITING a college, including visitor admission
and the postcode. Free visitor admission is not free tuition, a management course,
enrollment, a scholarship, or a training application. Names and compatible new
details may be adapted; keep the purpose, actor/action/object and action status.
Emily takes the information-seeker role; James takes the helper role. Emily asks
for a suitable college; James suggests one. Emily asks about visitor admission;
James supplies the information. Emily requests the postcode; James provides it.
These are the source roles and interaction, not historical facts already said by
Emily or James. Let the model choose compatible motivation and natural wording.
Do not add direct speaking turns for any other person. The complete new scene
must add actual information or a decision. Do not output an outline or a review.

Return one JSON object with root_id, route, scene_id, dataset_card_id, status,
turns:[{speaker,text}], feedback_topic_ids:[], no_scene_reason. Aim for 12-16
natural turns, status=scene; if impossible, return no_scene with an honest reason.
TARGET_AND_ORIGINAL_CARD_JSON
'''
    tail += json.dumps(target, ensure_ascii=False) + '\nEND CURRENT TASK. Generate only the new scene now.\n'
    request['messages'][0]['content'] += tail
    proof.update(diagnostic_scope='consumed_source_generation_only',
        source_run=str(SOURCE), source_id=card['source_id'],
        source_card_with_evidence_sha256=c.r.base.digest(card),
        final_task_sha256=c.r.base.digest(tail), source_reservation_created=False,
        feedback_receipt_scope='intermediate_request_before_current_task_append',
        pre_tail_payload_sha256=before_tail_hash, final_payload_sha256=c.r.base.digest(request),
        fresh_collection_body_tokens=0, training_pool_eligible=False)
    return request, proof


def verify_inputs(run):
    run = Path(run).resolve()
    c.r.base.require(run.parent == HERE / 'diagnostics', 'diagnostic_path_invalid')
    manifest = c.r.base.read(run / 'MANIFEST.json')
    c.r.base.require(manifest['scope'] == 'consumed_source_diagnostic' and
        manifest['logical_max'] == 1 and manifest['http_max'] == 2 and
        manifest['training_pool_eligible'] is False, 'diagnostic_scope_changed')
    c.r.base.verify_bindings(manifest['files'])
    c.r.base.verify_bindings(c.r.base.read(HERE / 'BASELINE_FILES.json')['files'])
    c.verify(SOURCE)
    inputs = c.r.base.read(run / 'INPUT.json')
    history = c.r.base.read(SOURCE / 'FORMAL_BASELINE.json')['turns']
    card = c.r.base.read(SOURCE / 'extractions/01.json')['card']
    c.r.base.require(inputs['history'] == history and len(history) == 132 and
        c.r.base.digest(history) == c.r.AUDITED_V34_A3_HISTORY_SHA256, 'full_A3_history_changed')
    c.r.base.require(inputs['source_card'] == card, 'original_card_changed')
    reservations = c.r.base.read(SOURCE / 'SOURCE_RESERVATIONS.json')['reservations']
    c.r.base.require(any(x['source_id'] == card['source_id'] and
        x['source_item_sha256'] == card['lineage']['source_item_sha256'] for x in reservations),
        'diagnostic_source_not_consumed')
    c.r.base.require(not (run / 'SOURCE_RESERVATIONS.json').exists(), 'diagnostic_created_new_source_reservation')
    request, proof = make_request(run)
    c.r.base.require(c.r.base.read(run / 'FROZEN_REQUEST.json') == {'payload': request, 'proof': proof},
        'diagnostic_payload_changed')
    return manifest


def prepare(run):
    run = Path(run).resolve()
    c.r.base.require(run.parent == HERE / 'diagnostics' and not run.exists(), 'invalid_or_existing_diagnostic')
    c.verify(SOURCE)
    run.mkdir(parents=True)
    shutil.copytree(SOURCE / 'feedback', run / 'feedback')
    shutil.copyfile(SOURCE / 'provider_profile.json', run / 'provider_profile.json')
    c.r.base.write(run / 'INPUT.json', {'history': c.r.base.read(SOURCE / 'FORMAL_BASELINE.json')['turns'],
        'source_card': c.r.base.read(SOURCE / 'extractions/01.json')['card'],
        'provenance': 'reuses_consumed_source_diagnostic_only'})
    request, proof = make_request(run)
    c.r.base.write(run / 'FROZEN_REQUEST.json', {'payload': request, 'proof': proof})
    _profile, transport_files, tokenizer_files = c.r.transport_inputs()
    files = [Path(__file__), HERE / 'POLICY.md', HERE / 'BASELINE_FILES.json']
    files += list(V41.glob('*.py')) + list(c.r.LEGACY.glob('*.py')) + transport_files + tokenizer_files
    files += [p for p in run.rglob('*') if p.is_file()]
    files += [SOURCE / 'MANIFEST.json', SOURCE / 'SOURCE_RESERVATIONS.json', SOURCE / 'extractions/01.json',
              SOURCE / 'FORMAL_BASELINE.json', SOURCE / 'COLLECTION_BINDINGS.json']
    prior = c.r.source_inventory.collect_prior()
    c.r.base.write(run / 'MANIFEST.json', {'version': 'v42_original_evidence_tail_probe',
        'scope': 'consumed_source_diagnostic', 'logical_max': 1, 'http_max': 2,
        'model': c.r.MODEL, 'training_pool_eligible': False, 'source_run': str(SOURCE),
        'inventory_before_sha256': c.r.base.digest(prior), 'occupied_sources_before': len(prior['accepted_uses']),
        'prepared_at': c.r.now(), 'files': c.r.base.bindings(files)})
    verify_inputs(run)
    return {'status': 'PREPARED', 'full_history_turns': 132, 'logical_max': 1,
            'new_source_reservations': 0, 'training_pool_eligible': False}


def validate_response(raw, inputs):
    obj, errors, warnings, dup = None, [], [], None
    try:
        obj = c.r.parse_object(raw)
        obj, warnings = c.validate_dialogue(raw, c.identity(ROOT, ROUTE, SCENE),
            {t['speaker'] for t in inputs['history']}, inputs['source_card'])
        if obj['status'] == 'no_scene':
            errors.append('model_returned_no_scene')
    except (ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    # Still check readable text if identity was wrong; report every mechanical
    # finding, rather than letting the first error hide historical copying.
    if isinstance(obj, dict) and isinstance(obj.get('turns'), list) and obj['turns'] and all(
        isinstance(t, dict) and isinstance(t.get('speaker'), str) and isinstance(t.get('text'), str)
        for t in obj['turns']):
        dup = c.r.duplicate_check.inspect_duplicates(obj['turns'], {'complete_formal_history': inputs['history']})
    return {'parsed': obj, 'mechanical_failures': errors, 'quality_warnings': warnings,
            'duplicate_check': dup, 'semantic_certified': False, 'training_pool_eligible': False}


def live(run):
    run = Path(run).resolve()
    with c.r.LOCK_PATH.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = verify_inputs(run)
        c.r.base.require(not (run / 'LIVE_STARTED.json').exists(), 'no_automatic_restart')
        prior = c.r.source_inventory.collect_prior()
        c.r.base.write(run / 'LIVE_STARTED.json', {'started_at': c.r.now(),
            'inventory_before_sha256': c.r.base.digest(prior), 'scope': 'consumed_source_diagnostic',
            'new_source_reservations': 0, 'shared_live_lock_held': True})
        frozen = c.r.base.read(run / 'FROZEN_REQUEST.json')
        raw, directory = Caller(run).call('B-S01-grounded-final-task', frozen['payload'], frozen['proof'])
        result = validate_response(raw, c.r.base.read(run / 'INPUT.json'))
        c.r.base.write(run / 'GENERATION_CHECK.json', result)
        verify_inputs(run)
        c.r.base.require(prior == c.r.source_inventory.collect_prior(), 'diagnostic_inventory_changed')
        c.r.base.write(run / 'RESULT_BINDINGS.json', {'files': c.r.base.bindings(
            [p for p in run.rglob('*') if p.is_file()]), 'semantic_certified': False})
        return {'status': 'DIAGNOSTIC_GENERATED', 'request': str(directory),
            'mechanical_failures': result['mechanical_failures'],
            'duplicate_check': (result['duplicate_check'] or {}).get('status'),
            'fresh_collection_body_tokens': 0, 'new_source_reservations': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'live', 'verify'))
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = verify_inputs(args.run) if args.command == 'verify' else globals()[args.command](args.run)
    if args.command == 'verify':
        result = {'status': 'FROZEN_INPUTS_VERIFIED', 'logical_max': result['logical_max']}
    print(json.dumps(result, ensure_ascii=False, indent=2))
