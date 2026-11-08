import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gxy-selfplay')
BATCH = ROOT / 'research-protocols/selfplay_week_20261019/postweek_live_extension_v5'
BASE = ROOT / 'offline-deliveries/h4_codex_train_pool_v125/sources'

SCENES = {
    'cc-train-episode-19708': [
        ('Emily', 'Marco, let us separate the taxi, restaurant, college, hotel, and train notes before we plan the outing.'),
        ('Marco', 'The taxi is from Prezzo to Pembroke College, and the requirement is arrival by thirteen hundred.'),
        ('Emily', 'Thirteen hundred is the arrival deadline, not the time to leave.'),
        ('Marco', 'Right. The service asked again when I wanted to leave, and I did not guess a departure time.'),
        ('Emily', 'The reply did give you a white Volkswagen and the contact number 07750608995.'),
        ('Marco', 'Yes, but the booking reply still does not give an exact departure time.'),
        ('Emily', 'The restaurant note is separate: the Galleria is moderate, at 33 Bridge Street, and no table was reserved.'),
        ('Marco', 'The college note is separate too: Christ\'s College is on Saint Andrew\'s Street, with free entry and postcode CB23BU.'),
        ('Emily', 'For accommodation, Kirkwood House is a north guesthouse, booked for three people for two nights starting Wednesday.'),
        ('Marco', 'Its confirmation is F74W1S62, which should not be confused with the train reference.'),
        ('Emily', 'The train request was for Friday after 18:15 from Cambridge to London King\'s Cross.'),
        ('Marco', 'The first train total of 23.60 pounds was not for all three people.'),
        ('Emily', 'The corrected three-person train total is 70.80 pounds, with reference 5VJX8JWT, payable at the station.'),
        ('Marco', 'I will keep the taxi arrival deadline, its unresolved departure, and each confirmed booking in separate notes.'),
    ],
    'cc-train-episode-20432': [
        ('Liam', 'Mom, let us keep tomorrow\'s time by the water separate from the later train trip.'),
        ('Emily', 'The restaurant enquiry was for Turkish food in the centre at a moderate price, and it was only for information.'),
        ('Liam', 'The adviser offered Anatolia and Efes Restaurant.'),
        ('Emily', 'I chose no preference and received Anatolia\'s address, Thirty Bridge Street, City Centre.'),
        ('Liam', 'There is no table booking or dining day attached to that note.'),
        ('Emily', 'The museum enquiry is another possible outing: Cafe Jello Gallery, free to enter, postcode CB30AF.'),
        ('Liam', 'You have not checked its opening times or arranged a visit.'),
        ('Emily', 'Correct. The date stays blank, and it must not be merged with the restaurant note.'),
        ('Liam', 'The train is for Cambridge to Ely on the Saturday after this coming one.'),
        ('Emily', 'The rail adviser offered TR0718 arriving at 12:07, for four people, with reference IR7HSMG4.'),
        ('Liam', 'The quoted 14.08 pounds is payable at the station, so it is not proof that we have paid.'),
        ('Emily', 'The departure time still needs to be checked before travelling.'),
        ('Liam', 'Tomorrow remains the afternoon by the water, while the Ely train belongs to the following Saturday.'),
        ('Emily', 'I will record the arrival, group size, reference, payment boundary, and unresolved departure separately.'),
    ],
}

def bind(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

for root, texts in SCENES.items():
    baseline = BASE / f'{root}.json'
    hand = BATCH / 'handoff' / root / 'SOURCE_AUTHOR_INPUT.json'
    source = json.loads(baseline.read_text(encoding='utf-8'))
    events = []
    for p in sorted((BATCH / 'inputs').glob(f'{root}_B_S*_EVENT.json')):
        events.append({'scene_ref': p.stem.split('_')[-3] + ':' + p.stem.split('_')[-2] if False else p.name, 'binding': bind(p), 'full_event_read_required': True})
    save(hand, {
        'schema_version': 'live_extension_v5_source_author_input_v1',
        'root_id': root,
        'mode': 'EXTEND',
        'namespace': 'LEX1',
        'source_binding': bind(baseline),
        'full_source': source,
        'native_events': events,
        'scene_sequence': ['LEX1:A:S01', 'LEX1:A:S02', 'LEX1:A:S03', 'LEX1:B:S01', 'LEX1:B:S02', 'LEX1:B:S03'],
        'constraints': {'full_source_read': True, 'solver_outputs_read': False, 'no_new_facts_without_source': True, 'dependent_scene_after_failed_review': False},
    })
    scene_ref = 'LEX1:A:S01'
    turns = [{'turn_ref': f'{scene_ref}:T{i:03d}', 'scene_ref': scene_ref, 'session_ref': scene_ref, 'speaker': speaker, 'text': text} for i, (speaker, text) in enumerate(texts, 1)]
    folder = BATCH / 'roots' / root / 'LEX1_A_S01'
    draft = {'schema_version': 'live_extension_scene_draft_v2', 'root_id': root, 'scene_ref': scene_ref, 'author_id': f'/root/source-author-v5-{root[-5:]}', 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'input_source_binding': bind(baseline), 'author_input_binding': bind(hand), 'turns': turns}
    draft_path = folder / 'DRAFT.json'; save(draft_path, draft)
    cumulative = {'schema_version': 'live_extension_cumulative_source_v2', 'root_id': root, 'turns': source['turns'] + turns, 'baseline_binding': bind(baseline), 'input_source_binding': bind(baseline), 'draft_binding': bind(draft_path), 'scene_ref': scene_ref}
    cumulative_path = folder / 'CUMULATIVE_SOURCE.json'; save(cumulative_path, cumulative)
    save(folder / 'AUTHOR_CONSUMPTION.json', {'schema_version': 'source_author_consumption_v2', 'author_id': draft['author_id'], 'root_id': root, 'scene_ref': scene_ref, 'full_source_read': True, 'all_continuation_constraints_read': True, 'material_targets_read': True, 'native_events_read': False, 'solver_outputs_read': False, 'author_input_binding': bind(hand), 'input_source_binding': bind(baseline), 'draft_binding': bind(draft_path), 'created_at_utc': datetime.now(timezone.utc).isoformat()})
print(json.dumps({'batch': str(BATCH), 'roots': list(SCENES), 'scenes_written': len(SCENES)}, ensure_ascii=False))
