import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/gxy-selfplay')
BATCH = ROOT / 'research-protocols/selfplay_week_20261019/postweek_live_extension_v5'
BASE = ROOT / 'offline-deliveries/h4_codex_train_pool_v125/sources'

SCENES = {
    'cc-train-episode-19708': [
        ('Emily', 'Let us reconcile the outing notes without turning an unresolved request into a completed booking.'),
        ('Marco', 'The taxi request is from Prezzo to Pembroke College, with arrival required by 13:00.'),
        ('Emily', 'That is an arrival deadline only; the service still did not give us an exact departure time.'),
        ('Marco', 'The reply identifies a white Volkswagen and the contact number 07750608995.'),
        ('Emily', 'The Galleria note is separate: it is moderate in price at 33 Bridge Street, and no table was reserved.'),
        ('Marco', "Christ's College is on Saint Andrew's Street, has free entry, and uses postcode CB23BU in the note."),
        ('Emily', 'Kirkwood House is the north guesthouse for three people for two nights starting Wednesday.'),
        ('Marco', 'Its confirmation is F74W1S62, so I will keep that identifier with the accommodation record.'),
        ('Emily', "The rail request is for Friday after 18:15 from Cambridge to London King's Cross."),
        ('Marco', 'The earlier £23.60 figure was not the total for all three people.'),
        ('Emily', 'The corrected three-person figure is £70.80, reference 5VJX8JWT, payable at the station.'),
        ('Marco', 'I will retain the rail reference and payment condition without claiming that station payment is complete.'),
        ('Emily', 'The taxi departure remains unresolved, and no table was reserved for the restaurant; the supplied accommodation details stay attached to their own record.'),
        ('Marco', 'I will preserve the separate taxi, restaurant, college, accommodation, and train states for the next planning step.'),
    ],
    'cc-train-episode-20432': [
        ('Liam', "Let us reconcile tomorrow afternoon by the water with the later Ely train, rather than treating them as one plan."),
        ('Emily', 'The restaurant enquiry was for Turkish food in the centre at a moderate price and was information only.'),
        ('Liam', "The adviser offered Anatolia and Efes Restaurant; after no preference was given, the supplied address was for Anatolia at Thirty Bridge Street, City Centre."),
        ('Emily', 'No dining day or table was booked, so the restaurant item remains an enquiry.'),
        ('Liam', 'The separate gallery enquiry names Cafe Jello Gallery, which is free to enter and has postcode CB30AF.'),
        ('Emily', 'Its visit date remains blank, and its opening times remain unchecked; no gallery visit was arranged.'),
        ('Liam', 'The rail item is Cambridge to Ely on the Saturday after this coming one, separate from tomorrow afternoon by the water.'),
        ('Emily', 'The offered service is TR0718, arriving at 12:07, for four people, with reference IR7HSMG4.'),
        ('Liam', 'The quoted £14.08 is payable at the station and does not show that payment has already been made.'),
        ('Emily', 'The departure time still needs checking before travelling.'),
        ('Liam', 'The water plan and later train therefore keep different time contexts and should not be merged.'),
        ('Emily', 'I will retain the restaurant and gallery as information with unresolved details, rather than inventing dates or reservations.'),
        ('Liam', 'The train arrival, group size, reference, and payment boundary can be recorded while its departure remains open.'),
        ('Emily', 'This leaves a traceable plan: tomorrow by the water, a later Saturday train, and separate enquiry states for the restaurant and gallery.'),
    ],
}

def bind(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

created = datetime.now(timezone.utc).isoformat()
for root, texts in SCENES.items():
    baseline = BASE / f'{root}.json'
    hand = BATCH / 'handoff' / root / 'SOURCE_AUTHOR_INPUT.json'
    previous_dir = BATCH / 'roots' / root / 'LEX1_A_S02'
    previous_cumulative = previous_dir / 'CUMULATIVE_SOURCE.json'
    previous = json.loads(previous_cumulative.read_text(encoding='utf-8'))
    scene_ref = 'LEX1:A:S03'
    turns = [
        {'turn_ref': f'{scene_ref}:T{i:03d}', 'scene_ref': scene_ref, 'session_ref': scene_ref,
         'speaker': speaker, 'text': text}
        for i, (speaker, text) in enumerate(texts, 1)
    ]
    folder = BATCH / 'roots' / root / 'LEX1_A_S03'
    author_id = f'/root/source-author-v5-a3-{root[-5:]}'
    draft = {
        'schema_version': 'live_extension_scene_draft_v2',
        'root_id': root,
        'scene_ref': scene_ref,
        'author_id': author_id,
        'created_at_utc': created,
        'input_source_binding': bind(baseline),
        'previous_cumulative_binding': bind(previous_cumulative),
        'turns': turns,
    }
    draft_path = folder / 'DRAFT.json'
    save(draft_path, draft)
    cumulative = {
        'schema_version': 'live_extension_cumulative_source_v2',
        'root_id': root,
        'turns': previous['turns'] + turns,
        'baseline_binding': bind(baseline),
        'input_source_binding': bind(baseline),
        'previous_cumulative_binding': bind(previous_cumulative),
        'draft_binding': bind(draft_path),
        'scene_ref': scene_ref,
    }
    cumulative_path = folder / 'CUMULATIVE_SOURCE.json'
    save(cumulative_path, cumulative)
    author = {
        'schema_version': 'source_author_consumption_v2',
        'author_id': author_id,
        'root_id': root,
        'scene_ref': scene_ref,
        'full_source_read': True,
        'all_continuation_constraints_read': True,
        'material_targets_read': True,
        'native_events_read': False,
        'solver_outputs_read': False,
        'input_source_binding': bind(baseline),
        'previous_cumulative_binding': bind(previous_cumulative),
        'draft_binding': bind(draft_path),
        'created_at_utc': created,
    }
    save(folder / 'AUTHOR_CONSUMPTION.json', author)
    print(root, len(turns), bind(draft_path), bind(cumulative_path))
