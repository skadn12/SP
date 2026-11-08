"""Freeze deterministic, training-unseen test sources before QA generation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'SP-820-publish/first_training_pipeline/data'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    assert not (HERE / 'RUN_CONFIG.json').exists()
    train_paths = [DATA / 'matched_training' / f'm_{side}_policy_training_manifest.jsonl'
                   for side in ('clean', 'inclusive')]
    train_rows = [json.loads(l) for p in train_paths for l in p.read_text().splitlines()]
    train_roots = {r['root_id'] for r in train_rows}
    test_path = DATA / 'eval_common_test/clean_test/candidate_results.jsonl'
    # Consume identities only; no request outputs or model scores enter selection.
    test_roots = {json.loads(l)['root_id'] for l in test_path.read_text().splitlines()}
    selected = sorted(test_roots, key=lambda r: int(r.rsplit('-', 1)[1]))[:4]
    assert selected == [f'cc-train-episode-{x}' for x in (1456, 35167, 39120, 78460)]
    assert not train_roots & test_roots
    records = read(DATA / 'selection_manifest_h4_unexpanded122.json')['records']
    originals = {r['root_id']: r for r in records}
    sources = {root: read(Path(originals[root]['input_path']))['source'] for root in train_roots | test_roots}
    for key in ('source_id', 'source_record_sha256'):
        assert all(sources[r].get(key) for r in train_roots | test_roots), key
        assert not {sources[r][key] for r in train_roots} & {sources[r][key] for r in test_roots}, key
    inventory = {r['root_id']: r for r in read(DATA / 'qa_generation_only/manifest.json')['roots']}
    frozen = []
    for root in selected:
        item = inventory[root]
        original_path = Path(item['source_path'])
        assert sha(original_path) == item['source_sha256']
        doc = read(original_path)
        assert doc['root_id'] == root and len(doc['turns']) == doc['turn_count']
        directory = HERE / 'roots' / root
        directory.mkdir(parents=True)
        target = directory / 'SOURCE.json'
        with target.open('xb') as f:
            f.write(original_path.read_bytes())
        scene = doc['child_scene_refs'][-1]
        turns = [t for t in doc['turns'] if t['scene_ref'] == scene]
        assert len(turns) >= 12
        slots = []
        for slot, indices in enumerate(((0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11)), 1):
            chosen = [turns[i] for i in indices]
            order = list(range(6))
            random.Random(f'v106|{root}|ordering|{slot}').shuffle(order)
            anchors = {f'I{j + 1}': {k: chosen[i][k] for k in ('turn_ref', 'speaker', 'text')}
                       for j, i in enumerate(order)}
            slots.append({'slot': slot, 'scene_ref': scene, 'anchors': anchors})
        write(directory / 'ORDERING_PLAN.json', {'root_id': root, 'source_sha256': sha(target),
                                                'slots': slots, 'created_before_statements': True})
        with (directory / 'SOURCE.txt').open('x', encoding='utf-8') as f:
            f.write('\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in doc['turns']) + '\n')
        frozen.append({'root_id': root, 'source_id': sources[root]['source_id'],
                       'source_record_sha256': sources[root]['source_record_sha256'],
                       'original_source_path': str(original_path), 'source_path': str(target),
                       'source_sha256': sha(target), 'turns': len(doc['turns']),
                       'scene_ref': scene, 'scene_turns': len(turns),
                       'ordering_plan_sha256': sha(directory / 'ORDERING_PLAN.json')})
    write(HERE / 'RUN_CONFIG.json', {
        'batch_id': HERE.name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'EXPLORATORY_FIXED_ROOT_VALIDATION_CONFIGURED_BEFORE_QA_AND_SOLVER',
        'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
        'sampling_parameters': None, 'usage_tokens': None,
        'model_metadata_limitation': 'Exact underlying model, generation parameters, billing and usage not exposed by this session.',
        'roots': frozen, 'source_selection': 'First four numeric root IDs in historical common_test, root+source_id+source_record disjoint from both first-round training sets. No performance filtering.',
        'historical_test_evaluated': True, 'never_evaluated_claim': False,
        'new_dialogue_expansion': False, 'new_source_turns': 0,
        'source_rule': 'Read all turns; block unresolved actor/time/location/state/continuity contradictions. Preserve naturalness/repetition warnings. No source repair or replacement.',
        'qa_per_ready_root': {'gold2': 1, 'gold3': 1, 'gold4': 1, 'gold5': 1, 'gold6_control': 1, 'ordering': 2},
        'attempts_per_root': 1, 'semantic_repairs': 0, 'source_substitutions': 0,
        'failure_policy': 'Preserve all drafts; exclude failed whole questions or sources; report missing slots, do not repair or resample.',
        'mcq_rule': 'Six distinct facts, true/false sentence pair per fact; false must be explicitly contradicted within public scope. Unsupported/unknown is not false. Retain modality, actor, time and completion. No gender inference.',
        'mcq_composition': 'For Gold g, true fact indices from seeded shuffle(v106|root|gold|g), first g of six; display independent seeded shuffle. No model scores or answer count in prompt.',
        'ordering_rule': 'Six original full turns fixed before writing in ORDERING_PLAN; one speech act/event each; exact latest-scene appearance scope; enumerate alternative semantic anchors; unique strict order must match prewritten plan.',
        'review': 'Reviewer reads whole sources and public propositions before private Gold. Same inherited model, separate roles, no external semantic certification.',
        'solver_protocol': 'v15_full_only_random_mcq6x3_order12x1_v1',
        'models': {'base': 'M0', 'trained': 'M_clean_policy'},
        'primary_outcomes': ['Gold2-3 exact accuracy', 'Gold2-3 false-positive and false-negative rates'],
        'secondary_outcomes': ['Gold2/3/4/5 separate and equal-Gold exact', 'Gold6 all-select control', 'ordering exact and pair partial', 'per-root and paired-question results'],
        'statistical_unit': 'Root/family; repeated calls are correlated, no call-level significance claims.',
        'training_started': False, 'test_to_training_export': False,
        'training_files': [{'path': str(p), 'sha256': sha(p)} for p in train_paths],
        'source_index': {'path': str(DATA / 'selection_manifest_h4_unexpanded122.json'),
                         'sha256': sha(DATA / 'selection_manifest_h4_unexpanded122.json')},
    })
    print(json.dumps({'status': 'FIXED_BEFORE_QA', 'roots': selected, 'sources': len(frozen),
                      'planned_candidates': 28, 'planned_calls_per_model_if_all_content_pass': 456}))


if __name__ == '__main__':
    main()
