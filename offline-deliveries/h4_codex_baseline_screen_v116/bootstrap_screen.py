"""Freeze baseline candidates before this batch's full-text semantic reads."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATA = PROJECT / 'SP-820-publish/first_training_pipeline/data'

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write('\n')

def main():
    training = DATA / 'matched_training/m_clean_policy_training_manifest.jsonl'
    train_ids = {json.loads(line)['root_id'] for line in training.read_text().splitlines()}
    manifest = DATA / 'qa_generation_only/manifest.json'
    inventory = {r['root_id']: r for r in read(manifest)['roots']}
    originals_path = DATA / 'selection_manifest_h4_unexpanded122.json'
    originals = {r['root_id']: r for r in read(originals_path)['records']}
    exclude = {f'cc-train-episode-{n}' for n in [761,9557,14304,20432,222937,225575,29753,31457,45849,55788]}
    holdout_paths = [DATA / 'matched_training' / filename for filename in
                     ['dev_full_eval_manifest.jsonl', 'test_full_eval_manifest.jsonl']]
    holdout_ids = {json.loads(line)['root_id'] for path in holdout_paths for line in path.read_text().splitlines()}
    holdout_sources = [read(originals[rid]['input_path'])['source'] for rid in holdout_ids]
    trace, candidates = [], []
    for rid in sorted(train_ids, key=lambda item: int(item.rsplit('-',1)[1])):
        if rid in exclude:
            trace.append({'root_id': rid, 'eligible': False, 'reason': 'excluded_previous_full_validation_root'})
            continue
        upstream = Path(inventory[rid]['source_path'])
        # Structure only: body is not printed or used in selection.
        source = read(upstream)
        scenes = sorted({t['scene_ref'] for t in source['turns'] if t.get('scene_ref')})
        speakers = sorted({t['speaker'] for t in source['turns']})
        eligible = scenes == ['R01:S01','R01:S02','R01:S03'] and len(speakers) == 2
        trace.append({'root_id': rid, 'eligible': eligible, 'scene_refs': scenes,
                      'speakers': speakers, 'reason': 'structure_eligible' if eligible else 'structure_ineligible'})
        if not eligible:
            continue
        assert sha(upstream) == inventory[rid]['source_sha256']
        short = rid.rsplit('-',1)[1]
        target = HERE / 'inputs' / f'root{short}_BASELINE.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(upstream.read_bytes())
        original_path = Path(originals[rid]['input_path'])
        original = read(original_path)['source']
        assert rid not in holdout_ids
        for key in ['source_id','source_record_sha256']:
            assert original[key] not in {item[key] for item in holdout_sources}
        candidates.append({'root_id': rid, 'baseline_path': str(target), 'baseline_sha256': sha(target),
                           'upstream_path': str(upstream), 'upstream_sha256': sha(upstream),
                           'baseline_turn_count': len(source['turns']), 'baseline_speakers': speakers,
                           'original_source_id': original['source_id'],
                           'original_source_record_sha256': original['source_record_sha256']})
        if len(candidates) == 8:
            break
    assert [r['root_id'].rsplit('-',1)[1] for r in candidates] == ['57679','59126','63323','68549','75223','77882','78203','87964']
    prior = HERE.parent / 'h4_codex_newroots_ab_validation_v115'
    prior_files = read(prior / 'evidence/HISTORICAL_PROTECTION.json')['files']
    protection = {item['path']: item['sha256'] for item in prior_files}
    for path in prior.rglob('*'):
        if path.is_file():
            protection[str(path)] = sha(path)
    for path, digest in protection.items():
        assert sha(path) == digest, path
    save(HERE / 'HISTORICAL_PROTECTION.json', {'scope': 'v115 complete plus inherited sealed history',
         'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'files': [{'path': path, 'sha256': digest} for path, digest in sorted(protection.items())]})
    policy = HERE / 'BASELINE_POLICY.md'
    save(HERE / 'SCREEN_PLAN.json', {'schema_version': 'codex_baseline_screen_plan_v1',
         'batch_id': HERE.name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
         'policy_path': str(policy), 'policy_sha256': sha(policy),
         'status': 'FROZEN_BEFORE_THIS_BATCH_FULL_TEXT_SEMANTIC_READS',
         'candidate_denominator': 8, 'selection_max': 4, 'reviews_per_candidate': 2,
         'candidate_order': [r['root_id'] for r in candidates], 'candidates': candidates,
         'structural_trace': trace, 'excluded_previous_full_validation_roots': sorted(exclude),
         'selection_rule': 'First at most four in fixed candidate order with two distinct valid PASS/WARNING reviews; any HOLD or disagreement involving HOLD excludes; no replenishment beyond frozen eight.',
         'training_exposure': True, 'prior_codex_exposure': True, 'unseen_holdout_claimed': False,
         'holdout_root_overlap': [], 'holdout_source_id_overlap': [], 'holdout_source_record_overlap': [],
         'bindings': [{'path': str(path), 'sha256': sha(path)} for path in
                      [training, manifest, originals_path, *holdout_paths, HERE/'bootstrap_screen.py', HERE/'HISTORICAL_PROTECTION.json']],
         'generation_started': False, 'solver_started': False, 'training_started': False,
         'pool_merge_started': False, 'external_semantic_certification': False})
    print(json.dumps({'candidate_ids': [r['root_id'] for r in candidates],
                      'protected_files': len(protection), 'plan_sha256': sha(HERE/'SCREEN_PLAN.json')}))

if __name__ == '__main__':
    main()
