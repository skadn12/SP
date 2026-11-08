"""Read/save helpers only; dialogue and semantic judgments are supplied by a human-facing agent."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
BATCH = Path(__file__).resolve().parent.parent
PROJECT = BATCH.parent.parent
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(BATCH))
from codex_scene_payload import write_new
from codex_sequential_ab import SCENES, scene_folder, materialize_scene, load_config, load_baseline, reservation_for
import source_review

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
def context(root, scene):
    return Path('/tmp') / ('v117_' + root.rsplit('-', 1)[1] + '_' + scene.removeprefix('R02:').replace(':', '_') + '_author_read.json')
def print_source(data):
    for t in data['turns']: print(t['turn_ref'], t['speaker'] + ':', t['text'])
def print_native(event):
    print('NATIVE_EVENT', event['dialogue_id'], event['services'])
    for t in event['turns']: print(t['turn_id'], t['speaker'] + ':', t['utterance'])

def author_read(root, scene):
    cfg = load_config(BATCH); i = SCENES.index(scene)
    p = load_baseline(BATCH, cfg, root)[0] if not i else scene_folder(BATCH, root, SCENES[i-1]) / 'CUMULATIVE_SOURCE.json'
    data = read(p)
    c = {'started_at_utc': now(), 'input_path': str(p), 'input_source_sha256': sha(p), 'turn_refs_read': [t['turn_ref'] for t in data['turns']]}
    r = reservation_for(BATCH, cfg, root, scene)
    if r:
        ep = Path(r['source_item_path']); e = read(ep)
        c.update(native_event_path=str(ep), native_event_sha256=sha(ep), native_turn_ids_read=[t['turn_id'] for t in e['turns']])
    save(context(root, scene), c)
    print('CONSTRAINTS', json.dumps(source_review.constraints_for(root)[1]['continuation_constraints'], ensure_ascii=False))
    print_source(data)
    if r: print_native(e)

def author_save(root, scene, payload_path):
    c = read(context(root, scene)); assert sha(c['input_path']) == c['input_source_sha256']
    if 'native_event_path' in c: assert sha(c['native_event_path']) == c['native_event_sha256']
    draft = write_new(BATCH, root, scene, '/root', read(payload_path))
    print(materialize_scene(BATCH, root, scene))
    save(draft.parent / 'AUTHOR_RECEIPT.json', {'root_id': root, 'scene_ref': scene, 'author': '/root', **c,
         'finished_at_utc': now(), 'attempt': 1, 'full_input_source_read': True, 'continuation_constraints_read': True,
         'semantic_rewrites': 0, 'backend': 'CURRENT_CODEX_SESSION', 'exact_model_id': None,
         'draft_path': str(draft), 'draft_sha256': sha(draft)})

def review_read(root, scene):
    log = BATCH / 'evidence/main_review_inputs' / (root.rsplit('-',1)[1] + '_' + scene.replace(':','_') + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as f, contextlib.redirect_stdout(f): source_review.start(root, scene, '/root')
    print('CONSTRAINTS', json.dumps(source_review.constraints_for(root)[1]['continuation_constraints'], ensure_ascii=False))
    folder = scene_folder(BATCH, root, scene)
    print_source(read(folder / 'CUMULATIVE_SOURCE.json'))
    print('DRAFT_NOTES', json.dumps({k:v for k,v in read(folder/'EXPANSION_DRAFT.json').items() if k!='turns'}, ensure_ascii=False))
    r = reservation_for(BATCH, load_config(BATCH), root, scene)
    if r: print_native(read(r['source_item_path']))

if __name__ == '__main__':
    action, root, scene = sys.argv[1:4]
    if action == 'author-read': author_read(root, scene)
    elif action == 'author-save': author_save(root, scene, sys.argv[4])
    elif action == 'review-read': review_read(root, scene)
