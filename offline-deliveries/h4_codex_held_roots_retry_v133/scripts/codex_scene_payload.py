"""Serialize newly authored speaker/text pairs into the fixed scene schema.

This helper is for future drafts. It never rewrites an existing draft or clears
a failure receipt. Semantic content remains the declared author's work.
"""
from pathlib import Path
import argparse

from codex_controlled_batch import digest, read, save
from codex_sequential_ab import SCENES, inspect_root, load_config, load_baseline, reservation_for, scene_folder

def build_draft(root_id, scene_ref, author, input_path, messages, source_id=None, **notes):
    if scene_ref not in SCENES:
        raise ValueError('INVALID_SCENE')
    source = read(input_path)
    if source['root_id'] != root_id:
        raise ValueError('SOURCE_ROOT_MISMATCH')
    if not isinstance(author, str) or not author.strip():
        raise ValueError('AUTHOR_REQUIRED')
    if not isinstance(messages, list) or not 12 <= len(messages) <= 16:
        raise ValueError('TWELVE_TO_SIXTEEN_MESSAGES_REQUIRED')
    if (':A:' in scene_ref and source_id is not None) or (':B:' in scene_ref and not source_id):
        raise ValueError('ROUTE_SOURCE_ID_MISMATCH')
    speakers = {t['speaker'] for t in source['turns']}
    turns = []
    for i, message in enumerate(messages, 1):
        if set(message) != {'speaker', 'text'}:
            raise ValueError('ONLY_SPEAKER_AND_TEXT_ARE_AUTHORED')
        if message['speaker'] not in speakers or not isinstance(message['text'], str) or not message['text'].strip():
            raise ValueError('INVALID_MESSAGE')
        turns.append({'turn_ref': f'{scene_ref}:T{i:03d}', 'session_ref': scene_ref,
                      'scene_ref': scene_ref, 'speaker': message['speaker'],
                      'text': message['text'], 'origin': 'codex_session_generated'})
    return {'root_id': root_id, 'scene_ref': scene_ref, 'author': author,
            'input_source_sha256': digest(input_path), 'source_id': source_id,
            **{k: notes[k] for k in ('state_notes', 'event_mapping') if k in notes}, 'turns': turns}

def write_new(batch, root_id, scene_ref, author, payload):
    batch = Path(batch); cfg = load_config(batch)
    state = inspect_root(batch, cfg, root_id)
    if state['root_stopped']:
        raise ValueError('ROOT_ALREADY_STOPPED_NO_RECOVERY')
    index = SCENES.index(scene_ref)
    current = state['scenes'][index]
    if current['status'] != 'NOT_STARTED' or (index and state['scenes'][index-1]['status'] != 'READY'):
        raise ValueError('SCENE_NOT_READY_FOR_FIRST_DRAFT')
    input_path = load_baseline(batch, cfg, root_id)[0] if not index else scene_folder(batch, root_id, SCENES[index-1]) / 'CUMULATIVE_SOURCE.json'
    reservation = reservation_for(batch, cfg, root_id, scene_ref)
    value = build_draft(root_id, scene_ref, author, input_path, payload['messages'],
                        source_id=reservation['source_id'] if reservation else None,
                        **{k: payload[k] for k in ('state_notes','event_mapping') if k in payload})
    target = scene_folder(batch, root_id, scene_ref) / 'EXPANSION_DRAFT.json'
    save(target, value)
    return target

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-dir', type=Path, required=True)
    p.add_argument('--root-id', required=True); p.add_argument('--scene-ref', choices=SCENES, required=True)
    p.add_argument('--author', required=True); p.add_argument('--payload', type=Path, required=True)
    a = p.parse_args()
    print(write_new(a.batch_dir, a.root_id, a.scene_ref, a.author, read(a.payload)))
