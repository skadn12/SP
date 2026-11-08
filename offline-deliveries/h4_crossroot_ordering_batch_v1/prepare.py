from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path('/root/autodl-tmp/gxy-selfplay')
MANIFEST = PROJECT / 'offline-deliveries/h4_latest_expansion_samples_20260922/source_manifest.json'
ROOTS = {'cc-train-episode-136202', 'cc-train-episode-197635', 'cc-train-episode-254852'}
MODEL = 'Qwen/Qwen3.6-35B-A3B'

INSTRUCTIONS = '''Generate up to TWO genuinely different six-item ordering questions from the complete dialogue below. Return fewer with an abstention explanation when the source cannot support them. Output one JSON object with keys questions and abstentions.
Each question must use question_type ordering, chronology_kind conversation_order or within_scene_actions, exactly I1..I6 items, a complete gold permutation, and item_evidence for all six items. Each evidence entry must contain one exact turn_ref, a contiguous verbatim quote, a status, and a reason. Use six different anchor turns. Ask explicitly about transcript utterance order when using conversation_order; do not infer dates of recalled events. Preserve report, plan, promise, advice, and completed-action distinctions. Do not invent facts or force a quota.\n\nSOURCE:\n'''

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    selected = [x for x in manifest if x['root_id'] in ROOTS]
    if len(selected) != 3:
        raise SystemExit('selected_root_manifest_mismatch')
    out = {'version': HERE.name, 'requested_model': MODEL, 'roots': []}
    for row in selected:
        source = Path(row['source_path'])
        raw = source.read_bytes()
        if sha(raw) != row['source_sha256']:
            raise SystemExit(f"source_hash_changed:{row['root_id']}")
        data = json.loads(raw)
        turns = data.get('turns')
        if data.get('root_id') != row['root_id'] or len(turns or []) != row['total_turns']:
            raise SystemExit(f"source_identity_mismatch:{row['root_id']}")
        surface = {'root_id': data['root_id'], 'turn_count': len(turns), 'round_count': data.get('round_count'), 'round_boundaries': data.get('round_boundaries'), 'turns': turns}
        prompt = INSTRUCTIONS + json.dumps(surface, ensure_ascii=False, indent=2) + '\n'
        root_dir = HERE / row['root_id']
        root_dir.mkdir(parents=True, exist_ok=False)
        (root_dir / 'source.json').write_bytes(raw)
        (root_dir / 'prompt.txt').write_text(prompt)
        (root_dir / 'SOURCE_METADATA.json').write_text(json.dumps({'root_id': row['root_id'], 'source_path': str(source), 'source_sha256': row['source_sha256'], 'turn_count': len(turns), 'requested_model': MODEL, 'max_candidates': 2, 'api_calls': 0, 'solver_calls': 0, 'training_started': False, 'prompt_sha256': sha(prompt.encode())}, ensure_ascii=False, indent=2) + '\n')
        out['roots'].append({'root_id': row['root_id'], 'turn_count': len(turns), 'source_sha256': row['source_sha256'], 'prompt_sha256': sha(prompt.encode()), 'status': 'PREPARED_NO_API_CALL'})
    (HERE / 'BATCH_MANIFEST.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
