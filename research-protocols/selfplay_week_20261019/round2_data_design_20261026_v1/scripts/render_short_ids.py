#!/usr/bin/env python3
"""Render dated draft scenes with short display IDs while retaining full coordinates.

The display is for model-facing text. The private mapping is the audit address;
it is never allowed to replace a source turn_ref or source binding.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import date
from pathlib import Path

try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def binding(path: Path) -> dict:
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def validate_date(scene: dict) -> None:
    d = date.fromisoformat(scene['conversation_date'])
    assert WEEKDAYS[d.weekday()] == scene['weekday'], (scene['scene_ref'], d, scene['weekday'])

def render(draft_path: Path, out_path: Path, tokenizer_path: Path | None) -> dict:
    draft = json.loads(draft_path.read_text())
    assert draft['status'] == 'DESIGN_PILOT_NOT_TRAINING_DATA'
    plan_binding = draft['calendar_plan_binding']
    plan_path = Path(plan_binding['path'])
    assert binding(plan_path) == plan_binding
    root = draft['root_id']
    display_scenes, mapping, rendered = [], {}, []
    for scene in draft['scenes']:
        validate_date(scene)
        short_scene = scene['scene_ref'].split(':')[-1]
        turns, lines = [], [f"场景 {short_scene}｜对话日期：{scene['conversation_date']}（{scene['weekday']}）"]
        for index, turn in enumerate(scene['turns'], 1):
            short_id = f"T{index:03d}"
            map_key = f"{scene['scene_ref']}:{short_id}"
            assert map_key not in mapping
            canonical = f"{root}:{turn['turn_ref']}"
            mapping[map_key] = {
                'root_id': root,
                'display_scene': short_scene,
                'display_ref': f"[{short_id}]",
                'full_coordinate': canonical,
                'turn_ref': turn['turn_ref'],
                'session_ref': f"{root}:{scene['scene_ref']}",
                'speaker': turn['speaker'],
                'text': turn['text'],
                'source_binding': binding(draft_path),
            }
            turns.append({'display_ref': f"[{short_id}]", 'speaker': turn['speaker'], 'text': turn['text']})
            lines.append(f"[{short_id}] {turn['speaker']}: {turn['text']}")
        display_scenes.append({'scene_ref': scene['scene_ref'], 'conversation_date': scene['conversation_date'], 'weekday': scene['weekday'], 'turns': turns, 'temporal_checks': scene.get('temporal_checks', [])})
        rendered.extend(lines)
    rendered_text = '\n'.join(rendered)
    result = {
        'schema_version': 'short_display_ids_with_private_coordinates_v1',
        'status': 'DESIGN_PILOT_NOT_TRAINING_DATA',
        'source_draft_binding': binding(draft_path),
        'calendar_plan_binding': plan_binding,
        'display_id_scope': 'reset_per_scene; scene header supplies scene identity',
        'display_scenes': display_scenes,
        'private_coordinate_map': mapping,
        'rendered_text': rendered_text,
        'validation': {'unique_full_coordinates': len(mapping) == len(set(v['full_coordinate'] for v in mapping.values())), 'turns': len(mapping)},
        'model_calls': 0,
        'training': False,
    }
    if tokenizer_path:
        tok = Tokenizer.from_file(str(tokenizer_path))
        full_lines = []
        for scene in draft['scenes']:
            short_scene = scene['scene_ref'].split(':')[-1]
            full_lines.append(f"场景 {short_scene}｜对话日期：{scene['conversation_date']}（{scene['weekday']}）")
            for index, turn in enumerate(scene['turns'], 1):
                key = f"{scene['scene_ref']}:T{index:03d}"
                full_lines.append(f"[{mapping[key]['full_coordinate']}] {turn['speaker']}: {turn['text']}")
        full_text = '\n'.join(full_lines)
        full_count = len(tok.encode(full_text, add_special_tokens=False).ids)
        short_count = len(tok.encode(rendered_text, add_special_tokens=False).ids)
        result['token_counts'] = {'short_render': short_count, 'full_coordinate_render': full_count, 'estimated_saving': full_count - short_count}
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('draft', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--tokenizer', type=Path)
    args = ap.parse_args()
    result = render(args.draft, args.output, args.tokenizer)
    print(json.dumps({'status': 'PASS', **result.get('token_counts', {}), 'turns': result['validation']['turns']}, ensure_ascii=False))
