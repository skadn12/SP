#!/usr/bin/env python3
"""Create read-only calendar sidecars for the 19 valid diagnostic roots."""
from __future__ import annotations
import hashlib, json
from collections import OrderedDict
from pathlib import Path

WEEK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'calendar' / 'OLD_ROOT_SCENE_SIDECARS.json'

def bind(path: Path):
    data = path.read_bytes(); return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def main():
    analysis = json.loads((WEEK/'dialogue_tokens_time_question_forms_20261025_v1/ANALYSIS.json').read_text())
    roots = []
    for row in analysis['records']:
        source = Path(row['source_binding']['path'])
        source_obj = json.loads(source.read_text())
        assert bind(source) == row['source_binding']
        groups = OrderedDict()
        for turn in source_obj['turns']:
            key = turn.get('scene_ref') or turn.get('session_ref') or 'PARENT_UNKNOWN_SCENE'
            groups.setdefault(key, []).append(turn['turn_ref'])
        scenes = []
        for scene_ref, refs in groups.items():
            scenes.append({'scene_ref': scene_ref, 'scene_time': {'value': None, 'precision': 'unknown', 'status': 'not_recorded_in_source', 'evidence_refs': []}, 'calendar_scopes': [], 'mentioned_events': [], 'turn_refs': refs})
        roots.append({'root_id': source_obj['root_id'], 'source_binding': row['source_binding'], 'scenes': scenes})
    out = {'schema_version':'calendar_sidecars_v1','status':'READ_ONLY_MIGRATION_SIDECARS','source_roots':len(roots),'scene_time_policy':'unknown_unless_independently_recorded','roots':roots,'warning':'Empty event arrays do not claim that speech contains no dates; this conservative sidecar records only that no event extraction was authorized by this migration. Use OLD_ROOT_DATE_AUDIT.json for the separate date mention audit.','frozen_sources_modified':False,'training':False}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':'PASS','roots':len(roots),'scenes':sum(len(r['scenes']) for r in roots),'output':str(OUT)},ensure_ascii=False))

if __name__ == '__main__': main()
