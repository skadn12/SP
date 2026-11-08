"""Read actual request/stage progress without changing any production artifact."""
from pathlib import Path
import json

HOME = Path(__file__).resolve().parents[1]
RUN = HOME / 'expansion_engine/runs/four_roots_B_scene01'
def read(p): return json.loads(p.read_text())
def main():
    requests = []
    for folder in sorted((RUN / 'requests').glob('*')):
        proof = read(folder / 'PROOF.json')
        records = {}
        for name in ('VALIDATION.json', 'REVIEW.json', 'ACCEPTANCE.json', 'EVENT_STATE_DECISION.json'):
            p = folder / name
            if p.exists():
                obj = read(p)
                records[name] = {k: obj[k] for k in ('status', 'reason', 'requires_replan', 'schema_valid', 'failures') if k in obj}
        response = folder / 'RESPONSE.json'
        meta = read(response).get('metadata', {}) if response.exists() else {}
        requests.append({'request': folder.name, 'role': proof['role'], 'root': proof.get('root_id'),
                         'finish_reason': meta.get('finish_reason'), 'output_tokens': meta.get('output_tokens'),
                         'records': records})
    roots = []
    for path in sorted((RUN / 'roots').glob('*/B/RESULT.json')):
        obj = read(path)
        roots.append({k: obj[k] for k in ('root_id', 'accepted_scenes', 'added_turns', 'failures')})
    print(json.dumps({'request_count': len(requests), 'recent_requests': requests[-5:],
                      'completed_roots': roots, 'report_exists': (RUN / 'REPORT.json').exists()},
                     ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
