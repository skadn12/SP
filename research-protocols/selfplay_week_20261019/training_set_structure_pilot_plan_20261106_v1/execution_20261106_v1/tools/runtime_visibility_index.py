#!/usr/bin/env python3
"""Read-only visibility extraction aid, NOT an independent compliance verdict."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import pilot_artifacts as pa

E = Path(__file__).resolve().parents[1]


def strings(value):
    if isinstance(value, str):
        yield value
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return
        if decoded != value:
            yield from strings(decoded)
    elif isinstance(value, list):
        for x in value:
            yield from strings(x)
    elif isinstance(value, dict):
        for k in ('output', 'text', 'content'):
            if k in value:
                yield from strings(value[k])


def extract():
    rows = []
    for p in sorted(Path('/root/.codex/sessions/2026/11/06').glob('rollout-*.jsonl')):
        records = p.read_text().splitlines()
        if not records:
            continue
        meta = json.loads(records[0]).get('payload', {})
        agent = meta.get('agent_path')
        if not agent or agent == '/root':
            continue
        calls, outputs = [], []
        for n, line in enumerate(records, 1):
            x = json.loads(line)
            if x.get('type') != 'response_item':
                continue
            a = x.get('payload', {})
            base = {'line': n, 'record_sha256': hashlib.sha256(line.encode()).hexdigest(),
                    'timestamp': x.get('timestamp')}
            if a.get('type') in ('custom_tool_call', 'function_call'):
                calls.append({**base, 'name': a.get('name'),
                              'input': a.get('input', a.get('arguments'))})
            if a.get('type') in ('custom_tool_call_output', 'function_call_output'):
                texts = list(strings(a.get('output')))
                outputs.append({**base, 'texts': texts,
                                'truncated': any('tokens truncated' in t or 'Warning: truncated output' in t for t in texts)})
        # No semantic claims from this mechanical extraction. The auditor checks
        # allowed reads, freeze chronology and source visibility at role-specific cutoffs.
        rows.append({'agent_path': agent, 'session_path': str(p),
                     'session_meta_record_sha256': hashlib.sha256(records[0].encode()).hexdigest(),
                     'calls': calls, 'visible_outputs': outputs})
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'MECHANICAL_EXTRACTION_ONLY_NOT_INDEPENDENT_AUDIT',
            'policy': 'Only public response_item tool calls and model-visible tool outputs; never encrypted task bodies, reasoning or process event stdout.',
            'sessions': rows}


if __name__ == '__main__':
    out = E / 'review/runtime_visibility_index.json'
    pa.write_new(out, extract())
    print(json.dumps({'output': str(out), 'sha256': pa.digest(out)}))
