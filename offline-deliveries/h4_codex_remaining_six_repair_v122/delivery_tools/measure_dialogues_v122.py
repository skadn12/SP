"""Local tokenizer counts for completed dialogues; does not call any model."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import statistics
import sys

B = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(B / 'tools'))
import scene_controller_v120 as scene


def read(path):
    return json.loads(path.read_text())


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    import transformers
    tokenizer_path = '/root/autodl-tmp/Qwen3-4B'
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows, stopped = [], []
    for row in read(B / 'MANIFEST.json')['roots']:
        root = row['root_id']
        if not row['prefix_reuse_approved']:
            stopped.append({'root_id': root, 'reason': 'SOURCE_OR_PREFIX_HOLD'})
            continue
        unit = Path(row['unit_path'])
        state = scene.inspect_root(unit, scene.load_config(unit), root, record_reviews=False)
        if not state['full_expansion_complete']:
            assert state['root_stopped'], 'GENERATION_STILL_PENDING:' + root
            stopped.append({'root_id': root, 'reason': 'SCENE_HOLD'})
            continue
        path = scene.scene_folder(unit, root, scene.SCENES[-1]) / 'CUMULATIVE_SOURCE.json'
        source = read(path)
        turns = source['turns']
        six = [t for t in turns if t.get('scene_ref') in scene.SCENES]
        metrics = {}
        for prefix, subset in (('full', turns), ('six_scope', six)):
            rendered = '\n'.join(t['speaker'] + ': ' + t['text'] for t in subset)
            metrics.update({prefix + '_turns': len(subset), prefix + '_tokens': len(tokenizer.encode(rendered, add_special_tokens=False)),
                            prefix + '_render_sha256': sha(rendered.encode())})
        rows.append({'root_id': root, 'source_status': row.get('effective_source_status', row['source_status']), 'source_path': str(path),
                     'source_sha256': sha(path.read_bytes()), **metrics})
    keys = ('full_turns', 'full_tokens', 'six_scope_turns', 'six_scope_tokens')
    report = {'schema_version': 'v122_complete_dialogue_lengths_v1', 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'tokenizer': tokenizer_path, 'tokenizer_class': type(tokenizer).__name__, 'transformers_version': transformers.__version__,
        'python_executable': sys.executable, 'local_files_only': True, 'add_special_tokens': False,
        'render': 'speaker: text lines joined with a single newline; no trailing newline, chat template, turn IDs or metadata',
        'complete_six_scene_root_count': len(rows), 'roots': rows, 'incomplete_or_held_roots_excluded_from_mean': stopped,
        'arithmetic_mean': {k: statistics.mean(r[k] for r in rows) if rows else None for k in keys},
        'minimum': {k: min((r[k] for r in rows), default=None) for k in keys},
        'maximum': {k: max((r[k] for r in rows), default=None) for k in keys},
        'not_generation_usage_or_cost': True, 'solver_data_accessed': False}
    with (B / 'delivery/DIALOGUE_LENGTHS_ALL_ROOTS_V1.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('roots', 'incomplete_or_held_roots_excluded_from_mean')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
