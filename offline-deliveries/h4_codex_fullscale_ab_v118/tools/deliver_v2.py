"""Keep frozen delivery code and fix FP/FN reporting to use canonical Gold IDs.

The original artifact and code stay unchanged. No model calls, source, question,
Gold, scores, or selection policy are edited by this adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_export_v2 import export_root_questions

OLD = "    manifest = read(run / 'manifest.json')\n"
NEW = "    content_gold = read(run / 'gold.json')\n" + OLD
GOLD_OLD = "                expected = set(q['gold'])\n"
GOLD_NEW = "                expected = set(content_gold[r['candidate_id']])\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine():
    original = HERE / 'deliver.py'
    bindings = json.loads((HERE.parent / 'evidence/DELIVERY_TOOL_BINDINGS.json').read_text())['files']
    expected = next(x for x in bindings if x['path'] == str(original))
    assert sha(original) == expected['sha256'], 'ORIGINAL_DELIVERY_CHANGED'
    source = original.read_text()
    assert source.count(OLD) == source.count(GOLD_OLD) == 1, 'EXACT_CANONICAL_GOLD_PATCH_REQUIRED'
    source = source.replace(OLD, NEW).replace(GOLD_OLD, GOLD_NEW)
    ns = {'__file__': str(original), '__name__': 'preserved_delivery_v2'}
    exec(compile(source, str(original), 'exec'), ns)
    ns['export_root_questions'] = export_root_questions
    return ns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', choices=['validation', 'all'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    engine()['summarize'](args.wave, args.output.resolve())


if __name__ == '__main__':
    main()
