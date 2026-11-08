"""Exact ordered constraint-list compatibility; frozen audit and receipts preserved."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_export_v2 import export_root_questions
OLD = "assert receipt['full_source_read'] is True and receipt['all_continuation_constraints_read'] is True"
NEW = "assert receipt['full_source_read'] is True and receipt['all_continuation_constraints_read'] == constraints['continuation_constraints']"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def engine():
    original = HERE / 'audit_content.py'
    bindings = json.loads((HERE.parent / 'evidence/EVALUATION_TOOL_BINDINGS.json').read_text())['files']
    expected = next(x for x in bindings if x['path'] == str(original))
    assert sha(original) == expected['sha256'], 'ORIGINAL_AUDIT_CHANGED'
    source = original.read_text()
    assert source.count(OLD) == 1, 'EXPECTED_EXACT_LIST_PATCH_REQUIRED'
    ns = {'__file__': str(original), '__name__': 'preserved_content_audit_v2'}
    exec(compile(source.replace(OLD, NEW), str(original), 'exec'), ns)
    ns['export_root_questions'] = export_root_questions
    return ns
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wave', choices=['validation','production','all'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    for item in json.loads((HERE.parent/'evidence/AUDIT_RECEIPT_LIST_V2_BINDING.json').read_text())['files']:
        assert sha(item['path']) == item['sha256'], 'ADAPTER_BINDING_CHANGED:'+item['path']
    engine()['main'](a.wave,a.output.resolve())
if __name__=='__main__': main()
