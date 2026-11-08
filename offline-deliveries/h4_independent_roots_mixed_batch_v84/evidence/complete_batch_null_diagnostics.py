"""Close v84 while distinguishing absent comparison records from SHA bindings.

The original helper is retained. Only the explicit generation-audit diagnostic
old_public_assets entries may have null hashes; all actual bindings stay strict.
"""
from pathlib import Path
import re
import sys

EVIDENCE = Path(__file__).resolve().parent
sys.path.insert(0, str(EVIDENCE))
import complete_batch as original


def unavailable_comparisons():
    audit = original.read(original.HOME / 'evidence/GENERATION_AUDIT.json')
    config = original.read(original.HOME / 'CONFIG.json')
    old_batches = (
        'h4_ready_roots_mixed_batch_v78',
        'h4_output_contract_mixed_batch_v79',
        'h4_scoped_ordering_mixed_batch_v80',
        'h4_bound_scene_ordering_mixed_batch_v81',
    )
    expected = {
        str((original.HOME.parent / batch / 'multiselect/assets' / slot['slot_id'] /
             'rounds/initial/PUBLIC_ASSETS.json').resolve()): slot['root_id']
        for slot in config['multiselect_config']['slots'] for batch in old_batches
    }
    allowed = {}
    for index, row in enumerate(audit['repeated_per_fact_replacement_texts']):
        binding = row['old_public_assets']
        if binding['sha256'] is not None:
            continue
        path = binding['path']
        original.need(path in expected and row['root_id'] == expected[path],
                      'unexpected_unavailable_comparison_path')
        original.need(row['comparison_status'] == 'not_available' and
                      row['identical_per_fact_replacements'] is None,
                      'null_comparison_has_invalid_status')
        original.need(not Path(path).exists(), 'unavailable_comparison_file_now_exists')
        original.need(path not in allowed.values(), 'duplicate_unavailable_comparison')
        allowed[index] = path
    original.need(set(allowed.values()) == set(expected),
                  'v84_unavailable_comparison_coverage_changed')
    return allowed


ALLOWED_NULLS = unavailable_comparisons()


def verify_exact_bindings(value):
    bindings = {}

    def visit(obj, location=()):
        if isinstance(obj, dict):
            if set(obj) == {'path', 'sha256'}:
                path, digest = obj['path'], obj['sha256']
                if digest is None:
                    original.need(
                        len(location) == 4 and
                        location[:2] == ('GENERATION_AUDIT', 'repeated_per_fact_replacement_texts') and
                        type(location[2]) is int and location[3] == 'old_public_assets' and
                        ALLOWED_NULLS.get(location[2]) == path and not Path(path).exists(),
                        'unexpected_null_binding:' + str(path))
                    return
                original.need(isinstance(path, str) and isinstance(digest, str) and
                              re.fullmatch(r'[0-9a-f]{64}', digest) is not None,
                              'invalid_sha256_binding')
                original.need(path not in bindings or bindings[path] == digest,
                              'conflicting_artifact_binding')
                bindings[path] = digest
            else:
                for key, item in obj.items():
                    visit(item, location + (key,))
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                visit(item, location + (index,))

    visit(value)
    original.verify_bindings([{'path': path, 'sha256': digest}
                              for path, digest in bindings.items()])


original.verify_exact_bindings = verify_exact_bindings

if __name__ == '__main__':
    original.main()
