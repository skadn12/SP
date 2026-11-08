"""Offline tests of v88 row-based parent schema and current-only audit paths."""
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import unittest

BATCH = Path(__file__).resolve().parents[1]
EVIDENCE = BATCH / 'evidence'
ROOTS = tuple('cc-train-episode-' + r for r in ('103866', '109164', '112812', '135466'))
AUDITORS = ('audit_generation_v88.py', 'audit_results_v88.py')

def functions(filename):
    tree = ast.parse((EVIDENCE / filename).read_text())
    names = {'v88_parent_judgments_supported', 'validate_admission_contract', 'registry_contract_errors'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'ROOTS': ROOTS, 'datetime': datetime, 'Path': Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename + ':pure_functions', 'exec'), namespace)
    return namespace


def synthetic_fixture():
    tree = ast.parse((EVIDENCE / 'test_audit_bindings_v88.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fixture']
    namespace = {'ROOTS': ROOTS, 'Path': Path, 'deepcopy': deepcopy, 'hashlib': hashlib, 'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'legacy_fixture_only', 'exec'), namespace)
    home, data, read, bind, config, registry, sources = namespace['fixture']()
    for slot in config['multiselect_config']['slots']:
        judgment = data[slot['parent_review_path']]
        judgment['version'] = 'v88_parent_source_judgments_v1'
        judgment.pop('content_status')
        provenance = data[slot['parent_provenance_path']]
        provenance['parent_review_sha256'] = bind(slot['parent_review_path'])['sha256']
        slot['parent_review_sha256'] = provenance['parent_review_sha256']
        slot['parent_provenance_sha256'] = bind(slot['parent_provenance_path'])['sha256']
    return home, data, read, bind, config, registry, sources


class ParentSchemaRecoveryTests(unittest.TestCase):
    def actual(self):
        path = BATCH / 'parent_review/root103866_SOURCE_JUDGMENTS.json'
        judgment = json.loads(path.read_text())
        judgment.setdefault('coverage', {'options_reviewed':list('ABCDEF'), 'option_count':6, 'supported_count':6, 'contradicted_count':0, 'unknown_count':0})
        return judgment, judgment['public_question']

    def test_both_auditors_have_identical_strict_parent_functions(self):
        signatures = []
        for name in AUDITORS:
            tree = ast.parse((EVIDENCE / name).read_text())
            signatures.append([ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                ('v88_parent_judgments_supported', 'validate_admission_contract', 'registry_contract_errors')])
        self.assertEqual(*signatures)

    def test_actual_four_reviews_pass_with_or_without_optional_content_summary(self):
        for name in AUDITORS:
            validate = functions(name)['v88_parent_judgments_supported']
            for path in sorted((BATCH / 'parent_review').glob('*_SOURCE_JUDGMENTS.json')):
                judgment = json.loads(path.read_text())
                self.assertTrue(validate(judgment, judgment['public_question']), (name, path))
                judgment.pop('content_status', None)
                self.assertTrue(validate(judgment, judgment['public_question']), (name, path, 'without optional summary'))

    def test_full_admission_preserves_binding_and_time_contract_without_summary(self):
        for name in AUDITORS:
            home, data, read, bind, config, _, _ = synthetic_fixture()
            checks = []
            functions(name)['validate_admission_contract'](config, home, read, bind,
                lambda ok, label, *detail: checks.append((ok, label)))
            self.assertTrue(checks)
            self.assertTrue(all(ok for ok, label in checks), [label for ok, label in checks if not ok])

    def test_unknown_false_missing_duplicate_wrong_version_and_bad_summary_rejected(self):
        changes = {
            'unknown': lambda j: j['judgments'][0].update(source_judgment='unknown'),
            'contradicted': lambda j: j['judgments'][0].update(source_judgment='contradicted'),
            'not_entailed': lambda j: j['judgments'][0].update(entailed=False),
            'bool_not_int': lambda j: j['judgments'][0].update(entailed=1),
            'missing_support': lambda j: j['judgments'][0].pop('source_judgment'),
            'missing_entailed': lambda j: j['judgments'][0].pop('entailed'),
            'missing_row': lambda j: j['judgments'].pop(),
            'duplicate_option': lambda j: j['judgments'][0].update(option='B'),
            'missing_evidence': lambda j: j['judgments'][0].pop('required_evidence'),
            'empty_evidence': lambda j: j['judgments'][0].update(required_evidence=[]),
            'empty_quote': lambda j: j['judgments'][0]['required_evidence'][0].update(quote=''),
            'missing_reason': lambda j: j['judgments'][0].pop('reason'),
            'wrong_version': lambda j: j.update(version='v87_parent_source_judgments_v1'),
            'incompatible_summary': lambda j: j.update(content_status='CORE_UNUSABLE'),
            'null_summary': lambda j: j.update(content_status=None),
            'missing_coverage_option': lambda j: j['coverage']['options_reviewed'].pop(),
            'duplicate_coverage_option': lambda j: j['coverage']['options_reviewed'].__setitem__(0, 'B'),
            'wrong_coverage_counts': lambda j: j['coverage'].update(supported_count=5),
            'unknown_coverage_count': lambda j: j['coverage'].update(unknown_count=1),
        }
        for name in AUDITORS:
            validate = functions(name)['v88_parent_judgments_supported']
            for label, change in changes.items():
                with self.subTest(auditor=name, mutation=label):
                    judgment, public = self.actual(); change(judgment)
                    self.assertFalse(validate(judgment, public))
            judgment, public = self.actual(); judgment['content_status'] = 'CORE_USABLE'
            self.assertTrue(validate(judgment, public))
            judgment['judgments'][0]['entailed'] = False
            self.assertFalse(validate(judgment, public), 'summary must not override failed rows')

    def test_full_admission_still_rejects_inexact_evidence_and_private_timing(self):
        for name in AUDITORS:
            for mutation in ('quote', 'private_timing'):
                home, data, read, bind, config, _, _ = synthetic_fixture()
                slot = config['multiselect_config']['slots'][0]
                judgment = data[slot['parent_review_path']]
                provenance = data[slot['parent_provenance_path']]
                if mutation == 'quote':
                    judgment['judgments'][0]['required_evidence'][0]['quote'] = 'Not in original source'
                else:
                    provenance['current_private_parent_construction_at_utc'] = '2026-10-14T06:00:00+00:00'
                provenance['parent_review_sha256'] = bind(slot['parent_review_path'])['sha256']
                slot['parent_review_sha256'] = provenance['parent_review_sha256']
                slot['parent_provenance_sha256'] = bind(slot['parent_provenance_path'])['sha256']
                checks = []
                functions(name)['validate_admission_contract'](config, home, read, bind,
                    lambda ok, label, *detail: checks.append((ok, label)))
                failures = [label for ok, label in checks if not ok]
                expected = 'fresh_parent_exact_source_evidence:' if mutation == 'quote' else 'fresh_parent_provenance_saved_before_current_private:'
                self.assertTrue(any(label.startswith(expected) for label in failures), failures)

    def test_three_auditor_current_paths_and_historical_paths_are_distinct(self):
        generation = (EVIDENCE / AUDITORS[0]).read_text()
        result = (EVIDENCE / AUDITORS[1]).read_text()
        union = (EVIDENCE / 'audit_union_v88.py').read_text()
        self.assertIn("OUT = HOME / 'evidence/GENERATION_AUDIT.json'", generation)
        self.assertIn("'INPUT_AUDIT.json' if args.inputs else 'RESULT_AUDIT.json'", result)
        self.assertIn("HOME/'evidence/RESULT_AUDIT.json'", union)
        self.assertIn("HOME/'evidence/GENERATION_AUDIT.json'", union)
        self.assertIn("PREVIOUS/'evidence/RESULT_AUDIT.json'", union)
        self.assertNotIn("PREVIOUS/'evidence/RESULT_AUDIT_v2.json'", union)
        self.assertNotIn('def validate_admission_contract', union, 'union has no separate parent admission function')
        self.assertIn("previous / 'evidence/RESULT_AUDIT.json'", (EVIDENCE / 'export_audited.py').read_text())

if __name__ == '__main__':
    unittest.main(verbosity=2)
