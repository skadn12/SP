"""Offline regression tests for observed native failure receipts; no model calls.

Execute the actual auditor functions by AST extraction. The round disclosure
validator is the unchanged frozen native schema, and fixtures come from saved
failed requests rather than synthesized success records.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCHEMA_PATH = HERE.parents[1] / 'h4_output_contract_multiselect_v79/review_schema.py'
FUNCTION_NAMES = {'failed_transport_contract', 'failed_multiselect_contract', 'sha', 'check_binding'}
REJECTED = (AssertionError, ValueError, KeyError, TypeError)


def fail_check(value, name, detail=None):
    if not value:
        raise AssertionError(name)


def load_contracts():
    spec = importlib.util.spec_from_file_location('v89_frozen_native_review_schema', SCHEMA_PATH)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    def load_multi(name):
        if name != 'review_schema':
            raise AssertionError('unexpected_module:' + name)
        return schema
    path = HERE / 'audit_generation_core_v89.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTION_NAMES]
    if {n.name for n in nodes} != FUNCTION_NAMES:
        raise AssertionError('auditor_function_missing')
    env = {'datetime': datetime, 'hashlib': hashlib, 'Path': Path,
           'common': SimpleNamespace(load_multi=load_multi), 'require': fail_check}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def read(path):
    return json.loads(Path(path).read_text())


def request_fixture(request):
    error = read(request / 'ERROR.json')
    return {'completion': read(request / 'HTTP_RESULT.json'), 'error': error,
            'reservation': read(request / 'RESERVATION.json'),
            'request_names': sorted(p.name for p in request.iterdir() if p.is_file()),
            'model': error['metadata']['requested_model'], 'check': fail_check}


def round_fixture(home, family, round_id):
    run = home / 'multiselect/assets' / family
    folder = run / 'rounds' / round_id
    requests = list((folder / 'requests').iterdir())
    if len(requests) != 1:
        raise AssertionError('fixture_one_request_required')
    fixture = request_fixture(requests[0])
    fixture.update({key: read(folder / filename) for key, filename in (
        ('generation', 'GENERATION.COMPLETE.json'), ('public', 'PUBLIC_ASSETS.json'),
        ('assets', 'ASSETS.json'), ('judgment', 'SOURCE_JUDGMENTS.json'),
        ('review', 'ASSET_REVIEW.json'), ('receipt', 'REVIEW_RECEIPT.json'),
        ('version_map', 'VERSION_MAP.json'), ('manifest', 'MANIFEST.json'))})
    fixture['source'] = read(run / 'SOURCE.json')
    return fixture


class GenerationCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = Path(ARGS.home).resolve()
        cls.env = load_contracts()
        cls.multi = round_fixture(cls.home, 'root160549_family', 'initial')
        cls.other_multi = round_fixture(cls.home, 'root194259_family', 'supplement')
        cls.http = request_fixture(cls.home / 'multiselect/assets/root160549_family/rounds/initial/requests/001-root160549_family-initial')
        cls.timeout = request_fixture(cls.home / 'ordering/root194259_ordering/requests/001-root194259_ordering')

    def transport(self, fixture):
        return self.env['failed_transport_contract'](**fixture)

    def multiselect(self, fixture):
        return self.env['failed_multiselect_contract'](**fixture)

    def reject_changes(self, fixture, cases, contract):
        for label, change in cases:
            with self.subTest(change=label):
                altered = deepcopy(fixture)
                change(altered)
                with self.assertRaises(REJECTED):
                    contract(altered)

    def test_native_source_judgment_without_source_first_passes_without_mutation(self):
        for fixture in (self.multi, self.other_multi):
            with self.subTest(root=fixture['manifest']['root_id']):
                self.assertEqual(fixture['judgment']['review_schema_version'], 'source_judgment_v1')
                self.assertNotIn('source_first', fixture['judgment'])
                before = deepcopy(fixture)
                result = self.multiselect(fixture)
                self.assertEqual(result['failure_kind'], 'HTTP_ERROR')
                self.assertTrue(result['empty_source_review_complete'])
                self.assertEqual(result['public_variant_count'], 0)
                self.assertEqual(fixture, before)
        explicit_true = deepcopy(self.multi)
        explicit_true['judgment']['source_first'] = True
        self.assertTrue(self.multiselect(explicit_true)['empty_source_review_complete'])
        for value in (False, 1, 'true'):
            with self.subTest(invalid_optional_source_first=value):
                invalid = deepcopy(self.multi)
                invalid['judgment']['source_first'] = value
                with self.assertRaises(REJECTED):
                    self.multiselect(invalid)

    def test_native_disclosures_cannot_be_missing_wrong_or_inferred(self):
        expected = {'parent_all_true_provenance_known': True, 'parent_gold_known_before_review': True,
                    'fresh_blind_parent_review': False, 'current_solver_scores_read': False,
                    'solver_scores_used': False, 'new_variant_private_artifacts_read_before_save': False,
                    'full_source_read': True, 'derived_from_rounds': False}
        cases = []
        for key, value in expected.items():
            if key != 'derived_from_rounds':
                cases.append(('missing_' + key, lambda f, key=key: f['judgment'].pop(key)))
            cases.append(('wrong_' + key, lambda f, key=key, value=value: f['judgment'].update({key: not value})))
            cases.append(('numeric_' + key, lambda f, key=key, value=value: f['judgment'].update({key: int(value)})))
        cases.extend([
            ('missing_schema', lambda f: f['judgment'].pop('review_schema_version')),
            ('unsupported_schema', lambda f: f['judgment'].update(review_schema_version='source_judgment_v0')),
            ('missing_turn_count', lambda f: f['judgment'].pop('complete_source_turns_read')),
            ('incomplete_source', lambda f: f['judgment'].update(complete_source_turns_read=len(f['source']['turns']) - 1)),
            ('bool_turn_count', lambda f: f['judgment'].update(complete_source_turns_read=True))])
        self.reject_changes(self.multi, cases, self.multiselect)

    def test_native_private_projections_and_zero_candidate_claims_stay_strict(self):
        cases = []
        for record in ('review', 'receipt'):
            cases.append((record + '_missing_source_first', lambda f, record=record: f[record].pop('source_first')))
            cases.append((record + '_false_source_first', lambda f, record=record: f[record].update(source_first=False)))
        cases.extend([
            ('review_incomplete_source', lambda f: f['review'].update(full_source_read=False)),
            ('judgments_changed_after_private', lambda f: f['receipt'].update(source_judgments_bytes_changed_after_private_read=True)),
            ('invented_usable_contradiction', lambda f: f['receipt'].update(usable_contradicts=1)),
            ('invented_public_version', lambda f: f['public']['versions'].append({'fact_id': 'F01'})),
            ('invented_review_version', lambda f: f['review']['versions'].append({'fact_id': 'F01'})),
            ('invented_judgment_version', lambda f: f['judgment']['versions'].append({'fact_id': 'F01'})),
            ('invented_asset_version', lambda f: f['assets']['facts'][0].update(contradicted={'proposition': 'invented'})),
            ('changed_parent', lambda f: f['judgment']['parents'][0].update(proposition='changed text')),
            ('missing_parent', lambda f: f['judgment']['parents'].pop()),
            ('generated_counterversion', lambda f: f['generation'].update(counterversion_count=1)),
            ('semantic_retry', lambda f: f['generation'].update(semantic_retries=1)),
            ('mechanical_repair', lambda f: f['generation'].update(mechanical_repairs=1)),
            ('transport_retry', lambda f: f['generation'].update(transport_retries=1)),
            ('training_claim', lambda f: f['generation'].update(training_allowed=True))])
        self.reject_changes(self.multi, cases, self.multiselect)

    def test_saved_source_review_time_chain_cannot_be_missing_naive_or_reordered(self):
        timestamps = (('reservation', 'reserved_at'), ('completion', 'http_completed_at_utc'),
                      ('generation', 'completed_at_utc'), ('judgment', 'created_at_utc'),
                      ('receipt', 'private_crosscheck_started_at_utc'), ('receipt', 'review_completed_at_utc'))
        cases = []
        for record, field in timestamps:
            cases.append(('missing_' + field, lambda f, record=record, field=field: f[record].pop(field)))
            cases.append(('naive_' + field, lambda f, record=record, field=field: f[record].update({field: '2026-10-14T12:41:00'})))
        cases.extend([
            ('missing_saved_time', lambda f: f['receipt'].pop('source_judgments_saved_at_utc')),
            ('saved_time_not_created', lambda f: f['receipt'].update(source_judgments_saved_at_utc='2026-10-14T12:41:46.146196+00:00')),
            ('source_saved_after_private', lambda f: f['receipt'].update(private_crosscheck_started_at_utc='2026-10-14T00:00:00+00:00')),
            ('completion_before_reservation', lambda f: f['completion'].update(http_completed_at_utc='2026-10-13T00:00:00+00:00')),
            ('generation_after_source_save', lambda f: f['generation'].update(completed_at_utc='2026-10-15T00:00:00+00:00')),
            ('review_completed_before_private', lambda f: f['receipt'].update(review_completed_at_utc='2026-10-14T00:00:00+00:00'))])
        self.reject_changes(self.multi, cases, self.multiselect)

    def test_native_http_error_and_timeout_pass_as_distinct_nonresponses(self):
        for fixture, kind in ((self.http, 'HTTP_ERROR'), (self.timeout, 'TIMEOUT')):
            with self.subTest(kind=kind):
                before = deepcopy(fixture)
                result = self.transport(fixture)
                self.assertEqual(result['failure_kind'], kind)
                self.assertEqual(result['outcome'], 'TRANSPORT_FAILURE')
                self.assertEqual(result['generated_candidate_count'], 0)
                self.assertFalse(result['mechanical_validation_performed'])
                self.assertEqual(result['semantic_rejection_count'], 0)
                self.assertEqual(result['retry_count'], 0)
                self.assertEqual(fixture, before)
                if kind == 'TIMEOUT':
                    self.assertIsNone(result['http_status'])
                    self.assertIsNone(result['service_error_body'])
                    self.assertTrue(result['timeout'])
                    for key in ('http_status', 'raw_response_text', 'raw_response'):
                        self.assertNotIn(key, fixture['error']['metadata'])
                else:
                    self.assertEqual(result['http_status'], 503)
                    self.assertEqual(result['service_error_body'], fixture['error']['metadata']['raw_response_text'])
                    self.assertFalse(result['timeout'])

    def test_error_and_timeout_branches_are_mutually_exclusive(self):
        self.reject_changes(self.timeout, [
            ('null_status_without_timeout', lambda f: f['error']['metadata'].update(timeout=False)),
            ('missing_timeout', lambda f: f['error']['metadata'].pop('timeout')),
            ('numeric_timeout', lambda f: f['error']['metadata'].update(timeout=1)),
            ('timeout_embedded_status', lambda f: f['error']['metadata'].update(http_status=None)),
            ('timeout_embedded_body', lambda f: f['error']['metadata'].update(raw_response_text='service text')),
            ('timeout_embedded_raw_response', lambda f: f['error']['metadata'].update(raw_response={})),
            ('timeout_with_http_status', lambda f: f['completion'].update(http_status=503))], self.transport)
        self.reject_changes(self.http, [
            ('http_timeout_true', lambda f: f['error']['metadata'].update(timeout=True)),
            ('http_missing_timeout', lambda f: f['error']['metadata'].pop('timeout')),
            ('success_status_as_failure', lambda f: (f['completion'].update(http_status=200), f['error']['metadata'].update(http_status=200))),
            ('null_completion_http_metadata', lambda f: f['completion'].update(http_status=None)),
            ('wrong_metadata_status', lambda f: f['error']['metadata'].update(http_status=500)),
            ('float_metadata_status', lambda f: f['error']['metadata'].update(http_status=503.0)),
            ('http_missing_body', lambda f: f['error']['metadata'].pop('raw_response_text')),
            ('http_empty_body', lambda f: f['error']['metadata'].update(raw_response_text=' \n')),
            ('http_nonstring_body', lambda f: f['error']['metadata'].update(raw_response_text={})),
            ('http_parsed_response', lambda f: f['error']['metadata'].update(raw_response={}))], self.transport)

    def test_no_success_model_output_or_untyped_failure_can_be_claimed(self):
        fields = ('actual_model', 'returned_model', 'finish_reason', 'usage',
                  'input_tokens', 'output_tokens', 'reasoning_tokens', 'total_tokens')
        cases = [('non_null_' + key, lambda f, key=key: f['error']['metadata'].update({key: 'false-success'})) for key in fields]
        cases.extend([
            ('wrong_requested_model', lambda f: f['error']['metadata'].update(requested_model='other-model')),
            ('success_completion_status', lambda f: f['error']['metadata'].update(completion_status='COMPLETED')),
            ('unsupported_completion_status', lambda f: f['error']['metadata'].update(completion_status='REJECTED')),
            ('wrong_error_type', lambda f: f['error'].update(error_type='RuntimeError')),
            ('http_not_attempted', lambda f: f['error'].update(http_attempted=False)),
            ('numeric_http_attempted', lambda f: f['error'].update(http_attempted=1)),
            ('automatic_retry_claim', lambda f: f['error'].update(no_automatic_retry=False)),
            ('raw_exception_text_saved', lambda f: f['error'].update(raw_exception_text_saved=True)),
            ('extra_error_field', lambda f: f['error'].update(extra='invented')),
            ('extra_completion_field', lambda f: f['completion'].update(extra='invented')),
            ('response_outcome', lambda f: f['completion'].update(outcome='response'))])
        for fixture in (self.http, self.timeout):
            self.reject_changes(fixture, cases, self.transport)

    def test_bool_counters_retries_multiple_attempts_and_residual_success_files_fail(self):
        cases = []
        for key, value in {'transport_attempt_count': 1, 'http_attempt_count': 1,
                           'transport_retry_count': 0, 'rate_limit_retry_count': 0}.items():
            cases.append(('bool_' + key, lambda f, key=key, value=value: f['error']['metadata'].update({key: bool(value)})))
            cases.append(('excess_' + key, lambda f, key=key, value=value: f['error']['metadata'].update({key: value + 1})))
        for key, value in {'maximum': 1, 'request_number': 1, 'automatic_retries': 0}.items():
            cases.append(('bool_reservation_' + key, lambda f, key=key, value=value: f['reservation'].update({key: bool(value)})))
            cases.append(('excess_reservation_' + key, lambda f, key=key, value=value: f['reservation'].update({key: value + 1})))
        cases.extend([
            ('residual_response', lambda f: f['request_names'].append('RESPONSE.json')),
            ('residual_validation', lambda f: f['request_names'].append('VALIDATION.json')),
            ('missing_error_file', lambda f: f['request_names'].remove('ERROR.json'))])
        for fixture in (self.http, self.timeout):
            self.reject_changes(fixture, cases, self.transport)

    def test_original_caller_binding_checks_and_immutable_sha_guard_are_preserved(self):
        original = HERE / 'auditor_revisions/revision01_original/audit_generation_core_v89.py'
        def binding_nodes(path):
            tree = ast.parse(path.read_text())
            audit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
            names = {'failed_transport_error_and_request_bound_in_round_completion:',
                     'failed_transport_empty_projection_review_bindings:'}
            matches = []
            for node in ast.walk(audit):
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'recorded':
                    matches.append(ast.dump(node, include_attributes=False))
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'require':
                    if any(isinstance(v, ast.Constant) and v.value in names for v in ast.walk(node)):
                        matches.append(ast.dump(node, include_attributes=False))
            self.assertEqual(len(matches), 3)
            return matches
        self.assertEqual(binding_nodes(original), binding_nodes(HERE / 'audit_generation_core_v89.py'))
        with tempfile.TemporaryDirectory(prefix='v89-immutable-binding-') as task_tmp:
            path = Path(task_tmp) / 'SOURCE_JUDGMENTS.json'
            path.write_text('{"fixture":true}\n')
            row = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.env['check_binding'](row)
            path.write_text('{"fixture":false}\n')
            with self.assertRaises(AssertionError):
                self.env['check_binding'](row)
        for fixture in (self.multi, self.other_multi):
            for row in fixture['generation']['files'] + fixture['receipt']['files']:
                self.env['check_binding'](row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', required=True)
    ARGS, rest = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
