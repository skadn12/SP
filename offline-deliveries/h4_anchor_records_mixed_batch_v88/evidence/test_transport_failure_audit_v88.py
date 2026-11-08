"""Exercise strict failed-HTTP audit on real receipts and in-memory corruptions."""
import ast
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import unittest

HOME=Path(__file__).resolve().parents[1]
AUDITOR=HOME/'evidence/audit_generation_v88.py'
tree=ast.parse(AUDITOR.read_text())
namespace={'datetime':datetime}
body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='failed_multiselect_contract']
assert len(body)==1
exec(compile(ast.Module(body=body,type_ignores=[]),'pure_transport_failure_contract','exec'),namespace)
verify=namespace['failed_multiselect_contract']


def fixture():
    run=HOME/'multiselect/assets/root135466_family'
    folder=run/'rounds/supplement'
    request=folder/'requests/001-root135466_family-supplement'
    read=lambda p:json.loads(p.read_text())
    inputs={key:read(base/name) for key,base,name in (
        ('completion',request,'HTTP_RESULT.json'),('error',request,'ERROR.json'),
        ('reservation',request,'RESERVATION.json'),('generation',folder,'GENERATION.COMPLETE.json'),
        ('public',folder,'PUBLIC_ASSETS.json'),('assets',folder,'ASSETS.json'),
        ('judgment',folder,'SOURCE_JUDGMENTS.json'),('review',folder,'ASSET_REVIEW.json'),
        ('receipt',folder,'REVIEW_RECEIPT.json'),('version_map',folder,'VERSION_MAP.json'),
        ('source',run,'SOURCE.json'),('manifest',folder,'MANIFEST.json'))}
    inputs['request_names']=[p.name for p in request.iterdir() if p.is_file()]
    inputs['model']='Qwen/Qwen3.6-35B-A3B'
    return inputs


class FailureContractTests(unittest.TestCase):
    def checked(self,inputs):
        checks=[]
        def check(ok,name):
            checks.append(name)
            if not ok:raise AssertionError(name)
        result=verify(**inputs,check=check)
        self.assertEqual(len(checks),10)
        return result

    def test_actual_503_is_failure_counted_without_response_or_semantic_rejection(self):
        inputs=fixture();before=deepcopy(inputs);result=self.checked(inputs)
        self.assertEqual(inputs,before)
        self.assertEqual(result['outcome'],'TRANSPORT_FAILURE')
        self.assertEqual(result['http_status'],503)
        self.assertEqual(result['generated_candidate_count'],0)
        self.assertEqual(result['semantic_rejection_count'],0)
        self.assertFalse(result['mechanical_validation_performed'])
        self.assertEqual(json.loads(result['service_error_body'])['code'],50508)

    def test_success_status_or_disagreeing_error_status_cannot_use_failure_branch(self):
        for key,value in [('http_status',200),('outcome','response')]:
            inputs=fixture();inputs['completion'][key]=value
            with self.assertRaisesRegex(AssertionError,'failed_transport_http_error_receipt'):self.checked(inputs)
        inputs=fixture();inputs['error']['metadata']['http_status']=502
        with self.assertRaisesRegex(AssertionError,'no_success_or_model_output_claim'):self.checked(inputs)

    def test_retry_extra_attempt_and_unattempted_request_rejected(self):
        changes=[lambda x:x['error']['metadata'].update(transport_retry_count=1),
                 lambda x:x['error']['metadata'].update(http_attempt_count=2),
                 lambda x:x['reservation'].update(request_number=2),
                 lambda x:x['error'].update(http_attempted=False),
                 lambda x:x['error'].update(no_automatic_retry=False)]
        for change in changes:
            inputs=fixture();change(inputs)
            with self.assertRaises(AssertionError):self.checked(inputs)

    def test_response_or_validation_file_with_failed_receipt_rejected(self):
        for name in ('RESPONSE.json','VALIDATION.json','untracked_file.json'):
            inputs=fixture();inputs['request_names'].append(name)
            with self.assertRaisesRegex(AssertionError,'exact_request_inventory'):self.checked(inputs)
        inputs=fixture();inputs['request_names'].remove('ERROR.json')
        with self.assertRaisesRegex(AssertionError,'exact_request_inventory'):self.checked(inputs)

    def test_no_success_model_finish_or_usage_can_be_claimed(self):
        for key,value in [('returned_model','invented'),('finish_reason','stop'),('usage',{}),('output_tokens',0)]:
            inputs=fixture();inputs['error']['metadata'][key]=value
            with self.assertRaisesRegex(AssertionError,'no_success_or_model_output_claim'):self.checked(inputs)

    def test_nonempty_failed_round_assets_or_reviews_rejected(self):
        changes=[lambda x:x['generation'].update(counterversion_count=1),
                 lambda x:x['public']['versions'].append({'proposition':'Invented'}),
                 lambda x:x['assets']['facts'][0].update(contradicted={'proposition':'Invented'}),
                 lambda x:x['judgment']['versions'].append({'relation':'unknown'}),
                 lambda x:x['receipt'].update(usable_contradicts=1)]
        for change in changes:
            inputs=fixture();change(inputs)
            with self.assertRaises(AssertionError):self.checked(inputs)

    def test_empty_review_does_not_permit_missing_parents_or_source_read(self):
        changes=[lambda x:x['judgment']['parents'].pop(),
                 lambda x:x['judgment'].update(full_source_read=False),
                 lambda x:x['judgment'].update(complete_source_turns_read=1),
                 lambda x:x['receipt'].update(source_judgments_bytes_changed_after_private_read=True)]
        for change in changes:
            inputs=fixture();change(inputs)
            with self.assertRaisesRegex(AssertionError,'empty_public_projection_has_complete_source_review'):self.checked(inputs)

    def test_review_cannot_precede_http_failure(self):
        inputs=fixture();inputs['judgment']['created_at_utc']='2026-10-14T00:00:00+00:00'
        with self.assertRaisesRegex(AssertionError,'receipts_precede_saved_empty_review'):self.checked(inputs)


if __name__=='__main__':unittest.main(verbosity=2)
