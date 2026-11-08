import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_v7 as builder
import preflight_v7


class FailClosedTests(unittest.TestCase):
    def history(self):
        dims={k:'reviewed descriptive value' for k in ('event','natural_information_need','query_scope','answer_operation','evidence_family')}
        coverage=[{'history_item_id':f'H{i:04d}','review_status':'INDEPENDENTLY_REVIEWED','eligibility':'ELIGIBLE_VERIFIED',**dims} for i in range(135)]
        known=[{'history_item_id':f'H{i:04d}','review_status':'INDEPENDENTLY_ANNOTATED','eligibility':'UNKNOWN',**dims} for i in range(135,200)]
        unknown=[{'history_item_id':f'H{i:04d}','review_status':'UNKNOWN_RETAINED','eligibility':'UNKNOWN',**{k:'UNKNOWN' for k in dims},'unknown_reason':'Semantic record unavailable; novelty comparison unresolved.'} for i in range(200,212)]
        return {'schema_version':'stage_b_history_projection_v1','status':'COMPLETE_FOR_SELECTOR','coverage_set':coverage,'overlap_set':known+unknown,'source_bindings':[{'logical_name':name,'path':'/synthetic','sha256':value} for name,value in builder.PARENT_HASHES.items()],'annotation_audits':{'R02':{'known':31,'unknown':3},'R03':{'known':34,'unknown':9}},'unique_union_count':212}

    def assert_rejected(self, h):
        with self.assertRaises(builder.BuildError): builder.validate_history(h)

    def test_split_selector_sets(self):
        h=self.history(); coverage,overlap=builder.validate_history(h)
        projected=builder.selector_projection(h,coverage,overlap)
        self.assertEqual((len(projected['coverage_set']),len(projected['overlap_set'])),(135,77))
        self.assertNotIn('records',projected)

    def test_reject_partial_unknown_and_not_treat_operation_as_full_tuple(self):
        h=self.history(); h['overlap_set'][0]['natural_information_need']='UNKNOWN'
        self.assert_rejected(h)
        h=self.history(); h['coverage_set'][0]['query_scope']='PENDING independent review'
        self.assert_rejected(h)

    def test_reject_leaked_question_and_scene(self):
        for text in ('Which actions apply?', 'R03:A:S01:T002', 'cc-train-episode-123'):
            h=self.history(); h['overlap_set'][0]['query_scope']=text
            self.assert_rejected(h)

    def test_reject_duplicate_ids_and_bindings(self):
        h=self.history(); h['overlap_set'][0]['history_item_id']=h['coverage_set'][0]['history_item_id']; self.assert_rejected(h)
        h=self.history(); h['source_bindings'].append(copy.deepcopy(h['source_bindings'][0])); self.assert_rejected(h)

    def test_reject_unknown_eligibility_promoted_to_coverage(self):
        h=self.history(); h['overlap_set'][-1]['eligibility']='ELIGIBLE_VERIFIED'; self.assert_rejected(h)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'a.json'; p.write_text('{"key":1,"key":2}')
            with self.assertRaises(builder.BuildError): builder.readj(p)

    def test_builder_remains_closed_even_for_count_valid_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json'; path.write_text(json.dumps(self.history()))
            dest=Path(temp)/'execution'
            with patch.object(builder,'DEST',dest), patch.object(builder,'validate_raw_bindings'), patch.object(builder,'validate_contract_review',return_value='a'*64):
                with self.assertRaisesRegex(builder.BuildError,'HISTORY_BASELINE_INCOMPLETE'): builder.build(path)
            self.assertFalse(dest.exists())

    def test_missing_execution_cannot_pass_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            result=preflight_v7.check_execution(Path(temp))
            self.assertFalse(result['all_pass'])
            self.assertIn('HISTORY_BASELINE_INCOMPLETE', result['issues'][0])


if __name__=='__main__': unittest.main()
