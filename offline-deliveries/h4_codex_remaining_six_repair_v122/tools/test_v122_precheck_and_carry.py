"""Synthetic checks for new schema retry and exact historical carry adapters."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import historical_fixed_import_v122 as carry
import test_public_anchor_workflow_v1 as fixture


class PrecheckWorkflowTests(unittest.TestCase):
    setUp = fixture.PublicAnchorWorkflowTests.setUp

    def start_public(self):
        fixture.record(self.batch,self.unit,self.folder,self.helper,'asset')
        with redirect_stdout(io.StringIO()):self.helper.start(self.batch,fixture.ROOT,'public',fixture.REVIEWER)
        return fixture.semantic_judgment(self.folder,'public')

    def pre(self,raw,label):
        p=self.unit/(label+'.json');fixture.write(p,raw)
        with redirect_stdout(io.StringIO()):
            result=self.helper.preflight_review(self.batch,fixture.ROOT,'public',fixture.REVIEWER,p)
        return p,result

    def test_missing_identifier_can_be_added_without_second_semantic_judgment(self):
        raw=self.start_public();cid=next(iter(raw['questions']));del raw['questions'][cid]['candidate_id']
        p,first=self.pre(raw,'missing');self.assertFalse(first['pass'])
        self.assertIn('CANDIDATE',first['error'])
        raw['questions'][cid]['candidate_id']=cid
        p,second=self.pre(raw,'corrected');self.assertTrue(second['pass'],second)
        self.assertEqual(len(second['allowed_identifier_delta']),1)
        with redirect_stdout(io.StringIO()):self.helper.save_review(self.batch,fixture.ROOT,'public',fixture.REVIEWER,p)
        result=fixture.gate.check(self.batch,fixture.ROOT,'public')
        self.assertTrue(result['pass'],result['errors'])

    def test_reason_change_after_schema_failure_is_preserved_and_blocked(self):
        raw=self.start_public();cid=next(iter(raw['questions']));del raw['questions'][cid]['candidate_id']
        self.pre(raw,'missing');raw['questions'][cid]['candidate_id']=cid
        raw['questions'][cid]['reason']='Changed semantic justification.'
        p,result=self.pre(raw,'changed_reason');self.assertFalse(result['pass'])
        self.assertIn('SEMANTIC_PAYLOAD_CHANGED',result['error'])
        with self.assertRaisesRegex(ValueError,'PASSING_SCHEMA_PRECHECK_IDENTITY_REQUIRED'):
            self.helper.save_review(self.batch,fixture.ROOT,'public',fixture.REVIEWER,p)
        directory=fixture.review_paths(self.unit,'public')
        self.assertFalse((directory/'RAW_JUDGMENT.json').exists())
        self.assertFalse((directory/'SAVE_FAILURE.json').exists())
        self.assertEqual(len(list((directory/'SCHEMA_CANDIDATES').glob('*.json'))),2)

    def test_save_without_precheck_cannot_consume_raw(self):
        raw=self.start_public();p=self.unit/'unprechecked.json';fixture.write(p,raw)
        with self.assertRaisesRegex(ValueError,'PASSING_SCHEMA_PRECHECK_REQUIRED'):
            self.helper.save_review(self.batch,fixture.ROOT,'public',fixture.REVIEWER,p)
        self.assertFalse((fixture.review_paths(self.unit,'public')/'RAW_JUDGMENT.json').exists())

    def test_precheck_receipt_required_by_gate(self):
        fixture.record(self.batch,self.unit,self.folder,self.helper,'asset')
        directory=fixture.review_paths(self.unit,'asset')
        next(directory.glob('PRECHECK_*.json')).unlink()
        result=fixture.gate.check(self.batch,fixture.ROOT,'asset')
        self.assertFalse(result['pass'])
        self.assertIn('PRECHECK:FILE_MISSING',{e['code'] for e in result['errors']})


class FixedCarryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.batch=Path(self.tmp.name)/'batch';self.unit=self.batch/'units/root146700'
        self.folder=self.unit/'roots/root146700';self.old=Path(self.tmp.name)/'history'
        self.old.mkdir(parents=True);self.folder.mkdir(parents=True)
        original={'CUMULATIVE_SOURCE.json':{'root_id':'cc-train-episode-146700','turns':[{'text':'Synthetic fixed source.'}]},
                  'ORDERING_ANCHOR_PLAN.json':{'slots':[{'candidate_id':'historical-fixed-id','anchors':[{'item_id':'I1','anchor_id':'A1'}]}]},
                  'QA_DRAFT.json':{'author':'/root/historical_fixture_author','facts':[{'true_text':'Exact immutable proposition.'}]},
                  'QA_AUTHOR_RECEIPT.json':{'author':'/root/historical_fixture_author','executor':'/root/historical_fixture_author','source_sha256':'synthetic','ordering_anchor_plan_binding':{'path':'/old/path','sha256':'synthetic'}}}
        for n,d in original.items():carry.save(self.old/n,d)
        policy={'allowed_roots':['cc-train-episode-146700','cc-train-episode-254852'],
                'qa':{'root_id':'cc-train-episode-146700','candidate_namespace':'historical-fixed-namespace',
                      'artifacts':{n:carry.bind(self.old/n) for n in original}},'scene':{}}
        self.pp=self.batch/'evidence/FIXED_IMPORT_POLICY.json';carry.save(self.pp,policy)
        carry.save(self.batch/'MASTER_FREEZE.json',{'files':[carry.bind(self.pp),carry.bind(carry.__file__)]})
        carry.copy(self.old/'CUMULATIVE_SOURCE.json',self.folder/'CUMULATIVE_SOURCE.json')
        for n in ('ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json'):carry.copy(self.old/n,self.folder/n)
        spec,old,adapted=carry.qa_expected(self.batch,self.unit,'cc-train-episode-146700',False)
        carry.save(self.folder/'QA_AUTHOR_RECEIPT.json',adapted)
        carry.save(self.folder/'HISTORICAL_QA_IMPORT.json',{'root_id':'cc-train-episode-146700','policy_binding':carry.bind(self.pp),
            'historical_author':adapted['author'],'artifacts':{n:{'original_binding':carry.bind(self.old/n),'imported_binding':carry.bind(self.folder/n)} for n in ('ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json')},
            'adapted_qa_receipt_binding':carry.bind(self.folder/'QA_AUTHOR_RECEIPT.json')})

    def test_exact_import_retains_historical_author(self):
        _,_,receipt=carry.qa_expected(self.batch,self.unit,'cc-train-episode-146700')
        self.assertEqual(receipt['author'],'/root/historical_fixture_author')
        self.assertTrue(receipt['historical_import_only'])
        self.assertFalse(receipt['new_qa_authoring_attempt'])

    def test_fixed_question_text_edit_is_rejected(self):
        p=self.folder/'QA_DRAFT.json';obj=carry.read(p);obj['facts'][0]['true_text']='Changed proposition.'
        p.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'FIXED_QA_CONTENT_CHANGED'):
            carry.qa_expected(self.batch,self.unit,'cc-train-episode-146700')

    def test_anchor_label_edit_is_rejected(self):
        p=self.folder/'ORDERING_ANCHOR_PLAN.json';obj=carry.read(p);obj['slots'][0]['anchors'][0]['item_id']='I2';p.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError,'FIXED_QA_CONTENT_CHANGED'):
            carry.qa_expected(self.batch,self.unit,'cc-train-episode-146700')

    def test_other_root_cannot_claim_fixed_qa_exception(self):
        with self.assertRaisesRegex(ValueError,'FIXED_QA_IMPORT_ROOT'):
            carry.qa_expected(self.batch,self.unit,'cc-train-episode-197635')

class FixedSceneCarryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.batch=Path(self.tmp.name)/'batch';self.unit=self.batch/'units/root254852'
        self.folder=self.unit/'roots/root254852/scenes/A_S02'
        previous=self.unit/'roots/root254852/scenes/A_S01/CUMULATIVE_SOURCE.json'
        carry.save(previous,{'root_id':'cc-train-episode-254852','turns':[{'text':'Synthetic retained prefix.'}]})
        self.original=Path(self.tmp.name)/'history/EXPANSION_DRAFT.json'
        carry.save(self.original,{'root_id':'cc-train-episode-254852','scene_ref':'R02:A:S02',
                   'author':'/root/historical_fixture_author','input_source_sha256':carry.sha(previous),
                   'turns':[{'text':'Synthetic fixed A2 content.'}]})
        self.pp=self.batch/'evidence/FIXED_IMPORT_POLICY.json'
        carry.save(self.pp,{'allowed_roots':['cc-train-episode-146700','cc-train-episode-254852'],
                   'scene':{'root_id':'cc-train-episode-254852','scene_ref':'R02:A:S02','draft_binding':carry.bind(self.original)}})
        carry.save(self.batch/'MASTER_FREEZE.json',{'files':[carry.bind(self.pp),carry.bind(carry.__file__)]})
        carry.copy(self.original,self.folder/'EXPANSION_DRAFT.json')
        carry.save(self.folder/'HISTORICAL_SCENE_IMPORT.json',{'root_id':'cc-train-episode-254852','scene_ref':'R02:A:S02',
                   'policy_binding':carry.bind(self.pp),'historical_author':'/root/historical_fixture_author','new_authoring_attempt':False,
                   'original_draft_binding':carry.bind(self.original),'imported_draft_binding':carry.bind(self.folder/'EXPANSION_DRAFT.json')})

    def test_exact_fixed_scene_retains_old_author(self):
        _,_,draft=carry.scene_expected(self.batch,self.unit,'cc-train-episode-254852','R02:A:S02')
        self.assertEqual(draft['author'],'/root/historical_fixture_author')

    def test_fixed_scene_dialogue_edit_rejected(self):
        p=self.folder/'EXPANSION_DRAFT.json';d=carry.read(p);d['turns'][0]['text']='Changed text.';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'FIXED_SCENE_DRAFT_BINDING'):
            carry.scene_expected(self.batch,self.unit,'cc-train-episode-254852','R02:A:S02')

    def test_other_scene_cannot_claim_carry_exception(self):
        with self.assertRaisesRegex(ValueError,'FIXED_SCENE_IMPORT_SCOPE'):
            carry.scene_expected(self.batch,self.unit,'cc-train-episode-254852','R02:A:S03')

    def test_changed_predecessor_blocks_import(self):
        p=self.unit/'roots/root254852/scenes/A_S01/CUMULATIVE_SOURCE.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'FIXED_SCENE_PREDECESSOR_CHANGED'):
            carry.scene_expected(self.batch,self.unit,'cc-train-episode-254852','R02:A:S02')


class NewSceneLanguageTests(unittest.TestCase):
    def test_new_chinese_turn_is_blocked(self):
        import scene_controller_v120 as controller
        with self.assertRaisesRegex(ValueError,'NEW_SCENE_CJK_TEXT_BLOCKED'):
            controller.new_scene_english_guard({'turns':[{'speaker':'Existing','text':'这是中文新场。'}]})

    def test_english_new_turn_passes_mechanical_guard(self):
        import scene_controller_v120 as controller
        controller.new_scene_english_guard({'turns':[{'speaker':'Existing','text':'This new scene is written in English.'}]})


if __name__=='__main__':unittest.main()
