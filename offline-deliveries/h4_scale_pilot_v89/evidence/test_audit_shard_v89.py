"""Offline negative tests for independent scoring and saved-record validation."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PROJECT=Path('/root/autodl-tmp/gxy-selfplay')
sys.path.insert(0,str(HERE))
import audit_results_core_v89 as results
from audit_contract_v89 import source_contract,validate_admission_contract


def fail_check(value,name,detail=None):
    if not value:
        raise AssertionError(name)


class IndependentAuditTests(unittest.TestCase):
    def test_raw_scoring_and_selection_have_all_select_and_order_negative_cases(self):
        a=results.Audit();results.selfcheck(a)
        self.assertTrue(all(c['status']=='PASS' for c in a.checks))
        req={'variant':{'display_to_content':dict(zip('ABCDEF','123456'))}}
        q={'question_type':'multi_select','candidate_id':'fixture','root_id':'fixture'}
        for g in (2,3,4,5):
            self.assertEqual(results.score('<box>A,B,C,D,E,F</box>',q,req,list('123456')[:g])['partial_score_fraction'],'0')
        for bad in ('A,C','<box>A,A</box>','<box>G</box>','<box>A</box> explanation'):
            with self.assertRaises(ValueError):results.score(bad,q,req,list('12'))
        rows=[{'variant_id':f'perm_{i:02d}','independent_score':{'partial_score_fraction':'0','exact_match':False}} for i in range(6) for _ in range(3)]
        self.assertFalse(results.select_independent(rows,q)['selected_by_policy'])
        for row in rows:row['independent_score']['partial_score_fraction']='1/2'
        self.assertTrue(results.select_independent(rows,q)['selected_by_policy'])
        for row in rows:row['independent_score'].update(partial_score_fraction='1',exact_match=True)
        self.assertFalse(results.select_independent(rows,q)['selected_by_policy'])

    def test_seeded_displays_are_gold_blind_and_type_specific(self):
        base={'candidate_id':'fixture_question','root_id':'fixture','question_type':'multi_select','options':{c:'statement '+c for c in 'ABCDEF'},'gold':['A','C']}
        altered=deepcopy(base);altered['gold']=list('ABCDEF')
        self.assertEqual(results.variants(base),results.variants(altered));self.assertEqual(len(results.variants(base)),6)
        ordering={**base,'question_type':'ordering','items':{'I'+str(i):'statement '+str(i) for i in range(1,7)}}
        self.assertEqual(len(results.variants(ordering)),12)
        self.assertEqual(len({tuple(v['display_order_content_ids']) for v in results.variants(ordering)}),12)

    def test_real_source_contract_rejects_missing_coverage_inferred_exposure_and_hold(self):
        home=Path(ARGS.home).resolve(); config=json.loads((home/'ADMISSION_INPUT.json').read_text()); row=config['roots'][0]
        source=json.loads(Path(row['source']['path']).read_text());review=json.loads(Path(row['review']['path']).read_text())
        source_contract(source,review,row,fail_check)
        variants=[]
        v=deepcopy(review);v['turn_refs_read']=v['turn_refs_read'][:-1];variants.append(v)
        v=deepcopy(review);v['fresh_blind_review']=True;variants.append(v)
        v=deepcopy(review);v['current_question_gold_read']=True;variants.append(v)
        v=deepcopy(review);v['units'][0]['decision']='HOLD';variants.append(v)
        v=deepcopy(review);v['decision']='HOLD';variants.append(v)
        v=deepcopy(review);v['checks_completed'].pop('actor_identity');variants.append(v)
        for v in variants:
            with self.subTest(change=v):
                with self.assertRaises(AssertionError):source_contract(source,v,row,fail_check)

    def test_real_parent_and_provenance_tampering_are_rejected(self):
        home=Path(ARGS.home).resolve();a=results.Audit();config=a.read(home/'CONFIG.json')
        validate_admission_contract(config,home,a.read,a.bind,fail_check)
        slot=config['multiselect_config']['slots'][0]; review=Path(slot['parent_review_path']).resolve(); provenance=Path(slot['parent_provenance_path']).resolve()
        cases=[(review,lambda v:v['judgments'][0].update(source_judgment='unknown')),
               (review,lambda v:v.update(current_parent_private_gold_read_before_judgments_saved=True)),
               (review,lambda v:v['judgments'][0]['required_evidence'][0].update(quote='fixture quote absent from real source')),
               (provenance,lambda v:v.update(current_private_parent_created_after_saved_source_judgments=False)),
               (provenance,lambda v:v.update(created_at_utc='2000-01-01T00:00:00+00:00'))]
        for target,mutate in cases:
            def read(path):
                value=a.read(path)
                if Path(path).resolve()==target:mutate(value)
                return value
            with self.assertRaises(AssertionError):validate_admission_contract(config,home,read,a.bind,fail_check)

    def test_real_audit_hook_blocks_network_subprocess_and_artifact_mutation(self):
        script = r"""
import ast,json,os,socket,subprocess,sys,tempfile,types
from pathlib import Path
sys.dont_write_bytecode=True
path=Path(sys.argv[1]); tree=ast.parse(path.read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
node=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='guard')
with tempfile.TemporaryDirectory(prefix='v89-audit-guard-') as task_tmp:
 home=Path(task_tmp); output=home/'evidence.json'; opened=set(); args=types.SimpleNamespace(stage='inputs')
 env={'Path':Path,'os':os,'home':home,'output':output,'opened':opened,'args':args}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
 sys.addaudithook(env['guard'])
 blocked=[]
 for label,action in [('network',lambda:socket.socket()),('subprocess',lambda:subprocess.Popen([sys.executable,'-B','-c','pass'])),
                      ('artifact_write',lambda:Path('/root/v89-audit-forbidden-write').open('w')),
                      ('solver_response',lambda:(home/'solver/trained/responses/fixture.json').open())]:
  try:action()
  except RuntimeError as error:
   assert str(error).startswith(('audit_forbids_external:','audit_forbids_artifact_write:','input_audit_forbids_solver_response_read:'))
   blocked.append(label)
  else:raise AssertionError('guard_failed:'+label)
 with output.open('x') as stream:json.dump({'blocked':blocked},stream)
 print(output.read_text())
"""
        run=subprocess.run([sys.executable,'-B','-c',script,str(HERE/'audit_shard.py')],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(run.stdout)['blocked'],['network','subprocess','artifact_write','solver_response'])

    def test_independent_reader_rejects_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory(prefix='v89-audit-json-') as task_tmp:
            p=Path(task_tmp)/'invalid.json'
            for value in ('{"x":1,"x":2}','{"x":NaN}'):
                p.write_text(value)
                with self.assertRaises(ValueError):results.Audit().read(p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--home',required=True);ARGS,rest=parser.parse_known_args()
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
