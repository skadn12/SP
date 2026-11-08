"""Standard-library fixtures for mixed row admission and ordering scoring.

Load only pure function ASTs; do not install audit hooks or invoke a formal audit.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re
import unittest

HOME=Path(__file__).resolve().parents[1]
tree=ast.parse((HOME/'evidence/audit_results_v83.py').read_text())
names={'typed_pool_training_forbidden','content_ids','variants','score','select_independent','aggregate'}
module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
namespace={'Fraction':Fraction,'hashlib':hashlib,'random':random,'re':re,'SEED':2026100800,
           'POLICY':'r02_groundless_full_only_random_position_v2'}
exec(compile(module,'pure_auditor_fixture_functions','exec'),namespace)
ban=namespace['typed_pool_training_forbidden']


class TypedSchemaTests(unittest.TestCase):
    def fixture(self):
        return ({'training_allowed':False,'formal_training_admission':False,
                 'questions':[{'question_type':'multi_select','training_allowed':False},
                              {'question_type':'ordering'}]},
                {'status':'FROZEN','training_allowed':False})

    def test_ordering_absence_or_explicit_false_inherits_both_bans(self):
        pool,frozen=self.fixture(); self.assertTrue(ban(pool,frozen))
        pool['questions'][1]['training_allowed']=False; self.assertTrue(ban(pool,frozen))

    def test_pool_and_frozen_ban_cannot_be_missing_or_weakened(self):
        for container,field in [('pool','training_allowed'),('pool','formal_training_admission'),('frozen','training_allowed')]:
            for value in (True,None,0,'false','MISSING'):
                with self.subTest(container=container,field=field,value=value):
                    pool,frozen=self.fixture(); obj=pool if container=='pool' else frozen
                    if value=='MISSING':obj.pop(field)
                    else:obj[field]=value
                    self.assertFalse(ban(pool,frozen))
        pool,frozen=self.fixture(); frozen['status']='DRAFT'; self.assertFalse(ban(pool,frozen))

    def test_multiselect_cannot_omit_its_explicit_ban(self):
        pool,frozen=self.fixture(); pool['questions'][0].pop('training_allowed')
        self.assertFalse(ban(pool,frozen))

    def test_ordering_cannot_explicitly_allow_or_use_falsy_nonboolean(self):
        for value in (True,None,0,'false'):
            pool,frozen=self.fixture(); pool['questions'][1]['training_allowed']=value
            self.assertFalse(ban(pool,frozen))

    def test_unknown_question_type_cannot_inherit_ordering_schema(self):
        pool,frozen=self.fixture(); pool['questions'][1]['question_type']='other'
        self.assertFalse(ban(pool,frozen))

    def test_ordering_exact_and_reverse_scores_and_missing_item(self):
        q={'question_type':'ordering'}; mapping={'I'+str(i):str(i) for i in range(1,7)}
        req={'variant':{'display_to_content':mapping}}; gold=list('123456'); score=namespace['score']
        self.assertEqual(score('<box>I1,I2,I3,I4,I5,I6</box>',q,req,gold)['partial_score_fraction'],'1')
        self.assertEqual(score('<box>I6,I5,I4,I3,I2,I1</box>',q,req,gold)['partial_score_fraction'],'-1')
        with self.assertRaisesRegex(ValueError,'incomplete_order'):score('<box>I1,I2,I3,I4,I5</box>',q,req,gold)

    def test_actual_frozen_mixed_schema_and_request_permutations(self):
        read=lambda p:json.loads(p.read_text())
        pool=read(HOME/'EVALUATION_POOL.json'); frozen=read(HOME/'BATCH_REVIEW.COMPLETE.json')
        self.assertTrue(ban(pool,frozen))
        ordering=[q for q in pool['questions'] if q['question_type']=='ordering']; self.assertTrue(ordering)
        for side in ('base','trained'):
            manifest=read(HOME/'solver'/side/'manifest.json')
            for q in ordering:
                expected=namespace['variants'](q)
                actual=[r['variant'] for r in manifest['requests'] if r['candidate_id']==q['candidate_id']]
                self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main(verbosity=2)
