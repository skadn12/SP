import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import postcheck as p
from scripts.current_method.restart_n5_spell.backends import BackendResponse
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import ComparisonTransportError, build_payload

class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):return json.dumps(messages)
 def encode(self,text,**kwargs):return [0]*100

class Tests(unittest.TestCase):
 def setUp(self):
  block=patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK_FORBIDDEN'))
  block.start();self.addCleanup(block.stop)
  self.slot=p.SLOTS[0]
  q=p.SOURCE/'qa/roots'/self.slot['root_id']
  pool=p.read(q/'candidate_pool.json')
  c=next(c for c in pool if c['candidate_id']==self.slot['candidate_id'])
  self.raw=p.read(q/'raw/responses'/f"{c['admission_request_id']}.json.txt")['candidates'][self.slot['raw_row_index']-1]
  self.source=p.read(q/'qa_source.json')
  self.review={'version':p.VERSION,'candidate_id':self.slot['candidate_id'],
   'item_checks':[{'item_id':n['item_id'],'support':'SUPPORTED','evidence_refs':n['evidence_refs'],'reason':'Mock only, not semantic approval'} for n in self.raw['event_nodes']],
   'original_relation_checks':[{'ordered_pair':e['ordered_pair'],'support':'SUPPORTED','evidence_refs':e['evidence_refs'],'reason':'Mock only'} for e in self.raw['temporal_relations']],
   'question_check':{'status':'CONSISTENT','reason':'Mock only','suggested_question':None},
   'supported_relations':[{'ordered_pair':e['ordered_pair'],'evidence_refs':e['evidence_refs'],'reason':'Mock only'} for e in self.raw['temporal_relations']],
   'decision':'READY','decision_reason':'Mock only'}
  self.profile=copy.deepcopy(p.read(p.SOURCE/'frozen/provider_profile.json'))
  self.profile['request_parameters']={'temperature':.1,'max_tokens':8192,'response_format':{'type':'json_object'}}
  self.profile['output_limit']=8192
 def test_id_only_materialization_and_source_immutable(self):
  before=copy.deepcopy(self.raw)
  result=p.materialize(self.review,self.slot,self.raw,self.source)
  self.assertTrue(result['program_acceptance'])
  self.assertEqual(self.raw,before)
  self.assertEqual(result['semantic_support_status'],'NOT_ASSESSED_BY_PROGRAM')
  byref={r['turn_ref']:r['text'] for r in self.source['turns']}
  for e in result['rebound_contract']['event_nodes']+result['rebound_contract']['temporal_relations']:
   for sp in e['evidence_spans']:self.assertEqual(sp['quote'],byref[sp['turn_ref']])
 def test_unknown_ref_fails(self):
  self.review['item_checks'][0]['evidence_refs']=['NOT_REAL']
  self.assertIn('EVIDENCE_ID_UNKNOWN',p.validate_review(self.review,self.slot,self.raw,self.source)['errors'])
 def test_duplicate_node_fails(self):
  self.review['item_checks'][1]=copy.deepcopy(self.review['item_checks'][0])
  self.assertIn('ITEM_COVERAGE_INVALID',p.validate_review(self.review,self.slot,self.raw,self.source)['errors'])
 def test_missing_original_edge_check_fails(self):
  self.review['original_relation_checks'].pop()
  self.assertIn('ORIGINAL_RELATION_COVERAGE_INVALID',p.validate_review(self.review,self.slot,self.raw,self.source)['errors'])
 def test_ready_with_false_item_fails(self):
  self.review['item_checks'][0]['support']='UNSUPPORTED'
  self.assertFalse(p.materialize(self.review,self.slot,self.raw,self.source)['program_acceptance'])
 def test_reject_not_materialized(self):
  self.review['decision']='REJECT'
  self.assertNotIn('rebound_contract',p.materialize(self.review,self.slot,self.raw,self.source))
 def test_nonunique_graph_not_accepted(self):
  self.review['supported_relations']=self.review['supported_relations'][:1]
  r=p.materialize(self.review,self.slot,self.raw,self.source)
  self.assertFalse(r['program_acceptance'])
  self.assertIn('ORDER_NOT_DETERMINED_BY_CONSTRAINTS',r['graph_errors'])
 def test_extra_item_rewrite_forbidden(self):
  self.review['item_checks'][0]['item_text']='Changed meaning'
  self.assertEqual(p.validate_review(self.review,self.slot,self.raw,self.source)['status'],'FAIL')
 def test_minor_stem_fix_explicit(self):
  self.review['question_check']={'status':'MINOR_WORDING_FIX','reason':'Mock','suggested_question':'Order the four unchanged events.'}
  self.assertEqual(p.materialize(self.review,self.slot,self.raw,self.source)['rebound_contract']['question'],'Order the four unchanged events.')
 def factory(self,mode='ok'):
  parent=self
  def make(profile,*,on_http_attempt):
   class Mock:
    def complete(self,*,prompt,max_tokens,json_object):
     on_http_attempt(build_payload(profile,prompt))
     if mode=='double':on_http_attempt(build_payload(profile,prompt))
     if mode=='timeout':raise ComparisonTransportError('timeout_unknown',{'timeout':True,'completion_status':'UNKNOWN','http_attempt_count':1,'transport_retry_count':0})
     meta={'actual_model':'mock','finish_reason':'length' if mode=='length' else 'stop','completion_status':'RESPONSE_RECEIVED','usage':{'prompt_tokens':10,'completion_tokens':10,'total_tokens':20},'input_tokens':10,'output_tokens':10,'total_tokens':20,'http_status':200,'timeout':False}
     text='{"x":1,"x":2}' if mode=='duplicatejson' else json.dumps(parent.review)
     return BackendResponse(text,'mock',meta)
   return Mock()
  return make
 def test_runtime_success_no_resume(self):
  with tempfile.TemporaryDirectory() as d:
   with p.PostcheckCaller(d,self.profile,Tokenizer(),client_factory=self.factory()) as c:
    c.call(self.slot['root_id'],'Q01',{'messages':[{'role':'user','content':'offline mock'}]},8192)
    self.assertEqual(c.state['http_attempt_count'],1)
    with self.assertRaises(p.DevelopmentRuntimeError):c.call(self.slot['root_id'],'Q01',{'prompt':'again'},8192)
   with self.assertRaises(p.DevelopmentRuntimeError):p.PostcheckCaller(d,self.profile,Tokenizer(),client_factory=self.factory())
 def test_runtime_failures_consume_one_stop_all(self):
  for mode in ('timeout','length','double','duplicatejson'):
   with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
    with p.PostcheckCaller(d,self.profile,Tokenizer(),client_factory=self.factory(mode)) as c:
     with self.assertRaises(p.DevelopmentRuntimeError):c.call(self.slot['root_id'],'Q01',{'prompt':'mock'},8192)
     self.assertEqual(c.state['http_attempt_count'],1)
     if mode!='duplicatejson':self.assertEqual(c.state['status'],'STOPPED')
 def test_global_budget_and_slot_before_http(self):
  with tempfile.TemporaryDirectory() as d:
   with p.PostcheckCaller(d,self.profile,Tokenizer(),client_factory=self.factory()) as c:
    c.state['http_attempt_count']=8
    with self.assertRaises(p.DevelopmentRuntimeError):c.call(self.slot['root_id'],'Q01',{'prompt':'mock'},8192)
    self.assertEqual(c.state['http_attempt_count'],8)
  with tempfile.TemporaryDirectory() as d:
   with p.PostcheckCaller(d,self.profile,Tokenizer(),client_factory=self.factory()) as c:
    with self.assertRaises(p.DevelopmentRuntimeError):c.call(self.slot['root_id'],'Q08',{'prompt':'mock'},8192)
    self.assertEqual(c.state['http_attempt_count'],0)

if __name__=='__main__':unittest.main()
