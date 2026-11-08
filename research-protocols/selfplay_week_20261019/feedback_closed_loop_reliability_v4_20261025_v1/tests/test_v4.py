import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline_v4 as p
class V4Tests(unittest.TestCase):
 def setUp(self):
  self.data={'root_id':'r','source_draft':{'path':'/tmp/x','sha256':'0','bytes':0},'turns':[{'turn_ref':'R03:A:S01:T001','speaker':'A','text':'At 10:00, I planned two visits.'},{'turn_ref':'R03:A:S01:T002','speaker':'B','text':'Because we need time, it remains planned.'},{'turn_ref':'R03:A:S01:T003','speaker':'A','text':'The booking is not complete.'},{'turn_ref':'R03:A:S01:T004','speaker':'B','text':'Correction: Tuesday replaces Monday.'},{'turn_ref':'R03:A:S01:T005','speaker':'A','text':'I want a quiet walk.'},{'turn_ref':'R03:A:S01:T006','speaker':'B','text':'Six notes are ready.'}], 'facts':[]}
  for i,t in enumerate(self.data['turns'],1): self.data['facts'].append({'fact_id':f'f{i}','turn_ref':t['turn_ref'],'speaker':t['speaker'],'order':i,'start':0,'end':len(t['text']),'text':t['text'],'category':p.category(t['text'])})
 def test_literal_and_mutation(self):
  s=p.compile_slot(self.data,'F','MC2');p.validate_slot(s,self.data);self.assertEqual(len(s['gold']),2);self.assertTrue(all(v['claim'] in '\n'.join(t['text'] for t in self.data['turns']) for v in s['options'].values() if v['truth']))
 def test_order_and_permutations(self):
  s=p.compile_slot(self.data,'G','ORDER1');p.validate_slot(s,self.data);self.assertEqual(s['gold'],list('ABCDEF'));self.assertTrue(all(v['nonidentity'] for v in s['variants']))
 def test_false_is_not_literal_source(self):
  s=p.compile_slot(self.data,'F','MC5');source='\n'.join(t['text'] for t in self.data['turns']);self.assertTrue(all(o['claim'] not in source for o in s['options'].values() if not o['truth']))
if __name__=='__main__':unittest.main()
