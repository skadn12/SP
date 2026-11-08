import copy,json,unittest
from event_ledger import classify,WARNING,turn_fact,mechanical,inputs
from common import BASE,PILOT,load,local

FINAL='{"ok":true}'
def stream():
    return [{'type':'thread.started','thread_id':'fixed-test'}, {'type':'turn.started'},
       {'type':'item.completed','item':{'type':'agent_message','text':FINAL}},
       {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}]
class Events(unittest.TestCase):
    def test_clean(self):self.assertTrue(classify(stream(),FINAL)['accepted'])
    def test_exact_warnings(self):
        e=stream();e.insert(1,{'type':'item.completed','item':{'type':'error','message':WARNING}})
        self.assertTrue(classify(e,FINAL)['accepted'])
    def test_prefinal_reconnect(self):
        e=stream();e.insert(2,{'type':'error','message':'Reconnecting... 2/5 (stream disconnected before completion: stream closed before response.completed)'})
        self.assertTrue(classify(e,FINAL)['accepted'])
    def test_postfinal_reconnect_reject(self):
        e=stream();e.insert(3,{'type':'error','message':'Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)'})
        self.assertFalse(classify(e,FINAL)['accepted'])
    def test_unknowns_reject(self):
        for typ in ['command_execution','command','tool','function_call','mcp_tool_call','web_search','file_change','unknown','error']:
            e=stream();e.insert(2,{'type':'item.completed','item':{'type':typ,'text':'{}','message':'unknown'}})
            with self.subTest(typ=typ):self.assertFalse(classify(e,FINAL)['accepted'])
    def test_unknown_lifecycle_reject(self):
        for typ in ['item.started','turn.failed','foo','response.completed']:
            e=stream();e.insert(2,{'type':typ});self.assertFalse(classify(e,FINAL)['accepted'])
        e=stream();e[1]['item']={'type':'function_call','text':'{}'}
        self.assertFalse(classify(e,FINAL)['accepted'])
    def test_reconnect_before_active_turn_rejected(self):
        e=stream();e.insert(1,{'type':'error','message':'Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)'})
        self.assertFalse(classify(e,FINAL)['accepted'])
    def test_changed_warning_reject(self):
        e=stream();e.insert(1,{'type':'item.completed','item':{'type':'error','message':WARNING+' '}})
        self.assertFalse(classify(e,FINAL)['accepted'])
    def test_unknown_reconnect_reject(self):
        for msg in ['Reconnecting... 1/5 (unknown error)','stream closed before response.completed','Reconnecting... 6/5 (stream disconnected before completion: stream closed before response.completed)']:
            e=stream();e.insert(2,{'type':'error','message':msg});self.assertFalse(classify(e,FINAL)['accepted'])
    def test_missing_or_mismatching_final(self):
        for final in ['', '{"ok":false}', '{} garbage','[] garbage','true']:
            self.assertFalse(classify(stream(),final)['accepted'])
    def test_completion_required(self):
        self.assertFalse(classify(stream()[:-1],FINAL)['accepted'])
        e=stream();e[-1]['usage']={};self.assertFalse(classify(e,FINAL)['accepted'])
    def test_single_turn_only(self):self.assertFalse(classify(stream()+stream(),FINAL)['accepted'])
    def test_prefinal_agent_text_last_message_matches(self):
        e=stream();e.insert(2,{'type':'item.completed','item':{'type':'agent_message','text':'Interim text'}})
        self.assertTrue(classify(e,FINAL)['accepted'])
        e[2]['item']['type']='function_call'
        self.assertFalse(classify(e,FINAL)['accepted'])
    def test_failed_transport_with_final(self):
        self.assertFalse(classify(stream(),FINAL,returncode=1)['accepted'])
        self.assertFalse(classify(stream(),FINAL,timed_out=True)['accepted'])
    def test_malformed_and_blank(self):
        self.assertFalse(classify('invalid',FINAL)['accepted'])
        self.assertFalse(classify('\n'.join(json.dumps(e) for e in stream())+'\n\n',FINAL)['accepted'])
class Ledger(unittest.TestCase):
    def test_exact_span_and_canonical_negation(self):
        t={'turn_ref':'R03:A:S01:T001','speaker':'A','text':"I haven't booked it;  09:00 remains proposed."}
        f=turn_fact(t,'R03:A:S01');s=f['span']
        self.assertEqual(t['text'][s['char_start']:s['char_end']],s['quote'])
        self.assertIn("haven't",f['canonical_fact']['proposition'])
        self.assertIn('proposed',f['explicit_state_tokens'])
    def test_outside_write_rejected(self):
        with self.assertRaises(ValueError):local(BASE.parent/'forbidden.json')
    def test_native_quotes_validated_against_exact_original(self):
        from source_engine import quote_errors
        ledger={'facts':[],'native_utterances':[{'source_id':'native:test','turns':[{'turn_id':'4','speaker':'SYSTEM','utterance':'The booking is not confirmed.'}]}]}
        self.assertEqual(quote_errors([{'turn_ref':'native:test:4','quote':'not confirmed'}],ledger),[])
        self.assertTrue(quote_errors([{'turn_ref':'native:test:4','quote':'is confirmed'}],ledger))
        self.assertTrue(quote_errors([{'turn_ref':'native:test:5','quote':'not confirmed'}],ledger))
    def test_pilot_fixed_order(self):
        from common import ROOTS
        self.assertEqual(PILOT,ROOTS[:4])
        self.assertEqual(PILOT,['cc-train-episode-109164','cc-train-episode-136202','cc-train-episode-14304','cc-train-episode-156136'])
    def test_missing_mapping_and_wrong_quotes_rejected(self):
        inp,parent,native=inputs(PILOT[0]);c=load(BASE/'sources'/PILOT[0]/'CANDIDATE_0.json')
        self.assertEqual(mechanical(c,inp,parent,native),[])
        bad=copy.deepcopy(c);bad['native_evidence'].pop()
        self.assertIn('missing_native_mapping',[e['code'] for e in mechanical(bad,inp,parent,native)])
        bad=copy.deepcopy(c);bad['native_evidence'][0]['source_evidence'][0]['quote']='Fabricated quotation'
        self.assertIn('source_span_mapping_mismatch',[e['code'] for e in mechanical(bad,inp,parent,native)])
    def test_wrong_actor_and_changed_native_are_rejected(self):
        inp,parent,native=inputs(PILOT[0]);c=load(BASE/'sources'/PILOT[0]/'CANDIDATE_0.json')
        bad=copy.deepcopy(c);bad['scenes'][0]['turns'][0]['speaker']='Unestablished person'
        self.assertIn('new_speaker',[e['code'] for e in mechanical(bad,inp,parent,native)])
        bad=copy.deepcopy(c);bad['native_evidence'][0]['native_quote']='Fabricated native utterance'
        self.assertIn('native_quote_mismatch',[e['code'] for e in mechanical(bad,inp,parent,native)])

class Questions(unittest.TestCase):
    def test_graph_requires_unique_total_order_and_gold_identity(self):
        from question_engine import production
        c={'items':{k:k for k in 'ABCDEF'},'gold':list('ABCDEF')}
        edges=[{'before':a,'after':b} for a,b in zip('ABCDE','BCDEF')]
        self.assertTrue(production._graph_audit(c,edges)['graph_order_matches_gold'])
        self.assertFalse(production._graph_audit(c,edges[:-1])['unique_topological_order'])
        self.assertFalse(production._graph_audit(c,edges+[{'before':'F','after':'A'}])['acyclic'])
        c['gold']=list('BACDEF')
        self.assertFalse(production._graph_audit(c,edges)['graph_order_matches_gold'])
    def test_all_slot_schemas_enforce_quota(self):
        from question_engine import schema,SLOTS
        for slot,(_,_,n) in SLOTS.items():
            s=schema(PILOT[0],'G',slot,{'path':'/test','sha256':'0'*64,'bytes':1})
            self.assertEqual(s['properties']['gold']['minItems'],n)
            self.assertEqual(s['properties']['gold']['maxItems'],n)
            self.assertFalse(s['additionalProperties'])
    def test_source_response_is_not_a_review_verdict(self):
        from source_engine import review_errors
        ledger={'facts':[],'native_utterances':[]}
        raw_source={'root_id':PILOT[0],'scenes':[],'native_evidence':[]}
        es=review_errors({'accepted':True,'response':raw_source},PILOT[0],ledger,['R03:A:S01'])
        self.assertIn('semantic_source_not_pass',es)
        self.assertIn('incomplete_scene_review',es)

class Concurrency(unittest.TestCase):
    def test_three_slots_shared_across_processes(self):
        import pathlib,subprocess,sys,tempfile,time
        from codex_client import global_slot
        import codex_client
        with tempfile.TemporaryDirectory(dir=BASE/'runtime',prefix='slot_test_') as tmp:
            old=codex_client.BASE;codex_client.BASE=pathlib.Path(tmp)
            try:
                with global_slot(),global_slot(),global_slot():
                    script=('import pathlib;import codex_client;codex_client.BASE=pathlib.Path('+repr(tmp)+');'
                            'ctx=codex_client.global_slot();ctx.__enter__();print("ACQUIRED",flush=True);ctx.__exit__(None,None,None)')
                    p=subprocess.Popen([sys.executable,'-B','-c',script],cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    time.sleep(.2)
                    self.assertIsNone(p.poll(),'fourth process acquired an occupied semantic slot')
                out,err=p.communicate(timeout=5)
                self.assertEqual(p.returncode,0,err.decode())
                self.assertEqual(out.decode().strip(),'ACQUIRED')
            finally:codex_client.BASE=old

class Solver(unittest.TestCase):
    def test_initialization_recovery_never_repeats_submitted_work(self):
        from solver_engine import validate_initialization_recovery
        validate_initialization_recovery({'returned_count':0},0,0)
        for returned,submitted,raw in [(1,0,0),(0,1,0),(0,0,1),(1,1,1)]:
            with self.subTest(returned=returned,submitted=submitted,raw=raw):
                with self.assertRaises(RuntimeError):validate_initialization_recovery({'returned_count':returned},submitted,raw)
    def test_relative_ipc_path_binds_inside_run(self):
        import os,pathlib,socket,uuid
        from solver_engine import runtime_env
        env=runtime_env();self.assertEqual(env['VLLM_RPC_BASE_PATH'],'.ipc')
        relative='.ipc/test-'+uuid.uuid4().hex
        self.assertLess(len(relative.encode()),107)
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            self.assertEqual(pathlib.Path.cwd().resolve(),BASE)
            sock.bind(relative)
            self.assertTrue((BASE/relative).is_socket())
        finally:
            sock.close()
            if (BASE/relative).exists():(BASE/relative).unlink()

if __name__=='__main__':unittest.main()
