import copy,json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from events import classify,EXACT_WARNING
class EventTests(unittest.TestCase):
    def setUp(self):
        self.final='{"x":1}'
        self.events=[{'type':'thread.started','thread_id':'test'},{'type':'turn.started'},
          {'type':'item.completed','item':{'id':'a','type':'agent_message','text':self.final}},
          {'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}]
    def run_case(self,events=None,final=None,**kw):return classify('\n'.join(json.dumps(e) for e in (events or self.events)),self.final if final is None else final,**kw)
    def test_clean_accept(self):self.assertTrue(self.run_case()['accepted'])
    def test_exact_warning_accept(self):
        self.events.insert(1,{'type':'item.completed','item':{'type':'error','message':EXACT_WARNING}});self.assertTrue(self.run_case()['accepted'])
    def test_prefinal_reconnect_accept(self):
        self.events.insert(2,{'type':'error','message':'Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)'});self.assertTrue(self.run_case()['accepted'])
    def test_postfinal_reconnect_reject(self):
        self.events.insert(3,{'type':'error','message':'Reconnecting... 1/5 (stream disconnected before completion: stream closed before response.completed)'});self.assertFalse(self.run_case()['accepted'])
    def test_unknown_and_tool_reject(self):
        for typ in ['command_execution','tool','function_call','unknown','web_search','mcp_tool_call']:
            with self.subTest(typ=typ):
                e=copy.deepcopy(self.events);e.insert(2,{'type':'item.completed','item':{'type':typ,'text':'x'}});self.assertFalse(self.run_case(e)['accepted'])
    def test_unknown_errors_reject(self):
        for e in [{'type':'error','message':'Reconnecting... 1/5 (different error)'},{'type':'item.completed','item':{'type':'error','message':EXACT_WARNING+'!'}},{'type':'turn.failed'}]:
            v=copy.deepcopy(self.events);v.insert(2,e);self.assertFalse(self.run_case(v)['accepted'])
    def test_completion_usage_required(self):
        self.assertFalse(self.run_case(self.events[:-1])['accepted']);self.events[-1]['usage']={};self.assertFalse(self.run_case()['accepted'])
    def test_matching_message_required(self):self.assertFalse(self.run_case(final='{"x":2}')['accepted'])
    def test_final_json_required(self):self.events[2]['item']['text']='broken';self.assertFalse(self.run_case(final='broken')['accepted'])
    def test_exit_timeout_reject(self):self.assertFalse(self.run_case(returncode=1)['accepted']);self.assertFalse(self.run_case(timed_out=True)['accepted'])
    def test_duplicate_response_reject(self):self.events.insert(3,copy.deepcopy(self.events[2]));self.assertFalse(self.run_case()['accepted'])
    def test_no_semantic_retry_after_malformed_response(self):
        self.events[2]['item']['text']='broken';r=self.run_case(final='broken');self.assertTrue(r['semantic_response_present']);self.assertFalse(r['accepted'])
    def test_invalid_json_stream_reject(self):self.assertFalse(classify('not json',self.final)['accepted'])
if __name__=='__main__':unittest.main()
