import json,unittest
from codex_client import strict_response
from event_policy import classify_events,EXACT_STARTUP_WARNING
from source_resume import mechanical,review_ok
class Tests(unittest.TestCase):
 def stream(self):return [{'type':'thread.started','thread_id':'test'},{'type':'item.completed','item':{'type':'error','message':EXACT_STARTUP_WARNING}},{'type':'turn.started'},{'type':'item.completed','item':{'type':'agent_message','text':'{"decision":"PASS"}'}},{'type':'turn.completed'}]
 def test_exact_warning_and_response(self):
  c=classify_events(self.stream());self.assertTrue(c['accepted']);self.assertEqual(strict_response('{"decision":"PASS"}',c)[1],[])
 def test_near_warning_rejected(self):
  e=self.stream();e[1]['item']['message']+=' ';self.assertFalse(classify_events(e)['accepted'])
 def test_tools_unknown_and_stream_error(self):
  for e in [{'type':'error','message':'reconnect'},{'type':'item.completed','item':{'type':'function_call'}},{'type':'item.started','item':{'type':'command_execution'}}]:self.assertFalse(classify_events(self.stream()+[e])['accepted'])
 def test_malformed_missing_timeout(self):
  for raw in ['',None,'{bad','{}\n\n']:self.assertFalse(classify_events(raw)['accepted'])
  self.assertFalse(classify_events(self.stream(),timed_out=True)['accepted'])
 def test_missing_completion_and_mismatch(self):
  self.assertIn('missing_turn_completed',strict_response('{"decision":"PASS"}',classify_events(self.stream()[:-1]))[1]);self.assertIn('last_message_event_mismatch',strict_response('{}',classify_events(self.stream()))[1])
 def test_invalid_json_response(self):self.assertIn('invalid_or_missing_response_json',strict_response('no',classify_events(self.stream()))[1])
 def test_mechanical_incomplete_source(self):self.assertIn('not_48_new_turns',mechanical({'root_id':'r','source_status':'s','scenes':[]},{'root_id':'r','source_status':'s','scene_sequence':[]},{'turns':[]}))
 def test_pass_requires_real_quote(self):
  c={'root_id':'r','scenes':[]};r={'accepted':True,'response':{'root_id':'r','decision':'PASS','source_quotes':[{'turn_ref':'t','quote':'invented'}],'failure_reasons':[]}}
  self.assertIn('invalid_review_quote',review_ok(r,c,{'turns':[{'turn_ref':'t','text':'actual'}]}))
if __name__=='__main__':unittest.main()
