import contextlib
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import run_gate_comparison as runner
from types import SimpleNamespace

class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        return json.dumps(messages)
    def encode(self,text,**kwargs):
        return range(max(1,len(text)//8))

class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.run=Path(self.tmp.name)/'run'
        with patch.object(runner,'resolve_credential',return_value='mock'),patch.object(socket.socket,'connect',side_effect=AssertionError('network')):
            runner.prepare(self.run,tokenizer=Tokenizer())
        self.sent=[]
    def tearDown(self):self.tmp.cleanup()
    def factory(self,profile,*,on_http_attempt):
        parent=self
        class Client:
            def complete(self,*,prompt,max_tokens,json_object):
                outbound=runner.build_payload(profile,prompt)
                on_http_attempt(outbound);parent.sent.append(outbound)
                root=runner.ROOTS[len(parent.sent)-1]
                output=runner.read(parent.run/'frozen'/root/'qwen_response.json')
                metadata={'actual_model':'mock-tju','finish_reason':'stop','completion_status':'RESPONSE_RECEIVED',
                    'input_tokens':100,'output_tokens':10,'total_tokens':110,'reasoning_tokens':None,
                    'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110},
                    'http_attempt_count':1,'transport_attempt_count':1,'transport_retry_count':0}
                return SimpleNamespace(raw_text=json.dumps(output),metadata=metadata)
        return Client()
    def test_exact_two_messages_and_baseline_parameters_with_no_resume(self):
        with patch.object(socket.socket,'connect',side_effect=AssertionError('network')),contextlib.redirect_stdout(io.StringIO()):
            result=runner.execute(self.run,client_factory=self.factory,tokenizer=Tokenizer())
        self.assertEqual(len(self.sent),2)
        self.assertEqual(result['cost']['http_attempts'],2)
        self.assertEqual(result['cost']['retries'],0)
        for root,payload in zip(runner.ROOTS,self.sent):
            prior=runner.read(self.run/'frozen'/root/'request.json')['payload']
            self.assertEqual({k:v for k,v in payload.items() if k!='model'},{k:v for k,v in prior.items() if k!='model'})
            self.assertEqual(payload['model'],'tju-llm')
        with self.assertRaises(runner.DevelopmentRuntimeError):
            runner.execute(self.run,client_factory=self.factory,tokenizer=Tokenizer())
        self.assertEqual(len(self.sent),2)
    def test_technical_failure_consumes_one_and_stops_unsent_root(self):
        from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import ComparisonTransportError
        parent=self
        def factory(profile,*,on_http_attempt):
            class Client:
                def complete(self,*,prompt,max_tokens,json_object):
                    p=runner.build_payload(profile,prompt);on_http_attempt(p);parent.sent.append(p)
                    raise ComparisonTransportError('timeout_unknown',{'timeout':True,'completion_status':'UNKNOWN'})
            return Client()
        with patch.object(socket.socket,'connect',side_effect=AssertionError('network')),contextlib.redirect_stdout(io.StringIO()):
            result=runner.execute(self.run,client_factory=factory,tokenizer=Tokenizer())
        self.assertEqual(result['status'],'STOPPED')
        self.assertEqual(result['cost']['http_attempts'],1)
        self.assertEqual(result['unsent_roots'],[runner.ROOTS[1]])
        self.assertEqual(len(self.sent),1)
    def test_changed_frozen_request_cannot_send(self):
        f=self.run/'frozen'/runner.ROOTS[0]/'strong_request.json'
        f.write_text('{}')
        with self.assertRaises(ValueError):
            runner.execute(self.run,client_factory=self.factory,tokenizer=Tokenizer())
        self.assertEqual(self.sent,[])

if __name__=='__main__':unittest.main()
