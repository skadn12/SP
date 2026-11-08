#!/usr/bin/env python3
"""CPU-only tests of the v2 provider guard. No secrets, network or pilot data."""
import copy, hashlib, importlib.util, json, tempfile
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
CLIENT=WORK/'execution/provider_client.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())

class SyntheticResponse:
    status=200
    def __init__(self,value):self.body=json.dumps(value).encode()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return self.body

def run():
    backend=load(WORK/'EXECUTION_AMENDMENT.json')['backend']
    endpoint=backend['endpoint'];assert endpoint.endswith('/responses')
    base=endpoint[:-len('/responses')];model=backend['model_requested']
    spec=importlib.util.spec_from_file_location('day2_v2_provider_client_cpu',CLIENT)
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    baseline={'id':'synthetic_response','status':'completed','model':model,'previous_response_id':None,'tools':None,'tool_choice':'auto','tool_usage':{'image_gen':{'input_tokens':0,'output_tokens':0,'total_tokens':0},'web_search':{'num_requests':0}},'output':[{'type':'reasoning','summary':[]},{'type':'message','content':[{'type':'output_text','text':'{"ready":true}'}]}],'usage':{'input_tokens':11,'output_tokens':5,'total_tokens':16}}
    cases=[('auto_with_null_tools_no_activity_accept',None,None),('auto_with_empty_tools_no_activity_accept',lambda r:r.update(tools=[]),None),('nonzero_tool_usage_reject',lambda r:r['tool_usage']['web_search'].update(num_requests=1),'NONZERO_TOOL_USAGE'),('nested_nonzero_tool_tokens_reject',lambda r:r['tool_usage']['image_gen'].update(output_tokens=1),'NONZERO_TOOL_USAGE'),('nonempty_returned_tools_reject',lambda r:r.update(tools=[{'type':'web_search'}]),'RETURNED_TOOLS_NOT_EMPTY'),('tool_output_reject',lambda r:r['output'].append({'type':'function_call','name':'synthetic','arguments':'{}','call_id':'fake'}),'UNEXPECTED_TOOL_ACTIVITY'),('previous_chain_reject',lambda r:r.update(previous_response_id='synthetic_previous'),'CROSS_TASK_LINK')]
    results=[];mock_calls=0
    with tempfile.TemporaryDirectory(prefix='day2_v2_provider_cpu_') as tmp:
        for name,mutate,error in cases:
            response=copy.deepcopy(baseline)
            if mutate:mutate(response)
            directory=Path(tmp)/name
            requests=[]
            def fake_urlopen(request,timeout):
                payload=json.loads(request.data)
                assert request.full_url==endpoint
                assert payload['tools']==[] and payload['tool_choice']=='none'
                assert 'previous_response_id' not in payload
                assert payload['input']==[{'role':'user','content':'CPU SYNTHETIC CHECK ONLY'}]
                assert payload['store'] is False and timeout==300
                requests.append(payload)
                return SyntheticResponse(response)
            with patch.object(client,'config',return_value=(base,model,'FAKE_CPU_KEY_NEVER_READ_FROM_DISK')),patch.object(client.urllib.request,'urlopen',side_effect=fake_urlopen):
                value,meta=client.call('CPU SYNTHETIC CHECK ONLY',32,directory)
            assert len(requests)==1,'Unexpected retry'
            mock_calls+=len(requests)
            submitted=load(directory/'SUBMITTED.json')
            assert submitted['automatic_retry'] is False
            assert submitted['payload']['tool_choice']=='none' and submitted['payload']['tools']==[]
            assert load(directory/'RAW_RESPONSE.json')==response
            if error is None:
                assert value=={'ready':True} and meta['status']=='RETURNED'
                assert (directory/'COMPLETE.json').exists() and not (directory/'FAILED.json').exists()
                assert meta['input_tokens']==11 and meta['output_tokens']==5
                assert meta['returned_tool_choice']=='auto' and meta['tool_choice_echo_matches_request'] is False
                assert load(directory/'RAW_RESPONSE.json')['tool_choice']=='auto'
            else:
                assert value is None and meta['status']=='FAILED'
                assert error in meta['error'],(name,meta)
                assert (directory/'FAILED.json').exists() and not (directory/'COMPLETE.json').exists()
                assert meta['usage']==baseline['usage'],'Failed attempt usage must be retained'
            assert not any('FAKE_CPU_KEY' in p.read_text() for p in directory.iterdir() if p.is_file())
            results.append({'name':name,'status':'PASS','expected':'ACCEPT' if error is None else 'REJECT','mock_transport_calls':len(requests)})
    result={'status':'PASS','client_path':str(CLIENT),'client_sha256':sha(CLIENT),'check_count':len(results),'checks':results,'real_api_calls':0,'mock_transport_calls':mock_calls,'credential_reads':0,'pilot_scene_reads':0}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
