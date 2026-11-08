"""One independent, tool-free Responses request; no retry or previous history."""
from pathlib import Path
import json,re,urllib.request,urllib.error,time,datetime,hashlib

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def config():
    c=Path('/root/.codex/config.toml').read_text()
    base=re.search(r'^base_url\s*=\s*"([^"]+)"',c,re.M).group(1).rstrip('/')
    model=re.search(r'^model\s*=\s*"([^"]+)"',c,re.M).group(1)
    return base,model,json.loads(Path('/root/.codex/auth.json').read_text())['OPENAI_API_KEY']
def strict_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise ValueError('DUPLICATE_JSON_KEY')
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE')))
def tool_activity(value):
    if isinstance(value,dict): return any(tool_activity(v) for v in value.values())
    if isinstance(value,list): return any(tool_activity(v) for v in value)
    return isinstance(value,(int,float)) and not isinstance(value,bool) and value!=0
def call(text,max_output_tokens,directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    assert not (d/'SUBMITTED.json').exists(), 'NO_RETRY'
    base,model,key=config()
    freeze=json.loads((Path(__file__).resolve().parent.parent/'EXECUTION_AMENDMENT.json').read_text())['backend']
    assert base+'/responses'==freeze['endpoint'] and model==freeze['model_requested'],'BACKEND_CONFIG_DRIFT'
    payload={'model':model,'input':[{'role':'user','content':text}],
        'max_output_tokens':max_output_tokens,'reasoning':{'effort':'low'},
        'store':False,'tools':[],'tool_choice':'none','truncation':'disabled'}
    started=now();start=time.perf_counter()
    save(d/'SUBMITTED.json',{'started_at_utc':started,'endpoint':base+'/responses',
        'payload':payload,'automatic_retry':False,'credential_source':'existing Codex auth; secret never archived'})
    response=None
    try:
        req=urllib.request.Request(base+'/responses',data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'gxy-day2-eval/1.0'})
        with urllib.request.urlopen(req,timeout=300) as r:
            raw=r.read(); http=r.status
        (d/'RAW_RESPONSE.json').write_bytes(raw)
        response=strict_json(raw)
        ended=now();wall=time.perf_counter()-start
        assert response.get('status')=='completed', 'BACKEND_NOT_COMPLETE'
        assert response.get('model')==model,'RETURNED_MODEL_DRIFT'
        assert response.get('previous_response_id') is None, 'CROSS_TASK_LINK'
        assert not response.get('tools'),'RETURNED_TOOLS_NOT_EMPTY'
        out=response.get('output',[])
        assert all(x.get('type') in ('message','reasoning') for x in out),'UNEXPECTED_TOOL_ACTIVITY'
        assert not tool_activity(response.get('tool_usage',{})),'NONZERO_TOOL_USAGE'
        chunks=[c['text'] for x in out if x.get('type')=='message' for c in x.get('content',[]) if c.get('type')=='output_text']
        assert len(chunks)==1,'OUTPUT_COUNT'
        output=chunks[0];(d/'RAW_OUTPUT.txt').write_text(output)
        value=strict_json(output)
        usage=response.get('usage',{})
        result={'status':'RETURNED','started_at_utc':started,'finished_at_utc':ended,'wall_seconds':wall,
            'http_status':http,'requested_model':model,'returned_model':response.get('model'),
            'response_id':response.get('id'),'input_tokens':usage.get('input_tokens'),
            'output_tokens':usage.get('output_tokens'),'usage':usage,
            'returned_max_output_tokens':response.get('max_output_tokens'),
            'temperature':response.get('temperature'),'top_p':response.get('top_p'),
            'seed':None,'output_ceiling_enforced':False,'tools_activity_observed':False,
            'returned_tool_choice':response.get('tool_choice'),
            'tool_choice_echo_matches_request':response.get('tool_choice')=='none',
            'tool_choice_echo_limit':'auto/null alone is metadata, not evidence of tool activity; output types and actual tool usage checked separately',
            'scope':'provider-reported usage includes opaque gateway instructions; not local prompt estimate'}
        save(d/'COMPLETE.json',result)
        return value,result
    except Exception as error:
        failure={'status':'FAILED','started_at_utc':started,'finished_at_utc':now(),
            'wall_seconds':time.perf_counter()-start,'error_type':type(error).__name__,
            'error':str(error).replace(key,'[REDACTED]'),'automatic_retry':False}
        if response is not None:
            failure['usage']=response.get('usage')
            failure['returned_model']=response.get('model')
        if isinstance(error,urllib.error.HTTPError):
            failure['http_status']=error.code
            failure['error_body']=error.read().decode(errors='replace').replace(key,'[REDACTED]')[:4000]
        save(d/'FAILED.json',failure)
        return None,failure
