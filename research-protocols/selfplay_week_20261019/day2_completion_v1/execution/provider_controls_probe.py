from pathlib import Path
import json,re,urllib.request,urllib.error,time,datetime
HERE=Path(__file__).resolve().parent
c=Path('/root/.codex/config.toml').read_text();base=re.search(r'^base_url\s*=\s*"([^"]+)"',c,re.M).group(1).rstrip('/');model=re.search(r'^model\s*=\s*"([^"]+)"',c,re.M).group(1)
key=json.loads(Path('/root/.codex/auth.json').read_text())['OPENAI_API_KEY']
payload={'model':model,'input':[{'role':'user','content':'Protocol compatibility probe only. Do not use any tools or files. Reply exactly {"ready":true}. Do not add anything.'}],'max_output_tokens':64,'reasoning':{'effort':'low'},'store':False,'tools':[],'tool_choice':'none','temperature':0.8,'top_p':0.9,'truncation':'disabled'}
p=HERE/'PROVIDER_CONTROLS_PROBE.json';assert not p.exists()
start=time.time();result={'purpose':'ONE_CONTROLS_CHECK_NOT_PILOT_CONTENT','endpoint':base+'/responses','requested':payload,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retry':False}
try:
 req=urllib.request.Request(base+'/responses',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'gxy-day2-eval/1.0'})
 with urllib.request.urlopen(req,timeout=180) as r: data=json.load(r); result.update({'http_status':r.status,'response':data})
except urllib.error.HTTPError as e:result.update({'http_status':e.code,'error_body':e.read().decode(errors='replace').replace(key,'[REDACTED]')})
except Exception as e:result.update({'error_type':type(e).__name__,'error':str(e).replace(key,'[REDACTED]')})
result['wall_seconds']=time.time()-start;p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if 'response' in result:print(json.dumps({k:data.get(k) for k in ['model','status','max_output_tokens','tools','tool_choice','temperature','top_p','previous_response_id','metadata','usage','output']},ensure_ascii=False))
else:print(json.dumps({k:v for k,v in result.items() if k!='requested'},ensure_ascii=False))
