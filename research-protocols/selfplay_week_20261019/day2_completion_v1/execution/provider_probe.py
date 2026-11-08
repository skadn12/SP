from pathlib import Path
import json,re,urllib.request,urllib.error,time,datetime,hashlib
HERE=Path(__file__).resolve().parent
c=Path('/root/.codex/config.toml').read_text();base=re.search(r'^base_url\s*=\s*"([^"]+)"',c,re.M).group(1).rstrip('/')
model=re.search(r'^model\s*=\s*"([^"]+)"',c,re.M).group(1)
key=json.loads(Path('/root/.codex/auth.json').read_text())['OPENAI_API_KEY']
payload={'model':model,'input':[{'role':'user','content':'Connectivity check only. Reply exactly {"ready":true}.'}],'max_output_tokens':128,'reasoning':{'effort':'low'},'store':False}
p=HERE/'PROVIDER_COMPATIBILITY.json';assert not p.exists()
start=time.time(); result={'purpose':'ONE_COMPATIBILITY_CALL_NOT_PILOT_CONTENT','endpoint':base+'/responses','requested':payload,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retry':False}
try:
 req=urllib.request.Request(base+'/responses',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key,'User-Agent':'gxy-day2-eval/1.0'})
 with urllib.request.urlopen(req,timeout=180) as r: data=json.load(r);result.update({'http_status':r.status,'response':data})
except urllib.error.HTTPError as e:result.update({'http_status':e.code,'error_body':e.read().decode(errors='replace').replace(key,'[REDACTED]')})
except Exception as e:result.update({'error_type':type(e).__name__,'error':str(e).replace(key,'[REDACTED]')})
result['wall_seconds']=time.time()-start
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('requested','response','error_body')},ensure_ascii=False))
if 'response' in result:print(json.dumps({'response_model':data.get('model'),'status':data.get('status'),'usage':data.get('usage'),'output':data.get('output')},ensure_ascii=False))
