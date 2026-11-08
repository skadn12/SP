"""Read-only public-record extraction aid. No verdicts or execution of logged code."""
from pathlib import Path
import json, hashlib, re
E=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/execution_20261106_v1')
MAIN=Path('/root/.codex/sessions/2026/11/05/rollout-2026-11-05T16-37-01-01a1aada-b034-7901-ad25-df290435bb6d.jsonl')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(p):
 out=[]
 for n,line in enumerate(Path(p).read_text().splitlines(),1):
  x=json.loads(line)
  out.append({'path':str(p),'line':n,'record_sha256':hashlib.sha256(line.encode()).hexdigest(),'timestamp_utc':x.get('timestamp'),'type':x.get('type'),'payload':x.get('payload',{})})
 return out
def ref(r): return {k:r[k] for k in ('path','line','record_sha256','timestamp_utc')}
def strings(x,depth=0):
 if depth>14:return
 if isinstance(x,str):
  try:d=json.loads(x)
  except (ValueError,TypeError):yield x;return
  if d!=x:yield from strings(d,depth+1)
 elif isinstance(x,list):
  for v in x:yield from strings(v,depth+1)
 elif isinstance(x,dict):
  if 'output' in x:yield from strings(x['output'],depth+1)
  elif 'text' in x:yield from strings(x['text'],depth+1)
  elif 'value' in x:yield from strings(x['value'],depth+1)
  elif 'content' in x:yield from strings(x['content'],depth+1)
  else:
   for v in x.values():
    if isinstance(v,(dict,list)):yield from strings(v,depth+1)
def objects(s):
 dec=json.JSONDecoder();i=0
 while i<len(s):
  positions=[v for v in (s.find('{',i),s.find('[',i)) if v>=0]
  if not positions:break
  i=min(positions)
  try:x,n=dec.raw_decode(s[i:])
  except ValueError:i+=1;continue
  yield x;i+=n
def walk(x):
 if isinstance(x,dict):
  yield x
  for v in x.values():yield from walk(v)
 elif isinstance(x,list):
  for v in x:yield from walk(v)
def toolrows(rs):
 out=[]
 for r in rs:
  p=r['payload'];t=p.get('type')
  if r['type']!='response_item':continue
  if t in ('custom_tool_call','function_call'):
   a=p.get('input',p.get('arguments',''))
   out.append({**ref(r),'kind':'call','name':p.get('name'),'input':a,'call_id':p.get('call_id')})
  elif t in ('custom_tool_call_output','function_call_output'):
   ss=list(strings(p.get('output')))
   out.append({**ref(r),'kind':'output','texts':ss,'call_id':p.get('call_id'),'truncated':any('tokens truncated' in s or 'Warning: truncated output' in s for s in ss)})
 return out
def sessions():
 out={}
 for p in sorted(Path('/root/.codex/sessions/2026/11/06').glob('rollout-*.jsonl')):
  line=p.open().readline();m=json.loads(line).get('payload',{});a=m.get('agent_path','')
  if re.fullmatch(r'/root/(author|selector)_\d+',a):out[a]={'path':str(p),'meta':m,'records':records(p)}
 return out
def chunks(rows,source,cutoff=10**20):
 expected={t['turn_ref']:{k:t[k] for k in ('turn_ref','speaker','text','origin')} for t in source['turns']}
 result=[]
 for r in rows:
  if r['kind']!='output' or r['line']>=cutoff:continue
  for s in r['texts']:
   os=[o for x in objects(s) for o in walk(x)]
   matches=[];bad=[]
   for o in os:
    if all(k in o for k in ('turn_ref','speaker','text','origin')) and o['turn_ref'] in expected:
     if all(o[k]==v for k,v in expected[o['turn_ref']].items()):matches.append(o['turn_ref'])
     else:bad.append(o['turn_ref'])
   if matches or bad:
    unique=list(dict.fromkeys(matches))
    result.append({**{k:r[k] for k in ('path','line','record_sha256','timestamp_utc')},'turn_refs':unique,'turn_count':len(unique),'text_chars':len(s),'truncated':r['truncated'],'valid_compact':len(unique)<=30 and len(s)<=12000 and not r['truncated'],'mismatched_turns':bad})
 return result
def firstwrite(rows,name):
 hits=[r for r in rows if r['kind']=='call' and name in str(r['input']) and re.search(r"write|open\(|O_EXCL",str(r['input']))]
 return hits[0] if hits else None
def allobjects(rows,cutoff=10**20):
 return [(r,s,o) for r in rows if r['kind']=='output' and r['line']<cutoff for s in r['texts'] for x in objects(s) for o in walk(x)]
