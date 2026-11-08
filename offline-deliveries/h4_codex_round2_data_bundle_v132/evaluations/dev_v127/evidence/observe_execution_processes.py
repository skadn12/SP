import datetime,json,os,time
from pathlib import Path
scope=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_dev_eval_v127')
out=Path('/tmp/v127_execution_process_observations.jsonl')
seen=set()
if out.exists():
 for line in out.read_text().splitlines():
  r=json.loads(line);seen.add((r.get('active_side'),r['pid'],r['proc_starttime_ticks']))
for _ in range(480):
 try:
  s=json.loads((scope/'RUN_STATUS.json').read_text())
  if s.get('status') in ['EVALUATION_REPORTED_AWAITING_INDEPENDENT_AUDIT','STOPPED_WITH_ERROR']:
   print(json.dumps({'watcher_status':'STOPPED_AFTER_CONTROLLER','controller_status':s['status'],'observations':len(seen)}),flush=True);break
  pid=s.get('pid');proc=Path('/proc')/str(pid)
  if s.get('active_side') in ['baseline','new'] and s.get('status')=='EVALUATION_RUNNING' and proc.exists():
   start=(proc/'stat').read_text().split()[21];key=(s['active_side'],pid,start)
   if key not in seen:
    cmd=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    r={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'active_side':s['active_side'],'status':s['status'],'pid':pid,'cmdline':cmd,'exe':os.readlink(proc/'exe'),'cwd':os.readlink(proc/'cwd'),'proc_starttime_ticks':start}
    with out.open('a') as f:f.write(json.dumps(r)+'\n')
    seen.add(key);print(json.dumps({'observed_side':s['active_side'],'pid':pid,'cmdline':cmd}),flush=True)
 except (FileNotFoundError,ProcessLookupError,json.JSONDecodeError):pass
 time.sleep(15)
