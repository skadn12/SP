"""Record only PYTHONPATH during live base; never claim unseen past environment."""
from pathlib import Path
import json,os,datetime
HOME=Path(__file__).resolve().parents[1]
observed=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().split(b'\0')
  cwd=(p/'cwd').resolve()
  if cwd!=HOME or (b'live-base' not in cmd and not(any(b'run_solver.py' in c for c in cmd) and b'live' in cmd)):continue
  entries=(p/'environ').read_bytes().split(b'\0')
  values=[x.split(b'=',1)[1].decode() for x in entries if x.startswith(b'PYTHONPATH=')]
  observed.append({'pid':int(p.name),'command':[c.decode() for c in cmd if c],
                   'pythonpath_present':bool(values),'pythonpath_value':values[0] if values else None})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(observed)>=2,'live base parent and solver must be observed, never fabricated'
record={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'shell_pythonpath_present':'PYTHONPATH' in os.environ,'shell_pythonpath_value':os.environ.get('PYTHONPATH'),
 'live_processes':observed,'limitation':'Observed during live base; not a pre-start environment capture of the earlier completed trained process. No other environment values retained.'}
with (HOME/'evidence/RUNTIME_ENVIRONMENT_OBSERVATION.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print({'processes_observed':len(observed),'PYTHONPATH_present':any(p['pythonpath_present'] for p in observed)})
