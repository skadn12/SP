"""Freeze the concrete repository closure and verify it before future sends."""
from __future__ import annotations
from contextlib import contextmanager
import json,sys
from pathlib import Path
from ..core import atomic_json,sha256_file
from ..h4_r02_basic_miner_v3_off_paired_v1.dependencies import (
 record,static_closure,loaded_repository_files,environment_identity,forbid_network,
 repository_execution_guard,check_loaded_coverage,
)
from .constants import CODE,METHOD,VERSION,SCHEDULE,CONFIG

def collect_coverage(run,dry):
 observed=loaded_repository_files()|{Path(p) for p in dry['repository_modules']}
 traced,edges,dynamic=static_closure(set(CODE.glob('*.py'))|observed)
 required={METHOD/p for p in ('backends.py','simple_append_v1/live_runtime.py','live_backends.py','ledger.py','core.py')}
 checks=dict(dry_run_pass=dry['status']=='PASS',observed_covered=observed<=traced,required_in_closure=required<=traced,required_observed=required<=observed,bytecode_disabled=sys.dont_write_bytecode)
 result=dict(version=VERSION,status='PASS' if all(checks.values()) else 'FAIL',checks=checks,repository_files=[record(p) for p in sorted(traced)],observed_repository_modules=[record(p) for p in sorted(observed)],static_import_edges=edges,nonliteral_dynamic_import_sites=dynamic,environment=environment_identity(),dry_run=dry,method='AST recursive local import closure plus network-blocked execution of experiment scheduler, shared Writer and original HTTP transport wrapper. Includes loaded and error-path repository modules; future execution profile guard rejects unfrozen repository files.')
 atomic_json(Path(run)/'audit/dependency_coverage_report.json',result)
 return result

def verify_frozen(run,config_path=None):
 from .runtime import GuardrailStop
 run=Path(run);p=run/'audit/frozen_manifest.json'
 if not p.is_file():raise GuardrailStop('PREPARATION_NOT_FROZEN')
 f=json.loads(p.read_text())
 if f.get('version')!=VERSION or f.get('schedule')!=SCHEDULE:raise GuardrailStop('FROZEN_SCOPE_DRIFT')
 if Path(config_path or CONFIG).resolve()!=Path(f['config']['path']).resolve():raise GuardrailStop('CONFIG_PATH_DRIFT')
 for row in f['code']+f['inputs']+f['sources']+[f['config'],f['preflight']]:
  p=Path(row['path'])
  if not p.is_file() or p.stat().st_size!=row['size'] or sha256_file(p)!=row['sha256']:raise GuardrailStop('FROZEN_IDENTITY_DRIFT:'+str(p))
 current={str(p.resolve()) for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 expected={r['path'] for r in f['code'] if Path(r['path']).is_relative_to(CODE)}
 if current!=expected:raise GuardrailStop('NEW_CODE_INVENTORY_DRIFT')
 for folder in ('inputs','policy_snapshot','audit/service_evidence'):
  directory=run/folder
  current={str(p.resolve()) for p in directory.rglob('*') if p.is_file()}
  expected={r['path'] for r in f['inputs'] if Path(r['path']).is_relative_to(directory)}
  if current!=expected:raise GuardrailStop('FROZEN_INPUT_INVENTORY_DRIFT:'+folder)
 check_loaded_coverage(x['path'] for x in f['code'])
 if environment_identity()!=f['environment']:raise GuardrailStop('ENVIRONMENT_DRIFT')
 if json.loads((run/'audit/preflight_report.json').read_text()).get('status')!='PASS':raise GuardrailStop('PREFLIGHT_NOT_PASS')
 return f
