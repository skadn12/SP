"""Explicit prepare/live/report CLI. Default is offline preparation only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from ..core import atomic_text,atomic_json
from .constants import DEFAULT_RUN,CONFIG

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('mode',nargs='?',choices=('prepare','live','report'),default='prepare')
 parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
 parser.add_argument('--config',type=Path,default=CONFIG)
 parser.add_argument('--execute-after-authorization',action='store_true',help='Only use after a subsequent explicit live execution instruction; no authorization file is created.')
 args=parser.parse_args(argv)
 if args.run_dir.resolve()!=DEFAULT_RUN.resolve() or args.config.resolve()!=CONFIG.resolve():parser.error('The frozen experiment identity and configuration paths are fixed.')
 if args.mode=='live':
  if not args.execute_after_authorization:parser.error('live requires a subsequent explicit instruction and --execute-after-authorization; this preparation task is not authorization.')
  from .runtime import execute_live
  result=execute_live(args.run_dir,args.config)
 elif args.mode=='prepare':
  if args.execute_after_authorization:parser.error('prepare never accepts or records live authorization')
  from .prepare import prepare
  already_frozen=(args.run_dir/'audit/frozen_manifest.json').is_file()
  result=prepare(args.run_dir)
  if result['status']=='PASS' and already_frozen:
   result={'status':'PASS','mode':'FROZEN_IDENTITY_VERIFIED','preflight_report':str(args.run_dir/'audit/preflight_report.json'),'artifacts_rewritten':False,'external_calls':0}
  elif result['status']=='PASS':
   from .delivery import write_delivery
   write_delivery(args.run_dir)
  else:atomic_text(args.run_dir/'RUN_STATUS.md','PREPARATION_STATUS = BLOCKED\nACTUAL_EXTERNAL_CALLS = 0\nLIVE_STATUS = NOT_RUN\nSee audit/preflight_report.json.\n')
 else:
  from .reporting import report
  result=report(args.run_dir)
 print(json.dumps(result,ensure_ascii=False,indent=2))
 return 0 if result.get('status') not in ('FAIL','BLOCKED') else 1

if __name__=='__main__':raise SystemExit(main())
