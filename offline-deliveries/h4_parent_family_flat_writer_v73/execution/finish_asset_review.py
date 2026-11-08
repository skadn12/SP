"""Crosscheck a saved source-first round judgment; no model calls."""
from pathlib import Path
import argparse
import sys
R=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(R))
import asset_generation as assets
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',choices=('initial','supplement'),required=True);p.add_argument('--slot',required=True)
    args=p.parse_args();r=assets.finish_review(R/'assets'/args.slot,args.round)
    print(args.slot,args.round,r['usable_parents'],r['usable_contradicts'])
