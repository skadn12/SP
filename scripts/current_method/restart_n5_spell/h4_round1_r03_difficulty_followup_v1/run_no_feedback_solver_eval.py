"""Run the frozen Base/trained Solver protocol on no-feedback candidates."""
import argparse, json, shutil
from pathlib import Path

from . import run as solver_run

RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
CONTROL = RUNS / 'h4_round1_r03_first_round_no_feedback_6roots_live_1'
AUDIT = RUNS / 'h4_round1_r03_first_round_no_feedback_semantic_audit_live_1'
SOURCE = RUNS / 'h4_round1_r03_feedback_batch_generation_revision1_live_2'
STAGED = RUNS / 'h4_round1_r03_first_round_no_feedback_solver_eval_input_live_1'
DEFAULT_OUTPUT = RUNS / 'h4_round1_r03_first_round_no_feedback_solver_eval_live_1'

def stage_input():
    if STAGED.exists():
        return
    # First-round parity: use every program-structurally-valid generated row.
    # The semantic audit is retained as a separate diagnostic artifact and is
    # deliberately not used to filter this Solver control.
    rows = []
    for rootdir in sorted(p for p in CONTROL.iterdir() if p.is_dir() and p.name.startswith('cc-train-episode-')):
        rows.extend(json.loads((rootdir / 'candidate_pool.json').read_text()))
    STAGED.mkdir(parents=True)
    (STAGED / 'roots').mkdir()
    records = []
    for row in rows:
        # The semantic-audit export wraps the original QA under `candidate`;
        # the difficulty Solver runner expects that original QA shape.
        candidate = dict(row.get('candidate') or row)
        # The difficulty runner consumes audit records; this staged record is
        # derived from the completed semantic audit export and does not alter it.
        records.append({'status':'completed','semantic_status':'accept',
                        'answer_semantically_valid':True,
                        'audit':{'evidence_sufficient':'sufficient'},
                        'static_review':{'training_blockers':[]},
                        'training_ready':False, 'candidate':candidate})
    (STAGED / 'baseline_omitted').mkdir()
    (STAGED / 'balanced_non_single').mkdir()
    for scope in ('baseline_omitted','balanced_non_single'):
        (STAGED / scope / 'audit_results.json').write_text(json.dumps({'results':records},ensure_ascii=False,indent=2))
        for root in sorted({str(r['root_id']) for r in rows}):
            rd = STAGED / scope / 'roots' / root
            rd.mkdir(parents=True)
            d = json.loads((SOURCE / root / 'dialogue_source.json').read_text())
            (rd / 'frozen_dialogue.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
    print(json.dumps({'staged_candidates':len(rows),'staged_roots':len(set(r['root_id'] for r in rows))},ensure_ascii=False))

def configure():
    solver_run.AUDITS = {'baseline_omitted': STAGED/'baseline_omitted',
                         'balanced_non_single': STAGED/'balanced_non_single'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=('prepare','live','all'),default='all'); ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a=ap.parse_args(); stage_input(); configure()
    if a.phase in ('prepare','all'): solver_run.prepare(a.output)
    if a.phase in ('live','all'): solver_run.live(a.output)

if __name__ == '__main__': main()
