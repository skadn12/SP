"""Single owner, pilot gates, automatic full continuation only on complete PASS."""
from __future__ import annotations
import argparse, collections, fcntl, os, random, statistics, subprocess, sys, traceback
sys.dont_write_bytecode=True
from common import *
import source_engine,question_engine,solver_engine

def integrity(name):
    rows=[]
    for b in load(BASE/'INPUT_BINDINGS.json')['bindings']:
        actual=bind(b['path']);rows.append({'binding':b,'actual':actual,'verified':actual==b})
    atomic(BASE/name,rows)
    return all(x['verified'] for x in rows)

def counts(roots,entries,q,scores,raw,prep):
    sq={r:{a:[] for a in ('F','G')} for r in roots}
    for x in scores.get('records',[]):
        if x['root_id'] in sq:sq[x['root_id']][x['arm']].append(x)
    source_roots={e['root_id'] for e in entries if e['status']=='PASS'}
    valid=collections.Counter(x['root_id'] for x in q['slots'] if x['status']=='PASS')
    prepared=collections.Counter()
    p=BASE/'execution'/('PILOT' if roots==PILOT else 'FULL')/'PUBLIC_REQUESTS.json'
    if p.exists():prepared.update(x['root_id'] for x in load(p)['requests'])
    returned=collections.Counter(x['root_id'] for x in scores.get('records',[]))
    paired=[r for r,arms in sq.items() if all(len(arms[a])==16 and len({(x['slot_id'],x['variant_id']) for x in arms[a]})==16 for a in ('F','G'))]
    return {'planned_roots':len(roots),'source_pass_roots':len(source_roots),'question_valid_roots':sum(valid[r]==16 for r in roots),
      'solver_prepared_roots':sum(prepared[r]==32 for r in roots),'solver_returned_roots':sum(returned[r]==32 for r in roots),
      'paired_valid_roots':len(paired),'planned_slots':len(roots)*16,'valid_slots':q['valid_slots'],
      'missing_slots':len(roots)*16-q['valid_slots'],'planned_presentations':len(roots)*32,
      'prepared_presentations':prep.get('prepared_presentations',0),'returned_presentations':raw.get('returned_count',0),
      'contract_valid_presentations':scores.get('contract_valid',0),'solver_exact_presentations':scores.get('exact',0)}

def paired_analysis(records,roots):
    out=[]
    for root in roots:
        arms={a:[r for r in records if r['root_id']==root and r['arm']==a] for a in ('F','G')}
        if any(len(v)!=16 or len({(r['slot_id'],r['variant_id']) for r in v})!=16 for v in arms.values()):continue
        acc={a:statistics.mean(float(r['exact']) for r in v) for a,v in arms.items()}
        out.append({'root_id':root,'F_accuracy':acc['F'],'G_accuracy':acc['G'],'G_minus_F':acc['G']-acc['F']})
    diffs=[r['G_minus_F'] for r in out];ci=None;mean=None
    if diffs:
        mean=statistics.mean(diffs);rng=random.Random(20261025)
        samples=sorted(statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(10000));ci=[samples[250],samples[9749]]
    return {'eligible_roots':len(out),'root_results':out,'mean_G_minus_F':mean,'bootstrap_ci95':ci,
      'bootstrap_seed':20261025,'resamples':10000,'interpretation':'Paired descriptive difficulty only; no training, benefit or confirmed mechanism claim.'}

def sealed():
    # Validate the manifest binding and the exact score entry before any read.
    manifest_binding=next(b for b in load(BASE/'INPUT_BINDINGS.json')['bindings'] if b['path'].endswith('FINAL_MANIFEST.json'))
    verify(manifest_binding);p=pathlib.Path(manifest_binding['path']);manifest=load(p)
    row=next(r for r in manifest['files'] if r['path']=='execution/SCORES.json');sp=p.parent/row['path']
    b=bind(sp);assert b['sha256']==row['sha256'] and b['bytes']==row['bytes']
    results=[]
    for r in load(sp)['records']:
        s=r['strict'];root=s['root_id']
        if root in PROTOCOL['targets']['sealed_original_roots']:
            results.append({'root_id':root,'arm':r['arm'],'slot_id':r['slot_id'],'variant_id':s['variant_id'],'exact':s['exact'],'contract_valid':s['valid']})
    atomic(BASE/'SEALED_READONLY_RECEIPT.json',{'at':now(),'manifest':manifest_binding,'scores':b,'records_read':len(results),'reruns':0})
    return results

def report(label,roots,entries,q):
    d=BASE/'execution'/label
    prep=load(d/'PREPARATION.json') if (d/'PREPARATION.json').exists() else {}
    raw=load(d/'RAW_RESPONSE_FREEZE.json') if (d/'RAW_RESPONSE_FREEZE.json').exists() else {}
    scores=load(d/'SCORES.json') if (d/'SCORES.json').exists() else {}
    c=counts(roots,entries,q,scores,raw,prep)
    gates={'source':c['source_pass_roots']==len(roots),'questions':c['valid_slots']==len(roots)*16,
      'presentations':c['prepared_presentations']==c['returned_presentations']==len(roots)*32,
      'solver_output_contract':c['contract_valid_presentations']==len(roots)*32,
      'concurrency':not (BASE/'CONCURRENCY_INCIDENT.json').exists(),'immutable_inputs':integrity('INPUT_INTEGRITY_AFTER.json')}
    recs=[{'root_id':r['root_id'],'arm':r['arm'],'slot_id':r['slot_id'],'variant_id':r['variant_id'],
      'exact':r['score']['exact'],'contract_valid':r['contract_valid']} for r in scores.get('records',[])]
    old=sealed();analysis={'new_fixed_roots':paired_analysis(recs,roots),
      'original_four_readonly':paired_analysis(old,PROTOCOL['targets']['sealed_original_roots']),
      'combined_descriptive':paired_analysis(recs+old,roots+PROTOCOL['targets']['sealed_original_roots'])}
    receipts=[];sessions=[];failures=[]
    for p in (BASE/'cli_calls').glob('*/RECEIPT.json'):
        r=load(p);receipts.append(bind(p))
        for a in r.get('attempts',[]):
            be=a.get('backend',{});sessions.append({'model':be.get('actual_model'),'provider':be.get('actual_provider'),'thread_id':be.get('thread_id')})
        if not r['accepted']:failures.append({'receipt':bind(p),'failure_reasons':r['failure_reasons']})
    o={'schema_version':'stable_closed_loop_v2_report','at':now(),'status':'PASS' if all(gates.values()) else 'PARTIAL_STABILITY_BLOCKED',
      'phase':label.lower(),'fixed_roots':roots,'pilot_selection':'first four in PROTOCOL_LOCK target list, independent of past outcomes',
      'counts':c,'gates':gates,'source_outcomes':entries,
      'missing_slots':[{'root_id':x['root_id'],'arm':x['arm'],'slot_id':x['slot_id'],'failure_reasons':x['failure_reasons']} for x in q['slots'] if x['status']!='PASS'],
      'analysis':analysis,'semantic_call_receipts':receipts,'failed_calls':failures,'backend_sessions':sessions,
      'concurrency_incident':bind(BASE/'CONCURRENCY_INCIDENT.json') if (BASE/'CONCURRENCY_INCIDENT.json').exists() else None,
      'sealed_old_reruns':0,'training':False,'nonreturns_scored_as_incorrect':False,
      'solver_coverage_gate':'All planned presentations prepared and returned; all returned outputs must satisfy the frozen contract. Exact answer accuracy is descriptive and is not a stability gate.',
      'full_run_started':False,'backend_attestation_limit':'Local Codex session model/provider only; upstream routing not independently observed.'}
    atomic(BASE/f'{label}_REPORT.json',o);return o

def blocker(r):
    c=r['counts'];lines=['# Stability blocker','',f"Pilot status: {r['status']}. Full fixed 16-root production was not started.",'',
      f"Source: {c['source_pass_roots']}/4; slots: {c['valid_slots']}/64; prepared presentations: {c['prepared_presentations']}/128; returned: {c['returned_presentations']}/128.",'',
      'Failed gates: '+', '.join(k for k,v in r['gates'].items() if not v)+'.','',
      'The previous four roots remain sealed; no training or old-root solver rerun occurred. Missing outputs remain missing.','',
      'Concrete failures and smallest corrections:','']
    if r.get('concurrency_incident'):
        lines.append('- Two independently launched workers overlapped, reaching at least six semantic child calls. Their reciprocal interruption left required reviews/repairs incomplete. Preserve this pilot as invalid. The implemented cross-process owner lock and three file-lock slots prevent a compliant launcher from repeating the overlap; launch only `python -B pipeline.py` for a separately authorized clean run. No automatic rerun of failed semantic calls is permitted here.')
    for e in r['source_outcomes']:
        if e['status']=='PASS':continue
        lines.append('- '+e['root_id']+': '+json_compact(e['failure_reasons']))
        issues=e.get('failed_issues',[])
        for issue in issues:
            correction=issue.get('smallest_correction')
            if correction:lines.append('  '+str(issue.get('scene_ref'))+' / '+str(issue.get('fact_id'))+': '+correction)
    failed_questions=[x for x in r['missing_slots'] if x['failure_reasons']!=['source_not_pass']]
    if failed_questions:
        lines+=['','Slot interface corrections:','',
          '- The provider returned `invalid_request` for some initial schemas. Use the simplified supported schema types already implemented and check schema compatibility before a fresh experimental run; do not renew failed slot budgets.',
          '- Some semantic outputs echoed the input wrapper, omitted identity/verdict fields, changed a source binding, or used combined/nonexact quote coordinates. The author prompt now explicitly specifies flat root fields; planner identities use enum constraints, and all outputs still fail closed under program validation.',
          '- Independent Gold disagreement remains a semantic failure. Keep the failed slot after its one revision; do not coerce the assessor answer, silently relabel options, or retry the review.','']
    for x in failed_questions:
        concise=[str(e).split('\n',1)[0] for e in x['failure_reasons']]
        lines.append('- '+x['root_id']+'/'+x['arm']+'/'+x['slot_id']+': '+json_compact(concise))
    recovery=BASE/'execution/PILOT/INITIALIZATION_RECOVERY.json'
    if recovery.exists():
        lines+=['','Solver initialization correction: the absolute ZeroMQ IPC path exceeded 107 bytes. The implementation now sets `VLLM_RPC_BASE_PATH=.ipc` with cwd inside this run. The archived failure had zero submitted requests and zero outputs; recovery changed no experimental input or decode parameter.']
    scores_path=BASE/'execution/PILOT/SCORES.json'
    if scores_path.exists():
        invalid=[x for x in load(scores_path)['records'] if not x['contract_valid']]
        if invalid:
            lines+=['','Returned solver output failures:','']
            for x in invalid:lines.append('- '+x['request_id']+': '+', '.join(x['errors']))
            lines+=['','These ordering outputs omit required labels. They remain recorded failures with no semantic retries. A future protocol revision can encode the existing all-six ordering contract directly in the generation schema; this pilot retains the frozen generator and scorer.']
    lines+=['','Source failures block their 16 downstream slots and 32 presentations. Review PASS must come from two actual independent, accepted receipts for the same final source revision.','',
      'Engineering entry point: `pipeline.py`. Deterministic checks: `python -B -m unittest -v test_stability`. Full artifacts and exact gate counts: `PILOT_REPORT.json`.']
    write(BASE/'STABILITY_BLOCKER.md','\n'.join(lines)+'\n')

def run():
    owner=(BASE/'.pipeline_owner.lock').open('a')
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another_pipeline_owner_is_running')
    atomic(BASE/'OWNER.json',{'pid':os.getpid(),'at':now(),'entry_point':bind(__file__)})
    if (BASE/'PILOT_REPORT.json').exists():
        r=load(BASE/'PILOT_REPORT.json')
        if r['status']!='PASS':status('blocked',status='STABILITY_BLOCKED',pilot_report=bind(BASE/'PILOT_REPORT.json'));return r
    assert integrity('INPUT_INTEGRITY_BEFORE_PIPELINE.json')
    # Inventory all original drafts without changing any existing review-bound
    # candidate/ledger revision artifacts.
    ledger_rows=[]
    for root in ROOTS:
        from event_ledger import initial_ledger_with_byte_spans
        p=BASE/'sources'/root/'INITIAL_FACT_LEDGER.json'
        if not p.exists():atomic(p,initial_ledger_with_byte_spans(root))
        ledger_rows.append({'root_id':root,'ledger':bind(p)})
    atomic(BASE/'INITIAL_LEDGER_INVENTORY.json',{'at':now(),'roots':ledger_rows})
    cp=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_stability'],cwd=BASE,capture_output=True,text=True)
    atomic(BASE/'UNIT_TEST_RECEIPT.json',{'at':now(),'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
    if cp.returncode:raise RuntimeError('deterministic_tests_failed')
    p=BASE/'PILOT_SOURCE_FREEZE.json'
    entries=load(p)['roots'] if p.exists() else source_engine.run(PILOT)
    if not p.exists():atomic(p,{'at':now(),'roots':entries})
    q=question_engine.run(entries,PILOT,'PILOT');solver_engine.run('PILOT')
    r=report('PILOT',PILOT,entries,q)
    if r['status']!='PASS':
        blocker(r);status('blocked',status='STABILITY_BLOCKED',pilot_report=bind(BASE/'PILOT_REPORT.json'),counts=r['counts'],full_run_started=False);return r
    # Only the strict pilot PASS reaches this branch. Pilot roots, slots and raw
    # outputs are carried forward by identity, never regenerated for full run.
    status('full_source',status='RUNNING',full_run_started=True)
    full_entries=entries+source_engine.run(ROOTS[4:]);atomic(BASE/'FULL_SOURCE_FREEZE.json',{'roots':full_entries,'at':now()})
    fq=question_engine.run(full_entries,ROOTS,'FULL');solver_engine.run('FULL')
    fr=report('FULL',ROOTS,full_entries,fq);fr['full_run_started']=True;atomic(BASE/'FULL_REPORT.json',fr)
    status('complete' if fr['status']=='PASS' else 'partial',status='COMPLETE' if fr['status']=='PASS' else 'PARTIAL',counts=fr['counts'],final_report=bind(BASE/'FULL_REPORT.json'))
    return fr

if __name__=='__main__':
    try:run()
    except Exception:
        atomic(BASE/'PIPELINE_FAILURE.json',{'at':now(),'traceback':traceback.format_exc()});status('engineering_failure',status='FAILED');raise
