"""Offline reports for the frozen source diagnostic; no model calls."""
import collections
import json

SAMPLERS = ('SCORE', 'RANDOM_FROM_REMAINDER')

def source_decisions(mapping, primary, independent, adjudication):
    p={r['sample_id']:r for r in primary}; i={r['sample_id']:r for r in independent}; a={r['sample_id']:r for r in adjudication}
    out={}
    for m in mapping:
        sid=m['sample_id']; x=p.get(sid); y=i.get(sid)
        if x and y and x['status']==y['status']:
            final=x; status=x['status']
        elif x and y and sid in a:
            final=a[sid]; status=final['status']
        else:
            final={}; status='UNKNOWN'
        out[sid]={'status':status,'double_reviewed':bool(x and y),'review':final}
    return out

def history_order(mapping, decisions, seed):
    import hashlib
    by={s:sorted([m['sample_id'] for m in mapping if m['sampler']==s and decisions[m['sample_id']]['status']=='PASS'],
                key=lambda sid:hashlib.sha256((seed+':history:'+sid).encode()).hexdigest()) for s in SAMPLERS}
    first={s:by[s][:6] for s in SAMPLERS}
    left=12-sum(map(len,first.values()))
    # Transfer only capacity left by a group with fewer than six source passes.
    for s in SAMPLERS:
        extra=by[s][6:6+left]; first[s]+=extra; left-=len(extra)
    return [sid for s in SAMPLERS for sid in first[s]]

def signatures(review):
    certs=review.get('certificates', [])
    values=[c.get('operation_signature') for c in certs if c.get('operation_signature')]
    if not values:
        sig=review.get('operation_signature')
        values=[sig] if sig else []
    return sorted(set(map(str,values)))

def build(mapping, ledgers):
    decisions=source_decisions(mapping,ledgers['primary'],ledgers['independent'],ledgers['adjudication'])
    hist={r['sample_id']:r for r in ledgers['history']}; groups={}; readiness=[]; comparators=[]
    for sampler in SAMPLERS:
        ms=[m for m in mapping if m['sampler']==sampler]; counts=collections.Counter(); sigs=collections.Counter(); families=collections.Counter(); failures=collections.Counter()
        for m in ms:
            sid=m['sample_id']; d=decisions[sid]; r=d['review']; h=hist.get(sid)
            counts[d['status']]+=1; counts['double_reviewed']+=d['double_reviewed']
            if h: counts['history_reviewed']+=1; counts['history_'+h['status']]+=1
            if d['status']=='PASS':
                for sig in signatures(r): sigs[sig]+=1
                families[str(r.get('semantic_family','UNSPECIFIED'))]+=1
            if d['status']!='PASS': failures[str(r.get('failure_reason','UNKNOWN'))]+=1
            # A source pass is distinct from an explicit qualified history pass.
            qualified=bool(d['status']=='PASS' and h and h['status']=='PASS'
                           and h.get('qualification_status')=='ELIGIBLE_FOR_DRAFT'
                           and h.get('candidate_history_boundary')
                           and h.get('source_binding_verified') is True
                           and h.get('identity_checks_pass') is True)
            counts['eligible_for_draft']+=qualified
            gaps=[] if qualified else (h.get('gaps',[]) if h else ['history deep review not completed'])
            if d['status']!='PASS': gaps=['source status '+d['status']]+gaps
            readiness.append({'sample_id':sid,'root_id':m['root_id'],'sampler':sampler,'source_status':d['status'],
                              'double_reviewed':d['double_reviewed'],'history_status':h['status'] if h else 'NOT_REVIEWED',
                              'eligible_for_draft':qualified,'gaps':gaps,'operation_signatures':signatures(r),
                              'candidate_history_boundary':h.get('candidate_history_boundary') if h else None})
            if h:
                comparators.append({'sample_id':sid,'root_id':m['root_id'],'comparators':h.get('comparators',[]),
                                    'same_root_event':h.get('same_root_event','UNKNOWN'),
                                    'same_task_near_duplicate':h.get('same_task_near_duplicate','UNKNOWN'),
                                    'broad_semantic_family':h.get('broad_semantic_family','UNKNOWN'),
                                    'operation_signature':h.get('operation_comparison','UNKNOWN'),
                                    'boundary':h.get('candidate_history_boundary'),'gaps':h.get('gaps',[])})
        n=len(ms)
        times=[float(r['elapsed_seconds']) for k in ledgers for r in ledgers[k] if r['sample_id'] in {m['sample_id'] for m in ms} and isinstance(r.get('elapsed_seconds'),(int,float))]
        groups[sampler]={'target':20,'allocated':n,'allocation_shortfall':20-n,**dict(counts),
                         'source_pass_rate':counts['PASS']/n if n else None,'operation_signature_root_counts':dict(sigs),
                         'semantic_family_root_counts':dict(families),'max_family_share_of_passes':max(families.values(),default=0)/counts['PASS'] if counts['PASS'] else None,
                         'failure_reasons':dict(failures),'review_elapsed_seconds_recorded':sum(times),'records_with_cost':len(times)}
    a,b=(groups[s] for s in SAMPLERS)
    enabled=all(g['allocated']==20 and g.get('double_reviewed',0)>=18 for g in (a,b)) and abs(a.get('UNKNOWN',0)-b.get('UNKNOWN',0))<=2
    signal={'enabled':enabled,'decision':'DESCRIPTIVE_ONLY'}
    if enabled:
        pa,pb=a.get('PASS',0),b.get('PASS',0); lead=SAMPLERS[0] if pa>pb else SAMPLERS[1]; diff=abs(pa-pb)
        lead_roots=[r for r in readiness if r['sampler']==lead and r['source_status']=='PASS']
        patterns=set(s for r in lead_roots for s in r['operation_signatures'])
        if pa<=1 and pb<=1: signal['decision']='STOP_ADDITION_LOW_YIELD'
        elif diff>=3 and len(lead_roots)>=3 and len(patterns)>=2:
            signal.update(decision='EXPANDED_CONFIRMATION_CANDIDATE',leading_sampler=lead,pass_count_difference=diff,verified_patterns=sorted(patterns))
        else: signal['decision']='NO_STRATEGY_CHANGE_SUPPORTED'
    eligible=[r for r in readiness if r['eligible_for_draft']]
    return {'status':'FROZEN','groups':groups,'denominator':'actual frozen allocations; missing/conflicting reviews UNKNOWN',
            'count_based_signal':signal,'eligible_for_draft_roots':len(eligible),'new_root_shortfall_to_8':max(0,8-len(eligible)),
            'formal_admission':'NOT_AUTOMATIC; source and historical qualification reported per root',
            'solver_requests':0,'training_runs':0,'readiness':readiness,'historical_comparators':comparators,
            'limits':['Source signatures do not prove the structure of any future question.',
                      'Unresolved historical records remain UNKNOWN; scoped boundaries do not establish global novelty.',
                      'Count signals are engineering diagnostics, not causal or significance claims.']}

def write_reports(root, clock, results):
    def js(name,value): (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    js('RESULTS_BY_SAMPLER.json',results)
    js('ROOT_READINESS.json',results['readiness'])
    (root/'HISTORICAL_COMPARATORS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results['historical_comparators']),encoding='utf-8')
    lines=['# Root readiness','', '| Sample | Root | Source | History | Eligible for draft | Gaps |','|---|---|---|---|---|---|']
    for r in results['readiness']:
        lines.append(f"| {r['sample_id']} | {r['root_id']} | {r['source_status']} | {r['history_status']} | {r['eligible_for_draft']} | {'; '.join(map(str,r['gaps'])).replace('|','/')} |")
    (root/'ROOT_READINESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    gaps=['# Eligibility gaps','']+[f"- {r['sample_id']} / {r['root_id']}: {'; '.join(map(str,r['gaps']))}" for r in results['readiness'] if not r['eligible_for_draft']]
    (root/'ELIGIBILITY_GAPS.md').write_text('\n'.join(gaps)+'\n',encoding='utf-8')
    draft=['# Next author batch draft','',f"Qualified for draft: {results['eligible_for_draft_roots']}; shortfall to eight: {results['new_root_shortfall_to_8']}.",
           'A future 8-root × 4-arm contract requires resolved eligibility and protocol conflicts. This diagnostic generates no question, Solver call or training item.',
           'OLD_F/OLD_G/NEW_F/NEW_G retain equal opportunity budgets; F uses fixed FC09 and G has no QA feedback. Certificates and discovery groups remain hidden from authors.']
    (root/'NEXT_AUTHOR_BATCH_DRAFT.md').write_text('\n\n'.join(draft)+'\n',encoding='utf-8')
    report=['# Morning report','',f"Clock: {clock['started_utc']} → {clock['closed_utc']}; fixed deadline: {clock['deadline_utc']}.",
            '```json',json.dumps({k:v for k,v in results.items() if k not in ('readiness','historical_comparators')},ensure_ascii=False,indent=2),'```',
            'Per-root evidence and eligibility: ROOT_READINESS.md; comparator boundaries: HISTORICAL_COMPARATORS.jsonl; unresolved items: ELIGIBILITY_GAPS.md.']
    (root/'MORNING_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
