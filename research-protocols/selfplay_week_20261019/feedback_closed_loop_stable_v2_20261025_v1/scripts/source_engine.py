"""Scene-local audit/repair; two independent review streams and two repairs/root."""
import concurrent.futures,copy,json,pathlib,traceback,fcntl
from common import BASE,atomic,bind,load,now,protocol,status
from client import BASIC,STR,QUOTE,arr,obj,call
from ledger import build,ledger,related_parent,canonical
ISSUE=obj(dict(scene_ref=STR,fact_id=STR,reason=STR,smallest_correction=STR))
REVIEW=obj(dict(root_id=STR,scene_ref=STR,decision={'type':'string','enum':['PASS','REVISE','FAIL']},source_quotes=arr(QUOTE),issues=arr(ISSUE),warnings=arr(STR)))
TURN=obj(dict(turn_ref=STR,speaker=STR,text=STR,origin={'type':'string','enum':['codex_synthetic_extension']}))
SCENE=obj(dict(scene_ref=STR,route={'type':'string','enum':['A','B']},turns=dict(type='array',items=TURN,minItems=8,maxItems=8)))
MAP=obj(dict(source_id=STR,native_turn_id=STR,native_quote=STR,disposition={'type':'string','enum':['PRESERVED','OMITTED_NONMATERIAL']},rationale=STR,source_evidence=arr(QUOTE)))
REPAIR=obj(dict(root_id=STR,scene=SCENE,native_evidence=arr(MAP),canonical_facts=arr(obj(dict(turn_ref=STR,canonical_fact=STR))),changes=arr(STR)))
def packet(scene,candidate,inp,parent,native,led):
    index=inp['scene_sequence'].index(scene['scene_ref'])
    preceding=[dict(scene_ref=s['scene_ref'],facts=[dict(turn_ref=t['turn_ref'],speaker=t['speaker'],quote=t['text']) for t in s['turns']]) for s in candidate['scenes'][:index] if s['route']=='A']
    return dict(root_id=inp['root_id'],scene_ref=scene['scene_ref'],facts=[{k:canonical(t)[k] for k in ['turn_ref','speaker','canonical_fact','quote']} for t in scene['turns']],
                parent_evidence=related_parent(scene,parent),preceding_synthetic_scenes=preceding,
                reserved_native=[n for n in native if n['scene_ref']==scene['scene_ref']],
                native_mappings=[m for m in led['native_mappings'] if m['scene_ref']==scene['scene_ref']],
                constraints=inp['continuity_constraints'],warnings=candidate.get('warnings',[]),
                mechanical_errors=[e for e in led['mechanical_errors'] if e['scene_ref'] in [None,scene['scene_ref']]])
def review_errors(receipt,root,scene,available):
    errors=[];r=receipt.get('response')
    if not receipt['accepted']:return ['review_call_rejected:'+','.join(receipt['failure_reasons'])]
    if r['root_id']!=root or r['scene_ref']!=scene:errors.append('review_identity_mismatch')
    if r['decision']!='PASS':errors.append('semantic_'+r['decision'])
    if r['decision']=='PASS' and r['issues']:errors.append('pass_has_issues')
    if not r['source_quotes']:errors.append('missing_review_quotes')
    for q in r['source_quotes']:
        if q['turn_ref'] not in available or not q['quote'] or q['quote'] not in available[q['turn_ref']]:errors.append('review_quote_mismatch')
    for issue in r['issues']:
        if issue['scene_ref']!=scene:errors.append('issue_not_scene_local')
    return errors
def process(root):
    d=BASE/'sources'/root
    if (d/'OUTCOME.json').exists():return load(d/'OUTCOME.json')
    candidate,inp,parent,native,led,event=build(root)
    outcome=dict(root_id=root,status='FAIL',repairs_used=0,scene_outcomes=[],failure_reasons=[],initial_draft=led['initial_draft_binding'])
    if not event['accepted']:
        outcome['failure_reasons']=['initial_event_rejected']+event['failure_reasons'];atomic(d/'OUTCOME.json',outcome);return outcome
    for si in range(len(candidate['scenes'])):
        scene=candidate['scenes'][si];sr=scene['scene_ref'];attempt=0
        while True:
            p=packet(scene,candidate,inp,parent,native,led);atomic(d/'scene_audits'/sr.replace(':','_')/f'PACKET_{attempt}.json',p)
            available={f['turn_ref']:f['quote'] for f in p['facts']+p['parent_evidence']}
            for s in p['preceding_synthetic_scenes']:
                available.update({f['turn_ref']:f['quote'] for f in s['facts']})
            reviews=[];errs=[]
            for reviewer in [1,2]:
                role=('Check continuity, speaker/action bindings, plans versus completion, and natural sequential dialogue.' if reviewer==1 else 'Independently check semantic fidelity of every native utterance and mapping, including omissions, role/direction, booking state, clocks and ambiguity. Also check synthetic continuity.')
                prompt=BASIC+'You are independent source reviewer '+str(reviewer)+'. '+role+''' Audit only this scene against the supplied frozen fact ledger and evidence. Synthetic A planning is allowed; unrelated historical weekdays are not the same week. Native utterances are authoritative; annotations and frame values are not spoken facts. Exact native spelling/inconsistency/ambiguity must remain, not be resolved. Omitted native courtesy is acceptable only when genuinely nonmaterial. Old parent inconsistencies alone are warnings unless this scene amplifies/resolves them. Check material relations: quantities and coverage, distinct people/actions, plan/confirmation/completion, explicit motives, dates and revisions. Do not require every objective in this one scene, or impose question-specific output quotas. Mechanical errors are binding. PASS only if sound; otherwise name the exact failed fact/turn and a scene-local correction. Quote 1-3 exact substrings from this scene. Keep the response under 500 words. Do not rewrite source or invent a verdict for another reviewer.\nDATA\n'''+json.dumps(p,ensure_ascii=False,separators=(',',':'))
                r=call(root+'_'+sr.replace(':','_')+f'_a{attempt}_review{reviewer}',prompt,REVIEW)
                reviews.append(dict(receipt=str(pathlib.Path(r['directory'])/'RECEIPT.json'),response=r.get('response'),accepted=r['accepted']))
                errs+=review_errors(r,root,sr,available)
            me=[e for e in led['mechanical_errors'] if e['scene_ref'] in [None,sr]]
            errs+=['mechanical:'+e['code'] for e in me]
            audit=dict(scene_ref=sr,attempt=attempt,scene_sha256=bind(d/f'CANDIDATE_{outcome["repairs_used"]}.json')['sha256'],reviews=reviews,errors=errs,status='PASS' if not errs else 'FAIL')
            atomic(d/'scene_audits'/sr.replace(':','_')/f'AUDIT_{attempt}.json',audit)
            if not errs:
                outcome['scene_outcomes'].append(audit);break
            # Transport failure cannot justify editing any source fact.
            if outcome['repairs_used']>=2 or any(not r['accepted'] for r in reviews):
                outcome['scene_outcomes'].append(audit);break
            number=outcome['repairs_used']+1
            repair_packet=dict(scene=scene,source_evidence=p,identified_failures=[r['response'] for r in reviews],mechanical_errors=me,
                               native_evidence=[m for m in candidate.get('native_evidence',[]) if any(n['source_id']==m.get('source_id') and n['scene_ref']==sr for n in native)])
            prompt=BASIC+'''You are a targeted source repair author. Repair ONLY the named scene and its native fact mapping, addressing the exact failed facts. Return eight turns with the same IDs, route, two speakers and origin. No other scenes or parent turns may be changed. Preserve correct content. Native facts must match the reserved utterances without inference; synthetic A planning may be clarified naturally. Scene mean <=18 English words/turn. Return canonical_facts for each turn as whitespace-normalized exact utterance text, not inferred truths. Return complete native_evidence for this scene only (empty for A). No new people, retrospective completion, date conflation, or task/answer hints.\nDATA\n'''+json.dumps(repair_packet,ensure_ascii=False,separators=(',',':'))
            r=call(root+'_'+sr.replace(':','_')+f'_repair{number}',prompt,REPAIR)
            outcome['repairs_used']=number
            if not r['accepted']:
                audit['errors'].append('repair_call_rejected');outcome['scene_outcomes'].append(audit);break
            patch=r['response'];sc=patch['scene']
            facts={t['turn_ref']:' '.join(t['text'].split()) for t in sc['turns']}
            if patch['root_id']!=root or sc['scene_ref']!=sr or {f['turn_ref']:f['canonical_fact'] for f in patch['canonical_facts']}!=facts:
                audit['errors'].append('repair_scope_or_ledger_mismatch');outcome['scene_outcomes'].append(audit);break
            allowed={n['source_id'] for n in native if n['scene_ref']==sr}
            if any(m['source_id'] not in allowed for m in patch['native_evidence']):
                audit['errors'].append('repair_native_scope_violation');outcome['scene_outcomes'].append(audit);break
            candidate=copy.deepcopy(candidate);candidate['scenes'][si]=sc
            candidate['native_evidence']=[m for m in candidate.get('native_evidence',[]) if m['source_id'] not in allowed]+patch['native_evidence']
            atomic(d/f'CANDIDATE_{number}.json',candidate)
            led=ledger(candidate,inp,parent,native,led['initial_draft_binding']);atomic(d/f'LEDGER_{number}.json',led)
            atomic(d/f'REPAIR_{number}.json',dict(target_scene=sr,changed_only_scene=True,receipt=str(pathlib.Path(r['directory'])/'RECEIPT.json'),identified_failures=errs,changes=patch['changes']))
            scene=sc;attempt+=1
        atomic(d/'PROGRESS.json',outcome)
    errors=[dict(scene_ref=x['scene_ref'],errors=x['errors']) for x in outcome['scene_outcomes'] if x['status']!='PASS']
    if led['mechanical_errors']:errors.append(dict(mechanical_errors=led['mechanical_errors']))
    outcome['failure_reasons']=errors
    if not errors and len(outcome['scene_outcomes'])==6:
        full=copy.deepcopy(parent);full['turns']=parent['turns']+[t for s in candidate['scenes'] for t in s['turns']]
        full.update(root_id=root,source_status=inp['source_status'],warnings=candidate.get('warnings',[]),stable_v2_provenance=dict(initial=led['initial_draft_binding'],parent=inp['parent_source_binding'],repairs_used=outcome['repairs_used']))
        assert full['turns'][:len(parent['turns'])]==parent['turns']
        atomic(d/'FULL_SOURCE.json',full);atomic(d/'FACT_LEDGER.json',led)
        outcome.update(status='PASS',source=bind(d/'FULL_SOURCE.json'),ledger=bind(d/'FACT_LEDGER.json'),independent_review_streams=2)
    atomic(d/'OUTCOME.json',outcome);return outcome
def run(roots,label='pilot'):
    run_lock=(BASE/'.source_engine.lock').open('a')
    try:fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another source driver owns the workspace lock')
    outcomes=[];status('source_'+label,status='RUNNING')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(process,r):r for r in roots}
        for f in concurrent.futures.as_completed(futures):
            root=futures[f]
            try:r=f.result()
            except Exception as exc:
                r=dict(root_id=root,status='FAIL',failure_reasons=['engineering_exception:'+str(exc)],traceback=traceback.format_exc());atomic(BASE/'sources'/root/'OUTCOME.json',r)
            outcomes.append(r);atomic(BASE/f'{label.upper()}_SOURCE_PROGRESS.json',dict(outcomes=outcomes,at=now()))
            status(source_finished=len(outcomes),source_pass=sum(x['status']=='PASS' for x in outcomes))
    outcomes=sorted(outcomes,key=lambda x:roots.index(x['root_id']));result=dict(roots=roots,outcomes=outcomes,passed=sum(x['status']=='PASS' for x in outcomes),at=now())
    atomic(BASE/f'{label.upper()}_SOURCE_FREEZE.json',result)
    fcntl.flock(run_lock,fcntl.LOCK_UN);run_lock.close();return result
if __name__=='__main__':run(protocol()['targets']['new_train_roots'][:4])
