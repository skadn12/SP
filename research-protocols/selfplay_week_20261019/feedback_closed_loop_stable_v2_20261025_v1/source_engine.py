"""Independent compact audits and at most two single-scene repairs per root."""
from __future__ import annotations
import concurrent.futures, copy, traceback
from common import *
from event_ledger import inputs,build,compact,recover
from codex_client import call,object_schema,STR,STRS,QUOTES

ISSUE=object_schema({'scene_ref':STR,'fact_id':STR,'code':STR,'reason':STR,'smallest_correction':STR})
REVIEW=object_schema({'root_id':STR,'decision':{'type':'string','enum':['PASS','REVISE','FAIL']},
    'checked_scenes':STRS,'source_quotes':QUOTES,
    'issues':{'type':'array','items':ISSUE},'warnings':STRS,
    'material_evidence':{'type':'array','items':object_schema({'capability':STR,'turn_refs':STRS,'rationale':STR})}})
TURN=object_schema({'turn_ref':STR,'speaker':STR,'text':STR,'origin':{'type':'string','enum':['codex_synthetic_extension']}})
MAPPING=object_schema({'source_id':STR,'native_turn_id':STR,'native_quote':STR,
    'disposition':{'type':'string','enum':['PRESERVED','OMITTED_NONMATERIAL']},'rationale':STR,
    'source_evidence':{'type':'array','items':object_schema({'turn_ref':STR,'quote':STR})}})
REPAIR=object_schema({'root_id':STR,'scene':object_schema({'scene_ref':STR,'route':STR,'turns':{'type':'array','items':TURN,'minItems':8,'maxItems':8}}),
    'scene_native_mappings':{'type':'array','items':MAPPING},
    'ledger_facts':{'type':'array','items':object_schema({'turn_id':STR,'canonical_proposition':STR}),'minItems':8,'maxItems':8},
    'changes':STRS})
CAPS=['quantities','conjunction','evidence_states','causes','clocks','procedure','discourse']

def quote_errors(qs,ledger):
    ts={f['turn_id']:f['span']['quote'] for f in ledger['facts']}
    ts.update({n['source_id']+':'+str(t['turn_id']):t['utterance'] for n in ledger.get('native_utterances',[]) for t in n['turns']})
    if not qs:return ['missing_exact_quotes']
    return ['quote_mismatch:'+str(q.get('turn_ref')) for q in qs if not q.get('quote') or q.get('quote') not in ts.get(q.get('turn_ref'),'')]

def review_errors(r,root,ledger,scenes):
    if not r['accepted']:return r['failure_reasons']
    o=r.get('response',{});es=[]
    if o.get('root_id')!=root:es.append('review_root_mismatch')
    if o.get('decision')!='PASS' or o.get('issues'):es.append('semantic_source_not_pass')
    if set(o.get('checked_scenes',[]))!=set(scenes):es.append('incomplete_scene_review')
    es+=quote_errors(o.get('source_quotes'),ledger)
    caps={x['capability']:x for x in o.get('material_evidence',[])}
    if o.get('decision')=='PASS':
        if set(caps)!=set(CAPS):es.append('missing_material_capability_evidence')
        refs={f['turn_id'] for f in ledger['facts']}
        for k,v in caps.items():
            if not v['turn_refs'] or not set(v['turn_refs'])<=refs or not v['rationale']:es.append('invalid_capability_refs:'+k)
    return es

def packet(root,candidate,ledger):
    inp,parent,native=inputs(root)
    # Exact quotes and their normalized propositions are identical apart from whitespace.
    # No raw CLI events, frames, author rationale, question contracts or old outcomes.
    return {'root_id':root,'scene_sequence':inp['scene_sequence'],'speakers':sorted({t['speaker'] for t in parent['turns']}),
       'source_status':ledger['source_status'],'source_warnings':ledger['warnings'],
       'facts':[{'id':r,'speaker':s,'assertion':q} for r,s,q in compact(ledger)],
       'native_utterances':[[n['scene_ref'],n['source_id'],[[t['turn_id'],t['speaker'],t['utterance']] for t in n['turns']]] for n in native],
       'native_mapping':[[m['source_id'],m['native_turn_id'],m['disposition'],[[q['turn_ref'],q['quote']] for q in m['source_evidence']]] for m in candidate['native_evidence']],
       'mechanical_failures':ledger['mechanical_failures']}

def audit(root,candidate,ledger,rev,reviewer):
    p=packet(root,candidate,ledger)
    prompt='''Act as independent source auditor. This is a fresh review; there is no other reviewer verdict to follow.
Audit ONLY this root. Facts are speaker reports, preserving negation, plans, corrections, uncertainty and identity; they are not inferred world truth. Parent turns are immutable; historical weekday plans do not imply one week or completion. A scenes may add a fresh synthetic planning cycle. B scenes must faithfully report the exact reserved native utterances, including role/direction, requests, offers, refusals, confirmations, ambiguity and clocks; frames are not evidence. Courtesy can be omitted. Old-parent inconsistencies alone are warnings unless the extension amplifies or silently resolves them.
Check every six R03 scenes for parent continuity, natural sequential two-person dialogue, no new identity, source/native mapping fidelity and explicit bindings. Check material sufficiency across the cumulative ledger, with necessary R03 contribution: quantities (durations/coverage/units), conjunction, evidence_states, causes (at least five distinct explicitly motivated choices across source), clocks (six distinct clock values usable without inferring dates), procedure (six distinguishable actions with a source-defined unique dependency order, at least one necessary R03 fact; discourse order or a generic common-sense template is insufficient), discourse (six uniquely located complete speech contributions). Existing earlier-round explicit procedures may contribute, but explain the necessary R03 edge. Do not require finished actions when a prescribed procedure is explicit. Do not invent missing relations. No questions or rewritten source.
Return PASS only if all checks pass. On failure name a precise R03 scene and fact for each issue, and the smallest scene-local correction. If capability missing identify one scene where adding it preserves continuity. List checked_scenes as exactly the six scene IDs. Supply 3-6 exact source quotes. material_evidence must contain one row for each capability: quantities, conjunction, evidence_states, causes, clocks, procedure, discourse. For procedure cite the actual action-bearing turns and explain dependencies. Concise response under 900 words.
DATA\n'''+json_compact(p)
    return call(f'{root}_source_r{rev}_review{reviewer}',prompt,REVIEW)

def process(root):
    d=BASE/'sources'/root
    if (d/'OUTCOME.json').exists():return load(d/'OUTCOME.json')
    recovered=recover(root);out={'root_id':root,'status':'FAIL','repairs_used':0,'reviews':[],'failure_reasons':[]}
    # Preserve and count any author calls made by the duplicate worker against
    # the same root budget, including interrupted calls. Never hide them.
    external_repairs=list((BASE/'cli_calls').glob(root+'_R03_*_repair*'))
    out['duplicate_worker_repair_calls']=[str(p) for p in external_repairs]
    out['total_repair_budget']=2
    if not recovered['accepted']:
        out['failure_reasons']=recovered['failure_reasons'];atomic(d/'OUTCOME.json',out);return out
    candidate=load(d/'CANDIDATE_0.json')
    for rev in range(3):
        ledger=build(root,candidate,rev);atomic(d/f'LEDGER_{rev}.json',ledger)
        inp,parent,native=inputs(root);issues=list(ledger['mechanical_failures']);fail=[];reviews=[]
        if not issues:
            for reviewer in (1,2):
                rr=audit(root,candidate,ledger,rev,reviewer);es=review_errors(rr,root,ledger,inp['scene_sequence'])
                reviews.append({'reviewer':reviewer,'receipt':str(pathlib.Path(rr['directory'])/'RECEIPT.json'),'errors':es,'decision':(rr.get('response') or {}).get('decision')})
                fail+=es
                if rr['accepted'] and isinstance(rr.get('response'),dict):issues+=rr['response']['issues']
        else:fail=['mechanical_source_failed']
        out['reviews'].append({'revision':rev,'candidate':bind(d/f'CANDIDATE_{rev}.json'),'ledger':bind(d/f'LEDGER_{rev}.json'),'independent_reviews':reviews,'issues':issues,'failure_reasons':fail})
        if not fail and len(reviews)==2:
            full=copy.deepcopy(parent);full['turns']=parent['turns']+[t for s in candidate['scenes'] for t in s['turns']]
            full.update(source_status=inp['source_status'],warnings=candidate.get('warnings',[]),stable_v2_provenance={'candidate':bind(d/f'CANDIDATE_{rev}.json'),'ledger':bind(d/f'LEDGER_{rev}.json'),'parent':inp['parent_source_binding']})
            assert full['turns'][:len(parent['turns'])]==parent['turns']
            atomic(d/'FULL_SOURCE.json',full)
            out.update(status='PASS',source=bind(d/'FULL_SOURCE.json'),ledger=bind(d/f'LEDGER_{rev}.json'),failure_reasons=[]);break
        out['failure_reasons']=fail or ['source_not_pass'];out['failed_issues']=issues
        if rev==2:break
        if out['repairs_used']+len(external_repairs)>=2:
            out['failure_reasons'].append('total_root_repair_budget_exhausted_including_duplicate_worker');break
        target=next((x['scene_ref'] for x in issues if x.get('scene_ref') in inp['scene_sequence']),None)
        if not target:
            out['failure_reasons'].append('no_scene_identified_no_fabricated_repair');break
        scene=next(s for s in candidate['scenes'] if s['scene_ref']==target)
        prompt='''You are a targeted single-scene source repair author. Change only identified_scene and its fact ledger. This is a fixed repair budget, not source regeneration. Keep exactly eight turns, established speakers, same turn IDs and origin=codex_synthetic_extension. Return the complete scene and eight ledger_facts, each canonical_proposition exactly whitespace-normalized text of its corresponding repaired turn (reported assertion, no inference). Retain all sound facts and continuity with other scenes. Parent is immutable. Address the identified issues without inventing native facts; synthetic A planning is allowed. Mean words over all 48 new turns must remain <=18. If procedure missing, naturally specify a source-specific dependency chain of six concrete distinct actions within the scene; at least five adjacent source-defined dependencies, explicitly distinguishing prescribed from completed actions. Preserve other existing material. For B return complete mappings for only its reserved event; for A scene_native_mappings=[]. No other scene or entire root output.
DATA\n'''+json_compact({'root_id':root,'identified_scene':target,'issues':[x for x in issues if x.get('scene_ref')==target],
        'current_scene':scene,'context_ledger':compact(ledger),'native_event':[n for n in native if n['scene_ref']==target],
        'other_scene_word_count':sum(len(t['text'].split()) for s in candidate['scenes'] if s['scene_ref']!=target for t in s['turns'])})
        rr=call(f'{root}_scene_repair_{rev+1}',prompt,REPAIR);out['repairs_used']+=1
        if not rr['accepted']:
            out['failure_reasons']+=['scene_repair_call_failed']+rr['failure_reasons'];break
        o=rr['response'];new=o['scene'];es=[]
        if o['root_id']!=root or new['scene_ref']!=target or new['route']!=scene['route']:es.append('repair_scene_identity_mismatch')
        expected={t['turn_ref']:' '.join(t['text'].split()) for t in new['turns']}
        if {f['turn_id']:f['canonical_proposition'] for f in o['ledger_facts']}!=expected:es.append('repair_ledger_mismatch')
        if target.startswith('R03:A:') and o['scene_native_mappings']:es.append('A_repair_changed_native')
        if es:out['failure_reasons']+=es;break
        before=copy.deepcopy(candidate);candidate=copy.deepcopy(candidate)
        candidate['scenes']=[new if s['scene_ref']==target else s for s in candidate['scenes']]
        if target.startswith('R03:B:'):
            sid=next(n['source_id'] for n in native if n['scene_ref']==target)
            if any(m['source_id']!=sid for m in o['scene_native_mappings']):out['failure_reasons'].append('repair_cross_event_mapping');break
            candidate['native_evidence']=[m for m in candidate['native_evidence'] if m['source_id']!=sid]+o['scene_native_mappings']
        assert [s for s in before['scenes'] if s['scene_ref']!=target]==[s for s in candidate['scenes'] if s['scene_ref']!=target]
        atomic(d/f'CANDIDATE_{rev+1}.json',candidate)
        atomic(d/f'REPAIR_{rev+1}.json',{'scene_ref':target,'changes':o['changes'],'receipt':str(pathlib.Path(rr['directory'])/'RECEIPT.json'),
          'unchanged_scene_hashes':{s['scene_ref']:sha(json_compact(s).encode()) for s in candidate['scenes'] if s['scene_ref']!=target}})
    atomic(d/'OUTCOME.json',out);return out

def run(roots):
    results=[];status('source',status='RUNNING',pilot_roots=PILOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(process,root):root for root in roots}
        for f in concurrent.futures.as_completed(jobs):
            try:o=f.result()
            except Exception:
                o={'root_id':jobs[f],'status':'FAIL','failure_reasons':['engineering_exception'],'traceback':traceback.format_exc()}
                atomic(BASE/'sources'/jobs[f]/'OUTCOME.json',o)
            results.append(o);atomic(BASE/'SOURCE_PROGRESS.json',results)
            status(source_roots_finished=len(results),source_roots_passed=sum(x['status']=='PASS' for x in results))
    return [next(x for x in results if x['root_id']==r) for r in roots]

def revalidate_saved(root):
    """Correct mechanical quote indexing using saved actual final reviews only.

    No semantic call, verdict manufacture, root substitution or repair budget.
    Both prior outcome and exact validator binding are retained for audit.
    """
    d=BASE/'sources'/root;o=load(d/'OUTCOME.json')
    if o['status']=='PASS' or not o.get('reviews'):return o
    final=o['reviews'][-1];rs=final['independent_reviews']
    if len(rs)!=2:return o
    ledger=load(final['ledger']['path']);candidate=load(final['candidate']['path']);inp,parent,native=inputs(root)
    es=list(ledger['mechanical_failures']);checked=[]
    for row in rs:
        r=load(row['receipt']);errors=review_errors(r,root,ledger,inp['scene_sequence']);es+=errors
        checked.append({'receipt':bind(row['receipt']),'actual_saved_decision':(r.get('response') or {}).get('decision'),'errors':errors})
    receipt={'at':now(),'validator':bind(__file__),'scope':'Mechanical native quote lookup correction; no new semantic calls or verdicts',
      'prior_outcome':o,'final_candidate':final['candidate'],'final_ledger':final['ledger'],'reviews':checked,'failure_reasons':es}
    atomic(d/'SAVED_REVIEW_REVALIDATION.json',receipt)
    if es:return o
    full=copy.deepcopy(parent);full['turns']=parent['turns']+[t for s in candidate['scenes'] for t in s['turns']]
    full.update(source_status=inp['source_status'],warnings=candidate.get('warnings',[]),stable_v2_provenance={'candidate':final['candidate'],'ledger':final['ledger'],'parent':inp['parent_source_binding']})
    assert full['turns'][:len(parent['turns'])]==parent['turns']
    atomic(d/'FULL_SOURCE.json',full)
    o.update(status='PASS',source=bind(d/'FULL_SOURCE.json'),ledger=final['ledger'],failure_reasons=[],mechanical_revalidation=bind(d/'SAVED_REVIEW_REVALIDATION.json'))
    o['reviews'][-1]['independent_reviews']=[dict(row,errors=[]) for row in rs]
    o['reviews'][-1]['failure_reasons']=[]
    atomic(d/'OUTCOME.json',o);return o

if __name__=='__main__':
    raise SystemExit('Use python -B pipeline.py: single owner lock is mandatory.')
