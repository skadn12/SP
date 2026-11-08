"""One slot per fresh author, blind Gold and content-review session."""
from __future__ import annotations
import concurrent.futures, copy, re, sys, traceback
sys.dont_write_bytecode=True
from common import *
from codex_client import call,object_schema,STR,STRS,QUOTES
from source_engine import quote_errors

sys.path.insert(0,'/root/autodl-tmp/gxy-selfplay/scripts/current_method')
import production_question_gate as production
SLOTS={'MC2':('multi_select','MC_QUANTITATIVE_RELATION',2),
       'MC3':('multi_select','MC_CONSTRAINT_CONJUNCTION',3),
       'MC4':('multi_select','MC_EVIDENCE_SELECTION',4),
       'MC5':('multi_select','MC_CAUSAL_SUPPORT',5),
       'MC6':('multi_select','MC_CONSTRAINT_CONJUNCTION',6),
       'ORDER1':('ordering','ORDER_EXPLICIT_TIME',6),
       'ORDER2':('ordering','ORDER_EXPLICIT_PROCEDURE',6),
       'ORDER3':('ordering','ORDER_DISCOURSE_RECONSTRUCTION',6)}
LABELS=list('ABCDEF')
EVIDENCE=object_schema({'turn_ref':STR,'speaker':STR,'quote':STR})
EVS={'type':'array','items':EVIDENCE,'minItems':1}
GOLD=object_schema({'candidate_id':STR,'decision':{'type':'string','enum':['PASS','FAIL']},
                  'gold':STRS,'source_quotes':QUOTES,'failure_reasons':STRS})
CONTENT=object_schema({'candidate_id':STR,'decision':{'type':'string','enum':['PASS','REVISE','FAIL']},
    'source_quotes':QUOTES,'failure_reasons':STRS,'target_realization':{'type':'string','enum':['SUPPORTED','HOLD','NOT_APPLICABLE']},'target_reason':STR})
PLAN=object_schema({'candidate_id':STR,'task_operation':STR,'source_relations':STRS,
    'necessary_r03_turn_refs':STRS,'r03_necessity':STR,'quality_acceptance':STRS,
    'feedback_dispositions':{'type':'array','items':object_schema({'feedback_id':STR,'status':{'type':'string','enum':['APPLICABLE','INAPPLICABLE']},'reason':STR})}})

CONTRACT='''English question 30-90 words; each of six options/items 10-35 words, with matched detail and syntax. Public wording never reveals intended Gold count. No synonymous claims to fill quotas. For MC false alternatives change one verifiable relation, all other details grounded; supported criterion must distinguish unsupported from explicitly false. Exact evidence for ALL A-F, including false items; should_select must match private Gold. Every slot requires a necessary R03 premise, explained in r03_necessity. MC2 requires quantitative combination, units and scoped participant coverage. MC3/MC6 require conjunctions of entity/action/state constraints; MC6 is an all-select control. MC4 evidence unit (single turn or named combination) must be explicit; every printed quote/coordinate exact. MC5 requires five distinct explicit cause-action relations and one controlled wrong edge joining individually true facts. ORDER1 compares six distinct clock-face values, independent of dates; items hide direct clock values. ORDER2 uses a source-prescribed six-action procedure with unique source-defined dependencies; do not substitute discourse order/common sense or misrepresent prospective actions as completed. ORDER3 reconstructs actual source speech contributions, each complete action/object/request combination uniquely located, not real event order. No editorial ranking words (first/later/finally/last/initially) in ordering items. At least five exact-source-evidenced adjacent edges must define a unique Gold total order; each edge must be entailed for the stated operation. Canonical item writing order is arbitrary; display is shuffled programmatically. Source status, warnings, entity identity and uncertainty are binding. No world knowledge, hidden source facts, Gold quota hints, solver outputs or other arm information. Distinguish content quality from whether a hypothesized error is realized.'''

def cid(root,arm,slot):return 'STABLE2-'+arm+'-'+root.split('-')[-1]+'-'+slot

def output_instruction(root,arm,slot,binding):
    kind,op,n=SLOTS[slot]
    fixed={'candidate_id':cid(root,arm,slot),'root_id':root,'arm':arm,'slot_id':slot,
      'author_id':arm,'question_type':kind,'task_operation':op,'source_binding':binding}
    return ('Return the CANDIDATE ITSELF as the root JSON object, with no candidate/slot/source wrapper. '
      'Use these exact root identity fields: '+json_compact(fixed)+'. '
      'Also include root question (string), '+('items' if kind=='ordering' else 'options')+
      ' (object mapping A,B,C,D,E,F to strings), gold (array of '+str(n)+' distinct labels), '
      'r03_necessity (string), item_analysis (object A..F, each with evidence [{turn_ref,speaker,quote}], rationale'+
      (',should_select boolean,changed_relation string or null' if kind=='multi_select' else '')+'). '+
      ('Also include ordering_analysis {operation_class,source_dependence_rationale,edges:[{before,after,evidence:[{turn_ref,speaker,quote}],rationale}],mention_order_boundary,editorial_cue_policy,common_sense_boundary}. ' if kind=='ordering' else '')+
      'Do not echo input slot metadata, input source packet, audit verdicts or feedback as your answer. ')

def schema(root,arm,slot,binding):
    kind,op,n=SLOTS[slot]
    props={k:{'type':'string','enum':[v]} for k,v in {'candidate_id':cid(root,arm,slot),'root_id':root,'arm':arm,
      'slot_id':slot,'author_id':arm,'question_type':kind,'task_operation':op}.items()}
    props['source_binding']=object_schema({'path':STR,'sha256':STR,'bytes':{'type':'integer'}})
    props.update(question=STR,gold={'type':'array','items':{'type':'string','enum':LABELS},'minItems':n,'maxItems':n},r03_necessity=STR)
    props['items' if kind=='ordering' else 'options']=object_schema({k:STR for k in LABELS})
    analysis={'evidence':EVS,'rationale':STR}
    if kind=='multi_select':analysis.update(should_select={'type':'boolean'},changed_relation={'type':['string','null']})
    props['item_analysis']=object_schema({k:object_schema(analysis) for k in LABELS})
    if kind=='ordering':
        props['ordering_analysis']=object_schema({'operation_class':{'type':'string','enum':[{'ORDER1':'clock_value','ORDER2':'source_defined_dependency','ORDER3':'discourse_reconstruction'}[slot]]},
        'source_dependence_rationale':STR,'edges':{'type':'array','minItems':5,'items':object_schema({'before':STR,'after':STR,'evidence':EVS,'rationale':STR})},
        'mention_order_boundary':STR,'editorial_cue_policy':STR,'common_sense_boundary':STR})
    return object_schema(props)

def feedback(slot):
    op=SLOTS[slot][1];p=BASE.parent/'codex_route_a_feedback_batch_20261024_v1'
    cards=load(p/'CHALLENGER_PUBLIC_CARDS.json')['cards'];task=load(p/'QUESTIONER_FEEDBACK.json')
    result=[];dispositions=[]
    for c in cards:
        qt=c.get('question_task',{});applies=op in qt.get('operation_ids',[])
        dispositions.append({'feedback_id':c['feedback_id'],'status':'APPLICABLE' if applies else 'INAPPLICABLE',
          'reason':'frozen operation match' if applies else 'card operation outside this frozen slot'})
        if applies:
            result.append({'feedback_id':c['feedback_id'],'candidate_hypothesis':c.get('candidate_hypothesis'),
                'question_task':qt,'questioner_feedback':c.get('questioner_feedback'),
                'boundary_conditions':c.get('boundary_conditions',[]),
                'limitations':['Exploratory transfer of a candidate hypothesis, not confirmed mechanism or training benefit.']})
    return {'projection':'verbatim task fields and feedback; no original answers, labels or scores',
            'cards':result,'all_card_dispositions':dispositions,'bindings':[bind(p/n) for n in ['CHALLENGER_PUBLIC_CARDS.json','QUESTIONER_FEEDBACK.json']]}

def evidence_packet(entry,slot):
    ledger=load(entry['ledger']['path']);facts=ledger['facts'];selected=set()
    scenes={'MC2':['R03:A:S01','R03:A:S03'],'MC3':['R03:A:S01','R03:A:S02'],
      'MC4':['R03:A:S02','R03:A:S03'],'MC5':['R03:A:S01','R03:A:S02','R03:A:S03'],
      'MC6':['R03:A:S02','R03:A:S03'],'ORDER1':['R03:A:S01','R03:A:S02','R03:A:S03'],
      'ORDER2':[],'ORDER3':['R03:B:S01','R03:B:S02','R03:B:S03']}[slot]
    for f in facts:
        if f['scope'] in scenes:selected.add(f['turn_id'])
        if slot=='ORDER1' and f['explicit_clocks']:selected.add(f['turn_id'])
        if slot=='MC5' and re.search(r'\b(because|so that|in order to|reason|since|want|avoid)\b',f['span']['quote'],re.I):selected.add(f['turn_id'])
    if slot=='ORDER2':
        # Use both independently reviewed capability receipts to find the exact
        # dependency-bearing spans, retaining their scene context.
        for rev in entry['reviews'][-1]['independent_reviews']:
            o=load(rev['receipt'])['response']
            for cap in o['material_evidence']:
                if cap['capability']=='procedure':selected.update(cap['turn_refs'])
        scopes={f['scope'] for f in facts if f['turn_id'] in selected}
        for f in facts:
            if f['scope'] in scopes and f['scope']!='parent':selected.add(f['turn_id'])
    # Include immediate neighbors of selected parent assertions for antecedent
    # resolution; do not ship the entire parent or full native event packets.
    originally_selected=set(selected)
    for i,f in enumerate(facts):
        if f['scope']=='parent' and f['turn_id'] in originally_selected:
            for g in facts[max(0,i-1):i+2]:selected.add(g['turn_id'])
    rows=[[f['turn_id'],f['speaker'],f['span']['quote']] for f in facts if f['turn_id'] in selected]
    return {'root_id':entry['root_id'],'source_binding':entry['source'],'ledger_binding':entry['ledger'],
       'source_status':ledger['source_status'],'source_warnings':ledger['warnings'],
       'evidence_scope':'Exact selected assertions and needed antecedents; do not infer absence outside supplied evidence.',
       'fact_ledger':rows},ledger

def candidate_errors(c,entry,arm,slot,packet):
    if not isinstance(c,dict):return ['missing_candidate']
    import jsonschema
    es=[]
    try:jsonschema.validate(c,schema(entry['root_id'],arm,slot,entry['source']))
    except Exception as e:return ['candidate_schema:'+str(e)[:300]]
    if c['source_binding']!=entry['source']:es.append('source_binding_mismatch')
    if len(set(c['gold']))!=len(c['gold']):es.append('duplicate_gold_label')
    if not 30<=len(c['question'].split())<=90:es.append('question_words_outside_30_90')
    choices=c.get('options',c.get('items'))
    for k,v in choices.items():
        if not 10<=len(v.split())<=35:es.append('item_words_outside_10_35:'+k)
    if len(set(v.lower().strip() for v in choices.values()))!=6:es.append('duplicate_item')
    allowed={r[0] for r in packet['fact_ledger']}
    for k,a in c['item_analysis'].items():
        if any(q['turn_ref'] not in allowed for q in a['evidence']):es.append('quote_outside_supplied_packet:'+k)
    try:
        a=production.audit_candidate(c,BASE/'SOURCE_SELECTION.json')
        es += ['production:'+json_compact(e) for e in a['mechanical_errors']]
        es += ['presentation:'+f['code'] for f in a['presentation_lint']['flags'] if f['code'] in ('QUESTION_REVEALS_SELECTION_COUNT','ALL_ORDERING_ITEMS_EXPOSE_CLOCK_TIME','ORDERING_DISCOURSE_CUE')]
    except Exception as e:es.append('production_exception:'+str(e))
    return es

def audit(entry,arm,slot,c,packet,ledger,attempt):
    root=entry['root_id'];public={k:c[k] for k in ['candidate_id','question_type','task_operation','question','options','items'] if k in c}
    # Gold and reviewer are independent fresh sessions; reviewer does not see
    # Gold assessor output, private author analyses or another arm.
    gs=copy.deepcopy(GOLD);gs['properties']['candidate_id']={'type':'string','enum':[c['candidate_id']]}
    gs['properties']['gold']={'type':'array','items':{'type':'string','enum':LABELS},'maxItems':6}
    cs=copy.deepcopy(CONTENT);cs['properties']['candidate_id']={'type':'string','enum':[c['candidate_id']]}
    gr=call(f'{root}_{arm}_{slot}_a{attempt}_gold',
      'You are the independent blind Gold assessor. Solve exactly this one question from supplied exact evidence. No author answer, analyses or Gold quota is supplied. Derive ALL supported labels or unique ordering; reject ambiguity, insufficient evidence, nonunique order or wrong operation. Give exact quotes sufficient to justify every item, including rejection, and failure_reasons. PASS only if this question is uniquely solvable. Return the review itself at root with candidate_id, decision, gold, source_quotes and failure_reasons.\n'+json_compact({'question':public,'source':packet}),gs)
    cr=call(f'{root}_{arm}_{slot}_a{attempt}_content',
      'You are the independent content reviewer. Review exactly this one slot, independently deriving its correctness. Check every contract requirement, six distinct items, plausible minimal relation distractors, exact evidence, private Gold, necessary R03 premise, source-specific unique order/dependencies, no editorial/count/clock shortcuts, proper prospective/completed status. Content PASS does not require hypothesized solver errors; mark target HOLD honestly. Return precise failures and exact quotes, without rewriting. Return the review itself at root with candidate_id, decision, source_quotes, failure_reasons, target_realization and target_reason.\nCONTRACT\n'+CONTRACT+'\n'+json_compact({'candidate':c,'source':packet,'required_gold_count':SLOTS[slot][2]}),cs)
    es=candidate_errors(c,entry,arm,slot,packet)
    for name,r in [('gold',gr),('content',cr)]:
        if not r['accepted']:es+=[name+':'+x for x in r['failure_reasons']];continue
        o=r['response']
        if o['candidate_id']!=c['candidate_id']:es.append(name+':identity')
        if o['decision']!='PASS' or o['failure_reasons']:es.append(name+':not_pass')
        es += [name+':'+x for x in quote_errors(o['source_quotes'],ledger)]
        if any(q['turn_ref'] not in {r[0] for r in packet['fact_ledger']} for q in o['source_quotes']):es.append(name+':quote_outside_evidence')
    if gr['accepted']:
        ag=c['gold'];gg=gr['response']['gold']
        if (ag!=gg if c['question_type']=='ordering' else set(ag)!=set(gg)):es.append('blind_gold_disagreement')
    return {'status':'PASS' if not es else 'FAIL','errors':es,'gold':gr,'content':cr}

def process(entry,arm,slot):
    root=entry['root_id'];d=BASE/'questions'/root/arm/slot
    if (d/'OUTCOME.json').exists():return load(d/'OUTCOME.json')
    out={'root_id':root,'arm':arm,'slot_id':slot,'candidate_id':cid(root,arm,slot),'status':'MISSING','failure_reasons':[],'revisions_used':0}
    packet,ledger=evidence_packet(entry,slot);fb=feedback(slot) if arm=='F' else None
    atomic(d/'EVIDENCE_PACKET.json',packet)
    if fb:atomic(d/'FEEDBACK_PACKET.json',fb)
    meta={'root_id':root,'arm':arm,'slot_id':slot,'candidate_id':out['candidate_id'],'question_type':SLOTS[slot][0],
      'task_operation':SLOTS[slot][1],'private_required_gold_count':SLOTS[slot][2]}
    plan_schema=copy.deepcopy(PLAN)
    plan_schema['properties']['candidate_id']={'type':'string','enum':[out['candidate_id']]}
    plan_schema['properties']['task_operation']={'type':'string','enum':[SLOTS[slot][1]]}
    pl=call(f'{root}_{arm}_{slot}_plan','Plan exactly this one slot BEFORE drafting. Freeze source relations, necessary R03 refs, r03_necessity and quality checks; no candidate text or Gold now. Return the plan itself at the root, including candidate_id exactly '+out['candidate_id']+' and task_operation exactly '+SLOTS[slot][1]+'. For F disposition every card in all_card_dispositions, with honest source instantiation and boundary; G feedback_dispositions=[].\nCONTRACT\n'+CONTRACT+'\n'+json_compact({'slot':meta,'source':packet,'feedback':fb}),plan_schema)
    if not pl['accepted']:
        out['failure_reasons']=['plan_call_failed']+pl['failure_reasons'];atomic(d/'OUTCOME.json',out);return out
    p=pl['response'];refs=set(p['necessary_r03_turn_refs']);allowed={r[0] for r in packet['fact_ledger']}
    es=[]
    if p['candidate_id']!=out['candidate_id'] or p['task_operation']!=SLOTS[slot][1]:es.append('plan_identity')
    if not refs or not refs<=allowed or not any(r.startswith('R03:') for r in refs):es.append('plan_necessary_R03_refs')
    if arm=='F' and {x['feedback_id'] for x in p['feedback_dispositions']}!={x['feedback_id'] for x in fb['all_card_dispositions']}:es.append('plan_feedback_coverage')
    if es:out['failure_reasons']=es;atomic(d/'OUTCOME.json',out);return out
    atomic(d/'TASK_PLAN_FREEZE.json',{'plan':p,'receipt':str(pathlib.Path(pl['directory'])/'RECEIPT.json'),'at':now()})
    initial=call(f'{root}_{arm}_{slot}_draft','Author exactly ONE question slot implementing the frozen plan. '+output_instruction(root,arm,slot,entry['source'])+'Do not author other slots. All references must be exact substrings within supplied evidence. Keep output compact; include full required analyses.\nCONTRACT\n'+CONTRACT+'\n'+json_compact({'slot':meta,'source':packet,'plan':p,'feedback':fb}),schema(root,arm,slot,entry['source']))
    if not initial['accepted']:
        out['failure_reasons']=['initial_author_call_failed']+initial['failure_reasons'];atomic(d/'OUTCOME.json',out);return out
    c=initial['response']
    for attempt in (0,1):
        atomic(d/f'CANDIDATE_{attempt}.json',c)
        # Mechanical failure still gets explicit feedback, and consumes the
        # single revision rather than retrying the semantic response.
        mech=candidate_errors(c,entry,arm,slot,packet)
        a=audit(entry,arm,slot,c,packet,ledger,attempt) if not mech else {'status':'FAIL','errors':mech}
        atomic(d/f'AUDIT_{attempt}.json',a)
        if a['status']=='PASS':out.update(status='PASS',candidate=c,final_audit=bind(d/f'AUDIT_{attempt}.json'),failure_reasons=[]);break
        out['failure_reasons']=a['errors']
        if attempt==1:break
        rr=call(f'{root}_{arm}_{slot}_revision1','Make the ONE allowed targeted revision to this ONE existing slot. '+output_instruction(root,arm,slot,entry['source'])+'Preserve identity, operation, source and fixed Gold quota. Repair named mechanical/Gold/content errors; no new facts or alternate slot. Return full candidate.\nCONTRACT\n'+CONTRACT+'\n'+json_compact({'candidate':c,'source':packet,'frozen_plan':p,'errors':a['errors'],
            'independent_gold':a.get('gold',{}).get('response'),'content_review':a.get('content',{}).get('response')}),schema(root,arm,slot,entry['source']))
        out['revisions_used']=1
        if not rr['accepted']:out['failure_reasons']+=['revision_call_failed']+rr['failure_reasons'];break
        c=rr['response']
    atomic(d/'OUTCOME.json',out);return out

def run(entries,roots,label):
    atomic(BASE/'SOURCE_SELECTION.json',{'roots':[{'root_id':e['root_id'],**e['source']} for e in entries if e['status']=='PASS']})
    results=[];jobs={};status('questions')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for e in entries:
            if e['status']!='PASS':
                results.extend({'root_id':e['root_id'],'arm':arm,'slot_id':s,'candidate_id':cid(e['root_id'],arm,s),'status':'MISSING','failure_reasons':['source_not_pass']} for arm in ('F','G') for s in SLOTS)
                continue
            for arm in ('F','G'):
                for s in SLOTS:jobs[pool.submit(process,e,arm,s)]=(e['root_id'],arm,s)
        for f in concurrent.futures.as_completed(jobs):
            root,arm,s=jobs[f]
            try:o=f.result()
            except Exception:
                o={'root_id':root,'arm':arm,'slot_id':s,'candidate_id':cid(root,arm,s),'status':'MISSING','failure_reasons':['engineering_exception'],'traceback':traceback.format_exc()}
                atomic(BASE/'questions'/root/arm/s/'OUTCOME.json',o)
            results.append(o);atomic(BASE/f'{label}_QUESTION_PROGRESS.json',results)
            status(question_valid_slots=sum(x['status']=='PASS' for x in results),question_finished_slots=len(results),question_planned_slots=len(roots)*16)
    seen={}
    for x in sorted(results,key=lambda x:x['candidate_id']):
        if x['status']!='PASS':continue
        c=x['candidate'];key=json_compact([c['question_type'],c['question'],c.get('options',c.get('items'))])
        if key in seen:x.update(status='MISSING',failure_reasons=['full_question_duplicate_of:'+seen[key]])
        else:seen[key]=x['candidate_id']
    result={'at':now(),'planned_slots':len(roots)*16,'valid_slots':sum(x['status']=='PASS' for x in results),'slots':results}
    assert len(results)==len(roots)*16
    atomic(BASE/f'{label}_QUESTION_FREEZE.json',result)
    atomic(BASE/f'{label}_CANDIDATES_PRIVATE.json',{'candidates':[x['candidate'] for x in results if x['status']=='PASS']})
    return result
