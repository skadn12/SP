#!/usr/bin/env python3
"""Reproducible score-free evidence union; no semantic-clearance claim."""
import collections, hashlib, itertools, json, pathlib, re, unicodedata
BASE = pathlib.Path('/root/autodl-tmp/gxy-selfplay')
WEEK = BASE / 'research-protocols/selfplay_week_20261019'
OUT = pathlib.Path(__file__).resolve().parent
FIELDS = ['exclusion_record_id','root_id','original_split','source_sha256','canonical_fact_family_id','semantic_fact_fingerprints','related_family_ids','evidence_turn_refs','scope','exposure_role','exclusion_manifest_sha256']
BOUND, GAPS, OCC, EXPOSURES, SOURCES = {}, [], [], {}, {}
COUNTS, ROWS, INTERNAL = {}, [], []

def sha(data): return hashlib.sha256(data).hexdigest()
def blob(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode()
def write(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+chr(10))
def bind(path, role='metadata'):
    p=pathlib.Path(path)
    if not p.is_absolute(): p=BASE/p
    if not p.is_file():
        GAPS.append({'code':'MISSING_INPUT','path':str(p),'role':role}); return None
    prev=BOUND.get(str(p))
    h=prev['sha256'] if prev else sha(p.read_bytes())
    if prev and prev['sha256']!=h: raise RuntimeError('Input mutated during extraction: '+str(p))
    BOUND[str(p)]={'path':str(p),'sha256':h,'bytes':p.stat().st_size,'roles':sorted(set((prev or {}).get('roles',[])+[role]))}
    return p
def read(path,role='metadata'):
    p=bind(path,role)
    if p is None: return None
    if p.suffix=='.jsonl': return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    return json.loads(p.read_text())
def norm(s):
    s=unicodedata.normalize('NFKC',str(s)).replace('’',"'").replace('“','"').replace('”','"')
    return re.sub(r'\s+',' ',s).strip()
def source(path,expected,root,role):
    if not path: return None
    p=bind(path,'source')
    if not p: return None
    actual=BOUND[str(p)]['sha256']
    if expected and actual!=expected:
        GAPS.append({'code':'SOURCE_HASH_MISMATCH','root_id':root,'path':str(p),'expected':expected,'actual':actual})
        return None
    key=(root,actual)
    if key not in SOURCES:
        d=json.loads(p.read_text());turns=d.get('turns',[])
        td={t.get('turn_ref',t.get('turn_id')):t for t in turns if isinstance(t,dict)}
        if len(td)!=len(turns):GAPS.append({'code':'DUPLICATE_OR_MISSING_TURN_REF','root_id':root,'path':str(p)})
        SOURCES[key]={'path':str(p),'sha256':actual,'turns':td,'count':len(turns)}
    z=EXPOSURES.setdefault(key,{'root_id':root,'source_sha256':actual,'source_paths':set(),'exposure_roles':set(),'turn_count':SOURCES[key]['count'],'meaning':'ROOT_SOURCE_EXPOSURE_ONLY; not all-turn family exposure and not automatic root exclusion'})
    z['source_paths'].add(str(p));z['exposure_roles'].add(role)
    return SOURCES[key]

REF_RX=re.compile(r'^(?:R\d{2}(?::[AB])?:S\d{2}:T\d{3}|cc-train-episode-\d+:P\d+:T\d{3})$')
EVIDENCE_KEYS={'evidence_refs','evidence_groups','option_evidence','evidence_by_label','item_evidence','order_anchor_spans','minimal_evidence_refs','evidence_span','all_mentions','all_mentions_by_label','all_occurrences','anchor_by_label','fixed_turns','source_turn_ref','evidence','included_mentions','occurrences_by_label','required_refs'}
def harvest(obj):
    refs=set();quotes=collections.defaultdict(set)
    def go(x):
        if isinstance(x,dict):
            rr=x.get('turn_ref') or x.get('turn_id') or x.get('source_turn_ref')
            if isinstance(rr,str) and REF_RX.match(rr):
                refs.add(rr); q=x.get('quote')
                if isinstance(q,str) and q.strip():quotes[rr].add(norm(q))
            for k,v in x.items():
                if k in {'reason','rationale','proposition','statement','true_text','false_text','question','options','items','gold','answer','prediction','score','status','semantic_role','relation','verdict'}:continue
                if isinstance(v,(dict,list)):go(v)
                elif isinstance(v,str) and REF_RX.match(v):refs.add(v)
        elif isinstance(x,list):
            for v in x:go(v)
        elif isinstance(x,str) and REF_RX.match(x):refs.add(x)
    go(obj);return refs,quotes
def selected_evidence(q):
    obj={k:v for k,v in q.items() if k in EVIDENCE_KEYS}
    if isinstance(q.get('ordering'),dict):
        oo=q['ordering'];obj['ordering_occurrences']=[{'all_occurrences':x.get('all_occurrences',[])} for x in oo.get('items',[])]
    return harvest(obj)
def source_binding(q,wrapper=None,fallback=None):
    w=wrapper or {}; s=q.get('source_binding') or w.get('source_binding') or {}
    path=q.get('source_path') or q.get('dialogue_path') or s.get('source_path') or s.get('path') or w.get('dialogue_path')
    h=q.get('source_sha256') or q.get('dialogue_sha256') or s.get('source_sha256') or s.get('sha256') or w.get('dialogue_sha256')
    if not path and fallback:path,h=fallback
    return path,h
def fingerprint(turn,text):
    speaker=norm(turn.get('speaker','UNKNOWN')); t=norm(text)
    # Keep the complete source clause. Lexical components aid retrieval; do not invent resolved objects.
    ns=norm(t).lower(); actor=speaker
    names=sorted(set(re.findall(r'\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+){0,3}\b',t))-{ 'I','The','It','That','This','Yes','No','We','You','They','My','But','And','So','Both','Before','After'})
    neg=bool(re.search(r"\b(?:not|no|never|without|neither|nothing|nobody|cannot|can't|won't|haven't|hasn't|didn't|don't|doesn't|wasn't|weren't)\b",ns))
    cond=[norm(x) for x in re.findall(r'\b(?:if|unless|only after|only when|provided that|until|before|after)\b[^.!?;]*',t,re.I)]
    times=sorted(set(re.findall(r'\b(?:\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\.?|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow|yesterday|tonight|next week|last week|now|later|earlier|first|then)\b',t,re.I)))
    modes=[m for m,pat in [('reported_completed',r"\b(?:have|has|I've|we've|had)\s+(?:[a-z]+ed|sent|put|written|made|taken|left|done|begun|bought|seen)\b"),('future_or_plan',r"\b(?:will|I'll|we'll|going to|plan|planning|intend)\b"),('conditional_or_uncertain',r'\b(?:may|might|could|would|perhaps|maybe)\b'),('question',r'\?')] if re.search(pat,t,re.I)]
    key=norm(actor).lower()+'|reports|'+ns
    return {'fingerprint_id':'SFACT-'+sha(key.encode())[:24], 'canonical_proposition':speaker+' reports: '+t,'entities':sorted(set([speaker]+names)),'actor':actor,'action':'reports_source_clause','object':t,'condition':cond,'polarity':'contains_explicit_negation' if neg else 'no_explicit_negation_detected','time':times,'modality':modes or ['source_report_not_resolved'],'projection_method':'exact_source_clause_with_lexical_components','semantic_normalization_status':'PENDING_ENTITY_ACTION_OBJECT_MODALITY_REVIEW'}

def add(q,role,input_path,wrapper=None,fallback=None,forced_split=None):
    root=q.get('root_id') or (wrapper or {}).get('root_id'); cid=q.get('candidate_id') or q.get('fact_id') or q.get('slot_id')
    if not root or not cid:
        GAPS.append({'code':'MISSING_QUESTION_ID_OR_ROOT','input':str(input_path),'role':role});return
    split=forced_split or q.get('split') or (wrapper or {}).get('split') or SPLIT.get(root,'UNKNOWN')
    path,h=source_binding(q,wrapper,fallback)
    src=source(path,h,root,role)
    refset,quotes=selected_evidence(q);errors=[]
    if not src:errors.append('MISSING_OR_UNVERIFIED_SOURCE')
    if not refset:errors.append('NO_SOURCE_EVIDENCE_REFS')
    facts={};refs=[]
    if src:
        for ref in sorted(refset):
            turn=src['turns'].get(ref)
            if not turn:errors.append('MISSING_EVIDENCE_TURN:'+ref);continue
            text=norm(turn.get('text',turn.get('utterance','')));qs=quotes.get(ref,set())
            chosen=[]
            for quote in sorted(qs):
                if norm(quote).lower() in text.lower():chosen.append(quote)
                else:errors.append('QUOTE_NOT_EXACT_SOURCE_SUBSTRING:'+ref)
            if not chosen:chosen=[text]
            for textpiece in chosen:
                fp=fingerprint(turn,textpiece);facts[fp['fingerprint_id']]=fp
            refs.append(ref)
    # Never condition facts on relation/verdict/Gold. All evidence branches are unioned.
    typ=q.get('question_type','material'); group=q.get('option_evidence') or q.get('evidence_by_label')
    opts=q.get('options') or q.get('options_or_items')
    if not group and isinstance(q.get('evidence'),dict) and isinstance(opts,dict):group=q['evidence']
    option_coverage='NOT_APPLICABLE'
    if typ in ('multi_select','single_choice'):
        nested=q.get('multi_select')
        if isinstance(nested,dict):
            spans=q.get('evidence_span',{}).get('evidence_refs',[])
            available={v.get('ref_id') for v in spans if harvest(v)[0]}
            branches=nested.get('options',[])
            all_mapped=len(branches)==6 and all(set(x.get('support_ref_ids',[])+x.get('refute_ref_ids',[])) and set(x.get('support_ref_ids',[])+x.get('refute_ref_ids',[]))<=available for x in branches)
            option_coverage='ALL_OPTION_EVIDENCE_PRESENT' if all_mapped else 'ALL_OPTION_EVIDENCE_INCOMPLETE'
            if not all_mapped:errors.append(option_coverage)
        elif isinstance(group,dict) and isinstance(opts,dict):
            option_coverage='ALL_OPTION_EVIDENCE_PRESENT' if set(opts)<=set(group) and all(harvest(group[k])[0] for k in opts) else 'ALL_OPTION_EVIDENCE_INCOMPLETE'
            if option_coverage.endswith('INCOMPLETE'):errors.append(option_coverage)
        else:option_coverage='LEGACY_ALL_OPTION_EVIDENCE_UNVERIFIED';errors.append(option_coverage)
    atomic=[]
    if typ=='ordering':
        if isinstance(q.get('ordering'),dict):
            atomic=[norm(x.get('event_semantic_key','')) for x in q['ordering'].get('items',[]) if x.get('event_semantic_key')]
        else:
            ev=q.get('item_evidence') or q.get('order_anchor_spans') or q.get('evidence')
            if not ev and isinstance(q.get('fixed_turns'),dict):ev={k:{'turn_ref':v} for k,v in q['fixed_turns'].items()}
            if isinstance(ev,dict):
                for v in ev.values():
                    rr,qq=harvest(v);parts=[]
                    for ref in sorted(rr):
                        turn=(src or {}).get('turns',{}).get(ref)
                        if turn:parts.append(norm(turn.get('speaker',''))+' reports '+norm(v.get('quote') or turn.get('text','')))
                    if parts:atomic.append(' ; '.join(parts))
        if len(atomic)!=len(q.get('items',q.get('ordering',{}).get('items',[]))):errors.append('ORDER_ATOMIC_EVENT_PROJECTION_UNRESOLVED')
    keys=sorted(facts)
    family='FACTSET-v1-'+sha(blob(keys))[:24] if keys else 'UNRESOLVED_FAMILY_MAPPING-'+sha(blob([root,cid,h]))[:24]
    rec={'exclusion_record_id':'EXCL-'+sha(blob([str(input_path),cid,role]))[:24],'root_id':root,'original_split':split,'source_sha256':src['sha256'] if src else h,'canonical_fact_family_id':family,'semantic_fact_fingerprints':[facts[k] for k in keys],'related_family_ids':[],'evidence_turn_refs':sorted(set(refs)),'scope':{'comparison_unit':'actual_question_all_evidence_union_not_root','question_type':typ,'evidence_mapping_status':'UNRESOLVED_FAMILY_MAPPING' if errors else 'EVIDENCE_UNION_MAPPED_SEMANTIC_REVIEW_PENDING','all_options_evidence_coverage':option_coverage,'unordered_atomic_event_set':sorted(set(atomic)),'semantic_near_duplicate_clearance':'PENDING','all_source_turns_excluded':False,'source_item_count':len(q.get('items') or q.get('ordering',{}).get('items',[])) if typ=='ordering' else None,'mapping_limitation_interpretation':'Missing per-option schema or item-to-turn links is a support-completeness uncertainty; not proof cited facts are invalid. Boundary semantic reviewers may restore mappings from original source-grounded propositions without exposing question labels to authors.' if errors else None},'exposure_role':[role],'exclusion_manifest_sha256':None}
    ROWS.append(rec);INTERNAL.append({'exclusion_record_id':rec['exclusion_record_id'],'candidate_id':cid,'input_path':str(input_path),'role':role,'root_id':root,'family_id':family,'mapping_errors':sorted(set(errors)),'recorded_evidence_ref_count':len(refset),'mapped_evidence_ref_count':len(refs),'fact_count':len(keys),'source_sha256':src['sha256'] if src else h})
    OCC.append((role,cid,root));return rec

def questions(path,key,role,forced_split=None):
    d=read(path,'question_inventory:'+role)
    if d is None:return
    qs=d[key] if key else d;COUNTS[role]={'input_rows':len(qs),'input_path':str(BASE/path) if not pathlib.Path(path).is_absolute() else str(path)}
    for q in qs:add(q,role,path,forced_split=forced_split)

def main():
    global SPLIT
    OUT.mkdir(exist_ok=True)
    bind(pathlib.Path(__file__).resolve(),'extraction_algorithm')
    protocol=read(WEEK/'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json','frozen_boundary_protocol')
    read(OUT.parent/'protocol/EXPOSURE_AND_FAMILY_INTERPRETATION.json','prospective_boundary_interpretation')
    read(OUT/'COVERAGE_PLAN.json','extraction_coverage_plan')
    read(WEEK/'day1/reuse/INPUT_BINDINGS.json','legacy_reuse_bindings')
    for package,name in [('h4_codex_dev_test_ab_v124','MANIFEST.json'),('h4_codex_dev_eval_v127','SCOPE.json'),('h4_codex_held_roots_eval_v134','evaluation/SCOPE.json'),('h4_codex_structure_pilot_v140','PLAN.json'),('h4_codex_historical_frontier_eval_v131','SCOPE.json')]:read(BASE/'offline-deliveries'/package/name,'historical_scope_binding')
    splitpath='restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl'
    SPLIT={x['root_id']:x['split'] for x in read(splitpath,'authoritative_split')}
    histpath='restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/difficulty_manifest.jsonl'
    hist=read(histpath,'historical_all_question_inventory');byid={x['candidate_id']:x for x in hist}
    COUNTS['historical_all_question_exposure']={'input_rows':len(hist),'roots':len({x['root_id'] for x in hist})}
    for x in hist:add(x['candidate'],'historical_all_question_exposure',histpath,wrapper=x)
    for name,role in [('m_clean_policy_training_manifest.jsonl','historical_M_clean_basic_training'),('m_inclusive_policy_training_manifest.jsonl','historical_inclusive_training_exposure')]:
        p='restart-n5-runs/h4_round0_b0_matched_training_v1_live_2/'+name;xs=read(p,role);COUNTS[role]={'input_rows':len(xs)}
        for x in xs:
            h=byid.get(x['candidate_id'])
            if h:add(h['candidate'],role,p,wrapper=x)
            else:GAPS.append({'code':'MISSING_HISTORICAL_QUESTION','candidate_id':x['candidate_id'],'role':role})
    p='restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/candidate_transition_matrix.jsonl';xs=read(p,'historical_train_diagnostic_ids');COUNTS['historical_train_diagnostic']={'input_rows':len(xs)}
    for x in xs:
        h=byid.get(x['candidate_id'])
        if h:add(h['candidate'],'historical_train_diagnostic',p,wrapper=h)
        else:GAPS.append({'code':'MISSING_HISTORICAL_DIAGNOSTIC_QUESTION','candidate_id':x['candidate_id']})
    p=WEEK/'day1/reuse/INVENTORY.json';xs=read(p,'legacy_reuse_inventory')['questions'];COUNTS['legacy_reuse']={'input_rows':len(xs)}
    for x in xs:add(x['original_question_record'],'legacy_reuse',p,wrapper=x,forced_split=x.get('split'))
    sidx=read(WEEK/'day1/reuse/SOURCE_INDEX.json','reuse_source_index')
    for x in sidx['roots']:
        sb=x['source_binding'];source(sb['path'],sb['sha256'],x['root_id'],'reuse_source_only_metadata')
    questions(WEEK/'day1/diagnosis_inputs/PRIVATE_QUESTIONS.json','questions','train_day1_diagnostic')
    questions('offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json','questions','train_v125_foundation')
    rm=read('offline-deliveries/h4_codex_train_pool_v125/data/ROOT_MANIFEST.json','v125_train_root_manifest')
    for x in rm:
        sb=x['source_binding'];source(sb['path'],sb['sha256'],x['root_id'],'train_v125_source_metadata')
    p='offline-deliveries/h4_codex_historical_frontier_eval_v131/QUESTIONS.json';xs=read(p,'historical_frontier_inventory')['candidates'];COUNTS['historical_frontier']={'input_rows':len(xs)}
    pre=read('restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/preflight_report.json','historical_source_inventory')
    for x in xs:add(x,'historical_frontier',p,wrapper=pre['source_root_inventory'].get(x['root_id'],{}),forced_split='train')
    for split in ['dev','test']:questions('offline-deliveries/h4_codex_dev_test_ab_v124/delivery/'+split+'/QUESTIONS_WITH_ANSWERS.json','questions','historical_v124_'+split,split)
    questions('offline-deliveries/h4_codex_dev_eval_v127/EVALUATION_POOL.json','questions','historical_v127_dev','dev')
    questions('offline-deliveries/h4_codex_held_roots_eval_v134/EVALUATION_POOL.json','questions','historical_v134_heldout')
    for name in ['ADMITTED_QUESTIONS.json','ALL_ORIGINAL_QUESTIONS.json']:questions('offline-deliveries/h4_codex_structure_pilot_v140/private/'+name,'questions','historical_v140_'+name[:-5],'dev')
    # Unpublished/failed materials still carry source-fact exposure; never filter by outcome.
    for package in ['h4_codex_dev_test_ab_v124','h4_codex_held_roots_eval_v134']:
        for p in sorted((BASE/'offline-deliveries'/package/'units').glob('*/roots/*/QA_DRAFT.json')):
            d=read(p,'historical_material_draft');src=p.parent/'CUMULATIVE_SOURCE.json'
            if 'root_id' not in d:
                sd=read(src,'material_source_root_binding');d=dict(d,root_id=sd['root_id'])
            fallback=(str(src),d.get('source_sha256'))
            role='historical_material_'+package;COUNTS.setdefault(role,{'input_rows':0,'input_roots':set()});COUNTS[role]['input_roots'].add(d['root_id'])
            for ix,f in enumerate(d.get('facts',[])):
                q={'candidate_id':'MATERIAL-FACT-'+d['root_id']+'-'+str(f.get('fact_id',ix)),'root_id':d['root_id'],'question_type':'source_fact_material','evidence_refs':[f['source_turn_ref']]};add(q,role,p,fallback=fallback);COUNTS[role]['input_rows']+=1
            private=p.parent/'PRIVATE_ASSET_DESIGN.json';pd=read(private,'historical_material_evidence') if private.exists() else {}
            for x in pd.get('ordering',[]):
                q={'candidate_id':x['candidate_id'],'root_id':d['root_id'],'question_type':'ordering','items':{k:'' for k in x.get('fixed_turns',{})},'evidence_refs':list(x.get('fixed_turns',{}).values()),'fixed_turns':x.get('fixed_turns',{})};add(q,role,p,fallback=fallback);COUNTS[role]['input_rows']+=1
            assets=p.parent/'PUBLIC_ASSETS.json'
            if assets.exists():
                ad=read(assets,'historical_material_inventory_no_text_export')
                sidmap={}
                for f in pd.get('facts',[]):
                    for k in ['true_statement_id','false_statement_id']:
                        if f.get(k):sidmap[f[k]]=f.get('source_turn_ref')
                role='historical_public_asset_'+package
                ct=COUNTS.setdefault(role,{'statements':0,'statements_with_private_source_ref':0,'ordering_questions':0,'ordering_with_private_fixed_turns':0,'missing_statement_ids':[],'missing_ordering_ids':[]})
                oid={x['candidate_id']:x for x in pd.get('ordering',[])}
                for st in ad.get('statements',[]):
                    sid=st['statement_id'];ref=sidmap.get(sid);ct['statements']+=1
                    if ref:ct['statements_with_private_source_ref']+=1
                    else:ct['missing_statement_ids'].append(sid)
                    add({'candidate_id':sid,'root_id':d['root_id'],'question_type':'source_fact_material','evidence_refs':[ref] if ref else []},role,assets,fallback=fallback)
                for oq in ad.get('ordering_questions',[]):
                    ct['ordering_questions']+=1
                    if oq['candidate_id'] in oid:ct['ordering_with_private_fixed_turns']+=1
                    else:ct['missing_ordering_ids'].append(oq['candidate_id'])
                ct['coverage_interpretation']='All statements are mapped through both source-judged variants to the same original fact support; no truth flag or statement text is released. Ordering coverage means fixed source turns are indexed, not that source semantics is independently cleared.'

    # Published v124 extra not in delivery; preserve unresolved exact scope.
    known_v124={x['candidate_id'] for x in INTERNAL if x['role'].startswith('historical_v124_')}
    for p in sorted((BASE/'offline-deliveries/h4_codex_dev_test_ab_v124/units').glob('*/roots/*/PUBLIC_QUESTIONS.json')):
        d=read(p,'historical_v124_all_public_ids');qs=d['questions'];COUNTS.setdefault('historical_v124_public_all',{'input_rows':0});COUNTS['historical_v124_public_all']['input_rows']+=len(qs)
        for q in qs:
            if q['candidate_id'] in known_v124:continue
            draft=read(p.parent/'QA_DRAFT.json','historical_extra_public_evidence')
            q=dict(q);q['evidence_refs']=[f['source_turn_ref'] for f in (draft or {}).get('facts',[])]
            add(q,'historical_v124_extra_public',p,fallback=(str(p.parent/'CUMULATIVE_SOURCE.json'),(draft or {}).get('source_sha256')))
    for package,expected in [('day3_production_v1',20),('day3_production_batch2_v1',175)]:
        ps=sorted((WEEK/package/'final_records').glob('*.json'));COUNTS[package]={'input_rows':len(ps),'expected':expected}
        for p in ps:
            q=read(p,'sealed_new_train_private_evidence');add(q,'train_new_'+package,p,forced_split='train')
    # Include the Day2 pilot final, original, revised and rejected author attempts.
    pilot=WEEK/'day2_pilot_v1'
    questions(pilot/'private/PRIVATE_QUESTIONS.json','questions','train_day2_pilot_final')
    for p in sorted((pilot/'drafts').glob('*/*.json')):
        if not any(tag in p.name for tag in ['.DRAFT.','.REVISION_','.CLARIFICATION1.']):continue
        d=read(p,'train_day2_pilot_all_attempts')
        qs=d.get('questions',[])
        role='train_day2_pilot_all_attempts';COUNTS.setdefault(role,{'input_rows':0,'files':0});COUNTS[role]['files']+=1;COUNTS[role]['input_rows']+=len(qs)
        for q in qs:
            sp=pilot/'common/sources'/(q['root_id']+'.json')
            add(q,role,p,fallback=(str(sp),None))
    for p in [pilot/'AUTHORING_ATTEMPTS_FREEZE.json',pilot/'CONTENT_FREEZE.json',pilot/'EXECUTION_FREEZE.json']:read(p,'train_day2_pilot_scope_binding')
    # Retain all diagnosis probe evidence and verify feedback IDs are represented without releasing feedback.
    for package in ['day2_order_probe_v1','day2_frozen_evidence_probe_v2']:
        p=WEEK/package/'run/PUBLIC_QUESTIONS.json';d=read(p,'train_probe_inventory')
        if d:
            COUNTS['train_diagnostic_probe_'+package]={'input_rows':len(d['questions']),'unique_candidate_ids':len({x['candidate_id'] for x in d['questions']})}
            for q in d['questions']:
                backup=next((x for x in read(BASE/'offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json','train_probe_evidence_lookup')['questions'] if x['candidate_id']==q['candidate_id']),None)
                if backup:add(backup,'train_diagnostic_probe_'+package,p)
                else:add(q,'train_diagnostic_probe_'+package,p)
    p=WEEK/'day2_frozen_evidence_probe_v2/run/FROZEN_EVIDENCE.json';d=read(p,'train_frozen_probe_evidence')
    if d:
        COUNTS['train_frozen_probe_evidence']={'input_rows':len(d.get('records',[])),'unique_candidate_ids':len({x['candidate_id'] for x in d.get('records',[])}),'excluded_records_not_used_as_evidence':len(d.get('excluded_records',[]))}
        for x in d.get('records',[]):
            q={'candidate_id':x['frozen_record_id'],'root_id':x['root_id'],'question_type':'ordering','source_binding':x['source_binding'],'evidence':x['evidence'],'item_evidence':{str(i):{'turn_ref':z['turn_ref'],'quote':z['quote']} for i,z in enumerate(x['evidence'])},'items':{str(i):'' for i in range(len(x['evidence']))}};add(q,'train_frozen_probe_evidence',p)
    feedback_paths=[WEEK/'day2_completion_v2/diagnosis/PROVENANCE.json',WEEK/'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json',WEEK/'day3_production_v1/packets/B/ORDERING_FEEDBACK_BOUNDED.json']
    feedback_ids=set();feedback_roots=set()
    def idscan(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k=='candidate_id' and isinstance(v,str):feedback_ids.add(v)
                if k=='root_id' and isinstance(v,str):feedback_roots.add(v)
                if isinstance(v,(dict,list)):idscan(v)
        elif isinstance(x,list):
            for v in x:idscan(v)
    for p in feedback_paths:
        d=read(p,'feedback_id_boundary_only')
        if d:idscan(d)
        if p.name=='PROVENANCE.json' and d:
            for inp in d.get('inputs',[]):
                ip=bind(inp['path'],'diagnostic_feedback_id_scan')
                if ip and ip.suffix in ('.json','.jsonl'):
                    if inp.get('sha256') and BOUND[str(ip)]['sha256']!=inp['sha256']:GAPS.append({'code':'DIAGNOSTIC_PROVENANCE_HASH_MISMATCH','path':str(ip)})
                    idscan(read(ip,'diagnostic_feedback_id_scan'))
    known_ids={x['candidate_id'] for x in INTERNAL};unknown_ids=sorted(feedback_ids-known_ids)
    for cid in unknown_ids:GAPS.append({'code':'UNMAPPED_FEEDBACK_REFERENCED_CANDIDATE','candidate_id':cid})
    finalize(feedback_ids,feedback_roots,protocol)

def finalize(feedback_ids,feedback_roots,protocol):
    # Merge repeated exposure occurrences but never collapse on root identity.
    merged={};prov=collections.defaultdict(list)
    for r,t in zip(ROWS,INTERNAL):
        key=(t['candidate_id'],r['root_id'],r['source_sha256'],r['canonical_fact_family_id'])
        if key not in merged:merged[key]=r
        else:merged[key]['exposure_role']=sorted(set(merged[key]['exposure_role']+r['exposure_role']))
        prov[key].append(t)
    records=[];family_facts={};factfamilies=collections.defaultdict(set)
    for key,r in merged.items():
        r['exclusion_record_id']='EXCL-'+sha(blob(key))[:24]
        errors=sorted({e for t in prov[key] for e in t['mapping_errors']})
        if errors:r['scope']['evidence_mapping_status']='UNRESOLVED_FAMILY_MAPPING'
        r['scope']['uncertainty_reasons']=errors
        for t in prov[key]:t['final_exclusion_record_id']=r['exclusion_record_id']
        fam=r['canonical_fact_family_id'];facts={x['fingerprint_id'] for x in r['semantic_fact_fingerprints']};family_facts.setdefault(fam,set()).update(facts)
        for f in facts:factfamilies[f].add(fam)
        records.append(r)
    for r in records:
        own=r['canonical_fact_family_id'];related=set()
        for f in family_facts[own]:related.update(factfamilies[f])
        r['related_family_ids']=sorted(related-{own})
    records.sort(key=lambda r:r['exclusion_record_id'])
    for v in COUNTS.values():
        if isinstance(v.get('input_roots'),set):v['input_roots']=sorted(v['input_roots'])
    unresolved=[t for t in INTERNAL if t['mapping_errors']]
    for role,ct in COUNTS.items():
        occ=[t for t in INTERNAL if t['role']==role]
        if occ:ct['unique_candidate_ids']=len({t['candidate_id'] for t in occ});ct['roots']=len({t['root_id'] for t in occ})
    coverage={'schema_version':'common_eval_exclusion_coverage_v1','status':'EXTRACTION_COMPLETE_WITH_UNRESOLVED_FAMILY_MAPPING_SEMANTIC_CLEARANCE_PENDING','input_categories':COUNTS,'input_occurrences':len(INTERNAL),'unique_export_records':len(records),'unique_candidate_ids':len({x['candidate_id'] for x in INTERNAL}),'source_versions_verified':len(SOURCES),'roots':len({r['root_id'] for r in records}),'root_source_exposure_is_separate_from_fact_family_exposure':True,'canonical_family_policy':'Evidence-fact-set identity, never root identity. Exact source-clause equality links only observed lexical overlap; semantic equivalence remains unreviewed.','unique_fact_sets':len(family_facts),'evidence_mapped_occurrences':len(INTERNAL)-len(unresolved),'unresolved_occurrences':len(unresolved),'evidence_mapped_unique_records':sum(r['scope']['evidence_mapping_status']!='UNRESOLVED_FAMILY_MAPPING' for r in records),'unresolved_unique_records':sum(r['scope']['evidence_mapping_status']=='UNRESOLVED_FAMILY_MAPPING' for r in records),'unresolved_by_role':dict(collections.Counter(t['role'] for t in unresolved)),'errors_by_kind':dict(collections.Counter(e.split(':',1)[0] for t in unresolved for e in t['mapping_errors'])),'missing_inputs_and_unresolved_references':GAPS,'unresolved_family_mapping':unresolved,'feedback_candidate_id_count':len(feedback_ids),'feedback_root_id_count':len(feedback_roots),'feedback_candidate_ids_missing':[x for x in sorted(feedback_ids) if x not in {t['candidate_id'] for t in INTERNAL}],'export_schema_whitelist':FIELDS,'semantic_near_duplicate_review_status':'PENDING_FOR_ALL_EVAL_SLOTS','no_solver_or_training_calls':True,'no_score_or_answer_conditioned_selection':True,'question_ids_and_internal_bindings_are_not_author_inputs':True}
    # Private, root-local work queue; never give this file to fresh authors.
    rootgaps=[]
    for sb in protocol['source_bindings']:
        root=sb['root_id'];rr=[r for r in records if r['root_id']==root];detail=[]
        for r in rr:
            pp=[t for t in INTERNAL if t.get('final_exclusion_record_id')==r['exclusion_record_id']]
            reasons=sorted({e for t in pp for e in t['mapping_errors']});refs=r['evidence_turn_refs']
            groups={'PARENT':[],'R01':[],'R02':[],'OTHER':[]}
            for ref in refs:groups['PARENT' if ref.startswith('cc-train-episode-') else 'R01' if ref.startswith('R01:') else 'R02' if ref.startswith('R02:') else 'OTHER'].append(ref)
            formal=all(e in ['LEGACY_ALL_OPTION_EVIDENCE_UNVERIFIED','ORDER_ATOMIC_EVENT_PROJECTION_UNRESOLVED'] for e in reasons)
            detail.append({'exclusion_record_id':r['exclusion_record_id'],'candidate_ids':sorted({t['candidate_id'] for t in pp}),'source_sha256':r['source_sha256'],'canonical_fact_family_id':r['canonical_fact_family_id'],'input_paths':sorted({t['input_path'] for t in pp}),'exposure_roles':r['exposure_role'],'mapping_status':r['scope']['evidence_mapping_status'],'mapping_uncertainties':reasons,'gap_classification':'OLD_FORMAT_UNPAIRED_SUPPORT_REQUIRES_SEMANTIC_COMPLETENESS_REVIEW' if reasons and formal else 'NECESSARY_SUPPORT_MISSING_OR_UNVERIFIED' if reasons else 'EXACT_SUPPORT_MAPPED_SEMANTICS_PENDING','known_evidence_turn_refs':refs,'evidence_origin_refs':groups,'parent_only':bool(refs) and len(groups['PARENT'])==len(refs),'R01_present':bool(groups['R01']),'question_item_count':r['scope']['source_item_count'],'semantic_review_required':True})
        rootgaps.append({'root_id':root,'original_split':sb['original_split'],'unique_candidate_source_family_records':len(rr),'exact_support_mapped_records':sum(r['scope']['evidence_mapping_status']!='UNRESOLVED_FAMILY_MAPPING' for r in rr),'unresolved_records':sum(r['scope']['evidence_mapping_status']=='UNRESOLVED_FAMILY_MAPPING' for r in rr),'source_versions':sorted({r['source_sha256'] for r in rr}),'records':detail})
    write('ROOT_LOCAL_CLEARANCE_GAPS.json',{'schema_version':'heldout_root_local_clearance_gaps_v1','not_author_input':True,'status':'SEMANTIC_REVIEW_QUEUE_NOT_CLEARANCE','deduplication_unit':'candidate_id + source version + necessary evidence-set family; repeated roles merged','heldout_roots':len(rootgaps),'global_unique_unresolved_records':coverage['unresolved_unique_records'],'legacy_schema_interpretation':'No modern per-option fields does not invalidate source facts. It leaves completeness unverified until boundary semantic reconstruction; old ordering uses actual n items, never forced to six. Train-only old-format gaps do not by themselves block unrelated heldout families.','roots':rootgaps})
    write('COVERAGE.json',coverage);write('INPUT_BINDINGS.json',{'schema_version':'common_eval_boundary_private_bindings_v1','inputs':[BOUND[k] for k in sorted(BOUND)],'all_question_occurrences':INTERNAL})
    ex=[dict(v,source_paths=sorted(v['source_paths']),exposure_roles=sorted(v['exposure_roles'])) for _,v in sorted(EXPOSURES.items())];write('ROOT_SOURCE_EXPOSURE.json',{'schema_version':'root_source_exposure_metadata_v1','not_author_input':True,'not_fact_family_exclusion':True,'source_versions':ex})
    payload_digest=sha(blob([{k:v for k,v in r.items() if k!='exclusion_manifest_sha256'} for r in records]))
    manifest={'schema_version':'author_exclusion_manifest_v1','status':'EVIDENCE_EXCLUSION_INDEX_FROZEN_SEMANTIC_CLEARANCE_PENDING','fields':FIELDS,'record_count':len(records),'index_payload_sha256_without_manifest_field':payload_digest,'input_bindings':[BOUND[k] for k in sorted(BOUND)],'coverage':{'input_occurrences':len(INTERNAL),'unique_candidate_ids':coverage['unique_candidate_ids'],'unique_export_records':len(records),'evidence_mapped_occurrences':coverage['evidence_mapped_occurrences'],'unresolved_occurrences':len(unresolved),'evidence_mapped_unique_records':coverage['evidence_mapped_unique_records'],'unresolved_unique_records':coverage['unresolved_unique_records']},'algorithm':'Union all explicit source evidence branches without reading Gold; verify source sha and exact cited quote; represent evidence sets and unordered events; no semantic equivalence claim.','family_identity':'provisional unordered exact source-evidence set; canonical necessary-fact identity and semantic normalization remain pending; never original root identity','root_source_metadata_policy':'All31 historically exposed heldout roots remain REGRESSION_ONLY; source exposure alone does not mark every source fact as examined. Any fresh-heldout scope requires separate authorization and review.','review_requirement':'Use source facts for semantic near-duplicate review. Unresolved family mappings and related overlap remain HOLD/PENDING. Root ID/source hash inequality is insufficient.','semantic_pass_claimed':False,'gold_or_model_answers_included':False,'source_facts_included':True}
    write('EXCLUSION_MANIFEST.json',manifest);mh=sha((OUT/'EXCLUSION_MANIFEST.json').read_bytes())
    for r in records:r['exclusion_manifest_sha256']=mh
    (OUT/'EXCLUSION_INDEX.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+chr(10) for r in records))
    forbidden={'gold','answer','correct','correctness','score','prediction','question','options','label','weakness','feedback','rationale','verdict','relation','candidate_id'}
    def audit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in forbidden:raise RuntimeError('forbidden export key '+k)
                audit(v)
        elif isinstance(x,list):
            for v in x:audit(v)
    for r in records:
        assert set(r)==set(FIELDS);audit(r)
    assert all(sha(pathlib.Path(p).read_bytes())==v['sha256'] for p,v in BOUND.items())
    verify={'status':'PASS_EXPORT_WHITELIST_AND_INPUT_HASHES_ONLY','records':len(records),'manifest_sha256':mh,'index_sha256':sha((OUT/'EXCLUSION_INDEX.jsonl').read_bytes()),'forbidden_key_leaks':0,'input_hash_changes':0,'semantic_clearance':'PENDING','source_fact_projection':'SOURCE_EVIDENCE_UNION; no Gold-conditioned support selection','coverage_status':'EXACT_SUPPORT_MAPPED_OR_EXPLICITLY_UNRESOLVED; NOT_COMPLETE_SEMANTIC_FAMILY_CLEARANCE','replay_command':'python '+str(pathlib.Path(__file__).resolve())};write('EXPORT_VALIDATION.json',verify)
    write('OUTPUT_BINDINGS.json',{'schema_version':'common_eval_exclusion_output_bindings_v1','manifest':{'path':str(OUT/'EXCLUSION_MANIFEST.json'),'sha256':mh},'index':{'path':str(OUT/'EXCLUSION_INDEX.jsonl'),'sha256':verify['index_sha256']},'coverage':{'path':str(OUT/'COVERAGE.json'),'sha256':sha((OUT/'COVERAGE.json').read_bytes())},'heldout_root_local_gaps':{'path':str(OUT/'ROOT_LOCAL_CLEARANCE_GAPS.json'),'sha256':sha((OUT/'ROOT_LOCAL_CLEARANCE_GAPS.json').read_bytes())}})
    lines=['# 共同评估排除索引边界报告','','仅 EXCLUSION_INDEX.jsonl 与 EXCLUSION_MANIFEST.json 是作者可读输入。其他文件均为边界审计内部材料，不应转交作者。','','覆盖历史全部题目及后续题池，不按答对答错筛选。根/源暴露另列 ROOT_SOURCE_EXPOSURE.json，不自动排除同根未重复事实。正式排除粒度为实际题目的全部证据并集：所有选项证据和排序事件支持，移除标签、真假判断、Gold及模型表现。','','## 实际结果','',f'- 输入题目/材料出现次数：{len(INTERNAL)}；合并索引记录：{len(records)}；不同候选/材料标识数：{coverage["unique_candidate_ids"]}。',f'- 实际核验源版本：{len(SOURCES)}；涉及根：{coverage["roots"]}；事实集合：{len(family_facts)}。',f'- 证据完整映射出现次数：{coverage["evidence_mapped_occurrences"]}；未解决映射：{len(unresolved)}。',f'- 去重后的证据完整映射记录：{coverage["evidence_mapped_unique_records"]}；未解析记录：{coverage["unresolved_unique_records"]}。', '- 源事实指纹保留可读命题、人物、动作、对象、条件、极性、时间及模态。词法分解仅用于检索，未声称自动语义等价。','- legacy 仅有 evidence_refs 而无逐选项证据，或排序原子事件投影不完整的记录保持 UNRESOLVED_FAMILY_MAPPING；不能据此宣布语义排重 PASS。','','## 限制','','每个新评估槽仍需完成冻结规则规定的5类事实族比较。已知同事实或近义关系需人工记录匹配排除ID及理由。缺失、未解析及歧义不得以根分离或hash不同替代。未运行Solver或训练。原封存材料未修改。','','复现：python '+str(pathlib.Path(__file__).resolve()),''];(OUT/'REPORT_ZH.md').write_text(chr(10).join(lines));print(json.dumps(verify,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in coverage.items() if k not in ['unresolved_family_mapping','missing_inputs_and_unresolved_references']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
