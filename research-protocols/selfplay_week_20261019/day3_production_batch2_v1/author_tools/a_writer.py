import json, hashlib, pathlib, datetime, itertools

BASE=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019')
D=BASE/'day3_production_batch2_v1'
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,d):
    p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise RuntimeError('Refuse overwrite '+str(p))
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

class Root:
    def __init__(self,r):
        self.r=r; self.p=D/'packets/A'/f'{r}.json';self.packet=json.loads(self.p.read_text())
        self.s=pathlib.Path(self.packet['source_binding']['path']);self.source=json.loads(self.s.read_text())
        self.review=pathlib.Path(self.packet['source_review_path']);self.rv=json.loads(self.review.read_text())
        assert self.rv['status'] in ['PASS','PASS_FOR_SCOPED_AUTHORING'],self.rv
        assert sha(self.s)==self.packet['source_binding']['sha256']
        self.out=D/'candidates/A'/r; self.turns={t['turn_ref']:t for t in self.source['turns']}
        self.pos={t['turn_ref']:i for i,t in enumerate(self.source['turns'])}
    def freeze(self,plans,notes):
        lib=pathlib.Path(self.packet['public_library']['path'])
        reads=[(self.p,'entire A packet'),(D/'RECORD_FORMAT.json','entire file'),(lib,'all public semantic design sections; schema export deferred'),(self.s,'entire source: all parent, R01 and R02 turns, read semantically in source order'),(self.review,'entire independent source review')]
        manifest={'root_id':self.r,'arm':'A','author_role':'fresh no-feedback author','created_at_utc':now(),'read_records':[{'path':str(p),'sha256':sha(p),'scope':scope,'byte_range':[0,p.stat().st_size]} for p,scope in reads], 'full_source_turn_range':[self.source['turns'][0]['turn_ref'],self.source['turns'][-1]['turn_ref']],'full_source_turn_count':len(self.source['turns']), 'read_boundary':'Only named A packet, record format, public design, bound full source and bound source review. No B packets, cards, feedback, scores, status, other candidates or evaluation sources.', 'isolation':'Process boundary on shared filesystem, not a technical sandbox. R02:A/B are source scenes.', 'model_revision':None,'actual_model_revision_available':False}
        dump(self.out/'READ_MANIFEST.json',manifest)
        plan={'root_id':self.r,'arm':'A','created_at_utc':now(),'source_binding':self.packet['source_binding'],'source_review_path':str(self.review),'source_review_sha256':sha(self.review),'full_source_semantic_read_complete':True,'scope':'entire supplied full source; target R02 reports; first full-source semantic report for ordering','slots':[dict(s,planned_realization=plans.get(f"{s['question_type']}-{s['local_slot']:02}",'SHORTFALL')) for s in self.packet['slots']],'semantic_read_notes':notes,'gold_and_scene_profiles_unchanged':True,'initial_draft_limit':1,'clarification_limit':1,'no_solver_or_feedback':True}
        dump(self.out/'PLAN.json',plan)
    def refs(self,refs):
        return [{'turn_ref':r,'quote':self.turns[r]['text']} for r in refs]
    def common(self,typ,n):
        s=next(s for s in self.packet['slots'] if s['question_type']==typ and s['local_slot']==n)
        return {'candidate_id':s['candidate_slot_id'],'root_id':self.r,'question_type':typ,'slot_id':s['slot_id'],'design_id':s['design_id'],'target_gold_count_inherited':s['target_gold_count'],'created_at_utc':now(),'read_manifest_sha256':sha(self.out/'READ_MANIFEST.json'),'plan_sha256':sha(self.out/'PLAN.json'),'source_binding':self.packet['source_binding'],'source_review_path':str(self.review),'source_review_sha256':sha(self.review),'budget':{'initial_used':1,'clarification_used':0,'requested_output_tokens':3072,'actual_cap':None,'actual_usage':None,'actual_cap_support':'UNKNOWN','provider_request_id':None,'mechanical_batch_write_is_not_api_call':True},'feedback_binding_location':None,'qa_status':'PENDING_INDEPENDENT_RECONSTRUCTION','author_final_pass':False}
    def mc(self,n,question,rows,realization):
        c=self.common('multi_select',n); assert len(rows)==6
        labels='ABCDEF'; gold=[l for l,row in zip(labels,rows) if row[1]]
        assert len(gold)==c['target_gold_count_inherited']
        pub={k:c[k] for k in ['candidate_id','root_id','question_type']};pub.update(question=question,options_or_items={l:r[0] for l,r in zip(labels,rows)})
        private=dict(c,gold=gold,evidence_by_label={l:{'verdict':'SUPPORTED' if row[1] else 'REFUTED','rationale':row[3],'evidence':self.refs(row[2])} for l,row in zip(labels,rows)},design_realization=realization)
        self.write('multi_select',n,pub,private)
    def order(self,n,events,realization='Six atomic reporting acts, ordered by first full-source semantic report; source scenes retained.'):
        # Input events are manually selected in chronological source order, with complete manually judged mention sets.
        assert len(events)==6
        c=self.common('ordering',n);labels='ABCDEF'; perm=[3,0,5,1,4,2]
        items={};texts={};gold=[];chron=[]
        for label,j in zip(labels,perm):
            e=events[j];txt,ref,key,mentions=e; assert ref in self.turns
            allm=[{'turn_ref':m[0],'quote':self.turns[m[0]]['text'],'included_for_event_key':m[1],'rationale':m[2]} for m in mentions]
            assert any(m['turn_ref']==ref and m['included_for_event_key'] for m in allm)
            assert self.pos[ref]==min(self.pos[m['turn_ref']] for m in allm if m['included_for_event_key'])
            texts[label]=txt;items[label]={'event_semantic_key':key,'selected_turn_ref':ref,'scene_ref':self.turns[ref]['scene_ref'] or self.turns[ref]['session_ref'],'mention_policy':'FIRST_IN_DECLARED_SCOPE','all_mentions':allm,'all_reasonable_mentions_searched_scope':'entire parent+R01+R02 source; semantic full read','repeat_status':'REPEATED_SAME_FACT' if sum(m['included_for_event_key'] for m in allm)>1 else 'SINGLE'};chron.append((self.pos[ref],label))
        assert len(set(x[0] for x in chron))==6
        gold=[l for _,l in sorted(chron)]
        counts={}
        for v in items.values():counts[v['scene_ref']]=counts.get(v['scene_ref'],0)+1
        profile={1:[3,2,1],2:[2,2,2],3:[4,2]}[n];assert sorted(counts.values(),reverse=True)==profile,counts
        pairs=[{'earlier':a,'later':b,'within_scene':items[a]['scene_ref']==items[b]['scene_ref'],'earlier_turn_ref':items[a]['selected_turn_ref'],'later_turn_ref':items[b]['selected_turn_ref'],'rationale':'Earlier selected report occurs first in the full source turn sequence.'} for a,b in itertools.combinations(gold,2)]
        pub={k:c[k] for k in ['candidate_id','root_id','question_type']};pub.update(question='Put these reports in the order in which they first appear in the entire supplied conversation history. Use the first semantic report of each event, not when the underlying activity happened.',options_or_items=texts)
        priv=dict(c,gold=gold,items=items,evidence_by_label={l:{'verdict':'SUPPORTED','rationale':'Atomic report grounded at the earliest included semantic mention.','evidence':self.refs([v['selected_turn_ref']])} for l,v in items.items()},pair_checks=pairs,scene_event_counts=profile,scene_counts_by_id=counts,design_realization={'description':realization,'gold_definition':'SOURCE_APPEARANCE_ORDER','all_mentions_review':'author semantic inventory, independent QA pending','same_six_facts_gold_shift':False,'scene_only_partial_order_has_multiple_valid_linearizations':True})
        self.write('ordering',n,pub,priv)
    def write(self,typ,n,pub,priv):
        name=f'{typ}-{n:02}.json'
        dump(self.out/'public'/name,pub);dump(self.out/'private'/name,priv)
        dump(self.out/'initial_drafts'/name,{'public':pub,'private':priv,'immutable_initial_draft':True})
    def shortfall(self,typ,n,reason,refs=[]):
        c=self.common(typ,n);c['budget']['initial_used']=0
        dump(self.out/'shortfall'/f'{typ}-{n:02}.json',dict(c,disposition='SHORTFALL',reason_code='SHORTFALL_SOURCE_RELATION',reason=reason,evidence=self.refs(refs),public_written=False))
    def finish(self):
        public=sorted(p.name for p in (self.out/'public').glob('*.json'))
        shortfall=sorted(p.name for p in (self.out/'shortfall').glob('*.json'))
        assert len(public)+len(shortfall)==6
        dump(self.out/'COMPLETED.json',{'root_id':self.r,'arm':'A','created_at_utc':now(),'public_files':public,'shortfall_files':shortfall,'real_initial_questions':len(public),'shortfall_count':len(shortfall),'qa_status':'PENDING_INDEPENDENT_RECONSTRUCTION','initials_immutable':True})
        print(self.r,len(public),'real questions;',len(shortfall),'shortfalls')

def ev(text,ref,key,extra=[]):
    return (text,ref,key,[(ref,True,'Earliest report of this semantic event in full history.')]+extra)
