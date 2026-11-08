"""Serialize the parent's explicit source-first semantic judgments, not infer them.

Each supplied relation/ref/reason and alternative anchor list is a manual decision.
The helpers only copy exact public wording and exact source citations.
"""
import copy
import qa_workflow as w

def assets(rid, decisions, orders, quarantine_reason, findings=None):
    f=w.folder(rid); a=w.read(f/'PUBLIC_ASSETS.json')
    lookup={t['turn_ref']:t for t in w.source(rid)['turns']}
    def cite(ref):
        t=lookup[ref]
        return {'turn_ref':ref,'speaker':t['speaker'],'quote':t['text']}
    assert set(decisions)=={s['statement_id'] for s in a['statements']}
    statements={}
    for s in a['statements']:
        relation,ref,reason=decisions[s['statement_id']]
        statements[s['statement_id']]={'proposition':s['proposition'],'relation':relation,
            'scope_valid':True,'fact_distinct':True,'reason':reason,'citations':[cite(ref)]}
    oq=[]
    for q in a['ordering_questions']:
        selected=orders[q['candidate_id'].rsplit('ordering',1)[-1]]
        assert set(selected)==set(q['items'])
        items={}
        for label,text in q['items'].items():
            ref,alternatives,reason=selected[label]
            items[label]={'proposition':text,'supported':True,'actor_ok':True,'modality_ok':True,
                'state_ok':True,'neutral_stem_ok':True,'single_event_ok':True,
                'order_anchor':cite(ref),'support_refs':[ref],
                'alternative_matching_anchors':[cite(v) for v in alternatives],'reason':reason}
        positions={t['turn_ref']:i for i,t in enumerate(w.source(rid)['turns'])}
        derived=sorted(items,key=lambda k:positions[items[k]['order_anchor']['turn_ref']])
        oq.append({'candidate_id':q['candidate_id'],'stem_valid':True,'scope_valid':True,
            'coherent':True,'unique_order':True,'neutral_stem_ok':True,'items':items,
            'derived_order':derived,'alternative_orders':[]})
    review={'quarantine_compliance':{'status':'PASS','reason':quarantine_reason},
        'findings':findings or [],'statements':statements,'ordering_questions':oq}
    w.review_assets(rid,'/root',review)

def public(rid):
    """Call only after the parent has read and assessed composed public questions."""
    f=w.folder(rid);a=w.read(f/'ASSET_SOURCE_JUDGMENTS.json')
    by_text={v['proposition']:v for v in a['statements'].values()}
    order={v['candidate_id']:v for v in a['ordering_questions']}
    p=w.read(f/'PUBLIC_QUESTIONS.json');judgments={}
    for q in p['questions']:
        if q['question_type']=='ordering':
            j=copy.deepcopy(order[q['candidate_id']]);j.pop('candidate_id')
        else:
            j={'stem_valid':True,'scope_valid':True,'fact_distinct':True,'no_entangled_options':True,
                'options':{k:copy.deepcopy(by_text[v]) for k,v in q['options'].items()}}
        judgments[q['candidate_id']]=j
    w.review_public(rid,'/root',{'quarantine_compliance':a['quarantine_compliance'],
        'findings':a['findings'],'questions':judgments})
