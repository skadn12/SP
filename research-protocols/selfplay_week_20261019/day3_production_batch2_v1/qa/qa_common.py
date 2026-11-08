"""Write reviewers' explicit semantic decisions; never infer semantic PASS.

The reviewer supplies independently reconstructed answers and anchors after
reading full sources and public content. This writer does not read private
author answers. All files are exclusive-create.
"""
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from collections import Counter
import hashlib, json

D=Path(__file__).resolve().parents[1]
P=D.parent/'day3_production_v1'
def resolve(p):
    p=Path(p)
    if p.is_absolute():return p
    choices=[x for x in [D/p,D.parents[2]/p,P/p,Path.cwd()/p] if x.exists()]
    choices=list(dict.fromkeys(x.resolve() for x in choices))
    if len(choices)!=1:raise ValueError(('Unresolved or ambiguous relative artifact path',str(p),choices))
    return choices[0]
def read(p): return json.loads(resolve(p).read_text())
def sha(p): return hashlib.sha256(resolve(p).read_bytes()).hexdigest()
def binding(p): return {'path':str(p),'sha256':sha(p)}
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
def seal(root, reviewer, rows, read_notes, namespace=None):
    source=P/'sources'/f'{root}.json'
    turns=read(source)['turns']
    refs={t['turn_ref']:(i,t) for i,t in enumerate(turns)}
    expanded=[]
    for row in rows:
        row=dict(row)
        pub=D/'candidates'/row['arm']/root/'public'/f"{row['local_name']}.json"
        row.update(candidate_id=read(pub)['candidate_id'],public_binding=binding(pub))
        if 'anchors' in row:
            assert set(row['anchors'])==set('ABCDEF')
            positions={k:refs[v][0] for k,v in row['anchors'].items()}
            assert sorted(positions,key=positions.get)==row['gold']
            assert len(set(positions.values()))==6
            scenes={k:refs[v][1].get('scene_ref') or refs[v][1]['session_ref'] for k,v in row['anchors'].items()}
            row['scene_event_counts']=sorted(Counter(scenes.values()).values(),reverse=True)
            row['pairs']=[{'earlier':a,'later':b,'same_scene':scenes[a]==scenes[b]}
                          for a,b in combinations(row['gold'],2)]
        cited=set(row.get('anchors',{}).values())
        for values in row.get('evidence_by_label',{}).values(): cited.update(values['turn_refs'])
        for occurrence in row.get('mention_review',[]): cited.add(occurrence['turn_ref'])
        row['source_evidence']=[{'turn_ref':r,'global_turn_index':refs[r][0],
                                'speaker':refs[r][1]['speaker'],'scene_id':refs[r][1].get('scene_ref') or refs[r][1]['session_ref'],
                                'quote':refs[r][1]['text']} for r in sorted(cited,key=lambda r:refs[r][0])]
        expanded.append(row)
    out=D/'qa'/'independent'
    if namespace: out=out/namespace
    out=out/root/'RECONSTRUCTION.json'
    obj={'schema_version':'batch2_independent_source_first_v1','reviewer':reviewer,
         'sealed_at_utc':datetime.now(timezone.utc).isoformat(),'root_id':root,
         'source_binding':binding(source),'full_history_turns_read':len(turns),
         'scope':'Full parent+R01+R02 original utterances, all source positions.',
         'private_author_answers_read_before_seal':False,
         'read_notes':read_notes,'rows':expanded,
         'limits':'Process separation on shared filesystem; no technical blinding. Reviewer may have historical feedback exposure; no solver scores used.'}
    dump(out,obj)
    dump(out.with_name('RECONSTRUCTION_SEAL.json'),{'binding':binding(out),'sealed_at_utc':datetime.now(timezone.utc).isoformat()})
    return out

def compare(reconstruction_path, reviewer, annotation_notes=None):
    """Compare AFTER seal. Mechanical agreement cannot promote semantic FAIL."""
    p=Path(reconstruction_path)
    seal_path=p.with_name('RECONSTRUCTION_SEAL.json')
    assert read(seal_path)['binding']['sha256']==sha(p)
    reconstructed=read(p)
    assert reconstructed['reviewer']==reviewer
    source=read(reconstructed['source_binding']['path'])
    turns={t['turn_ref']:t for t in source['turns']}
    rows=[]
    for r in reconstructed['rows']:
        public_path=Path(r['public_binding']['path'])
        assert sha(public_path)==r['public_binding']['sha256']
        priv=public_path.parent.parent/'private'/public_path.name
        author=read(priv)
        errors=[]
        if author['gold']!=r['gold']: errors.append('GOLD_DISAGREEMENT')
        anchors=author.get('selected_turn_ref')
        if anchors is None and 'items' in author:
            if isinstance(author['items'],dict):
                anchors={k:x['selected_turn_ref'] for k,x in author['items'].items()}
            else:
                anchors={x['item_id']:x['selected_turn_ref'] for x in author['items']}
        if 'anchors' in r and anchors!=r['anchors']: errors.append('ANCHOR_DISAGREEMENT')
        # Check every quoted leaf against immutable source. This is byte checking,
        # not a semantic endorsement of every author inclusion/exclusion.
        def sweep(node):
            if isinstance(node,dict):
                if 'turn_ref' in node and 'quote' in node:
                    t=turns.get(node['turn_ref'])
                    if t is None or node['quote'] not in t['text']:
                        errors.append('AUTHOR_EVIDENCE_QUOTE_MISMATCH')
                for value in node.values(): sweep(value)
            elif isinstance(node,list):
                for value in node: sweep(value)
        sweep(author)
        pairs=author.get('all15pairs',author.get('pair_checks',[]))
        if 'anchors' in r:
            expected={(x['earlier'],x['later']) for x in r['pairs']}
            actual={(x['earlier'],x['later']) for x in pairs}
            if len(pairs)!=15 or actual!=expected: errors.append('PAIR_DISAGREEMENT')
        eligible=r['content_result']=='PASS' and r['target_result']=='PASS' and not errors
        rows.append({'candidate_id':r['candidate_id'],'independent_reconstruction':binding(p),
                     'public_binding':r['public_binding'],'private_binding':binding(priv),
                     'gold_agreement':author['gold']==r['gold'],
                     'anchor_agreement':None if 'anchors' not in r else anchors==r['anchors'],
                     'mechanical_comparison_errors':errors,
                     'content_status':'PASS' if eligible else 'HOLD',
                     'target_status':r['target_result'],
                     'independent_semantic_rationale':r['target_rationale'],
                     'shortcut_review':r['shortcut_review'],
                     'annotation_notes':(annotation_notes or {}).get(r['candidate_id'],[])})
    output=p.with_name('COMPARISON.json')
    dump(output,{'reviewer':reviewer,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                 'reconstruction_seal':binding(seal_path),'rows':rows,
                 'deduplication':'PENDING_SEPARATE_REVIEW','solver_calls':0,
                 'note':'Content/target decisions are explicit reviewer judgments; agreement checks do not prove semantic quality or difficulty.'})
    return output
