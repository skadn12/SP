"""Exact historical draft imports for two explicitly frozen retry cases.

This adapter never authors dialogue, questions, or judgments. Historical authors
remain historical authors. Each exception is restricted to a frozen root/stage
and exact source artifact hashes; all subsequent judgments are fresh and assigned.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

B = Path(__file__).resolve().parents[1]


def need(ok, why):
    if not ok: raise ValueError(why)

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(Path(p).resolve()), 'sha256': sha(p)}
def verify(r):
    p = Path(r['path']).resolve()
    need(bind(p) == r, 'FIXED_IMPORT_INPUT_CHANGED:' + str(p))
    return p

def save(p, d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')

def copy(p,q):
    q=Path(q);q.parent.mkdir(parents=True,exist_ok=True)
    with q.open('xb') as f:f.write(Path(p).read_bytes())

def policy(batch):
    batch=Path(batch).resolve()
    fp=batch/'MASTER_FREEZE.json';freeze=read(fp)
    pp=batch/'evidence/FIXED_IMPORT_POLICY.json'
    for p in (pp,Path(__file__)):
        rows=[r for r in freeze['files'] if r['path']==str(p.resolve())]
        need(len(rows)==1 and rows[0]==bind(p),'FIXED_IMPORT_POLICY_OR_RUNTIME_NOT_FROZEN')
    data=read(pp)
    need(data['allowed_roots']==['cc-train-episode-146700','cc-train-episode-254852'],'FIXED_IMPORT_COHORT')
    return data,pp

def qa_expected(batch,unit,root,check_target=True):
    data,pp=policy(batch)
    need(root=='cc-train-episode-146700','FIXED_QA_IMPORT_ROOT')
    spec=data['qa'];need(spec['root_id']==root,'FIXED_QA_POLICY_ROOT')
    originals={n:verify(r) for n,r in spec['artifacts'].items()}
    folder=Path(unit)/'roots/root146700'
    need((folder/'CUMULATIVE_SOURCE.json').read_bytes()==originals['CUMULATIVE_SOURCE.json'].read_bytes(),'FIXED_QA_SOURCE_CHANGED')
    receipt=read(originals['QA_AUTHOR_RECEIPT.json'])
    adapted={**receipt,'ordering_anchor_plan_binding':{'path':str((folder/'ORDERING_ANCHOR_PLAN.json').resolve()),'sha256':sha(originals['ORDERING_ANCHOR_PLAN.json'])},
             'historical_import_only':True,'new_qa_authoring_attempt':False,
             'original_qa_receipt_binding':bind(originals['QA_AUTHOR_RECEIPT.json'])}
    if check_target:
        rp=folder/'HISTORICAL_QA_IMPORT.json';carry=read(rp)
        need(carry['policy_binding']==bind(pp) and carry['root_id']==root,'FIXED_QA_IMPORT_POLICY_CHANGED')
        need(carry['historical_author']==receipt['author'],'FIXED_QA_HISTORICAL_AUTHOR')
        for name in ('ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json'):
            need((folder/name).read_bytes()==originals[name].read_bytes(),'FIXED_QA_CONTENT_CHANGED:'+name)
            need(carry['artifacts'][name]=={'original_binding':bind(originals[name]),'imported_binding':bind(folder/name)},'FIXED_QA_ARTIFACT_BINDING')
        need(read(folder/'QA_AUTHOR_RECEIPT.json')==adapted,'FIXED_QA_RECEIPT_ADAPTATION_CHANGED')
        need(carry['adapted_qa_receipt_binding']==bind(folder/'QA_AUTHOR_RECEIPT.json'),'FIXED_QA_RECEIPT_CHANGED')
    return spec,originals,adapted

def scene_expected(batch,unit,root,scene,check_target=True):
    data,pp=policy(batch)
    need(root=='cc-train-episode-254852' and scene=='R02:A:S02','FIXED_SCENE_IMPORT_SCOPE')
    spec=data['scene'];need(spec['root_id']==root and spec['scene_ref']==scene,'FIXED_SCENE_POLICY_SCOPE')
    original=verify(spec['draft_binding']);draft=read(original)
    previous=Path(unit)/'roots/root254852/scenes/A_S01/CUMULATIVE_SOURCE.json'
    need(sha(previous)==draft['input_source_sha256'],'FIXED_SCENE_PREDECESSOR_CHANGED')
    folder=Path(unit)/'roots/root254852/scenes/A_S02'
    if check_target:
        carry=read(folder/'HISTORICAL_SCENE_IMPORT.json')
        need(carry['policy_binding']==bind(pp) and carry['root_id']==root and carry['scene_ref']==scene,'FIXED_SCENE_IMPORT_POLICY_CHANGED')
        need(carry['original_draft_binding']==bind(original) and carry['imported_draft_binding']==bind(folder/'EXPANSION_DRAFT.json'),'FIXED_SCENE_DRAFT_BINDING')
        need((folder/'EXPANSION_DRAFT.json').read_bytes()==original.read_bytes(),'FIXED_SCENE_DRAFT_CHANGED')
        need(carry['historical_author']==draft['author'] and carry['new_authoring_attempt'] is False,'FIXED_SCENE_AUTHOR_CHANGED')
    return spec,original,draft

def expected_author(batch,unit,root,phase,scene=None):
    role=next(r for r in read(Path(batch)/'ROLE_ASSIGNMENTS.json')['roots'] if r['root_id']==root)
    if root=='cc-train-episode-146700' and phase in ('qa','asset','public'):
        return qa_expected(batch,unit,root)[2]['author']
    if root=='cc-train-episode-254852' and phase=='scene' and scene=='R02:A:S02':
        return scene_expected(batch,unit,root,scene)[2]['author']
    return role['author']

def import_qa(batch,root,executor):
    need(executor=='/root','PARENT_IMPORTER_REQUIRED')
    unit=Path(batch)/'units/root146700';folder=unit/'roots/root146700'
    import scene_controller_v120 as controller
    state=controller.inspect_root(unit,controller.load_config(unit),root,record_reviews=False)
    need(state['full_expansion_complete'],'FIXED_QA_REQUIRES_SIX_READY_SCENES')
    # Root summary is an exact B3 copy, as in the ordinary author qa-read.
    for name in ('CUMULATIVE_SOURCE.json','SOURCE_REVIEW.json'):
        source=folder/'scenes/B_S03'/name
        if (folder/name).exists():need((folder/name).read_bytes()==source.read_bytes(),'FIXED_ROOT_SUMMARY_CHANGED')
        else:copy(source,folder/name)
    spec,originals,adapted=qa_expected(batch,unit,root,False)
    for name in ('ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json'):copy(originals[name],folder/name)
    # Recompute after anchor copy; only path binding changes in carried receipt.
    spec,originals,adapted=qa_expected(batch,unit,root,False)
    save(folder/'QA_AUTHOR_RECEIPT.json',adapted)
    _,pp=policy(batch)
    save(folder/'HISTORICAL_QA_IMPORT.json',{'root_id':root,'actual_importer':executor,
        'imported_at_utc':datetime.now(timezone.utc).isoformat(),'policy_binding':bind(pp),
        'historical_author':adapted['author'],'new_qa_authoring_attempt':False,
        'semantic_content_changes':0,'candidate_namespace':spec['candidate_namespace'],
        'artifacts':{n:{'original_binding':bind(originals[n]),'imported_binding':bind(folder/n)} for n in ('ORDERING_ANCHOR_PLAN.json','QA_DRAFT.json')},
        'adapted_qa_receipt_binding':bind(folder/'QA_AUTHOR_RECEIPT.json'),
        'receipt_adaptation':'Only ordering_anchor_plan_binding relocated; original identity/times retained and explicit carry metadata added.'})
    qa_expected(batch,unit,root)
    print(json.dumps({'root_id':root,'status':'FIXED_HISTORICAL_QA_IMPORTED','new_qa_semantic_drafts':0}))

def import_scene(batch,root,scene,executor):
    need(executor=='/root','PARENT_IMPORTER_REQUIRED')
    unit=Path(batch)/'units/root254852'
    import scene_controller_v120 as controller
    state=controller.inspect_root(unit,controller.load_config(unit),root,record_reviews=False)
    need(state['scenes'][0]['status']=='READY' and state['scenes'][1]['status']=='NOT_STARTED','FIXED_SCENE_PREDECESSOR_NOT_READY')
    spec,original,draft=scene_expected(batch,unit,root,scene,False)
    folder=unit/'roots/root254852/scenes/A_S02';copy(original,folder/'EXPANSION_DRAFT.json')
    _,pp=policy(batch)
    save(folder/'HISTORICAL_SCENE_IMPORT.json',{'root_id':root,'scene_ref':scene,'actual_importer':executor,
        'imported_at_utc':datetime.now(timezone.utc).isoformat(),'policy_binding':bind(pp),
        'historical_author':draft['author'],'new_authoring_attempt':False,'semantic_content_changes':0,
        'original_draft_binding':bind(original),'imported_draft_binding':bind(folder/'EXPANSION_DRAFT.json')})
    result=controller.materialize_scene(unit,root,scene)
    need(result['status']=='STRUCTURAL_PASS_ONLY','FIXED_SCENE_MATERIALIZATION_FAILED')
    print(json.dumps({'root_id':root,'scene_ref':scene,'status':'FIXED_HISTORICAL_DRAFT_AWAITING_FRESH_REVIEW','new_scene_semantic_drafts':0}))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['import-qa','import-scene'])
    parser.add_argument('--batch',type=Path,default=B);parser.add_argument('--root',required=True)
    parser.add_argument('--executor',required=True);parser.add_argument('--scene')
    a=parser.parse_args()
    if a.action=='import-qa':import_qa(a.batch.resolve(),a.root,a.executor)
    else:import_scene(a.batch.resolve(),a.root,a.scene,a.executor)

if __name__=='__main__':main()
