"""Temporary, explicit old-source fixtures only; no production admission claims."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

HOME=Path(__file__).resolve().parents[1]
ENGINE=HOME/'engine'
OLD=HOME.parent/'h4_anchor_records_mixed_batch_v88'
PYTHON='/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
spec=importlib.util.spec_from_file_location('v89_offline_installer',ENGINE/'install_shard.py')
installer=importlib.util.module_from_spec(spec);spec.loader.exec_module(installer)


def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def bind(path):return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(home,n):
    installer.install(home)
    original=read(OLD/'CONFIG.json');cfg=deepcopy(original)
    cfg['version']='v89_full_turn_mixed_shard_v1';cfg['batch_id']='v89_fixture_'+str(n)
    cfg['multiselect_config']['slots']=cfg['multiselect_config']['slots'][:n]
    cfg['ordering_slots']=cfg['ordering_slots'][:n]
    multi=cfg['multiselect_config']
    multi.update(max_http=2*n,max_questions=5*n,max_calls_per_model=90*n,family_count=n,
                 target_distribution={**{str(g):n for g in range(2,6)},'6':0},
                 new_child_target_distribution={str(g):n for g in range(2,6)},
                 control_target_distribution={'6':n},final_review_groups=list('abcd')[:n])
    cfg.update(max_generation_http=3*n,max_candidates=7*n,max_solver_calls_per_model=114*n)
    roots=[slot['root_id'] for slot in multi['slots']]
    pilot=read(HOME/'PILOT_PLAN.json')
    pilot['exclude_repeated_roots']=[]
    pilot['fixture_only']='Explicit historical v88 source fixtures; never used for actual v89 admission.'
    write(home/'PILOT_FIXTURE.json',pilot)
    selected=[]
    for slot in multi['slots']:
        review=OLD/'source_reviews'/('root'+slot['root_id'].rsplit('-',1)[-1]+'_REVIEW.json')
        selected.append({'root_id':slot['root_id'],'split':'train',
                         'source':{'path':slot['source_path'],'sha256':slot['source_sha256']},'review':bind(review)})
    write(home/'ADMISSION_INPUT.json',{'version':'v89_shard_admission_input_v1',
          'shard_id':'synthetic_'+str(n),'split':'train','pilot_plan':bind(home/'PILOT_FIXTURE.json'),
          'source_gate':bind(HOME.parent/'h4_source_consistency_minfix_v77/source_gate.py'),'roots':selected})
    plans=deepcopy(read(Path(original['ordering_anchor_plan']['path'])))
    plans['plans']=[p for p in plans['plans'] if p['root_id'] in roots]
    for plan in plans['plans']:
        slot=next(s for s in multi['slots'] if s['root_id']==plan['root_id'])
        turns={t['turn_ref']:t for t in read(Path(slot['source_path']))['turns']}
        for anchor in plan['anchors']:anchor['quote']=turns[anchor['turn_ref']]['text']
    write(home/'SCENE_PLANS.json',plans);cfg['ordering_anchor_plan']=bind(home/'SCENE_PLANS.json')
    cfg['source_consistency']['selected_roots']=roots
    write(home/'CONFIG_DRAFT.json',cfg)
    return cfg


WORKER_HEADER = """
import json,sys,socket,subprocess
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path.cwd()))
import common,source_admission,source_subset,batch,ordering_plan,ordering_generation
from common import read,write,bind
home=Path.cwd().resolve()
with ExitStack() as guards:
    for target in ('socket.socket.connect','socket.socket.connect_ex','socket.create_connection','socket.getaddrinfo','subprocess.Popen'):
        guards.enter_context(patch(target,side_effect=AssertionError('External operation prohibited')))
    assert common.HERE==home
    source_admission.finalize()
    cfg=read(home/'CONFIG_DRAFT.json')
    cfg['source_consistency']['upstream_complete']=bind(home/'SOURCE_ADMISSION.COMPLETE.json')
    write(home/'CONFIG.json',cfg);write(home/'CONFIG_TEMPLATE.json',cfg)
"""


def worker(home,body):
    result=subprocess.run([PYTHON,'-B','-c',WORKER_HEADER+body],cwd=home,capture_output=True,text=True)
    return result


class EngineTests(unittest.TestCase):
    def test_two_three_four_root_configs_and_real_two_root_prepare(self):
        for n in (2,3,4):
            with self.subTest(n=n),tempfile.TemporaryDirectory(prefix='v89_engine_'+str(n)+'_') as directory:
                home=Path(directory);fixture(home,n)
                body="""
    source_subset.finalize()
    current=batch.config()
    assert len(current['ordering_slots'])==N
    assert (current['max_generation_http'],current['max_candidates'],current['max_solver_calls_per_model'])==(3*N,7*N,114*N)
    try:batch.config(home/'other')
    except ValueError as error:assert 'shard_home_must_equal' in str(error)
    else:raise AssertionError('Different HOME accepted')
    changed=dict(current);changed['max_generation_http']+=1
    try:source_subset.validate(changed)
    except ValueError:pass
    else:raise AssertionError('Budget mutation accepted')
    for slot in current['ordering_slots']:
        source=read(slot['source_path'])
        private=ordering_plan.build(current,slot,source,ordering_generation.candidate_ids(current,slot))
        gold=ordering_plan.gold_questions(source,private)
        public=ordering_plan.writer_plan(source,private)
        assert all(a['quote']==next(t['text'] for t in source['turns'] if t['turn_ref']==a['turn_ref']) for q in private['questions'] for a in q['anchors'].values())
        assert all(len(q['gold'])==6 for q in gold)
        assert '\"I1\"' not in json.dumps(public) and '\"gold\"' not in json.dumps(public)
    if N==2:
        result=batch.prepare()
        assert result['status']=='PREPARED'
        assert not list(home.rglob('HTTP_ATTEMPT.json')) and not list(home.rglob('GENERATION.STARTED.json'))
        for slot in current['ordering_slots']:
            run=home/'ordering'/slot['slot_id']
            original_open=Path.open
            def barrier(path,*args,**kwargs):
                assert path.name not in {'ANCHOR_PLAN.json','GOLD_PLAN.json'},str(path)
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',barrier):payload,proof=ordering_generation.payload_for(run)
            assert payload==read(run/'REQUEST.json')
            assert proof['writer_plan']==bind(run/'WRITER_PLAN.json')
        batch.verify_plan()
    assert not list(home.rglob('__pycache__'))
    print('PASS dynamic roots, full-turn plans, fixed labels/Gold, '+('actual both-lane prepare' if N==2 else 'config validation'))
"""
                body=re.sub(r'\bN\b',str(n),body)
                result=worker(home,body)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_single_source_admission_waits_for_second_family(self):
        with tempfile.TemporaryDirectory(prefix='v89_single_source_') as directory:
            home=Path(directory);fixture(home,1)
            result=worker(home,"""
    assert source_admission.verify()['production_readiness']=='SOURCE_READY_AWAITING_SECOND_FAMILY'
    try:source_subset.finalize()
    except ValueError as error:assert 'SOURCE_READY_AWAITING_SECOND_FAMILY' in str(error)
    else:raise AssertionError('Singleton production enabled')
    assert not (home/'SHARD_INPUTS.COMPLETE.json').exists()
    assert not (home/'BATCH_PREPARATION.STARTED.json').exists()
""")
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)

    def test_held_review_rejected_without_repair(self):
        with tempfile.TemporaryDirectory(prefix='v89_hold_') as directory:
            home=Path(directory);fixture(home,2)
            admission=read(home/'ADMISSION_INPUT.json')
            row=admission['roots'][0];review=read(Path(row['review']['path']))
            review['decision']='HOLD';write(home/'INVALID_REVIEW.json',review);row['review']=bind(home/'INVALID_REVIEW.json')
            write(home/'ADMISSION_INPUT.json',admission)
            result=worker(home,'')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('root_decision_disagrees_with_findings',result.stderr)
            self.assertFalse((home/'SOURCE_ADMISSION.COMPLETE.json').exists())

    def test_shared_engine_and_reinstallation_rejected(self):
        with tempfile.TemporaryDirectory(prefix='v89_install_') as directory:
            home=Path(directory);installer.install(home)
            with self.assertRaisesRegex(ValueError,'refuse_reinstall'):installer.install(home)
        with self.assertRaisesRegex(ValueError,'separate_shard'):installer.install(ENGINE)

    def test_full_turn_range_keeps_semantic_actor_and_cross_turn_guards(self):
        with tempfile.TemporaryDirectory(prefix='v89_full_turn_semantics_') as directory:
            home=Path(directory);fixture(home,2)
            result=worker(home,"""
    from copy import deepcopy
    from types import SimpleNamespace
    import ordering_review
    labels=['I3','I1','I6','I2','I5','I4']
    source={'root_id':'synthetic-source','turns':[
        {'turn_ref':f'CANDIDATE:A:S01:T{i:03d}','speaker':'Actor',
         'text':f'Actor recorded event {i}. Actor discussed topic {i}.'} for i in range(1,7)]}
    plan={'candidate_id':'synthetic-question','root_id':'synthetic-source','scene_id':'CANDIDATE:A:S01',
          'question':ordering_plan.STEM.format(scene_id='CANDIDATE:A:S01'),
          'anchors':{label:{'turn_ref':source['turns'][i]['turn_ref'],'speaker':'Actor','quote':source['turns'][i]['text']} for i,label in enumerate(labels)}}
    record={'root_id':'synthetic-source','source_sha256':'fixture','questions':[plan]}
    answer=ordering_plan.gold_questions(source,record)[0]
    assert all(p['quote_start']==0 and p['quote_end']==len(source['turns'][p['turn_index']]['text']) for p in answer['anchor_positions'].values())
    bad=deepcopy(record);bad['questions'][0]['anchors']['I3']['quote']='Actor recorded event 1.'
    try:ordering_plan.gold_questions(source,bad)
    except ValueError as error:assert 'complete_source_turn' in str(error)
    else:raise AssertionError('Short excerpt accepted')
    run=home/'synthetic_fixed';run.mkdir();source_path=run/'SOURCE.json';write(source_path,source)
    record['source_sha256']=bind(source_path)['sha256'];write(run/'ANCHOR_PLAN.json',record)
    gold_record={'version':'v89_program_gold_plan_v1','created_at_utc':common.now(),'source':bind(source_path),
                 'anchor_plan':bind(run/'ANCHOR_PLAN.json'),'gold_origin':ordering_plan.GOLD_ORIGIN,
                 'semantic_certification':False,'gold_supplied_to_writer':False,'questions':[answer]}
    write(run/'GOLD_PLAN.json',gold_record)
    write(run/'MANIFEST.json',{'root_id':'synthetic-source','candidate_ids':['synthetic-question'],
          'source':bind(source_path),'anchor_plan':bind(run/'ANCHOR_PLAN.json'),'gold_plan':bind(run/'GOLD_PLAN.json'),
          'gold_origin':ordering_plan.GOLD_ORIGIN,'prepared_at_utc':common.now()})
    for mode in ('within_same_turn','wrong_other_turn','wrong_actor'):
        place=home/mode;folder=place/'ordering';folder.mkdir(parents=True)
        raw={'candidate_id':'synthetic-question','root_id':'synthetic-source','records':[
            {'anchor_id':t['turn_ref'],'predicate':f'discussed topic {i+1}.'} for i,t in enumerate(source['turns'])]}
        if mode=='wrong_other_turn':
            raw['records'][0]['predicate']='discussed topic 2.'
            raw['records'][1]['predicate']='discussed topic 1.'
        candidate=ordering_plan.project(raw,plan,answer)
        original=deepcopy(raw);raw['records'].reverse()
        assert ordering_plan.project(raw,plan,answer)==candidate
        for key in ('gold','items','subject'):
            extra={**original,key:'invented'}
            try:ordering_plan.project(extra,plan,answer)
            except ValueError:pass
            else:raise AssertionError('Writer extra field accepted')
        errors,q=common.load_ordering_gate().validate_generated_ordering(candidate,plan['candidate_id'],plan['root_id'],source,source_path)
        assert not errors,errors
        q.update(gold_origin=ordering_plan.GOLD_ORIGIN,program_gold_plan=bind(run/'GOLD_PLAN.json'))
        public=ordering_generation.public(q);judgment=deepcopy(public)
        judgment.update(complete_source_turns_read=6,coherent=True,unique_order=True,neutral_stem_ok=True,alternative_orders=[])
        judgment['items']={}
        for label,anchor in plan['anchors'].items():
            quote=q['items'][label];ref=anchor['turn_ref']
            if mode=='wrong_other_turn' and label=='I3':ref=source['turns'][1]['turn_ref']
            if mode=='wrong_other_turn' and label=='I1':ref=source['turns'][0]['turn_ref']
            judgment['items'][label]={'proposition':quote,**{key:True for key in ordering_review.ITEM_BOOLEANS},
                'order_anchor':{'turn_ref':ref,'quote':quote},'support_refs':[ref],'alternative_matching_anchors':[]}
        if mode=='wrong_actor':judgment['items']['I3']['actor_ok']=False
        checked=ordering_review._review_row(judgment,public,source,common.load_ordering_gate())
        judgment_path=folder/'SOURCE_JUDGMENTS.json';write(judgment_path,{'fixture_only':True,'questions':[judgment]})
        write(folder/'POOL.json',{'questions':[q]})
        pool_bind=bind(folder/'POOL.json');jbind=bind(judgment_path)
        context={'created_at_utc':common.now(),'generation_files':{pool_bind['path']:pool_bind},
                 'questions':{q['candidate_id']:public},'sources':{q['root_id']:source},'checks':[checked],
                 'bindings':[jbind],'judgment_binding':jbind,
                 'judgment':{'reviewer':'synthetic-fixture','public_questions':{'sha256':'0'*64}}}
        result=ordering_review._private_review(place,context)
        if mode=='within_same_turn':assert result['checks'][0]['status']=='USABLE',result
        else:assert result['checks'][0]['status']=='EXCLUDED',result
    print('PASS complete-turn span permits same-turn evidence while wrong turn and actor stay excluded')
""")
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
