import hashlib, itertools, json, random, tempfile
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import importlib.util

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('compose', HERE/'compose.py'); c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

def sh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wr(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def make_slot(report, sid, root, k, fact_count=6):
    run=report/'assets'/sid; run.mkdir(parents=True)
    turns=[{'turn_ref':f'{root}:T{i+1}','speaker':'S','text':f'Statement {i+1} is true.'} for i in range(fact_count)]
    source={'root_id':root,'turns':turns}
    wr(run/'SOURCE.json',source)
    facts=[]
    for i in range(fact_count):
        facts.append({'fact_id':f'F{i+1}','scope':{'actor':'S','event':f'event{i+1}','time':'today','condition':'always','action_state':'stated'},'anchor':{'turn_ref':turns[i]['turn_ref'],'speaker':'S','quote':turns[i]['text']},'supported':{'proposition':f'Statement {i+1} is true.','relation':'supports','rationale':'source says it.'},'contradiction_unavailable_reason':'No explicit alternative supplied in this synthetic fixture.' if i < k else None,'contradicted':None if i < k else {'proposition':f'Statement {i+1} is false.','relation':'contradicts','rationale':'explicit alternative.'}})
    assets={'root_id':root,'slot_id':sid,'public_stem':'Which statements are supported?','facts':facts}; wr(run/'ASSETS.json',assets)
    man={'source_sha256':sh(run/'SOURCE.json')}
    pub={'root_id':root,'slot_id':sid,'source_path':str(run/'SOURCE.json'),'source_sha256':sh(run/'SOURCE.json'),'assets_sha256':sh(run/'ASSETS.json'),'public_stem':assets['public_stem'],'versions':[], 'contains_relations':False,'contains_solver_scores':False}
    mapping={'source_sha256':sh(run/'SOURCE.json'),'assets_sha256':sh(run/'ASSETS.json'),'public_assets_sha256':'', 'versions':[]}
    judgments={'source_sha256':sh(run/'SOURCE.json'),'public_assets_sha256':'','gold_read_before_save':False,'solver_scores_used':False,'stem_valid':True,'versions':[]}
    review={'source_sha256':sh(run/'SOURCE.json'),'assets_sha256':sh(run/'ASSETS.json'),'public_assets_sha256':'','source_judgments_sha256':'','solver_scores_used':False,'source_first':True,'stem_valid':True,'versions':[]}
    for i,f in enumerate(facts):
        for variant,rel in [('supported','supports')]+([('contradicted','contradicts')] if f['contradicted'] else []):
            vid=f'v{i+1}{variant[0]}'
            val=f[variant]; pub['versions'].append({'version_id':vid,'fact_id':f['fact_id'],'proposition':val['proposition']}); mapping['versions'].append({'version_id':vid,'fact_id':f['fact_id'],'internal_version_id':f['fact_id']+':'+('s' if variant=='supported' else 'c'),'variant':variant,'relation':rel}); judgments['versions'].append({'version_id':vid,'relation':rel,'scope_valid':True,'fact_distinct':True,'reason':'ok','citations':[f['anchor']['turn_ref']]}); review['versions'].append({'version_id':vid,'usable':True,'relation':rel,'reason':'ok'})
    wr(run/'PUBLIC_ASSETS.json',pub); mapping['public_assets_sha256']=sh(run/'PUBLIC_ASSETS.json'); wr(run/'VERSION_MAP.json',mapping); judgments['public_assets_sha256']=sh(run/'PUBLIC_ASSETS.json'); wr(run/'SOURCE_JUDGMENTS.json',judgments); review['public_assets_sha256']=sh(run/'PUBLIC_ASSETS.json'); review['source_judgments_sha256']=sh(run/'SOURCE_JUDGMENTS.json'); wr(run/'ASSET_REVIEW.json',review)
    config_path=report/'CONFIG.json'
    manifest={'version':'v69_frozen_evidence_assets_v1','slot':next(s for s in json.loads(config_path.read_text())['slots'] if s['slot_id']==sid),'config':{'path':str(config_path),'sha256':sh(config_path)},'root_id':root,'slot_id':sid,'source_path':str(run/'SOURCE.json'),'source_sha256':sh(run/'SOURCE.json'),'training_allowed':False,'files':[]}
    wr(run/'MANIFEST.json',manifest)
    complete={'root_id':root,'slot_id':sid,'status':'ASSETS_READY','fact_count':fact_count,'semantic_retries':0,'training_allowed':False,'files':[]}; wr(run/'GENERATION.COMPLETE.json',complete)

def test_compose_and_verify():
    with tempfile.TemporaryDirectory() as td:
        report=Path(td); cfg={'version':'x','baseline_pool':str(report/'BASE.json'),'baseline_result_bindings':str(report/'BIND.json'),'slots':[],'max_questions':10,'training_allowed':False}
        for n,(sid,root,k) in enumerate([('s1','r1',2),('s2','r2',6)]): cfg['slots'].append({'slot_id':sid,'root_id':root,'history_path':'h','prior_pool_paths':[],'focus':{'scene_refs':[],'description':'topic'},'initial_target_count':k})
        wr(report/'CONFIG.json',cfg); wr(report/'BASE.json',{'questions':[]}); wr(report/'BIND.json',{})
        for sid,root,k in [('s1','r1',2),('s2','r2',6)]: make_slot(report,sid,root,k)
        out=c.compose(report); assert out['candidate_count']==2
        assert sorted(len(q['gold']) for q in out['questions'])==[2,6]
        assert c.verify_assembly(report)['questions']==out['questions']
        # Public projection never exposes gold or evidence.
        pub=c.read(report/'PUBLIC_QUESTIONS.json'); assert all('gold' not in q and 'option_evidence' not in q for q in pub['questions'])

def test_writer_relation_tamper_rejected():
    with tempfile.TemporaryDirectory() as td:
        # Build basic fixture then tamper review relation; test fixture setup is shared through first test logic.
        report=Path(td); cfg={'version':'x','baseline_pool':str(report/'BASE.json'),'baseline_result_bindings':str(report/'BIND.json'),'slots':[{'slot_id':'s','root_id':'r','history_path':'h','prior_pool_paths':[],'focus':{'scene_refs':[],'description':'topic'},'initial_target_count':2}],'max_questions':10,'training_allowed':False}; wr(report/'CONFIG.json',cfg); wr(report/'BASE.json',{'questions':[]}); wr(report/'BIND.json',{}); make_slot(report,'s','r',2)
        p=report/'assets/s/ASSET_REVIEW.json'; x=json.loads(p.read_text()); x['versions'][0]['relation']='contradicts'; wr(p,x)
        try: c.compose(report)
        except ValueError as e: assert 'review_judgment_disagreement' in str(e)
        else: assert False



def fixture(report, specs):
    cfg={'version':'x','baseline_pool':str(report/'BASE.json'),'baseline_result_bindings':str(report/'BIND.json'),
         'slots':[{'slot_id':sid,'root_id':root,'history_path':'h','prior_pool_paths':[],
                   'focus':{'scene_refs':[],'description':'topic'},'initial_target_count':target}
                  for sid,root,target,minimum in specs], 'max_questions':10,'training_allowed':False}
    wr(report/'CONFIG.json',cfg);wr(report/'BASE.json',{'questions':[]});wr(report/'BIND.json',{})
    for sid,root,target,minimum in specs: make_slot(report,sid,root,minimum)


def raises(reason, fn):
    try: fn()
    except ValueError as exc:
        assert reason in str(exc), str(exc)
    else: raise AssertionError('expected '+reason)


def change(report, sid, filename, mutator):
    p=report/'assets'/sid/filename; obj=c.read(p);mutator(obj);wr(p,obj)


def rebind_reviews(report,sid):
    run=report/'assets'/sid
    change(report,sid,'PUBLIC_ASSETS.json',lambda x:x.update(assets_sha256=sh(run/'ASSETS.json'),source_sha256=sh(run/'SOURCE.json')))
    for filename in ['VERSION_MAP.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json']:
        def mutate(x):
            x.update(source_sha256=sh(run/'SOURCE.json'),public_assets_sha256=sh(run/'PUBLIC_ASSETS.json'))
            if filename != 'SOURCE_JUDGMENTS.json': x['assets_sha256']=sh(run/'ASSETS.json')
            if filename == 'ASSET_REVIEW.json': x['source_judgments_sha256']=sh(run/'SOURCE_JUDGMENTS.json')
        change(report,sid,filename,mutate)


def test_cross_root_allocation_and_native_contract():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        initial=[(2,5),(4,2),(5,3),(3,6),(6,4)]
        specs=[(f's{root}{n}',f'r{root}',k,2) for root,pair in enumerate(initial) for n,k in enumerate(pair)]
        fixture(r,specs)
        out=c.compose(r); allocation=c.read(r/'ALLOCATION.json')
        assert allocation['assembled_distribution']=={str(k):2 for k in range(2,7)}
        assert all(len(roots)==2 for roots in allocation['distinct_roots_by_count'].values())
        assert not allocation['deviations']
        assert len(out['questions'])==10
        forbidden={'initial_target_count','target_count','feasible_counts','version_id','fact_id','internal_version_id'}
        def keys(value):
            if isinstance(value,dict):
                for k,v in value.items(): yield k;yield from keys(v)
            if isinstance(value,list):
                for v in value:yield from keys(v)
        for q in out['questions']:
            assert not forbidden.intersection(keys(q))
            assert list(q['options'])==list('ABCDEF')
            assert q['gold']==[l for l in 'ABCDEF' if q['option_evidence'][l]['relation']=='supports']
            assert len(set(c.norm(t) for t in q['options'].values()))==6
        # Existing native question contract agrees with assembled schema.
        native_path=HERE.parent/'h4_feedback_training_pool_v15/assemble_pool.py'
        spec=importlib.util.spec_from_file_location('native_contract',native_path)
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        for q in out['questions']:native.question_contract(q)
        assert c.verify_assembly(r)==out
        before={p.name:p.read_bytes() for p in r.glob('*.json')}
        raises('assembly_already_exists',lambda:c.compose(r))
        assert before=={p.name:p.read_bytes() for p in r.glob('*.json')}


def test_reject_mixed_root_and_modified_text():
    for kind in ['mixed_root','modified_text','duplicate_fact','extra_public_field']:
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);fixture(r,[('s','r',2,2)])
            if kind=='mixed_root':
                change(r,'s','ASSETS.json',lambda x:x.update(root_id='another'))
                reason='mixed_asset_root'
            elif kind=='modified_text':
                change(r,'s','PUBLIC_ASSETS.json',lambda x:x['versions'][0].update(proposition='modified text'))
                reason='public_writer_text_changed'
            elif kind=='duplicate_fact':
                change(r,'s','ASSETS.json',lambda x:x['facts'][1].update(fact_id=x['facts'][0]['fact_id']))
                reason='duplicate_or_missing_fact_id'
            else:
                change(r,'s','PUBLIC_ASSETS.json',lambda x:x.update(initial_target_count=2))
                reason='public_asset_private_field_leak'
            rebind_reviews(r,'s')
            raises(reason,lambda:c.build(r))


def test_reject_false_label_even_if_review_repeats_it():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',2,2)])
        for filename in ['SOURCE_JUDGMENTS.json','ASSET_REVIEW.json']:
            change(r,'s',filename,lambda x:x['versions'][0].update(relation='contradicts'))
        rebind_reviews(r,'s')
        raises('usable_unverified_or_false_label',lambda:c.build(r))


def test_review_coverage_and_independence():
    for filename,mutator,reason in [
        ('SOURCE_JUDGMENTS.json',lambda x:x.update(gold_read_before_save=True),'judgment_not_source_first'),
        ('ASSET_REVIEW.json',lambda x:x.update(solver_scores_used=True),'review_not_source_first'),
        ('ASSET_REVIEW.json',lambda x:x['versions'].pop(),'review_version_coverage_mismatch'),
        ('SOURCE_JUDGMENTS.json',lambda x:x['versions'][0].update(scope_valid=False),'usable_unverified'),
        ('SOURCE_JUDGMENTS.json',lambda x:x['versions'][0].update(fact_distinct=False),'usable_unverified')]:
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);fixture(r,[('s','r',2,2)]);change(r,'s',filename,mutator);rebind_reviews(r,'s')
            raises(reason,lambda:c.build(r))


def test_gaps_keep_valid_all_six_and_unknown_does_not_become_false():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',2,2)])
        for filename in ['SOURCE_JUDGMENTS.json','ASSET_REVIEW.json']:
            def mutate(x):
                for v in x['versions']:
                    if v['relation']=='contradicts':
                        v['relation']='unknown'
                        if filename=='ASSET_REVIEW.json':v['usable']=False
            change(r,'s',filename,mutate)
        rebind_reviews(r,'s')
        out=c.compose(r);q=out['questions'][0];a=c.read(r/'ALLOCATION.json')
        assert q['gold']==list('ABCDEF') and a['slots'][0]['feasible_counts']==[6]
        assert a['missing_target_counts']['2']==2
        assert a['deviations'][0]['actual_gold_count']==6


def test_exact_prior_dedup_and_same_batch():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',6,6)])
        previous={'root_id':'r','question_type':'multi_select','options':dict(zip('FEDCBA',[f'  STATEMENT   {i+1} IS TRUE. ' for i in range(6)]))}
        wr(r/'BASE.json',{'questions':[previous]})
        out=c.compose(r);assert out['candidate_count']==0
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s1','r',6,6),('s2','r',6,6)])
        out=c.compose(r);assert out['candidate_count']==1
        a=c.read(r/'ALLOCATION.json');assert [s for s in a['slots'] if not s['assembled']][0]['reason']=='SAME_BATCH_EXACT_SET_CONFLICT'


def test_tamper_outputs_detected_even_after_receipt_hash_update():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',2,2)]);c.compose(r)
        q=c.read(r/'POOL.json');q['questions'][0]['options']['A']='Changed after assembly';wr(r/'POOL.json',q)
        receipt=c.read(r/'ASSEMBLY_RECEIPTS.json')
        for b in receipt['outputs']:
            if b['path']==str(r/'POOL.json'):b['sha256']=sh(r/'POOL.json')
        wr(r/'ASSEMBLY_RECEIPTS.json',receipt)
        raises('assembly_replay_changed',lambda:c.verify_assembly(r))


def test_input_hash_tamper_detected():
    for filename in ['SOURCE.json','ASSETS.json','PUBLIC_ASSETS.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json']:
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);fixture(r,[('s','r',2,2)]);c.compose(r)
            p=r/'assets/s'/filename;p.write_text(p.read_text()+'\n')
            raises('assembly_bound_file_changed',lambda:c.verify_assembly(r))


def test_failed_generation_keeps_shortage():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',2,2)])
        for name in ['ASSETS.json','PUBLIC_ASSETS.json','VERSION_MAP.json','SOURCE_JUDGMENTS.json','ASSET_REVIEW.json']:(r/'assets/s'/name).unlink()
        change(r,'s','GENERATION.COMPLETE.json',lambda x:x.update(status='NO_VALID_ASSETS',fact_count=0))
        out=c.compose(r);assert out['candidate_count']==0
        assert c.read(r/'ALLOCATION.json')['slots'][0]['generation_status']=='NO_VALID_ASSETS'


def test_eight_fact_dual_version_feasibility():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',2,0)])
        import shutil
        shutil.rmtree(r/'assets/s');make_slot(r,'s','r',0,8)
        outputs,_,_=c.build(r)
        assert outputs['ALLOCATION.json']['slots'][0]['distinct_candidate_counts']=={'2':420,'3':560,'4':420,'5':168,'6':28}
        q=outputs['POOL.json']['questions'][0]
        assert len(q['gold'])==2
        assert len(set(v['fact_id'] for v in c.build(r)[2][0]['selected_versions']))==6


def declare_targets(report, target, priorities, required=None):
    path=report/'CONFIG.json';config=c.read(path)
    config.update(target_distribution=target, allocation_priority_counts=priorities)
    if required is not None: config['allocation_required_counts']=required
    wr(path,config)
    for slot in config['slots']:
        change(report,slot['slot_id'],'MANIFEST.json',lambda row:row['config'].update(sha256=sh(path)))


def test_priority_three_distribution_uses_verified_versions_and_preserves_gold():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        initial=[(2,3),(4,3),(5,3),(3,6),(3,3)]
        specs=[(f's{root}{n}',f'r{root}',k,2) for root,pair in enumerate(initial) for n,k in enumerate(pair)]
        fixture(r,specs)
        target={'2':1,'3':6,'4':1,'5':1,'6':1}
        declare_targets(r,target,[3])
        pool=c.compose(r);allocation=c.read(r/'ALLOCATION.json')
        assert allocation['assembled_distribution']==target
        assert allocation['distinct_root_targets']['3']==5
        assert len(allocation['distinct_roots_by_count']['3'])==5
        assert allocation['allocation_objective']['priority_count_coverage']=={'3':6}
        assert all(q['gold']==[label for label in 'ABCDEF' if q['option_evidence'][label]['relation']=='supports'] for q in pool['questions'])
        assert c.verify_assembly(r)==pool


def test_priority_shortage_retains_other_valid_counts_without_relabeled_versions():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s1','r1',3,2),('s2','r2',3,4),('s3','r3',3,6)])
        declare_targets(r,{'2':0,'3':3,'4':0,'5':0,'6':0},[3])
        outputs,_,_=c.build(r)
        questions=outputs['POOL.json']['questions'];allocation=outputs['ALLOCATION.json']
        assert len(questions)==3
        assert allocation['assembled_distribution']['3']==1
        assert allocation['missing_target_counts']['3']==2
        assert all(q['gold']==[label for label in 'ABCDEF' if q['option_evidence'][label]['relation']=='supports'] for q in questions)
        assert len(next(q for q in questions if q['root_id']=='r3')['gold'])==6


def test_invalid_declared_target_or_priority_rejected():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s','r',3,2)])
        declare_targets(r,{'2':0,'3':2,'4':0,'5':0,'6':0},[3])
        raises('target_distribution_slot_total_mismatch',lambda:c.build(r))
        declare_targets(r,{'2':0,'3':1,'4':0,'5':0,'6':0},[3,3])
        raises('invalid_allocation_priority_counts',lambda:c.build(r))


BALANCED_TARGET = {'2':4, '3':3, '4':3, '5':0, '6':0}


def allocation_candidate(root, count, token):
    """Small immutable six-fact candidate, shared across slots when token matches."""
    versions=[]
    for i in range(6):
        supported=i<count
        text=f'{token} fact {i} is '+('true.' if supported else 'false.')
        versions.append({'version_id':f'{token}-v{i}', 'fact_id':f'F{i}',
                         'internal_version_id':f'F{i}:'+('s' if supported else 'c'),
                         'proposition':text, 'relation':'supports' if supported else 'contradicts',
                         'rationale':'Frozen synthetic source relation.',
                         'anchor':{'turn_ref':f'{root}:T{i}', 'speaker':'S', 'quote':f'{token} fact {i} is true.'}})
    key=c.statement_key(root,[v['proposition'] for v in versions])
    return {'versions':tuple(versions),'key':key,'id':c.digest(key)}


def allocation_fixture(specs):
    assets=[]; available=[]
    for i,(root,preferred,counts) in enumerate(specs):
        assets.append({'slot':{'slot_id':f's{i}','root_id':root,'initial_target_count':preferred,
                              'focus':{'description':'Synthetic source facts.'}},
                       'stem':'Which statements are supported?', 'source_path':'synthetic/SOURCE.json',
                       'source_sha256':'0'*64})
        available.append({k:[allocation_candidate(root,k,'shared')] if k in counts else [] for k in c.COUNTS})
    return assets,available


def assignment_score(assets, selected, target, required, priorities):
    """Independent full-assignment oracle; never uses capped DP states/helpers."""
    actual=[]; keys=[]; preferred=0
    for i, candidate in enumerate(selected):
        if candidate is None: continue
        keys.append(candidate['key'])
        count=sum(v['relation']=='supports' for v in candidate['versions'])
        actual.append((assets[i]['slot']['root_id'],count))
        preferred+=count==assets[i]['slot']['initial_target_count']
    if len(keys)!=len(set(keys)): return None
    hist=Counter(k for _,k in actual)
    root_count=len({a['slot']['root_id'] for a in assets})
    root_coverage=sum(min(target[str(k)],root_count,len({r for r,j in actual if j==k})) for k in c.COUNTS)
    if required:
        prefix=(sum(hist[k]>0 for k in required),tuple(sorted(Fraction(min(hist[k],target[str(k)]),target[str(k)])
                                                             for k in required if target[str(k)]>0)))
    else:
        prefix=tuple(min(hist[k],target[str(k)]) for k in priorities)
    return (*prefix,sum(min(hist[k],target[str(k)]) for k in c.COUNTS),root_coverage,len(actual),preferred)


def brute_force_score(assets,available,target,required,priorities=()):
    choices=[[None]+[candidate for k in c.COUNTS for candidate in row[k]] for row in available]
    scores=(assignment_score(assets,selection,target,required,priorities) for selection in itertools.product(*choices))
    return max(score for score in scores if score is not None)


def assert_optimal_allocation(assets,available,target,required,priorities=()):
    before=deepcopy((assets,available))
    selected,objective=c.allocate(assets,available,target,priorities,required)
    assignment=[selected.get(i) for i in range(len(assets))]
    assert assignment_score(assets,assignment,target,required,priorities)==brute_force_score(assets,available,target,required,priorities)
    assert (assets,available)==before  # Allocation must not change any relation, version, or text.
    hist=Counter(sum(v['relation']=='supports' for v in row['versions']) for row in assignment if row is not None)
    assert objective['required_counts_covered']==sum(hist[k]>0 for k in required)
    ratios={str(k):Fraction(min(hist[k],target[str(k)]),target[str(k)]) for k in required if target[str(k)]>0}
    exact=lambda v:f'{v.numerator}/{v.denominator}'
    assert objective['balance_target_fractions']=={k:exact(v) for k,v in ratios.items()}
    assert objective['balance_sorted_target_fractions']==[exact(v) for v in sorted(ratios.values())]
    assert objective['balance_excluded_zero_target_counts']==[k for k in required if target[str(k)]==0]
    assert objective['target_coverage']==sum(min(hist[k],target[str(k)]) for k in c.COUNTS)
    for i,candidate in enumerate(assignment):
        if candidate is None:continue
        count=sum(v['relation']=='supports' for v in candidate['versions'])
        assert candidate in available[i][count]
        question,receipt=c.assemble_question(assets[i],candidate)
        versions={v['version_id']:v for v in candidate['versions']}
        for chosen in receipt['selected_versions']:
            original=versions[chosen['version_id']];letter=chosen['label']
            assert question['options'][letter]==original['proposition']
            assert question['option_evidence'][letter]['relation']==original['relation']
            assert all(question['option_evidence'][letter][k]==v for k,v in original['anchor'].items())
        assert question['gold']==[l for l in 'ABCDEF' if question['option_evidence'][l]['relation']=='supports']
        assert len(question['gold'])==count
    return selected,objective,hist


def test_required_coverage_prevents_priority_three_starving_feasible_two_and_four():
    assets,available=allocation_fixture([(f'r{i}',3,[2,3,4]) for i in range(3)])
    # The old priority rule can exhaust all three slots on count 3 even though
    # count 2 and count 4 are individually and jointly feasible.
    _,_,old_hist=assert_optimal_allocation(assets,available,BALANCED_TARGET,[],[3])
    assert old_hist=={3:3}
    _,objective,hist=assert_optimal_allocation(assets,available,BALANCED_TARGET,[2,3,4])
    assert hist=={2:1,3:1,4:1}
    assert objective['required_counts_covered']==3
    assert objective['balance_sorted_target_fractions']==['1/4','1/3','1/3']


def test_leximin_improves_lowest_target_fractions_before_initial_three_preference():
    assets,available=allocation_fixture([(f'r{i}',3,[2,3,4]) for i in range(4)])
    _,objective,hist=assert_optimal_allocation(assets,available,BALANCED_TARGET,[2,3,4],[3])
    assert hist=={2:2,3:1,4:1}
    assert objective['balance_sorted_target_fractions']==['1/3','1/3','1/2']


def test_required_jointly_infeasible_and_shared_set_conflicts_match_brute_force():
    target={'2':1,'3':1,'4':1,'5':0,'6':0}
    assets,available=allocation_fixture([('r0',3,[2,3]),('r1',4,[4]),('r2',6,[6])])
    _,objective,hist=assert_optimal_allocation(assets,available,target,[2,3,4])
    assert set(k for k in c.COUNTS if any(row[k] for row in available))=={2,3,4,6}
    assert objective['required_counts_covered']==2 and sum(hist.values())==3
    # One complete statement set may be retained once, even if several slots
    # offer it. Required coverage still wins over an initial preference.
    assets,available=allocation_fixture([('same',3,[2,3]),('same',3,[2,3]),('same',4,[2,4]),('other',6,[6])])
    _,objective,hist=assert_optimal_allocation(assets,available,target,[2,3,4])
    assert objective['required_counts_covered']==3 and hist=={2:1,3:1,4:1,6:1}
    # A non-first candidate is necessary for the best valid assignment.
    available[0][2].append(allocation_candidate('same',2,'alternative'))
    available[1]={k:available[1][k] if k==2 else [] for k in c.COUNTS}
    available[2]={k:available[2][k] if k==2 else [] for k in c.COUNTS}
    assert_optimal_allocation(assets,available,target,[2,3,4])


def test_zero_target_fallbacks_and_required_presence_have_no_zero_denominator():
    target={'2':1,'3':1,'4':1,'5':0,'6':0}
    assets,available=allocation_fixture([('r0',2,[2]),('r1',3,[3]),('r2',4,[4]),('r3',5,[5]),('r4',6,[6])])
    _,objective,hist=assert_optimal_allocation(assets,available,target,[2,3,4])
    assert hist=={2:1,3:1,4:1,5:1,6:1}
    assert objective['balance_counts']==[2,3,4]
    assert objective['balance_sorted_target_fractions']==['1/1']*3
    assert objective['target_coverage']==3 and objective['retained_questions']==5
    # Presence must survive DP capping even when all target counts are zero.
    assets,available=allocation_fixture([('r0',6,[2,6]),('r1',6,[3,6]),('r2',6,[4,6])])
    _,objective,hist=assert_optimal_allocation(assets,available,{str(k):0 for k in c.COUNTS},[2,3,4])
    assert hist=={2:1,3:1,4:1}
    assert objective['balance_sorted_target_fractions']==[]
    assert objective['balance_excluded_zero_target_counts']==[2,3,4]
    assert objective['required_counts_covered']==3 and objective['target_coverage']==0


def test_capped_dp_pruning_against_exhaustive_assignments():
    rng=random.Random(690234)
    for case in range(64):
        count=rng.randrange(2,6)
        specs=[(f'r{rng.randrange(2)}',rng.choice(c.COUNTS),[]) for _ in range(count)]
        assets,available=allocation_fixture(specs)
        for i,asset in enumerate(assets):
            root=asset['slot']['root_id']
            for k in rng.sample(c.COUNTS,rng.randrange(4)):
                # Repeated roots share candidate sets; a second option tests
                # matching without collapsing away future slot feasibility.
                available[i][k]=[allocation_candidate(root,k,str(token)) for token in rng.sample(range(2),rng.randrange(1,3))]
        target={str(k):rng.randrange(4) for k in c.COUNTS}
        required=[2,3,4] if case%3 else []
        priorities=[3] if case%2 else []
        assert_optimal_allocation(assets,available,target,required,priorities)


def test_balanced_build_reports_exact_distribution_and_preserves_frozen_inputs():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        fixture(r,[(f's{i}',f'r{i//2}',3,2) for i in range(10)])
        declare_targets(r,BALANCED_TARGET,[],[2,3,4])
        inputs={str(p):p.read_bytes() for p in r.rglob('*.json')}
        pool=c.compose(r);allocation=c.read(r/'ALLOCATION.json')
        assert allocation['assembled_distribution']==BALANCED_TARGET
        assert allocation['allocation_required_counts']==[2,3,4]
        assert allocation['allocation_priority_counts']==[]
        assert allocation['required_counts_covered']==[2,3,4]
        assert allocation['missing_required_counts']==[]
        assert allocation['allocation_objective']['required_counts_covered']==3
        assert allocation['allocation_objective']['balance_sorted_target_fractions']==['1/1']*3
        assert all(Path(p).read_bytes()==body for p,body in inputs.items())
        assert len({c.statement_key(q['root_id'],q['options'].values()) for q in pool['questions']})==10
        assert all(q['gold']==[l for l in 'ABCDEF' if q['option_evidence'][l]['relation']=='supports'] for q in pool['questions'])
        assert c.verify_assembly(r)==pool


def test_balanced_build_reports_joint_shortage_and_retains_six_fallback():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);fixture(r,[('s0','r0',3,2),('s1','r1',4,4),('s2','r2',6,6)])
        declare_targets(r,{'2':1,'3':1,'4':1,'5':0,'6':0},[],[2,3,4])
        for sid,maximum in [('s0',3),('s1',4)]:
            def limit(review):
                for v in review['versions']:
                    if v['relation']=='supports' and int(v['version_id'][1:-1])>maximum:
                        v.update(usable=False,reason='Excluded synthetic version.')
            change(r,sid,'ASSET_REVIEW.json',limit)
        pool=c.compose(r);allocation=c.read(r/'ALLOCATION.json')
        assert [s['feasible_counts'] for s in allocation['slots']]==[[2,3],[4],[6]]
        assert allocation['required_counts_covered']==[3,4]
        assert allocation['missing_required_counts']==[2]
        assert allocation['missing_target_counts']=={'2':1,'3':0,'4':0,'5':0,'6':0}
        assert allocation['assembled_distribution']['6']==1 and pool['candidate_count']==3
        assert allocation['allocation_objective']['required_counts_covered']==2
        assert allocation['allocation_objective']['balance_sorted_target_fractions']==['0/1','1/1','1/1']
        assert c.verify_assembly(r)==pool


def test_invalid_required_configuration_rejected():
    for required in [[2,2],['2'],[True],[1],[7],{},'234']:
        with tempfile.TemporaryDirectory() as td:
            r=Path(td);fixture(r,[('s','r',2,2)])
            declare_targets(r,{'2':1,'3':0,'4':0,'5':0,'6':0},[],required)
            raises('invalid_allocation_required_counts',lambda:c.build(r))


if __name__ == '__main__':
    tests=sorted((name,fn) for name,fn in globals().items() if name.startswith('test_') and callable(fn))
    for name,fn in tests:
        fn();print('PASS',name)
    print(f'{len(tests)} tests passed')
