"""Offline checks of review freezing, native pairing and complete raw replay."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import contextlib
import io
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import evaluate as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='v75-evaluation-test-')
        self.addCleanup(temp.cleanup)
        self.home=Path(temp.name)
        self.block=patch('socket.socket',side_effect=AssertionError('Offline tests cannot call models'))
        self.block.start();self.addCleanup(self.block.stop)

    def fixture(self,counts=(2,3),statuses=('CORE_USABLE','KNOWN_SUBSTANTIVE_ISSUE')):
        source={'root_id':'fixture-root','turns':[{'turn_ref':f'P:T{i}','speaker':'A',
            'text':f'The value for event {i} is {i}.'} for i in range(1,7)]}
        source_path=self.home/'SOURCE.json';ev.write(source_path,source)
        ev.write(self.home/'HISTORY.json',source)
        questions=[]
        judgments=[]
        for index,(count,status) in enumerate(zip(counts,statuses)):
            options={label:f'The value for event {i+1} is {i+1 if i<count else i+11}.'
                     for i,label in enumerate('ABCDEF')}
            if status=='KNOWN_SUBSTANTIVE_ISSUE':
                options['F']='Event 6 is also planned to have an additional value next year.'
            evidence={}
            judged={}
            for i,label in enumerate('ABCDEF'):
                relation='supports' if i<count else 'contradicts'
                if label=='F' and status=='KNOWN_SUBSTANTIVE_ISSUE': relation='unknown'
                anchor={'turn_ref':f'P:T{i+1}','speaker':'A','quote':source['turns'][i]['text']}
                evidence[label]={**anchor,'relation':relation,'reason':'Explicit current value; future plan is absent.'}
                judged[label]={'proposition':options[label],'relation':relation,'scope_valid':True,
                    'fact_distinct':True,'reason':'Explicit current value; future plan is absent.','citations':[anchor]}
            q={'candidate_id':f'fixture-q{index}','root_id':'fixture-root','question_type':'multi_select',
                'question':f'Which statements about event set {index} are supported?',
                'options':options,'gold':list('ABCDEF'[:count]),'option_evidence':evidence,
                'source_path':str(source_path),'source_sha256':ev.sha(source_path),'training_allowed':False}
            questions.append(q)
            judgments.append({**{k:q[k] for k in ('candidate_id','root_id','question','source_sha256')},
                              'complete_source_turns_read':6,'stem_valid':True,'options':judged})
        pool={'questions':questions,'candidate_count':len(questions),'training_allowed':False,
              'provenance':{'requested_model':'Qwen/Qwen3.6-35B-A3B'}}
        ev.write(self.home/'POOL.json',pool)
        ev.write(self.home/'PUBLIC_QUESTIONS.json',{'questions':[
            {k:q[k] for k in ('candidate_id','root_id','question_type','question','options','source_path','source_sha256')}
            for q in questions]})
        ev.write(self.home/'ALLOCATION.json',{})
        ev.write(self.home/'ASSEMBLY_RECEIPTS.json',{'inputs':[ev.bind(source_path),ev.bind(self.home/'HISTORY.json')]})
        ev.write(self.home/'CONFIG.json',{'training_allowed':False,'max_questions':10,'max_calls_per_model':180,
            'review_schema_version':ev.review_schema.SCHEMA_VERSION,'final_review_groups':['a'],
            'slots':[{'root_id':'fixture-root','history_path':str(self.home/'HISTORY.json')}]})
        disclosures={'parent_all_true_provenance_known':True,'parent_gold_known_before_review':True,
            'fresh_blind_parent_review':False,'current_solver_scores_read':False,'solver_scores_used':False,
            'child_gold_read_before_save':False,'current_pool_read_before_save':False}
        timestamp=datetime.now(timezone.utc)-timedelta(seconds=3)
        saved_at=timestamp.isoformat()
        self.judgment_path=self.home/'final_review/group_a_SOURCE_JUDGMENTS.json'
        judgment={'review_schema_version':ev.review_schema.SCHEMA_VERSION,**disclosures,
            'group':'a','reviewer':'offline-fixture','created_at_utc':saved_at,
            'public_questions':ev.bind(self.home/'PUBLIC_QUESTIONS.json'),
            'source':[{'root_id':'fixture-root',**ev.bind(source_path),'complete_turn_count':6}],
            'current_complete_source_read':{'fixture-root':{'complete':True,'turn_positions_inclusive':[1,6]}},
            'questions':judgments}
        ev.write(self.judgment_path,judgment)
        self.audit_path=self.home/'final_review/group_a_AUDIT.json'
        rows=[{'candidate_id':q['candidate_id'],'root_id':q['root_id'],'source_sha256':q['source_sha256'],
               'core_status':status} for q,status in zip(questions,statuses)]
        audit={'group':'a','created_at_utc':(timestamp+timedelta(seconds=2)).isoformat(),
            'pool_sha256':ev.sha(self.home/'POOL.json'),'source_judgments_sha256':ev.sha(self.judgment_path),
            'current_solver_scores_read':False,'solver_scores_used':False,'source_first_for_current_children':True,
            'child_gold_read_before_save':False,'source_judgments_bytes_changed_after_private_read':False,
            'current_pool_read_after_both_group_source_judgments_save':True,
            'first_private_pool_read_at_utc':(timestamp+timedelta(seconds=1)).isoformat(),
            'both_source_judgments_saved_bindings':[{'group':'a',**ev.bind(self.judgment_path),
                'source_judgments_saved_at_utc':saved_at}],
            'question_or_gold_edits':[],'training_allowed':False,'questions':rows}
        ev.write(self.audit_path,audit)
        review={'review_schema_version':ev.review_schema.SCHEMA_VERSION,**disclosures,
            'derived_from_saved_final_reviews':True,'pool_sha256':ev.sha(self.home/'POOL.json'),
            'reviewed_at_utc':(timestamp+timedelta(seconds=2,microseconds=500000)).isoformat(),
            'independent_review_files':[ev.bind(self.judgment_path),ev.bind(self.audit_path)],'questions':rows}
        ev.write(self.home/'FINAL_CONTENT_REVIEW.json',review)
        checker=patch.object(ev,'verify_assembly',side_effect=lambda home:ev.read(home/'POOL.json'))
        self.assembly_checker=checker.start();self.addCleanup(checker.stop)
        return pool

    def rewrite(self,path,value):
        Path(path).write_text(json.dumps(value)+'\n')

    def script(self,name):
        path=ev.HERE/'execution'/f'{name}.py'
        spec=importlib.util.spec_from_file_location('offline_'+name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_frozen_usable_subset_can_be_prepared(self):
        pool=self.fixture()
        ev.freeze_review(self.home)
        with patch.object(ev,'execute') as execute,patch.object(ev,'paired_inputs',return_value={'request_count':18}):
            result=ev.prepare(self.home)
        self.assertEqual(result['candidate_count'],1)
        self.assertEqual(ev.read(self.home/'EVALUATION_POOL.json')['questions'],pool['questions'][:1])
        self.assertEqual(execute.call_count,2)
        self.assertEqual({call.args[1][-1] for call in execute.call_args_list},{'M0','M_clean_policy'})

    def test_review_is_required_before_prepare_side_effects(self):
        self.fixture()
        with patch.object(ev,'execute') as execute,self.assertRaises(FileNotFoundError): ev.prepare(self.home)
        execute.assert_not_called()
        self.assertFalse((self.home/'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_missing_review_row_and_source_tampering_fail(self):
        self.fixture()
        path=self.home/'FINAL_CONTENT_REVIEW.json'
        original=ev.read(path);review=deepcopy(original);review['questions'].pop();path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'merged_review_changed'):ev.freeze_review(self.home)
        review['questions'].append({'candidate_id':'fixture-q1','root_id':'fixture-root','source_sha256':'bad','core_status':'CORE_USABLE'})
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'merged_review_changed'):ev.freeze_review(self.home)
        self.rewrite(path,original)
        source_path=self.home/'SOURCE.json';source=ev.read(source_path)
        source['turns'][0]['text']='The source has changed after the saved judgment.'
        self.rewrite(source_path,source)
        self.assembly_checker.reset_mock()
        with self.assertRaisesRegex(ValueError,'source:sha_changed'):ev.freeze_review(self.home)
        self.assembly_checker.assert_not_called()

    def test_frozen_review_cannot_change_and_started_blocks_freeze(self):
        self.fixture();ev.freeze_review(self.home)
        path=self.home/'FINAL_CONTENT_REVIEW.json';review=ev.read(path)
        review['questions'][1]['core_status']='CORE_USABLE';path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'merged_review_changed'):ev.verify_review(self.home)
        ev.write(ev.solver_dir(self.home,'trained')/'STARTED.json',{})
        with self.assertRaisesRegex(ValueError,'cannot_change_after_solver'):ev.freeze_review(self.home)

    def test_empty_usable_pool_skips_both_models(self):
        self.fixture(statuses=('KNOWN_SUBSTANTIVE_ISSUE','KNOWN_SUBSTANTIVE_ISSUE'))
        ev.freeze_review(self.home)
        with patch.object(ev,'execute') as execute, patch.object(ev,'native',side_effect=AssertionError('Empty pool must not load solver')):
            result=ev.prepare(self.home)
            ev.live(self.home,'trained');ev.live(self.home,'base')
            audited=ev.verify(self.home)
        execute.assert_not_called()
        self.assertEqual(result['calls_per_model'],0)
        self.assertEqual(audited['calls_per_model'],0)
        self.assertEqual(result['evaluation_status'],'SKIPPED_EMPTY_CONTENT_USABLE_POOL')
        self.assertEqual(audited['evaluation_status'],'SKIPPED_EMPTY_CONTENT_USABLE_POOL')

    def test_missing_disclosure_stops_freeze_before_private_assembly(self):
        self.fixture()
        judgment=ev.read(self.judgment_path);judgment.pop('current_solver_scores_read')
        self.rewrite(self.judgment_path,judgment)
        with self.assertRaisesRegex(ValueError,'current_solver_scores_read:required'):
            ev.freeze_review(self.home)
        self.assembly_checker.assert_not_called()
        self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json').exists())

    def test_saved_final_record_tampering_stops_prepare_before_native(self):
        self.fixture();frozen=ev.freeze_review(self.home)
        self.assertEqual(frozen['review_schema_code'],ev.bind(Path(ev.review_schema.__file__)))
        audit=ev.read(self.audit_path);audit['additional_note']='Changed after freezing.'
        self.rewrite(self.audit_path,audit)
        # Even a summary updated coherently to match the new valid audit cannot
        # replace the records already frozen for this evaluation.
        path=self.home/'FINAL_CONTENT_REVIEW.json';review=ev.read(path)
        review['independent_review_files'][1]=ev.bind(self.audit_path)
        self.rewrite(path,review)
        with patch.object(ev,'execute') as execute,patch.object(ev,'native') as native:
            with self.assertRaisesRegex(ValueError,'frozen_review_changed'):ev.prepare(self.home)
        execute.assert_not_called();native.assert_not_called()
        self.assertFalse((self.home/'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_round_and_historical_records_are_frozen_explicitly(self):
        self.fixture()
        round_j=self.home/'assets/family/rounds/initial/SOURCE_JUDGMENTS.json'
        round_receipt=round_j.with_name('REVIEW_RECEIPT.json')
        old_parent=self.home/'historical_parent_SOURCE_JUDGMENTS.json'
        old_provenance=self.home/'historical_parent_PROVENANCE.json'
        for path in (round_j,round_receipt,old_parent,old_provenance):ev.write(path,{'fixture':'saved'})
        config_path=self.home/'CONFIG.json';config=ev.read(config_path)
        config['slots'][0].update(parent_review_path=str(old_parent),parent_provenance_path=str(old_provenance))
        self.rewrite(config_path,config)
        frozen=ev.freeze_review(self.home)
        paths={row['path'] for row in frozen['review_records']}
        self.assertTrue({str(p) for p in (round_j,round_receipt,old_parent,old_provenance)}<=paths)
        self.rewrite(round_j,{'fixture':'changed'})
        with patch.object(ev,'execute') as execute,self.assertRaisesRegex(ValueError,'frozen_review_changed'):
            ev.prepare(self.home)
        execute.assert_not_called()
        self.assertFalse((self.home/'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_invalid_review_stops_launch_and_runner_before_any_side_effect(self):
        self.fixture();ev.freeze_review(self.home)
        judgment=ev.read(self.judgment_path);judgment.pop('current_solver_scores_read')
        self.rewrite(self.judgment_path,judgment)
        launch=self.script('launch_evaluations');runner=self.script('run_evaluations')
        self.assembly_checker.reset_mock()
        with patch.object(launch.subprocess,'Popen') as popen,patch.object(runner.subprocess,'run') as run:
            for module in (launch,runner):
                with self.subTest(module=module.__name__),self.assertRaisesRegex(ValueError,'current_solver_scores_read:required'):
                    module.main(self.home)
        popen.assert_not_called();run.assert_not_called();self.assembly_checker.assert_not_called()
        self.assertFalse((self.home/'execution/evaluation_sequence.log').exists())
        self.assertFalse((self.home/'execution/evaluation_sequence.LAUNCH.json').exists())
        self.assertFalse((self.home/'execution/evaluation_sequence.STARTED.json').exists())

    def test_frozen_metadata_is_checked_before_preparation(self):
        self.fixture();ev.freeze_review(self.home)
        path=self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json';original=ev.read(path)
        cases=[('status','COMPLETE','review_not_frozen'),
               ('completed_at_utc',None,'timestamp_required'),
               ('completed_at_utc',(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),'future_timestamp'),
               ('usable_count',True,'counts_changed'),('candidate_count',99,'counts_changed')]
        for key,value,why in cases:
            changed=deepcopy(original);changed[key]=value;self.rewrite(path,changed)
            with self.subTest(key=key,value=value),patch.object(ev,'execute') as execute:
                with self.assertRaisesRegex(ValueError,why):ev.prepare(self.home)
                execute.assert_not_called()
            self.assertFalse((self.home/'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_freeze_timestamp_cannot_precede_audit_or_merged_summary(self):
        self.fixture();ev.freeze_review(self.home)
        path=self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json';original=ev.read(path)
        audit_at=datetime.fromisoformat(ev.read(self.audit_path)['created_at_utc'])
        summary_at=datetime.fromisoformat(ev.read(self.home/'FINAL_CONTENT_REVIEW.json')['reviewed_at_utc'])
        for bound in (audit_at,summary_at):
            frozen=deepcopy(original);frozen['completed_at_utc']=(bound-timedelta(microseconds=1)).isoformat()
            self.rewrite(path,frozen)
            with self.subTest(bound=bound),patch.object(ev,'execute') as execute:
                with self.assertRaisesRegex(ValueError,'content_freeze_before_review_completion'):
                    ev.prepare(self.home)
                execute.assert_not_called()
            self.assertFalse((self.home/'EVALUATION_PREPARATION.STARTED.json').exists())

    def test_live_and_verify_check_freeze_time_before_stage_side_effects(self):
        self.fixture(statuses=('KNOWN_SUBSTANTIVE_ISSUE','KNOWN_SUBSTANTIVE_ISSUE'))
        ev.freeze_review(self.home);ev.prepare(self.home)
        path=self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json';frozen=ev.read(path)
        audit_at=datetime.fromisoformat(ev.read(self.audit_path)['created_at_utc'])
        frozen['completed_at_utc']=(audit_at-timedelta(seconds=1)).isoformat();self.rewrite(path,frozen)
        with patch.object(ev,'execute') as execute,patch.object(ev,'native') as native:
            for action in (lambda:ev.live(self.home,'trained'),lambda:ev.verify(self.home)):
                with self.assertRaisesRegex(ValueError,'content_freeze_before_review_completion'):action()
        execute.assert_not_called();native.assert_not_called()
        self.assertFalse((self.home/'TRAINED_EVALUATION.STARTED.json').exists())
        self.assertFalse((self.home/'TRAINED_EVALUATION.ERROR.json').exists())

    def test_summary_canonical_fields_cannot_be_missing_or_coerced(self):
        self.fixture()
        path=self.home/'FINAL_CONTENT_REVIEW.json';original=ev.read(path)
        cases=[('parent_gold_known_before_review',None,'required'),
               ('current_solver_scores_read','false','explicit_bool_required'),
               ('derived_from_saved_final_reviews',1,'explicit_bool_required'),
               ('review_schema_version',None,'unsupported_or_missing'),
               ('fresh_blind_parent_review',True,'derived_disclosure_changed')]
        for key,value,why in cases:
            changed=deepcopy(original)
            if value is None:changed.pop(key)
            else:changed[key]=value
            self.rewrite(path,changed)
            with self.subTest(key=key),patch.object(ev,'execute') as execute:
                with self.assertRaisesRegex(ValueError,key+':'+why):ev.freeze_review(self.home)
                execute.assert_not_called()
            self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json').exists())

    def test_audit_private_read_must_follow_saved_source_judgment(self):
        self.fixture()
        saved_at=datetime.fromisoformat(ev.read(self.judgment_path)['created_at_utc'])
        audit=ev.read(self.audit_path)
        audit['first_private_pool_read_at_utc']=(saved_at-timedelta(microseconds=1)).isoformat()
        self.rewrite(self.audit_path,audit)
        with patch.object(ev,'execute') as execute,self.assertRaisesRegex(ValueError,'source_before_private_required'):
            ev.freeze_review(self.home)
        execute.assert_not_called()
        self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json').exists())

    def test_source_judgment_bytes_cannot_change_after_public_validation(self):
        self.fixture()
        original_read=ev.review_schema.read;changed=False
        def read_and_change(path):
            nonlocal changed
            value=original_read(path)
            if Path(path)==self.home/'POOL.json' and not changed:
                self.judgment_path.write_bytes(self.judgment_path.read_bytes()+b' ')
                changed=True
            return value
        with patch.object(ev.review_schema,'read',side_effect=read_and_change),patch.object(ev,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'review_records:sha_changed'):ev.freeze_review(self.home)
        self.assertTrue(changed);execute.assert_not_called()
        self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json').exists())

    def test_audit_bytes_cannot_change_during_private_gold_check(self):
        self.fixture()
        original_read=ev.review_schema.read;private_seen=False;changed=False
        def read_and_change(path):
            nonlocal private_seen,changed
            value=original_read(path)
            if Path(path)==self.home/'POOL.json':private_seen=True
            if Path(path)==self.home/'SOURCE.json' and private_seen and not changed:
                self.audit_path.write_bytes(self.audit_path.read_bytes()+b' ')
                changed=True
            return value
        with patch.object(ev.review_schema,'read',side_effect=read_and_change),patch.object(ev,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'audit:bytes_changed_during_validation'):ev.freeze_review(self.home)
        self.assertTrue(changed);execute.assert_not_called()
        self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.COMPLETE.json').exists())

    def test_merge_accepts_real_records_and_result_can_be_frozen(self):
        self.fixture();(self.home/'FINAL_CONTENT_REVIEW.json').unlink()
        merge=self.script('merge_final_review')
        with contextlib.redirect_stdout(io.StringIO()):result=merge.main(self.home)
        self.assertEqual(result['independent_review_files'],[ev.bind(self.judgment_path),ev.bind(self.audit_path)])
        self.assertTrue(result['parent_gold_known_before_review'])
        self.assertFalse(result['fresh_blind_parent_review'])
        self.assertTrue(result['derived_from_saved_final_reviews'])
        frozen=ev.freeze_review(self.home)
        self.assertEqual(frozen['usable_count'],1)
        self.assertEqual(ev.verify_review(self.home),frozen)

    def test_merge_rejects_incomplete_source_without_writing_summary(self):
        self.fixture();(self.home/'FINAL_CONTENT_REVIEW.json').unlink()
        judgment=ev.read(self.judgment_path);judgment.pop('parent_gold_known_before_review')
        self.rewrite(self.judgment_path,judgment)
        merge=self.script('merge_final_review')
        with self.assertRaisesRegex(ValueError,'parent_gold_known_before_review:required'):merge.main(self.home)
        self.assertFalse((self.home/'FINAL_CONTENT_REVIEW.json').exists())

    def test_importing_execution_scripts_has_no_process_or_record_side_effects(self):
        with patch('subprocess.Popen') as popen,patch('subprocess.run') as run,patch.object(ev,'write') as write:
            for name in ('launch_evaluations','run_evaluations','merge_final_review'):self.script(name)
        popen.assert_not_called();run.assert_not_called();write.assert_not_called()

    def test_native_temporary_preflight_and_all_raw_scores_replay(self):
        self.fixture(counts=(2,),statuses=('CORE_USABLE',))
        ev.freeze_review(self.home)
        n=ev.native()
        def offline_execute(home,args,log_name):
            self.assertEqual(args[0],'preflight')
            pool=Path(args[args.index('--pool')+1]);run=Path(args[args.index('--run')+1])
            model=args[args.index('--solver-model')+1]
            with contextlib.redirect_stdout(io.StringIO()):
                n.preflight(pool,run,solver_model=model,require_model=False)
        with patch.object(ev,'execute',side_effect=offline_execute):ev.prepare(self.home)
        for side in ev.MODELS:
            run=ev.solver_dir(self.home,side);m=ev.read(run/'manifest.json');gold=ev.read(run/'gold.json')
            qs={q['candidate_id']:q for q in ev.read(run/'questions.json')['candidates']}
            ev.write(run/'STARTED.json',{'started_at_utc':(datetime.now(timezone.utc)+timedelta(seconds=1)).isoformat(),
                'manifest_sha256':ev.sha(run/'manifest.json')})
            for index,request in enumerate(m['requests']):
                qid=request['candidate_id'];inverse={cid:label for label,cid in request['variant']['display_to_content'].items()}
                raw='<box>'+','.join(inverse[cid] for cid in gold[qid])+ '</box>' if index%2==0 else '<box>A,B,C,D,E,F</box>'
                score=n.score_response({**qs[qid],'gold_content_ids':gold[qid]},request,raw)
                meta={key:m['decoder_contract'][key] for key in ('temperature','top_p','max_tokens','structured_output_api','structured_output_version')}
                meta.update(sampling_seed=request['seed'],prompt_token_count=request['prompt_tokens'],finish_reason='stop')
                record={'request_id':request['request_id'],'candidate_id':qid,'root_id':request['root_id'],
                    'question_type':'multi_select','variant_id':request['variant']['variant_id'],'slot':request['slot'],
                    'mode':'full_context','seed':request['seed'],'solver_model':m['solver_model'],'model_path':m['model'],
                    'raw_output':raw,'metadata':meta,'evaluation_valid':True,'data_errors':[],'score':score}
                ev.write(run/'responses'/(request['request_id']+'.json'),record)
            ev.write(run/'COMPLETE.json',{'actual_call_count':18})
            with contextlib.redirect_stdout(io.StringIO()):n.report(run)
            ev.write(self.home/(side.upper()+'_EVALUATION.COMPLETE.json'),{'calls':18})
        result=ev.verify(self.home)
        self.assertEqual(result['calls_per_model'],18)
        self.assertEqual(result['models']['trained']['questions'][0]['full_partial_mean'],'1/2')
        self.assertTrue(result['models']['trained']['questions'][0]['selected_by_policy'])
        response=next((ev.solver_dir(self.home,'base')/'responses').glob('*.json'))
        value=ev.read(response);value['score']['partial_score_fraction']='999';response.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'score_replay_mismatch'):ev.verify(self.home)


if __name__=='__main__':unittest.main()
