"""Offline checks of review freezing, native pairing and complete raw replay."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import evaluate as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='v70-evaluation-test-')
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
        for index,count in enumerate(counts):
            questions.append({'candidate_id':f'fixture-q{index}','root_id':'fixture-root','question_type':'multi_select',
                'question':f'Which statements about event set {index} are supported?',
                'options':{label:f'Event statement {index}/{label}' for label in 'ABCDEF'},'gold':list('ABCDEF'[:count]),
                'option_evidence':{label:{'turn_ref':f'P:T{i+1}','speaker':'A','quote':source['turns'][i]['text'],
                    'relation':'supports' if i<count else 'contradicts','reason':'Synthetic explicitly bound version.'} for i,label in enumerate('ABCDEF')},
                'source_path':str(source_path),'source_sha256':ev.sha(source_path),'training_allowed':False})
        pool={'questions':questions,'candidate_count':len(questions),'training_allowed':False,
              'provenance':{'requested_model':'Qwen/Qwen3.6-35B-A3B'}}
        ev.write(self.home/'POOL.json',pool)
        ev.write(self.home/'ALLOCATION.json',{})
        ev.write(self.home/'ASSEMBLY_RECEIPTS.json',{})
        ev.write(self.home/'CONFIG.json',{'training_allowed':False,'max_questions':10,'max_calls_per_model':180,
            'slots':[{'root_id':'fixture-root','history_path':str(self.home/'HISTORY.json')}]})
        review={'pool_sha256':ev.sha(self.home/'POOL.json'),'solver_scores_used':False,'questions':[
            {'candidate_id':q['candidate_id'],'root_id':q['root_id'],'source_sha256':q['source_sha256'],
             'core_status':status} for q,status in zip(questions,statuses)]}
        ev.write(self.home/'FINAL_CONTENT_REVIEW.json',review)
        checker=patch.object(ev,'verify_assembly',side_effect=lambda home:ev.read(home/'POOL.json'))
        checker.start();self.addCleanup(checker.stop)
        return pool

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
        review=ev.read(path);review['questions'].pop();path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'cover_every'):ev.freeze_review(self.home)
        review['questions'].append({'candidate_id':'fixture-q1','root_id':'fixture-root','source_sha256':'bad','core_status':'CORE_USABLE'})
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'source_binding'):ev.freeze_review(self.home)

    def test_frozen_review_cannot_change_and_started_blocks_freeze(self):
        self.fixture();ev.freeze_review(self.home)
        path=self.home/'FINAL_CONTENT_REVIEW.json';review=ev.read(path)
        review['questions'][1]['core_status']='CORE_USABLE';path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError,'frozen_review_changed'):ev.verify_review(self.home)
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
