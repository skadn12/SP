"""Synthetic CPU fixtures only: no current input seal, model, tokenizer, or API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import types
import sys

import adapter as a


class CaptureTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps({'messages': messages, 'kwargs': kwargs}, ensure_ascii=False)


class AdapterFixtures(unittest.TestCase):
    def fixture(self, qtype, directory, k=4):
        root = 'synthetic-root'
        source = Path(directory) / 'SOURCE.json'
        if not source.exists():
            a.write_new(source, {'root_id': root, 'turns': [
                {'turn_ref': 't1', 'speaker': 'Synthetic A', 'text': 'First, alpha; then beta; then gamma. Delta is unconstrained.'},
                {'turn_ref': 't2', 'speaker': 'Synthetic B', 'text': 'This is the complete synthetic source.'}]})
        stable = ['stable_' + str(i) for i in range(k)]
        candidate = {'candidate_id': 'fixture_' + qtype, 'root_id': root, 'question_type': qtype,
                     'question': 'Answer the synthetic inventory or relation question.',
                     'options': [{'option_id': oid, 'text': 'Synthetic distinct item ' + str(i)} for i, oid in enumerate(stable)],
                     'gold_option_ids': stable[:2] if qtype=='multi_select' else stable if qtype=='ordering' else [],
                     'source_path': str(source), 'source_sha256': a.bind(source)['sha256'], 'source_turn_count': 2}
        if qtype=='partial_order':
            pairs = [[stable[0], stable[1]]] if k==2 else [[stable[0], stable[1]], [stable[1], stable[2]], [stable[0], stable[2]]]
            candidate.update(gold_pairs=pairs, query_pairs=[[x,y] for x in stable for y in stable if x!=y],
                             query_scope={'relation': 'strict temporal precedence', 'items': 'all displayed items'},
                             answer_semantics='all_and_only_entailed_pairs_in_query_scope')
        record = {'candidate_id': candidate['candidate_id'], 'root_id': root, 'group': 'NEW_G',
                  'source': {**a.bind(source), 'turn_count': 2}, 'protocol_markers': [],
                  'structure_result': {'independent': True, 'status': 'D', 'actual_structure_class': 'D'},
                  'overlap_result': {'independent': True, 'status': 'HISTORICAL_DUPLICATE', 'historical_duplicate': True},
                  'quality_status': 'PASS', 'gold_agreement': True}
        return candidate, record

    def request(self, item, variant):
        k = len(item['options'])
        request = {**variant, 'question_type': item['question_type'], 'gold_option_ids': item['gold_option_ids'],
                   'context_mode': 'full_context', 'n': k if item['question_type']=='multi_select' else k*(k-1),
                   'g': len(item['gold_option_ids']) if item['question_type']=='multi_select' else len(item.get('gold_pairs', [])) if item['question_type']=='partial_order' else k*(k-1)//2}
        if item['question_type']=='partial_order':
            request['query_pairs'] = item['query_pairs']
        return request

    def raw(self, request, answer, malformed_aux=False):
        evidence = {label: {'judgment': 'supported', 'turn_refs': ['t1'], 'reason': 'Synthetic evidence.'}
                    for label in request['display_to_option_id']}
        obj = {'item_evidence': evidence, 'answer_reason': 'Synthetic answer.', 'final_answer': '<box>' + answer + '</box>'}
        if malformed_aux:
            del obj['item_evidence']
        return {'status': 'RETURNED', 'finish_reason': 'stop', 'raw_prediction': json.dumps(obj)}

    def answer(self, request):
        if request['question_type']=='partial_order':
            return ';'.join('<'.join(p) for p in request['gold_labels'])
        return ' '.join(request['gold_labels'])

    def checked(self, raw, request):
        return a.validate(raw, request, a.schema_for(list(request['display_to_option_id']), ['t1','t2'], request['question_type']))

    def test_permutations_gold_independence_and_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering','partial_order'):
                candidate, record = self.fixture(qtype, directory)
                item = a.normalize(candidate, record)
                shown = a.variants(item)
                changed = copy.deepcopy(item)
                if qtype=='partial_order':
                    changed['gold_pairs'] = []
                elif qtype=='multi_select':
                    changed['gold_option_ids'] = list(item['label_to_option_id'].values())[2:]
                else:
                    changed['gold_option_ids'].reverse()
                self.assertEqual([v['display_to_option_id'] for v in shown], [v['display_to_option_id'] for v in a.variants(changed)])
                self.assertEqual(len(shown), 6)
                self.assertEqual(len({tuple(v['display_to_option_id'].values()) for v in shown}), 6)
                for variant in shown:
                    request = self.request(item, variant)
                    result = self.checked(self.raw(request, self.answer(request)), request)
                    self.assertTrue(result['final_answer_valid'], result)
                    self.assertTrue(result['auxiliary_protocol_valid'], result)
                    self.assertTrue(result['answer_only_score']['exact_correct'])
                small, small_record = self.fixture(qtype, directory, 2)
                if qtype=='multi_select':
                    small['gold_option_ids'] = small['gold_option_ids'][:1]
                self.assertEqual(len(a.variants(a.normalize(small, small_record))), 2)

    def test_wrong_answers_and_partial_missing_transitive_edge(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering'):
                c, r = self.fixture(qtype, directory)
                item = a.normalize(c,r)
                request = self.request(item,a.variants(item)[0])
                wrong = [x for x in request['display_to_option_id'] if x not in request['gold_labels']] if qtype=='multi_select' else request['gold_labels'][::-1]
                result = self.checked(self.raw(request,' '.join(wrong)),request)
                self.assertEqual(result['answer_only_score']['J'], -1)
            c,r = self.fixture('partial_order',directory)
            item = a.normalize(c,r)
            request = self.request(item,a.variants(item)[0])
            incomplete = ';'.join('<'.join(p) for p in request['gold_labels'][:2])
            result = self.checked(self.raw(request,incomplete),request)['answer_only_score']
            self.assertFalse(result['exact_correct'])
            self.assertEqual(result['partial_score'], .8)
            self.assertNotIn('J',result)
            self.assertEqual(len(result['predicted_option_ids']),2)
            c['gold_pairs'] = c['gold_pairs'][:2]
            with self.assertRaisesRegex(AssertionError,'TRANSITIVE'):
                a.normalize(c,r)

    def test_auxiliary_format_separation_and_invalid_labels(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering','partial_order'):
                c,r = self.fixture(qtype,directory)
                item=a.normalize(c,r);request=self.request(item,a.variants(item)[0])
                result=self.checked(self.raw(request,self.answer(request),True),request)
                self.assertTrue(result['final_answer_valid'])
                self.assertTrue(result['answer_only_score']['exact_correct'])
                self.assertFalse(result['auxiliary_protocol_valid'])
                bad = self.answer(request)
                bad = bad+';'+bad.split(';')[0] if qtype=='partial_order' else bad+' '+bad.split(' ')[0]
                self.assertFalse(self.checked(self.raw(request,bad),request)['final_answer_valid'])
                self.assertFalse(self.checked(self.raw(request,'INVALID_LABEL'),request)['final_answer_valid'])
                raw=self.raw(request,self.answer(request))
                raw['raw_prediction']='{"final_answer":"<box></box>","final_answer":"<box></box>"}'
                self.assertFalse(self.checked(raw,request)['final_answer_valid'])
                raw=self.raw(request,self.answer(request));raw['finish_reason']='length'
                self.assertFalse(self.checked(raw,request)['final_answer_valid'])

    def test_boundaries_and_four_group_proxy(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            c,r=self.fixture('multi_select',directory)
            for gold in ([c['options'][0]['option_id']], [o['option_id'] for o in c['options']]):
                c['gold_option_ids']=gold
                item=a.normalize(c,r)
                source={**r['source']}
                for group in a.GROUPS:
                    item['arm']=item['group']=group
                    checked,turns=a.legacy_check(item,source)
                    self.assertEqual(checked['arm'],group);self.assertEqual(len(turns),2)
                request=self.request(item,a.variants(item)[0])
                self.assertEqual(self.checked(self.raw(request,self.answer(request)),request)['answer_only_score']['J'],1)
            c['gold_option_ids']=[]
            with self.assertRaisesRegex(AssertionError,'UNDEFINED_EMPTY_GOLD_J'):
                a.normalize(c,r)
            c,r=self.fixture('partial_order',directory);c['gold_pairs']=[]
            item=a.normalize(c,r)
            checked,_=a.legacy_check(item,r['source'])
            self.assertEqual(checked['gold_option_ids'],[])
            request=self.request(item,a.variants(item)[0])
            result=self.checked(self.raw(request,''),request)['answer_only_score']
            self.assertTrue(result['exact_correct']);self.assertEqual(result['partial_score'],0)

    def test_prompt_full_k_scope_no_hidden_fields(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering','partial_order'):
                c,r=self.fixture(qtype,directory)
                c['author_certificate']='HIDDEN_CERTIFICATE';c['evidence']=[{'quote':'HIDDEN_EVIDENCE'}]
                item=a.normalize(c,r);variant=a.variants(item)[0]
                turns=a.read(r['source']['path'])['turns']
                prompt=a.prompt_for(CaptureTokenizer(),item,variant,turns)
                for turn in turns:self.assertIn(turn['text'],prompt)
                self.assertNotIn('HIDDEN_CERTIFICATE',prompt);self.assertNotIn('HIDDEN_EVIDENCE',prompt)
                self.assertNotIn('gold_option_ids',prompt);self.assertNotIn('historical_duplicate',prompt)
                if qtype=='ordering':
                    self.assertIn('<box>I1 I2 I3 I4</box>',prompt)
                    self.assertIn('all 4 displayed labels exactly once',prompt)
                if qtype=='partial_order':
                    self.assertIn('query_pairs',prompt);self.assertIn('transitive',prompt);self.assertIn('unconstrained',prompt)
                no_context=a.prompt_for(CaptureTokenizer(),item,variant,[])
                self.assertNotIn(turns[0]['text'],no_context)
                self.assertIn('empty []',no_context)

    def test_prompt_exposes_only_permuted_display_and_remapped_pair_domain(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering','partial_order'):
                c,r=self.fixture(qtype,directory)
                original=copy.deepcopy(c)
                item=a.normalize(c,r)
                stable=list(item['label_to_option_id'].values())
                shown=a.variants(item)
                variant=next(v for v in shown if list(v['display_to_option_id'].values())!=stable)
                prompt=json.loads(a.prompt_for(CaptureTokenizer(),item,variant,[]))['messages'][-1]['content']
                self.assertNotIn('Original option_id',prompt)
                self.assertNotIn('displayed-label mapping',prompt)
                for oid in stable:self.assertNotIn(oid,prompt)
                displayed=prompt.split('Displayed items:\n',1)[1].split('\n\n',1)[0]
                self.assertEqual(displayed,'\n'.join(o['id']+'. '+o['text'] for o in variant['options']))
                self.assertIn(c['question'],prompt)
                if qtype=='partial_order':
                    fields=json.loads(prompt.split('Original visible scope/instructions (query pair endpoints use displayed labels):\n',1)[1].split('\n\n',1)[0])
                    inverse={oid:label for label,oid in variant['display_to_option_id'].items()}
                    self.assertEqual(fields['query_pairs'],[[inverse[left],inverse[right]] for left,right in c['query_pairs']])
                self.assertEqual(c,original)

    def test_stable_id_prose_reference_hold_and_articles_do_not_hold(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            c,r=self.fixture('multi_select',directory)
            for field in ('question','query_scope'):
                changed=copy.deepcopy(c)
                changed[field]='Compare stable_0 with the other displayed item.'
                with self.assertRaisesRegex(AssertionError,'EVAL_HOLD_STABLE_ID_REFERENCE'):
                    a.normalize(changed,r)
            changed=copy.deepcopy(c);changed['options'][0]['text']='The next item follows stable_1.'
            with self.assertRaisesRegex(AssertionError,'EVAL_HOLD_STABLE_ID_REFERENCE'):
                a.normalize(changed,r)
            original_ids=['A','B','C','D']
            c['options']=[{'option_id':oid,'text':text} for oid,text in zip(original_ids,
                ['A blanket','A portable radio','A first-aid kit','Bottled water'])]
            c['gold_option_ids']=['A','D']
            c['question']='A neighbour asks what was brought; which supplies were reported?'
            item=a.normalize(c,r)
            self.assertIn('A blanket',a.prompt_for(CaptureTokenizer(),item,a.variants(item)[0],[]))
            a.check_visible_id_references('I brought a blanket.', ['A blanket'], {}, ['I','A','B'])
            for question in ('Does option A precede option B?','Which of A and B was reported?','Does A<B hold?','Does [A] qualify?'):
                changed=copy.deepcopy(c);changed['question']=question
                with self.assertRaisesRegex(AssertionError,'EVAL_HOLD_STABLE_ID_REFERENCE'):
                    a.normalize(changed,r)
            held=copy.deepcopy(c);held['question']='Does option A qualify?'
            draft=Path(directory)/'REFERENCE_DRAFT.json';a.write_new(draft,held)
            review=Path(directory)/'REFERENCE_REVIEW.json';a.write_new(review,{'synthetic':True})
            r.update(raw_draft=a.bind(draft),review_artifacts=[a.bind(review)])
            seal={'schema_version':'structure_pilot_input_review_seal_v1','namespace':a.NAMESPACE,'frozen':True,
                  'frozen_at_utc':a.now(),'reviews_frozen':dict.fromkeys(('quality','gold','structure','overlap'),True),
                  'files':[a.bind(draft),a.bind(review),a.bind(r['source']['path'])],'items':[r]}
            path=Path(directory)/'REFERENCE_SEAL.json';a.write_new(path,seal)
            items,holds,_,_=a.check_inputs(path,verify_model=False)
            self.assertEqual(items,[])
            self.assertEqual(holds[0]['status'],'EVAL_HOLD')
            self.assertEqual(holds[0]['quality_status'],'PASS')
            self.assertEqual(holds[0]['reason'],'EVAL_HOLD_STABLE_ID_REFERENCE_IN_VISIBLE_PROSE')

    def test_seal_d_and_overlap_not_quality_exclusions_and_tamper(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            c,r=self.fixture('multi_select',directory)
            draft=Path(directory)/'DRAFT.json';a.write_new(draft,{'candidates':[c]})
            review=Path(directory)/'REVIEW.json';a.write_new(review,{'synthetic':True})
            r.update(raw_draft=a.bind(draft),review_artifacts=[a.bind(review)])
            seal={'schema_version':'structure_pilot_input_review_seal_v1','namespace':a.NAMESPACE,'frozen':True,
                  'frozen_at_utc':a.now(),'reviews_frozen':dict.fromkeys(('quality','gold','structure','overlap'),True),
                  'files':[a.bind(draft),a.bind(review),a.bind(r['source']['path'])],'items':[r]}
            path=Path(directory)/'SEAL.json';a.write_new(path,seal)
            items,holds,_,_=a.check_inputs(path,verify_model=False)
            self.assertEqual(len(items),1);self.assertEqual(holds,[])
            duplicate=copy.deepcopy(r);duplicate['candidate_id']='other_fixture';seal['items'].append(duplicate)
            path2=Path(directory)/'DUPLICATE.json';a.write_new(path2,seal)
            with self.assertRaisesRegex(AssertionError,'DUPLICATE_ROOT_GROUP'):
                a.check_inputs(path2,verify_model=False)
            before=a.bind(review);review.write_text('{"synthetic":false}')
            with self.assertRaisesRegex(AssertionError,'FROZEN_INPUT_CHANGED'):
                a.require_binding(before)

    def test_partial_domain_and_missing_denominator_and_one_shot_ledger(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            c,r=self.fixture('partial_order',directory)
            c['query_pairs']=[c['gold_pairs'][0]];c['gold_pairs']=[c['gold_pairs'][0]]
            item=a.normalize(c,r);request=self.request(item,a.variants(item)[0])
            first,second=request['gold_labels'][0]
            self.assertFalse(self.checked(self.raw(request,second+'<'+first),request)['final_answer_valid'])
            ledger=Path(directory)/'LEDGER.json';a.reserve(ledger,36,['synthetic'])
            with self.assertRaises(FileExistsError):a.reserve(ledger,36,['synthetic'])
            output=Path(directory)/'output';output.mkdir()
            a.write_new(output/'REQUESTS.json',[{'request_id':'r1'},{'request_id':'r2'}])
            a.write_new(output/'raw/r1.json',{'request_id':'r1','status':'RETURNED','raw_prediction':'bad','finish_reason':'stop'})
            with patch.object(a,'OUTPUT',output):
                raws=a.collect_raws()
            self.assertEqual([r['status'] for r in raws],['RETURNED','MISSING'])
            summary=a.summarize([{'execution_status':'RETURNED','valid':False,'exact_correct':False,'question_type':'ordering','J':0},
                                 {'execution_status':'MISSING','valid':False,'exact_correct':False,'question_type':'ordering','J':0}])
            self.assertEqual(summary['executed_returned_calls'],1)
            self.assertEqual(summary['missing_calls'],1)
            self.assertEqual(summary['mean_J_executed'],0)
            self.assertIsNone(summary['planned_accuracy'])

    def test_actual_preflight_roundtrip_and_mocked_backend_interruption(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            for qtype in ('multi_select','ordering','partial_order'):
                c,r=self.fixture(qtype,directory)
                item=a.normalize(c,r);request=self.request(item,a.variants(item)[0])
                schema=a.schema_for(list(request['display_to_option_id']),['t1','t2'],qtype)
                if qtype=='partial_order':request['gold_pairs']=item['gold_pairs']
                a.preflight_answer_check(request,schema)
            output=Path(directory)/'output';output.mkdir()
            requests=[]
            for index in range(2):
                prompt=b'Synthetic prompt, no model.';schema=b'{}\n'
                pp,sp='prompts/r'+str(index)+'.txt','schemas/r'+str(index)+'.json'
                a.write_new(output/pp,prompt);a.write_new(output/sp,schema)
                requests.append({'request_id':'r'+str(index),'prompt_path':pp,'schema_path':sp,
                                 'prompt_sha256':a.digest(prompt),'schema_sha256':a.digest(schema),'sampling_seed':index})
            a.write_new(output/'REQUESTS.json',requests)
            a.write_new(output/'PREFLIGHT.json',{'status':'PASS','requests_sha256':a.bind(output/'REQUESTS.json')['sha256']})
            a.write_new(output/'INPUT_BINDINGS.json',{'files':[]})
            class FakeLLM:
                def __init__(self,**kwargs):pass
                def generate(self,*args,**kwargs):
                    completion=types.SimpleNamespace(text='synthetic returned text',finish_reason='stop',stop_reason=None,token_ids=[])
                    return [types.SimpleNamespace(outputs=[completion])]
            fake=types.ModuleType('vllm');fake.__version__='synthetic-only';fake.LLM=FakeLLM
            fake.SamplingParams=lambda **kwargs:kwargs
            sampling=types.ModuleType('vllm.sampling_params');sampling.StructuredOutputsParams=lambda **kwargs:kwargs
            with patch.object(a,'OUTPUT',output), patch.dict(sys.modules,{'vllm':fake,'vllm.sampling_params':sampling}):
                h=a.configure_legacy_run()
                with self.assertRaisesRegex(AssertionError,'INCOMPLETE_BACKEND_BATCH'):h.run()
                self.assertEqual(a.read(output/'raw/r0.json')['raw_prediction'],'synthetic returned text')
                self.assertFalse((output/'raw/r1.json').exists())
                with self.assertRaisesRegex(AssertionError,'NO_RETRY'):h.run()

    def test_synthetic_prepare_holds_overlength_without_truncation(self):
        with tempfile.TemporaryDirectory(dir=a.HERE) as directory:
            c,r=self.fixture('multi_select',directory)
            item=a.normalize(c,r);turns=a.read(r['source']['path'])['turns']
            long_item=copy.deepcopy(item);long_item['candidate_id']='synthetic_overlength'
            long_item['question']='SYNTHETIC_OVERLENGTH_MARKER'
            class FakeTokenizer(CaptureTokenizer):
                def __call__(self,prompt,**kwargs):
                    return types.SimpleNamespace(input_ids=[0]*(20000 if 'SYNTHETIC_OVERLENGTH_MARKER' in prompt else 100))
            fake=types.ModuleType('transformers')
            fake.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args,**kwargs:FakeTokenizer())
            root=Path(directory)/'synthetic_prepare';root.mkdir()
            identity={'model':'SYNTHETIC_MODEL_NEVER_LOADED','tokenizer':'SYNTHETIC_TOKENIZER'}
            with patch.object(a,'HERE',root), patch.object(a,'OUTPUT',root/'evaluation'), \
                 patch.object(a,'check_inputs',return_value=([(item,turns),(long_item,turns)],[],[],identity)), \
                 patch.dict(sys.modules,{'transformers':fake}):
                result=a.prepare(Path(directory)/'UNUSED_SYNTHETIC_SEAL')
                self.assertEqual(result['request_count'],36)
                self.assertEqual(result['evaluated_candidate_count'],1)
                self.assertFalse(result['source_truncated'])
                self.assertEqual(result['holds'][0]['candidate_id'],'synthetic_overlength')
                self.assertEqual(result['holds'][0]['status'],'EVAL_HOLD')
                self.assertEqual(result['holds'][0]['quality_status'],'PASS')
                requests=a.read(root/'evaluation/REQUESTS.json')
                for request in requests:
                    if request['context_mode']=='full_context':
                        prompt=(root/'evaluation'/request['prompt_path']).read_text()
                        self.assertIn(turns[-1]['text'],prompt)
                with self.assertRaisesRegex(AssertionError,'NO_PREPARE_RETRY'):
                    a.prepare()


if __name__=='__main__':
    unittest.main(verbosity=2)
