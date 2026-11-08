"""Meaningful negative cases for the new source-first question gates."""
import copy
import unittest
import test_codex_controlled_batch as fixtures
import codex_controlled_batch as c
import codex_batch_questions as q

class QuestionGates(unittest.TestCase):
    setUp=fixtures.ControllerTests.setUp
    tearDown=fixtures.ControllerTests.tearDown
    init=fixtures.ControllerTests.init
    draft=fixtures.ControllerTests.draft
    def ready_assets(self):
        self.init();f=c.root_folder(self.batch,self.root)
        c.save(f/'EXPANSION_DRAFT.json',self.draft());c.materialize(self.batch)
        source=c.read(f/'CUMULATIVE_SOURCE.json');refs=[t['turn_ref'] for t in source['turns']]
        checks={k:{'status':'PASS','reason':'synthetic fixture'} for k in ('time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity')}
        c.save(f/'SOURCE_REVIEW.json',{'source_sha256':c.digest(f/'CUMULATIVE_SOURCE.json'),'full_source_read':True,'turn_refs_read':refs,'checks':checks,'findings':[],'decision':'READY'})
        c.save(f/'ORDERING_ANCHOR_PLAN.json',{'source_sha256':c.digest(f/'CUMULATIVE_SOURCE.json'),'slots':[]})
        c.save(f/'QA_DRAFT.json',{'stem':'Which claims are supported?','facts':[{'fact_id':str(i),'true_text':f'Test true {i}','false_text':f'Test false {i}' if i<4 else None} for i in range(6)],'ordering':[]})
        q.publish_assets(self.batch);assets=c.read(f/'PUBLIC_ASSETS.json')
        header={'source_sha256':c.digest(f/'CUMULATIVE_SOURCE.json'),'public_sha256':c.digest(f/'PUBLIC_ASSETS.json'),'full_source_read':True,'turn_refs_read':refs,'private_answers_read_before_save':False,'independent_of_question_author':True}
        citation={'turn_ref':source['turns'][1]['turn_ref'],'speaker':'Morgan','quote':source['turns'][1]['text']}
        js={}
        for a in assets['statements']:
            relation='supports' if 'true' in a['proposition'] else 'contradicts'
            if a['proposition']=='Test false 3':relation='unknown'
            js[a['statement_id']]={'proposition':a['proposition'],'relation':relation,'scope_valid':True,'fact_distinct':True,'reason':'synthetic human-judgment fixture, not real semantic evidence','citations':[citation]}
        c.save(f/'ASSET_SOURCE_JUDGMENTS.json',{**header,'statements':js,'ordering_questions':[]})
        return f,header,js

    def test_unknown_variant_not_composed_and_gold2_shortfall_retained(self):
        f,_,js=self.ready_assets();q.compose(self.batch)
        receipt=c.read(f/'COMPOSITION_RECEIPT.json')
        self.assertEqual(len(receipt['valid_false_fact_ids']),3)
        self.assertEqual(receipt['unfilled_slots'][0]['intended_cardinality'],2)
        self.assertEqual(receipt['public_count'],4)
        self.assertNotIn('Test false 3',str(c.read(f/'PUBLIC_QUESTIONS.json')))
        self.assertFalse((f/'PRIVATE_GOLD.json').exists())

    def test_final_conflict_excludes_without_rewriting_design_gold(self):
        f,header,js=self.ready_assets();q.compose(self.batch)
        public=c.read(f/'PUBLIC_QUESTIONS.json');by_text={j['proposition']:j for j in js.values()}
        judgments={}
        for item in public['questions']:
            judgments[item['candidate_id']]={'stem_valid':True,'scope_valid':True,'fact_distinct':True,'no_entangled_options':True,'options':{label:copy.deepcopy(by_text[text]) for label,text in item['options'].items()}}
        first=public['questions'][0]['candidate_id'];judgments[first]['options']['A']['relation']='contradicts'
        header['public_sha256']=c.digest(f/'PUBLIC_QUESTIONS.json')
        c.save(f/'PUBLIC_SOURCE_JUDGMENTS.json',{**header,'questions':judgments})
        q.finalize(self.batch)
        gold=c.read(f/'PRIVATE_GOLD.json')['questions'][0]['gold']
        self.assertEqual(gold,list('ABCDEF'))
        self.assertEqual(c.read(f/'FINAL_CROSSCHECK.json')['results'][0]['status'],'EXCLUDED')

    def test_incomplete_final_review_never_creates_gold(self):
        f,header,_=self.ready_assets();q.compose(self.batch)
        header['public_sha256']=c.digest(f/'PUBLIC_QUESTIONS.json')
        c.save(f/'PUBLIC_SOURCE_JUDGMENTS.json',{**header,'questions':{}})
        with self.assertRaises(ValueError):q.finalize(self.batch)
        self.assertFalse((f/'PRIVATE_GOLD.json').exists())

    def test_ordering_rejects_multi_event_and_reversing_semantic_match(self):
        source={'turns':[{'turn_ref':str(i),'scene_ref':'s','text':f'unique {i}'} for i in range(7)]}
        question={'items':{f'I{i}':f'claim {i}' for i in range(1,7)}}
        j={'stem_valid':True,'scope_valid':True,'coherent':True,'unique_order':True,'neutral_stem_ok':True,'derived_order':list(question['items']),'alternative_orders':[],'items':{}}
        for i in range(1,7):
            j['items'][f'I{i}']={'proposition':f'claim {i}','supported':True,'actor_ok':True,'modality_ok':True,'state_ok':True,'neutral_stem_ok':True,'single_event_ok':True,'order_anchor':{'turn_ref':str(i),'quote':f'unique {i}'},'support_refs':[str(i)],'alternative_matching_anchors':[],'reason':'test'}
        self.assertEqual(q.check_ordering(question,j,source,'s')[0],[])
        j['items']['I1']['single_event_ok']=False
        self.assertIn('I1:single_event_ok_failed',q.check_ordering(question,j,source,'s')[0])
        j['items']['I6']['alternative_matching_anchors']=[{'turn_ref':'0','quote':'unique 0'}]
        self.assertIn('semantic_order_not_unique',q.check_ordering(question,j,source,'s')[0])

if __name__=='__main__':unittest.main()
