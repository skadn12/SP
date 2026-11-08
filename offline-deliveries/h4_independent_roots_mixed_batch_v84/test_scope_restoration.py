"""Scope must follow the public stem; full source still supplies context."""
from copy import deepcopy
import unittest
import ordering_review as review
from common import load_ordering_gate

class PublicScopeTests(unittest.TestCase):
    def fixture(self, restriction):
        source = {'root_id':'r', 'turns':[{'turn_ref':f'CANDIDATE:A:S01:T{i:03d}', 'speaker':'Actor', 'text':f'fact {i}'} for i in range(1,7)] + [{'turn_ref':'CANDIDATE:A:S02:T001','speaker':'Actor','text':'fact 2'}]}
        public = {'candidate_id':'q','root_id':'r','question_type':'ordering','chronology_kind':'conversation_order','question':'Order the statements as they APPEAR IN THE SUPPLIED DIALOGUE' + restriction,'items':{f'I{i}':f'fact {i}' for i in range(1,7)},'source_path':'/tmp/source.json','source_sha256':'0'*64}
        row = deepcopy(public)
        row.update(complete_source_turns_read=7, coherent=True, unique_order=True, neutral_stem_ok=True, alternative_orders=[])
        row['items']={f'I{i}':{'proposition':f'fact {i}', **{k:True for k in review.ITEM_BOOLEANS},'order_anchor':{'turn_ref':f'CANDIDATE:A:S01:T{i:03d}','quote':f'fact {i}'},'support_refs':[f'CANDIDATE:A:S01:T{i:03d}'],'alternative_matching_anchors':[]} for i in range(1,7)}
        row['items']['I2']['alternative_matching_anchors']=[{'turn_ref':'CANDIDATE:A:S02:T001','quote':'fact 2'}]
        return public,row,source

    def test_outside_explicit_scene_repeat_does_not_change_answer(self):
        pub,row,source=self.fixture(' within the CANDIDATE:A:S01 scene.')
        result=review._review_row(row,pub,source,load_ordering_gate())
        self.assertEqual(result['reasons'],[])
        self.assertEqual(result['ordering_match_scope']['context_turn_refs'],[t['turn_ref'] for t in source['turns']])
        self.assertEqual(len(result['ordering_match_scope']['matching_turn_refs']),6)
        self.assertEqual(len(result['out_of_scope_alternative_anchor_spans']['I2']),1)

    def test_same_repeat_inside_public_scene_union_excludes(self):
        pub,row,source=self.fixture(' within the CANDIDATE:A:S01 and CANDIDATE:A:S02 scenes.')
        result=review._review_row(row,pub,source,load_ordering_gate())
        self.assertTrue(any('alternative_anchor_changes_or_ties_order' in x for x in result['reasons']))

    def test_no_explicit_scope_keeps_full_dialogue_and_excludes(self):
        pub,row,source=self.fixture('.')
        result=review._review_row(row,pub,source,load_ordering_gate())
        self.assertEqual(result['ordering_match_scope']['kind'],'full_dialogue')
        self.assertTrue(result['reasons'])

    def test_implicit_topic_or_unknown_scene_cannot_shrink_scope(self):
        for restriction in [' about the first conversation.',' within the imaginary scene.']:
            pub,row,source=self.fixture(restriction)
            result=review._review_row(row,pub,source,load_ordering_gate())
            self.assertEqual(result['ordering_match_scope']['kind'],'full_dialogue')
            self.assertTrue(result['reasons'])

if __name__=='__main__':unittest.main()
