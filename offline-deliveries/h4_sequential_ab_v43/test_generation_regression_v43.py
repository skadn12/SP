"""Known live failures remain errors even with improved prompt generation."""
from pathlib import Path
import unittest
import candidate_runner as c

class HistoricalGenerationTests(unittest.TestCase):
    def test_wrong_scene_id_also_reports_full_history_copy(self):
        old=c.HERE.parent/'h4_sequential_ab_v41/runs/fresh_b2_candidates_01'
        history=c.r.base.read(old/'FORMAL_BASELINE.json')['turns']
        card=c.r.base.read(old/'extractions/01.json')['card']
        raw=c.r.base.read(old/'requests/006-cc-train-episode-104814-B-S01-realizer-1/RESPONSE.json')['text']
        obj,failures,warnings,duplicate=c.inspect_realization(raw,c.identity(c.ROOT,c.r.ROUTES[1],'S01'),history,history,card)
        self.assertEqual(failures,['dialogue_identity_mismatch'])
        self.assertEqual(duplicate['status'],'REJECT')
        self.assertEqual(duplicate['duplicate_turn_count'],12)
        self.assertEqual(obj['scene_id'],'S02')
        decision=c.contract.decide_candidate(mechanical_failures=failures,duplicate_result=duplicate)
        self.assertEqual(decision['disposition'],'hard_reject')

    def test_semantic_risk_is_not_programmatically_accepted_by_clean_mechanics(self):
        prior=c.HERE.parent/'h4_sequential_ab_v41'
        history=c.r.base.read(prior/'runs/fresh_b2_candidates_01/FORMAL_BASELINE.json')['turns']
        card=c.r.base.read(prior/'runs/fresh_b2_candidates_01/extractions/01.json')['card']
        raw=c.r.base.read(next((prior/'diagnostics/prompt_tail_b2_s01_01').glob('requests/*/RESPONSE.json')))['text']
        _,failures,_,duplicate=c.inspect_realization(raw,c.identity(c.ROOT,c.r.ROUTES[1],'S01'),history,history,card)
        self.assertEqual(failures,[])
        self.assertEqual(duplicate['status'],'PASS')
        decision=c.contract.decide_candidate(duplicate_result=duplicate)
        self.assertEqual(decision['disposition'],'retained_pending_review')
        self.assertFalse(decision['semantic_certified'])
        self.assertFalse(decision['training_pool_eligible'])

if __name__=='__main__':unittest.main()
