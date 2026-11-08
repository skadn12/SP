"""Source/ordering isolation and frozen empirical context contract regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import feedback as f
from ..h4_batch30_memory_event_k3_probe_v1.qa_generation_only import _load_feedback_context


class EligibilityTests(unittest.TestCase):
    def test_ordering_is_removed_before_any_safe_frontier_or_rank_read(self):
        poison={'question_type':'ordering','root_id':f.ROOTS[0], 'candidate_id':'ordering-sentinel',
                'models':None,'flags':None,'gap_type':'POISON_DERIVED_GUIDANCE'}
        with patch.object(f.original,'_safe_frontier',side_effect=AssertionError('ordering reached original safe')):
            self.assertFalse(f.eligibility(poison,set())[0])
        with patch.object(f.original,'_frontier_rank',side_effect=AssertionError('ordering reached rank')):
            self.assertEqual(f.selected_rows([poison],set()),{root:[] for root in f.ROOTS})

    def test_unknown_scope_fails_before_reading_metrics(self):
        for row in ({'question_type':'single_choice','split':'dev','root_id':f.ROOTS[0]},
                    {'question_type':'multi_select','split':'train','root_id':'another-root'}):
            self.assertFalse(f.eligibility(row,set())[0])

    def test_rank_and_confirmed_defect_fallback_reuse_original_rules(self):
        transitions=list(f.jsonl(f.COMPILER/'candidate_primary_transition.jsonl'))
        expected=f.selected_rows(transitions,set())
        for root,rows in expected.items():
            self.assertEqual(rows,sorted(rows,key=f.original._frontier_rank))
            self.assertTrue(all(f.original._safe_frontier(row,set())[0] for row in rows))
            removed=rows[0]['candidate_id']
            self.assertEqual(f.selected_rows(transitions,{removed})[root],rows[1:])


class FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_preflight_report.json')

    def test_all_six_loader_contexts_match_original_empirical_template(self):
        template=f.legacy.TEMPLATE.read_text()
        miner=f.legacy.template_block(template,'## Intent Miner 附加段（intent_suffix）')
        writer=f.legacy.template_block(template,'## QA Writer 附加段（slot_suffixes）')
        for root in f.ROOTS:
            off=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_off/feedback_context_manifest.json')['roots'][root]
            on=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_on/feedback_context_manifest.json')['roots'][root]
            self.assertEqual(_load_feedback_context(off,root),{'intent_suffix':'','slot_suffixes':['','','']})
            context=_load_feedback_context(on,root)
            packet=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_on'/root/'feedback_packet.json')
            self.assertIn(packet['historical_qa']['question_type'],f.ALLOWED)
            before,after=miner.split('{INTENT_FEEDBACK_JSON}')
            self.assertTrue(context['intent_suffix'].startswith(before))
            self.assertTrue(context['intent_suffix'].endswith(after))
            embedded=context['intent_suffix'][len(before):len(context['intent_suffix'])-len(after) if after else None]
            self.assertEqual(json.loads(embedded),packet)
            self.assertEqual(context['slot_suffixes'],[writer]*3)
            off_packet=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_off'/root/'feedback_packet.json')
            self.assertEqual(off_packet['feedback_status'],'NO_ACTIVE_FEEDBACK')
            self.assertEqual(off_packet['solver_observations'],{})
            self.assertEqual(set(context),{'intent_suffix','slot_suffixes'})

    def test_raw_request_observations_include_all_three_models_and_contexts(self):
        for root in f.ROOTS:
            packet=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_on'/root/'feedback_packet.json')
            self.assertEqual(set(packet['solver_observations']),set(f.legacy.MODELS))
            request_keys=[]
            for model in f.legacy.MODELS:
                result=f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_on'/root/'historical_solver_requests'/f'{model}.json')
                observed=packet['solver_observations'][model]['by_context']
                self.assertEqual(set(observed),{'full_context','no_context'})
                for context,data in observed.items():
                    rows=[x for x in result['requests'] if x['context_mode']==context]
                    self.assertEqual(data['scored_count'],len(rows))
                    self.assertEqual(data['correct_count'],sum(x['correct'] is True for x in rows))
                    self.assertEqual(sum(x['count'] for x in data['canonical_prediction_counts']),len(rows))
                    self.assertTrue(all(x['canonical_gold']==packet['historical_qa']['gold'] for x in rows))
                request_keys.append({(x['request_id'],x['context_mode'],x['variant_id'],x['slot_id'],x['seed']) for x in result['requests']})
            self.assertTrue(all(x==request_keys[0] for x in request_keys))

    def test_all_history_preserved_without_prompt_or_second_gate(self):
        history=f.legacy.load(f.DEFAULT_OUTPUT/'history_preservation.json')
        rows=list(f.jsonl(history['history']['path']))
        expected={x['candidate_id'] for x in f.jsonl(f.COMPILER/'candidate_primary_transition.jsonl')
                  if x['root_id'] in f.ROOTS and x['question_type']=='ordering'}
        self.assertEqual({x['candidate_id'] for x in rows},expected)
        self.assertFalse(history['delivered_to_feedback_off'])
        self.assertFalse(history['delivered_to_feedback_on'])
        self.assertFalse(history['second_generation_dedup_gate_added'])
        self.assertTrue(all(x['stem'] and x['solver_prompt_fingerprint'] and not x['model_visible'] for x in rows))

    def test_planner_old_derived_guidance_removed_as_whole(self):
        audit=f.legacy.load(f.DEFAULT_OUTPUT/'planner_feedback_preview/manifest.json')
        self.assertEqual(len(audit['roots']),6)
        for item in audit['roots']:
            self.assertTrue(item['old_derived_performance_removed_entirely'])
            preview=f.legacy.load(item['preview']['path'])
            text=f.legacy.canonical(preview['feedback_view'])
            self.assertNotIn(item['excluded_old_seed'],text)
            for forbidden in ('coverage_gaps','gap_type','global_priority','desired_event_operator'):
                self.assertNotIn(forbidden,text)
            self.assertEqual(preview['feedback_view']['historical_qa']['question_type'],
                             f.legacy.load(f.DEFAULT_OUTPUT/'inputs/feedback_on'/item['root_id']/'feedback_packet.json')['historical_qa']['question_type'])

    def test_selection_excludes_confirmed_defects_and_retains_ambiguity(self):
        rows=list(f.jsonl(f.DEFAULT_OUTPUT/'feedback_selection_audit.jsonl'))
        selected=[x for x in rows if x['selected']]
        self.assertEqual({x['root_id'] for x in selected},set(f.ROOTS))
        defects={x['candidate_id'] for x in f.manual_reviews() if x['confirmed_source_defect']}
        self.assertFalse(defects & {x['candidate_id'] for x in selected})
        self.assertEqual(len(defects),4)
        for x in rows:
            if x['candidate_id'] in defects:
                self.assertIn('confirmed_defect',x['exclusion_reasons'])
                self.assertIsNotNone(x['rank_before_manual_confirmed_defects'])
        self.assertTrue(any(x['manual_review']['review_status']=='REVIEWED_WITH_UNRESOLVED_AMBIGUITY' for x in selected))

    def test_source_hash_drift_and_output_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.json';p.write_text('{}')
            source=f.legacy.source(p)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'frozen_source_sha_mismatch'):
                f.legacy.source(p,source['sha256'])
            out=Path(folder)/'out.json';first=f.legacy.write_json(out,{'a':1})
            self.assertEqual(first,f.legacy.write_json(out,{'a':1}))
            with self.assertRaisesRegex(ValueError,'refuse_to_replace_frozen_output'):
                f.legacy.write_json(out,{'a':2})

if __name__=='__main__':unittest.main()
