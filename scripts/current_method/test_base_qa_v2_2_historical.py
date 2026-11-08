"""Read-only historical regressions for V2.2's hard-target boundary.

These tests reuse frozen QA and retrospective regression annotations. They do
not generate QA, certify material admission, perform a new semantic review, or
measure model difficulty. Historical PASS is not a hard-QA label.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    from . import base_qa_pipeline_v2 as old
    from . import base_qa_pipeline_v2_2 as qa
except ImportError:
    import base_qa_pipeline_v2 as old
    import base_qa_pipeline_v2_2 as qa

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research-protocols/selfplay_week_20261019'
RUN = BASE / 'base_qa_generation_v2_2_20261102'
FIXTURES = BASE / 'base_qa_generation_v2_20261031/HISTORICAL_RELATION_FIXTURES.json'


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controlled_quantity_card(*, separate_identity=False):
    """Same simple subtraction under two labels; never an admitted source."""
    source = dict(root_id='v22-controlled-quantity', source_id='offline-fixture-only', turns=[
        dict(turn_ref='T1', speaker='Jo', scene_ref='S1', text='For consignment C4, 24 bottles were allocated.'),
        dict(turn_ref='T2', speaker='Jo', scene_ref='S2', text=(
            'Reserve six bottles for the speakers.' if separate_identity else
            'Of those C4 bottles, reserve six for the speakers.')),
        dict(turn_ref='T3', speaker='Jo', scene_ref='S3', text=(
            'That reservation is from consignment C4.' if separate_identity else
            'Separate consignment C9 is ready.')),
    ])
    def evidence(ref):
        turn = next(t for t in source['turns'] if t['turn_ref'] == ref)
        return [dict(turn_ref=ref, speaker=turn['speaker'], quote=turn['text'])]
    def fact(nid, ref, claim):
        return dict(node_id=nid, type='fact', event_id='C4', claim=claim,
                    supports=[dict(turn_refs=[ref], evidence=evidence(ref),
                                   entailment_rationale='Controlled premise for the numeric count.')])
    bridge_ref = 'T3' if separate_identity else 'T2'
    operation = dict(node_id='answer', type='operation', kind='arithmetic',
                     inputs=['allocated', 'reserved'], claim='24 minus 6 is 18.',
                     rationale='Subtract the speaker reservation from the allocation.')
    bridge = dict(identity_basis='explicit_reference', identity_evidence=evidence(bridge_ref),
                  event_ids=['C4'], rationale='The cited statement binds the reservation to C4.',
                  change_or_constraint='Exclude the reserved subset from the audience supply.')
    card = dict(information_need='How many C4 bottles remain available to the audience?',
                natural_value='PASS', uncertainties=[], dependency_graph=dict(
                    nodes=[fact('allocated', 'T1', '24 allocated'),
                           fact('reserved', 'T2', '6 reserved'), operation], answer_node='answer'),
                path_audit=dict(semantic_completeness_proved=False,
                    natural_value_rationale='Audience supply count in a controlled fixture.',
                    source_scan=[dict(turn_refs=[t['turn_ref']], rationale='Examined controlled sentence.')
                                 for t in source['turns']], counterpaths=[]))
    return source, card, bridge


class HistoricalHardTargetTests(unittest.TestCase):
    def test_three_frozen_local_calculations_are_not_hard_targets(self):
        specs = [
            ('109164_v2_61_minute_hold', BASE / 'feedback_replication_20root_resume_20261025_v1/sources/cc-train-episode-109164/FULL_SOURCE.json'),
            ('103866_v21_arrival_gap', BASE / 'day3_production_v1/sources/cc-train-episode-103866.json'),
            ('104814_v21_hold_endpoint', BASE / 'day3_production_v1/sources/cc-train-episode-104814.json'),
        ]
        for name, source_path in specs:
            with self.subTest(case=name):
                case = read(RUN / 'historical_cases' / (name + '.json'))
                target = read(case['target_binding']['path'])
                candidate = read(case['candidate_binding']['path'])
                result = qa.classify_legacy_target(target, read(source_path), candidate)
                self.assertEqual(result['level'], 'L1')
                self.assertFalse(result['eligible_for_hard'])
                self.assertFalse(result['semantic_quality_proved'])

    def test_old_relation_positives_remain_valid_without_promoting_pass_to_difficulty(self):
        fixtures = read(FIXTURES)
        self.assertEqual(len(fixtures['relation_examples']), 2)
        for fixture in fixtures['relation_examples']:
            with self.subTest(case=fixture['fixture_id']):
                source = read(fixture['source_binding']['path'])
                self.assertEqual(file_sha(fixture['source_binding']['path']), fixture['source_binding']['sha256'])
                self.assertEqual(old.validate_discovery(fixture['discovery'], source), [])
                self.assertEqual(fixture['joint_necessity'], 'NOT_CERTIFIED')
                self.assertEqual(fixture['difficulty'], 'NOT_CERTIFIED')
                self.assertEqual(fixture['historical_candidate']['gold'], fixture['historical_review_record']['reconstructed_gold'])
                # No V2.2 dependency graph: old representation alone never certifies L3.
                result = qa.classify_legacy_target(fixture['historical_candidate'], source)
                self.assertFalse(result['eligible_for_hard'])
                self.assertFalse(result['semantic_quality_proved'])

    def test_historical_freeze_failure_is_preserved_as_failure_not_reinterpreted(self):
        case = read(RUN / 'historical_cases/104814_v21_hold_endpoint.json')
        diag = read(case['freeze_order_failure_binding']['path'])
        self.assertEqual(file_sha(case['freeze_order_failure_binding']['path']), case['freeze_order_failure_binding']['sha256'])
        self.assertTrue(diag['canonical_fingerprints_equal'])
        self.assertNotEqual(diag['original_packet_sha256'], diag['frozen_recomputed_packet_sha256'])
        self.assertNotEqual(diag['delivered_display_to_canonical'], diag['recomputed_display_to_canonical'])
        self.assertEqual(diag['delivered_display_to_canonical']['B'], 'end_r6')
        self.assertFalse(diag['official_gold_comparison_performed'])

    def test_arithmetic_cannot_become_hard_only_by_relational_retagging(self):
        source, card, bridge = controlled_quantity_card()
        baseline = qa.classify_relation(card, source)
        self.assertEqual(baseline['level'], 'L1')
        self.assertFalse(baseline['eligible_for_hard'])
        retagged = copy.deepcopy(card)
        retagged['dependency_graph']['nodes'][-1].update(kind='constraint_apply', bridge=bridge)
        result = qa.classify_relation(retagged, source)
        self.assertFalse(result['eligible_for_hard'],
                         'Same scalar subtraction must not become hard solely through a high-order tag.')

    def test_standalone_identity_bridge_is_part_of_support_and_deletion_audit(self):
        source, card, bridge = controlled_quantity_card(separate_identity=True)
        card['dependency_graph']['nodes'][-1].update(kind='constraint_apply', bridge=bridge)
        result = qa.classify_relation(card, source)
        # The identity link may be rejected as an incomplete higher-order contract.
        # If the ledger is accepted, its independent required bridge cannot vanish.
        if not result.get('graph_audit'):
            self.assertFalse(result['eligible_for_hard'])
            return
        paths = result['graph_audit']['minimal_support_sets']
        self.assertTrue(all('T3' in route for route in paths),
                        'Without T3, reservation identity is not supplied by either numeric premise.')
        removed = {row['removed_turn_ref'] for row in result['graph_audit']['turn_deletions']}
        self.assertIn('T3', removed)

    def test_historical_annotations_keep_original_artifacts_bound(self):
        for path in sorted((RUN / 'historical_cases').glob('*.json')):
            with self.subTest(case=path.stem):
                case = read(path)
                self.assertEqual(case['annotation_kind'], 'RETROSPECTIVE_REGRESSION_ANNOTATION')
                self.assertTrue(case['not_new_qa'])
                self.assertTrue(case['not_independent_semantic_review'])
                self.assertFalse(case['expected_formal_qa_admitted'])
                for key in ('target_binding', 'candidate_binding', 'decision_binding', 'freeze_order_failure_binding', 'original_regression_binding'):
                    if key in case:
                        self.assertEqual(file_sha(case[key]['path']), case[key]['sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
