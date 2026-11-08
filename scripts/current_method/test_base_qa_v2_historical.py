"""Small semantic regressions over frozen, historically reviewed QA.

Only discovery representation is tested: retained source relations are not new
QA or a certification of joint necessity, difficulty, or material admission.
No model, API, Solver, training, or historical-artifact mutation is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    from . import base_qa_pipeline_v2 as qa
except ImportError:
    import base_qa_pipeline_v2 as qa

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research-protocols/selfplay_week_20261019/base_qa_generation_v2_20261031'
FIXTURES = RUN / 'HISTORICAL_RELATION_FIXTURES.json'


def read(path):
    return json.loads(Path(path).read_text())


def file_binding(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def historical_rows():
    return read(FIXTURES)


def regression_report():
    fixtures = historical_rows()
    audit = read(fixtures['audit_path'])
    by_id = {e['candidate']['candidate_id']: e for e in audit['selected_examples']}
    rows = []
    for fixture in fixtures['relation_examples']:
        source = read(fixture['source_binding']['path'])
        original = by_id[fixture['fixture_id']]
        binding = file_binding(fixture['source_binding']['path'])
        errors = qa.validate_discovery(fixture['discovery'], source)
        rows.append(dict(candidate_id=fixture['fixture_id'],
            question=fixture['historical_candidate']['question'],
            historical_gold=fixture['historical_candidate']['gold'],
            historical_review_tier=fixture['historical_review_tier'],
            historical_review_collection=fixture['historical_review_collection'],
            historical_candidate_snapshot_sha256=fixture['historical_candidate_snapshot_sha256'],
            candidate_unchanged=fixture['historical_candidate']==original['candidate'],
            gold_unchanged=fixture['historical_candidate']['gold']==original['candidate']['gold'],
            source_binding=fixture['source_binding'],
            source_hash_matches=binding['sha256']==fixture['source_binding']['sha256'],
            source_bytes_match=binding['bytes']==fixture['source_binding']['bytes'],
            validate_discovery_errors=errors, retained=not errors,
            relation_count=len(fixture['discovery']['relations']),
            semantic_scope=fixture['semantic_scope'],
            relationship_qualification=fixture['relationship_qualification'],
            joint_necessity='NOT_CERTIFIED', difficulty='NOT_CERTIFIED',
            source_scan_semantic_completeness='NOT_CERTIFIED',
            material_admission='NOT_REASSESSED_RESEARCH_REGRESSION_ONLY'))
    denominator = len(rows)
    numerator = sum(r['retained'] for r in rows)
    return dict(schema_version='base_qa_v2_historical_relation_regression_v1',
        fixtures_binding=file_binding(FIXTURES),
        pipeline_binding=file_binding(Path(qa.__file__)),
        audit_binding=file_binding(fixtures['audit_path']),
        denominator_definition='Two deliberately selected historically content-approved cross-scene relation/order examples; not all historical PASS questions, not a random or unseen sample.',
        historical_relation_retention=dict(numerator=numerator, denominator=denominator,
            rate=numerator/denominator if denominator else None),
        joint_reasoning_quality_certified_count=0,
        joint_necessity='NOT_CERTIFIED', difficulty='NOT_CERTIFIED',
        relation_examples=rows, independent_fact_controls=fixtures['independent_fact_controls'],
        independent_fact_controls_counted_in_relation_denominator=False,
        limitations=[
            'Day1 is same-model source-first blind content review on a reused, outcome-selected cohort.',
            'Production prerequisite example has separate author/reviewer roles but Gold was visible.',
            'All sources include generated extensions; no pristine-natural-dialogue claim.',
            'Exact statement chronology is not a causal or joint-reasoning certificate.',
            'validate_discovery verifies representation constraints, not unrestricted semantic truth or necessary-evidence completeness.',
            'No new QA or empirical Solver weakness claim is produced by these regressions.'],
        new_qa_generated=False, model_calls=0, solver_calls=0, training_runs=0)


class HistoricalRelationRegressionTests(unittest.TestCase):
    def test_historical_gold_source_and_versions_remain_bound(self):
        fixtures = historical_rows()
        self.assertEqual(file_binding(fixtures['audit_path'])['sha256'], fixtures['audit_sha256'])
        report = regression_report()
        for row in report['relation_examples']:
            with self.subTest(candidate=row['candidate_id']):
                self.assertTrue(row['candidate_unchanged'])
                self.assertTrue(row['gold_unchanged'])
                self.assertTrue(row['source_hash_matches'])
                self.assertTrue(row['source_bytes_match'])
        for fixture in fixtures['relation_examples']:
            self.assertEqual(qa.fp(fixture['historical_candidate']), fixture['historical_candidate_snapshot_sha256'])
            review = fixture['historical_review_record']
            self.assertEqual(fixture['historical_candidate']['gold'], review['reconstructed_gold'])

    def test_cross_scene_historical_relations_survive_identity_rules(self):
        fixtures = historical_rows()
        for fixture in fixtures['relation_examples']:
            with self.subTest(candidate=fixture['fixture_id']):
                source = read(fixture['source_binding']['path'])
                refs = {t['turn_ref']: t for t in source['turns']}
                scenes = {refs[e['turn_ref']].get('scene_ref') or refs[e['turn_ref']].get('session_ref')
                    for r in fixture['discovery']['relations'] for e in r['evidence']}
                self.assertGreaterEqual(len(scenes), 2)
                self.assertEqual(qa.validate_discovery(fixture['discovery'], source), [])
                self.assertEqual(fixture['joint_necessity'], 'NOT_CERTIFIED')
        self.assertEqual(regression_report()['historical_relation_retention'],
                         dict(numerator=2, denominator=2, rate=1.0))

    def test_cross_scene_reference_cannot_be_replaced_by_actor_continuity(self):
        fixture = historical_rows()['relation_examples'][0]
        source = read(fixture['source_binding']['path'])
        broken = copy.deepcopy(fixture['discovery'])
        broken['events'][0]['mentions'][1]['identity_basis'] = 'local_continuation'
        errors = qa.validate_discovery(broken, source)
        self.assertTrue(any('cross-scene' in error for error in errors), errors)

    def test_independent_fact_content_pass_is_not_joint_reasoning_positive(self):
        fixtures = historical_rows()
        audit = read(fixtures['audit_path'])
        by_id = {e['candidate']['candidate_id']: e for e in audit['selected_examples']}
        controls = fixtures['independent_fact_controls']
        self.assertEqual(len(controls), 2)
        for control in controls:
            with self.subTest(candidate=control['candidate_id']):
                self.assertEqual(control['historical_candidate'], by_id[control['candidate_id']]['candidate'])
                self.assertEqual(control['historical_gold'], control['historical_candidate']['gold'])
                self.assertEqual(qa.fp(control['historical_candidate']), control['historical_candidate_snapshot_sha256'])
                self.assertEqual(file_binding(control['source_binding']['path'])['sha256'], control['source_binding']['sha256'])
                self.assertFalse(control['joint_reasoning_positive'])
                self.assertEqual(control['joint_necessity'], 'NOT_CERTIFIED')
                self.assertEqual(control['relation_selection_decision'], 'REJECT_AS_JOINT_REASONING_POSITIVE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HistoricalRelationRegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = regression_report()
    report['tests'] = dict(run=result.testsRun, failures=len(result.failures),
                           errors=len(result.errors), passed=result.wasSuccessful())
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
