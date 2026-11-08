"""Three controlled replays of the immutable first-HOLD failure, no model calls.

These tests validate known historical failure boundaries. They are not an
independent-model discovery rate and do not rewrite the historical question.
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

ROOT=Path(__file__).resolve().parents[2]
HISTORY=ROOT/'research-protocols/selfplay_week_20261019/codex_qwen_first_root_20261031_v1'
PATHS={
    'relation':HISTORY/'qa_run/inputs/RELATION.json',
    'candidate':HISTORY/'qa_run/inputs/CANDIDATE.json',
    'correction':HISTORY/'qa_run_reviewed_v2/state/COMPARISON_CORRECTION.json',
    'v1_blind_packet':HISTORY/'qa_run/state/BLIND_PACKET.json',
    'source':ROOT/'research-protocols/selfplay_week_20261019/feedback_replication_20root_resume_20261025_v1/sources/cc-train-episode-109164/FULL_SOURCE.json',
}
EXPECTED_SHA256={
    'relation':'57d30da109686ea8da1ec1d6c3c1baf8aa50486449d66430863d7ce1f19b72ba',
    'candidate':'23023c6f936d545d6c290f9e3a3329b08a0aabcdf66a18fd9bfd58bce6151c17',
    'correction':'b681ab040cd7c5e1351dd128028f3f1c45a66e900ffd1faa3e790a6b72eed99a',
    'v1_blind_packet':'925524f334dc5325364d78ec6fd6dd46a09aac7b46a392bc0e939176cd35123f',
    'source':'98d0a54c28e3472186804cfa2b9902d5fbd72179e49ad412751840adf7665fa1',
}
ALTERNATIVE_REFS=['R01:S01:T003','R02:A:S01:T003','R02:A:S01:T005',
                  'R02:A:S01:T007','R03:A:S02:T002','R03:A:S01:T008']


def bound_inputs():
    out={}
    for name,path in PATHS.items():
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=EXPECTED_SHA256[name]:raise ValueError('historical artifact drift: '+name)
        out[name]=json.loads(path.read_text())
    return out


def historical_replay(data):
    """Adapt only an in-memory proof ledger; the original relation stays frozen."""
    source=data['source']; relation=copy.deepcopy(data['relation'])
    for fact in relation['facts']:
        fact['supports']=[dict(turn_refs=fact['evidence_turn_ids'][:],evidence=copy.deepcopy(fact['evidence']),
                               entailment_rationale='Historical author proposition, replayed without recertifying semantic truth.')]
    scan=[]
    used={ref for fact in relation['facts'] for ref in fact['evidence_turn_ids']}
    for turn in source['turns']:
        ref=turn['turn_ref']
        if ref in ALTERNATIVE_REFS:
            scan.append(dict(turn_refs=[ref],classification='possible_alternative',resolution='unresolved',
                rationale='Old duration plus a later Monday start is a plausible path, but the earlier trial is explicitly separate; no duration transfer is proved.'))
        else:
            scan.append(dict(turn_refs=[ref],classification='support' if ref in used else 'irrelevant',
                rationale='Original author cited evidence.' if ref in used else
                    'Not part of this narrowly controlled known-failure replay; natural-language completeness is not certified.'))
    relation['evidence_certificate']=dict(primary_turn_refs=[t['turn_ref'] for t in source['turns'] if t['turn_ref'] in used],
        all_minimal_support_sets=[[t['turn_ref'] for t in source['turns'] if t['turn_ref'] in used]],
        necessity_claim='premise_groups',full_source_scan=scan)
    result=qa.evidence_audit(relation,source,qa.source_manifest(source))
    return relation,result


def regression_report():
    data=bound_inputs(); source=data['source']; correction=data['correction']; relation=data['relation']
    by_ref={t['turn_ref']:t for t in source['turns']}
    _,audit=historical_replay(data)
    removed=set(next(g for g in relation['necessity_groups'] if g['group_id']=='monday_block_details')['turn_refs'])
    remaining={t['turn_ref'] for t in source['turns'] if t['turn_ref'] not in removed}
    nine_refs={e['turn_ref'] for row in data['candidate']['item_analysis'].values() for e in row['evidence']}
    nine_refs.update(r for g in relation['necessity_groups'] for r in g['turn_refs'])
    packet=qa.blind_packet(data['candidate'],source,qa.source_manifest(source))
    public={t['turn_ref']:t for t in packet['source']['turns']}
    conditions=[dict(condition='event_identity_and_duration_transfer',passed=
        correction['overall_decision']=='HOLD' and 'separate' in by_ref['R03:A:S01:T008']['text']
        and 'does not unambiguously establish' in correction['corrected_necessity_review']['rationale'],
        outcome='PLAUSIBLE_UNRESOLVED_NOT_CONFIRMED_BYPASS',confirmed_bypass_proved=False,
        reasoning='Same participants and similar activities do not establish that an earlier trial duration transfers to the later plan.'),
      dict(condition='full_source_alternative_risk_after_monday_removal',passed=
        set(ALTERNATIVE_REFS)<=remaining and not audit['mechanical_pass']
        and any('unresolved alternative' in e for e in audit['errors']),
        outcome='V2_HOLD_UNRESOLVED_ALTERNATIVE',removed_turn_refs=sorted(removed),
        surviving_risk_evidence=[dict(turn_ref=ref,text=by_ref[ref]['text']) for ref in ALTERNATIVE_REFS],
        evidence_audit=audit,scope='In-memory schema replay of known failure; no new Gold or regenerated question.'),
      dict(condition='source_scope_and_provenance_visible_to_blind_reviewer',passed=
        len(data['v1_blind_packet']['source_warnings'])==0 and len(source['warnings'])==9
        and packet['source_scope']['warnings']==source['warnings'] and len(packet['source']['turns'])==233
        and len(nine_refs)==9 and all(public[r]['provenance']['provenance_type']==by_ref[r]['origin'] for r in nine_refs),
        outcome='V2_PRESERVES_SOURCE_WARNINGS_AND_NINE_EVIDENCE_ORIGINS',v1_warning_count=0,
        complete_source_warning_count=len(source['warnings']),v2_warning_count=len(packet['source_scope']['warnings']),
        v2_turn_count=len(packet['source']['turns']),evidence_origin_count=len(nine_refs),
        evidence_origins={r:public[r]['provenance']['provenance_type'] for r in sorted(nine_refs)},
        all_nine_research_only=all(not public[r]['provenance']['formal_qa_allowed'] for r in nine_refs))]
    return dict(schema_version='base_qa_v2_first_hold_negative_regression_v1',
        candidate_id=data['candidate']['candidate_id'],relation_id=relation['relation_id'],
        historical_artifacts_unchanged=True,
        bindings={name:dict(path=str(PATHS[name]),sha256=EXPECTED_SHA256[name],bytes=PATHS[name].stat().st_size) for name in PATHS},
        pipeline_sha256=hashlib.sha256(Path(qa.__file__).read_bytes()).hexdigest(),
        negative_regression=dict(passed=sum(c['passed'] for c in conditions),total=len(conditions)),conditions=conditions,
        denominator_definition='Three deliberately specified known first-HOLD failure conditions; controlled replay, not independent-model discovery rate.',
        first_hold_rewritten=False,new_qa_generated=False,model_calls=0,solver_calls=0,training_runs=0)


class FirstHoldNegativeRegressionTests(unittest.TestCase):
    def test_identity_risk_is_unresolved_and_not_proven_bypass(self):
        report=regression_report(); condition=report['conditions'][0]
        self.assertTrue(condition['passed'],condition)
        self.assertFalse(condition['confirmed_bypass_proved'])
        self.assertEqual(condition['outcome'],'PLAUSIBLE_UNRESOLVED_NOT_CONFIRMED_BYPASS')

    def test_full_source_residual_path_risk_is_not_ignored(self):
        report=regression_report(); condition=report['conditions'][1]
        self.assertTrue(condition['passed'],condition)
        self.assertFalse(condition['evidence_audit']['mechanical_pass'])
        original=bound_inputs()['relation']
        self.assertNotIn('evidence_certificate',original)
        self.assertTrue(all('supports' not in f for f in original['facts']))

    def test_blind_scope_preserves_nine_warnings_and_nine_evidence_origins(self):
        condition=regression_report()['conditions'][2]
        self.assertTrue(condition['passed'],condition)
        self.assertEqual(condition['v2_warning_count'],9)
        self.assertEqual(condition['evidence_origin_count'],9)
        self.assertEqual(set(condition['evidence_origins'].values()),{'codex_synthetic_extension'})
        self.assertTrue(condition['all_nine_research_only'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FirstHoldNegativeRegressionTests))
    report=regression_report();report['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful())
    if args.report:args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+chr(10))
    raise SystemExit(0 if result.wasSuccessful() else 1)
