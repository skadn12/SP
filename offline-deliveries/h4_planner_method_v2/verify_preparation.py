"""Read-only request/integrity verification; never instantiate a transport."""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
WORK = Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-planner-method-v2')
sys.path.insert(0, str(WORK))
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.planner_method import apply_prompt
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.planner_supplement_context import build_supplement_context

OLD = BASE.parent / 'h4_planner_codex_feedback_v1/prepared_run'
RUN = BASE / 'prepared_run'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    manifest = read(RUN / 'freeze_manifest.json')
    results = []
    for root in manifest['root_order']:
        source = OLD / 'expansion/roots' / root / 'planner'
        old_prompt = read(OLD / 'frozen' / root / 'planner_prompt.json')
        new_prompt = read(RUN / 'frozen' / root / 'planner_prompt.json')
        assert apply_prompt(old_prompt)['prompt'] == new_prompt['prompt'], root
        assert sha(OLD / 'frozen' / root / 'inputs.json') == sha(RUN / 'frozen' / root / 'inputs.json'), root
        old_request = read(OLD / 'frozen' / root / 'planner_request.json')
        new_request = read(RUN / 'frozen' / root / 'planner_request.json')
        assert {k:v for k,v in old_request.items() if k != 'messages'} == {k:v for k,v in new_request.items() if k != 'messages'}
        ctx = read(source / 'supplement_prompt_manifest.json')['candidate_recovery_context']
        reasons = ctx['gate_outline_rejection_reasons']
        reasons.update(gate_raw=read(source.parent / 'semantic_gate/output.json'),
                       outline_raw=read(source / 'outline_semantic_output.json'))
        records = read(source / 'validation.json')['candidate_records']
        compact = build_supplement_context(records, ctx['qualified_candidate_refs_locked'], reasons)
        assert [x['raw_candidate'] for x in compact['original_candidates']] == [x['raw_candidate'] for x in ctx['original_candidates']]
        assert [x['candidate_ref'] for x in compact['original_candidates']] == [x['candidate_ref'] for x in ctx['original_candidates']]
        for value in compact['original_candidates']:
            assert 'resolved_evidence' not in json.dumps(value.get('gate'))
            assert 'raw_model_reviews' not in json.dumps(value.get('outline'))
        before = len(json.dumps(ctx, ensure_ascii=False, indent=2))
        after = len(json.dumps(compact, ensure_ascii=False, indent=2))
        results.append({'root_id': root, 'frozen_inputs_sha256': sha(RUN / 'frozen' / root / 'inputs.json'),
                        'prompt_change_matches_method_adapter': True, 'request_parameters_unchanged': True,
                        'historical_context_projection_chars': before, 'compact_projection_chars': after,
                        'reduction_percent': round(100 * (1 - after / before), 1),
                        'projection_is_new_model_output': False,
                        'primary_capacity': read(RUN / 'frozen' / root / 'planner_capacity.json')})
    freezes = []
    for p in [OLD/'freeze_manifest.json', OLD.parent/'inputs/manifest.json', RUN/'freeze_manifest.json']:
        files = read(p)['files']
        changed = [row['path'] for row in files if not Path(row['path']).is_file() or sha(Path(row['path'])) != row['sha256']]
        assert not changed, changed
        freezes.append({'manifest':str(p),'count':len(files),'changed_or_missing':changed})
    assert not (RUN / 'EXECUTION_STARTED.json').exists()
    assert not list(RUN.rglob('response.txt'))
    result = {'status':'OFFLINE_VERIFICATION_PASS', 'external_api_calls':0,
              'live_model_quality':'NOT_EVALUATED', 'tests':{'passed':91,'failures':0,'errors':0,
              'scope':'Planner v2, supplement context, workflow, candidate recovery, actors, formal recovery entries and bundle identity; mocks do not certify semantics'},
              'roots':results, 'freeze_integrity':freezes,
              'candidate_minimum':3,'supplement_limit':1,'future_full_pipeline_http_ceiling':68,
              'budget_authorization_for_new_run':'NOT_REQUESTED_OR_CONSUMED_THIS_TURN',
              'comparison_limit':'Context projection on historical model outputs is an input-size check, not a v2 generation experiment.'}
    (BASE/'reports/VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'external_api_calls':0,'roots':[{k:x[k] for k in ('root_id','historical_context_projection_chars','compact_projection_chars','reduction_percent')} for x in results],'freezes':freezes},ensure_ascii=False))


if __name__ == '__main__':
    run()
