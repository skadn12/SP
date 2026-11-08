"""Current four-root baseline capacity estimate; never a READY/actual QA claim."""
from pathlib import Path
import json
import sys
import tempfile

HOME = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HOME / 'qa_engine'))
import ordering_generation as generation
import ordering_prior
from common import bind


def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    e = generation.engine(); rows = []
    _, _, annotate, capacity, dev = e.base.transport(); tokenizer = dev.tokenizer_from(str(e.base.TOKENIZER))
    precall = read(HOME / 'PRECALL_PLAN.json')
    old_config = read(HOME.parent / 'h4_v95_ready_mixed_qa_v96/shards/shard_01/CONFIG.json')
    provider = e._dependencies()[0]
    registry_path = HOME / 'expansion_engine/SOURCE_REGISTRY.json'
    registry = {r['root_id']: r for r in read(registry_path)['roots']}
    for root in precall['candidate_roots']:
        root_row = registry[root]
        source_path = Path(root_row['source_path']); source = read(source_path)
        assert bind(source_path)['sha256'] == root_row['source_sha256']
        assert len(source['turns']) == root_row['source_turns']
        feedback = e.source_adapter.feedback_data(root)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); e.source_adapter.freeze_feedback(root, run / 'feedback', data=feedback)
            save(run / 'MANIFEST.json', {'solver_model': e.legacy.TARGET})
            compact, files = e.source_adapter.prior_candidates(root, feedback,
                run / 'feedback' / root / 'candidate_history.json', e.v15)
            priors, receipt, _ = ordering_prior.collect(root, run / 'feedback' / root / 'candidate_history.json',
                [old_config['baseline_pool']], compact_rows=compact, compact_files=files)
            public = [{k:q[k] for k in ('candidate_id','root_id','question_type','question','items')} for q in priors]
            ctx = {'root_id': root, 'candidate_ids': [root + '-capacity-estimate-01', root + '-capacity-estimate-02'],
                   'source_id_anchor_records': {'location': 'Not authored before expansion; reserve shown separately.'},
                   'actor_reference_policy': {'location': 'policy follows below'},
                   'source_turns': source['turns'], 'source_metadata': {k:v for k,v in source.items() if k != 'turns'},
                   'previous_public_questions_dedup_only': {'question_index': [{k:q[k] for k in ('candidate_id','root_id','question_type')} for q in priors]},
                   'source_mode': 'complete_frozen_sources', 'training_allowed': False}
            payload, _ = e.legacy.legacy.payload_with_feedback(run, root,
                'Read the complete frozen source and historical feedback.\n', ctx, 6144)
            policy = {'root_id': root, 'canonical_source_speaker_names': sorted({t['speaker'] for t in source['turns']}),
                      'forbidden_gendered_pronouns': ['he','him','his','himself','she','her','hers','herself'],
                      'output_style': 'use_source_names_or_explicit_roles_for_people', 'new_source_facts':False,'automatic_repair':False}
            payload['messages'][0]['content'] += '\n[ORDERING_BATCH_V89]\n' + generation.TASK + '\nPUBLIC HISTORICAL ORDERING QUESTIONS:\n' + json.dumps({'public_ordering_questions':public},ensure_ascii=False,separators=(',',':')) + '\nACTOR_REFERENCE_POLICY:\n' + json.dumps(policy,ensure_ascii=False,separators=(',',':')) + '\n[/ORDERING_BATCH_V89]'
            profile = dict(provider); profile['output_limit'] = 6144
            profile['request_parameters'] = {**profile['request_parameters'], 'max_tokens':6144}
            cap = annotate(capacity(profile, tokenizer, payload['messages'], 6144))
            estimate = cap['required_context_tokens'] + 4096
            content = payload['messages'][0]['content']
            assert all(json.dumps(t['text'], ensure_ascii=False)[1:-1] in content for t in source['turns'])
            assert feedback['writer_suffix'] in content
            rows.append({'root_id':root, 'historical_source':bind(source_path), 'historical_turn_count':len(source['turns']),
                         'historical_feedback_candidates':receipt['historical_candidate_count'],
                         'supplied_public_ordering_prior_count':len(priors),'prior_item_counts':receipt['historical_item_counts'],
                         'baseline_capacity':cap,'added_scene_and_anchor_reserve_estimate_tokens':4096,
                         'estimated_required_context_tokens':estimate,'estimated_remaining_tokens':cap['context_window_tokens']-estimate,
                         'estimate_status':'PASS' if estimate<=cap['context_window_tokens'] else 'BLOCKED',
                         'estimate_is_actual_future_qa_request':False,'current_new_scene_or_anchors_authored':False,
                         'full_historical_source_and_feedback_preserved':True,'http_calls':0})
    result = {'version':'v97_four_root_pre_expansion_ordering_capacity_estimate_v1',
              'status':'PASS_ESTIMATE_ONLY' if all(r['estimate_status']=='PASS' for r in rows) else 'CAPACITY_RISK_ESTIMATE',
              'precall_scope':bind(HOME/'PRECALL_PLAN.json'),'source_registry':bind(registry_path),'rows':rows,'historical_source_turn_total':sum(r['historical_turn_count'] for r in rows),
              'limitation':'4096 is a planning reserve, not a certified bound for unknown new scene plus duplicated anchor records. Actual full-source request must pass prepare capacity after expansion; no truncation or automatic source fallback.',
              'new_source_readiness_or_answer_certified':False,'model_calls':0,'script':bind(__file__)}
    save(HOME/'evidence/ORDERING_PRE_EXPANSION_CAPACITY.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
