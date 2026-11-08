"""Bind all content-accepted Train candidates before any S0 inference."""
from pathlib import Path
from datetime import datetime, timezone
import collections
import hashlib
import json

HERE = Path(__file__).resolve().parent
WEEK = HERE.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def read(p):
    return json.loads(Path(p).read_text())


def check(b):
    assert sha(b['path']) == b['sha256'], b


def save(name, obj):
    p = HERE / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return binding(p)


def main():
    now = datetime.now(timezone.utc).isoformat()
    publics, privates, provenance, metadata, pairs = [], [], [], [], []
    production_slots, candidate_slots = {}, {}
    for batch in ['day3_production_v1', 'day3_production_batch2_v1']:
        base = WEEK / batch
        policy = read(base / 'EXECUTION_FREEZE.json')
        for slot in policy['slots']:
            production_slots[slot['candidate_slot_id']] = slot
        pairs += read(base / 'MATCHED_SLOT_PAIRS.json')['pairs']
        mapping_path = base / 'presentation/PRIVATE_LABEL_MAP.json'
        presentation = {r['candidate_id']: r for r in read(mapping_path)['rows']} if mapping_path.exists() else {}
        export = base / 'exports/content_accepted.private.jsonl'
        for line in export.read_text().splitlines():
            r = json.loads(line)
            for key in ['public', 'source', 'record']:
                check(r[key])
            original = read(r['public']['path'])
            assert original['candidate_id'] == r['candidate_id']
            shown = original
            shown_gold = r['gold']
            presentation_row = presentation.get(r['candidate_id'])
            if presentation_row:
                assert presentation_row['original_public'] == r['public']
                assert presentation_row['original_gold'] == r['gold']
                check(presentation_row['display_public'])
                shown = read(presentation_row['display_public']['path'])
                shown_gold = presentation_row['display_gold']
                m = presentation_row['mapping_old_to_new']
                assert set(m) == set('ABCDEF') == set(m.values())
                assert all(original['options_or_items'][k] == shown['options_or_items'][v] for k,v in m.items())
                expected_gold = [m[k] for k in r['gold']]
                assert (set(expected_gold) == set(shown_gold) if r['question_type'] == 'multi_select' else expected_gold == shown_gold)
            assert all(shown[k] == original[k] for k in ['candidate_id','root_id','question_type','question'])
            labels = list(shown['options_or_items'])
            assert set(labels) == set('ABCDEF')
            native = {k: (k if r['question_type'] == 'multi_select' else 'I'+str(i+1)) for i,k in enumerate('ABCDEF')}
            field = 'options' if r['question_type'] == 'multi_select' else 'items'
            public = {k: shown[k] for k in ['candidate_id','root_id','question_type','question']}
            public[field] = {native[k]: shown['options_or_items'][k] for k in 'ABCDEF'}
            public.update(source_path=r['source']['path'], source_sha256=r['source']['sha256'])
            gold = [native[k] for k in shown_gold]
            orig_gold_texts = [original['options_or_items'][k] for k in r['gold']]
            new_gold_texts = [public[field][k] for k in gold]
            assert (set(orig_gold_texts) == set(new_gold_texts) if field == 'options' else orig_gold_texts == new_gold_texts)
            publics.append(public)
            privates.append(public | {'gold': gold})
            record = read(r['record']['path'])
            slot_key = r['arm'] + '-' + r['slot_id']
            slot = production_slots[slot_key]
            candidate_slots[r['candidate_id']] = slot
            metadata.append({k:r[k] for k in ['candidate_id','root_id','arm','slot_id','question_type','fact_family_id']} | {
                'batch':batch, 'design_id':slot.get('design_id'),
                'generation_mode':slot.get('generation_mode'), 'target_cause_id':slot.get('target_cause_id'),
                'target_gold_count':slot.get('target_gold_count'),
                'actual_gold_count':len(gold), 'content_state':r['state'],
                'fact_family':record.get('fact_family'), 'target_realization':record.get('target_realization')})
            provenance.append({'candidate_id':r['candidate_id'], 'accepted_export':binding(export),
                'original_public':r['public'], 'original_record':r['record'], 'source':r['source'],
                'presentation':presentation_row, 'presentation_label_to_native_label':native,
                'original_gold':r['gold'], 'presentation_gold':shown_gold, 'native_gold':gold,
                'semantic_text_and_gold_content_roundtrip':'PASS'})
    publics.sort(key=lambda r:r['candidate_id'])
    privates.sort(key=lambda r:r['candidate_id'])
    metadata.sort(key=lambda r:r['candidate_id'])
    ids = {r['candidate_id'] for r in publics}
    counts = collections.Counter(r['question_type'] for r in publics)
    assert len(publics) == len(ids) == 187
    assert counts == {'multi_select':74,'ordering':113}
    assert len({r['root_id'] for r in publics}) == 20
    assert len(production_slots) == 240 and len(pairs) == 88
    assert all(p['A'] in ids and p['B'] in ids for p in pairs)
    pub = save('inputs/PUBLIC_QUESTIONS.json', {'questions':publics})
    private = save('inputs_private/PRIVATE_QUESTIONS.json', {'questions':privates})
    Path(private['path']).chmod(0o600)
    prov = save('inputs_private/PROVENANCE.json', {'questions':provenance})
    Path(prov['path']).chmod(0o600)
    meta = save('CANDIDATE_METADATA.json', {'questions':metadata})
    pair_bind = save('MATCHED_SLOT_PAIRS.json', {'pairs':pairs})
    admitted_slot_keys = {m['arm']+'-'+m['slot_id'] for m in metadata}
    ledger = save('PRODUCTION_SLOT_LEDGER.json', {'production_denominator':240,'content_admitted':187,
        'content_hold':8,'source_shortfall':42,'generation_hold':3,
        'slots':[{'candidate_slot_id':qid, 'slot':slot, 'diagnostic_state':'ADMITTED_EVALUATION_PLANNED' if qid in admitted_slot_keys else 'NOT_CONTENT_ADMITTED_NOT_EVALUATED'} for qid,slot in sorted(production_slots.items())]})
    solver = {'name':'S0_M_clean',
        'model':'/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model',
        'tokenizer':'/root/autodl-tmp/Qwen3-4B', 'MC_mappings':6,'MC_samples_each_mapping':3,
        'ordering_mappings':12,'ordering_samples_each_mapping':1,'temperature':0.8,'top_p':0.9,
        'max_tokens':32,'enable_thinking':False,'max_model_len':16384,'source_truncation':False,
        'automatic_retry':False,'display_seed':2026101903,'sampling_seed':2026101904}
    slots = [{'slot_id':candidate_slots[q['candidate_id']]['slot_id'],
        'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_type':q['question_type'],
        'item_count':6,'source':{'path':q['source_path'],'sha256':q['source_sha256']}} for q in publics]
    contract = save('EXECUTION_FREEZE.json', {'schema_version':'day3_solver187_diagnostic_v1',
        'status':'FROZEN_BEFORE_NEW_SOLVER_INFERENCE','created_at_utc':now,
        'namespace':'train.feedback/diagnostic_only','solver':solver,
        'budget':{'initial_question_slots':187,'max_direct_requests':2688,'production_planned_slots':240},
        'slots':slots,'public_questions':pub,'private_questions_binding':private,
        'metadata_binding':meta,'matched_pairs_binding':pair_bind,'production_ledger_binding':ledger,
        'day1_protocol':binding(WEEK/'day1/PROTOCOL_V1.json'),
        'analysis':{'primary':'question_equal_exact_by_arm_and_question_type',
            'also_report':['root_equal_exact','MC_gold_count','MC_FP_FN_Jaccard_overselection','ordering_pair_accuracy',
                'all_correct_some_correct_never_correct','invalid_operational_errors','88_matched_slot_pairs'],
            'position_consistency_caveat':'MC repeated stochastic samples confound position sensitivity with sampling variability.',
            'A_B_caveat':'Different generated questions and shared roots/facts; descriptive diagnostic difference cannot establish training benefit or B superiority.'},
        'candidate_policy':{'all_187_retained_regardless_of_solver_score':True,
            'content_admission_uses_solver_scores':False,'training_selection':'NOT_PERFORMED; requires prospective rule',
            'A_generator_receives_this_feedback':False,'training_started':False,'dev_test_evaluation':False},
        'review_scope':'Codex internal roles and programs share filesystem; no external blinded review claim.'})
    admission = save('CONTENT_ADMISSION.json', {'status':'CONTENT_FROZEN','namespace':'train.feedback/diagnostic_only',
        'planned_slots':187,'admitted_count':187,'production_planned_slots':240,
        'public_questions':pub,'private_questions':private,
        'questions':[{'candidate_id':q['candidate_id']} for q in publics],
        'content_admission_uses_solver_scores':False,'scope':'Reuse exact accepted records; no new content review performed in this assembly.'})
    summary = save('INPUT_ASSEMBLY.json', {'status':'PASS','created_at_utc':now,'questions':187,
        'roots':20,'by_type':dict(counts),'requests':2688,'matched_pairs':88,
        'new_solver_calls_at_assembly':0,'script':binding(__file__),
        'files':[pub,private,prov,meta,pair_bind,ledger,contract,admission]})
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
