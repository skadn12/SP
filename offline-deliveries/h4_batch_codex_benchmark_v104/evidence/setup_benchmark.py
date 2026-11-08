"""Freeze inputs for a Codex-session authored benchmark, without Qwen HTTP."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
OLD = BASE.parent / 'h4_batch_restore_v103'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started = datetime.now(timezone.utc).isoformat()
    registry = json.loads((OLD/'expansion_engine/SOURCE_REGISTRY.json').read_text())
    pool_path = BASE.parent/'h4_external_event_source_refresh_v1/soda_train_sample.json'
    pool = {int(x['row']['original_index']): x for x in json.loads(pool_path.read_text())['rows']}
    assignments = {'14304':6028354,'20432':6028058,'24719':6028328,'38601':6028310}
    reservations = sorted(BASE.parent.glob('**/SOURCE_RESERVATIONS.json'))
    evidence = [{'path':str(p),'sha256':sha(p)} for p in reservations if not p.is_relative_to(BASE)]
    consumed_text = '\n'.join(p.read_text() for p in reservations if not p.is_relative_to(BASE))
    allocation=[]
    for short, index in assignments.items():
        root='cc-train-episode-'+short
        assert f'soda:train:{index}' not in consumed_text
        spec=next(x for x in registry['roots'] if x['root_id']==root)
        src=Path(spec['source_path'])
        assert sha(src)==spec['source_sha256']
        target=BASE/'inputs'/f'root{short}_BASELINE.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists()
        target.write_bytes(src.read_bytes())
        source=pool[index]
        save(BASE/'inputs'/f'root{short}_EVENT.json',source)
        allocation.append({'root_id':root,'source_id':f'soda:train:{index}',
            'baseline_path':str(target),'baseline_sha256':sha(target),
            'source_item_path':str(BASE/'inputs'/f'root{short}_EVENT.json'),
            'source_item_sha256':hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'status':'PERMANENTLY_RESERVED_BEFORE_CODEX_AUTHORING'})
    save(BASE/'SOURCE_RESERVATIONS.json',{'schema_version':'codex_session_reservations_v1','created_at_utc':started,'allocation':allocation,'historical_reservation_files':evidence})
    save(BASE/'BENCHMARK_STARTED.json',{'started_at_utc':started,'author_backend':'CURRENT_CODEX_SESSION_AND_INHERITED_SUBAGENTS','qwen_transport_used':False,'codex_cli_used':False,'exact_model_id':'Not exposed by session runtime; inherited Codex agent model','exact_usage_tokens':None,'billing_cost':None,'usage_limitation':'Session tools expose neither exact input/output usage nor billing. Artifact tokens can be counted separately and are not billed usage.','roots':list(assignments),'attempts_per_root':1,'planned_new_turns_per_root':[12,16],'qa_per_ready_root':{'multiselect_gold_2_3_4_5':4,'gold6_control':1,'ordering':2},'solver_run':False,'training_run':False,'sampling':'Purposive exposed roots and compatible unused events; not random holdout.','method':'Codex reads complete baseline, assigned event and public historical feedback; authors one suffix; independent agent reads complete cumulative source before QA; another agent authors QA; independent source-first answer review precedes private Gold comparison. No writer repair or source substitution in this benchmark.'})
    save(BASE/'evidence/HISTORICAL_PRESERVATION_BEFORE.json',{'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(OLD.rglob('*')) if p.is_file()]})
    feedback=OLD/'expansion_engine/compiled_history_v91/COMPILED_FEEDBACK.json'
    (BASE/'inputs/HISTORICAL_FEEDBACK.json').write_bytes(feedback.read_bytes())
    print(json.dumps({'started_at_utc':started,'allocation':allocation},indent=2))

if __name__=='__main__':
    main()
