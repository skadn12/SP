"""CPU-only structural/token and dynamic-display checks; never reads Gold or scores."""
from __future__ import annotations
import hashlib
import importlib.util
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
HERE = Path(__file__).resolve().parent
WEEK = HERE.parent
REPO = WEEK.parents[1]
LOCK = WEEK / 'day3_production_v1/evaluation/EVAL_SOURCE_FAMILY_LOCK.json'
EXPECTED_LOCK = 'd6738f6808b96d50238bc7c914442a3e717aaf10f5c08e69ca3135366f29ef89'
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
ADAPTER = WEEK / 'day2/solver/adapter.py'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def bind(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}

def check(ok, label):
    if not ok:
        raise RuntimeError(label)

def main():
    check(Path(sys.prefix).resolve() == (REPO / '.venv-local4b').resolve(), 'EXACT_LOCAL4B_ENV_REQUIRED')
    check(sha(LOCK) == EXPECTED_LOCK, 'SOURCE_LOCK_CHANGED')
    lock = json.loads(LOCK.read_text())
    bindings = json.loads((WEEK/'day1/MODEL_BINDINGS.json').read_text())
    tokenizer_files = [row for row in bindings['models']['base']['files']
                       if Path(row['path']).name in {'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'}]
    check(len(tokenizer_files) == 4, 'FOUR_TOKENIZER_BINDINGS_REQUIRED')
    for row in tokenizer_files:
        check(sha(row['path']) == row['sha256'], 'TOKENIZER_CHANGED:'+row['path'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    spec = importlib.util.spec_from_file_location('common_eval_dynamic_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    source_rows = []
    roots = []
    for row in lock['source_bindings']:
        path = Path(row['source_binding']['path'])
        check(sha(path) == row['source_binding']['sha256'], 'SOURCE_HASH_CHANGED')
        source = json.loads(path.read_text())
        turns = source['turns']
        refs = [t['turn_ref'] for t in turns]
        check(source['root_id'] == row['root_id'] and len(refs) == len(set(refs)), 'SOURCE_IDENTITY')
        check(all(all(isinstance(t.get(k),str) and t[k].strip() for k in ['turn_ref','speaker','text']) for t in turns), 'TURN_SCHEMA')
        check(len(turns) == row['turn_count'], 'TURN_COUNT_CHANGED')
        scenes = sorted({t['scene_ref'] for t in turns if t.get('scene_ref') is not None})
        check(scenes == row['explicit_scene_refs'], 'SCENE_STRUCTURE_CHANGED')
        dialogue = '\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        source_rows.append({'root_id':row['root_id'],'original_split':row['original_split'],
            'source_binding':bind(path),'turn_count':len(turns),'explicit_scene_refs':scenes,
            'source_text_tokens_without_question_or_template':len(tokenizer.encode(dialogue,add_special_tokens=False)),
            'token_count_scope':'Dialogue block only; not a request length check or request readiness PASS.',
            'role':'REGRESSION_ONLY'})
        roots.append(row['root_id'])
    check(len(roots) == len(set(roots)) == 31, 'ROOT_COUNT')
    slots = lock['itemized_slot_family_ledger']
    check(len(slots) == 186 and len({s['slot_id'] for s in slots}) == 186, 'SLOT_COUNT')
    source_map = {r['root_id']:r for r in source_rows}
    structural_shortfalls = []
    for slot in slots:
        required = slot['design_target']['minimum_explicit_scene_count']
        if len(source_map[slot['root_id']]['explicit_scene_refs']) < required:
            structural_shortfalls.append({'slot_id':slot['slot_id'],'root_id':slot['root_id'],
                'question_type':slot['question_type'],'minimum_explicit_scene_count':required,
                'actual_explicit_scene_count':len(source_map[slot['root_id']]['explicit_scene_refs']),
                'status':'SOURCE_SHORTFALL_STRUCTURE','reason':'Fixed scene composition cannot be supported by the admitted explicit scenes. No source or design replacement.'})
    check([s['slot_id'] for s in structural_shortfalls] == lock['deduplication_protocol']['known_structural_scene_shortfall_slot_ids'], 'SHORTFALL_SET_CHANGED')
    # These records are synthetic algorithm checks only. They are never questions
    # in the evaluation set, never source material and never model requests.
    mapping_checks = []
    for n, qtype in itertools.product([4,5,6], ['multi_select','ordering']):
        field = 'options' if qtype == 'multi_select' else 'items'
        labels = adapter.labels(qtype,n)
        q = {'candidate_id':f'SYNTHETIC-CHECK-{qtype}-{n}', 'root_id':'SYNTHETIC-CHECK',
             'question_type':qtype,'question':'Synthetic mapping validation only.',
             field:{label:f'Synthetic content {i}' for i,label in enumerate(labels)},
             'source_path':'/synthetic/no-source','source_sha256':'0'*64}
        adapter.public_question(q)
        count = 6 if field == 'options' else 12
        variants = adapter.variants(q,20261020,count)
        check(variants == adapter.variants(q,20261020,count), 'NONDETERMINISTIC_MAPPING')
        canonical = adapter.content_ids(q)
        check(len({tuple(v['display_order_content_ids']) for v in variants}) == count, 'REPEATED_DISPLAY')
        answer_space = ([list(c) for k in range(1,n+1) for c in itertools.combinations(labels,k)]
                        if field == 'options' else list(itertools.permutations(labels)))
        round_trips = 0
        for variant in variants:
            mapping = variant['display_to_content']
            check(set(mapping) == set(labels) and set(mapping.values()) == set(canonical.values()), 'NOT_BIJECTIVE')
            inverse = {v:k for k,v in mapping.items()}
            for gold in answer_space:
                content = [canonical[k] for k in gold]
                display = [inverse[k] for k in content]
                check([mapping[k] for k in display] == content, 'ANSWER_ROUNDTRIP_FAILED')
                round_trips += 1
        negative=dict(q,gold=list(labels))
        rejected=False
        try: adapter.public_question(negative)
        except ValueError: rejected=True
        check(rejected, 'PRIVATE_GOLD_NOT_REJECTED')
        choices=adapter.choices(qtype,q[field])
        check(len(choices)==len(answer_space), 'FINITE_CHOICE_COUNT')
        expected_choices={'<box>'+','.join(answer)+'</box>' for answer in answer_space}
        check(set(choices)==expected_choices and len(set(choices))==len(choices), 'FINITE_CHOICE_SYNTAX')
        mapping_checks.append({'item_count':n,'question_type':qtype,'distinct_variants':count,
            'synthetic_valid_answer_space':len(answer_space),'answer_roundtrip_checks':round_trips,
            'private_field_rejection':'PASS','deterministic_gold_blind_mapping':'PASS'})
    native_paths=sorted({Path(module.__file__).resolve() for module in list(sys.modules.values())
        if getattr(module,'__file__',None) and Path(module.__file__).suffix=='.py'
        and Path(module.__file__).resolve().is_relative_to(adapter.WORKTREE.resolve())})
    out={'schema_version':'common_eval_source_and_mapping_cpu_preflight_v1',
         'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'status':'STRUCTURAL_AND_SYNTHETIC_MAPPING_PASS_ONLY',
         'script_binding':bind(__file__),'source_lock':bind(LOCK),'adapter_code_binding':bind(ADAPTER),
         'imported_native_code_bindings':[bind(path) for path in native_paths],
         'runtime_versions':{name:importlib.metadata.version(name) for name in ['transformers','tokenizers','torch','vllm']},
         'model_identity_policy_binding':bind(WEEK/'day1/MODEL_BINDINGS.json'),
         'actual_tokenizer_bindings':[bind(r['path']) for r in tokenizer_files],
         'sources':source_rows,'structural_shortfall_slots':structural_shortfalls,
         'mapping_checks':mapping_checks,'actual_evaluation_questions':0,'model_backend_created':False,
         'new_solver_calls':0,'new_training_runs':0,'actual_question_request_token_checks':0,
         'full_evaluation_ready':False,
         'limits':['No semantic admission, exclusion or unique-answer verdict is made by this script.',
                   'Dialogue-only token lengths omit the question and chat template; actual full prompts still require zero-truncation checks.',
                   'Synthetic checks validate mapping and syntax only; they never count as generated evaluation data.']}
    (HERE/'audit/SOURCE_AND_MAPPING_PREFLIGHT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'source_count':len(source_rows),
          'structural_shortfalls':len(structural_shortfalls),'mapping_cases':len(mapping_checks),
          'answer_roundtrips':sum(r['answer_roundtrip_checks'] for r in mapping_checks),
          'source_text_token_range':[min(r['source_text_tokens_without_question_or_template'] for r in source_rows),
                                     max(r['source_text_tokens_without_question_or_template'] for r in source_rows)],
          'full_evaluation_ready':False},ensure_ascii=False))

if __name__ == '__main__': main()
