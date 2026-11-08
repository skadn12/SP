"""Prepare score-independent Train-only Day 5 diagnosis; never instantiate a backend."""
from __future__ import annotations
import collections
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WEEK = HERE.parents[1]
OLD = WEEK / 'day3_solver187_v1/runtime/direct_v2'
SEED = 'day5-train-diagnosis-v1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def binding(p):
    p = Path(p).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def save(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
        f.write('\n')


def hashkey(s):
    return hashlib.sha256((SEED + '|' + s).encode()).hexdigest()


def import_file(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def main():
    if (HERE / 'PREPARATION_FREEZE.json').exists():
        raise RuntimeError('PREPARATION_ALREADY_FROZEN')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B', local_files_only=True)
    policy = read(HERE / 'POLICY.json')
    selection = read(WEEK / 'day4_schedule_v1/SELECTION.json')
    metadata = {q['candidate_id']: q for q in read(WEEK / 'day3_solver187_v1/CANDIDATE_METADATA.json')['questions']}
    allowed = set(read(WEEK / 'day1/TRAIN_ROOTS.json')['root_ids'])
    forbidden = set()
    for c in read(WEEK / 'day4_schedule_v1/EVALUATION_LOCK.json')['cohorts']:
        forbidden.update(c['root_ids'])
    selected_sets = {a: set(selection['selected_new_' + a]) for a in ('A', 'B')}
    paired = [r for r in selection['paired_slots'] if r['A'] in selected_sets['A'] and r['B'] in selected_sets['B']]
    root_pairs = collections.defaultdict(list)
    for r in paired:
        m = metadata[r['A']]
        assert metadata[r['B']]['root_id'] == m['root_id']
        root_pairs[m['root_id']].append(r)
    rank = sorted(root_pairs, key=hashkey)
    chosen, slots = [], []
    for root in rank:
        idx = len(chosen)
        mc_design = ('MC_BINDING', 'MC_FINAL_STATE_REVISION')[idx % 2]
        order_design = ('ORDER_3_2_1', 'ORDER_2_2_2', 'ORDER_4_2')[idx % 3]
        choices = []
        for typ, design in [('multi_select', mc_design), ('ordering', order_design)]:
            candidates = [r for r in root_pairs[root] if metadata[r['A']]['question_type'] == typ and metadata[r['A']]['design_id'] == design]
            if not candidates:
                break
            choices.append(min(candidates, key=lambda r: hashkey(r['slot_id'])))
        if len(choices) == 2:
            chosen.append(root)
            slots.extend(choices)
        if len(chosen) == 6:
            break
    assert len(chosen) == 6 and set(chosen) <= allowed and not set(chosen) & forbidden
    cids = [r[a] for r in slots for a in ('A', 'B')]
    assert len(cids) == len(set(cids)) == 24
    manifest = read(OLD / 'manifest.json')
    public = {q['candidate_id']: q for q in read(OLD / 'PUBLIC_QUESTIONS.json')['questions'] if q['candidate_id'] in cids}
    maps = read(OLD / 'DISPLAY_MAPPINGS.json')
    direct = [r for r in manifest['requests'] if r['candidate_id'] in cids]
    assert len(direct) == 360
    direct_maps = {'requests': {r['request_id']: maps['requests'][r['request_id']] for r in direct},
                   'public_to_canonical': {cid: maps['public_to_canonical'][cid] for cid in cids}}
    prompt_bindings = []
    for r in direct:
        p = OLD / 'prompts' / (r['request_id'] + '.txt')
        text = p.read_text()
        ids = tok.encode(text, add_special_tokens=False)
        assert sha(p) == r['prompt_sha256'] and len(ids) == r['prompt_tokens'] and len(ids) + 32 <= 16384
        dest = HERE / 'public/prompts' / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        r['prompt_path'] = str(dest)
        prompt_bindings.append(binding(p))
    evidence_module = import_file('day5_prepared_brief_evidence', WEEK / 'day3_diagnostic_handoff_v1/runtime/evidence_handoff.py')
    evidence = []
    evidence_maps = {}
    for r in direct:
        if r['variant_id'] != 'perm_00' or r['slot'] != 0:
            continue
        q = public[r['candidate_id']]
        source = read(q['source_path'])
        assert sha(q['source_path']) == q['source_sha256']
        v = maps['requests'][r['request_id']]
        text = evidence_module.prompt(tok, q, v, source)
        rid = r['request_id'].replace('__direct__', '__brief_evidence__')
        p = HERE / 'public/prompts' / (rid + '.txt')
        p.write_text(text)
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) + 2048 <= 16384
        row = {k: v for k, v in r.items() if k != 'prompt_path'}
        row.update(request_id=rid, branch='brief_evidence', direct_request_id=r['request_id'], prompt_path=str(p),
                   prompt_sha256=sha(p), prompt_tokens=len(ids), max_output_tokens=2048, json_schema=evidence_module.schema(q))
        evidence.append(row)
        evidence_maps[rid] = v
    assert len(evidence) == 24
    annotations = []
    for cid in cids:
        m = metadata[cid]
        annotations.append({'candidate_id': cid, 'root_id': m['root_id'], 'source_arm': m['arm'],
            'question_type': m['question_type'], 'design_id': m['design_id'], 'fact_family_id': m['fact_family_id'],
            'model_training_exposure': {role + '1': 'OWN_ARM_TRAINED_QUESTION' if cid in selected_sets[role] else 'OPPOSITE_ARM_QUESTION_SHARED_ROOT_MAY_SHARE_FACTS' for role in ('A', 'B')},
            'not_heldout_generalization': True, 'source_sha256': public[cid]['source_sha256']})
    save(HERE / 'public/PUBLIC_QUESTIONS.json', {'questions': list(public.values())})
    save(HERE / 'public/DIRECT_REQUESTS.json', {'requests': direct})
    save(HERE / 'public/DIRECT_DISPLAY_MAPS.json', direct_maps)
    save(HERE / 'public/BRIEF_EVIDENCE_REQUESTS.json', {'requests': evidence, 'maps': evidence_maps,
        'system': evidence_module.SYSTEM, 'task': evidence_module.TASK, 'constrained_decoding': False,
        'meaning': 'Observable auditable judgments, citations and final answer; not a faithful internal reasoning trace.'})
    save(HERE / 'SELECTION.json', {'status': 'FIXED_BEFORE_A1_B1_SOLVER', 'seed': SEED, 'root_hash_rank': rank,
        'selected_roots': chosen, 'paired_slots': slots, 'question_annotations': annotations,
        'selection_reads_scores': False, 'new_solver_calls': 0})
    # Keep S0 results outside the public Solver input. Selection never reads scores.
    raw_binding = [binding(OLD / 'responses' / (r['request_id'] + '.json')) for r in direct]
    save(HERE / 'private/S0_BASELINE_BINDINGS.json', {'status': 'REUSE_EXACT_OLD_REQUESTS_ONLY',
        'manifest': binding(OLD / 'manifest.json'), 'raw_response_freeze': binding(OLD / 'RAW_RESPONSE_FREEZE.json'),
        'scores_binding_not_used_for_selection': binding(OLD / 'SCORES.json'), 'selected_raw_responses': raw_binding,
        'gold_binding': binding(WEEK / 'day3_solver187_v1/runtime/direct_v2_private/PRIVATE_QUESTIONS.json'),
        'new_S0_calls_required': 0, 'brief_evidence_S0_baseline': 'NOT_AVAILABLE_SAME_PROMPT_FOR_THIS24_NO_RECOMPUTATION_PLANNED'})
    probe = read(WEEK / 'day2_order_probe_v1/run/manifest.json')
    repl = [r for r in read(WEEK / 'day2_order_probe_v1/SELECTION.json')['questions'] if r['cohort'] == 'replication']
    repl = sorted(repl, key=lambda r: hashkey('controlled-probe|' + r['candidate_id']))[:3]
    pcids = {r['candidate_id'] for r in repl}
    prows = [r for r in probe['requests'] if r['candidate_id'] in pcids and r['variant_id'] in ('perm_00', 'perm_01')]
    assert len(prows) == 12 and {r['root_id'] for r in prows} <= allowed and not {r['root_id'] for r in prows} & forbidden
    for r in prows:
        p = WEEK / 'day2_order_probe_v1/run/prompts' / (r['request_id'] + '.txt')
        assert sha(p) == r['prompt_sha256']
        r['prompt_path'] = str(p)
    save(HERE / 'CONTROLLED_PROBE_RESERVE.json', {'status': 'PREPARED_NOT_RELEASED_NOT_EXECUTED',
        'selected_replication_questions': repl, 'requests_per_model': prows, 'requests_per_model_count': 12,
        'maximum_A1_B1_requests_total': 24, 'remaining_day6_probe_cap_if_all_used': 0,
        'selection_independent_of_scores': True, 'source_protocol': binding(WEEK / 'day2_order_probe_v1/PROTOCOL.json'),
        'release_gate': 'Only after direct records and source-reviewed residual errors establish a concrete unresolved ordering hypothesis; a separately audited release freeze must bind exact model endpoints. No automatic execution or extra mappings.',
        'not_sufficient_by_itself': 'Three roots and two repeated mappings per root do not guarantee controlled support, faithful internal reasoning or reproduced admission.',
        'S0_exact_baseline': [binding(WEEK / 'day2_order_probe_v1/run/responses' / (r['request_id'] + '.json')) for r in prows]})
    files = [binding(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'PREPARATION_FREEZE.json' and '__pycache__' not in p.parts]
    inputs = [binding(WEEK / p) for p in ('DAILY_EXECUTION_V2_ZH.md', 'FEEDBACK_CONTRACT_V2.json', 'day1/TRAIN_ROOTS.json',
        'day4_schedule_v1/SELECTION.json', 'day4_schedule_v1/EVALUATION_LOCK.json', 'day3_solver187_v1/CANDIDATE_METADATA.json',
        'day2_order_probe_v1/PROTOCOL.json', 'day2_order_probe_v1/SELECTION.json',
        'day3_diagnostic_handoff_v1/runtime/evidence_handoff.py', 'day2/solver/adapter.py')]
    save(HERE / 'PREPARATION_FREEZE.json', {'status': 'CPU_PREPARED_ENDPOINT_BINDINGS_AND_INDEPENDENT_AUDIT_REQUIRED',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files, 'upstream_bindings': inputs,
        'upstream_direct_prompt_bindings': prompt_bindings, 'namespace': 'train.feedback.day5_stage1',
        'questions': 24, 'roots': 6, 'direct_requests_per_model': 360, 'direct_requests_total': 720,
        'brief_evidence_max_per_model': 24, 'brief_evidence_max_total': 48, 'controlled_probe_shared_day5_day6_cap': 24,
        'new_solver_calls': 0, 'model_endpoint_hashes': 'NOT_YET_BOUND', 'source_gold_content_changed': False,
        'dev_test_requests': 0, 'public_gold_disjoint': True, 'policy_binding': binding(HERE / 'POLICY.json')})
    print(json.dumps({'status': 'CPU_PREPARED', 'questions': 24, 'roots': chosen, 'direct_per_model': 360, 'brief_evidence_per_model': 24, 'new_solver_calls': 0}))


if __name__ == '__main__':
    main()
