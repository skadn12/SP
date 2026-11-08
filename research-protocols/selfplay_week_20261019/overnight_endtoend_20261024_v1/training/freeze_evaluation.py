"""Freeze an isolated, historically exposed DEV-only shared greedy evaluation."""
from __future__ import annotations
from common import *


def main():
    from transformers import AutoTokenizer
    original = WEEK / 'day5_training_v1/evaluation'
    out = HERE / 'evaluation'
    freeze_path = out / 'EVALUATION_FREEZE.json'
    require(not freeze_path.exists(), 'EVALUATION_ALREADY_FROZEN')
    requests = lines(original / 'DEV_REQUESTS.jsonl')
    gold = {r['candidate_id']: r for r in lines(original / 'private/DEV_GOLD.jsonl')}
    maps = read(original / 'DEV_DISPLAY_MAPS.json')
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    require(len(gold) == 42, 'EXISTING_DEV_COUNT_DRIFT')
    final, periodic, private, checks, sources = [], [], [], [], {}
    public = {q['candidate_id']: q for q in lines(original / 'DEV_PUBLIC_QUESTIONS.jsonl')}
    for cid in sorted(gold):
        require(gold[cid]['split'] == 'dev', 'NON_DEV_GOLD')
        variants = {}
        for r in requests:
            if r['candidate_id'] == cid:
                require(r['split'] == 'dev' and r['branch'] == 'direct', 'NON_DEV_REQUEST')
                variants.setdefault(r['variant_id'], r)
        chosen = [variants[v] for v in sorted(variants)[:2]]
        require(len(chosen) == 2, 'TWO_DEV_VARIANTS_REQUIRED')
        signatures = set()
        for j, r in enumerate(chosen):
            p = Path(r['prompt_file']); require(sha(p) == r['prompt_sha256'], 'OLD_PROMPT_CHANGED')
            m = maps[r['prompt_id']]; mapping = m['display_to_content']
            require(len(mapping) == len(set(mapping.values())), 'DEV_MAPPING_NOT_BIJECTIVE')
            signatures.add(tuple(mapping.values()))
            ids = tokenizer(p.read_text(), add_special_tokens=False).input_ids
            require(len(ids) == r['prompt_tokens'] and len(ids) + 32 <= 16384, 'DEV_FULL_TOKEN_DRIFT')
            new = {k: r[k] for k in ('candidate_id', 'root_id', 'cohort_id', 'question_type', 'variant_id', 'prompt_sha256', 'prompt_tokens', 'prompt_file')}
            new.update(request_id='oet1-dev-' + canonical_sha([cid, r['variant_id']])[:24], split='dev',
                       display_to_content=mapping, prompt_binding=binding(p), prompt_token_ids_sha256=canonical_sha(ids))
            final.append(new)
            if j == 0: periodic.append(new)
            checks.append({'request_id': new['request_id'], 'prompt_tokens': len(ids), 'mapping_bijective': True, 'full_source_untruncated': True})
        require(len(signatures) == 2, 'DEV_VARIANTS_NOT_DISTINCT')
        g = gold[cid]
        private.append({k: g[k] for k in ('candidate_id', 'root_id', 'cohort_id', 'question_type', 'gold_content_ids', 'reporting_role')})
        q = public[cid]; require(sha(q['source_path']) == q['source_sha256'], 'DEV_SOURCE_CHANGED')
        sources[q['source_path']] = binding(q['source_path'])
    save_lines(out / 'FINAL_REQUESTS.jsonl', final)
    save_lines(out / 'PERIODIC_REQUESTS.jsonl', periodic)
    save_lines(out / 'private/GOLD.jsonl', private)
    save(out / 'TOKENIZATION_CHECKS.json', {'status': 'PASS', 'checks': checks})
    input_paths = [original / name for name in ('DEV_REQUESTS.jsonl', 'DEV_PUBLIC_QUESTIONS.jsonl', 'DEV_DISPLAY_MAPS.json', 'private/DEV_GOLD.jsonl', 'DEV_REQUEST_FREEZE.json')]
    save(freeze_path, {'schema_version': 'oet1_shared_dev_greedy_42x2_v1', 'status': 'FROZEN_BEFORE_TRAINING', 'created_at_utc': now(),
                      'models': ['B1', 'B1_GENERIC', 'B1_TARGETED'], 'question_count': len(gold), 'root_count': len({r['root_id'] for r in final}),
                      'final_requests': binding(out / 'FINAL_REQUESTS.jsonl'), 'periodic_requests': binding(out / 'PERIODIC_REQUESTS.jsonl'),
                      'private_gold': binding(out / 'private/GOLD.jsonl'), 'tokenization_checks': binding(out / 'TOKENIZATION_CHECKS.json'),
                      'final_request_count': len(final), 'periodic_request_count': len(periodic),
                      'question_strata': dict(collections.Counter(x['reporting_role'] for x in private)),
                      'input_bindings': [binding(p) for p in input_paths] + list(sources.values()),
                      'decoding': {'runtime': 'transformers.generate', 'do_sample': False, 'num_beams': 1, 'max_new_tokens': 32,
                                   'repetition_penalty': 1., 'length_penalty': 1., 'stop_at_first_closed_box': True,
                                   'strict_box': True, 'finite_answer_constraint': False, 'seed': 20261024},
                      'source_truncation': False, 'test_requests': 0, 'checkpoint_selection': 'FIXED_FINAL_NO_DEV_SELECTION',
                      'historically_exposed': True, 'unseen_root_claim': False, 'old_sampling_scores_not_combined': True,
                      'controller_and_challenger_access': 'aggregate counts/hash only; no item facts, Gold, responses, or diagnosis',
                      'wrapper': binding(__file__), 'protocol': binding(PROTOCOL)})
    print(json.dumps({'status': 'FROZEN', 'questions': len(gold), 'final_requests': len(final), 'periodic_requests': len(periodic), 'freeze': binding(freeze_path)}))

if __name__ == '__main__': main()
