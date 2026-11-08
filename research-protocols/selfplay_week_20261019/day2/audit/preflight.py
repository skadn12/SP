"""Independent, public-only reconstruction of prepared Day 2 inference requests."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOLVER = HERE.parent / 'solver'
DAY1 = HERE.parents[1] / 'day1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOLVER))
import adapter
import run as runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    check_names = []
    def check(value, name):
        if not value:
            raise AssertionError(name)
        check_names.append(name)

    # Runtime performs complete byte checks; independently rebuild public requests below.
    m = runtime.verify(args.run)
    check(m['namespace'] == 'train.feedback' and not m['use_as_final_eval'], 'train_feedback_only')
    check(m['solver_model'] == 'S0_M_clean', 'fixed_S0')
    check(m['batch_size'] == 8 and not m['automatic_retry'] and not m['source_truncation'], 'runtime_policy')
    check(m['content_admission_binding'] is not None, 'admission_bound')
    check(runtime.read(m['content_admission_binding']['path'])['status'] == 'PASS', 'prior_content_pass')
    qs = runtime.read(args.run / 'PUBLIC_QUESTIONS.json')['questions']
    public0 = runtime.read(DAY1 / 'diagnosis_inputs/PUBLIC_QUESTIONS.json')['questions']
    check({q['candidate_id']: q for q in qs} == {q['candidate_id']: q for q in public0}, 'unchanged_39_public_inputs')
    allowed = set(runtime.read(DAY1 / 'evaluation/ROOT_SELECTION.json')['wave1']['root_ids'])
    check({q['root_id'] for q in qs} == allowed, '20_train_roots')
    check(Counter(q['question_type'] for q in qs) == {'multi_select': 19, 'ordering': 20}, 'types_19_20')
    maps = runtime.read(args.run / 'private/DISPLAY_MAPPINGS.json')
    ids = runtime.read(args.run / 'PROMPT_TOKEN_IDS.json')
    byqid = defaultdict(list)
    for r in m['requests']:
        byqid[r['candidate_id']].append(r)
    check(len({r['request_id'] for r in m['requests']}) == len(m['requests']), 'unique_requests')
    check(set(maps['requests']) == set(ids) == {r['request_id'] for r in m['requests']}, 'complete_mapping_tokenids')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
    spec = importlib.util.spec_from_file_location('independent_original_protocol', adapter.NATIVE / 'protocol.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    direct = m['branch'] == 'direct'
    check(len(m['requests']) == (582 if direct else 39), 'fixed_denominator')
    for q in qs:
        qid = q['candidate_id']
        check(set(q) <= adapter.PUBLIC_KEYS, 'public_whitelist:' + qid)
        source = runtime.read(q['source_path'])
        check(runtime.sha(q['source_path']) == q['source_sha256'], 'source_sha:' + qid)
        check(runtime.sha(args.run / 'sources' / (q['root_id'] + '.json')) == q['source_sha256'], 'copied_source_sha:' + qid)
        rows = byqid[qid]
        nvariants = (6 if q['question_type'] == 'multi_select' else 12) if direct else 1
        variants = original.make_variants(q, seed=2026101903, count=nvariants)
        variants = {v['variant_id']: v for v in variants}
        check(len(rows) == ((18 if q['question_type'] == 'multi_select' else 12) if direct else 1), 'per_question_denominator:' + qid)
        seen = set()
        for row in rows:
            rid = row['request_id']
            v = variants[row['variant_id']]
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            check(v == maps['requests'][rid], 'native_gold_blind_variant:' + rid)
            check(row['display'] == {field: v[field]}, 'display_matches:' + rid)
            check(len(set(v['display_to_content'].values())) == 6, 'bijection:' + rid)
            prompt = (args.run / 'prompts' / (rid + '.txt')).read_text()
            check(hashlib.sha256(prompt.encode()).hexdigest() == row['prompt_sha256'], 'prompt_hash:' + rid)
            if rid not in seen:
                prompt2 = adapter.direct_prompt(tok, q, v, source) if direct else adapter.evidence_prompt(tok, q, v, source)
                check(prompt == prompt2, 'public_prompt_reconstruction:' + rid)
                check(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns']), 'all_turns_present:' + rid)
                encoded = tok.encode(prompt, add_special_tokens=False)
                check(encoded == ids[rid], 'exact_tokenization:' + rid)
                seen.add(rid)
            check(row['prompt_tokens'] == len(ids[rid]), 'token_count:' + rid)
            check(row['prompt_tokens'] + row['max_output_tokens'] <= m['max_model_len'], 'no_overflow:' + rid)
            check(row['max_output_tokens'] == (32 if direct else 1024), 'output_budget:' + rid)
            check(not set(v['display_to_content'].values()) & set(prompt.split()), 'private_content_map_absent:' + rid)
            check(row['seed'] == 2026101904 + m['requests'].index(row) + (0 if direct else 100000), 'seed_schedule:' + rid)
    check(not (args.run / 'STARTED.json').exists(), 'not_started_before_audit')
    out = {'status': 'PASS', 'auditor': '/root', 'review_scope': 'Internal independent request reconstruction; shared model/filesystem.',
           'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': len(check_names),
           'new_solver_calls': 0, 'private_gold_json_read_by_this_audit': False,
           'audited_files': [runtime.bind(args.run / 'PREFLIGHT_FREEZE.json'), runtime.bind(Path(__file__))],
           'questions': len(qs), 'requests': len(m['requests']), 'check_names': check_names}
    runtime.save(args.out, out)
    print(json.dumps({k: out[k] for k in ['status', 'checks', 'questions', 'requests']}))


if __name__ == '__main__':
    main()
