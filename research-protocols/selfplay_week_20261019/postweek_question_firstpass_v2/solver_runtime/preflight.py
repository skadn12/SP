"""CPU interface reconstruction for an actual independent reviewing agent."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import run as wrapper


def main():
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviewer-id', required=True)
    ap.add_argument('--prior-exposure-note', required=True)
    args = ap.parse_args()
    assert args.reviewer_id.startswith('/root/') and args.reviewer_id != '/root'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_REVIEW_MUST_MASK_GPU'
    r = wrapper.runtime
    out = wrapper.HERE / 'execution'
    config = argparse.Namespace(output=out)
    prep, rows, token_ids = r.verify(config)
    from transformers import AutoTokenizer
    import xgrammar as xgr
    from jsonschema import Draft7Validator
    endpoint = r.read(prep['endpoint_binding']['path'])
    tokenizer = AutoTokenizer.from_pretrained(endpoint['tokenizer_directory'], local_files_only=True)
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size), max_threads=4)
    results = []
    for row in rows:
        source = r.read(row['source_binding']['path'])
        prompt = r.direct_prompt(tokenizer, row, source) if row['branch'] == 'direct' else r.diag.make_prompt(tokenizer, row, row['display'], source)
        assert prompt == Path(row['prompt_path']).read_text()
        assert tokenizer.encode(prompt, add_special_tokens=False) == token_ids[row['request_id']]
        assert all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in prompt for t in source['turns'])
        assert row['candidate_id'] not in prompt and row['request_id'] not in prompt
        assert 'MC_RECORD_AUDIT_CONTROL' not in prompt and 'all-correct' not in prompt.lower()
        material = 'qfp2:firstpass:20261024:' + row['candidate_id'] + ':' + row['branch']
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], 'big') % 2147483647
        assert seed == row['sampling_seed']
        schema = r.read(row['schema_path'])
        if row['branch'] == 'direct':
            expected = {'choice': r.direct_choices(row['question_type'], row['labels'])}
            assert schema == expected
            assert len(schema['choice']) == (63 if row['question_type'] == 'multi_select' else 24)
        else:
            expected = r.diag.repair.schema(row['labels'], [t['turn_ref'] for t in source['turns']], row['question_type'])
            assert schema == expected
            Draft7Validator.check_schema(schema)
            compiler.compile_json_schema(json.dumps(schema), any_whitespace=True)
        assert row['prompt_tokens'] + row['max_output_tokens'] <= 16384
        results.append({'request_id': row['request_id'], 'full_source_turns': len(source['turns']),
                        'prompt_tokens': row['prompt_tokens'], 'context_headroom': 16384 - row['prompt_tokens'] - row['max_output_tokens'],
                        'prompt_exact_match': True, 'token_ids_exact_match': True, 'schema_exact_match': True,
                        'seed_exact_match': True, 'prompt_binding': r.bind(row['prompt_path'])})
    report = {
        'status': 'PASS_CPU_INTERFACE_ONLY', 'reviewer_id': args.reviewer_id, 'author_id': '/root',
        'independent_of_runtime_author': True, 'questions': prep['questions'], 'requests': len(rows),
        'cpu_only': True, 'gpu_execution_performed': False, 'model_calls': 0,
        'new_raw_or_scores_read': False, 'private_scoring_read': False, 'private_question_drafts_read': False,
        'read_flags_scope': 'This CPU interface reconstruction only; they do not assert lifetime blindness.',
        'prior_exposure_disclosure': args.prior_exposure_note,
        'public_sources_prompts_schemas_and_token_ids_reconstructed': True,
        'max_prompt_tokens': max(r['prompt_tokens'] for r in rows),
        'min_context_headroom': min(16384 - r['prompt_tokens'] - r['max_output_tokens'] for r in rows),
        'elapsed_seconds': time.perf_counter() - started,
        'requests_checked': results,
        'bindings': prep['bindings'] + [r.bind(out / 'PREPARATION.json'), r.bind(__file__)],
        'limits': ['CPU interface reconstruction does not verify question truth, answer Gold or reason semantics.',
                   'The reviewing agent is responsible for truthfully supplying its own identity after inspecting the reconstruction code.',
                   'Frozen imported solver code retains historical internal cohort string FBC1.'],
    }
    r.save(wrapper.HERE / 'PREFLIGHT_REVIEW.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'questions', 'requests', 'max_prompt_tokens', 'min_context_headroom', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
