#!/usr/bin/env python3
"""Eight frozen no-dialogue controls; main pilot is imported read-only.

Prepare/analyze may read private scoring. Run uses only public requests,
prompts, schemas and bindings, and delegates unchanged B1 generation to the
primary runner. This is a supplemental shortcut diagnostic, not a new test set.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PRIMARY = HERE.parent
OUT = HERE / 'execution'

spec = importlib.util.spec_from_file_location('question_only_primary_read_only', PRIMARY / 'run.py')
primary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = primary
spec.loader.exec_module(primary)
read, save, textfile = primary.read, primary.save, primary.textfile
need, binding, check = primary.need, primary.binding, primary.check
COUNT = 8


QUESTION_ONLY_INSTRUCTION = """No source dialogue or source references are supplied in this condition.
Give your best prediction using the question and displayed items only. For each
displayed label, give a judgment (supported, contradicted, not_supported or
ambiguous) and a concise reason (at most 320 characters). Here supported means
your best assessment from the visible wording, not a verified source claim.
Every turn_refs array must be empty []; do not invent or cite hidden dialogue.
Then provide answer_reason (at most 800 characters), explaining the selected
set. Finally provide final_answer in the format <box>A B C</box>: selected labels
for multi_select. The example only illustrates format. Your final response must
be JSON, with item_evidence first, answer_reason second and final_answer last.
Keep the explanations in the final JSON brief and based on visible wording."""


def no_dialogue_messages(row):
    """Keep the full condition's visible task block verbatim before instructions."""
    full = primary.original_messages(Path(row['prompt_path']).read_text())[1]['content']
    original_common = primary.orig.diagnostic.COMMON.replace(
        'Return only JSON, with\nitem_evidence first, answer_reason second and final_answer last. Provide brief\nevidence-based explanations, not extended private deliberation.',
        'Your final response must be JSON, with item_evidence first, answer_reason second and final_answer last. Keep the explanations in the final JSON brief and evidence-based.')
    prefix = (f"Question ({row['question_type']}):\n{row['question']}\n\nDisplayed items:\n"
              + '\n'.join(f'{label}. {text}' for label, text in row['display'].items())
              + f"\n\nRequested operation: {row['task_operation']}\n")
    need(full.count('\n\n' + prefix) == 1, 'VISIBLE_TASK_PREFIX_DRIFT')
    task = full.split('\n\n' + prefix, 1)[1]
    need(task.endswith('\n\n' + original_common), 'FULL_INSTRUCTION_DRIFT')
    response_instruction = task[:-len('\n\n' + original_common)]
    need(response_instruction == 'Return every selected option letter; do not infer a required number of choices.', 'RESPONSE_INSTRUCTION_DRIFT')
    user = prefix + response_instruction + '\n\n' + QUESTION_ONLY_INSTRUCTION
    need('Dialogue:\n' not in user and 'R03:' not in user and 'R02:' not in user, 'SOURCE_CONTEXT_LEAK')
    return [
        {'role': 'system', 'content': 'Make a best prediction from the question and displayed items. No dialogue is supplied. Never fabricate source references. Give concise, verifiable explanations before the final boxed answer.'},
        {'role': 'user', 'content': user},
    ]


def prepare():
    need(not OUT.exists(), 'NO_PREPARATION_OVERWRITE')
    base_prep, base_rows, _ = primary.public()
    selected = [r for r in base_rows if r['cohort'] == 'prototype_difficulty' and r['format'] == 'ORIGINAL']
    need(len(selected) == COUNT and {r['candidate_id'] for r in selected} == {f'MC-PROT-{i:02}' for i in range(1, 5)}, 'FROZEN_SELECTION_DRIFT')
    need(all(sum(s['candidate_id'] == r['candidate_id'] for s in selected) == 2 for r in selected), 'PERMUTATION_COUNT_DRIFT')
    primary.check(base_prep['private_scoring_binding'])
    base_private = {r['request_id']: r for r in read(PRIMARY / 'execution/PRIVATE_SCORING.json')['requests']}
    endpoint, config = primary.orig.endpoint_receipt()
    need(config == base_prep['config_receipt'] and primary.orig.versions() == base_prep['runtime_versions'], 'RUNTIME_CONFIG_DRIFT')
    from transformers import AutoTokenizer
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    tok = AutoTokenizer.from_pretrained(str(primary.orig.TOKENIZER), local_files_only=True)
    reasoner = Qwen3ReasoningParser(tokenizer=tok)
    rows, tokens, scoring, public_bindings, checks = [], {}, [], [], []
    for base in selected:
        messages = no_dialogue_messages(base)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(prompt, add_special_tokens=False)
        need(len(ids) + primary.MAX_TOKENS <= primary.MODEL_LEN, 'CONTEXT_OVERFLOW_NO_TRUNCATION')
        need(reasoner.is_reasoning_end(ids), 'THINK_OFF_GATE_FAILURE')
        rid = f"{base['candidate_id']}__{base['variant_id']}__QUESTION_ONLY"
        pp, sp = OUT / 'prompts' / (rid + '.txt'), OUT / 'schemas' / (rid + '.json')
        schema = primary.orig.base.question_only_schema(base['labels'], base['question_type'])
        need(all(x['properties']['turn_refs']['maxItems'] == 0 for x in schema['properties']['item_evidence']['properties'].values()), 'SCHEMA_REFERENCE_LEAK')
        textfile(pp, prompt)
        save(sp, schema)
        row = {k: base[k] for k in ('candidate_id', 'root_id', 'question_type', 'task_operation', 'question', 'variant_id', 'display', 'sampling_seed', 'labels')}
        row.update(request_id=rid, cohort='prototype_question_only', format='QUESTION_ONLY', condition='QUESTION_ONLY',
                   mode='THINK_OFF', original_request_id=base['request_id'], prompt_path=str(pp), schema_path=str(sp),
                   prompt_tokens=len(ids), messages_sha256=hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
        rows.append(row)
        tokens[rid] = ids
        private = base_private[base['request_id']]
        scoring.append({'request_id': rid, 'original_request_id': base['request_id'],
                        **{k: private[k] for k in ('display_to_canonical', 'canonical_gold', 'display_gold')}})
        public_bindings.extend([binding(pp), binding(sp), binding(base['prompt_path']), binding(base['schema_path'])])
        checks.append({'request_id': rid, 'original_request_id': base['request_id'],
                       'identical_question_display_operation_seed': all(row[k] == base[k] for k in ('question', 'display', 'task_operation', 'sampling_seed', 'labels', 'variant_id')),
                       'dialogue_supplied': False, 'references_allowed': False, 'prompt_tokens': len(ids)})
    need(all(x['identical_question_display_operation_seed'] for x in checks), 'PAIR_CONTROLS_DRIFT')
    save(OUT / 'PUBLIC_REQUESTS.json', {'requests': rows})
    save(OUT / 'PROMPT_TOKEN_IDS.json', tokens)
    save(OUT / 'PRIVATE_SCORING.json', {'requests': scoring})
    (OUT / 'PRIVATE_SCORING.json').chmod(0o600)
    protocol = {
        'experiment_id': HERE.name, 'parent_experiment': str(PRIMARY), 'exploratory': True,
        'independent_validation': False, 'purpose': 'Compare full-context performance with question/options-only performance to diagnose shortcuts; not proof of overall difficulty or actual model shortcut use.',
        'selection_rule': 'Every primary PUBLIC_REQUESTS row with cohort=prototype_difficulty and format=ORIGINAL; no primary Solver scores read.',
        'semantic_questions': 4, 'source_roots': 1, 'dependent_permutations_per_question': 2,
        'requests': COUNT, 'condition': 'QUESTION_ONLY', 'source_context_supplied': False,
        'paired_controls': ['question', 'display permutation', 'operation', 'response_instruction', 'gold', 'sampling seed', 'sampling settings', 'output budget'],
        'intervention': 'Remove complete dialogue; replace source-grounded instructions with best-effort visible-wording instructions and require empty turn_refs. Therefore this is a context-ablation diagnostic, not a pure rendering comparison.',
        'sampling': {'temperature': .8, 'top_p': .9, 'top_k': -1, 'repetition_penalty': 1., 'engine_seed': 0},
        'max_tokens': primary.MAX_TOKENS, 'max_model_len': primary.MODEL_LEN, 'batch_size': primary.BATCH,
        'thinking': False, 'structured_outputs_config': primary.engine_module.STRUCTURED_CONFIG,
        'prefix_caching': False, 'automatic_retries': 0, 'source_truncation': False, 'training': False,
        'primary_mc_partial': 'J = TP/g - FP/(n-g); negative values retained, no clipping. J*100 is a score, not accuracy.',
        'invalid_policy': 'Strict exact=0 with all 8 frozen requests as denominator; original partial J=None/unscored for invalid outputs. Report valid J denominator.',
        'limits': 'Four previously exposed semantic questions from one root, two dependent permutations and one checkpoint; high question-only performance indicates answerability without dialogue but does not establish which strategy full-context Solver used. Failure does not prove absence of shortcuts. No population-difficulty, equivalence or training-effect claim.'}
    save(HERE / 'PROTOCOL_FREEZE.json', protocol)
    save(HERE / 'CPU_PREPARATION_AUDIT.json', {'checks': checks, 'model_calls': 0, 'weights_loaded': False, 'primary_files_modified': False,
          'schema_function': 'primary.orig.base.question_only_schema', 'plan_frozen_before_supplement_inference': True})
    public_bindings.extend(binding(p) for p in [Path(__file__), PRIMARY / 'run.py', HERE / 'PROTOCOL_FREEZE.json', OUT / 'PUBLIC_REQUESTS.json', OUT / 'PROMPT_TOKEN_IDS.json', PRIMARY / 'execution/PUBLIC_REQUESTS.json'])
    public_bindings.extend(binding(p) for p in primary.orig.code_paths())
    public_bindings.append(binding(primary.WEEK / 'thinking_switch_20261024_structured_v2/run.py'))
    save(OUT / 'PREPARATION.json', {'requests': COUNT, 'fixed_order': [r['request_id'] for r in rows],
          'bindings': list({b['path']: b for b in public_bindings}.values()),
          'private_scoring_binding': binding(OUT / 'PRIVATE_SCORING.json'),
          'primary_preparation_binding': binding(PRIMARY / 'execution/PREPARATION.json'),
          'runtime_versions': primary.orig.versions(), 'config_receipt': config,
          'max_prompt_tokens': max(r['prompt_tokens'] for r in rows)})
    print(json.dumps({'prepared': COUNT, 'max_prompt_tokens': max(r['prompt_tokens'] for r in rows), 'model_calls': 0}), flush=True)


def public():
    prep = read(OUT / 'PREPARATION.json')
    forbidden = {str((OUT / 'PRIVATE_SCORING.json').resolve()), str((PRIMARY / 'execution/PRIVATE_SCORING.json').resolve())}
    for b in prep['bindings']:
        need(b['path'] not in forbidden and '/private_maps/' not in b['path'], 'PRIVATE_BINDING_IN_GENERATION')
        check(b)
    rows = read(OUT / 'PUBLIC_REQUESTS.json')['requests']
    tokens = read(OUT / 'PROMPT_TOKEN_IDS.json')
    need(prep['fixed_order'] == [r['request_id'] for r in rows] and len(rows) == len(tokens) == COUNT, 'PUBLIC_SET_DRIFT')
    for row in rows:
        need(not ({'gold', 'canonical_gold', 'display_gold', 'display_to_canonical', 'citation_to_original'} & set(row)), 'PRIVATE_PUBLIC_FIELD')
        need(row['condition'] == row['format'] == 'QUESTION_ONLY', 'CONDITION_DRIFT')
        need(row['prompt_tokens'] == len(tokens[row['request_id']]) and row['prompt_tokens'] + primary.MAX_TOKENS <= primary.MODEL_LEN, 'TOKEN_BUDGET_DRIFT')
    return prep, rows, tokens


def run():
    # This changes only the imported Python module's in-memory output/callback;
    # primary source and frozen execution artifacts are never changed.
    primary.HERE, primary.OUT, primary.public = HERE, OUT, public
    primary.run()


def analyze():
    prep, rows, _ = public()
    freeze = read(OUT / 'RAW_RESPONSE_FREEZE.json')
    need(freeze['request_ids'] == prep['fixed_order'], 'RAW_FREEZE_SET_DRIFT')
    for b in freeze['bindings']:
        check(b)
    check(prep['private_scoring_binding'])
    private = {r['request_id']: r for r in read(OUT / 'PRIVATE_SCORING.json')['requests']}
    # Read scored FULL results only now; no adaptive selection or rerunning.
    full_score_path = PRIMARY / 'execution/SCORES.json'
    full_doc = read(full_score_path)
    check(full_doc['raw_freeze_binding'])
    full_freeze = read(PRIMARY / 'execution/RAW_RESPONSE_FREEZE.json')
    for b in full_freeze['bindings']:
        check(b)
    full_records = {r['request_id']: r for r in full_doc['records']}
    records, paired = [], []
    for row in rows:
        raw = read(OUT / 'raw' / (row['request_id'] + '.json'))
        scoring = private[row['request_id']]
        valid, parsed, answer, errors = False, None, [], []
        try:
            need(raw['status'] == 'RETURNED' and raw.get('finish_reason') == 'stop', 'BACKEND_OR_LENGTH_FAILURE')
            need(not raw.get('capture_errors'), 'CAPTURE_ERRORS')
            thinking, final, _ = primary.engine_module.split_output(raw['raw_output'])
            need(not thinking, 'UNEXPECTED_THINKING')
            parsed, answer = primary.orig.base.validate_output(final, row, read(row['schema_path']))
            valid = True
        except Exception as e:
            errors.append(str(e))
        canonical = sorted(scoring['display_to_canonical'][a] for a in answer) if valid else []
        gold = scoring['canonical_gold']
        tp, fp = len(set(canonical) & set(gold)), len(set(canonical) - set(gold))
        g, n = len(gold), len(row['labels'])
        need(0 < g < n, 'J_REQUIRES_NONDEGENERATE_GOLD')
        j = tp / g - fp / (n - g) if valid else None
        exact = valid and set(canonical) == set(gold)
        record = {k: row[k] for k in ('request_id', 'candidate_id', 'cohort', 'root_id', 'question_type', 'task_operation', 'variant_id', 'format', 'prompt_tokens', 'original_request_id')}
        record.update(valid=valid, exact=exact, canonical_prediction=canonical, canonical_gold=gold,
                      TP=tp, FP=fp, FN=len(set(gold) - set(canonical)), J=j, pairwise_accuracy=None,
                      errors=errors, parsed=parsed, citations=[], finish_reason=raw.get('finish_reason'), output_tokens=len(raw.get('output_token_ids', [])))
        records.append(record)
        full = full_records[row['original_request_id']]
        need(full['canonical_gold'] == gold and full['format'] == 'ORIGINAL' and full['cohort'] == 'prototype_difficulty', 'FULL_BASELINE_MISMATCH')
        paired.append({'candidate_id': row['candidate_id'], 'variant_id': row['variant_id'], 'original_request_id': row['original_request_id'],
                       'question_only_request_id': row['request_id'], 'full_valid': full['valid'], 'question_only_valid': valid,
                       'full_exact': full['exact'], 'question_only_exact': exact, 'full_J': full['J'], 'question_only_J': j,
                       'J_difference_question_only_minus_full': j - full['J'] if j is not None and full['J'] is not None else None,
                       'same_canonical_prediction': full['valid'] and valid and full['canonical_prediction'] == canonical})
    selected_full = [full_records[r['original_request_id']] for r in rows]
    questions = []
    for cid in dict.fromkeys(r['candidate_id'] for r in rows):
        rs = [r for r in records if r['candidate_id'] == cid]
        fs = [r for r in selected_full if r['candidate_id'] == cid]
        questions.append({'candidate_id': cid, 'root_id': rs[0]['root_id'], 'gold_count': len(rs[0]['canonical_gold']),
                          'FULL_ORIGINAL': primary.summarize(fs), 'QUESTION_ONLY': primary.summarize(rs),
                          'question_only_permutation_consistency': all(r['valid'] for r in rs) and len({tuple(r['canonical_prediction']) for r in rs}) == 1})
    save(OUT / 'SCORES.json', {'records': records, 'raw_freeze_binding': binding(OUT / 'RAW_RESPONSE_FREEZE.json')})
    summary = {'summaries': {'FULL_ORIGINAL': primary.summarize(selected_full), 'QUESTION_ONLY': primary.summarize(records)},
               'paired': paired, 'questions': questions, 'requests': COUNT, 'unique_semantic_questions': 4, 'source_roots': 1,
               'model_calls': len(freeze['submitted_ids']), 'returned': freeze['returned'], 'failure': freeze['failure'],
               'wall_seconds': freeze['wall_seconds'], 'total_prompt_tokens': sum(r['prompt_tokens'] for r in records),
               'total_output_tokens': sum(r['output_tokens'] for r in records), 'full_scores_binding': binding(full_score_path),
               'limits': read(HERE / 'PROTOCOL_FREEZE.json')['limits']}
    save(OUT / 'ANALYSIS_SUMMARY.json', summary)
    print(json.dumps({'summaries': summary['summaries'], 'model_calls': summary['model_calls'], 'returned': summary['returned']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    globals()[parser.parse_args().action]()
