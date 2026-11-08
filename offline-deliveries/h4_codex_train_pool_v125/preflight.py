"""Validate the sealed package; start training only with an explicit --execute."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v125_builder_helpers', HERE / 'build_package.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check():
    seal = b.read(HERE / 'COMPLETE_MANIFEST.json')
    for row in seal['files']:
        path = HERE / row['path']
        require(path.is_file() and b.sha(path) == row['sha256'], 'PACKAGE_FILE_CHANGED:' + str(path))
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'COMPLETE_MANIFEST.json'}
    require(actual == {r['path'] for r in seal['files']}, 'PACKAGE_UNLISTED_OR_MISSING_FILES')
    for row in b.read(HERE / 'INPUT_FREEZE.json')['files']:
        path = Path(row['path'])
        require(path.is_file() and b.sha(path) == row['sha256'], 'EXTERNAL_INPUT_CHANGED:' + str(path))
    for filename in ['INDEPENDENT_PACKAGE_AUDIT.json', 'FINAL_ENGINEERING_AUDIT.json']:
        audit = b.read(HERE / 'evidence' / filename)
        require(audit['status'] == 'PASS', 'AUDIT_NOT_PASS:' + filename)
        for row in audit['audited_package_files']:
            require(b.sha(HERE / row['path']) == row['sha256'], 'AUDIT_STALE:' + row['path'])

    protocol = b.read(HERE / 'PROTOCOL.json')
    summary = b.read(HERE / 'SUMMARY.json')
    plan = b.read(HERE / 'TRAINING_PLAN.json')
    raw = b.read(HERE / 'data/RAW_SELECTED_QUESTIONS.json')['questions']
    effective = b.read(HERE / 'data/COMBINED_QUESTIONS.json')['questions']
    rows = b.readl(HERE / 'data/TRAIN_NEW_ONLY.jsonl')
    lineages = b.readl(HERE / 'data/QUESTION_LINEAGE.jsonl')
    require(len(rows) == len(raw) == len(effective) == len(lineages) == 363, 'COUNT_MISMATCH')
    require(len({q['candidate_id'] for q in rows}) == 363, 'DUPLICATE_IDS')
    require(protocol['replay']['count'] == summary['replay_question_count'] == 0, 'REPLAY_POLICY_CHANGED')
    require(protocol['length']['max_length'] == 12288, 'LENGTH_POLICY_CHANGED')
    original_inputs = [q for row in protocol['inputs'] for q in b.read(row['path'])['questions']]
    require(raw == original_inputs, 'RAW_CANDIDATE_EXPORT_NOT_LOSSLESS')
    patched = json.loads(json.dumps(raw))
    qmap = {q['candidate_id']: q for q in patched}
    for edit in b.read(HERE / 'evidence/EVIDENCE_OVERLAY.json')['corrections']:
        q = qmap[edit['candidate_id']]
        require(q['option_evidence'][edit['label']] == edit['original_evidence'], 'OVERLAY_ORIGINAL_MISMATCH')
        q['option_evidence'][edit['label']] = edit['effective_evidence']
    require(patched == effective, 'UNRECORDED_EFFECTIVE_QUESTION_CHANGE')
    qmap = {q['candidate_id']: q for q in effective}
    lmap = {q['candidate_id']: q for q in lineages}
    roots = {row['root_id'] for row in rows}
    heldout = {r['root_id'] for r in b.read(protocol['split']['heldout_population_binding']['path'])['roots']}
    splitmap = {r['root_id']: r['split'] for r in b.readl(protocol['split']['authority']['path'])}
    require(len(roots) == 68 and roots.isdisjoint(heldout) and all(splitmap.get(r) == 'train' for r in roots), 'SPLIT_LEAKAGE')
    require(len({l['prompt_sha256'] for l in lineages}) == 363, 'DUPLICATE_PROMPTS')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(b.BASE), local_files_only=True)
    require(b.hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == protocol['length']['chat_template_sha256'], 'CHAT_TEMPLATE_CHANGED')
    native = b.module(b.TRAINER, 'v125_preflight_native_trainer')
    display = b.module(b.DISPLAY, 'v125_preflight_display')
    for row in rows:
        q, lin = qmap[row['candidate_id']], lmap[row['candidate_id']]
        require(b.digest(row['messages']) == lin['messages_sha256'], 'MESSAGES_BINDING_MISMATCH')
        source = b.read(HERE / 'sources' / (row['root_id'] + '.json'))
        require(b.sha(HERE / 'sources' / (row['root_id'] + '.json')) == q['source_sha256'], 'PACKAGED_SOURCE_CHANGED')
        variant = display.make_variants(q, seed=b.SEED, count=1)[0]
        require(variant == lin['display_variant'], 'DISPLAY_NOT_REPRODUCIBLE')
        # Check the whole dialogue/public-only prompt by exact reconstruction.
        expected_user = display.prompt(q, variant, source)
        require(row['messages'][:2] == [{'role': 'system', 'content': b.SYSTEM}, {'role': 'user', 'content': expected_user}], 'PROMPT_CONTENT_MISMATCH')
        mapping = lin['display_to_canonical']
        labels = list(q['options'] if q['question_type'] == 'multi_select' else q['items'])
        require(set(mapping) == set(mapping.values()) == set(labels), 'DISPLAY_MAP_NOT_BIJECTIVE')
        back = [mapping[label] for label in lin['displayed_gold']]
        require((set(back) == set(q['gold'])) if q['question_type'] == 'multi_select' else back == q['gold'], 'GOLD_REBINDING_INCORRECT')
        require(row['messages'][2] == {'role': 'assistant', 'content': '<box>' + ','.join(lin['displayed_gold']) + '</box>'}, 'ASSISTANT_TARGET_MISMATCH')
        prompt_ids = tokenizer(native.render_chat(tokenizer, row['messages'][:-1], True), add_special_tokens=False).input_ids
        full_ids = tokenizer(native.render_chat(tokenizer, row['messages'], False), add_special_tokens=False).input_ids
        if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
            full_ids.append(tokenizer.eos_token_id)
        require(full_ids[:len(prompt_ids)] == prompt_ids and len(full_ids) <= b.MAX_LENGTH, 'TRUNCATION_OR_PROMPT_PREFIX_MISMATCH')
        feature = native.tokenize_record(tokenizer, row, b.MAX_LENGTH)
        require(feature['input_ids'] == full_ids, 'NATIVE_TRAINER_TRUNCATED')
        require(feature['labels'] == [-100] * len(prompt_ids) + full_ids[len(prompt_ids):], 'ASSISTANT_MASK_MISMATCH')
        require(len(full_ids) == lin['input_token_count'], 'TOKEN_COUNT_MISMATCH')
    require(b.sha(plan['data_binding']['path']) == plan['data_binding']['sha256'], 'TRAINING_PLAN_DATA_CHANGED')
    require(plan['argv'][plan['argv'].index('--train-file') + 1] == str(HERE / 'data/TRAIN_NEW_ONLY.jsonl'), 'TRAINING_COMMAND_USES_OTHER_DATA')
    require(plan['argv'][plan['argv'].index('--max-length') + 1] == str(b.MAX_LENGTH), 'TRAINING_COMMAND_LENGTH_CHANGED')
    require(plan['argv'][plan['argv'].index('--init-adapter-dir') + 1] == str(b.ADAPTER), 'TRAINING_COMMAND_CHECKPOINT_CHANGED')
    print('PASS: 363 new questions / 68 train roots / 0 replay / 0 truncated / heldouts excluded', flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='After verification, start the frozen training command and write to its separate output directory.')
    args = parser.parse_args()
    plan = check()
    if not args.execute:
        print('Preflight only. Training has not started.')
        return
    out = Path(plan['run_output_directory'])
    require(not out.exists(), 'OUTPUT_ALREADY_EXISTS: use a new reviewed run name rather than overwriting')
    out.mkdir(parents=True)
    (out / 'LAUNCH_BINDING.json').write_text(json.dumps({'package_manifest_sha256': b.sha(HERE / 'COMPLETE_MANIFEST.json'), 'training_plan': plan}, indent=2) + '\n')
    os.environ.update(plan['environment'])
    os.chdir(plan['cwd'])
    os.execv(sys.executable, [sys.executable, *plan['argv'][1:]])


if __name__ == '__main__':
    main()
