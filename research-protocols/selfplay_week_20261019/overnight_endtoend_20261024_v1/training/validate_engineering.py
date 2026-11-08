"""CPU-only checks on historical training questions; does not admit a new pool."""
from __future__ import annotations
from common import *
from build_schedule import variants_for, messages_for


def check_cpu_resume():
    """Exercise the real checkpoint functions on a tiny numeric LoRA module."""
    import tempfile
    import torch
    from runner import save_checkpoint, load_lora, set_rng
    require(not torch.cuda.is_initialized(), 'CPU_VALIDATION_MUST_NOT_INITIALIZE_CUDA')
    original_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        def make():
            torch.manual_seed(123)
            base = torch.nn.Linear(4, 3)
            for p in base.parameters(): p.requires_grad_(False)
            model = native().LoRALinear(base, rank=2, alpha=4, dropout=.25)
            optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            return model, optim
        x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20
        target = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 15
        def step(model, optim):
            optim.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean(); loss.backward(); optim.step()
            return float(loss.detach())
        with tempfile.TemporaryDirectory(prefix='oet1_cpu_resume_') as directory:
            path = Path(directory) / 'resume.pt'
            # Prefix the wrapper to use the same nested naming convention as Qwen.
            nested, nested_optim = make()
            nested = torch.nn.Sequential(nested)
            nested_optim = torch.optim.AdamW([p for p in nested.parameters() if p.requires_grad], lr=.01)
            step(nested, nested_optim); step(nested, nested_optim)
            save_checkpoint(path, nested, nested_optim, {'step': 2, 'cursor': 8}, torch)
            expected_losses = [step(nested, nested_optim), step(nested, nested_optim)]
            expected_nested = {k: v.detach().clone() for k, v in nested.state_dict().items()}
            fresh, _ = make(); fresh = torch.nn.Sequential(fresh)
            fresh_optim = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad], lr=.01)
            state = torch.load(path, map_location='cpu', weights_only=False)
            require(state['step'] == 2 and state['cursor'] == 8, 'CPU_RESUME_CURSOR_LOST')
            load_lora(fresh, state['adapter']); fresh_optim.load_state_dict(state['optimizer']); set_rng(torch, state['rng'])
            actual_losses = [step(fresh, fresh_optim), step(fresh, fresh_optim)]
            require(actual_losses == expected_losses and all(torch.equal(v, fresh.state_dict()[k]) for k, v in expected_nested.items()), 'CPU_EXACT_RESUME_DIVERGED')
        require(not torch.cuda.is_initialized(), 'CPU_VALIDATION_INITIALIZED_CUDA')
    finally:
        torch.cuda.is_available = original_available
    return {'numeric_lora_resume_exact': True, 'optimizer_and_dropout_rng_restored': True, 'cuda_initialized': False}


def main():
    from transformers import AutoTokenizer
    import itertools
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    private_path = WEEK / 'postweek_question_firstpass_v2/inputs/PRIVATE_SCORING.json'
    private = read(private_path)['questions']
    totals = collections.defaultdict(lambda: collections.Counter())
    measured = []
    forbidden_by_slot = collections.defaultdict(list)
    for record in private:
        if record['question_type'] != 'ordering': continue
        q = read(record['question_binding']['path'])
        slot = 'PILOT_QFP2_HISTORICALLY_OBSERVED:' + q['root_id'] + ':' + q['slot_id']
        forbidden_by_slot[slot].extend([q['gold'], list(reversed(q['gold']))])
    transforms = {}
    for row in private:
        if row['slot_id'] == 'MC6': continue
        q = read(row['question_binding']['path']); source = read(q['source_binding']['path'])
        arm = q['condition']; slot_key = 'PILOT_QFP2_HISTORICALLY_OBSERVED:' + q['root_id'] + ':' + q['slot_id']
        variants = variants_for(q, q['gold'], slot_key, 20261024, forbidden_by_slot[slot_key])
        transforms[arm, slot_key] = [v['display_to_canonical'] for v in variants]
        for variant in variants:
            messages = messages_for(q, variant, source)
            field = 'options' if q['question_type'] == 'multi_select' else 'items'
            contract, _ = adapter().native()
            require(contract.build_prompt(tokenizer, {'question_type': q['question_type'], 'question': q['question']}, {field: variant[field]}, contract.dialogue_rows(source), context_mode='full_context') == native().render_chat(tokenizer, messages, True), 'PROMPT_MISMATCH')
            messages.append({'role': 'assistant', 'content': '<box>' + ','.join(variant['gold_display_labels']) + '</box>'})
            feature, check = tokenize(tokenizer, messages, 16384)
            totals[arm].update(full=check['expected_full_input_tokens'] * 2, supervised=check['expected_supervised_tokens'] * 2, variants=1)
            measured.append(check['expected_full_input_tokens'])
            req = {'request_id': 'engineering', 'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                   'variant_id': variant['variant_id'], 'question_type': q['question_type'], 'cohort_id': 'engineering_train_only',
                   'display_to_content': variant['display_to_content']}
            scored = score_one(messages[-1]['content'], req, {'gold_content_ids': variant['gold_content_ids'], 'reporting_role': 'TRAIN_FIT_PRIMARY'})
            require(scored['exact'] == 1 and (scored.get('pairwise', 1) == 1) and scored.get('partial', 1) == 1, 'REMAPPED_GOLD_NOT_EXACT')
    for arm, slot in transforms:
        require(transforms['GENERIC', slot] == transforms['TARGETED', slot], 'PAIRED_TRANSFORMS_DIFFER')
    # A paired four-item ordering with different canonical Gold still shares transforms.
    q1 = {'question_type': 'ordering', 'items': {f'I{i}': f'event {i}' for i in range(1, 5)}}
    g1, g2 = ['I1','I2','I3','I4'], ['I2','I4','I1','I3']
    forbidden = [g1, g1[::-1], g2, g2[::-1]]
    v1 = variants_for(q1, g1, 'paired-engineering-check', 20261024, forbidden)
    v2 = variants_for(q1, g2, 'paired-engineering-check', 20261024, forbidden)
    require([v['display_to_canonical'] for v in v1] == [v['display_to_canonical'] for v in v2], 'DIFFERENT_GOLD_BREAKS_SHARED_TRANSFORM')
    # Strict parsing must not silently rescue broken/outside/duplicate answers.
    for text in ('<box>A,C', '<box>A,A</box>', 'reason <box>A,C</box>', '<box>A,C</box> extra', '<box>A,C</box><box>A,C</box>'):
        require(strict_parse(text, 'multi_select', dict.fromkeys('ABCDEF')) is None, 'STRICT_PARSER_RESCUED_INVALID')
    require(strict_parse('<box>I1,I2</box>', 'ordering', dict.fromkeys(['I1', 'I2', 'I3', 'I4'])) is None, 'PARTIAL_ORDER_ACCEPTED')
    # Exhaust every four-item ordering, including pairwise partial score and exact.
    mapping = {f'I{i + 1}': f'content_{i}' for i in range(4)}
    req = {'request_id': 'check', 'candidate_id': 'check', 'root_id': 'check', 'variant_id': 'check',
           'question_type': 'ordering', 'cohort_id': 'engineering_train_only', 'display_to_content': mapping}
    priv = {'gold_content_ids': list(mapping.values()), 'reporting_role': 'TRAIN_FIT_PRIMARY'}
    for order in itertools.permutations(mapping):
        s = score_one('<box>' + ','.join(order) + '</box>', req, priv)
        independent = sum(order.index(a) < order.index(b) for i, a in enumerate(mapping) for b in list(mapping)[i + 1:]) / 6
        require(s['pairwise'] == independent and s['exact'] == float(list(order) == list(mapping)), 'ORDER_PAIRWISE_INCORRECT')
    require(totals['GENERIC']['supervised'] == totals['TARGETED']['supervised'], 'HISTORICAL_SUPERVISED_TOKEN_BUDGET_DIFFERS')
    evaluation = read(HERE / 'evaluation/EVALUATION_FREEZE.json')
    eval_gold = {r['candidate_id']: r for r in lines(evaluation['private_gold']['path'])}
    eval_scored = []
    for request in lines(evaluation['final_requests']['path']):
        private_record = eval_gold[request['candidate_id']]
        inverse = {content: label for label, content in request['display_to_content'].items()}
        display_gold = [inverse[x] for x in private_record['gold_content_ids']]
        if request['question_type'] == 'multi_select': display_gold.sort()
        scored = score_one('<box>' + ','.join(display_gold) + '</box>', request, private_record)
        require(scored['exact'] == 1., 'DEV_PRIVATE_MAP_ROUNDTRIP_FAILED')
        eval_scored.append(scored)
    eval_counts = {k: v['requests'] for k, v in aggregate(eval_scored)['by_task'].items()}
    resume_check = check_cpu_resume()
    out = {'status': 'PASS', 'mode': 'CPU_ENGINEERING_ONLY_NOT_NEW_POOL_ADMISSION', 'checked_at_utc': now(),
           'historical_main_questions': len(measured) // 2, 'variants_tokenized': len(measured),
           'max_full_input_tokens': max(measured), 'min_full_input_tokens': min(measured),
           'two_epoch_arm_token_budgets': {k: dict(v) for k, v in totals.items()},
           'strict_parser_invalid_cases_rejected': 6, 'four_item_orderings_exhausted': 24,
           'dev_map_cpu_gold_roundtrips': len(eval_scored), 'dev_stratum_request_counts': eval_counts,
           'cpu_checkpoint_resume': resume_check,
           'paired_transform_equality': True, 'different_gold_joint_exclusion_checked': True,
           'native_prompt_byte_equivalence': True, 'gold_remap_exact_all': True, 'source_truncation': False,
           'model_loaded': False, 'gpu_started': False,
           'input_binding': binding(private_path), 'code_bindings': [binding(HERE / x) for x in ('common.py', 'build_schedule.py', 'runner.py', 'validate_engineering.py')]}
    save(HERE / 'ENGINEERING_VALIDATION_V4.json', out)
    print(json.dumps({k: out[k] for k in ('status', 'historical_main_questions', 'variants_tokenized', 'max_full_input_tokens', 'two_epoch_arm_token_budgets')}))

if __name__ == '__main__': main()
