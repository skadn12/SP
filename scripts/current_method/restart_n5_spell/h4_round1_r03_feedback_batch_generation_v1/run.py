"""Prepare frozen inputs and execute the feedback-policy batch generation."""

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

from . import (API_CONFIG, GENERATOR_TOKENS, HERE, INTENT_TOKENS, MAX_CALLS_PER_ROOT,
              MODEL, ROOTS, ROUND1, RUNS, TEMPERATURE, VERSION)
from .contract import reasoning_contract, validate_intents
from .inputs import compile_inputs, read, sha
from .prompts import generator_prompt, intent_prompt
from ..core import atomic_json, atomic_text, canonical_json
from ..h4_qa_mining_hybrid_generation_v1.candidate_contract import normalize_batch
from ..h4_qa_multiroot_k3_t08_probe_v1.permutation import audit_gold_rebinding, position_variants
from ..h4_qa_multiroot_k3_t08_probe_v1.validator import validate_batch
from ..ledger import RequestLedger
from ..live_backends import OpenAICompatibleChallengerClient, load_challenger_config
from ..simple_append_v1.live_runtime import load_json_object_strict


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save(path, value):
    atomic_json(path, value)
    atomic_text(path.with_name(path.name + '.sha256'), sha(path) + '  ' + path.name + '\n')


def configuration():
    # Read the historical contract and ledger, rather than guessing from defaults.
    reference = ROUND1 / 'cc-train-episode-197635'
    plan = read(reference / 'qa_feedback_generation_contract.json')
    states = [read(p) for p in (reference / 'ledger/requests').glob('*.json')]
    models = {x['backend_identity']['model'] for x in states}
    if models != {MODEL} or plan['temperature'] != TEMPERATURE:
        raise ValueError('round1_configuration_drift')
    return {
        'version': VERSION, 'round1_reference': str(reference), 'model': MODEL,
        'temperature': plan['temperature'], 'enable_thinking': False,
        'intent_max_tokens': plan['intent_miner_max_tokens'], 'qa_max_tokens': plan['max_tokens'],
        'sources': ['contract_guided', 'free_exploration'],
        'target_counts': {'contract_guided': {'single_choice': 2, 'multi_select': 1, 'ordering': 1},
                          'free_exploration': {'single_choice': 4, 'multi_select': 2, 'ordering': 2}},
        'option_keys': {'single_choice': 'A-D', 'multi_select': 'A-E', 'ordering': 'I1-I4'},
        'application_resend_on_timeout_or_parse_failure': False,
        'transport_policy': 'same_existing_OpenAICompatibleChallengerClient',
        'max_calls_per_root': MAX_CALLS_PER_ROOT,
        'intent_slots': ['SUCCESSOR_FRONTIER', 'COVERAGE_GAP', 'FRESH_R03'],
        'intent_count_is_safe_intent_quota': False,
        'intent_and_candidate_semantic_status': 'PENDING_REVIEW',
        'intentional_changes_from_round1': [
            'human_frozen_policy_and_slot_strategy_schema',
            'R03_baseline_cumulative_transcript_and_root_scope',
            'canonical_turn_refs_instead_of_numeric_intent_positions',
            'NO_SAFE_allowed_per_slot_and_fewer_candidates_allowed',
            'individually_valid_rows_retained_when_batch_under_target',
            'at_most_one_missing_type_repair_per_root_type_after_primary_filtering',
            'reference_coverage_and_readable_historical_replay_metadata',
            'explicit_feedback_provenance_and_semantic_review_pending'],
    }


def prepare(output, roots):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'output_already_exists:{output}')
    output.mkdir(parents=True, exist_ok=True)
    config = configuration()
    if (config['intent_max_tokens'], config['qa_max_tokens']) != (INTENT_TOKENS, GENERATOR_TOKENS):
        raise ValueError('token_configuration_mismatch')
    code_hashes = {str(p): sha(p) for p in HERE.iterdir() if p.suffix in {'.py', '.json', '.md'}}
    manifest = {'version': VERSION, 'roots': roots, 'configuration': config,
                'implementation_hashes': code_hashes, 'inputs': {}}
    for filename in ('qa_feedback_policy_v2.md', 'intent_contract_v1.json'):
        atomic_text(output / filename, (HERE / filename).read_text())
    for root in roots:
        dialogue, packet = compile_inputs(root)
        rd = output / root
        save(rd / 'dialogue_source.json', dialogue)
        save(rd / 'feedback_packet.json', packet)
        prompt = intent_prompt(dialogue, packet)
        atomic_text(rd / 'intent_prompt.txt', prompt)
        manifest['inputs'][root] = {
            'dialogue_sha256': sha(rd / 'dialogue_source.json'),
            'packet_sha256': sha(rd / 'feedback_packet.json'),
            'intent_prompt_sha256': digest(prompt),
            'source_manifest': packet['source_manifest'],
            'historical_question_count': len(packet['history']),
            'feedback_count': len(packet['eligible_feedback']),
            'coverage_gap_count': len(packet['coverage_gaps']),
        }
    save(output / 'manifest.json', manifest)
    save(output / 'configuration_comparison.json', config)
    return manifest


def verify_frozen(output, manifest):
    for path, expected in manifest['implementation_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'implementation_changed_since_freeze:{path}')
    for root, entry in manifest['inputs'].items():
        for filename, field in [('dialogue_source.json', 'dialogue_sha256'),
                                ('feedback_packet.json', 'packet_sha256'),
                                ('intent_prompt.txt', 'intent_prompt_sha256')]:
            if sha(output / root / filename) != entry[field]:
                raise ValueError(f'frozen_input_changed:{root}:{filename}')


def call_json(rd, ledger, client, request_id, prompt, tokens, role):
    atomic_text(rd / 'raw/prompts' / (request_id + '.txt'), prompt)
    try:
        response = ledger.invoke(
            request_id=request_id, intent=role, attempt=1, prompt_sha256=digest(prompt),
            call=lambda: client.complete(prompt=prompt, max_tokens=tokens, json_object=True),
            budget_class=VERSION, budget_limit=MAX_CALLS_PER_ROOT, total_budget_limit=MAX_CALLS_PER_ROOT,
            request_metadata={'version': VERSION, 'temperature': TEMPERATURE,
                              'enable_thinking': False, 'max_tokens': tokens},
            backend_identity={'model': client.config.model, 'base_url': client.config.base_url,
                              'temperature': TEMPERATURE})
    except Exception as exc:
        save(rd / 'failures' / (request_id + '.json'),
             {'stage': role, 'error_type': type(exc).__name__, 'request_id': request_id,
              'retry': False, 'delivery_status': 'UNCONFIRMED_SEE_LEDGER'})
        return None, 'TRANSPORT_OR_LEDGER_ERROR'
    try:
        if response.metadata.get('finish_reason') == 'length':
            raise ValueError('truncated_generation')
        return load_json_object_strict(response.raw_text), None
    except (ValueError, TypeError):
        save(rd / 'failures' / (request_id + '.json'),
             {'stage': role, 'request_id': request_id, 'error_type': 'INVALID_OR_TRUNCATED_JSON', 'retry': False})
        return None, 'PARSE_ERROR'


def norm(text):
    return re.sub(r'[^\w]+', ' ', str(text).casefold()).strip()


def content_key(row):
    options = row.get('items') if row.get('question_type') == 'ordering' else row.get('options')
    return (norm(row.get('question')), tuple(sorted(norm(v) for v in (options or {}).values())))


def filter_candidates(candidates, dialogue, history):
    validity = validate_batch(candidates, valid_turn_refs={t['turn_ref'] for t in dialogue['turns']})
    seen_stems = {norm(h.get('question')) for h in history}
    seen_content = {content_key(h) for h in history}
    accepted, audits, permutations = [], [], []
    for row, audit in zip(candidates, validity['rows']):
        errors = list(audit['blocking_errors']) + list(row.get('schema_errors') or [])
        stem = norm(row.get('question'))
        if content_key(row) in seen_content:
            errors.append('duplicate_content_in_root_bank')
        refs = set(row.get('evidence_refs') or [])
        anchors = set(row['intent_anchors'])
        if not refs.intersection(anchors):
            errors.append('intent_anchor_not_cited')
        if row['slot_type'] in {'SUCCESSOR_FRONTIER', 'FRESH_R03'}:
            if not refs.intersection(row['required_r03_evidence']):
                errors.append('required_r03_anchor_not_cited')
        similarities = [SequenceMatcher(None, stem, s).ratio() for s in seen_stems]
        audit['lexical_similarity_review'] = bool(similarities and max(similarities) >= .9)
        if not errors:
            try:
                variants = position_variants(row)
                pa = audit_gold_rebinding(row, variants)
                if not pa['pass']:
                    errors.append('permutation_rebinding_failed')
            except (KeyError, ValueError, TypeError):
                errors.append('permutation_generation_failed')
        if not errors:
            accepted.append(row)
            seen_stems.add(stem)
            seen_content.add(content_key(row))
            permutations.append({'candidate_id': row['candidate_id'], 'variants': variants,
                                 'gold_rebinding_audit': pa})
        audits.append({**audit, 'decision': 'reject' if errors else 'structurally_accepted_pending_semantics',
                       'filter_reasons': sorted(set(errors)), 'novelty_status': 'PENDING_SEMANTIC_REVIEW',
                       'semantic_evidence_support': 'NOT_VALIDATED',
                       'r03_necessity': 'UNKNOWN'})
    return accepted, audits, permutations


def generate_root(output, root, manifest, client, backend_mode='live', reused_intents=None,
                  balanced_type_minimums=None, type_focus_only=False):
    if type_focus_only and not balanced_type_minimums:
        raise ValueError('type_focus_only_requires_explicit_type_targets')
    rd = output / root
    ledger = RequestLedger(rd, digest(canonical_json(manifest)), backend_mode)
    dialogue, packet = read(rd / 'dialogue_source.json'), read(rd / 'feedback_packet.json')
    if reused_intents is None:
        raw, error = call_json(rd, ledger, client, root + '-intent-miner',
                              (rd / 'intent_prompt.txt').read_text(), INTENT_TOKENS, 'intent_miner')
    else:
        raw, error = read(reused_intents)['raw'], None
        save(rd / 'reused_intent_source.json', {'path': str(reused_intents), 'sha256': sha(reused_intents),
                                              'new_intent_api_calls': 0})
    checked = validate_intents(raw, dialogue, packet)
    save(rd / 'intents.json', {'raw': raw, 'call_error': error, 'validation': checked})
    intents = [s['intent'] for s in checked['slots'] if s['generation_allowed']]
    candidates, records = [], []

    def batch(intent, mode, repair_type=None, repair_target_count=1):
        suffix = ('repair-' + repair_type) if repair_type else mode
        rid = f'{root}-{intent["intent_id"]}-{suffix}'
        prompt_args = (dialogue, intent, packet['history'], mode,
                       [r['question'] for r in candidates], repair_type)
        prompt = (generator_prompt(*prompt_args, repair_target_count=repair_target_count)
                  if type_focus_only else generator_prompt(*prompt_args))
        raw_batch, failure = call_json(rd, ledger, client, rid, prompt, GENERATOR_TOKENS, 'qa_type_repair' if repair_type else 'qa_generator')
        try:
            normalized = normalize_batch(raw_batch, intent_id=intent['intent_id'], source=mode,
                                         contract=reasoning_contract(intent), expanded=dialogue)
        except (ValueError, TypeError, AttributeError) as exc:
            normalized = {'candidates': [], 'errors': ['malformed_batch:' + type(exc).__name__]}
        for ordinal, row in enumerate(normalized['candidates'], 1):
            row.update({'candidate_id': f'{rid}:Q{ordinal:03}', 'root_id': root,
                        'slot_type': intent['intent_type'], 'feedback_intent_id': intent['intent_id'],
                        'strategy': intent['strategy'], 'generation_mode': mode, 'batch_id': rid,
                        'source_feedback_ids': intent['source_feedback_ids'],
                        'source_coverage_gap_ids': intent['coverage_gap_ids'],
                        'intent_anchors': sorted({r for g in intent['reasoning_chain'].values() for r in g}
                                                 | set(intent['required_r03_evidence'])),
                        'required_r03_evidence': intent['required_r03_evidence'],
                        'r03_evidence_used': any(r.startswith('R03:') for r in row['evidence_refs']),
                        'feedback_alignment': 'PENDING_SEMANTIC_REVIEW',
                        'dialogue_sha256': packet['dialogue_sha256'],
                        'repair_type': repair_type})
            if repair_type and row['question_type'] != repair_type:
                row['schema_errors'].append('repair_question_type_out_of_scope')
            candidates.append(row)
        records.append({'request_id': rid, 'intent_id': intent['intent_id'], 'mode': mode,
                        'repair_type': repair_type, 'call_error': failure,
                        'count_deviations': normalized['errors'], 'candidate_count': len(normalized['candidates'])})
        save(rd / 'candidate_generation.json', records)
        save(rd / 'candidate_pool.json', candidates)

    if not type_focus_only:
        for intent in intents:
            for mode in ('contract_guided', 'free_exploration'):
                batch(intent, mode)
    accepted, audits, permutations = filter_candidates(candidates, dialogue, packet['history'])
    present = {r['question_type'] for r in accepted}
    # Failure is not an invitation to regenerate an ambiguous paid completion.
    if intents and not any(r['call_error'] for r in records):
        repair_targets = tuple(q for q in ('single_choice', 'multi_select', 'ordering')
                               if q not in present)
        if balanced_type_minimums:
            counts = Counter(r['question_type'] for r in accepted)
            repair_targets = tuple(q for q in ('single_choice', 'multi_select', 'ordering')
                                   if counts[q] < int(balanced_type_minimums.get(q, 0)))
        for qtype in repair_targets:
            if intents:
                batch(intents[0], 'contract_guided', qtype,
                      int(balanced_type_minimums[qtype]) if type_focus_only else 1)
    accepted, audits, permutations = filter_candidates(candidates, dialogue, packet['history'])
    save(rd / 'candidate_pool.json', candidates)
    save(rd / 'candidate_audit.json', audits)
    save(rd / 'candidates_valid.json', accepted)
    save(rd / 'permutations.json', permutations)
    report = {'root_id': root, 'status': 'GENERATED_PENDING_SEMANTIC_REVIEW',
              'candidate_count': len(candidates), 'structurally_accepted_count': len(accepted),
              'scheduled_intent_count': len(intents),
              'intent_validation_ok': checked['valid'],
              'no_safe_slots': [s['intent']['intent_type'] for s in checked['slots'] if s['intent']['status'] == 'NO_SAFE_EVOLUTION'],
              'question_type_counts': dict(Counter(r['question_type'] for r in accepted)),
              'call_failures': int(bool(error)) + sum(bool(r['call_error']) for r in records),
              'ledger': ledger.summary(), 'semantic_auditor_calls': 0, 'solver_calls': 0,
              'training_calls': 0, 'semantic_admission_status': 'NOT_RUN'}
    if report['call_failures'] or not checked['valid']:
        report['status'] = 'GENERATION_WITH_GAPS'
    if balanced_type_minimums:
        report['type_targets'] = balanced_type_minimums
        report['type_focus_only'] = type_focus_only
        report['type_shortfalls'] = {q: max(0, int(n) - report['question_type_counts'].get(q, 0))
                                    for q, n in balanced_type_minimums.items()}
    save(rd / 'generation_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUNS / (VERSION + '_live_1'))
    parser.add_argument('--root-id', action='append')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    roots = args.root_id or list(ROOTS)
    if len(set(roots)) != len(roots) or set(roots) - set(ROOTS):
        raise ValueError('invalid_root_scope')
    manifest = read(output / 'manifest.json') if args.resume else prepare(output, roots)
    if manifest['roots'] != roots:
        raise ValueError('root_scope_drift')
    verify_frozen(output, manifest)
    if not args.live:
        print(json.dumps({'status': 'INPUTS_AND_POLICY_FROZEN', 'output': str(output), 'roots': roots}))
        return 0
    config = load_challenger_config(API_CONFIG)
    if config.model != MODEL:
        raise ValueError('backend_model_not_round1_model')
    client = OpenAICompatibleChallengerClient(replace(config, temperature=TEMPERATURE,
        enable_thinking=False, timeout_seconds=max(180, config.timeout_seconds)))
    reports = []
    for root in roots:
        report_path = output / root / 'generation_report.json'
        reports.append(read(report_path) if report_path.exists() else generate_root(output, root, manifest, client))
        save(output / 'batch_progress.json', {'completed_roots': len(reports), 'total_roots': len(roots), 'reports': reports})
    verify_frozen(output, manifest)
    save(output / 'final_batch_report.json', {
        'status': 'BATCH_COMPLETE_PENDING_SEMANTIC_REVIEW', 'roots': reports,
        'candidate_count': sum(r['candidate_count'] for r in reports),
        'structurally_accepted_count': sum(r['structurally_accepted_count'] for r in reports),
        'application_request_count': sum(r['ledger']['request_count'] for r in reports),
        'confirmed_completed_requests': sum(r['ledger']['completed_request_count'] for r in reports),
        'semantic_auditor_calls': 0, 'solver_calls': 0, 'training_calls': 0})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
