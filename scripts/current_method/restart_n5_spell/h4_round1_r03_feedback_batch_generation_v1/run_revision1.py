"""Execute the compact intent-prompt revision under the same policy/schema."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import API_CONFIG, MODEL, ROOTS, RUNS, TEMPERATURE
from .intent_prompt_revision1 import render
from .run import (digest, generate_root, prepare, read, save, sha,
                  verify_frozen, OpenAICompatibleChallengerClient, load_challenger_config)
from ..core import atomic_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUNS / 'h4_round1_r03_feedback_batch_generation_revision1_live_1')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-intent', action='append', default=[], metavar='ROOT=PATH')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.resume:
        manifest = read(out / 'manifest.json')
    else:
        manifest = prepare(out, list(ROOTS))
        for root in ROOTS:
            rd = out / root
            prompt = render(read(rd / 'dialogue_source.json'), read(rd / 'feedback_packet.json'))
            atomic_text(rd / 'intent_prompt.txt', prompt)
            manifest['inputs'][root]['intent_prompt_sha256'] = digest(prompt)
        manifest['prompt_presentation_revision'] = 'compact_intent_fields_and_terminal_r03_allowlist_v1'
        manifest['previous_attempt'] = str(RUNS / 'h4_round1_r03_feedback_batch_generation_v1_live_3')
        manifest['policy_or_schema_changes'] = False
        reuse = {}
        for value in args.reuse_intent:
            root, source = value.split('=', 1)
            if root not in ROOTS:
                raise ValueError('invalid_reused_root')
            source = str(Path(source).resolve())
            reuse[root] = {'path': source, 'sha256': sha(source)}
        manifest['reused_intents'] = reuse
        manifest['reference_compatibility'] = 'independent_existing_r03_refs_and_neutral_empty_metadata_defaults'
        save(out / 'manifest.json', manifest)
    verify_frozen(out, manifest)
    if not args.live:
        print(json.dumps({'status': 'PREPARED_COMPACT_PROMPT', 'output': str(out)}))
        return 0
    config = load_challenger_config(API_CONFIG)
    if config.model != MODEL:
        raise ValueError('model_drift')
    client = OpenAICompatibleChallengerClient(replace(config, temperature=TEMPERATURE,
        enable_thinking=False, timeout_seconds=max(180, config.timeout_seconds)))
    reports = []
    for root in ROOTS:
        path = out / root / 'generation_report.json'
        reuse = manifest.get('reused_intents', {}).get(root)
        if reuse and sha(reuse['path']) != reuse['sha256']:
            raise ValueError('reused_intent_hash_drift')
        report = read(path) if path.exists() else generate_root(out, root, manifest, client,
            reused_intents=Path(reuse['path']) if reuse else None)
        reports.append(report)
        save(out / 'batch_progress.json', {'completed_roots': len(reports), 'total_roots': len(ROOTS), 'reports': reports})
        # One root's data/intent failure does not resubmit it or discard other roots.
    verify_frozen(out, manifest)
    save(out / 'final_batch_report.json', {
        'status': 'BATCH_COMPLETE_PENDING_SEMANTIC_REVIEW', 'roots': reports,
        'candidate_count': sum(r['candidate_count'] for r in reports),
        'structurally_accepted_count': sum(r['structurally_accepted_count'] for r in reports),
        'application_request_count': sum(r['ledger']['request_count'] for r in reports),
        'confirmed_completed_requests': sum(r['ledger']['completed_request_count'] for r in reports),
        'semantic_auditor_calls': 0, 'solver_calls': 0, 'training_calls': 0})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
