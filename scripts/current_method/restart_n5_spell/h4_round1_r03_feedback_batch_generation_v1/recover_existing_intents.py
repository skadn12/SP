"""Reuse saved miner output after per-slot compatibility correction; no re-mining."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from . import API_CONFIG, MODEL, ROOTS, RUNS, TEMPERATURE
from .contract import validate_intents
from .run import (prepare, generate_root, read, save, sha, verify_frozen,
                  OpenAICompatibleChallengerClient, load_challenger_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=RUNS / 'h4_round1_r03_feedback_batch_generation_revision1_live_2')
    parser.add_argument('--output', type=Path, default=RUNS / 'h4_round1_r03_feedback_batch_generation_slot_recovery_live_1')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    recover = []
    for root in ROOTS:
        rd = args.source / root
        report = read(rd / 'generation_report.json')
        if report['scheduled_intent_count'] or report['candidate_count']:
            continue
        checked = validate_intents(read(rd / 'intents.json')['raw'],
                                   read(rd / 'dialogue_source.json'), read(rd / 'feedback_packet.json'))
        if any(x['generation_allowed'] for x in checked['slots']):
            recover.append(root)
    manifest = prepare(args.output, recover)
    manifest['intent_recovery_sources'] = {
        root: {'path': str((args.source / root / 'intents.json').resolve()),
               'sha256': sha(args.source / root / 'intents.json')} for root in recover}
    manifest['new_intent_miner_calls'] = 0
    manifest['recovery_reason'] = 'independently_valid_slots_retained_after_sibling_schema_failure'
    for root in recover:
        for name in ('dialogue_source.json', 'feedback_packet.json'):
            if sha(args.source / root / name) != sha(args.output / root / name):
                raise ValueError('recovery_input_drift')
    save(args.output / 'manifest.json', manifest)
    verify_frozen(args.output, manifest)
    if not args.live:
        print(json.dumps({'roots': recover, 'new_miner_calls': 0}))
        return 0
    config = load_challenger_config(API_CONFIG)
    if config.model != MODEL:
        raise ValueError('model_drift')
    client = OpenAICompatibleChallengerClient(replace(config, temperature=TEMPERATURE,
        enable_thinking=False, timeout_seconds=max(180, config.timeout_seconds)))
    reports = []
    for root in recover:
        source = manifest['intent_recovery_sources'][root]
        if sha(source['path']) != source['sha256']:
            raise ValueError('reused_intent_source_drift')
        reports.append(generate_root(args.output, root, manifest, client, reused_intents=Path(source['path'])))
        save(args.output / 'batch_progress.json', {'reports': reports})
    verify_frozen(args.output, manifest)
    save(args.output / 'recovery_report.json', {'status': 'RECOVERY_GENERATED_PENDING_SEMANTICS',
         'roots': reports, 'new_miner_calls': 0,
         'candidate_count': sum(r['candidate_count'] for r in reports),
         'structurally_accepted_count': sum(r['structurally_accepted_count'] for r in reports)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
