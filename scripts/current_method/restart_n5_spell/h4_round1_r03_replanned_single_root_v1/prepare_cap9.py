"""Offline-only target-eight/cap-nine preparation from an independent rebound config."""
import argparse
import json
from pathlib import Path
from shutil import copytree

from ..h4_round1_r03_incremental_plan_retention_v1 import planner_candidate_policy as policy
from ..h4_round1_r03_incremental_plan_retention_v1 import planner_preparation as prep
from ..h4_round1_r03_incremental_plan_retention_v1 import planner_relations, planner_templates
from ..h4_round1_r03_incremental_plan_retention_v1.run import offline_network_guard


def prepare(source_config, output):
    source_config, output = Path(source_config).resolve(), Path(output).resolve()
    with offline_network_guard():
        config = json.loads(source_config.read_text())
        # Validate every declared source before copying any of its bytes.
        for identity in config['input_files'].values():
            prep.read_verified(identity)
        source_bundle = prep.prepare_initial(config)['bundle']
        source_inputs = Path(config['input_files']['run_freeze']['path']).parent
        output.mkdir(parents=True, exist_ok=False)
        inputs = output / 'input_bundle'
        copytree(source_inputs, inputs)
        history_file = inputs / Path(source_bundle['input_files']['history']['path']).name
        if prep.file_identity(history_file)['sha256'] != source_bundle['input_files']['history']['sha256']:
            raise ValueError('cap9_copied_history_sha_mismatch')
        history_manifest = inputs / Path(config['input_files']['history_manifest']['path']).name
        rows = [json.loads(line) for line in history_manifest.read_text().splitlines() if line.strip()]
        if len(rows) != 1 or rows[0]['root_id'] != config['root_id']:
            raise ValueError('cap9_requires_single_root_rebound_manifest')
        rows[0]['input_dialogue_path'] = str(history_file)
        history_manifest.write_text(json.dumps(rows[0], ensure_ascii=False) + '\n')
        freeze_file = inputs / 'run_freeze_manifest.json'
        freeze = json.loads(freeze_file.read_text())
        freeze['input_path'] = str(history_file)
        freeze_file.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + '\n')
        budget_file = inputs / Path(config['input_files']['capacity_budget']['path']).name
        budget_file.write_text(json.dumps(policy.gate_budget(source_bundle['capacity_budget'], max_new_candidates=9), indent=2) + '\n')
        for key, identity in config['input_files'].items():
            if key != 'export_manifest':
                config['input_files'][key] = prep.file_identity(inputs / Path(identity['path']).relative_to(source_inputs))
        config.update(candidate_policy_version=policy.VERSION, template_version=planner_templates.CAP9_VERSION,
                      output_contract_version=planner_relations.CAP9_VERSION, append_enabled=False,
                      input_bundle_version=policy.VERSION)
        prep.write_json(output / 'config.json', config)
        result = prep.prepare_from_config(config)
        bundle = result['initial']['bundle']
        for key in ('parent', 'dialogue', 'candidates', 'verified_tracks', 'memory_overview', 'shared_feedback_view'):
            if bundle[key] != source_bundle[key]:
                raise ValueError('cap9_changed_authoritative_input:' + key)
        prep.write_preparation(output / 'prepared', result)
        prep.write_json(output / 'lineage.json', {
            'source_config': prep.file_identity(source_config), 'candidate_policy_version': policy.VERSION,
            'target_new_candidates': 8, 'hard_max_new_candidates': 9, 'max_continuations': 3,
            'gate_request_limit': 12, 'scene_quota': 6, 'external_calls': 0,
            'raw_responses_used_for_prompt': False, 'old_r03_plan_or_dialogue_consumed': False,
            'authoritative_inputs_unchanged': True,
            'feedback_export': config['input_files']['export_manifest'],
            'bundle_sha256': bundle['bundle_sha256'], 'request_sha256': bundle['request_sha256'],
        })
        return {'root_id': config['root_id'], 'output': str(output), 'status': result['status'], 'external_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_config, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
