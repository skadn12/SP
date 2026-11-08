"""Project a frozen name-only writer style without adding new source facts."""
from pathlib import Path
import re
from common import bind, need, read

TOKENS = ('he','him','his','himself','she','her','hers','herself')
PATTERN = re.compile(r'\b(?:'+'|'.join(TOKENS)+r')\b', re.I)


def policy_for(config, slot, source):
    path = Path(config['actor_reference_policy']['path'])
    registry_path = Path(config['character_registry']['path'])
    need(bind(path) == config['actor_reference_policy'] and bind(registry_path) == config['character_registry'],
         'actor_reference_policy_or_registry_changed')
    policy = read(path); registry = read(registry_path)
    need(policy['registry'] == config['character_registry'] and policy['forbidden_gendered_pronouns'] == list(TOKENS)
         and policy['output_style'] == 'use_source_names_or_explicit_roles_for_people'
         and policy['new_source_facts'] is False and policy['automatic_repair'] is False, 'actor_reference_contract_changed')
    root = slot['root_id']; names = sorted({t['speaker'] for t in source['turns']})
    characters = [c for c in registry['characters'] if c['root_id'] == root]
    need(sorted(c['canonical_name'] for c in characters) == names, 'actor_registry_source_names_mismatch')
    need(all(c['gender']['value'] is None and c['third_person_pronouns']['value'] is None for c in characters),
         'pilot_unknown_pronoun_scope_changed')
    source_binding = next(r for r in registry['source_bindings'] if r['root_id'] == root)
    need(source_binding['sha256'] == slot['source_sha256'], 'actor_registry_wrong_source_version')
    return {'version': policy['version'], 'root_id': root, 'canonical_source_speaker_names': names,
            'pronoun_status': 'unknown_for_all_source_speakers', 'output_style': policy['output_style'],
            'forbidden_gendered_pronouns': list(TOKENS),
            'scope': 'generated item propositions only; use source names or explicit roles for any person',
            'source_evidence_unchanged': True, 'new_source_facts': False, 'automatic_repair': False}


def name_policy_errors(items):
    return ['ordering_actor_reference_policy_forbidden_pronoun:'+label+':'+token
            for label, text in items.items() for token in sorted({m.group().lower() for m in PATTERN.finditer(text)})]
