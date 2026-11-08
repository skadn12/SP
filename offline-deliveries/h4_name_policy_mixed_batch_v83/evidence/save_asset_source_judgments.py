"""Save manually determined source/public-only round judgments.

This helper does not classify propositions. The reviewer supplies each current
version's semantic judgment after reading the complete source and public object.
It reads no current private variant, writer response, Gold or solver artifact.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HOME = Path(__file__).resolve().parents[1]
SCHEMA_PATH = HOME.parent / 'h4_output_contract_multiselect_v79' / 'review_schema.py'
spec = importlib.util.spec_from_file_location('v83_asset_public_review_schema', SCHEMA_PATH)
schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schema)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def parent_rows(slot, public):
    """Reuse exact historical judgments, with current full-source reread disclosed."""
    legacy = read(slot['parent_review_path'])
    if 'public_question' in legacy:
        indexed = {r['option']: r for r in legacy['judgments']}
        def judgment(label):
            old = indexed[label]
            return {
                'proposition': old['proposition'], 'relation': 'supports',
                'scope_valid': True, 'fact_distinct': True,
                'reason': old['reason'], 'citations': old['required_evidence'],
            }
        stem_reason = ('The stem asks which actions or plans Ethan explicitly undertook, '
                       'proposed or confirmed in the dialogue. It permits reported actions '
                       'and plans while preserving their respective action states.')
    else:
        historical = next(row for row in legacy['questions']
                          if row['candidate_id'] == slot['parent_candidate_id'])
        def judgment(label):
            return deepcopy(historical['options'][label])
        stem_reason = historical['stem_reason']
    rows = []
    for fact in public['parent_facts']:
        row = judgment(fact['parent_label'])
        assert row['proposition'] == fact['proposition']
        row['fact_id'] = fact['fact_id']
        row['historical_parent_judgment_reused'] = True
        row['reason'] = ('Immutable historical parent judgment reused after current complete '
                         'source reread; its cited fact remains supported. ' + row['reason'])
        rows.append(row)
    return rows, stem_reason


def save_family(family, round_id, manual_rows):
    config = read(HOME / 'CONFIG.json')['multiselect_config']
    slot = next(s for s in config['slots'] if s['family_id'] == family)
    run = HOME / 'multiselect' / 'assets' / family
    folder = run / 'rounds' / round_id
    public_path = folder / 'PUBLIC_ASSETS.json'
    public = read(public_path)
    source = read(run / 'SOURCE.json')
    assert len(source['turns']) == {'root194259_family': 149, 'root130221_family': 91}[family]
    assert sha(run / 'SOURCE.json') == slot['source_sha256']
    assert {r['version_id'] for r in manual_rows} == {r['version_id'] for r in public['versions']}
    public_versions = {r['version_id']: r for r in public['versions']}
    for row in manual_rows:
        assert row['proposition'] == public_versions[row['version_id']]['proposition']
        assert row['relation'] in ('supports', 'contradicts', 'unknown')
    parents, stem_reason = parent_rows(slot, public)
    now = datetime.now(timezone.utc).isoformat()
    judgment = {
        'version': 'v83_independent_asset_round_source_judgments_v1',
        'review_schema_version': schema.SCHEMA_VERSION,
        'created_at_utc': now,
        'reviewer': '/root/v83_asset_source_review',
        'review_mode': 'current_full_source_and_public_versions_with_immutable_historical_parent_judgment_reuse',
        'root_id': source['root_id'], 'family_id': family, 'round_id': round_id,
        'source_sha256': sha(run / 'SOURCE.json'),
        'public_assets_sha256': sha(public_path),
        'parent_all_true_provenance_known': True,
        'parent_gold_known_before_review': True,
        'fresh_blind_parent_review': False,
        'current_solver_scores_read': False,
        'solver_scores_used': False,
        'new_variant_private_artifacts_read_before_save': False,
        'full_source_read': True,
        'complete_source_turns_read': len(source['turns']),
        'current_complete_source_reread': True,
        'source_first': True,
        'derived_from_rounds': False,
        'parent_judgment_reuse': {
            'mode': 'immutable_historical_parent_judgment_reuse_after_current_complete_source_reread',
            'review': binding(slot['parent_review_path']),
            'provenance': binding(slot['parent_provenance_path']),
            'fresh_blind_parent_review': False,
            'current_source_reread_turns': len(source['turns']),
            'note': ('Six true parent facts and parent Gold status were supplied before review. '
                     'Historical judgments are reused without translating their legacy disclosures. '
                     'Every current public replacement is independently judged against the current '
                     'complete exact source; no current private variants or solver scores were read.'),
        },
        'stem_valid': True, 'stem_reason': stem_reason,
        'parents': parents, 'versions': manual_rows,
        'full_source_consistency': ('Complete exact source was read. Reports, proposals and plans '
                                   'retain their modality and event scope. Later scoped exchanges '
                                   'are considered; additional activities can coexist and mere '
                                   'absence of mention does not establish contradiction.'),
    }
    write_new(folder / 'SOURCE_JUDGMENTS.json', judgment)
    schema.validate_round(run, round_id)
    return {
        'family_id': family, 'round_id': round_id,
        'source': binding(run / 'SOURCE.json'),
        'public_assets': binding(public_path),
        'source_judgments': binding(folder / 'SOURCE_JUDGMENTS.json'),
        'complete_source_turns_read': len(source['turns']),
        'parent_count': len(parents),
        'version_count': len(manual_rows),
        'supported_versions': sum(r['relation'] == 'supports' for r in manual_rows),
        'contradicted_versions': sum(r['relation'] == 'contradicts' for r in manual_rows),
        'unknown_versions': sum(r['relation'] == 'unknown' for r in manual_rows),
        'scope_invalid_versions': sum(not r['scope_valid'] for r in manual_rows),
        'schema_validated': True,
    }


def main():
    manual = read(sys.argv[1])
    round_id = manual['round_id']
    results = [save_family(family, round_id, rows)
               for family, rows in manual['families'].items()]
    status = {
        'version': 'v83_asset_source_review_status_v1',
        'reviewer': '/root/v83_asset_source_review',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'round_id': round_id,
        'status': 'SOURCE_JUDGMENTS_SAVED_BEFORE_CURRENT_PRIVATE_CROSSCHECK',
        'parent_all_true_provenance_known': True,
        'parent_gold_known_before_review': True,
        'fresh_blind_parent_review': False,
        'current_solver_scores_read': False, 'solver_scores_used': False,
        'new_variant_private_artifacts_read_before_save': False,
        'finish_review_invoked': False, 'results': results,
    }
    status_name = ('ASSET_SOURCE_REVIEW_STATUS.json' if round_id == 'initial'
                   else 'ASSET_SOURCE_REVIEW_STATUS.' + round_id + '.json')
    write_new(HOME / 'evidence' / status_name, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
