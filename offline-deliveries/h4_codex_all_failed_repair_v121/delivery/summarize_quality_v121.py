"""Consolidate recorded judgments without certifying semantics or changing data."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    manifest = read(B / 'MANIFEST.json')
    states, checks, warnings, post = Counter(), Counter(), [], []
    for row in manifest['roots']:
        unit = Path(row['unit_path'])
        for draft in unit.glob('roots/*/scenes/*/EXPANSION_DRAFT.json'):
            path = draft.parent / 'SOURCE_REVIEW.json'
            scene = path.parent.name
            if 'R02:' + scene.replace('_', ':') in row['retained_scene_refs']:
                continue
            document = read(path) if path.exists() else {'decision': 'NO_SUCCESSFUL_REVIEW'}
            states[document['decision']] += 1
            for name, check in document.get('checks', {}).items():
                checks[name + ':' + check['status']] += 1
                if check['status'] != 'PASS':
                    warnings.append({'root_id': row['root_id'], 'scene': scene, 'check': name,
                                     **check, 'review_binding': bind(path)})
            finding = path.parent / 'POST_REVIEW_BLOCKING_FINDING.json'
            if finding.exists():
                post.append({'root_id': row['root_id'], 'scene': scene, 'finding_binding': bind(finding),
                             'original_decision': document['decision'], 'finding': read(finding)})
    now = datetime.now(timezone.utc)
    source_start = datetime.fromisoformat(read(B / 'SOURCE_PROTOCOL_FREEZE.json')['created_at_utc'])
    generation_start = datetime.fromisoformat(read(B / 'MASTER_FREEZE.json')['created_at_utc'])
    report = {'schema_version': 'v121_saved_quality_findings_v1', 'created_at_utc': now.isoformat(),
        'fixed_root_count': 24, 'saved_new_scene_review_decisions_before_later_findings': dict(states),
        'check_status_counts': dict(checks), 'non_pass_checks': warnings, 'post_review_blocking_findings': post,
        'elapsed_wall_seconds_since_source_protocol_freeze': (now - source_start).total_seconds(),
        'elapsed_wall_seconds_since_generation_freeze': (now - generation_start).total_seconds(),
        'wall_time_is_not_active_model_compute_or_billed_usage': True,
        'execution_deviation_binding': bind(B / 'evidence/PARENT_COLLECTOR_FREEZE_DEVIATION_V1.json'),
        'review_preflight_execution_deviation_binding': bind(B / 'evidence/REVIEW_PREFLIGHT_FALLTHROUGH_B_V1.json'),
        'generation_usage_tokens': None, 'exact_generator_model_id': None, 'generation_cost': None,
        'scope': 'Descriptive aggregation of saved judgments; includes the erroneous READY subsequently blocked. It is not an additional semantic review or proof of correctness.',
        'training_or_pool_merge': False}
    path = B / 'delivery/QUALITY_FINDINGS_V1.json'
    with path.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('non_pass_checks', 'post_review_blocking_findings')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
