"""Aggregate saved checks and typed imports; never perform semantic review."""
from collections import Counter
from datetime import datetime, timezone
import json
from state_v122 import B, bind, read


def main():
    states, carried_states, checks, warnings, post = Counter(), Counter(), Counter(), [], []
    scope = read(B / 'REPAIR_SCOPE.json')
    for row in scope['roots']:
        from pathlib import Path
        unit = Path(row['unit_path'])
        for draft in unit.glob('roots/*/scenes/*/EXPANSION_DRAFT.json'):
            scene = read(draft)['scene_ref']
            if scene in row['proposed_retained_scene_refs']: continue
            path = draft.parent / 'SOURCE_REVIEW.json'
            doc = read(path) if path.exists() else {'decision': 'NO_SUCCESSFUL_REVIEW'}
            target = carried_states if scene in row.get('fixed_carried_scene_refs', []) else states
            target[doc['decision']] += 1
            for name, check in doc.get('checks', {}).items():
                checks[name + ':' + check['status']] += 1
                if check['status'] != 'PASS':
                    warnings.append({'root_id': row['root_id'], 'scene': scene, 'check': name,
                                     **check, 'review_binding': bind(path)})
            finding = draft.parent / 'POST_REVIEW_BLOCKING_FINDING.json'
            if finding.exists(): post.append({'root_id': row['root_id'], 'scene': scene, 'finding_binding': bind(finding), 'finding': read(finding)})
    now = datetime.now(timezone.utc)
    report = {'schema_version': 'v122_saved_quality_findings_v1', 'created_at_utc': now.isoformat(),
              'fixed_root_count': scope['fixed_candidate_count'], 'new_scene_review_decisions': dict(states),
              'carried_scene_review_decisions': dict(carried_states), 'check_status_counts': dict(checks),
              'non_pass_checks': warnings, 'post_review_blocking_findings': post,
              'execution_issue_bindings': [bind(p) for p in sorted((B / 'evidence').glob('*.json')) if any(x in p.name for x in ('DEVIATION', 'FAILURE', 'FALLTHROUGH'))],
              'generation_usage_tokens': None, 'exact_generator_model_id': None, 'generation_cost': None,
              'wall_time_is_not_active_model_compute_or_billed_usage': True,
              'scope': 'Saved judgment aggregation, including held candidates and exact historical carries; no new semantic certification.',
              'training_or_pool_merge': False}
    for name, key in [('SOURCE_PROTOCOL_FREEZE.json', 'source_protocol'), ('MASTER_FREEZE.json', 'generation')]:
        start = datetime.fromisoformat(read(B / name)['created_at_utc'])
        report['elapsed_wall_seconds_since_' + key + '_freeze'] = (now - start).total_seconds()
    with (B / 'delivery/QUALITY_FINDINGS_V1.json').open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('non_pass_checks', 'post_review_blocking_findings')}, ensure_ascii=False))


if __name__ == '__main__': main()
