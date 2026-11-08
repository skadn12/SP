"""Read-only reconstruction of the completed trial's outcome funnel."""
from collections import Counter
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_planner_writer_repair_v2_writer_prepare_1')
WORK = RUN / 'executions/writer_repair_comparison'
REPORTS = WORK / 'reports'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    inventory = rows(REPORTS / 'audited_candidate_inventory.jsonl')
    calls = rows(REPORTS / 'all_call_costs.jsonl')
    results = []
    for root in ('cc-train-episode-198480', 'cc-train-episode-122711'):
        for arm in ('TEMPORAL_V1', 'GROUNDED_V2'):
            local = [r for r in inventory if (r['root_id'], r['arm_id']) == (root, arm)]
            retained = [r for r in local if r['final_retained']]
            result = {'root_id': root, 'arm_id': arm, 'raw_objects': len(local),
                'raw_question_types': dict(Counter(r['question_type'] or 'MALFORMED_FRAGMENT' for r in local)),
                'program_retained_types': dict(Counter(r['question_type'] for r in retained)),
                'question_type_details': {}}
            for kind, types in (('ordering', ('ordering',)), ('SCMS', ('single_choice', 'multi_select'))):
                final = [r for r in retained if r['question_type'] in types]
                supported = [r for r in final if r['content_judgment'] == 'PASS']
                clusters = sorted({r['semantic_cluster_key'] for r in supported})
                result['question_type_details'][kind] = {
                    'final_content': dict(Counter(r['content_judgment'] for r in final)),
                    'semantic_pass_before_dedup': len(supported), 'effective_after_dedup': len(clusters),
                    'supported_duplicate_excess': len(supported)-len(clusters), 'clusters': clusters,
                    'semantic_rejections_or_uncertainties': [{k: r[k] for k in (
                        'candidate_id', 'slot_id', 'view_id', 'content_judgment',
                        'source_content_judgment', 'citation_support_judgment', 'reason')}
                        for r in final if r['content_judgment'] != 'PASS']}
            result['calls'] = dict(Counter(r['stage'] for r in calls if (r['root_id'], r['arm_id']) == (root, arm)))
            result['total_tokens'] = sum(r['total_tokens'] for r in calls if (r['root_id'], r['arm_id']) == (root, arm))
            results.append(result)
    primary_responses = []
    for path in sorted((WORK / 'arms/GROUNDED_V2').glob('roots/*/stage_calls/*/primary.json')):
        stage = read(path)
        raw = stage['raw']
        primary_responses.append({k: stage[k] for k in ('root_id', 'slot_id')} | {
            'raw_ordering_candidates': sum(c.get('question_type') == 'ordering' for c in raw['candidates']),
            'raw_abstentions': raw.get('abstentions', [])})
    assert len(primary_responses) == 6 and all(r['raw_ordering_candidates'] == 0 for r in primary_responses)
    frozen = read(RUN / 'audit/frozen_manifest.json')
    frozen_count = 0
    for group in ('files', 'source_files', 'code'):
        for record in frozen[group]:
            path = Path(record['path'])
            path = path if path.is_absolute() else RUN / path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
            assert path.stat().st_size == record['size']
            frozen_count += 1
    closed = read(REPORTS / 'closed_execution_manifest.json')
    for record in closed['files']:
        path = WORK / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], str(path)
        assert path.stat().st_size == record['size']
    http = rows(WORK / 'audit/http_events.jsonl')
    assert len(http) == len(calls) == 19
    result = {'status': 'PASS', 'by_root_arm': results,
        'grounded_primary_raw_responses': primary_responses,
        'original_frozen_files_unchanged': frozen_count,
        'original_closed_execution_files_unchanged': len(closed['files']),
        'original_generation_http_attempts': len(http), 'diagnostic_api_calls': 0,
        'metrics_scope': 'Original Codex diagnostic labels; no new human truth or witness counted as model output.'}
    (OUT / 'metrics_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('by_root_arm', 'grounded_primary_raw_responses')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
