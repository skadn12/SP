"""Verify and index this completed run offline; no generation or network imports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

BASE = Path('/root/autodl-tmp/gxy-selfplay')
RUN = BASE / 'restart-n5-runs/h4_full_pipeline_planner_recovery_v1_live_1'
REPORTS = RUN / 'reports'
DELIVERY = BASE / 'offline-deliveries/h4_planner_candidate_recovery_v1'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    generation_sha = sha(RUN / 'generation_summary.json')
    summary = read(REPORTS / 'FINAL_SUMMARY.json')
    audit = read(REPORTS / 'execution_audit.json')
    costs = read(REPORTS / 'complete_cost_audit.json')
    qa = read(REPORTS / '136202_qa_content_diagnosis.json')['summary']
    generation = read(RUN / 'generation_summary.json')
    status = read(RUN / 'run_status.json')

    # Make the acceptance boundary explicit in both the report and its generator.
    old = '两条都选出了 3 个合格计划场景，均未触发补充。'
    new = '两条均有 3 个计划场景通过现有 Gate／Outline 检查并被程序选中，均未触发补充；这不代表场景语义已全部合格。'
    for path in [REPORTS / 'FINAL_REPORT.md', DELIVERY / 'finalize_live_reports.py']:
        path.write_text(path.read_text().replace(old, new))
    definition = 'Selected after existing Gate/Outline and program checks; not semantic certification.'
    summary['qualified_plan_scenes_definition'] = definition
    write(REPORTS / 'FINAL_SUMMARY.json', summary)

    # Resolve the earlier cost audit's pending semantic denominator by citation.
    costs['semantic_diagnosis_source'] = str(REPORTS / '136202_qa_content_diagnosis.json')
    costs['semantic_diagnosis_method'] = summary['review_method']
    costs['semantic_effective_question_count'] = {
        'full_transcript_supported_semantic_clusters': 5,
        'citation_supported_subset': 4,
        'ordering': 0,
        'unevaluated_root': 'cc-train-episode-197635',
    }
    costs['cost_per_semantically_effective_question'] = summary['cost_per_supported_deduplicated_question']
    write(REPORTS / 'complete_cost_audit.json', costs)

    records = audit['raw_request_response_checks']
    assert len(records) == 16
    assert sum(r['http_attempts'] for r in records) == 16 <= 34
    assert all(r['http_status'] == 200 and not r['timeout'] and r['retries'] == 0 for r in records)
    assert all(r['actual_model'] == 'qwen3.6-35b-a3b' and r['finish_reason'] == 'stop' for r in records)
    for key, expected in [('input_tokens', 212284), ('output_tokens', 23448), ('total_tokens', 235732)]:
        assert sum(r[key] for r in records) == expected
        assert summary['cost'][key] == generation['cost'][key] == status['cost'][key] == expected
    assert summary['cost'] == audit['observed_cost']
    assert Counter(r['stage'] for r in records) == {
        'planner': 2, 'gate': 2, 'outline': 2, 'realizer': 2,
        'adherence': 1, 'miner': 1, 'writer_primary': 3, 'writer_repair': 3,
    }
    assert qa['finally_retained'] == 9
    assert qa['retained_content'] == {'PASS': 7, 'FAIL': 2}
    assert qa['full_transcript_supported_scms_after_semantic_dedup'] == 5
    assert qa['citation_complete_supported_scms_after_semantic_dedup'] == 4
    assert qa['supported_ordering_after_semantic_dedup'] == 0
    assert qa['repair_net_new_retained'] == 0
    assert qa['deduplicated_supported_scms_requiring_new_R02'] == 3

    raw_checks = []
    compiled_outputs = (REPORTS / 'ALL_MODEL_OUTPUTS.md').read_text()
    for record in records:
        for pkey, hkey in [('request_path', 'request_sha256'), ('response_path', 'response_sha256'),
                           ('full_metadata_path', 'full_metadata_sha256')]:
            assert sha(Path(record[pkey])) == record[hkey], record[pkey]
        raw = Path(record['response_path']).read_text()
        assert raw and raw in compiled_outputs, record['response_path']
        raw_checks.append({'root_id': record['root_id'], 'stage': record['stage'],
                           'intent_id': record['intent_id'], 'response_path': record['response_path'],
                           'sha256': record['response_sha256'], 'verbatim_in_compilation': True})

    freeze_checks = []
    for name, count in [('freeze_manifest.json', 296), ('frozen/input_bundle_manifest.json', 57)]:
        files = read(RUN / name)['files']
        assert len(files) == count
        for entry in files:
            assert sha(Path(entry['path'])) == entry['sha256'], entry['path']
        freeze_checks.append({'manifest': name, 'file_count': count, 'status': 'PASS'})
    assert sha(RUN / 'generation_summary.json') == generation_sha

    index = REPORTS / 'FILE_INDEX.md'
    index_text = index.read_text()
    additions = []
    for name in ['FINAL_REPORT.md', 'FINAL_SUMMARY.json', 'ALL_MODEL_OUTPUTS.md',
                 '136202_expansion_diagnosis.md', '197635_expansion_diagnosis.md',
                 '136202_qa_content_diagnosis.md', '136202_qa_content_diagnosis.json']:
        if str(REPORTS / name) not in index_text:
            additions.append(f'- [{name}]({REPORTS / name})')
    if additions:
        index.write_text(index_text + '\n## 最终收口与内容诊断\n\n' + '\n'.join(additions) + '\n')

    link_count = 0
    missing = []
    for path in REPORTS.glob('*.md'):
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            local = re.sub(r':\d+$', '', target.strip('<>').split('#')[0])
            resolved = Path(local) if Path(local).is_absolute() else path.parent / local
            link_count += 1
            if not resolved.exists():
                missing.append({'file': path.name, 'target': target})
    assert not missing, missing

    verification = {
        'status': 'PASS', 'verified_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Completed two-root run only; artifact consistency, not semantic certification.',
        'api_calls_added': 0, 'http_attempts': 16, 'approved_http_maximum': 34,
        'input_tokens': 212284, 'output_tokens': 23448, 'total_tokens': 235732,
        'model': 'qwen3.6-35b-a3b', 'retries': 0, 'http_technical_failures': 0,
        'root_197635_failure': 'Local Realizer scene contract; no adherence model or QA call.',
        'qa_retained': 9, 'qa_full_source_pass_fail': {'PASS': 7, 'FAIL': 2},
        'qa_deduplicated_full_source_supported': 5, 'qa_deduplicated_citation_supported': 4,
        'effective_ordering': 0, 'repair_net_new': 0, 'live_supplement_calls': 0,
        'markdown_links_checked': link_count, 'missing_links': missing,
        'raw_request_response_and_metadata_integrity': 'PASS',
        'all_raw_responses_verbatim': raw_checks, 'freeze_checks': freeze_checks,
        'generation_summary_unchanged_sha256': generation_sha,
        'independent_codex_consistency_review': 'PASS after clarifying scene acceptance is not semantic certification.',
        'production_code_or_raw_generation_modified': False,
    }
    write(REPORTS / 'FINAL_VERIFICATION.json', verification)
    artifacts = []
    for path in sorted(RUN.rglob('*')):
        if path.is_file() and path != REPORTS / 'ARTIFACT_MANIFEST.json':
            artifacts.append({'path': str(path.relative_to(RUN)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    write(REPORTS / 'ARTIFACT_MANIFEST.json', {
        'version': 'final_current_run_artifacts_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'run_root': str(RUN), 'scope': 'Current run only; manifest itself excluded.',
        'file_count': len(artifacts), 'files': artifacts,
    })
    assert all(sha(RUN / row['path']) == row['sha256'] for row in artifacts)
    print(json.dumps({'status': 'PASS', 'raw_responses': len(raw_checks), 'links': link_count,
                      'current_run_artifacts': len(artifacts), 'added_api_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
