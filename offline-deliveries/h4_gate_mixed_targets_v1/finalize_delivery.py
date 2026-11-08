"""Finalize local Gate-only artifacts; performs no model or network calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
RUN = BASE / 'prepared_run'
SOURCE = BASE.parent / 'h4_planner_output_contract_v1/prepared_run'
ROOTS = [f'cc-train-episode-{n}' for n in (259192, 96462, 112812, 218519)]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = read(BASE / 'reports/EXECUTION_AUDIT.json')
    tests = read(BASE / 'TEST_RESULTS.json')
    generation = read(RUN / 'generation_summary.json')
    assert audit['status'] == 'PASS' and not audit['errors']
    assert tests['tests_run'] == 329 and tests['failures'] == tests['errors'] == 0
    assert generation['status'] == 'GATE_ONLY_FINISHED'
    assert generation['cost']['http_attempts'] == 8
    assert generation['cost']['total_tokens'] == 184868
    assert generation['other_stage_calls'] == 0
    freeze_counts = {}
    for name, root in [('current', RUN), ('previous', SOURCE)]:
        records = read(root / 'freeze_manifest.json')['files']
        assert all(sha(Path(row['path'])) == row['sha256'] for row in records)
        freeze_counts[name] = len(records)
    for row in read(BASE / 'CHANGES.json'):
        assert sha(Path(row['path'])) == row['sha256']
    for filename in ['CONTENT_259192_96462.json', 'CONTENT_112812_218519.json']:
        assert read(BASE / 'reports' / filename)
    slot_index = []
    eligible = Counter()
    old_eligible = Counter()
    new_relations = Counter()
    for root in ROOTS:
        for stage in ['gate', 'gate_supplement']:
            frozen = RUN / 'frozen' / root / stage
            result = RUN / 'results' / root / stage
            calls = RUN / 'execution/calls' / root / stage
            paths = {
                'original_planner_output': frozen / 'planner_output.json',
                'original_planner_validation': frozen / 'planner_validation.json',
                'compiled_candidates': frozen / 'plan.json',
                'full_source_and_memory_catalog': frozen / 'context.json',
                'old_gate_prompt': frozen / 'old_prompt.txt',
                'old_gate_output': frozen / 'old_output.json',
                'old_gate_validation': frozen / 'old_validation.json',
                'new_actual_messages': calls / 'prompt.json',
                'new_actual_prompt_text': calls / 'prompt.txt',
                'new_actual_request': calls / 'request.json',
                'new_raw_model_output': calls / 'response.txt',
                'response_metadata_full_usage': calls / 'response_metadata.json',
                'new_parsed_output': result / 'output.json',
                'new_validation': result / 'validation.json',
                'comparison': result / 'comparison.json',
                'input_identity': frozen / 'identity.json',
            }
            assert all(p.is_file() for p in paths.values())
            old = read(paths['old_gate_validation'])
            new = read(paths['new_validation'])
            raw = read(paths['new_parsed_output'])
            old_eligible[root] += len(old['passed_candidate_refs'])
            eligible[root] += len(new['passed_candidate_refs'])
            new_relations.update(r['relation'] for r in raw['candidate_reviews'])
            slot_index.append(dict(root_id=root, stage=stage,
                                   paths={k: str(v) for k, v in paths.items()}))
    assert sum(eligible.values()) == 14 and sum(old_eligible.values()) == 11
    assert new_relations == {'NEW_PROPOSAL': 12, 'PLANNED_OR_INTENDED': 3}
    write(BASE / 'ARTIFACT_INDEX.json', slot_index)
    lines = ['# Gate原候选重跑：人工查看索引', '',
             '只重跑Gate；这里的Planner文件是旧输出冻结副本，不是重新生成。每槽位的context.json包含完整原文与Memory导航目录，plan.json为当时结构合格候选。', '',
             '[最终报告](reports/FINAL_REPORT.md) · [机器可读全部路径](ARTIFACT_INDEX.json) · [所有调用与usage](prepared_run/complete_call_records.json)', '',
             '推荐顺序：原Planner输出与候选 → 原文catalog → 新Gate实际请求 → 原始响应 → 程序验证 → 离线诊断。', '']
    for slot in slot_index:
        lines += [f"## {slot['root_id']} / {slot['stage']}", '']
        for name, path in slot['paths'].items():
            lines.append(f'- [{name}]({path})')
        lines.append('')
    (BASE / 'ARTIFACT_INDEX.md').write_text('\n'.join(lines) + '\n')
    summary = dict(status='COMPLETE_STOPPED', scope='GATE_ONLY_ORIGINAL_PLANNER_CANDIDATES',
                   roots=ROOTS, candidate_count=15, old_eligible=dict(old_eligible),
                   new_eligible=dict(eligible), new_quarantined=1,
                   new_relation_counts=dict(new_relations),
                   validator_matrix=audit['validator_matrix'], cost=audit['cost'],
                   semantic_diagnosis='COMPLETE_CODEX_OFFLINE_NOT_INDEPENDENT_HUMAN_CERTIFICATION',
                   diagnostic_categories=dict(scoped_reasonable=6, undetermined=5, clear_issue=4),
                   semantic_repair_success='NOT_ESTABLISHED',
                   live_mixed_unknown_branch_used=False,
                   original_planner_unchanged=True, other_stage_calls=0,
                   automatic_next_stage=False, freeze_records_verified=freeze_counts,
                   tests_run=tests['tests_run'], report=str(BASE / 'reports/FINAL_REPORT.md'))
    write(BASE / 'FINAL_SUMMARY.json', summary)
    checked_links = 0
    for path in [BASE / 'README.md', BASE / 'ARTIFACT_INDEX.md', BASE / 'reports/FINAL_REPORT.md']:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            target = Path(link)
            if not target.is_absolute():
                target = path.parent / target
            assert target.exists(), (str(path), link)
            checked_links += 1
    write(BASE / 'FINAL_VERIFICATION.json', dict(
        status='PASS', checked_links=checked_links, tests_run=329,
        current_freeze_entries=freeze_counts['current'],
        previous_freeze_entries=freeze_counts['previous'],
        changed_code_hashes_unchanged_since_test=True,
        formal_calls=8, formal_http_attempts=8, formal_tokens=184868,
        finalization_external_calls=0, other_stage_calls=0,
        generated_at=datetime.now(timezone.utc).isoformat()))
    files = [p for p in BASE.rglob('*') if p.is_file() and p.name != 'DELIVERY_MANIFEST.json']
    records = [dict(path=str(p.relative_to(BASE)), sha256=sha(p)) for p in sorted(files)]
    write(BASE / 'DELIVERY_MANIFEST.json', dict(
        status='COMPLETE_STOPPED', files=records,
        scope='Only this Gate delivery; excludes its self-hash.', count=len(records)))
    assert all(sha(BASE / r['path']) == r['sha256'] for r in records)
    print(json.dumps(dict(status='PASS_COMPLETE_STOPPED', delivery_files=len(records),
                          checked_links=checked_links, http_attempts=8, total_tokens=184868)))


if __name__ == '__main__':
    main()
