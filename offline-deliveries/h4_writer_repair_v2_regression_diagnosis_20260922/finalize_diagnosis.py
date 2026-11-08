"""Validate and publish a supplemental correction without touching closed data."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from metrics_check import OUT, RUN, WORK, REPORTS, main as check_metrics, read, rows


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    findings = read(OUT / 'admission_findings.json')
    scan = findings['citation_pass_scan']
    corrections = scan['corrections']
    inventory = rows(REPORTS / 'audited_candidate_inventory.jsonl')
    def identity(row):
        return row['arm_id'], row['root_id'], row['candidate_id']
    changes = {identity(row): row for row in corrections}
    assert len(changes) == len(corrections) == 6
    assert len({row['view_id'] for row in corrections}) == 5
    adjusted = []
    for row in inventory:
        change = changes.get(identity(row))
        if change:
            assert not row['final_retained']
            assert row['view_id'] == change['view_id']
            for key, value in change['original_judgments'].items():
                assert row[key] == value
            source = read(RUN / 'inputs/TRANSCRIPT_ONLY' / row['root_id'] / 'expanded.json')
            catalog = {t['turn_ref']: t for t in source['turns']}
            for unknown in change['unknown_refs']:
                assert unknown['submitted_nonexistent_ref'] not in catalog
                canonical = unknown['diagnostic_recoverable_canonical_ref']
                assert catalog[canonical] == unknown['source_turn_under_canonical_ref']
            adjusted.append({**row, **change['suggested_corrected_judgments']})
        else:
            adjusted.append(row)
    assert len(adjusted) == len(inventory) == 63
    effects = []
    for arm in ('TEMPORAL_V1', 'GROUNDED_V2'):
        original = [r for r in inventory if r['arm_id'] == arm]
        corrected = [r for r in adjusted if r['arm_id'] == arm]
        final = [r for r in corrected if r['final_retained']]
        effective = {kind: len({(r['root_id'], r['semantic_cluster_key']) for r in final
            if r['content_judgment'] == 'PASS' and
            ((r['question_type'] == 'ordering') if kind == 'ordering' else
             r['question_type'] in ('single_choice', 'multi_select'))}) for kind in ('ordering', 'SCMS')}
        assert effective == {'ordering': 0, 'SCMS': 5}
        effects.append({'arm_id': arm,
            'original_all_raw_content': dict(Counter(r['content_judgment'] for r in original)),
            'corrected_all_raw_content': dict(Counter(r['content_judgment'] for r in corrected)),
            'final_retained_content': dict(Counter(r['content_judgment'] for r in final)),
            'effective_counts_unchanged': effective})
    write('review_corrections.json', {
        'status': 'APPLIED_IN_SUPPLEMENT_ONLY',
        'reviewer': 'Codex offline diagnostic',
        'reason': 'Earlier exact-reference citation PASS labels were incorrect; source fact existence does not repair nonexistent submitted references.',
        'original_report_unchanged': True, 'original_runtime_admission_unchanged': True,
        'checked_citation_pass_occurrences': scan['citation_pass_occurrences_checked'],
        'affected_occurrences': 6, 'affected_unique_views': 5,
        'corrections': corrections, 'aggregate_effects': effects,
        'additional_model_or_review_api_calls': 0})
    assert findings['old_new_row_results_all_identical']
    assert findings['strict_scms_row_replays'] == 46
    assert findings['row_result_counts'] == {'PASS': 38, 'REJECT': 8}
    check_metrics()
    (OUT / 'INDEX.md').write_text('''# 排查产物

- [综合原因、证据和最小修改建议](DIAGNOSIS.md)
- [真实提示与六次弃答逐项核对](prompt_findings.md)
- [原文可排序性与诊断见证](orderability_findings.md)
- [接收与 repair 排查](admission_findings.md)
- [46 条旧新规则复放](admission_row_replay.json)
- [上次离线引用诊断的补充更正](review_corrections.json)
- [原指标、冻结输入与封存执行身份](metrics_check.json)
- [本次排查终检](validation.json)

对应 JSON 保存详细路径、引用、候选身份和机械核对结果。
诊断见证不计为模型产量，也没有进入生成提示。
生产代码、原冻结输入、19 次调用及封存报告均未改动；新增 API 调用为 0。
''')
    write('validation.json', {'status': 'PASS',
        'old_new_scms_replay_rows': 46, 'admission_decisions_identical': True,
        'supplemental_correction_occurrences': 6, 'affected_unique_views': 5,
        'final_supported_output_counts_unchanged': True,
        'original_frozen_files_unchanged': 111, 'original_execution_files_unchanged': 195,
        'original_generation_http_attempts': 19,
        'new_api_calls': 0, 'production_code_modified': False})
    paths = sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'diagnosis_manifest.json')
    write('diagnosis_manifest.json', {'files': [{'path': str(p.relative_to(OUT)),
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'size': p.stat().st_size} for p in paths]})
    print(json.dumps({'status': 'PASS', 'supplemental_corrections': len(corrections),
                      'effective_counts': [r['effective_counts_unchanged'] for r in effects],
                      'diagnostic_files': len(paths)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
