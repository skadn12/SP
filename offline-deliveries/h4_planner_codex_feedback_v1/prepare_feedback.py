"""Prepare Codex-reviewed feedback through the existing Planner route, offline."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

BASE = Path('/root/autodl-tmp/gxy-selfplay')
WORKTREE = BASE / 'worktrees/h4-planner-recovery-more-roots-v1'
DELIVERY = Path(__file__).resolve().parent
SOURCE = BASE / 'offline-deliveries/h4_planner_recovery_more_roots_v1_inputs'
OLD_RUN = BASE / 'restart-n5-runs/h4_full_pipeline_planner_recovery_more_roots_v1_live_1'
BUNDLE = DELIVERY / 'inputs'
RUN = DELIVERY / 'prepared_run'
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import full_pipeline_planner_recovery_more as entry
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live import _load_feedback_context


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def evidence(refs, turns):
    if not refs or len(refs) != len(set(refs)):
        raise ValueError('missing_or_duplicate_evidence_refs')
    result = []
    for ref in refs:
        if ref not in turns:
            raise ValueError('evidence_outside_frozen_P_R01:' + ref)
        t = turns[ref]
        result.append({'turn_ref': ref, 'speaker': t['speaker'], 'text': t['text']})
    return result


def main():
    if BUNDLE.exists() or RUN.exists():
        raise FileExistsError('feedback_delivery_already_prepared_no_overwrite')
    source_manifest = read(SOURCE / 'manifest.json')
    check_freeze(source_manifest)
    check_freeze(read(OLD_RUN / 'freeze_manifest.json'))
    files = {Path(r['path']).resolve() for r in source_manifest['files']}
    files.update([SOURCE / 'manifest.json', Path(__file__).resolve()])
    rows, cards = [], []
    for source_row in source_manifest['roots']:
        root = source_row['root_id']
        draft_path = DELIVERY / 'drafts' / (root.split('-')[-1] + '.json')
        draft = read(draft_path)
        if draft['root_id'] != root or not 1 <= len(draft['fact_checks']) <= 4:
            raise ValueError('invalid_feedback_draft_identity_or_count')
        original = read(source_row['input_path'])
        turns = {t['turn_ref']: t for t in original['prior_expanded']['turns']}
        checks, uncertainty = [], []
        for finding in draft['fact_checks']:
            checks.append({**finding, 'source_evidence': evidence(finding['evidence_refs'], turns)})
        for item in draft.get('uncertainties', []):
            uncertainty.append({**item, 'source_evidence': evidence(item['evidence_refs'], turns)})
        findings = {f['id'] for f in checks}
        if len(findings) != len(checks):
            raise ValueError('duplicate_feedback_finding_id')
        for example in draft.get('reviewed_failure_examples', []):
            source = Path(example['artifact_path']).resolve()
            if not (source.is_relative_to(OLD_RUN) or source.is_relative_to(SOURCE)) or not source.is_file():
                raise ValueError('failure_example_outside_original_run')
            if not set(example['linked_fact_check_ids']).issubset(findings):
                raise ValueError('failure_example_unknown_finding')
            files.add(source)
        card = {
            'feedback_kind': 'CODEX_OFFLINE_DIAGNOSIS',
            'author': 'Codex',
            'semantic_certification': 'NOT_INDEPENDENT_HUMAN_CERTIFICATION',
            'historical_fact_scope': 'FROZEN_P_PLUS_R01_ONLY',
            'evaluation_scope': 'KNOWN_CASES_WITH_CODEX_ASSISTANCE_NOT_AN_UNASSISTED_BASELINE',
            'solver_behavior_observations': 'NOT_PROVIDED_IN_THIS_REVIEW',
            'use': 'Use source-bound findings to preserve history and distinguish event identities and states. These are fallible review judgments, not new facts or a required scene list. Read the full transcript; choose useful new developments independently.',
            'fact_checks': [{k:v for k,v in finding.items() if k != 'source_evidence'} for finding in checks],
            'uncertainties': [{k:v for k,v in item.items() if k != 'source_evidence'} for item in uncertainty],
            'optional_planning_guidance': draft.get('optional_guidance', []),
            'reviewed_issues_not_additional_history': [
                {'observed_error': item['observed_error'],
                 'linked_fact_check_ids': item['linked_fact_check_ids'],
                 'status': 'REVIEW_OBSERVATION_NOT_AUTHORITATIVE_HISTORY'}
                for item in draft.get('reviewed_failure_examples', [])
            ],
            'date_boundary': 'Existing dates are synthetic conversation anchors. Preserve retrospective and prospective wording. New R02 dates remain UNKNOWN; do not force a total order from shared dates or missing timestamps.',
        }
        record = {
            'version': 'codex_source_bound_planner_feedback_v1', 'root_id': root,
            'policy': 'CODEX_AUTHORED_LOCAL_REVIEW_OF_FROZEN_HISTORY',
            'source_cluster_id': root + ':P_R01:codex_feedback_v1',
            'source_candidate_id': None,
            'source_feedback_ids': [root + ':CODEX_OFFLINE_V1'],
            'historical_qa': {'source_data_status': {
                'canonical_gold': 'missing', 'options_or_items': 'missing'}},
            'canonical_observation': {'basis': 'Codex offline review with exact frozen transcript references; not a Solver capability diagnosis.'},
            'performance': card,
        }
        target = BUNDLE / root
        write(target / 'resolved_evidence.json', {
            'root_id': root, 'scope': 'FROZEN_P_PLUS_R01_ONLY',
            'fact_checks': checks, 'uncertainties': uncertainty,
            'mechanical_check': 'Exact source lookup, not semantic entailment certification.',
        })
        provenance = {
            'review_author': 'Codex', 'review_type': 'CODEX_OFFLINE_DIAGNOSIS',
            'source_inputs_path': source_row['input_path'],
            'source_inputs_sha256': sha(source_row['input_path']),
            'replaced_feedback_path': source_row['source_feedback_path'],
            'replaced_feedback_sha256': sha(source_row['source_feedback_path']),
            'draft_path': str(draft_path), 'draft_sha256': sha(draft_path),
            'failure_examples_are_not_history': True,
            'failure_example_sources': [
                {**item, 'sha256': sha(item['artifact_path'])}
                for item in draft.get('reviewed_failure_examples', [])],
            'scope': 'Four known roots; only feedback input changed. Current observations may assist later generation, so later results are assisted diagnostics.',
            'extra_model_calls': 0,
        }
        write(target / 'feedback_input.json', {'record': record, 'offline_review_provenance': provenance})
        loaded, dialogue = _load_feedback_context(original['parent'], feedback_input=target / 'feedback_input.json')
        assert loaded == record and dialogue == original['feedback_dialogue']
        new = deepcopy(original)
        new['feedback_record'] = record
        new['provenance']['feedback_provenance'] = provenance
        new['provenance']['sources']['feedback'] = {'path': str(target / 'feedback_input.json'), 'sha256': sha(target / 'feedback_input.json')}
        write(target / 'inputs.json', new)
        for name in ['history_dated.txt', 'history_plain.txt']:
            shutil.copyfile(SOURCE / root / name, target / name)
        unchanged = ['parent', 'prior_expanded', 'snapshot', 'feedback_dialogue', 'conversation_date_segments']
        assert all(new[k] == original[k] for k in unchanged)
        row = deepcopy(source_row)
        row['input_path'] = str(target / 'inputs.json')
        row['source_feedback_path'] = str(target / 'feedback_input.json')
        row['feedback_review'] = provenance
        rows.append(row)
        cards.append({'root_id': root, 'card': card, 'provenance': provenance,
                      'resolved_checks': checks, 'resolved_uncertainties': uncertainty})
        files.add(draft_path)
    files.update(p for p in BUNDLE.rglob('*') if p.is_file())
    manifest = {
        'version': 'codex_source_bound_planner_feedback_inputs_v1',
        'status': 'OFFLINE_FEEDBACK_PREPARED_NO_EXECUTION_AUTHORIZED',
        'root_order': list(entry.ROOT_ORDER), 'roots': rows, 'external_calls': 0,
        'replaces': 'Historical derived feedback only; original history, dates and Memory unchanged.',
        'files': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sorted(files)],
    }
    write(BUNDLE / 'manifest.json', manifest)
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')) as network, \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')) as connect, \
         patch.object(entry.shared, 'resolve_credential', side_effect=AssertionError('NO_CREDENTIAL_READ')) as credentials, \
         patch.object(entry.shared, 'ProfileClient', side_effect=AssertionError('NO_TRANSPORT')) as transport, \
         patch.object(entry.shared, 'QARuntime', side_effect=AssertionError('NO_QA_RUNTIME')) as qa:
        prepared = entry.prepare(RUN, input_bundle=BUNDLE)
        for blocked in [network, connect, credentials, transport, qa]:
            blocked.assert_not_called()
    checks = []
    for root in entry.ROOT_ORDER:
        frozen = RUN / 'frozen' / root
        old_prompt = read(OLD_RUN / 'frozen' / root / 'planner_prompt.json')
        new_prompt = read(frozen / 'planner_prompt.json')
        old_view, new_view = canonical(old_prompt['feedback_view']), canonical(new_prompt['feedback_view'])
        assert old_prompt['prompt'].count(old_view) == new_prompt['prompt'].count(new_view) == 1
        assert old_prompt['prompt'].replace(old_view, new_view) == new_prompt['prompt']
        old_request = read(OLD_RUN / 'frozen' / root / 'planner_request.json')
        new_request = read(frozen / 'planner_request.json')
        assert {k:v for k,v in old_request.items() if k != 'messages'} == {k:v for k,v in new_request.items() if k != 'messages'}
        assert len(old_request['messages']) == len(new_request['messages']) == 1
        assert new_request['messages'][0]['content'] == new_prompt['prompt']
        assert old_request['messages'][0]['role'] == new_request['messages'][0]['role']
        for field in ['coverage_gaps', 'gap_type', 'frontier_status', 'desired_event_operator', 'global_priority', 'feedback_card']:
            assert ('"' + field + '"') not in new_view
        inputs = read(frozen / 'inputs.json')
        assert all(t['text'] in new_prompt['prompt'] and t['turn_ref'] in new_prompt['prompt'] for t in inputs['prior_expanded']['turns'])
        capacity = read(frozen / 'planner_capacity.json')
        assert capacity['status'] == 'PASS' and capacity['truncation'] is False
        checks.append({'root_id': root, 'feedback_only_prompt_change': True,
            'parameters_unchanged': True, 'complete_P_R01_visible': True,
            'old_derived_labels_absent': True, 'feedback_view_occurrences': 1,
            'planner_request_path': str(frozen / 'planner_request.json'),
            'planner_prompt_path': str(RUN / 'prepared' / root / 'planner/prompt.txt'),
            'capacity': capacity})
    check_freeze(manifest)
    check_freeze(prepared)
    check_freeze(source_manifest)
    check_freeze(read(OLD_RUN / 'freeze_manifest.json'))
    assert not (RUN / 'EXECUTION_STARTED.json').exists()
    report = {'status': 'PASS_OFFLINE_FEEDBACK_PREPARATION',
        'prepared_at': datetime.now(timezone.utc).isoformat(),
        'root_order': list(entry.ROOT_ORDER), 'external_calls': 0, 'credential_reads': 0,
        'transport_constructions': 0, 'network_attempts': 0,
        'existing_frozen_inputs_and_run_unchanged': True,
        'semantic_review': 'CODEX_OFFLINE_DIAGNOSIS_NOT_INDEPENDENT_CERTIFICATION',
        'reference_check': 'All feedback evidence resolves to exact frozen P+R01 text; this checks identity, not semantic entailment.',
        'implementation_changes': 'NONE; new local feedback records use the existing loader and formal preparation entry.',
        'live_quality': 'NOT_EVALUATED_NO_NEW_GENERATIONS',
        'run_root': str(RUN), 'input_bundle': str(BUNDLE), 'checks': checks,
        'generation_authorized': False,
        'known_unfixed_blockers': ['Realizer scene wrapping', 'Realizer CURRENT STATE authority conflict', 'Outline semantic compatibility']}
    write(DELIVERY / 'VERIFICATION.json', report)
    lines = ['# Codex 离线 Planner 反馈', '',
        '四条已知案例；事实范围仅完整 P+R01。原文、日期、Memory 和生成参数不变。',
        '这是 Codex 离线诊断，不是独立人工认证；后续使用此反馈的结果属于 Codex 辅助后的案例验证。',
        '旧编译卡已整体替换。所有原文片段由程序按 ID 回取；UNKNOWN 不升级为确定事实。', '']
    for item in cards:
        root, card = item['root_id'], item['card']
        lines += ['## ' + root, '']
        for f in item['resolved_checks']:
            lines += [f"**{f['id']} · {f['judgment']}**：{f['claim']}", '', '规划边界：' + f['planning_implication'], '']
            for ev in f['source_evidence']:
                lines += [f"- `{ev['turn_ref']}` · {ev['speaker']}：{ev['text']}"]
            lines.append('')
        for u in card['uncertainties']:
            lines += ['**不确定项**：' + u['issue'], '', '处理：' + u['handling'], '']
        for ex in card['reviewed_issues_not_additional_history']:
            lines += ['**来源或先前尝试的问题（不构成新增历史事实）**：' + ex['observed_error'], '']
        lines += [f"[反馈输入]({BUNDLE / root / 'feedback_input.json'}) · [实际准备的 Planner 请求]({RUN / 'prepared' / root / 'planner/prompt.txt'})", '']
    (DELIVERY / 'FEEDBACK_REVIEW.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'roots': list(entry.ROOT_ORDER),
        'external_calls': 0, 'feedback_only_prompt_change': True,
        'feedback_review': str(DELIVERY / 'FEEDBACK_REVIEW.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
