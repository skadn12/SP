"""Materialize authored local feedback and a hash-bound ordering review; no API."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

BASE = Path('/root/autodl-tmp/gxy-selfplay')
DEST = Path(__file__).resolve().parent
WORKTREE = BASE / 'worktrees/h4-realizer-state-contract-v1'
RUN = BASE / 'restart-n5-runs/h4_full_pipeline_realizer_state_contract_v1_live_1'
OLD = BASE / 'offline-deliveries/h4_planner_codex_feedback_v1/inputs'
BUNDLE = DEST / 'inputs'
ROOTS = [f'cc-train-episode-{n}' for n in (259192, 96462, 112812, 218519)]
AUTHORITY = 'CODEX_OFFLINE_DIAGNOSIS_NOT_INDEPENDENT_HUMAN_CERTIFICATION'
sys.path.insert(0, str(WORKTREE))

from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live import _load_feedback_context
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1 import full_pipeline_planner_recovery_more as entry
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import check_freeze


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def finding(code, status, explanation, action, candidate_ids=(), refs=()):
    return dict(finding_id=code, status=status, explanation=explanation,
                recommended_action=action, candidate_ids=list(candidate_ids), evidence_refs=list(refs))


def authored_feedback(root):
    """Codex-authored decisions; this function is not an automated semantic judge."""
    cid = lambda suffix: root + ':' + suffix
    pref = lambda suffix: root + ':' + suffix
    if root.endswith('259192'):
        return [
            finding('O_UNSUPPORTED_PLAN_ONSET', 'FAIL',
                '三道排序均将车损、购车决定、注册、通勤计划排成唯一链。注册已发生，但T011只表达当前持有的计划，没有说明该计划何时形成。提及顺序不能证明注册早于形成通勤计划。I001题干还把first use写成已发生；I003题干与选项中的购买/决定粒度不一致。',
                '排除这三道原题。不能仅换Gold来修复缺少的时间约束，也不能把计划改成完成。',
                [cid('I001-C-Q03:primary'), cid('I002-C-Q02:repair'), cid('I003-C-Q04:primary')],
                ['R02:S02:T003', 'R02:S02:T007', 'R02:S02:T009', 'R02:S02:T011']),
            finding('SC_IMMEDIATE_PREDECESSOR', 'FAIL',
                'I003单选的Gold把车损作为购车决定的directly preceded事件。原文支持较早的原因，但未支持紧邻前序，T001还叙述昨天去了车店才作决定。',
                '未来出题应区分原因、较早事件和紧邻前序；本次原题及Gold不改写。',
                [cid('I003-C-Q02:primary')], [pref('P03:T001'), pref('P03:T004'), 'R02:S02:T001', 'R02:S02:T002']),
            finding('MS_PURPOSE_SCOPE', 'UNDETERMINED',
                'I002多选Gold包含通勤计划。该计划确实被提到，但题干要求为安全/责任而采取的行动；T011责任语境容许宽松理解，仍不足以无歧义确定是否把通勤计划纳入这一目的集合。',
                '记录目的范围歧义，不宣称该计划不存在；未来题干应明确是在问陈述的事实还是特定目的的行动。',
                [cid('I002-C-Q02:primary')], ['R02:S02:T007', 'R02:S02:T009', 'R02:S02:T011']),
            finding('POSITIVE_STATE_DISTINCTION', 'PASS_EXAMPLE',
                'I003多选区分了已购买、已注册、已买头盔和计划通勤。T009 purchased a helmet along with the bike支持购车已发生，不能反过来误判所有购车陈述都仅是计划。',
                '保留这种状态区分；不要为了避错把已有完成事实一律降成未知。',
                [cid('I003-C-Q03:primary')], ['R02:S02:T003', 'R02:S02:T007', 'R02:S02:T009', 'R02:S02:T011']),
            finding('COVERAGE_AND_DUPLICATION', 'OBSERVATION',
                '最终10题中的三道排序反复使用同一交通链；涉及新增R02事实的题集中在自行车场景。新演讲安排与Sarah互动未形成对应的有效新增覆盖。相同链换题干不构成三种新知识。',
                '先做候选语义去重和覆盖诊断。Planner已经产生的新增事实不能因Writer未选中就判为没有新增；不要为排序硬造更强时间关系。'),
        ]
    if root.endswith('218519'):
        return [
            finding('O_BOUND_EVIDENCE_AMBIGUITY', 'HOLD',
                'I002修复排序的drove to可被读作完成抵达，而节点只引正在驾驶；湖泳日期与后来对话中的earlier也不能单独给相遇事件的时间下界。P03:T005证明曾完成一次到朋友家的旅程，但未唯一确认它就是P02那次；P05:T007可补强湖泳与相遇的前后理解。这些额外上下文未绑定在该题对应节点/边，也不能仅补引用就当作事件身份已证实。旧诊断直接写earlier that day不够严谨，现撤回无条件PASS。',
                '原题暂缓纳入。保留原Gold，不称确定反向；全文可能支持、当前题面及所绑证据不足，须分别记录。',
                [cid('I002-C-Q02:repair')], [pref('P02:T003'), pref('P03:T001'), pref('P03:T005'), pref('P05:T001'), pref('P05:T007')]),
            finding('O_SUPPORTED_SPEECH_ACTION_CHAIN', 'PASS',
                'I003修复排序由Emily表达紧张、James建议专业课、James完成报名到Emily到场。建议与回应在原文中直接发生，后续课程现场有连续对话佐证；保留原Gold I1/I2/I3/I4。',
                '保留原题。这只是该已知案例的Codex诊断，不证明所有排序可靠。',
                [cid('I003-C-Q02:repair')], ['R01:S02:T001', 'R01:S02:T002', 'R01:S02:T008', 'R02:S02:T001', 'R02:S02:T002', 'R02:S02:T010', 'R02:S02:T011']),
            finding('MS_ATTENDANCE_CITATION', 'CITATION_INCOMPLETE_FULL_TEXT_SUPPORTS',
                'I003多选的上课已发生在完整场景中有依据，但所引T001只说有三点预约；不能把独句预约说明当作充分完成证据。到场对话和实际练习在T002、T010/T011可核对。',
                '未来证据需覆盖该选项所需状态，区分引用不足和全文事实错误；当前不改原题引用或Gold。',
                [cid('I003-C-Q02:primary')], ['R02:S02:T001', 'R02:S02:T002', 'R02:S02:T010', 'R02:S02:T011']),
            finding('MS_REGISTRATION_ACTOR', 'UNDETERMINED',
                'I002多选E说Emily signed up，但实际James代她操作且Emily同意。普通用法可指Emily完成报名这一受益状态；精确行动主体解释则有歧义，不宜直接断言人物事实错误。',
                '明确谁执行报名、谁被报名；保留歧义标签，不把James改成Emily或反向改历史。',
                [cid('I002-C-Q02:primary')], ['R01:S02:T006', 'R01:S02:T007', 'R01:S02:T008']),
            finding('MS_DISTRACTOR_SPECIFICITY', 'NO_MISSING_D_ESTABLISHED',
                'I001多选D是James帮助Emily解二次方程；P05只说过去帮助数学，没有支持二次方程这一具体对象，因此不据此给ABC补D。原诊断earlier that day修正为earlier than that conversation；此题不要求A/B/C内部排序。驾驶完成表述的引用歧义仍适用，故不把本反馈作为整题严格认证。',
                '逐项匹配人物、行动与对象，不能用一般数学帮助支持特定二次方程事件。',
                [cid('I001-C-Q02:primary')], [pref('P03:T005'), pref('P05:T001'), pref('P05:T010'), pref('P05:T011'), 'R02:S01:T005']),
        ]
    if root.endswith('96462'):
        return [finding('QA_NOT_REACHED', 'NOT_EVALUATED',
            '此root在Planner最终合格数量2/3处停止，未续写、未出题。补充3个候选结构及Gate均通过，但Outline只留2个。不能编造此root的排序正确率、Solver薄弱点或QA反馈。',
            '先修Planner合同损耗，并复核报名/具体安排是否被低估为重复；维持最少3个合格场景。')]
    return [finding('QA_NOT_REACHED', 'NOT_EVALUATED',
        '此root已有2个保留候选，补充实际返回2个新候选，但套用primary三键外壳，触发recovery_top_level_shape，未进入逐候选验证。不是主动返回空候选，也没有新出题结果。',
        '清理补充请求中两种JSON顶层合同及冲突人物示例；不能用反馈伪造通过、推测缺失状态或增加未授权调用。')]


def main():
    if BUNDLE.exists():
        raise FileExistsError('delivery_inputs_already_materialized')
    frozen = read(RUN / 'freeze_manifest.json')
    old_manifest = read(OLD / 'manifest.json')
    check_freeze(frozen)
    check_freeze(old_manifest)
    # Only this run and its existing feedback inputs are protected/inspected.
    source_files = {p for p in RUN.rglob('*') if p.is_file()}
    source_files.update(p for p in OLD.rglob('*') if p.is_file())
    before_hashes = {p: digest(p) for p in source_files}
    review = read(DEST / 'reports/ORDERING_FILTER_DECISIONS.json')
    decisions = {d['candidate_id']: d for d in review['decisions']}
    assert len(decisions) == review['candidate_count'] == 5
    filtered_summary, qa_cards, all_order_ids = [], [], set()
    old_rows = {r['root_id']: r for r in old_manifest['roots']}
    bundle_rows = []
    for root in ROOTS:
        qa_root = RUN / 'qa/roots' / root
        feedback_target = DEST / 'feedback' / root
        findings = authored_feedback(root)
        pool = read(qa_root / 'candidate_pool.json') if (qa_root / 'candidate_pool.json').exists() else []
        source = read(qa_root / 'qa_source.json') if pool else None
        turns = {t['turn_ref']: t for t in source['turns']} if source else {}
        pool_ids = {q['candidate_id'] for q in pool}
        for f in findings:
            assert set(f['candidate_ids']).issubset(pool_ids)
            f['source_evidence'] = [{'turn_ref': ref, 'speaker': turns[ref]['speaker'],
                                     'text': turns[ref]['text']} for ref in f['evidence_refs']]
        exclusions, holds, keeps = [], [], []
        if pool:
            for q in pool:
                if q['question_type'] != 'ordering':
                    continue
                all_order_ids.add(q['candidate_id'])
                d = decisions[q['candidate_id']]
                assert canonical_hash(q) == d['candidate_json_sha256']
                assert digest(d['candidate_pool_path']) == d['candidate_pool_sha256']
                assert digest(d['source_path']) == d['source_sha256']
                assert q['gold'] == d['program_gold_unchanged']
                assert q['items'] == d['items_unchanged']
                raw = d['reviewed_raw']
                assert digest(raw['response_path']) == raw['response_file_sha256']
                assert digest(raw['derivation_path']) == raw['derivation_file_sha256']
                {'PASS': keeps, 'FAIL': exclusions, 'HOLD': holds}[d['decision']].append(q)
            target = DEST / 'filtered' / root
            kept_ids = {q['candidate_id'] for q in keeps}
            filtered = [q for q in pool if q['question_type'] != 'ordering' or q['candidate_id'] in kept_ids]
            assert [q for q in pool if q['question_type'] != 'ordering'] == [q for q in filtered if q['question_type'] != 'ordering']
            for q in keeps:
                rel = q['ordering_derivation']['path']
                dest_path = target / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(qa_root / rel, dest_path)
                assert digest(dest_path) == q['ordering_derivation']['sha256']
            for name, rows in [('candidate_pool.json', filtered), ('ordering_keep.json', keeps),
                               ('ordering_excluded.json', exclusions), ('ordering_hold.json', holds)]:
                write(target / name, rows)
            item = dict(root_id=root, original_total=len(pool), original_ordering=len(keeps)+len(exclusions)+len(holds),
                        ordering_keep=len(keeps), ordering_exclude=len(exclusions), ordering_hold=len(holds),
                        sc_ms_unchanged=len(pool)-len(keeps)-len(exclusions)-len(holds), filtered_total=len(filtered),
                        filtered_pool=identity(target / 'candidate_pool.json'), original_pool=identity(qa_root / 'candidate_pool.json'),
                        source=identity(qa_root / 'qa_source.json'),
                        certification_scope='ORDERING_ONLY_FILTERED_SC_MS_NOT_CERTIFIED',
                        mutation='NO_QUESTION_GOLD_CITATION_OR_METADATA_REWRITE',
                        decisions=[d for d in review['decisions'] if d['root_id']==root])
            write(target / 'FILTER_MANIFEST.json', item)
            filtered_summary.append({k:v for k,v in item.items() if k!='decisions'})
        provenance_paths = [RUN / 'frozen' / root / 'inputs.json', RUN / 'expansion/roots' / root / 'outcome.json']
        if pool:
            provenance_paths += [qa_root / n for n in ('candidate_pool.json','candidate_generation.json','qa_source.json','qa_transcript.txt')]
        else:
            provenance_paths += [RUN / 'expansion/roots' / root / 'planner' / n for n in ('validation.json','supplement_output.json','supplement_validation.json')]
        card = dict(version='codex_post_generation_feedback_v1', root_id=root, author='Codex',
                    authority=AUTHORITY, source_run=str(RUN), qa_status='GENERATED' if pool else 'NOT_REACHED',
                    scope='LOCAL_REVIEW_OF_EXISTING_OUTPUTS_NO_SOLVER_OBSERVATIONS',
                    qa_fact_scope='P_PLUS_R01_PLUS_THIS_RUN_R02' if pool else 'NO_QA_SOURCE',
                    planner_replay_fact_scope='P_PLUS_R01_ONLY_R02_REVIEW_IS_NOT_ADDITIONAL_HISTORY',
                    applied_to_existing_writer_repair=False, solver_accuracy='UNKNOWN_NOT_RUN',
                    root_program_accepted_counts=dict(Counter(q['question_type'] for q in pool)),
                    ordering_filter_counts={'keep':len(keeps),'exclude':len(exclusions),'hold':len(holds)},
                    findings=findings, provenance=[identity(p) for p in provenance_paths], external_calls=0)
        write(feedback_target / 'qa_feedback.json', card)
        qa_cards.append(card)

        original = read(OLD / root / 'inputs.json')
        # Refresh a loader-compatible record, keeping original source fact checks.
        record = deepcopy(original['feedback_record'])
        record['version'] = 'codex_source_bound_planner_feedback_v2'
        record['source_cluster_id'] = root + ':P_R01:codex_feedback_v2'
        record['source_feedback_ids'] += [root + ':CODEX_QA_REVIEW_V1']
        record['canonical_observation']['basis'] = 'Codex source-bound P+R01 checks plus observations from one frozen downstream attempt; no Solver run and no certified canonical Gold.'
        perf = record['performance']
        perf['post_generation_review'] = {
            'status': card['qa_status'], 'scope': 'REVIEW_OBSERVATION_NOT_ADDITIONAL_PARENT_HISTORY',
            'source_run': RUN.name, 'program_accepted_counts': card['root_program_accepted_counts'],
            'ordering_filter_counts': card['ordering_filter_counts'],
            'observations': [{'finding_id': f['finding_id'], 'status': f['status'],
                              'observation': f['explanation']} for f in findings],
            'source_feedback_artifact': identity(feedback_target / 'qa_feedback.json'),
            'interpretation': 'Past downstream output and review, not authoritative events at this P+R01 replay boundary. Do not copy R02 facts or require its scenes to reappear. Writer selection failure does not prove Planner added no information.',
        }
        perf['solver_behavior_observations'] = 'NOT_PROVIDED_SOLVER_NOT_RUN'
        perf['renderer_legacy_field_notice'] = 'The outer field solver_observations is a legacy renderer name. This payload is Codex offline feedback, never measured Solver performance.'
        perf['optional_planning_guidance'] += [
            'Distinguish known source facts, planned increments, realized outcomes, and what Writer actually chose. An unsupported QA temporal edge is not a demand to invent a stronger chronology in the history.',
            'Candidate field/envelope failures are interface defects, not evidence of missing developments in the source. Keep the qualified-scene minimum and preserve unknown facts.',
        ]
        provenance = dict(review_author='Codex', review_type=AUTHORITY,
                          previous_feedback=identity(OLD / root / 'feedback_input.json'),
                          post_generation_feedback=identity(feedback_target / 'qa_feedback.json'),
                          input_fact_scope='FROZEN_P_PLUS_R01_UNCHANGED',
                          downstream_R02_is_review_only=True, external_calls=0,
                          retained_v1_fact_checks=True)
        target = BUNDLE / root
        write(target / 'feedback_input.json', {'record': record, 'offline_review_provenance': provenance})
        loaded, dialogue = _load_feedback_context(original['parent'], feedback_input=target / 'feedback_input.json')
        assert loaded == record and dialogue == original['feedback_dialogue']
        new = deepcopy(original)
        new['feedback_record'] = record
        new['provenance']['feedback_provenance'] = provenance
        new['provenance']['sources']['feedback'] = identity(target / 'feedback_input.json')
        assert all(new[k] == original[k] for k in ('parent','prior_expanded','snapshot','feedback_dialogue','conversation_date_segments'))
        write(target / 'inputs.json', new)
        for name in ('history_dated.txt','history_plain.txt','resolved_evidence.json'):
            shutil.copyfile(OLD / root / name, target / name)
        row = deepcopy(old_rows[root])
        row.update(input_path=str(target / 'inputs.json'), source_feedback_path=str(target / 'feedback_input.json'), feedback_review=provenance)
        bundle_rows.append(row)
    assert all_order_ids == set(decisions)
    manifest = dict(version='codex_source_bound_planner_feedback_inputs_v2',
                    status='OFFLINE_FEEDBACK_READY_NOT_EXECUTED', root_order=ROOTS, roots=bundle_rows,
                    source_run=str(RUN), external_calls=0,
                    files=[identity(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()])
    write(BUNDLE / 'manifest.json', manifest)
    check_freeze(manifest)

    # Exercise the real route up to (and excluding) the first model call. Use
    # the already frozen request metadata; no credentials or provider discovery.
    class PreviewBoundary(Exception):
        pass
    preview_checks = []
    with patch('socket.socket.connect', side_effect=AssertionError('NETWORK_FORBIDDEN')) as net, \
         patch('socket.create_connection', side_effect=AssertionError('NETWORK_FORBIDDEN')) as conn, \
         patch.object(entry.shared, 'resolve_credential', side_effect=AssertionError('CREDENTIAL_READ_FORBIDDEN')) as creds, \
         patch.object(entry.shared, 'ProfileClient', side_effect=AssertionError('TRANSPORT_FORBIDDEN')) as transport, \
         patch.object(entry.shared, 'QARuntime', side_effect=AssertionError('QA_RUNTIME_FORBIDDEN')) as qa:
        tokenizer = entry.shared.tokenizer_from(frozen['tokenizer_path'])
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
        profile = entry.spec_for('v2','scoped-v1','keyed-state-v1').execution_profile
        for root in ROOTS:
            holder = {}
            def boundary(stage, prompt, maximum):
                assert stage == 'planner' and maximum == entry.shared.STAGE_BUDGETS['planner']
                holder.update(deepcopy(prompt))
                raise PreviewBoundary()
            try:
                entry.shared.run_round(read(BUNDLE / root / 'inputs.json'), boundary,
                    entry.shared._sink(DEST / 'preview' / root), counter, execution_profile=profile)
            except PreviewBoundary:
                pass
            assert holder
            old_prompt = read(RUN / 'frozen' / root / 'planner_prompt.json')
            old_view = json.dumps(old_prompt['feedback_view'], ensure_ascii=False, sort_keys=True, indent=2)
            new_view = json.dumps(holder['feedback_view'], ensure_ascii=False, sort_keys=True, indent=2)
            assert old_prompt['prompt'].count(old_view) == holder['prompt'].count(new_view) == 1
            assert old_prompt['prompt'].replace(old_view,new_view) == holder['prompt']
            request = read(RUN / 'frozen' / root / 'planner_request.json')
            old_messages = deepcopy(request['messages'])
            request['messages'] = entry.shared.messages(holder)
            assert [x['role'] for x in request['messages']] == [x['role'] for x in old_messages]
            write(DEST / 'preview' / root / 'planner_prompt.json', holder)
            write(DEST / 'preview' / root / 'planner_request.json', request)
            preview_checks.append(dict(root_id=root, feedback_only_prompt_change=True,
                loader_compatible=True, inherited_history_memory_dates_unchanged=True,
                planner_prompt_tokens_local_tokenizer=counter(holder['prompt']),
                token_count_is_service_guarantee=False, requests_sent=0))
        for mocked in (net,conn,creds,transport,qa):
            mocked.assert_not_called()
    assert all(digest(p)==sha for p,sha in before_hashes.items())
    check_freeze(frozen)
    check_freeze(old_manifest)
    report = dict(status='PASS_OFFLINE_DELIVERY', created_at=datetime.now(timezone.utc).isoformat(),
                  review_authority=AUTHORITY, external_calls=0, credential_reads=0,
                  source_run_freeze_entries_checked=len(frozen['files']),
                  source_run_and_feedback_files_unchanged=len(before_hashes),
                  ordering_counts={'program_accepted':5,'kept':1,'excluded':3,'held':1},
                  filtered_pools=filtered_summary, feedback_root_count=len(qa_cards),
                  planner_preview_checks=preview_checks,
                  source_code_changes='NONE_CONTRACT_REPAIR_RECOMMENDATIONS_ONLY',
                  application='Local filtered pool files and loader-compatible Planner feedback inputs. Existing frozen run and Writer prompts are not retroactively changed.',
                  no_new_generation=True)
    write(DEST / 'VERIFICATION.json', report)
    write(DEST / 'FEEDBACK_INDEX.json', {
        'version':'codex_qa_feedback_and_ordering_filter_v1','source_run':str(RUN),
        'feedback_bundle_manifest':identity(BUNDLE / 'manifest.json'),
        'review_decisions':identity(DEST / 'reports/ORDERING_FILTER_DECISIONS.json'),
        'filtered_roots':filtered_summary,
        'feedback_files':[identity(DEST / 'feedback' / r / 'qa_feedback.json') for r in ROOTS],
        'usage':'Use filtered/<root>/candidate_pool.json for this ordering-filtered view; SC/MS remain unchanged and are not certified. Use inputs/manifest.json for the next Planner preparation; preview is offline only.',
        'writer_feedback_injection':'NOT_IMPLEMENTED_EXISTING_FRESH_WRITER_ROUTE_UNCHANGED',
        'external_calls':0})
    write(DEST / 'manifest.json', {'version':'codex_offline_feedback_filter_delivery_v1',
        'files':[identity(p) for p in sorted(DEST.rglob('*')) if p.is_file() and p != DEST / 'manifest.json']})
    print(json.dumps({k:report[k] for k in ('status','external_calls','ordering_counts','source_run_and_feedback_files_unchanged')},ensure_ascii=False))


if __name__ == '__main__':
    main()
