#!/usr/bin/env python3
"""Create review records and conservative research-only decisions; never edit inputs."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = OUT.parent / 'training_set_rebuild_20261105_FG_exploration_run01'
NOW = datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def write_new(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def final_messages(path):
    found = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            p = r.get('payload', {})
            if (r.get('type') == 'response_item' and p.get('type') == 'message'
                    and p.get('role') == 'assistant' and p.get('phase') == 'final_answer'):
                found.append('\n'.join(c.get('text', '') for c in p.get('content', [])))
    return found


def parse_table(text):
    result = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if not cells or not re.fullmatch(r'(?:I\d\d|V2-\d\d)', cells[0]):
            continue
        if '/' in cells[1] or '／' in cells[1]:
            a = [int(x) for x in re.split(r'[/／]', cells[1])]
            b = [int(x) for x in re.split(r'[/／]', cells[2])]
            pref = cells[3]
            note = cells[4] if len(cells) > 4 else ''
        else:
            a = list(map(int, cells[1:3]))
            b = list(map(int, cells[3:5]))
            pref = cells[5]
            note = ''
        assert len(a) == len(b) == 2 and all(1 <= v <= 5 for v in a + b)
        pref = {'平局': 'tie', 'A': 'A', 'B': 'B'}[pref]
        result.append({'item_code': cells[0], 'label_scores': {
            'A': {'naturalness': a[0], 'clarity': a[1]},
            'B': {'naturalness': b[0], 'clarity': b[1]}},
            'preference_label': pref, 'raw_table_note': note})
    return result


raw = read('READABILITY_REVIEW_RAW.json')
corrected = read('BLIND_REVIEW_RESULTS_V1_CORRECTED.json')
score_checks = []
for reviewer in ['A', 'B']:
    parsed = parse_table(raw[f'V1_REVIEWER_{reviewer}']['final_response'])
    assert len(parsed) == 12
    by_code = {r['item_code']: r for r in corrected['randomized_label_scores']
               if r['reviewer'] == f'REVIEWER_{reviewer}'}
    for r in parsed:
        expected = by_code[r['item_code']]
        assert r['label_scores'] == expected['label_scores']
        assert r['preference_label'] == expected['preference_label']
        score_checks.append({'reviewer': reviewer, 'item_code': r['item_code'],
                             'scores_match_raw': True, 'preference_matches_raw': True})
write_new('READABILITY_SCORE_AUDIT.json', {
    'checked_at_utc': NOW,
    'authoritative_raw_record': 'READABILITY_REVIEW_RAW.json',
    'checked_aggregation': 'BLIND_REVIEW_RESULTS_V1_CORRECTED.json',
    'row_count': len(score_checks), 'all_rows_match': True, 'rows': score_checks,
    'note': 'Initial aggregation is preserved; corrected values are verified by parsing raw reviewer tables.'})

v2map = read('V2_BLIND_MAPPING_PRIVATE.json')
v2items = read('DERIVED_CANDIDATES_V2.json')['items']
v2rows = []
all_scores = defaultdict(list)
votes = Counter()
for reviewer in ['A', 'B']:
    pack = read(f'V2_REVIEWER_{reviewer}_BLIND_PACK.json')
    parsed = parse_table(raw[f'V2_REVIEWER_{reviewer}']['final_response'])
    assert len(parsed) == 4
    for score, stimulus in zip(parsed, pack['items']):
        assert score['item_code'] == stimulus['item_code']
        matches = []
        for item in v2items:
            mapping = v2map[f'reviewer_{reviewer.lower()}'][item['candidate_id']]
            texts = {'original': item['original_question'], 'v2': item['v2_derived_question']}
            if all(stimulus[f'version_{label}'] == texts[version] for label, version in mapping.items()):
                matches.append((item['candidate_id'], mapping))
        assert len(matches) == 1
        cid, mapping = matches[0]
        preferred = 'tie' if score['preference_label'] == 'tie' else mapping[score['preference_label']]
        score.update(reviewer=f'REVIEWER_{reviewer}', candidate_id=cid,
                     label_mapping=mapping, preferred_version=preferred)
        v2rows.append(score)
        votes[preferred] += 1
        for label, version in mapping.items():
            for metric, val in score['label_scores'][label].items():
                all_scores[f'{version}_{metric}'].append(val)
write_new('BLIND_REVIEW_RESULTS_V2.json', {
    'recorded_at_utc': NOW, 'ratings_are_internal_model_opinions_not_human_ratings': True,
    'comparison': 'Original versus V2 for four items, not V1 versus V2.',
    'source': 'READABILITY_REVIEW_RAW.json', 'randomized_label_scores': v2rows,
    'overall': {**{f'{k}_mean': sum(v) / len(v) for k, v in all_scores.items()},
                'preference_votes': dict(votes)},
    'limitations': 'Exploratory rewording after V1 review; not independent confirmation or a training effect. All four V2 versions remain unused.'})

blind_sessions = {
    'A': Path('/root/.codex/sessions/2026/11/06/rollout-2026-11-06T00-16-56-01a1ac7f-c122-7463-a4f9-80ef2262d7db.jsonl'),
    'B': Path('/root/.codex/sessions/2026/11/06/rollout-2026-11-06T00-17-07-01a1ac7f-edae-7360-90ab-149cdb246b0c.jsonl'),
}
blind_raw = {}
for key, path in blind_sessions.items():
    msgs = final_messages(path)
    assert msgs
    blind_raw[key] = {'session_path': str(path), 'final_response': msgs[-1]}
write_new('BLIND_SOURCE_RECONSTRUCTION_RAW.json', blind_raw)

semantic_session = Path('/root/.codex/sessions/2026/11/05/rollout-2026-11-05T23-36-02-01a1ac5a-5123-78f0-9681-3bd2731a7e9b.jsonl')
msgs = final_messages(semantic_session)
assert '11 条为 **PASS**' in msgs[-1] and '只读语义审查' in msgs[-2]
write_new('SOURCE_PRESERVATION_REVIEW_RAW.json', {
    'session_path': str(semantic_session),
    'responses': msgs[-2:],
    'note': 'Verbatim final summaries only. Earlier 11 PASS / 1 HOLD covers wording preservation; new Gold-blind reconstruction below identifies an additional ambiguity in root186046.'})

answers = {
    103866: ('A', 'I04', ['milk_added', 'timer_rings', 'carrot_firm', 'carrot_tender', 'james_tastes', 'curry_served']),
    130221: ('A', 'I05', ['plastic_aim', 'separate_intro', 'young_athletes']),
    162896: ('A', 'I10', ['food_water', 'blanket', 'lights_batteries']),
    186046: ('A', 'I09', ['liam_water', 'eden_map', 'liam_snacks']),
    191749: ('B', 'I11', ['opening', 'discussion', 'second_reading']),
    225575: ('B', 'I03', ['first', 'request', 'early_thursday', 'final']),
    45849: ('B', 'I07', ['timer_start', 'cars_tray', 'tower_photo', 'sofa_check', 'book_shelf']),
    75241: ('B', 'I02', ['park_nov3', 'park_nov5', 'band_nov6']),
}
bound_refs = {
    103866: ('R02:A:S03', [1, 2, 3, 5, 7, 9, 11, 13]),
    130221: ('R02:A:S03', [1, 5, 7, 9, 13]),
    162896: ('R02:A:S03', [1, 2, 3, 4, 5, 6, 9, 10]),
    186046: ('R02:A:S02', [3, 5, 6, 7, 8, 12]),
    191749: ('R02:A:S03', [4, 7, 12]),
    225575: ('R03:B:S03', [1, 3, 4, 5]),
    45849: ('R02:A:S01', [1, 3, 4, 5, 7, 8, 9, 10, 11]),
    75241: ('R03:A:S02', [1, 2, 5, 6, 7, 8]),
}
v1items = read('DERIVED_CANDIDATES.json')['items']
by_root = {i['root_id']: i for i in v1items}
recon = []
for root, (reviewer, code, answer) in answers.items():
    item = by_root[root]
    gold = item['original_candidate']['gold_option_ids']
    same = answer == gold if item['question_type'] == 'ordering' else set(answer) == set(gold)
    assert same
    source = json.loads(Path(item['source_path']).read_text(encoding='utf-8'))
    prefix, nums = bound_refs[root]
    evidence = []
    for n in nums:
        ref = f'{prefix}:T{n:03d}'
        turns = [t for t in source['turns'] if t['turn_ref'] == ref or t['turn_ref'].endswith(':' + ref)]
        assert len(turns) == 1, (root, ref)
        turn = turns[0]
        evidence.append({'turn_ref': ref, 'exact_source_turn_ref': turn['turn_ref'],
                         'quote': turn['text'], 'speaker': turn['speaker'], 'origin': turn['origin']})
    record = {'candidate_id': item['candidate_id'], 'root_id': root, 'reviewer': reviewer,
              'item_code_in_reviewer_a_pack': code,
              'raw_response_file': 'BLIND_SOURCE_RECONSTRUCTION_RAW.json',
              'source_path': item['source_path'], 'source_sha256': item['source_sha256'],
              'original_answer_ids': answer, 'v1_answer_ids': answer,
              'both_answers_match_frozen_gold': True, 'evidence': evidence,
              'interpretation_ambiguity': None}
    if root == 186046:
        record['v1_answer_ids'] = None
        record['v1_alternative_answers'] = [answer, ['liam_water', 'liam_snacks']]
        record['both_answers_match_frozen_gold'] = False
        record['interpretation_ambiguity'] = 'Completed packing actions versus final location: map was retrieved; return is future tense only. V1 is held.'
    if root == 45849:
        record['soft_note'] = 'Sofa check is supported by the immediate answer to a request, not an explicit I checked; brick scopes are slightly unclear. Preserved in raw response; no original quality approval.'
    recon.append(record)
write_new('BLIND_SOURCE_RECONSTRUCTION.json', {
    'recorded_at_utc': NOW, 'item_count': 8, 'blind_context_count': 2,
    'design': 'One fresh source-reading context per item, four items each; A/B question/options only plus full SOURCE. No Gold/mapping/original candidate read before answers.',
    'not_two_independent_source_answers_per_item': True,
    'original_answer_matches': 8, 'v1_unambiguous_answer_matches': 7,
    'items': recon,
    'limitations': 'Internal source reconstruction only. Not external human review, comprehensive source eligibility/quality review, semantic dedup, Solver testing, or training admission.'})

accept_roots = {130221, 162896, 191749, 225575, 45849, 75241}
reasons = {
    103866: '保义盲答一致，但V1清晰度均值4.5→4.0；原题明确的已发生限定更清楚，本轮保留原题。',
    104814: 'V1评审偏好分歧，时间锚点需核实；V2的进展更新可能指未来汇报，取得观察改为记录观察。',
    109164: 'V1最近一次调整不如原截止点明确，清晰度4.5→4.0；V2新增周二改周三答案线索且省略明确多选指令。',
    136202: 'V1组图顺序变发送顺序，保义复核HOLD；V2仍改变最初已收到照片的任务范围，不采用。',
    186046: '来源盲答揭示V1将过去装包动作改为结束时状态，地图放回只用将来时，不能断言最终归位。',
    229686: 'V1两名可读性评审均偏原题；V2这次讨论指代不清，可能将截止点扩到后续检查。',
    130221: 'T013明确保存三个方向；来源盲答和Gold一致，V1自然度与清晰度均改善。',
    162896: '来源盲答和Gold一致，保留实物已带来与待检查清单的区别；V1清晰度未下降。',
    191749: '来源盲答和Gold一致，保留修改后草案状态；去除内部编号，V1清晰度未下降。',
    225575: '来源盲答和Gold一致，路线与时间约束仍可定位；V1清晰度未下降。',
    45849: '来源盲答和Gold一致，保留下午清理范围，包括图画书；原有动作推断软问题继续保留。',
    75241: '来源盲答和Gold一致，保留双方各自已查且可行及calendar-only条件。',
}
scores = {i['candidate_id']: i for i in corrected['by_candidate']}
decisions = []
research_items = []
for item in v1items:
    root = item['root_id']
    accepted = root in accept_roots
    score = scores[item['candidate_id']]
    if accepted:
        assert score['preference_votes']['revised'] == 2
        assert score['revised_clarity_mean'] >= score['original_clarity_mean']
    decisions.append({'candidate_id': item['candidate_id'], 'root_id': root,
                      'selected_version': 'v1' if accepted else 'original',
                      'decision': 'RETAIN_AS_RESEARCH_WORDING_CANDIDATE' if accepted else 'KEEP_ORIGINAL_WORDING',
                      'reason_zh': reasons[root], 'v1_readability': score,
                      'candidate_quality_status': 'UNREVIEWED', 'training_admitted': False})
    research_items.append({'candidate_id': item['candidate_id'], 'root_id': root,
                           'arm': item['arm'], 'selected_version': 'v1' if accepted else 'original',
                           'candidate': item['derived_candidate'] if accepted else item['original_candidate']})
write_new('FINAL_WORDING_DECISIONS.json', {
    'recorded_at_utc': NOW, 'retained_v1_count': 6, 'kept_original_count': 6,
    'retained_v2_count': 0,
    'selection_rule': 'Conservative post-review editorial rule: both V1 readers prefer revision, mean clarity does not decrease, source reconstruction preserves the answer without new material ambiguity. Not a preregistered success threshold.',
    'items': decisions,
    'status': 'RESEARCH_ONLY_NO_TRAINING_ADMISSION',
    'parent_candidate_quality_status_unchanged': 'UNREVIEWED'})
write_new('FINAL_RESEARCH_CANDIDATES.json', {
    'status': 'RESEARCH_COMPARISON_ONLY_NOT_TRAINING_DATA',
    'original_run_unchanged': True, 'items': research_items})

print(json.dumps({'raw_score_rows_verified': len(score_checks),
                  'v2_means': {k: sum(v) / len(v) for k, v in all_scores.items()},
                  'v2_votes': dict(votes), 'retained_v1': len(accept_roots),
                  'kept_original': len(v1items) - len(accept_roots)}, ensure_ascii=False))
