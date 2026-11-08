#!/usr/bin/env python3
"""Additive independent source-only final semantic review.

All semantic decisions below were authored by executor /root/final_blind_review
after a complete source-first reading. This script only assembles exact citations,
enumerates authored candidate anchors, and checks structural consistency. It does
not independently certify semantic correctness.
"""
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_retry_qa_pilot_v119')
EXECUTOR = '/root/final_blind_review'
SCENES = [f'R02:{r}:S{s:02d}' for r in ('A', 'B') for s in range(1, 4)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Review:
    def __init__(self, rid):
        self.rid = rid
        self.unit = BASE / 'units' / ('root' + rid)
        self.folder = self.unit / 'roots' / ('root' + rid)
        self.source_path = self.folder / 'CUMULATIVE_SOURCE.json'
        self.constraint_path = self.unit / 'inputs' / 'CONTINUATION_CONSTRAINTS.json'
        self.public_path = self.folder / 'PUBLIC_QUESTIONS.json'
        # Allowlisted inputs only; source and constraints precede public questions.
        self.source = json.loads(self.source_path.read_text())
        self.constraints = json.loads(self.constraint_path.read_text())
        self.public = json.loads(self.public_path.read_text())
        self.source_sha = sha(self.source_path)
        self.turns = {t['turn_ref']: (i, t) for i, t in enumerate(self.source['turns'])}
        # Explicitly selected semantic primary citations for every multi-ref option.
        # A single-ref option has only one possible primary citation.
        self.multiref_primary = {
          '19708': {
            'Emily changes to the longer path toward the park.': 'R02:A:S03:T011',
            'The corrected booking is for three people at £70.80 with reference 5VJX8JWT.': 'R02:B:S03:T015',
            'The corrected booking is for three people at £23.60 with reference U4G7GGGH.': 'R02:B:S03:T015',
            'Emily moves the plant off the small table by the door.': 'R02:A:S01:T005',
            'Emily leaves the plant on the small table by the door.': 'R02:A:S01:T005',
          },
          '94289': {
            'Emily says the scholarship result is still unknown.': 'R02:A:S02:T013',
            'The first restaurant is Bedouin and has no availability.': 'R02:B:S02:T003',
            'Bedouin confirms an available table.': 'R02:B:S02:T003',
          },
          '103866': {
            'James requests two people for four nights starting on Sunday.': 'R02:B:S02:T005',
            'James requests two people for three nights starting on Sunday.': 'R02:B:S02:T005',
            'Train TR3732 leaves Cambridge at 09:01.': 'R02:B:S03:T003',
            'Train TR3732 leaves Cambridge at 09:10.': 'R02:B:S03:T003',
            'Emily has sent the follow-up message.': 'R02:A:S01:T007',
            'Emily leaves the follow-up as an unsent draft.': 'R02:A:S01:T007',
            'James finds the tent pegs in the small side pocket.': 'R02:A:S02:T005',
          },
        }[rid]
        self.records = []

    def cite(self, ref, phrase=None, note=None):
        i, t = self.turns[ref]
        assert t['scene_ref'] in SCENES
        start = t['text'].find(phrase) if phrase else 0
        assert start >= 0, (ref, phrase)
        out = dict(source_path=str(self.source_path), source_sha256=self.source_sha,
                   turn_ref=ref, scene_ref=t['scene_ref'], speaker=t['speaker'],
                   quote=t['text'], cumulative_turn_index_0based=i,
                   source_json_pointer=f'/turns/{i}/text', clause_start_character=start)
        if phrase:
            out['anchor_phrase'] = phrase
        if note:
            out['matching_note'] = note
        return out

    def ms(self, decisions):
        """Exact option text -> (supports/contradicts/unknown, refs, reason)."""
        for q in self.public['questions']:
            if q['question_type'] != 'multi_select':
                continue
            opts = {}
            for label, text in q['options'].items():
                judgment, refs, reason = decisions[text]
                primary_ref = refs[0] if len(refs) == 1 else self.multiref_primary[text]
                assert primary_ref in refs
                opts[label] = dict(option_text=text, judgment=judgment,
                                   supported=judgment == 'supports',
                                   fact_distinct=True, source_refs=[self.cite(r) for r in refs],
                                   primary_citation=self.cite(primary_ref),
                                   primary_citation_selection='Explicit semantic selection by /root/final_blind_review' if len(refs) > 1 else 'Only citation',
                                   support_refs=[self.cite(r) for r in refs] if judgment == 'supports' else [],
                                   contradiction_refs=[self.cite(r) for r in refs] if judgment == 'contradicts' else [],
                                   reason=reason)
            labels = [k for k, v in opts.items() if v['judgment'] == 'supports']
            self.records.append(dict(candidate_id=q['candidate_id'], question_type='multi_select',
                question=q['question'], stem_valid=True, scope_valid=True,
                no_entangled_options=True, fact_distinct=True, coherent=True,
                neutral_stem_ok=True, surface_leakage_detected=False, leakage_free=True,
                surface_leakage_reason='题干和各选项没有公开答案标签、source 位置或支持状态标记；均由六场原文独立判定。',
                option_reviews=opts, derived_answer=labels, derived_supported_labels=labels,
                reason='六项分别涉及不同场景中的独立事实；每项完整陈述均在限定六场内判定，未以选项数量或既有答案推断。'))

    def item(self, text, ref, phrase=None, alternatives=(), support=(), related=(),
             supported=True, actor_ok=True, modality_ok=True, state_ok=True,
             single_event_ok=True, reason=''):
        alt = [self.cite(*x) if isinstance(x, tuple) else self.cite(x) for x in alternatives]
        anchor = self.cite(ref, phrase)
        refs = list(dict.fromkeys([ref] + list(support)))
        return dict(item_text=text, supported=supported, actor_ok=actor_ok,
                    modality_ok=modality_ok, state_ok=state_ok, neutral_stem_ok=True,
                    single_event_ok=single_event_ok,
                    support_refs=[self.cite(r) for r in refs], order_anchor=anchor,
                    alternative_matching_anchors=alt,
                    related_nonmatching_anchors=[dict(anchor=self.cite(r), reason=why) for r, why in related],
                    all_matching_locations_searched=True, matching_search_scope=SCENES,
                    reason=reason)

    def ordering(self, suffix, items, reason, coherent=True, extra=None):
        q = next(q for q in self.public['questions'] if q['candidate_id'].endswith(suffix))
        assert q['items'] == {k: v['item_text'] for k, v in items.items()}
        choices = [[(key, a) for a in [val['order_anchor']] + val['alternative_matching_anchors']]
                   for key, val in items.items()]
        orders = {}
        for combo in itertools.product(*choices):
            ordered = sorted(combo, key=lambda x: (x[1]['cumulative_turn_index_0based'],
                                                   x[1]['clause_start_character']))
            positions = [(a['cumulative_turn_index_0based'], a['clause_start_character']) for _, a in ordered]
            assert len(set(positions)) == len(positions), ('unresolved equal anchors', suffix)
            labels = tuple(k for k, _ in ordered)
            orders.setdefault(labels, {k: a for k, a in combo})
        primary = sorted(items, key=lambda k: (items[k]['order_anchor']['cumulative_turn_index_0based'],
                                               items[k]['order_anchor']['clause_start_character']))
        alternatives = [list(o) for o in orders if list(o) != primary]
        valid = all(all(v[f] for f in ('supported', 'actor_ok', 'modality_ok', 'state_ok',
                                       'neutral_stem_ok', 'single_event_ok')) for v in items.values())
        record = dict(candidate_id=q['candidate_id'], question_type='ordering',
            chronology_kind='conversation_order', question=q['question'],
            coherent=coherent, neutral_stem_ok=True, stem_valid=True, scope_valid=True,
            items_valid=valid, unique_order=len(orders) == 1,
            derived_order=primary, alternative_orders=alternatives,
            items=items,
            alternative_matching_anchors={k: v['alternative_matching_anchors'] for k, v in items.items()},
            order_derivation='按累计 source 的 turns 数组位置排序；同一 turn 内按引文所指分句的字符起点排序。未把现实事件时间、标签编号或 items 展示顺序当作答案依据。',
            anchor_combination_count_checked=sum(1 for _ in itertools.product(*choices)),
            distinct_orders_count=len(orders),
            alternative_order_witnesses=[dict(derived_order=list(o), anchors=anchors) for o, anchors in orders.items() if list(o) != primary],
            surface_leakage_detected=False, leakage_free=True,
            surface_leakage_reason='公开 item 未标注对应 scene、turn 或 source offset，也没有先后序号词泄露源顺序。I1..I6 是 item 标签。展示顺序与一个派生顺序恰好一致另行记录，未据其作答。',
            displayed_item_order=list(q['items']),
            display_order_matches_primary_derived_order=list(q['items']) == primary,
            reason=reason)
        if extra:
            record.update(extra)
        self.records.append(record)

    def write(self):
        assert len(self.records) == len(self.public['questions'])
        assert {q['candidate_id'] for q in self.records} == {q['candidate_id'] for q in self.public['questions']}
        binding_ok = all(q['source_path'] == str(self.source_path) and q['source_sha256'] == self.source_sha for q in self.public['questions'])
        assert binding_ok
        invalid = [q['candidate_id'] for q in self.records
                   if not q['stem_valid'] or not q['scope_valid'] or
                   (q['question_type'] == 'ordering' and (not q['items_valid'] or not q['unique_order']))]
        out = dict(schema_version='final_blind_source_review_v1',
            root_id=self.source['root_id'], executor=EXECUTOR,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            semantic_correctness_certified_by_program=False,
            review_method='独立 source-first 最终公开题语义审核；人工逐项判断，脚本仅固化引文与枚举锚点组合。',
            input_bindings=[dict(role=role, path=str(p), sha256=sha(p)) for role, p in
                [('cumulative_source', self.source_path), ('continuation_constraints', self.constraint_path), ('public_questions', self.public_path)]],
            public_source_bindings_match=True,
            read_attestation=dict(executor=EXECUTOR,
                source_first=True, full_source_read_completed=True,
                full_continuation_constraints_read_completed=True,
                source_turn_count=len(self.source['turns']),
                all_source_and_constraints_completed_before_any_public_question_read=True,
                source_read_method='完整读取累计 source；19708 最初工具输出截断后补读遗漏的 turns，直到全体 turns 已读。94289 与 103866 分段完整读取。',
                allowed_input_content_reads=[str(self.source_path), str(self.constraint_path), str(self.public_path)],
                prohibited_input_content_reads=[],
                read_qa_draft=False, read_private=False, read_asset_source_judgments=False,
                read_composition=False, read_solver=False, read_other_reviews_or_answers=False,
                referenced_paths_in_constraints_followed=False,
                no_existing_answers_consulted=True),
            scope_scene_refs=SCENES,
            questions=self.records,
            summary=dict(question_count=len(self.records),
                multi_select_count=sum(q['question_type'] == 'multi_select' for q in self.records),
                ordering_count=sum(q['question_type'] == 'ordering' for q in self.records),
                all_multi_select_options_independently_judged=True,
                invalid_or_nonunique_candidates=invalid,
                all_candidates_valid_and_unique=not invalid),
            display_order_policy='展示 item 标签与源顺序分别记录；若二者一致，只报告核对结果，不视作支持证据。')
        dest = self.folder / 'FINAL_BLIND_SOURCE_REVIEW.json'
        with dest.open('x', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(json.dumps(dict(root_id=out['root_id'], path=str(dest), sha256=sha(dest),
                              invalid_or_nonunique_candidates=invalid), ensure_ascii=False))


def root19708():
    r = Review('19708')
    r.ms({
      'Emily keeps the walk to the short route they chose.': ('supports', ['R02:A:S03:T011'], 'Emily 明确说保持已经选定的短路线。'),
      'Emily changes to the longer path toward the park.': ('contradicts', ['R02:A:S03:T002','R02:A:S03:T003','R02:A:S03:T011'], '长路线只是 James 提供的备选；Emily 选择绕街区，并明确保持短路线。'),
      'Emily puts tomorrow afternoon in her calendar for the visit.': ('supports', ['R02:A:S02:T011'], '明确已把明天下午的探望记入日历。'),
      'Emily leaves tomorrow afternoon unscheduled.': ('contradicts', ['R02:A:S02:T011'], '已记入日历，反驳未安排。'),
      'The corrected booking is for three people at £70.80 with reference 5VJX8JWT.': ('supports', ['R02:B:S03:T015','R02:B:S03:T016'], '最终更正为三人、£70.80、5VJX8JWT，且说明最初总价不是三人确认。'),
      'The corrected booking is for three people at £23.60 with reference U4G7GGGH.': ('contradicts', ['R02:B:S03:T011','R02:B:S03:T015','R02:B:S03:T016'], '£23.60/U4G7GGGH 属于最初错误范围的确认；更正三人确认是 £70.80/5VJX8JWT。'),
      'Marco reports that the taxi is a white Volkswagen.': ('supports', ['R02:B:S01:T009'], 'Marco 直接报告 white Volkswagen。'),
      'Marco reports that the taxi is a black Volkswagen.': ('contradicts', ['R02:B:S01:T009'], '车色为 white，而不是 black。'),
      'The service recommends Christ’s College with free entry.': ('supports', ['R02:B:S02:T009'], 'Lila 报告服务推荐 Christ’s College 且 free entry；属于对服务答复的转述。'),
      'The service says Christ’s College requires an entrance fee.': ('contradicts', ['R02:B:S02:T009'], '原文明确 free entry。'),
      'Emily moves the plant off the small table by the door.': ('supports', ['R02:A:S01:T003','R02:A:S01:T005'], 'T003 确立门边小桌；T005 的 that table 明确承接，并报告已搬走植物。'),
      'Emily leaves the plant on the small table by the door.': ('contradicts', ['R02:A:S01:T003','R02:A:S01:T005'], '植物已经从该桌搬走。'),
    })
    r.ordering('root19708-ordering1', {
      'I1': r.item('Plant moved off the table','R02:A:S01:T005',"I've moved the plant",support=['R02:A:S01:T003'],reason='已完成的搬植物行为，唯一直接提及。'),
      'I2': r.item('Visit put in calendar','R02:A:S02:T011',"I've put tomorrow afternoon",reason='唯一日历记入陈述。'),
      'I3': r.item('Keep short route','R02:A:S03:T011',"Let's keep",related=[('R02:A:S03:T003','首次选择绕街区，并非其后明确保持既定路线。')],reason='准确对应保持而非改道的陈述。'),
      'I4': r.item('White Volkswagen taxi','R02:B:S01:T009',"They've booked a white Volkswagen",reason='唯一车色车款确认。'),
      'I5': r.item('Christ’s College free entry','R02:B:S02:T009',"They recommend Christ's College",reason='唯一免费入场陈述。'),
      'I6': r.item('Corrected three-person train booking','R02:B:S03:T015',"They've updated it for three people",alternatives=[('R02:B:S03:T016','Then the £23.60 message', '对同一更正的指代性复述：初始报价非三人确认，且 corrected train reference 不同。')],support=['R02:B:S03:T011','R02:B:S03:T016'],reason='T015 直接给出三人更正；T016 指代复述仍在最后一场，均不改变六项相对顺序。'),
    },'六项各定位到一个明确场景；更正火车确认在最后一场的重复指代不改变相对顺序，因此唯一为 I1,I2,I3,I4,I5,I6。')
    r.ordering('root19708-ordering2', {
      'I1': r.item('Tongs with dish','R02:A:S01:T008',"I'll bring serving tongs with the dish",alternatives=[('R02:A:S01:T012',"I'll keep the tongs with the dish",'同场再次说明把夹子与菜盘放一起。')],reason='两次均为将来带夹子的安排；item 没有把安排升级为已经到场。'),
      'I2': r.item('Message before setting off','R02:A:S02:T013',"I'll send a quick message",reason='为出发前发消息的承诺，未宣称已经发出。'),
      'I3': r.item('Glad she came outside','R02:A:S03:T013',"I'm glad I came outside",alternatives=[('R02:A:S03:T014',"I'm glad you came too",'James 也表示高兴 Emily 出来；第三人称 she 本身仍无来源性别依据。')],actor_ok=False,reason='“高兴 Emily 出来了”的事件核心受支持；但 item 增加 she 性别指代，六场与全 source 均未明确 Emily 的性别。约束禁止仅凭姓名补性别。累计 source 出现的 she/her 指母亲 Lila。'),
      'I4': r.item('Conversation ended','R02:B:S01:T011',"ended the conversation",alternatives=[('R02:B:S02:T015','finished the enquiry','Lila 的餐馆/学院问询也已结束。'),('R02:B:S03:T015','ended the enquiry','James 的住宿/火车问询也已结束。')],single_event_ok=False,reason='未限定说话者或问询主题，可匹配三个不同场景的结束陈述；其中两处可位于 I5 之后。'),
      'I5': r.item('No table reserved','R02:B:S02:T015',"I still haven't reserved a table",alternatives=[('R02:B:S02:T003','but I said no','拒绝服务提出的订桌，同一餐馆问询的未订桌状态已成立。')],reason='T015 明确仍未订桌；T003 拒绝订桌也可表达该状态。T015 中结束问询分句在未订桌分句之前。'),
      'I6': r.item('Train and guesthouse references distinct','R02:B:S03:T016',"The corrected train reference",reason='唯一直接比较两种 reference 的陈述。'),
    },'不唯一：I4 泛称结束谈话可指 taxi、restaurant/college 或 guesthouse/train 三个结束节点。采用最先明确结束与 T015 未订桌时是 I1,I2,I3,I4,I5,I6；采用先前拒绝订桌或最后结束问询时，I5 可在 I4 前。另 I3 的 she 加入无依据的性别属性。',coherent=False,
       extra={'gender_attribute_check': {'item_id':'I3','unsupported_attribute':'Emily -> she','constraint_basis':'不得据姓名补定性别','all_cumulative_source_gender_pronoun_search_completed':True}})
    r.write()


def root94289():
    r = Review('94289')
    r.ms({
      'Emily writes three choices: a small cleaning task, a short walk, or journal lines.': ('supports',['R02:A:S03:T003'],'已写三项，内容逐项对应；未把选择清单当作全都已实行。'),
      'Emily writes ten choices and plans to clean the whole apartment.': ('contradicts',['R02:A:S03:T003'],'明确写三项，且不想把每次坏心情都变成清扫整套公寓。'),
      'The service offers Ashley Hotel, Cityroomz and Lovell Lodge.': ('supports',['R02:B:S01:T003'],'服务列出这三家。'),
      'The service offers only one accommodation option.': ('contradicts',['R02:B:S01:T003'],'列出三家，反驳只有一家。'),
      'The taxi is a black Honda with contact number 07903665107.': ('supports',['R02:B:S03:T003'],'车色、品牌、号码同一确认中逐一给出。'),
      'The taxi is a white Volkswagen.': ('contradicts',['R02:B:S03:T003'],'实际为 black Honda。'),
      'Emily says the scholarship is already a relief.': ('supports',['R02:A:S02:T013'],'已获得奖学金所带来的 relief 为 Emily 明确自述。'),
      'Emily says the scholarship result is still unknown.': ('contradicts',['R02:A:S02:T011','R02:A:S02:T013'],'未知的是不能臆测的奖项要求；奖学金本身已是 relief，不是结果未定。'),
      'The first restaurant is Bedouin and has no availability.': ('supports',['R02:B:S02:T001','R02:B:S02:T003'],'第一家为 Bedouin，回复无可用餐位；“无位”限于已请求日期、人数和时间。'),
      'Bedouin confirms an available table.': ('contradicts',['R02:B:S02:T003','R02:B:S02:T007','R02:B:S02:T010'],'Bedouin 无位，成功预约的是 Kymmoy。'),
      'Emily says the sweeping is done and the broom is put away.': ('supports',['R02:A:S01:T001'],'已扫完且已收起扫帚，同句明确。'),
      'Emily says the sweeping is unfinished.': ('contradicts',['R02:A:S01:T001'],'明确 sweeping is done。'),
    })
    r.ordering('root94289-ordering1', {
      'I1': r.item('Sweeping is done and broom put away','R02:A:S01:T001','The sweeping is done',related=[('R02:A:S01:T014','提到地板完成，但没有扫帚已收起这个完整复合状态。')],reason='完成扫地与收起扫帚是同一轮连续的完成状态。'),
      'I2': r.item('Scholarship relief and study plan','R02:A:S02:T013','The scholarship is already a relief',support=['R02:A:S02:T001','R02:A:S02:T005','R02:A:S02:T009'],single_event_ok=False,
          related=[('R02:A:S03:T014','只提下一步学习计划，未陈述 scholarship relief；不满足完整 item。')],
          reason='内容均受第二场支持，但 item 是“奖学金 relief”与多轮制定学习计划的合并主题，缺少单一事件锚点。主锚 T013 是 relief 陈述，完整 plan 需 T005/T009；二者不应作为一个单事件 item。'),
      'I3': r.item('Three coping choices written','R02:A:S03:T003',"I've written three",reason='完成三项清单为唯一直接陈述。'),
      'I4': r.item('Three accommodation options','R02:B:S01:T003','They offered Ashley Hotel',reason='唯一三家住宿选择的列举。'),
      'I5': r.item('Bedouin unavailable','R02:B:S02:T003','the reply says nothing is available',alternatives=[('R02:B:S02:T010',"the Bedouin request didn't go through",'James 直接复述 Bedouin 请求未成功。')],reason='未成功的回顾仍在同一场，任一匹配均不改变与其他 item 的顺序。'),
      'I6': r.item('Black Honda and contact number','R02:B:S03:T003',"They've put me in a black Honda",reason='车及号码同轮给出，后续只说复制 car/contact 到 note，未再出现具体车色车型号码。'),
    },'各 item 的源场景依次为六场，所有完整或复述匹配都保持 I1,I2,I3,I4,I5,I6，故相对顺序唯一；但 I2 合并奖学金 relief 与制定学习计划，single_event_ok=false，整题 items_valid=false。',
       extra={'composite_item_spans':[{'item_id':'I2','scene_ref':'R02:A:S02','component_refs':{'study_plan':['R02:A:S02:T001','R02:A:S02:T005','R02:A:S02:T009'],'scholarship_relief':['R02:A:S02:T013']},'single_turn_full_match':False,'all_component_spans_precede_I3':True}]})
    r.write()


def root103866():
    r = Review('103866')
    r.ms({
      'James requests two people for four nights starting on Sunday.': ('supports',['R02:B:S02:T005','R02:B:S02:T011'],'请求两人、四晚、Sunday 起始；是预约请求，不宣称实际入住。'),
      'James requests two people for three nights starting on Sunday.': ('contradicts',['R02:B:S02:T005','R02:B:S02:T011'],'原文为 four nights。'),
      'Train TR3732 leaves Cambridge at 09:01.': ('supports',['R02:B:S03:T001','R02:B:S03:T003','R02:B:S03:T007'],'路线 departure Cambridge，车次 TR3732，出发 09:01；只是列车时刻，不宣称人已旅行。'),
      'Train TR3732 leaves Cambridge at 09:10.': ('contradicts',['R02:B:S03:T001','R02:B:S03:T003'],'出发时间为 09:01 而非 09:10。'),
      'Emily has sent the follow-up message.': ('supports',['R02:A:S01:T005','R02:A:S01:T007'],'先输入后明确已发送。'),
      'Emily leaves the follow-up as an unsent draft.': ('contradicts',['R02:A:S01:T005','R02:A:S01:T007'],'T005 输入后，T007 已发送，不停留为未发草稿。'),
      'The service identifies Ali Baba as a Lebanese restaurant in the centre.': ('supports',['R02:B:S01:T003'],'地点、菜系、名称完整对应服务答复的转述。'),
      'The service identifies Ali Baba as a Turkish restaurant.': ('contradicts',['R02:B:S01:T003'],'服务描述是 Lebanese，不是 Turkish。'),
      'James finds the tent pegs in the small side pocket.': ('supports',['R02:A:S02:T003','R02:A:S02:T004','R02:A:S02:T005','R02:A:S02:T011'],'先未看见，Emily 指示小侧袋，James 找到，之后收回该侧袋。'),
      'James says the curry is warmer with spice than expected.': ('supports',['R02:A:S03:T011'],'James 已尝味并直接说 warmer with spice than expected。'),
      'James says the curry is milder with spice than expected.': ('contradicts',['R02:A:S03:T011'],'原文为 warmer，不是 milder。'),
    })
    r.ordering('root103866-ordering2', {
      'I1': r.item('Emily does not turn the delayed reply into a new worry.','R02:A:S01:T013',"I'm not going to turn that into another worry",alternatives=[('R02:A:S01:T007',"I'm putting my phone aside",'不再盯候回复的当前行动，表达同一延迟回复不需制造新忧虑的立场。'),('R02:A:S01:T009',"a quiet phone doesn't need another explanation right now",'明确表示安静的手机此刻不需要新的解释，语义重述当前不再担忧。')],reason='当前 follow-up 尚无回复，Emily 表达不把它变成新忧虑的决定。按陈述顺序理解为当前立场；不提升成以后永不担忧或医学效果。三个可能匹配均在第一场。'),
      'I2': r.item('The departure remains nine in the morning.','R02:A:S02:T013','Our departure is still nine in the morning',reason='明确保留将来露营的出发时间；未把出发当作已发生。'),
      'I3': r.item('James puts rice and curry into bowls.','R02:A:S03:T013',"I've put rice in the bowls and spooned the curry over it",related=[('R02:A:S03:T008','只是拿碗准备，没有盛饭和咖喱。')],reason='James 直接报告已盛饭与咖喱，是同一装盘动作。'),
      'I4': r.item('Emily says the enquiry exchange is closed.','R02:B:S01:T013','The exchange is closed now',related=[('R02:B:S01:T011','只是第一次说 goodbye；随后服务又问一次是否需要帮助，最终 exchange closed 在 T013 才明确。')],reason='唯一明确终结整个问询 exchange 的陈述；不将先前说再见错误当成之后没有追加轮次。'),
      'I5': r.item('James saves the guesthouse details and reference in one note.','R02:B:S02:T013','I saved Finches Bed and Breakfast',reason='James 直接报告把 guesthouse 细节与 LT8DPU71 存在同一 note。'),
      'I6': r.item('Emily says the train and attraction information does not replace the camping plan.','R02:B:S03:T013','This separate train and attraction information',related=[('R02:B:S02:T010','只把 guesthouse 预约与 camping 分开，没有 train/attraction。'),('R02:B:S02:T011','明确 guesthouse 预约不改变 camping，但不是 item 的 train/attraction 陈述。')],reason='唯一完整匹配 train 与 attraction 信息不取代露营计划；不混入上一场住宿预约。'),
    },'六项对应第一场的延迟回复立场、第二场的露营时间、第三场的盛饭、第四场的最终关闭问询、第五场的保存住宿 note、第六场的独立火车/景点信息。重复语义匹配仅在第一场内部，唯一顺序为 I1,I2,I3,I4,I5,I6。')
    r.write()


if __name__ == '__main__':
    root19708()
    root94289()
    root103866()
