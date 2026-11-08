import json, hashlib, re
from pathlib import Path
A=Path(__file__).parent;E=A.parent/'feedback_closed_loop_20261024_v1'
def read(p):return json.loads(p.read_text())
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
C={x['candidate_id']:x for x in read(E/'CANDIDATES_PRIVATE.json')}
S=read(E/'execution/SCORES.json')['records'];PR={x['request_id']:x for x in read(E/'execution/PRIVATE_SCORING.json')['requests']}
protocol='''# F组理由与选择不一致复查

复用封存的F32题64次作答；不运行Solver、不训练、不改原题、Gold或旧分析。这里检查的是可见解释，不据此推断模型隐藏思考正确。

L1：局部决定性理由正确，但该项真假判断/最终取舍错误。须理由明确给出足以否定其判断的正确计算、事实或否定，不能只因复述半句真事实就计入。
L2：该项supported/reject判断与Gold一致，理由也能支持该判断，但最终box相反取舍。与L1可重叠，按响应/语义题去重。仅有judgment-final不一致、理由本身错的，不计L2，保留机械统计。
L3：整题可见解答已正确（完整六项理由/判断及综合说明无实质矛盾，或明确给出完整正确排序），但最后box错误。引用覆盖不足单列，不自动推翻解答逻辑；错误实体/状态/计算则不允许L3。L3绝不能由一项L1推断。

排序已经取对钟点却没有按升序排序属于推理操作失败，不能计为L3；独立字段retrieval_correct_but_order_wrong保留。最终答案正确、仅机械证据合同失败单列，不能说选择错。

每root所有8题16响应都审，包括正确回答。两展示是同题重复。审核继承既有会话，已知F/G结果及部分例子，不是盲审；此为事后描述性复查，不用于证明反馈因果收益。每条L1/L2/L3阳性须保存逐字raw quote与source turn原文；阴性需简短说明。不能确定时保留unknown。
'''
(A/'PROTOCOL_ZH.md').write_text(protocol)
all_mech=[]
for s in S:
 c=C[s['strict']['candidate_id']];p=PR[s['request_id']]['presentation'];parsed=s['parsed'];mech=dict(request_id=s['request_id'],arm=s['arm'],slot_id=s['slot_id'],exact=s['strict']['exact'],valid=s['strict']['valid'])
 if c['question_type']=='multi_select':
  judgments={p['display_to_canonical'][lab]:(v.get('judgment')=='supported') for lab,v in parsed['item_evidence'].items()};gold=set(c['gold']);chosen=set(s['strict']['prediction'] or [])
  mech.update(judgment_final_mismatch_labels=[k for k in judgments if judgments[k]!=(k in chosen)],gold_correct_judgment_bad_final_labels=[k for k in judgments if judgments[k]==(k in gold) and (k in chosen)!=(k in gold)],all_six_judgments_gold_correct=all(judgments[k]==(k in gold) for k in judgments) and len(judgments)==6)
 all_mech.append(mech)
(A/'MECHANICAL_SCREEN.json').write_text(json.dumps(all_mech,ensure_ascii=False,indent=2)+'\n')
for root in ['218519','222937','225575','229686']:
 b=A/'reviews'/root;b.mkdir(parents=True,exist_ok=True);records=[]
 source=E/'sources'/('cc-train-episode-'+root+'.json')
 for c in C.values():
  if c['arm']!='F' or not c['candidate_id'].startswith('CL1-F-'+root+'-'):continue
  responses=[]
  for s in S:
   if s['strict']['candidate_id']!=c['candidate_id']:continue
   raw=read(Path(s['raw_binding']['path']));responses.append(dict(request_id=s['request_id'],presentation=PR[s['request_id']]['presentation'],score=s,raw_output=raw['raw_output'],raw_binding=s['raw_binding']))
  records.append(dict(candidate=c,responses=responses))
 packet=dict(schema_version='f_reason_selection_review_packet_v1',root_id=root,source_binding=binding(source),protocol=protocol,records=records,
  required_review_schema={'root_id':root,'records':[{'candidate_id':'...','responses':[{'request_id':'...','l1_local_correct_reason_wrong_decision':False,'l2_correct_judgment_wrong_box':False,'l3_whole_visible_solution_correct_wrong_box':False,'retrieval_correct_but_order_wrong':False,'answer_correct_contract_failure':False,'uncertain':False,'evidence':[{'type':'L1|L2|L3|RETRIEVAL|CONTRACT','canonical_labels':['A'],'display_labels':['B'],'raw_quote':'exact substring','source_checks':[{'turn_ref':'...','source_quote':'exact source substring'}],'assessment':'...'}],'rationale':'...'}]}]})
 (b/'PACKET.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
(A/'INPUT_BINDINGS.json').write_text(json.dumps([binding(E/'CANDIDATES_PRIVATE.json'),binding(E/'execution/SCORES.json'),binding(E/'execution/PRIVATE_SCORING.json'),binding(E/'FINAL_MANIFEST.json')],ensure_ascii=False,indent=2)+'\n')
for arm in ['F','G']:
 m=[x for x in all_mech if x['arm']==arm and x['slot_id'].startswith('MC')]
 print(arm,'MC responses',len(m),'any crossfield mismatch',sum(bool(x['judgment_final_mismatch_labels']) for x in m),'gold correct judgment but bad final',sum(bool(x['gold_correct_judgment_bad_final_labels']) for x in m),'all judgments correct but final wrong',sum(x['all_six_judgments_gold_correct'] and not x['exact'] for x in m))
print('prepared four F root packets')
