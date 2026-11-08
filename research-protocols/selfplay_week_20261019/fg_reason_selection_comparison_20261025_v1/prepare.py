import json,hashlib
from pathlib import Path
A=Path(__file__).parent;W=A.parent
F=W/'f_reason_selection_analysis_20261025_v1';G=W/'g_solver_error_analysis_20261025_v1'
def rd(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
protocol=(F/'PROTOCOL_ZH.md').read_text()
(A/'PROTOCOL_ZH.md').write_text('# F/G同口径比例比较\n\nF沿用已封存逐条审核，G按同一标准复查全部多选（各root5题10响应，含MC6）；仅使用既有封存Solver输出，不运行模型、不训练、不改原分数。主比较使用Gold2—5的32次响应/16语义题；MC6控制单列。F/G都已知结果，不是盲审。\n\n以下原F口径原样引用：\n\n'+protocol)
bindings=[bind(F/'FINAL_MANIFEST.json'),bind(F/'SUMMARY.json'),bind(G/'FINAL_MANIFEST.json')]
for root in ['218519','222937','225575','229686']:
 b=A/'reviews'/root;b.mkdir(parents=True,exist_ok=True)
 p=rd(G/'reviews'/root/'ERROR_REVIEW_PACKET.json')
 p['records']=[x for x in p['records'] if x['candidate']['question_type']=='multi_select']
 p['schema_version']='g_reason_selection_same_F_criterion_packet_v1'
 p['protocol']=protocol
 p['required_review_schema']={'root_id':root,'records':[{'candidate_id':'...','responses':[{'request_id':'...','l1_local_correct_reason_wrong_decision':False,'l2_correct_judgment_wrong_box':False,'l3_whole_visible_solution_correct_wrong_box':False,'uncertain':False,'evidence':[{'type':'L1|L2|L3','canonical_labels':['A'],'display_labels':['B'],'raw_quote':'exact raw substring','source_checks':[{'turn_ref':'...','source_quote':'exact source substring'}],'assessment':'...'}],'rationale':'...'}]}]}
 p['scope']='G five MC questions, ten responses. No new ordering review; all semantic/choice rates must use matching MC denominators.'
 (b/'PACKET.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
 bindings+=[bind(G/'reviews'/root/'ERROR_REVIEW_PACKET.json'),bind(F/'reviews'/root/'REVIEW.json')]
(A/'INPUT_BINDINGS.json').write_text(json.dumps(bindings,ensure_ascii=False,indent=2)+'\n')
print('Prepared four packets: 20 G MC questions / 40 response repeats; F reviews bound unchanged.')
