from pathlib import Path
import json,hashlib,re,collections,copy
A=Path(__file__).parent
summary=json.load(open(A/'SUMMARY.json'));official=json.load(open(A/'OFFICIAL_SCORE_AUDIT.json'));report=(A/'REPORT_ZH.md').read_text();examples=(A/'EXAMPLES_ZH.md').read_text()
allq=[];allresp=[];cards=[];sources={};packets={};revset={};issues=[]
for root in ['218519','222937','225575','229686']:
 b=A/'reviews'/root;p=json.load(open(b/'ERROR_REVIEW_PACKET.json'));packets[root]=p;s=json.load(open(p['source_binding']['path']));turns={t['turn_ref']:t['text'] for t in s['turns']};sources[root]=turns
 effective=b/'ERROR_REVIEW_EFFECTIVE.json';rv=json.load(open(effective if effective.exists() else b/'ERROR_REVIEW.json'));revset[root]=rv
 byid={r['candidate_id']:r for r in rv['records']};cr=json.load(open(b/'CARD_MATCH_REVIEW.json'));cards+=cr['records']
 for q in p['records']:
  qr=byid[q['candidate']['candidate_id']];allq.append((q,qr))
  for o,v in zip(q['responses'],qr['responses']):
   assert o['request_id']==v['request_id'];st=o['score']['strict'];assert st['exact']==v['strict_exact'] and st['valid']==v['response_valid'];allresp.append((q,o,v))
   for e in v['evidence']:
    assert e['raw_quote'] in o['raw_output'];mp=o['presentation']['canonical_to_display'];can=e['canonical_label_or_pair'];assert e['display_label_or_pair']==([mp[x] for x in can] if isinstance(can,list) else ('<'.join(mp[x] for x in can.split('<')) if '<' in can else mp[can]))
    for z in e['source_checks']:assert z['source_quote'] in turns[z['turn_ref']]
   assert not v['strict_exact'] or (v['primary_failure_code'] is None and v['final_failure_codes']==[])
assert len(allq)==32 and len(allresp)==64
for k,a in summary['by_slot'].items():
 rr=[(q,o,v) for q,o,v in allresp if q['candidate']['slot_id']==k]
 assert a['responses']==len(rr) and a['exact']==sum(v['strict_exact'] for q,o,v in rr) and a['valid']==sum(v['response_valid'] for q,o,v in rr)
 assert a['citation_defect']==sum(v['citation_defect'] for q,o,v in rr)
 assert a['exact']==official['by_slot'][k]['exact_correct'] and a['responses']==official['by_slot'][k]['requests']
 qq=[(q,v) for q,v in allq if q['candidate']['slot_id']==k]
 assert a['both_correct']==sum(all(x['strict_exact'] for x in r['responses']) for q,r in qq)
 assert a['any_correct']==sum(any(x['strict_exact'] for x in r['responses']) for q,r in qq)
 assert a['both_wrong']==sum(not any(x['strict_exact'] for x in r['responses']) for q,r in qq)
for k,n in [('exact',sum(v['strict_exact'] for q,o,v in allresp)),('valid',sum(v['response_valid'] for q,o,v in allresp)),('citation_defect',sum(v['citation_defect'] for q,o,v in allresp))]:assert summary['overall'][k]==n
assert sum(summary['primary_failure_codes'].values())==49
assert summary['citation_defect']['among_correct']==sum(v['strict_exact'] and v['citation_defect'] for q,o,v in allresp)==12
assert summary['correct_with_non_citation_reason_errors']==sum(v['strict_exact'] and bool(set(v['reasoning_error_codes'])-{'EVIDENCE_SCOPE_OR_CITATION_BINDING_ERROR'}) for q,o,v in allresp)==4
mc=[(q,o,v) for q,o,v in allresp if q['candidate']['slot_id'] in ['MC2','MC3','MC4','MC5']]
assert sum(o['score']['strict']['false_positive'] for q,o,v in mc)==33
assert sum(o['score']['strict']['false_negative'] for q,o,v in mc)==5
assert sum(bool(o['score']['strict']['false_positive']) for q,o,v in mc)==25
order=[(q,o,v) for q,o,v in allresp if q['candidate']['question_type']=='ordering']
assert sum(o['score']['strict']['pairwise_correct'] for q,o,v in order)==234
assert sum(o['score']['strict']['pairwise_total'] for q,o,v in order)==360
assert sum(o['score']['strict']['pairwise_total'] for q,o,v in order if v['response_valid'])==315
assert official['exact_definition_audit']['citation_semantic_entailment_part_of_automatic_exact_gate'] is False
# Check example raw blocks, question text, options, mappings, predictions and entire source turns verbatim.
example_q_count=example_r_count=example_turn_count=0
chunks=re.split(r'^## (CL1-G-[^\n]+)\n',examples,flags=re.M)
for cid,section in zip(chunks[1::2],chunks[2::2]):
 example_q_count+=1;q=next(q for q,rv in allq if q['candidate']['candidate_id']==cid);c=q['candidate'];assert c['question'] in section
 for k,v in c.get('options',c.get('items')).items():assert f'- {k}: {v}' in section
 assert 'canonical Gold: '+' '.join(c['gold']) in section
 for x in q['responses']:
  m=re.search(r'^### '+re.escape(x['request_id'])+r'\n(.*?)(?=^### |\Z)',section,flags=re.M|re.S);assert m
  rs=m.group(1);raw=re.search(r'```json\n(.*?)\n```',rs,flags=re.S).group(1);assert raw==x['raw_output'];example_r_count+=1
  assert json.dumps(x['presentation']['display_to_canonical'],ensure_ascii=False) in rs
  st=x['score']['strict'];assert 'display Gold: `'+ ' '.join(st['display_gold'])+'`；display prediction: `'+ ' '.join(st['display_prediction'] or [])+'`' in rs
 src=section.split('### 对照来源turn\n',1)[1]
 root=cid.split('-')[2]
 for ref,quote in re.findall(r'^`([^`]+)`\n\n(.*?)(?=\n\n`|\Z)',src,flags=re.M|re.S):
  assert quote.strip()==sources[root][ref];example_turn_count+=1
assert example_q_count==7 and example_r_count==14
# Reapply explicit correction log to preserved phase1, verify effective record body exactly.
b=A/'reviews'/'218519';orig=json.load(open(b/'ERROR_REVIEW.json'));corr=json.load(open(b/'ERROR_REVIEW_CORRECTIONS.json'));effective=json.load(open(b/'ERROR_REVIEW_EFFECTIVE.json'));calc=copy.deepcopy(orig)
assert hashlib.sha256((b/'ERROR_REVIEW.json').read_bytes()).hexdigest()==corr['original_sha256'];assert corr['after_card_packet_read'] is True
for ch in corr['changes']:
 qr=next(r for r in calc['records'] if r['candidate_id']==ch['candidate_id']);rr=next(x for x in qr['responses'] if x['request_id']==ch['request_id'])
 if 'evidence_index' in ch:
  idx=ch['evidence_index']
  if 'field' in ch:assert rr['evidence'][idx][ch['field']]==ch['before'];rr['evidence'][idx][ch['field']]=ch['after']
  else:assert rr['evidence'][idx]==ch['before'];rr['evidence'][idx]=ch['after']
 else:assert rr[ch['field']]==ch['before'];rr[ch['field']]=ch['after']
assert calc['records']==effective['records']
# Review narrative boundaries.
assert '两排列属于同题重复测量' in report and '引用是否真的支持理由' in report and '不在自动exact门槛内' in report
assert '非盲探索' in report and '不能把6/15与F的16/24相减' in report and '同模型家族' in report
if not any(x in report for x in ['F/G共用','F/G共用相同','F/G共享','两臂共享','两组共享','相同R03','同一扩充来源']):
 issues.append({'issue_id':'FINAL-BOUNDARY-01','severity':'MINOR_DISCLOSURE','file':'REPORT_ZH.md','problem':'已披露Train既有roots、非盲与事后目标映射，但未直接写明F/G共用同一批R03扩充来源，因此本轮不能隔离扩充反馈收益与出题反馈收益。','recommended_fix':'在第5或第6节加入：F/G共用同一批R03扩充来源；本分析不能单独估计扩充反馈收益。'})
result={'schema_version':'independent_g_final_report_review_v1','reviewer':'/root/g_error_review_225575','status':'PASS_WITH_MINOR_DISCLOSURE' if issues else 'PASS','scope':['REPORT_ZH.md','SUMMARY.json','OFFICIAL_SCORE_AUDIT.json','EXAMPLES_ZH.md','ROOT218519_REVIEW_AUDIT.json','reviews/218519/ERROR_REVIEW_CORRECTIONS.json','reviews/218519/ERROR_REVIEW_EFFECTIVE.json'],'new_solver_calls':0,'training':False,'original_experiment_modified':False,'checks':{'64_response_and_32_semantic_question_counts':'PASS','slot_counts_against_original_packet_and_official_audit':'PASS','correct_and_wrong_final_reason_citation_separation':'PASS','MC_overselection_and_underselection':'PASS','ordering_pairwise_full_and_valid_only_denominators':'PASS','14_example_raw_outputs_exact':'PASS','7_example_questions_options_Gold_and_mappings':'PASS','example_source_turns_verbatim':'PASS','all_effective_evidence_raw_quotes_source_quotes_and_label_mappings':'PASS','218519_explicit_correction_reapplication':'PASS','citation_semantics_not_automatic_exact_gate':'PASS','posthoc_G_card_mapping_not_causal_FG_comparison':'PASS','same_model_nonblind_exposed_train_roots_limits':'PASS'},'verified_counts':{'questions':32,'responses':64,'exact':15,'valid':61,'example_questions':example_q_count,'example_responses':example_r_count,'example_turns':example_turn_count},'issues':issues,'limitations':['独立快审复用已有packet与审计，不重新加载模型。','审查者已参与225575阶段1/阶段2，因此不声称整项内容审查盲于该root。'],'input_bindings':{n:hashlib.sha256((A/n).read_bytes()).hexdigest() for n in ['REPORT_ZH.md','SUMMARY.json','OFFICIAL_SCORE_AUDIT.json','EXAMPLES_ZH.md']}}
(A/'FINAL_REPORT_REVIEW.json').open('x').write(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
