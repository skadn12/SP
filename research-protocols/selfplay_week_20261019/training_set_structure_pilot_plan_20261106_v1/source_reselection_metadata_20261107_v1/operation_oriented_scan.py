import json,re,heapq,hashlib
from pathlib import Path
SRC=Path('/root/autodl-tmp/gxy-selfplay/data/conversation_chronicles/train.jsonl'); OUT=Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_structure_pilot_plan_20261106_v1/source_reselection_metadata_20261107_v1')
EX={'46251','133062','165698','82382','40917','98734','238031','119609','9386','51072','53623','98366','25286','85772','215656','133418','192156','106092'}
rels=['Classmates','Neighbors','Co-workers','Mentee and Mentor','Parent and Child','Patient and Doctor','Husband and Wife','Student and Teacher','Athlete and Coach','Employee and Boss']
STOP=set('the a an and or but if then to of in on for with from by is are was were be been being i you he she it they we them this that these those have has had do did does can could should would will just really very about what when where how why there here their our your my his her its as at into out up down so than too also not no yes all any some more one two three first second third fourth fifth'.split())
MARKERS=re.compile(r'\b(remember|earlier|previous|last time|few weeks|a while ago|mentioned|again|before|since then|as you said|we talked|we discussed|reminds me|already|still|continue|instead|if|because|therefore|should|could|try|suggest|advice|plan|need to|have to|must|goal|decided|agreed)\b',re.I)
TEMP=re.compile(r'\b(before|after|later|then|first|next|finally|already|still|again|last|previous|earlier|week|month|year|tomorrow|yesterday)\b',re.I)

def toks(v): return set(re.findall(r"[a-zA-Z][a-zA-Z']+",' '.join(map(str,v)).lower()))-STOP
heaps={r:[] for r in rels}; count=0
for line in SRC.open(encoding='utf-8'):
 d=json.loads(line); n=d['dataID'].rsplit('-',1)[-1]; rel=d['relationship']; count+=1
 if n in EX or rel not in heaps: continue
 ss=[d.get(k,[]) for k in ('first_session_dialogue','second_session_dialogue','third_session_dialogue','fourth_session_dialogue','fifth_session_dialogue')]
 ts=[toks(x) for x in ss]; refs=sum(len(MARKERS.findall(' '.join(map(str,ss[j])))) for j in range(1,5)); temp=sum(len(TEMP.findall(' '.join(map(str,ss[j])))) for j in range(1,5))
 overlaps=[]
 for j in range(1,5):
  overlaps.append(len(ts[j]&ts[j-1]))
 cross=sum(overlaps); distinct=sum(1 for x in overlaps if x>=4)
 nums=sum(bool(re.search(r'\b\d+(?:\.\d+)?\b', ' '.join(map(str,x)))) for x in ss)
 strategy=sum(len(re.findall(r'\b(should|could|try|suggest|advice|recommend|plan|need to|must|goal|decided|agreed)\b',' '.join(map(str,x)),re.I)) for x in ss)
 score=2*distinct+0.15*cross+0.25*refs+0.15*temp+0.1*strategy+0.2*nums
 item=(score,n,d['dataID'],d['relationship'],d.get('source_line_index'),{'cross_adjacent_token_overlap':overlaps,'reference_markers_later':refs,'temporal_markers_later':temp,'strategy_markers':strategy,'numeric_sessions':nums})
 h=heaps[rel]
 if len(h)<10: heapq.heappush(h,item)
 elif score>h[0][0]: heapq.heapreplace(h,item)
res={'schema_version':'cc_operation_oriented_metadata_scan_v1','source_records_scanned':count,'excluded_current_review_roots':sorted(EX),'method':'heuristic source-text scan for adjacent-session content overlap plus later reference/temporal/strategy markers; candidates require manual review','per_relationship_top10':{r:[{'score':round(x[0],4),'data_id':x[1],'relationship':x[3],'features':x[5]} for x in sorted(heaps[r],reverse=True)] for r in rels},'generation_authorization':'NONE','limitations':['heuristic scores do not assign operation labels','high overlap can signal direct recap/template repetition','no questions, solver, training or export']}
(OUT/'OPERATION_ORIENTED_SCAN.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)); print(json.dumps({'records_scanned':count,'relationship_groups':len(rels)},ensure_ascii=False))
