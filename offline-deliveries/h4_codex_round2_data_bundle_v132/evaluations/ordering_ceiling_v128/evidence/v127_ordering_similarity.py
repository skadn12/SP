import collections,datetime,difflib,hashlib,json,math,re
from pathlib import Path
REPO=Path('/root/autodl-tmp/gxy-selfplay')
TRAIN=REPO/'offline-deliveries/h4_codex_train_pool_v125'
DEV=REPO/'offline-deliveries/h4_codex_dev_test_ab_v124/delivery/dev/QUESTIONS_WITH_ANSWERS.json'
alltrain=json.loads((TRAIN/'data/COMBINED_QUESTIONS.json').read_text())['questions']
train=[q for q in alltrain if q['question_type']=='ordering'];dev=[q for q in json.loads(DEV.read_text())['questions'] if q['question_type']=='ordering']
def norm(s):return ' '.join(re.findall(r"[a-z0-9]+",s.lower()))
def words(s):return set(norm(s).split())
def jac(a,b):return len(a&b)/len(a|b) if a|b else 1
def grams(s,n=5):
 w=norm(s).split();return set(tuple(w[i:i+n]) for i in range(len(w)-n+1))
def qcontent(q):return ' '.join(sorted(norm(v) for v in q['items'].values()))
def orderedcontent(q):return ' '.join(norm(q['items'][i]) for i in q['gold'])
def anchor(q,label):
 v=q['order_anchor_spans'][label]
 if isinstance(v,list):v=v[0]
 return v
for qs in [train,dev]:
 for q in qs:q['_itemsets']={k:words(v) for k,v in q['items'].items()};q['_grams']=grams(orderedcontent(q))
train_items=[{'candidate_id':q['candidate_id'],'root_id':q['root_id'],'label':k,'text':v,'tokens':words(v)} for q in train for k,v in q['items'].items()]
dev_items=[{'candidate_id':q['candidate_id'],'root_id':q['root_id'],'label':k,'text':v,'tokens':words(v)} for q in dev for k,v in q['items'].items()]
exact_pairs=[];normalized_pairs=[];nearest=[]
for di in dev_items:
 scores=[]
 for ti in train_items:
  score=jac(di['tokens'],ti['tokens']);scores.append((score,ti))
  if di['text']==ti['text']:exact_pairs.append([di['candidate_id'],di['label'],ti['candidate_id'],ti['label'],di['text']])
  if norm(di['text'])==norm(ti['text']):normalized_pairs.append([di['candidate_id'],di['label'],ti['candidate_id'],ti['label'],di['text']])
 score,ti=max(scores,key=lambda x:x[0]);nearest.append({'dev_candidate_id':di['candidate_id'],'dev_label':di['label'],'dev_text':di['text'],'train_candidate_id':ti['candidate_id'],'train_label':ti['label'],'train_text':ti['text'],'token_jaccard':score})
nearest.sort(key=lambda x:-x['token_jaccard'])
question_nearest=[]
for dq in dev:
 score,tq=max(((jac(dq['_grams'],tq['_grams']),tq) for tq in train),key=lambda x:x[0])
 question_nearest.append({'dev_candidate_id':dq['candidate_id'],'train_candidate_id':tq['candidate_id'],'ordered_item_5gram_jaccard':score,'train_items':[tq['items'][i] for i in tq['gold']],'dev_items':[dq['items'][i] for i in dq['gold']]})
def source_rows(questions):
 result={}
 for q in questions:
  if q['root_id'] in result:continue
  src=json.loads(Path(q['source_path']).read_text())
  turns=src['turns'];original=[t for t in turns if not str(t.get('turn_ref','')).startswith('R02:')];new=[t for t in turns if str(t.get('turn_ref','')).startswith('R02:')]
  text='\n'.join(t['speaker']+': '+t['text'] for t in turns);newtext='\n'.join(t['speaker']+': '+t['text'] for t in new)
  result[q['root_id']]={'path':q['source_path'],'sha256':q['source_sha256'],'source_id':src.get('source_id'),'source_item_sha256':src.get('source_item_sha256'),'normalized_full_sha256':hashlib.sha256(norm(text).encode()).hexdigest(),'normalized_new_sha256':hashlib.sha256(norm(newtext).encode()).hexdigest(),'full_grams':grams(text),'new_grams':grams(newtext),'turns':turns,'new_turns':new,'original_turns':original,'speakers':sorted({t['speaker'] for t in turns}),'new_speakers':sorted({t['speaker'] for t in new}),'new_scene_refs':sorted({t['scene_ref'] for t in new})}
 return result
trsrc=source_rows(alltrain);dvsrc=source_rows(dev)
source_nearest=[]
for dr,ds in dvsrc.items():
 scored=[(jac(ds['new_grams'],ts['new_grams']),rt,ts) for rt,ts in trsrc.items()];score,rt,ts=max(scored,key=lambda x:x[0])
 source_nearest.append({'dev_root':dr,'nearest_train_root':rt,'new_turn_5gram_jaccard':score,'full_turn_5gram_jaccard':jac(ds['full_grams'],ts['full_grams']),'dev_new_speakers':ds['new_speakers'],'train_new_speakers':ts['new_speakers']})
turntrain=collections.defaultdict(list)
for rt,ss in trsrc.items():
 for t in ss['new_turns']:turntrain[norm(t['text'])].append((rt,t['turn_ref'],t['text']))
turn_duplicates=[]
for dr,ss in dvsrc.items():
 for t in ss['new_turns']:
  key=norm(t['text'])
  if key in turntrain:
   turn_duplicates.append({'dev_root':dr,'turn_ref':t['turn_ref'],'text':t['text'],'word_count':len(key.split()),'train_matches':[list(x) for x in turntrain[key]]})
# Verify one distinct supported item per scene, no within-scene comparisons.
scene_summaries={}
for name,qs in [('train',train),('dev',dev)]:
 signatures=collections.Counter();same_scene=0;actor_prefix=collections.Counter();scene_semantics=collections.defaultdict(list)
 for q in qs:
  refs=[anchor(q,k)['turn_ref'].rsplit(':T',1)[0] for k in q['gold']];signatures[tuple(refs)]+=1
  same_scene+=len(set(refs))<len(refs)
  for k,text in q['items'].items():
   match=re.match(r'^\S+\s+(\S+)',text);actor_prefix[match.group(1).lower() if match else '?']+=1
   scene_semantics[anchor(q,k)['turn_ref'].rsplit(':T',1)[0]].append(text)
 scene_summaries[name]={'question_count':len(qs),'root_count':len({q['root_id'] for q in qs}),'stem_unique_count':len({q['question'] for q in qs}),'chronology_kind_counts':dict(collections.Counter(q['chronology_kind'] for q in qs)),'gold_scene_signatures':[{'signature':list(k),'count':v} for k,v in signatures.items()],'questions_with_any_two_items_same_scene':same_scene,'item_reporting_verb_counts':dict(actor_prefix),'scene_item_examples':{k:v[:3] for k,v in scene_semantics.items()}}
def duplicatefield(field):
 t={str(s[field]) for s in trsrc.values() if s[field] is not None};d={str(s[field]) for s in dvsrc.values() if s[field] is not None};return sorted(t&d)
result={'schema_version':'v127_train_dev_ordering_similarity_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only train and dev ordering/content comparison; no test question/source reads, no model calls.','counts':{'train_all_questions':len(alltrain),'train_ordering':len(train),'dev_ordering':len(dev),'train_ordering_items':len(train_items),'dev_ordering_items':len(dev_items),'all_train_sources':len(trsrc),'dev_delivered_sources':len(dvsrc)},'exact_stem_train_dev_identical':len({q['question'] for q in train+dev})==1,'common_stem':train[0]['question'],'scene_structure':scene_summaries,'exact_leakage_checks':{'root_overlap':sorted(set(trsrc)&set(dvsrc)),'candidate_id_overlap':sorted({q['candidate_id'] for q in train}&{q['candidate_id'] for q in dev}),'source_sha256_overlap':duplicatefield('sha256'),'source_id_overlap':duplicatefield('source_id'),'source_item_sha256_overlap':duplicatefield('source_item_sha256'),'normalized_full_dialogue_sha256_overlap':duplicatefield('normalized_full_sha256'),'normalized_new_dialogue_sha256_overlap':duplicatefield('normalized_new_sha256'),'exact_whole_question_itemset_overlap_count':len(set(qcontent(q) for q in train)&set(qcontent(q) for q in dev)),'exact_item_pairs_count':len(exact_pairs),'normalized_item_pairs_count':len(normalized_pairs),'exact_item_pairs':exact_pairs,'normalized_item_pairs':normalized_pairs},'item_similarity':{'method':'Best train-item set-of-alphanumeric-word Jaccard for each dev item, no stem included; descriptive screening, not a semantic leakage classifier.','dev_items_with_best_jaccard_ge_0_8':sum(r['token_jaccard']>=.8 for r in nearest),'dev_items_with_best_jaccard_ge_0_9':sum(r['token_jaccard']>=.9 for r in nearest),'nearest_pairs':nearest},'whole_question_similarity':{'method':'Best Jaccard over word 5-grams of gold-ordered item texts; excludes common stem and displayed labels. Gold only orders offline audit text; never used for inference.','nearest_pairs':question_nearest},'source_similarity':{'method':'Best train-vs-dev word5gram set Jaccard over generated R02 turn texts with speakers; full-dialogue similarity also shown. No semantic near-duplicate decision is inferred from this threshold alone.','nearest_pairs':source_nearest,'exact_generated_turn_overlap_occurrences':len(turn_duplicates),'exact_generated_turn_overlap_occurrences_ge10words':sum(x['word_count']>=10 for x in turn_duplicates),'exact_generated_turn_overlap_occurrences_ge20words':sum(x['word_count']>=20 for x in turn_duplicates),'exact_generated_turn_overlap_examples':sorted(turn_duplicates,key=lambda x:-x['word_count'])[:12],'train_new_speaker_pairs':dict(collections.Counter(', '.join(v['new_speakers']) for v in trsrc.values())),'dev_new_speaker_pairs':dict(collections.Counter(', '.join(v['new_speakers']) for v in dvsrc.values()))},'evidence_paths':{'train_questions':str(TRAIN/'data/COMBINED_QUESTIONS.json'),'dev_questions':str(DEV)},'interpretation':['All125 train and all14 dev ordering questions use exactly one identical stem and exactly one anchor from each of the same six scenes, in fixed A1,A2,A3,B1,B2,B3 chronology. There are no within-scene item pairs. This is a narrow cross-scene retrieval-and-ordering format; 12 display permutations do not add independent content difficulty.','Disjoint root/source IDs and no exact full source/itemset duplicate support absence of direct exact leakage in checked artifacts, but do not establish semantic diversity or causal generalization.','Same actors, restaurant/hotel/taxi/train booking motifs and reporting verbs can recur across distinct roots; literal overlap of generic dialogue turns must be separated from whole-dialogue/question duplication.']}
out=Path('/tmp/v127_ordering_similarity_details.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['counts','exact_stem_train_dev_identical','exact_leakage_checks']},ensure_ascii=False))
print('top_item_nearest',json.dumps(nearest[:6],ensure_ascii=False));print('source_nearest',json.dumps(source_nearest,ensure_ascii=False));print('turnoverlap',len(turn_duplicates),sum(x['word_count']>=10 for x in turn_duplicates),sum(x['word_count']>=20 for x in turn_duplicates));print('details',out)
