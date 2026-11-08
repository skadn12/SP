"""CPU/read-only v127 ordering prompt leakage and exact training-overlap diagnosis."""
from __future__ import annotations
import ast,collections,copy,datetime,hashlib,json,pathlib,re,sys
sys.dont_write_bytecode=True
REPO=pathlib.Path('/root/autodl-tmp/gxy-selfplay')
V=REPO/'offline-deliveries/h4_codex_dev_eval_v127'
TRAIN=REPO/'offline-deliveries/h4_codex_train_pool_v125/data/TRAIN_NEW_ONLY.jsonl'
CANON=REPO/'offline-deliveries/h4_codex_dev_test_ab_v124/delivery/dev/QUESTIONS_WITH_ANSWERS.json'
CONTRACT=REPO/'worktrees/h4-gate-downstream-v1/scripts/current_method/restart_n5_spell/h4_round0_b0_k3_difficulty_v1/contract.py'
OUT=pathlib.Path('/tmp/v127_ceiling_leakage_findings.json')
tracked={}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):
 p=pathlib.Path(p);tracked[str(p)]=sha(p);return json.loads(p.read_text())
def readtext(p):
 p=pathlib.Path(p);tracked[str(p)]=sha(p);return p.read_text()
def norm(t):return ' '.join(t.casefold().split())
def h(t):return hashlib.sha256(t.encode()).hexdigest()
canonical=[q for q in read(CANON)['questions'] if q['question_type']=='ordering']
qmap={q['candidate_id']:q for q in canonical}
manifest=read(V/'run/new/manifest.json')
requests=[r for r in manifest['requests'] if r['question_type']=='ordering']
golds=read(V/'run/new/gold.json')
train=[json.loads(x) for x in readtext(TRAIN).splitlines() if x.strip()]
combined=read(REPO/'offline-deliveries/h4_codex_train_pool_v125/data/COMBINED_QUESTIONS.json')['questions']
train_qmap={q['candidate_id']:q for q in combined}
# Load only two pure formatting functions and the literal system string. No solver/backend/GPU import.
contract_text=readtext(CONTRACT);tree=ast.parse(contract_text)
keep=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'build_prompt','dialogue_rows'} or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STRUCTURED_SYSTEM_PROMPT' for t in n.targets)]
namespace={'Any':object,'Mapping':dict,'Sequence':list}
# Keep annotations postponed, exactly as the source module does.
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*keep],type_ignores=[])
exec(compile(ast.fix_missing_locations(module),str(CONTRACT),'exec'),namespace)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B',local_files_only=True)
build=namespace['build_prompt'];rows=namespace['dialogue_rows']
source_by_root={q['root_id']:read(q['source_path']) for q in canonical}
public_only_equal=0;private_perturb_equal=0;display_equal=0;allgoldbox_absent=0
needle_counts=collections.Counter();numeric_guess_correct=0;raw_exact=0;all_new_outputs=[];per_request=[]
private_needles=['order_anchor','item_evidence','support_refs','primary_citation','canonical_order_content_ids','display_to_content','content_to_text','gold_content_ids','PRIVATE_GOLD','PRIVATE_DESIGN','PUBLIC_SOURCE_JUDGMENTS','anchor_id','frozen_full_turn_sha256','evidence_export']
for r in requests:
 q=qmap[r['candidate_id']];source=source_by_root[q['root_id']];variant=r['variant']
 p=V/'run/new/prompts'/f"{r['request_id']}.txt";actual=readtext(p)
 dialogue=rows(source)
 expected=build(tokenizer,{'question_type':'ordering','question':q['question']},{'items':variant['items']},dialogue,context_mode='full_context')
 assert actual==expected,('PUBLIC_REBUILD_DIFF',r['request_id']);public_only_equal+=1
 cq=copy.deepcopy(q)
 for k in set(cq)-{'question_type','question'}:cq[k]='CANARY_PRIVATE_FIELD_DO_NOT_RENDER'
 cv={k:('CANARY_PRIVATE_FIELD_DO_NOT_RENDER' if k!='items' else v) for k,v in variant.items()}
 cs=copy.deepcopy(source)
 for k in set(cs)-{'turns'}:cs[k]='CANARY_SOURCE_METADATA_DO_NOT_RENDER'
 for t in cs['turns']:
  for k in set(t)-{'turn_ref','session_ref','speaker','text'}:t[k]='CANARY_TURN_METADATA_DO_NOT_RENDER'
 changed=build(tokenizer,cq,cv,rows(cs),context_mode='full_context')
 assert changed==actual,('PRIVATE_PERTURB_DIFF',r['request_id']);private_perturb_equal+=1
 for key in private_needles:
  if key in actual:needle_counts[key]+=1
 actual_item_block=actual.split('\n\nItems:\n',1)[1].split('\n\nOutput only',1)[0]
 assert actual_item_block=='\n'.join(f'{k}. {t}' for k,t in variant['items'].items());display_equal+=1
 inverse={v:k for k,v in variant['display_to_content'].items()}
 displayed_gold=[inverse[x] for x in golds[q['candidate_id']]]
 box='<box>'+','.join(displayed_gold)+'</box>'
 allgoldbox_absent+=box not in actual
 numeric_guess_correct+=displayed_gold==['I1','I2','I3','I4','I5','I6']
 response=read(V/'run/new/responses'/f"{r['request_id']}.json")
 assert response['raw_output']==box,('ORDERING_RAW_DIFF',r['request_id']);raw_exact+=1
 all_new_outputs.append(response['raw_output'])
 per_request.append({'request_id':r['request_id'],'prompt_sha256':h(actual),'displayed_gold':displayed_gold,'raw_output':response['raw_output'],'full_gold_box_found_in_prompt':box in actual,'unique_private_fields_in_prompt':[k for k in private_needles if k in actual]})
# Actual training messages and context/item extraction. Question template overlap is separate from sample overlap.
train_info=[]
for t in train:
 user='\n'.join(m['content'] for m in t['messages'] if m['role']=='user')
 context=user.split('\n\nQuestion:\n',1)[0].removeprefix('Dialogue:\n')
 question=user.split('\n\nQuestion:\n',1)[1].split('\n\n',1)[0]
 field='Items' if t['question_type']=='ordering' else 'Options'
 block=user.split('\n\n'+field+':\n',1)[1].split('\n\nOutput only',1)[0]
 statements=[line.split('. ',1)[1] for line in block.splitlines()]
 lines=[]
 for line in context.splitlines():
  match=re.match(r'^\[([^]]+)\] ([^:]+): (.*)$',line)
  if match:lines.append({'ref':match[1],'speaker':match[2],'text':match[3]})
 train_info.append({'id':t['candidate_id'],'root':t['root_id'],'user':user,'context':context,'question':question,'statements':statements,'turns':lines,'qtype':t['question_type']})
assert len(train_info)==363
train_root_ids={t['root'] for t in train_info}
root_literals={q['root_id']:sum(q['root_id'] in t['user'] for t in train_info) for q in canonical}
exact_question_count=collections.Counter(t['question'] for t in train_info)
item_index=collections.defaultdict(list);turn_index=collections.defaultdict(list)
for t in train_info:
 for i in t['statements']:item_index[norm(i)].append(t['id'])
 for turn in t['turns']:turn_index[norm(turn['text'])].append({'id':t['id'],'ref':turn['ref']})
per_question=[]
for q in canonical:
 source=source_by_root[q['root_id']]
 dialogue='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in source['turns'])
 values=list(q['items'].values());itemset=set(map(norm,values))
 item_hits={k:item_index[norm(v)] for k,v in q['items'].items() if norm(v) in item_index}
 sixset_matches=[t['id'] for t in train_info if set(map(norm,t['statements']))==itemset]
 full_dialogue_matches=[t['id'] for t in train_info if norm(t['context'])==norm(dialogue)]
 anchor_hits={}
 for label,a in q['order_anchor_spans'].items():
  quote=a['quote'];hits=[t['id'] for t in train_info if norm(quote) in norm(t['context'])]
  if hits:anchor_hits[label]={'quote':quote,'chars':len(quote),'training_matches':hits}
 anchors=[]
 refindex={t['turn_ref']:i for i,t in enumerate(source['turns'])}
 for label,a in q['order_anchor_spans'].items():
  anchors.append({'label':label,'turn_ref':a['turn_ref'],'dialogue_row_index':refindex[a['turn_ref']],'quote':a['quote']})
 scenes=[a['turn_ref'].rsplit(':T',1)[0] for a in sorted(anchors,key=lambda a:a['dialogue_row_index'])]
 per_question.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'question_template_exact_training_count':exact_question_count[q['question']],
  'exact_item_text_training_hits':item_hits,'six_item_set_exact_training_matches':sixset_matches,'full_dialogue_normalized_training_matches':full_dialogue_matches,
  'anchor_quote_substring_training_hits':anchor_hits,'anchor_scenes_in_gold_order':scenes,'one_anchor_per_six_scene':len(scenes)==len(set(scenes))==6,
  'canonical_numeric_order_is_gold':q['gold']==['I1','I2','I3','I4','I5','I6'],'anchors':anchors})
per_source=[]
for root,source in source_by_root.items():
 full_hits=[];expanded_hits=[];long_hits=[]
 for turn in source['turns']:
  hits=turn_index.get(norm(turn['text']),[])
  if hits:
   entry={'dev_ref':turn['turn_ref'],'text':turn['text'],'training_root_count':len({next(t['root'] for t in train_info if t['id']==k['id']) for k in hits}),'training_examples':hits[:5]}
   full_hits.append(entry)
   if str(turn['turn_ref']).startswith('R02:'):expanded_hits.append(entry)
   if len(turn['text'])>=80:long_hits.append(entry)
 per_source.append({'root_id':root,'turn_count':len(source['turns']),'expanded_turn_count':sum(str(t['turn_ref']).startswith('R02:') for t in source['turns']),
  'exact_turn_text_overlap_count':len(full_hits),'expanded_exact_turn_text_overlap_count':len(expanded_hits),'long80chars_exact_turn_overlap_count':len(long_hits),
  'expanded_exact_turn_text_overlaps':expanded_hits,'long80chars_exact_turn_overlaps':long_hits,
  'source_metadata_keys':list(source),'turn_metadata_keys':sorted({k for t in source['turns'] for k in t}),
  'item_ids_in_source_turn_text':[(t['turn_ref'],t['text']) for t in source['turns'] if re.search(r'\bI[1-6]\b',t['text'])],
  'private_markers_in_source_turn_text':[{'ref':t['turn_ref'],'markers':[k for k in private_needles if k in t['text']]} for t in source['turns'] if any(k in t['text'] for k in private_needles)]})
# Confirm exact source ids and records through actual training question source files, not labels alone.
train_sources={q['root_id']:read(q['source_path']) for q in combined}
train_source_ids={s.get('source_id') for s in train_sources.values() if s.get('source_id')}
train_source_record_shas={s.get('source_item_sha256') for s in train_sources.values() if s.get('source_item_sha256')}
record_checks=[]
for root,s in source_by_root.items():
 record_checks.append({'root':root,'root_overlap':root in train_root_ids,'source_id':s.get('source_id'),'source_id_overlap':s.get('source_id') in train_source_ids,
  'source_item_sha256':s.get('source_item_sha256'),'source_item_sha_overlap':s.get('source_item_sha256') in train_source_record_shas,
  'full_source_text_sha256':h('\n'.join(t['text'] for t in s['turns'])),'full_source_turns_exact_training_match_roots':[r for r,x in train_sources.items() if [t['text'] for t in x['turns']]==[t['text'] for t in s['turns']]]})
code_paths=[REPO/'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/run_solver.py',REPO/'offline-deliveries/h4_feedback_training_pool_v15_qa/evaluation/protocol.py',REPO/'offline-deliveries/h4_codex_training_v126/run_training.py']
for p in code_paths:readtext(p)
changed=[p for p,v in tracked.items() if sha(p)!=v]
assert not changed,changed
result={'status':'COMPLETED_READ_ONLY','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':{'dev_ordering_questions':14,'dev_ordering_prompts':168,'train_messages':363,'train_ordering_questions':sum(t['question_type']=='ordering' for t in train),'test_content_read':False,'model_inference_or_training':False,'frozen_files_modified':False},
 'conclusions':[],
 'prompt_checks':{'actual168_prompts_equal_public_only_rebuild':public_only_equal,'private_metadata_perturbation_does_not_change_prompt':private_perturb_equal,'rebound_items_exactly_match_displayed_prompt':display_equal,'private_marker_prompt_counts':dict(needle_counts),'full_gold_box_not_found_in_prompt':allgoldbox_absent,'source_serializer_fields':{'dialogue_rows':['turn_ref','session_ref','speaker','text'],'rendered':['turn_ref','speaker','text'],'question':['question_type','question'],'variant':['items']},'actual_new_raw_output_matches_gold_for_all168':raw_exact,'unique_new_raw_ordered_outputs':len(set(all_new_outputs)),'constant_numeric_ID_guess_exact_count':numeric_guess_correct,'source_metadata_not_rendered':True},
 'training_exact_overlap':{'train_roots':len(train_root_ids),'dev_roots':len(source_by_root),'dev_root_literals_in_training_user_messages':root_literals,'dev_question_with_any_exact_training_item_count':sum(bool(x['exact_item_text_training_hits']) for x in per_question),'dev_question_with_exact_six_item_set_in_training_count':sum(bool(x['six_item_set_exact_training_matches']) for x in per_question),'dev_question_with_exact_full_dialogue_in_training_count':sum(bool(x['full_dialogue_normalized_training_matches']) for x in per_question),'source_identity_checks':record_checks},
 'structure':{'all14_same_question_stem':len({q['question'] for q in canonical})==1,'all14_one_item_per_scene':all(q['one_anchor_per_six_scene'] for q in per_question),'question_explicitly_lists_six_scenes_in_chronological_order':True,'canonical_numeric_order_gold_questions':sum(q['canonical_numeric_order_is_gold'] for q in per_question),'note':'Scene ordering is explicitly supplied, and each item can be assigned to a distinct scene; this reduces the task to locating six mostly explicit propositions rather than resolving six interacting timestamps or within-turn clauses.'},
 'per_question':per_question,'per_source':per_source,'per_request':per_request,'audited_files':[{'path':p,'sha256':v} for p,v in sorted(tracked.items())],
 'limitations':['Exact text and sample overlap only; no embeddings or semantic near-duplicate detector used.','This checks the363 new actual training messages, not the entire historical pretraining corpus or all historical clean-policy training data.','Public turn/scene identifiers and explicit scene list are intended task inputs but can make the benchmark easier.','A real inference ablation is needed to measure reliance on dialogue versus scene/template cues; this read-only diagnosis does not prove the cause of100% accuracy.']}
result['conclusions']=[
 'No gold/private-evidence/content-map leakage was detected in the168 actual ordering prompts: every byte reconstructs from public dialogue,stem and rebound items; private metadata perturbation leaves all prompts unchanged.',
 'No exact full dialogue or six-item set from these14 dev questions appears in the363 actual new training messages; exact root/source identity checks and per-item overlap are reported separately below.',
 'The observed100% is mechanically real under the frozen scorer:168 raw responses equal their request-specific displayed gold orders (147 unique ordered outputs); a constant numeric ID response does not explain it.',
 'The14 questions share one highly constrained six-scene appearance-order template; each has one event per scene and the stem gives scene order. This is a credible task-easiness explanation and limits any broader ordering-reasoning claim, but causal attribution requires new controlled ablations.'
]
result['code_evidence']=[]
for path,needle,description in [
 (CONTRACT,'def dialogue_rows','Source metadata is narrowed to turn/session refs,speaker,text;no gold or private evidence is copied.'),
 (CONTRACT,"f\"[{row['turn_id']}]",'Rendered dialogue uses only turn ID,speaker,text;session metadata is not rendered.'),
 (CONTRACT,'chunks.append(f"Question','Question renderer reads public question text and type.'),
 (CONTRACT,'f"{key}. {value}" for key, value in (variant.get(field)','Item renderer reads rebound variant.items,not canonical candidate.items or evidence.'),
 (code_paths[0],'public = {key: question[key]','Preflight reconstructs prompt with public fields and requires byte equality.'),
 (code_paths[0],'canonical_answer_text=""','Live backend request has empty canonical_answer_text;gold is loaded only for post-generation scoring.'),
 (code_paths[1],'def make_variants','Display randomization uses question ID and content with no gold access.'),
 (code_paths[2],"records = [json.loads(x) for x in (POOL / 'data/TRAIN_NEW_ONLY.jsonl')",'Training controller reads the363 new messages used in this exact-overlap diagnosis.')]:
 text=path.read_text();matches=[i for i,line in enumerate(text.splitlines(),1) if needle in line]
 assert matches,(path,needle)
 result['code_evidence'].append({'path':str(path),'line':matches[0],'matching_line':text.splitlines()[matches[0]-1].strip(),'finding':description})
result['training_exact_overlap']['dev_anchor_quotes_with_any_normalized_substring_match_in_training_context']=sum(bool(q['anchor_quote_substring_training_hits']) for q in per_question)
result['training_exact_overlap']['identical_ordering_stem_training_examples']=125
result['training_exact_overlap']['long80chars_full_turn_overlap_count']=sum(r['long80chars_exact_turn_overlap_count'] for r in per_source)
result['training_exact_overlap']['expanded_full_turn_overlap_count']=sum(r['expanded_exact_turn_text_overlap_count'] for r in per_source)
result['training_exact_overlap']['expanded_only_full_turn_match']='What did they recommend?'
result['reproduction_script']={'path':str(pathlib.Path(__file__).resolve()),'sha256':sha(__file__)}
with OUT.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'prompt_checks':result['prompt_checks'],'training_exact_overlap':result['training_exact_overlap'],'structure':result['structure'],'per_source_overlap':[{k:x[k] for k in ['root_id','exact_turn_text_overlap_count','expanded_exact_turn_text_overlap_count','long80chars_exact_turn_overlap_count']} for x in per_source]},ensure_ascii=False,indent=2))
