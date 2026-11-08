import ast,collections,datetime,hashlib,json,pathlib
REPO=pathlib.Path('/root/autodl-tmp/gxy-selfplay');V=REPO/'offline-deliveries/h4_codex_dev_eval_v127'
paths={k:pathlib.Path('/tmp')/name for k,name in {'script':'v127_public_ordering_baseline.py','input':'v127_public_ordering_baseline_input.json','predictions':'v127_public_ordering_baseline_predictions.json','scores':'v127_public_ordering_baseline_scores.json','analysis':'v127_ceiling_difficulty_analysis.py','findings':'v127_ceiling_difficulty_findings.json','independent_replay':'v127_independent_public_baseline_replay.json','replay_script':'v127_baseline_boundary_prediction_replay.py'}.items()}
tracked={}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):
 p=pathlib.Path(p);tracked[str(p)]=sha(p);return json.loads(p.read_text())
for p in paths.values():tracked[str(p)]=sha(p)
public=read(paths['input']);pred=read(paths['predictions']);replay=read(paths['independent_replay']);scores=read(paths['scores'])
assert pred['input_sha256']==sha(paths['input']) and pred['baseline_script_sha256']==sha(paths['script'])
assert replay['predictions']==pred['predictions'] and len(pred['predictions'])==504
assert replay['dict_insertion_order_invariance_pass']==replay['arbitrary_label_equivariance_pass']==504
assert replay['candidate_and_root_ids_not_passed_to_retrieval']
manifest=read(V/'run/new/manifest.json');gold=read(V/'run/new/gold.json')
requests={(r['candidate_id'],r['variant']['variant_id']):r for r in manifest['requests'] if r['question_type']=='ordering'}
qmap={q['candidate_id']:q for q in read(V/'EVALUATION_POOL.json')['questions'] if q['question_type']=='ordering'}
assert len(public['questions'])==len(qmap)==14
assert {q['candidate_id'] for q in public['questions']}==set(qmap)
for q in public['questions']:
 assert set(q)=={'candidate_id','root_id','question','turns','variants'}
 canonical=qmap[q['candidate_id']]
 assert q['root_id']==canonical['root_id'] and q['question']==canonical['question']
 source=read(V/'run/new/sources'/f"{q['root_id']}.json")
 assert q['turns']==[{k:t[k] for k in ['turn_ref','speaker','text']} for t in source['turns']]
 assert len(q['variants'])==12 and len({v['variant_id'] for v in q['variants']})==12
 for v in q['variants']:
  assert set(v)=={'variant_id','items'}
  req=requests[q['candidate_id'],v['variant_id']]
  assert v['items']==req['variant']['items']
# Static read boundary of the predictor: only the allowlisted INPUT and its own
# source for hashing; no imports of model/scoring/project modules or reflection.
tree=ast.parse(paths['script'].read_text())
imports=[];file_reads=[];sensitive_refs=[]
for node in ast.walk(tree):
 if isinstance(node,(ast.Import,ast.ImportFrom)):imports.append(ast.unparse(node))
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'read_text','read_bytes','open'}:file_reads.append({'line':node.lineno,'call':ast.unparse(node)})
 if isinstance(node,ast.Name) and node.id in {'gold','gold_content_ids','order_anchor_spans','item_evidence','canonical_order_content_ids','display_to_content','eval','exec','__import__'}:sensitive_refs.append({'line':node.lineno,'name':node.id})
assert not sensitive_refs
assert {x['call'] for x in file_reads}=={'INPUT.read_text()','INPUT.read_bytes()','Path(__file__).read_bytes()','OUTPUT.read_bytes()'}
assert all(not any(bad in x for bad in ['torch','transformers','scoring','protocol','solver','vllm']) for x in imports)
counts=collections.defaultdict(collections.Counter)
perq=collections.defaultdict(lambda:collections.defaultdict(list))
for row in replay['predictions']:
 req=requests[row['candidate_id'],row['variant_id']]
 content=[req['variant']['display_to_content'][label] for label in row['prediction']]
 target=gold[row['candidate_id']]
 correct=content==target
 ranks=[target.index(x) for x in content]
 inv=sum(ranks[i]>ranks[j] for i in range(6) for j in range(i+1,6))
 counts[row['method']]['calls']+=1;counts[row['method']]['exact_calls']+=correct;counts[row['method']]['wrong_pairs']+=inv
 perq[row['method']][row['candidate_id']].append(correct)
for method,values in counts.items():
 assert values['calls']==168
 expected=scores['method_summaries'][method]
 allcorrect=sum(all(v) for v in perq[method].values())
 assert allcorrect==expected['correct_questions_all_displays']
 assert values['exact_calls']/168==expected['exact_rate_equal_question']
 values['all_twelve_exact_questions']=allcorrect
assert counts['plain_tfidf_cosine']['all_twelve_exact_questions']==12
assert counts['stemmed_tfidf_cosine_actor']['all_twelve_exact_questions']==14
assert counts['stemmed_bm25_actor']['all_twelve_exact_questions']==14
changed=[p for p,v in tracked.items() if sha(p)!=v];assert not changed
result={'status':'PASS','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'/root/repair_v122_audit',
 'scope':'Independent CPU read-only baseline input-boundary review and replay;v127 frozen files unchanged;no model/no TEST content.',
 'checks':{'exact_public_allowlist_schema':True,'public_source_turns_exact_projection':True,'variants_exactly_actual_rebound_display_items':True,'no_gold_or_anchor_or_content_map_in_public_input':True,'no_canonical_item_labels_in_retrieval_input':True,'candidate_root_variant_identifiers_ignored_by_retrieval':True,'only_declared_public_input_and_own_script_read_by_predictor':True,'independent_gold_free_prediction_process_completed_before_separate_gold_scoring':True,'original504_predictions_exactly_reproduced':True,'all504_predictions_invariant_to_item_dictionary_insertion_order':True,'all504_predictions_equivariant_under_arbitrary_label_renaming':True,'score_recomputation_matches_report':True},
 'independently_recomputed_results':{k:dict(v) for k,v in counts.items()},'prediction_process_imports':imports,'prediction_process_file_reads':file_reads,
 'independent_prediction_binding':{'path':str(paths['independent_replay']),'sha256':sha(paths['independent_replay'])},
 'conclusion':'The lexical14/14 result is reproducible from public dialogue and shuffled items without Gold,review anchors,canonical I-order or model inference. The14-task ceiling is supported by a gold-free retrieval algorithm,not merely a private-annotation ordering oracle.',
 'limitations':['This is an exploratory diagnostic on the existing dev set,not a newly heldout accuracy estimate.','Static review and replay establish the executable input boundary;they cannot certify historical human choice of every heuristic or rule. All three declared methods are retained.','The method reads dialogue and named scenes;it is not a no-dialogue commonsense-only test.'],
 'inputs_changed_during_review':changed,'audited_files':[{'path':p,'sha256':v} for p,v in sorted(tracked.items())]}
out=pathlib.Path('/tmp/v127_lexical_boundary_review.json')
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':'PASS','output':str(out),'sha256':sha(out),'results':result['independently_recomputed_results']},ensure_ascii=False,indent=2))
