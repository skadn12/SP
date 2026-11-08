import collections,itertools,json,pathlib,re,statistics
P=pathlib.Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_codex_dev_eval_v127')
def load(p):return json.loads(p.read_text())
qs={q['candidate_id']:q for q in load(P/'EVALUATION_POOL.json')['questions'] if q['question_type']=='ordering'}
manifest=load(P/'run/baseline/manifest.json'); gold=load(P/'run/baseline/gold.json')
rows=[]
for req in manifest['requests']:
 if req['question_type']!='ordering':continue
 cid=req['candidate_id'];q=qs[cid];resp=load(P/'run/baseline/responses'/f"{req['request_id']}.json");v=req['variant'];s=resp['score']
 contents_to_item={c:next(i for i,t in q['items'].items() if t==v['content_to_text'][c]) for c in gold[cid]}
 scene={c:q['order_anchor_spans'][i]['turn_ref'].rsplit(':T',1)[0] for c,i in contents_to_item.items()}
 route={c:sc.split(':')[1] for c,sc in scene.items()}
 assert [contents_to_item[c] for c in gold[cid]]==q['gold']
 pred=s['content_prediction'];dp=s['display_prediction'];assert [v['display_to_content'][x] for x in dp]==pred
 assert pred and len(pred)==6 and len(set(pred))==6 and set(pred)==set(gold[cid])
 rank={c:n+1 for n,c in enumerate(gold[cid])};pos={c:n for n,c in enumerate(pred)};pair_details=[]
 for a,b in itertools.combinations(gold[cid],2):
  category=route[a]+'_'+route[b] if route[a]!=route[b] else route[a]+'_internal'
  pair_details.append({'gold_pair':[rank[a],rank[b]],'category':category,'correct':pos[a]<pos[b]})
 nwrong=sum(not p['correct'] for p in pair_details);assert s['tp']==15-nwrong and s['fp']==nwrong
 rr={'candidate_id':cid,'root_id':q['root_id'],'variant_id':v['variant_id'],'request_id':req['request_id'], 'valid':resp['evaluation_valid'],'parse_success':s['parse_status']=='success','six_unique_complete':len(pred)==6 and len(set(pred))==6,'raw_output':resp['raw_output'],'display_prediction':dp,'rank_prediction':[rank[c] for c in pred],'canonical_prediction':[contents_to_item[c] for c in pred],'scene_prediction':[scene[c] for c in pred],'route_prediction':''.join(route[c] for c in pred),'gold_display':[next(k for k,z in v['display_to_content'].items() if z==c) for c in gold[cid]],'gold_scenes':[scene[c] for c in gold[cid]],'wrong_pairs':nwrong,'pair_details':pair_details,'first_correct':pred[0]==gold[cid][0],'last_correct':pred[-1]==gold[cid][-1], 'both_route_internals_correct':all(p['correct'] for p in pair_details if p['category'].endswith('_internal')),'route_blocks_correct':all(p['correct'] for p in pair_details if not p['category'].endswith('_internal'))}
 rows.append(rr)

def stats(rs):
 pc=collections.defaultdict(lambda:[0,0]);dist=collections.Counter();rp=collections.Counter();dsp=collections.Counter();rankp=collections.Counter()
 for r in rs:
  dist[r['wrong_pairs']]+=1;rp[r['route_prediction']]+=1;dsp[','.join(r['display_prediction'])]+=1;rankp[','.join(map(str,r['rank_prediction']))]+=1
  for p in r['pair_details']:
   pc[p['category']][0]+=p['correct'];pc[p['category']][1]+=1
 result={'calls':len(rs),'exact_correct':sum(r['wrong_pairs']==0 for r in rs),'exact_rate':sum(r['wrong_pairs']==0 for r in rs)/len(rs),'valid_calls':sum(r['valid'] for r in rs),'parse_success_calls':sum(r['parse_success'] for r in rs),'six_unique_complete_calls':sum(r['six_unique_complete'] for r in rs),'correct_pairs':sum(15-r['wrong_pairs'] for r in rs),'total_pairs':15*len(rs),'pair_accuracy':sum(15-r['wrong_pairs'] for r in rs)/(15*len(rs)),'wrong_pairs_distribution':dict(sorted(dist.items())),'pair_category_stats':{k:{'correct':n,'total':d,'accuracy':n/d} for k,(n,d) in pc.items()},'first_correct':sum(r['first_correct'] for r in rs),'last_correct':sum(r['last_correct'] for r in rs),'both_route_internals_all_correct_calls':sum(r['both_route_internals_correct'] for r in rs),'all_cross_route_pairs_correct_calls':sum(r['route_blocks_correct'] for r in rs),'route_patterns':dict(rp.most_common()),'display_patterns':dict(dsp.most_common()),'gold_rank_patterns':dict(rankp.most_common()),'unique_display_sequences':len(dsp),'unique_content_rank_sequences':len(rankp),'canonical_display_identity_calls':dsp.get('I1,I2,I3,I4,I5,I6',0)}
 return result
out={'status':'READONLY_DIAGNOSTIC_COMPLETE','scope':'v127 baseline ordering DEV only; uses existing canonical, manifest, raw response scores. No inference. No TEST. No upstream writes.','overall':stats(rows),'per_question':{cid:stats([r for r in rows if r['candidate_id']==cid]) for cid in qs},'per_call':rows,'cautions':['Error pattern quantifies observed output only; it does not establish internal reasoning cause.','Structured choice decoding guaranteed six unique labels, so valid output format was enforced and is not independent evidence of free generation formatting skill.','A_internal/B_internal are three within-route scene-pair relations each per call; A_B are nine cross-route scene relations per call.','gold_rank_prediction indexes chronology positions 1..6, not original display label.']}
pathlib.Path('/tmp/v129_baseline_error_patterns.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out['overall'],ensure_ascii=False,indent=2))
print('PERQ')
for cid,v in out['per_question'].items():print(cid,v['exact_correct'],v['pair_accuracy'],v['wrong_pairs_distribution'],v['route_patterns'],v['unique_content_rank_sequences'])
