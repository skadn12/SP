"""Public-input lexical retrieval baseline. This program never opens Gold/evidence.

Input allowlist: candidate identifiers, public stem, displayed items, and source
turn ref/speaker/text. Three declared retrieval rules use fixed parameters, no
fitting. Scope follows only the scene identifiers printed in the public stem.
Predictions are saved before a separate process is allowed to open answer keys.
"""
import hashlib,json,math,re
from collections import Counter
from pathlib import Path
INPUT=Path('/tmp/v127_public_ordering_baseline_input.json')
OUTPUT=Path('/tmp/v127_public_ordering_baseline_predictions.json')
if OUTPUT.exists():raise RuntimeError('NO_OVERWRITE_PREDICTIONS')
d=json.loads(INPUT.read_text())
STOP=set('a an the i me my mine we us our ours you your yours he him his she her hers they them their theirs it its this that these those to of in on at for from with by as and or but if then than so very also just now already still only really please would could should will shall can may might must am is are was were be been being have has had do does did ive im ill id its thats theres reports report reported reporting says say said states state tells tell told recounts recount gives give asks ask'.split())
IRREGULAR={'sent':'send','sending':'send','wrote':'write','written':'write','writing':'write','gave':'give','given':'give','giving':'give','chosen':'choose','chose':'choose','choosing':'choose','bought':'buy','buying':'buy','putting':'put','put':'put','read':'read','reading':'read','paid':'pay','paying':'pay','left':'leave','leaving':'leave','took':'take','taken':'take','taking':'take','made':'make','making':'make','got':'get','getting':'get','kept':'keep','keeping':'keep','saw':'see','seen':'see','seeing':'see','went':'go','gone':'go','going':'go','ran':'run','running':'run','drew':'draw','drawn':'draw','drawing':'draw','worn':'wear','wearing':'wear','thought':'think','thinking':'think','brought':'bring','bringing':'bring'}
def stem(t):
 t=IRREGULAR.get(t,t)
 if t.endswith('ies') and len(t)>4:t=t[:-3]+'y'
 elif t.endswith('ing') and len(t)>5:t=t[:-3]
 elif t.endswith('ed') and len(t)>4:t=t[:-2]
 elif t.endswith('s') and len(t)>4 and not t.endswith(('ss','us')):t=t[:-1]
 if len(t)>4 and t[-1:]==t[-2:-1] and t[-1:] not in 'aeiou':t=t[:-1]
 if len(t)>4 and t.endswith('e'):t=t[:-1]
 return t
def tokens(s,stemming):
 ss=re.findall(r'[a-z0-9]+(?:[.:][a-z0-9]+)*',s.casefold().replace('’',"'"))
 return [stem(t) if stemming else t for t in ss if t not in STOP and len(t)>1]
def retrieve(question,variant,method):
 scene_refs=set(re.findall(r'R\d+:[A-Z]:S\d+',question['question']))
 turns=question['turns']
 indexed=[(i,t) for i,t in enumerate(turns) if not scene_refs or t['turn_ref'].rsplit(':T',1)[0] in scene_refs]
 stemming=method!='plain_tfidf_cosine';actor_aware=method!='plain_tfidf_cosine'
 docs=[Counter(tokens(t['text'],stemming)) for i,t in indexed]
 df=Counter(term for doc in docs for term in doc)
 N=len(docs);idf={term:math.log((N+1)/(freq+1))+1 for term,freq in df.items()}
 avglen=sum(sum(doc.values()) for doc in docs)/max(N,1)
 choices=[]
 for label,text in variant['items'].items():
  query=Counter(tokens(text,stemming));speaker=text.split(' ',1)[0]
  eligible=[j for j,(_,turn) in enumerate(indexed) if not actor_aware or turn['speaker']==speaker]
  if not eligible:eligible=list(range(N))
  scored=[]
  for j in eligible:
   doc=docs[j]
   if method=='stemmed_bm25_actor':
    score=0.0;dl=sum(doc.values())
    for term,tfq in query.items():
     freq=doc[term]
     if freq:
      weight=math.log(1+(N-df[term]+.5)/(df[term]+.5))
      score+=weight*freq*2.5/(freq+1.5*(.25+.75*dl/max(avglen,1)))
   else:
    query_weights={term:(1+math.log(freq))*idf.get(term,math.log(N+1)+1) for term,freq in query.items()}
    doc_weights={term:(1+math.log(freq))*idf[term] for term,freq in doc.items()}
    dot=sum(weight*doc_weights.get(term,0) for term,weight in query_weights.items())
    normq=math.sqrt(sum(weight*weight for weight in query_weights.values()));normd=math.sqrt(sum(weight*weight for weight in doc_weights.values()))
    score=dot/(normq*normd) if normq and normd else 0.0
   scored.append((score,indexed[j][0],j))
  scored.sort(key=lambda row:(-row[0],row[1]))
  first,second=scored[0],scored[1] if len(scored)>1 else None
  _,position,j=first;turn=indexed[j][1]
  choices.append({'display_label':label,'item':text,'predicted_turn_index':position,'predicted_turn_ref':turn['turn_ref'],
                  'predicted_speaker':turn['speaker'],'matched_turn_text':turn['text'],'score':first[0],
                  'margin_to_second':first[0]-second[0] if second else None,'second_turn_ref':indexed[second[2]][1]['turn_ref'] if second else None,
                  'zero_similarity':first[0]==0})
 choices.sort(key=lambda row:(row['predicted_turn_index'],row['item'].casefold()))
 return {'prediction':[row['display_label'] for row in choices],'matches':choices,'zero_similarity_items':sum(row['zero_similarity'] for row in choices),
         'distinct_predicted_turns':len({row['predicted_turn_index'] for row in choices})}
results=[]
for q in d['questions']:
 for v in q['variants']:
  for method in ['plain_tfidf_cosine','stemmed_tfidf_cosine_actor','stemmed_bm25_actor']:
   results.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'variant_id':v['variant_id'],'method':method,**retrieve(q,v,method)})
out={'scope':'PUBLIC_ONLY_NO_GOLD_OR_EVIDENCE','input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
     'baseline_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'methods':['plain_tfidf_cosine','stemmed_tfidf_cosine_actor','stemmed_bm25_actor'],
     'selection_or_parameter_fitting':False,'predictions':results}
OUTPUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PREDICTIONS_SAVED_BEFORE_SCORING','rows':len(results),'output':str(OUTPUT),'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}))
