"""Replay baseline from public allowlist only; no answer/source-file access."""
import ast,collections,hashlib,json,math,pathlib,re
script=pathlib.Path('/tmp/v127_public_ordering_baseline.py')
inputfile=pathlib.Path('/tmp/v127_public_ordering_baseline_input.json')
output=pathlib.Path('/tmp/v127_independent_public_baseline_replay.json')
d=json.loads(inputfile.read_text())
assert set(d)=={'questions','prohibited_inputs_absent'}
assert len(d['questions'])==14
for q in d['questions']:
 assert set(q)=={'candidate_id','root_id','question','turns','variants'},set(q)
 assert all(set(t)=={'turn_ref','speaker','text'} for t in q['turns'])
 assert len(q['variants'])==12
 assert all(set(v)=={'variant_id','items'} for v in q['variants'])
 assert all(set(v['items'])=={'I1','I2','I3','I4','I5','I6'} for v in q['variants'])
source=script.read_text();tree=ast.parse(source)
kept=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'STOP','IRREGULAR'} for t in n.targets)]
namespace={'math':math,'re':re,'Counter':collections.Counter}
exec(compile(ast.fix_missing_locations(ast.Module(body=kept,type_ignores=[])),str(script),'exec'),namespace)
methods=['plain_tfidf_cosine','stemmed_tfidf_cosine_actor','stemmed_bm25_actor']
results=[];insertion_agnostic=0;arbitrary_label_equivariance=0
for q in d['questions']:
 # IDs are omitted from the object passed to retrieve(), so ID names cannot influence predictions.
 publicq={k:q[k] for k in ['question','turns']}
 for v in q['variants']:
  for method in methods:
   r=namespace['retrieve'](publicq,{'items':v['items']},method)
   reversed_r=namespace['retrieve'](publicq,{'items':dict(reversed(list(v['items'].items())))},method)
   assert reversed_r['prediction']==r['prediction'];insertion_agnostic+=1
   renamed={label:f'X{7-i}' for i,label in enumerate(v['items'])}
   rv={'items':{renamed[k]:value for k,value in v['items'].items()}}
   rr=namespace['retrieve'](publicq,rv,method)
   assert rr['prediction']==[renamed[k] for k in r['prediction']];arbitrary_label_equivariance+=1
   results.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'variant_id':v['variant_id'],'method':method,**r})
result={'status':'PREDICTED_WITHOUT_GOLD','scope':'PUBLIC_ONLY_CPU_NO_MODEL','script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'public_input_sha256':hashlib.sha256(inputfile.read_bytes()).hexdigest(),'public_input_exact_schema_pass':True,'candidate_and_root_ids_not_passed_to_retrieval':True,'dict_insertion_order_invariance_pass':insertion_agnostic,'arbitrary_label_equivariance_pass':arbitrary_label_equivariance,'predictions':results}
with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'saved_prediction_file':str(output),'prediction_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'rows':len(results),'all_boundary_checks_passed':True}))
