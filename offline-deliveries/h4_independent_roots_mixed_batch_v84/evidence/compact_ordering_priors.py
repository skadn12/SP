"""Pre-prepare normalization preserves every historical whole-set exclusion."""
from pathlib import Path
import sys,json,re
HOME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HOME))
from common import read,write,sha,bind,now,need
need(not (HOME/'BATCH_PREPARATION.STARTED.json').exists(),'already_prepared')
config=read(HOME/'CONFIG.json');before=[bind(HOME/n) for n in ['CONFIG.json','CONFIG_TEMPLATE.json','source_subset.py']];records=[]
for slot in config['ordering_slots']:
 inputs=[bind(p) for p in slot['prior_pool_paths']];rows=[];bykey={};duplicates=[]
 for path in slot['prior_pool_paths']:
  for q in read(path)['questions']:
   if q['root_id']!=slot['root_id'] or q['question_type']!='ordering':continue
   key=tuple(sorted(' '.join(t.split()).casefold() for t in q['items'].values()))
   if key in bykey:duplicates.append({'candidate_id':q['candidate_id'],'input_path':path,'same_whole_set_as':bykey[key]})
   else:bykey[key]=q['candidate_id'];rows.append(q)
 p=HOME/'evidence'/('ORDERING_PRIORS_UNIQUE_'+slot['root_id'].split('-')[-1]+'.json')
 write(p,{'questions':rows,'inputs':inputs,'duplicates':duplicates,'rule':'Exact normalized six-statement sets, preserving first verbatim public row; no semantic merging.'})
 slot['prior_pool_paths']=[str(p)];records.append({'root_id':slot['root_id'],'unique_sets':len(rows),'duplicates':len(duplicates),'output':bind(p)})
for name in ['CONFIG.json','CONFIG_TEMPLATE.json']:(HOME/name).write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
p=HOME/'source_subset.py';text=p.read_text();old=next(r['sha256'] for r in before if r['path'].endswith('CONFIG_TEMPLATE.json'));text=text.replace(old,sha(HOME/'CONFIG_TEMPLATE.json'));p.write_text(text)
write(HOME/'evidence/ORDERING_PRIOR_COMPACTION.json',{'created_at_utc':now(),'before_preparation':True,'config_before':before,'config_after':[bind(HOME/n) for n in ['CONFIG.json','CONFIG_TEMPLATE.json','source_subset.py']],'records':records,'payload_change':'Public prior identity index in context plus every complete unique six-statement set once in task tail; source and feedback remain complete; no answer or semantic rule change.','code':bind(HOME/'ordering_generation.py')})
print(records)
