import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import requests

W=Path(__file__).resolve().parent
URLS={
 'multirc_index.html':'https://cogcomp.seas.upenn.edu/multirc/',
 'multirc_v2.zip':'https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip',
 'multirc_README.md':'https://raw.githubusercontent.com/CogComp/multirc/master/README.md',
 'multirc_paper.pdf':'https://aclanthology.org/N18-1023.pdf',
 'dream_README.md':'https://raw.githubusercontent.com/nlpdata/dream/master/README.md',
 'dream_train.json':'https://raw.githubusercontent.com/nlpdata/dream/master/data/train.json',
 'dream_dev.json':'https://raw.githubusercontent.com/nlpdata/dream/master/data/dev.json',
 'dream_annotator1_dev.json':'https://raw.githubusercontent.com/nlpdata/dream/master/annotation/annotator1_dev.json',
 'dream_annotator2_dev.json':'https://raw.githubusercontent.com/nlpdata/dream/master/annotation/annotator2_dev.json',
 'timedial_README.md':'https://raw.githubusercontent.com/google-research-datasets/TimeDial/main/README.md',
 'timedial_test.json':'https://raw.githubusercontent.com/google-research-datasets/TimeDial/main/test.json',
 'logiqa_README.md':'https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/README.md',
 'logiqa_Train.txt':'https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt',
 'logiqa_zh_train.txt':'https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/zh_train.txt',
 'logiqa_paper.pdf':'https://www.ijcai.org/Proceedings/2020/0501.pdf',
}
def fetch(item):
 name,url=item
 try:
  r=requests.get(url,timeout=50);r.raise_for_status();raw=r.content
  dest=W/'sources'/name
  with dest.open('xb') as f:f.write(raw)
  return {'name':name,'requested_url':url,'resolved_url':r.url,'status':r.status_code,'path':str(dest),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 except Exception as e:return {'name':name,'requested_url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 records=list(pool.map(fetch,URLS.items()))
with (W/'FETCH_MANIFEST.json').open('x') as f:json.dump(records,f,ensure_ascii=False,indent=2);f.write('\n')
for r in records:print(r['name'],r.get('status'),r.get('bytes'),r.get('error',''))
