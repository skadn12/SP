"""Readable finalized source/questions plus all exact native evidence JSON."""
from pathlib import Path
import argparse,importlib.util,json
B=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def export(root):
 short=root.rsplit('-',1)[-1];folder=B/'units'/('root'+short)/'roots'/('root'+short)
 cross=read(folder/'FINAL_CROSSCHECK.json');passed={q['candidate_id']for q in cross['results']if q['status']=='CONTENT_PASS'}
 source=read(folder/'CUMULATIVE_SOURCE.json');public=read(folder/'PUBLIC_QUESTIONS.json')['questions'];gold={q['candidate_id']:q for q in read(folder/'PRIVATE_GOLD.json')['questions']}
 lines=[f'# root{short}：完整对话与题目','',f'共 {len(source["turns"])} 条累计发言；{len(passed)} 题 CONTENT_PASS。Gold6 为对照；尚未因本文件导出而获得训练资格。','','## 完整对话','']
 prior=None
 for t in source['turns']:
  scene=t.get('scene_ref')or t.get('session_ref')or'原始对话'
  if scene!=prior:lines+=['',f'### {scene}',''];prior=scene
  lines += [f'**{t["turn_ref"]} — {t["speaker"]}**：{t["text"]}','']
 lines+=['## 题目与答案','']
 for i,q in enumerate(public,1):
  cid=q['candidate_id'];g=gold.get(cid);status='CONTENT_PASS'if cid in passed else'未通过'
  lines +=[f'### 题 {i}：{q["question_type"]}（{status}）','',q['question'],'']
  for k,v in q.get('options',q.get('items',{})).items():lines +=[f'- **{k}**：{v}']
  lines+=['', '**答案**：'+(' → 'if q['question_type']=='ordering'else'、').join(g['gold'])if g else'无冻结答案','',f'候选 ID：`{cid}`','']
  if g:lines+=['证据：','', '```json',json.dumps(g.get('evidence'),ensure_ascii=False,indent=2),'```','']
 target=B/'delivery'/f'root{short}_DIALOGUE_AND_QUESTIONS.md'
 with target.open('x')as f:f.write('\n'.join(lines)+'\n')
 print(str(target))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();export(a.root)
