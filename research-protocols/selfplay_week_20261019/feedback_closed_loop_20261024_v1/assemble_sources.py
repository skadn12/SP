"""Append reviewed scenes; never alter parent turn objects or reserved inputs."""
import copy,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def read(p): return json.loads(Path(p).read_text())
def binding(p):
 p=Path(p).resolve(); b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(p,d):
 with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 rows=[]; audits=[]
 for root in read(P/'CONFIG.json')['roots']:
  inp=read(P/f'inputs/cc-train-episode-{root}_SOURCE_AUTHOR_INPUT.json')
  parentb=inp['parent_source_binding'];assert all(binding(parentb['path'])[k]==parentb[k] for k in ['path','sha256','bytes'])
  parent=read(parentb['path']);reviewp=P/f'source_review/root{root}/REVIEW.json';review=read(reviewp)
  verdict=review.get('decision',review.get('overall_verdict',review.get('overall_status')));assert verdict in ['PASS','READY','READY_WITH_WARNINGS'],(root,verdict)
  draftp=P/f'source_authoring/root{root}/SCENES_DRAFT.json'
  if root=='218519':
   draftp=draftp.with_name('SCENES_REVISION_01.json');patchp=P/'source_review/root218519/PATCH_RECHECK.json';patch=read(patchp)
   assert patch.get('decision',patch.get('overall_verdict',patch.get('overall_patch_verdict')))=='PASS',patch
  draft=read(draftp);assert draft['root_id']==parent['root_id']==inp['root_id']
  scenes=draft['scenes'];assert [s['scene_ref'] for s in scenes]==inp['scene_sequence']
  new=[]
  for scene in scenes:
   assert len(scene['turns'])==8
   assert sum(len(t['text'].split()) for t in scene['turns'])/8<=18
   for i,t in enumerate(scene['turns'],1):
    assert t['turn_ref']==f"{scene['scene_ref']}:T{i:03}"
    assert t['speaker'] in {x['speaker'] for x in parent['turns']}
    z=copy.deepcopy(t);z['scene_ref']=scene['scene_ref'];z['session_ref']=f"{parent['root_id']}:{scene['scene_ref']}";new.append(z)
  final=copy.deepcopy(parent)
  final.update(schema_version='feedback_closed_loop_r03_source_v1',source_status=inp['source_status'],scene_ref='R03:B:S03',input_source_sha256=parentb['sha256'],draft_sha256=binding(draftp)['sha256'],previous_turn_count=len(parent['turns']),new_turn_count=len(new),cumulative_new_turn_count=parent['cumulative_new_turn_count']+len(new),turns=parent['turns']+new)
  final['closed_loop_provenance']={'round':'R03','mode':'A_then_B_sequential_cumulative','parent_source_binding':parentb,'draft_binding':binding(draftp),'review_binding':binding(reviewp),'native_events':inp['native_events'],'warnings':review.get('warnings',[]),'status':'SOURCE_REVIEWED_BEFORE_QUESTION_AUTHORING','extension_purpose':'TEST_HYPOTHESIS_NOT_CONFIRMED_SOURCE_SHORTFALL'}
  if root=='218519':final['closed_loop_provenance']['patch_recheck_binding']=binding(patchp)
  assert final['turns'][:len(parent['turns'])]==parent['turns'];assert len(final['turns'])==len(parent['turns'])+48
  assert len({t['turn_ref'] for t in final['turns']})==len(final['turns'])
  assert final['baseline_turn_count']+final['cumulative_new_turn_count']==len(final['turns'])
  dest=P/f'sources/cc-train-episode-{root}.json';save(dest,final)
  reread=read(dest);assert reread==final and reread['turns'][:len(parent['turns'])]==parent['turns']
  rows.append(dict(root_id=parent['root_id'],**binding(dest)))
  audits.append({'root_id':parent['root_id'],'source_status':inp['source_status'],'parent_turns':len(parent['turns']),'appended_turns':48,'final_turns':len(final['turns']),'prefix_equal':True,'source_review_verdict':verdict,'draft_binding':binding(draftp),'review_binding':binding(reviewp),'source_binding':binding(dest),'warnings':review.get('warnings',[])})
 save(P/'SOURCE_SELECTION_FINAL.json',{'schema_version':'closed_loop_source_selection_v1','split':'train','roots':rows})
 save(P/'SOURCE_FREEZE.json',{'status':'PASS','roots':audits,'selection_binding':binding(P/'SOURCE_SELECTION_FINAL.json'),'training':False,'old_sources_modified':False})
 print(json.dumps({'status':'PASS','sources':len(rows),'appended_turns':192,'final_turns':[a['final_turns'] for a in audits]}))
if __name__=='__main__':main()
