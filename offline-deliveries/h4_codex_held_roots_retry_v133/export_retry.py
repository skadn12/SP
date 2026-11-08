"""Export immutable retry sources and readable dialogues; no content generation."""
from pathlib import Path
import sys,json
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
import retry_workflow as w
from verify_retry import audit

def export():
 report=audit();assert report['status']=='PASS',report['errors']
 assert all(r['status'] in ['EXPANSION_COMPLETE','EXPANSION_STOPPED'] for r in report['rows']),'INCOMPLETE_WORK_REMAINS'
 for r in report['rows']:
  rid=r['root_id'];u=w.unit(rid);base=w.baseline(rid)
  source=base
  for scene in w.SCENES:
   p=w.native.scene_folder(u,rid,scene)/'CUMULATIVE_SOURCE.json'
   if p.exists():source=w.read(p)
  turns=source['turns'];new=turns[len(base['turns']):]
  d=B/'delivery'/r['split'];d.mkdir(parents=True,exist_ok=True)
  w.save(d/'dialogues'/(rid+'.json'),source)
  w.save(d/'new_turns'/(rid+'.json'),{'root_id':rid,'status':r['status'],'baseline_sha256':w.sha(u/'inputs/BASELINE.json'),'turns':new,'baseline_conflicts_resolved':False})
  plan=w.read(u/'SAFE_PLAN.json')
  text=[f'# {rid}',f"\nSplit: {r['split']} | Status: {r['status']} | 新增 {len(new)} turn | 累计 {len(turns)} turn",'\n本文件包含完整旧对话和本次新增 A3+B3。原文矛盾仍然存在；本次只推进不依赖该矛盾的线程，不能把完整对话标记为无矛盾。当前训练池和评估池未修改。','\n## 本次隔离范围']
  text.extend('- '+c['summary'] for c in plan['quarantined_conflicts'])
  text.extend(['\n## A 路线规划']+['- '+x['scene_ref']+': '+x['goal'] for x in plan['a_scene_plan']])
  text.append('\n## 原始完整对话（保持原文）')
  for t in base['turns']:text.append(f"\n[{t['turn_ref']}] **{t['speaker']}**: {t['text']}")
  for scene in w.SCENES:
   text.append('\n## 新增 '+scene)
   scene_turns=[t for t in new if t['scene_ref']==scene]
   if not scene_turns:text.append('\n未生成（前置环节未通过）。')
   for t in scene_turns:text.append(f"\n[{t['turn_ref']}] **{t['speaker']}**: {t['text']}")
  p=d/'readable'/(rid+'.md');p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('x') as f:f.write('\n'.join(text)+'\n')
 w.save(B/'FINAL_STATUS.json',report)
 w.save(B/'ROOT_INDEX.json',{'roots':[{**r,'readable_path':f"delivery/{r['split']}/readable/{r['root_id']}.md",'dialogue_path':f"delivery/{r['split']}/dialogues/{r['root_id']}.json",'new_turns_path':f"delivery/{r['split']}/new_turns/{r['root_id']}.json"} for r in report['rows']]})
 lines=['# 七个暂停 root 扩充重试（v133）','',f"固定 {len(report['rows'])} roots，六场审核全部通过 {report['complete_roots']}，已通过场景 {report['ready_scenes']}。",'','原有 v124/v132 全部保留。本次改变为冲突隔离后续写其他真实旧线程；A三场可以是安全旧线程的三阶段，与原 v124 的 R01 对应要求不同。历史矛盾没有被修复，不能以此声称整段数据无矛盾。','', '| Root | Split | 六场通过 | 新增turn | 总turn |','|---|---|---:|---:|---:|']
 for r in report['rows']:lines.append(f"| [{r['root_id']}](delivery/{r['split']}/readable/{r['root_id']}.md) | {r['split']} | {r['ready_scenes']}/6 | {r['new_turns']} | {r['total_turns']} |")
 lines.extend(['','每条对话均有 A/B 两条路线，每条三场，每场12–16 turn。B使用互不重复且未被此前消耗的原生 MultiWOZ TRAIN事件。每场首稿和实际非作者审核、精确引用、来源预留、读取与冻结记录保存在 units/。','', '本批只扩充，未出题、未运行 Solver、未训练、未修改当前评估池。后续如出题，只能从独立通过且不依赖隔离冲突的新内容取材。不能直接替换既有冻结评估结果。','', '作者和审核者是同一会话模型的不同代理，共享文件系统，不能视为外部或严格盲审。TEST已有历史曝光，本次扩充属于新增实验材料。','', '入口：[协议](PROTOCOL_ZH.md)、[逐root索引](ROOT_INDEX.json)、[审核汇总](FINAL_STATUS.json)。'])
 with (B/'README_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps({'complete_roots':report['complete_roots'],'ready_scenes':report['ready_scenes'],'new_turns':sum(r['new_turns'] for r in report['rows'])},ensure_ascii=False))
if __name__=='__main__':export()
