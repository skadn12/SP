"""Preserve prior derived decisions before a final independent-audit adjudication."""
import json
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parents[1]
archive=BASE/'evidence/attempt02_before_final_actor_adjudication'
assert not archive.exists()
assert json.loads((BASE/'FINAL_BATCH_ACCOUNTING.json').read_text())['content_passed_count']==23
names=['FINAL_BATCH_ACCOUNTING.json','ALL_OUTCOMES_INDEX.json','SELECTED_PUBLIC_QUESTIONS.json','SELECTED_PRIVATE_GOLD.json','CONTROL_PUBLIC_QUESTIONS.json','CONTROL_PRIVATE_GOLD.json','NEW_CONTENT_CANDIDATES.json','REJECTED_CANDIDATES.json','CUMULATIVE_DIALOGUES.txt','NEW_DIALOGUES.txt','COMPLETE_RESULTS_ZH.md','evidence/ORDERING_PROTOCOL_AUDIT.json','evidence/HISTORICAL_PRESERVATION_AFTER.json']
names += ['roots/root'+s+'/FINAL_CROSSCHECK.json' for s in ['14304','20404','24719','38601']]
names=[s.replace('root20404','root20432') for s in names]
for name in names:
    source=BASE/name;assert source.is_file(),str(source)
    target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
p=BASE/'evidence/FINAL_ACTOR_ADJUDICATION_RECEIPT.json';assert not p.exists()
p.write_text(json.dumps({'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_id':'v104-root20432-ordering01','trigger':'Independent final-delivery reviewer flagged a personal-agent inference in I4.','final_decision':'Exclude conservatively for unresolved actor attribution under the existing whole-proposition rule.','original_source_judgment_preserved':True,'source_question_gold_unchanged':True,'rule_changed':False,'prior_derived_outputs_archive':str(archive),'note':'This is a final content adjudication, not a new generation attempt, source substitution, or answer repair.'},ensure_ascii=False,indent=2)+'\n')
print('Prior derived outputs retained; immutable artifacts unchanged.')
