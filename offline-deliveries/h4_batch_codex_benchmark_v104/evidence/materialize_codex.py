import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    for short in ('14304','20432','24719','38601'):
        folder=BASE/'roots'/f'root{short}'
        draft=folder/'EXPANSION_DRAFT.json'
        out=folder/'CUMULATIVE_SOURCE.json'
        if not draft.exists() or out.exists() or (folder/'STRUCTURAL_FAILURE.json').exists():continue
        try:
            old=read(BASE/'inputs'/f'root{short}_BASELINE.json');new=read(draft)
            turns=new['turns'];assert 12<=len(turns)<=16,'turn_count'
            assert new['root_id']==old['root_id'],'root'
            speakers={t['speaker'] for t in old['turns']}
            for i,t in enumerate(turns,1):
                assert set(t)=={'turn_ref','session_ref','scene_ref','speaker','text','origin'},'fields'
                assert t['turn_ref']==f'R02:B:S01:T{i:03d}','turn_ref'
                assert t['session_ref']==t['scene_ref']=='R02:B:S01','scene'
                assert t['speaker'] in speakers,'speaker'
                assert isinstance(t['text'],str) and t['text'].strip(),'text'
                assert t['origin']=='codex_session_generated','origin'
            full=old['turns']+turns
            assert len({t['turn_ref'] for t in full})==len(full),'duplicate_refs'
            save(out,{'schema_version':'codex_session_cumulative_source_v1','root_id':old['root_id'],'baseline_sha256':sha(BASE/'inputs'/f'root{short}_BASELINE.json'),'draft_sha256':sha(draft),'baseline_turn_count':len(old['turns']),'new_turn_count':len(turns),'turns':full})
            save(folder/'MATERIALIZATION_RECEIPT.json',{'status':'STRUCTURAL_PASS_ONLY_NOT_SEMANTIC_READY','created_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(out),'prefix_exact':full[:len(old['turns'])]==old['turns'],'draft_sha256':sha(draft)})
            print(short,'PASS',len(turns))
        except Exception as e:
            save(folder/'STRUCTURAL_FAILURE.json',{'status':'FAIL','error':str(e),'created_at_utc':datetime.now(timezone.utc).isoformat()})
            print(short,'FAIL',str(e))
if __name__=='__main__':main()
