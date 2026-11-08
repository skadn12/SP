"""Retain a finalizer checksum-definition bug before rerunning derived outputs."""
import json
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parents[1]
archive=BASE/'evidence/finalizer_attempt01_hash_encoding_bug'
assert not archive.exists()
account=json.loads((BASE/'FINAL_BATCH_ACCOUNTING.json').read_text())
assert account['failure_code_counts']['INPUT_RESERVATION_HASH_MISMATCH']==28
names=['FINAL_BATCH_ACCOUNTING.json','ALL_OUTCOMES_INDEX.json','SELECTED_PUBLIC_QUESTIONS.json','SELECTED_PRIVATE_GOLD.json','CONTROL_PUBLIC_QUESTIONS.json','CONTROL_PRIVATE_GOLD.json','NEW_CONTENT_CANDIDATES.json','REJECTED_CANDIDATES.json','CUMULATIVE_DIALOGUES.txt','NEW_DIALOGUES.txt','evidence/ORDERING_PROTOCOL_AUDIT.json','evidence/HISTORICAL_PRESERVATION_AFTER.json']
names += ['roots/root'+s+'/FINAL_CROSSCHECK.json' for s in ['14304','20432','24719','38601']]
for name in names:
    source=BASE/name
    assert source.is_file(),str(source)
    target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
    source.rename(target)
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'issue':'Initial finalizer compared event file-byte SHA with the canonical-JSON SHA recorded by setup_benchmark.py. All four inputs were semantically unchanged; baseline byte SHA and event canonical SHA match their original reservation.','fix':'Use file-byte SHA for baseline and canonical JSON SHA for source_item, exactly matching the reservation constructor. No reserved input, source, public question, review or original Gold changed.','archived_derived_outputs':names,'archive':str(archive),'not_a_generation_or_semantic_failure':True}
p=BASE/'evidence/FINALIZER_CORRECTION_RECEIPT.json';assert not p.exists();p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('Archived initial derived outputs; immutable source/question/review/Gold artifacts unchanged.')
