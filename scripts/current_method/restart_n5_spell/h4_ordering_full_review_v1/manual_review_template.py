"""Create a non-overlapping manual-review worklist for remaining ordering items.

This intentionally creates blank adjudication rows. It never promotes the
machine triage output to a human decision.
"""
import csv, json
from pathlib import Path

BASE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_matched_training_v1_live_1')
OLD = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_ordering_candidate_review_v1_live_1/review_decisions_v2.tsv')
OUT = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_dev_test_ordering_review_v1_live_1/manual_remaining_review.tsv')

old = {r['candidate_id'] for r in csv.DictReader(OLD.open(), delimiter='\t')}
rows = []
for split in ('dev','test'):
    for line in (BASE / f'{split}_full_eval_manifest.jsonl').open():
        item = json.loads(line); c = item['candidate']
        if c.get('question_type') != 'ordering' or c['candidate_id'] in old:
            continue
        turns = json.loads(Path(item['dialogue_path']).read_text())['turns']
        text = {str(t['turn_ref']): str(t.get('text','')) for t in turns}
        rows.append({
            'split': split, 'candidate_id': c['candidate_id'], 'root_id': c['root_id'],
            'question': c['question'], 'gold': json.dumps(c.get('gold',[]), ensure_ascii=False),
            'items': json.dumps(c.get('items',{}), ensure_ascii=False, sort_keys=True),
            'evidence_refs': json.dumps(c.get('evidence_refs',[]), ensure_ascii=False),
            'evidence_text': json.dumps({ref: text.get(str(ref), '<MISSING>') for ref in c.get('evidence_refs',[])}, ensure_ascii=False),
            'dialogue_path': item['dialogue_path'],
            'human_gold_status': '', 'human_evidence_status': '', 'human_actual_order': '',
            'human_item_refs': '', 'human_note': '', 'reviewer': 'manual',
        })
rows.sort(key=lambda r: (r['split'], r['root_id'], r['candidate_id']))
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t'); writer.writeheader(); writer.writerows(rows)
print('wrote', OUT, 'rows', len(rows))
