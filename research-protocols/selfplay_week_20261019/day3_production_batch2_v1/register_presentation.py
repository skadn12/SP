"""Freeze a common display-only label permutation before any Solver/training.

This does not change facts, answer sets, source anchors, or author drafts.
The same inherited slot gets the same mapping in both arms. Labels are sorted
by SHA-256 of a fixed seed + slot ID + old label, without reading the answer.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
D=Path(__file__).resolve().parent
p=D/'PRESENTATION_TRANSFORM_FREEZE.json'
with p.open('x') as f:
    json.dump({
      'id':'DAY3_BATCH2_COMMON_PRESENTATION_V1',
      'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
      'motivation':'A writer uses one fixed ordering-label permutation; some MC drafts also have position patterns. Preserve initial drafts and remove deterministic displayed-answer patterns before use.',
      'seed':'gxy-day3-common-presentation-20261020-v1',
      'algorithm':'For each inherited slot_id, sort old labels A-F by sha256(seed + "|" + slot_id + "|" + old_label); assign that sequence to new display labels A-F. Both arms share slot-specific map. No answer, evidence or score enters mapping.',
      'variants':1,
      'scope':'Audited candidate display export only. Training augmentation and actual common evaluation require their own explicit request/pool freeze.',
      'source_drafts_modified':False,
      'facts_gold_membership_or_source_anchors_changed':False,
      'solver_calls_before_this_freeze':0,
      'training_runs_before_this_freeze':0,
      'acceptance_depends_on_mapping_or_scores':False,
      'does_not_guarantee_exact_finite_sample_label_balance':True,
      'required_validation':['Bijection A-F','Round-trip public texts and Gold','Same slot same map in both arms','Report displayed answer histogram','Hash-bind original and displayed artifacts']
    },f,ensure_ascii=False,indent=2)
    f.write('\n')
print(p)
