"""New-root scope and budget changes must fail before any generation preparation."""
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import batch,source_subset,source_admission
from common import HERE,read
class GateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.config=read(HERE/'CONFIG.json');cls.summary=source_subset.verified_summary()
 def test_actual_four_ready_roots_and_bounded_targets(self):
  c=batch.config();self.assertEqual(len(c['ordering_slots']),4);self.assertEqual((c['max_generation_http'],c['max_candidates'],c['max_solver_calls_per_model']),(12,28,456))
  self.assertEqual(c['multiselect_config']['new_child_target_distribution'],{'2':4,'3':4,'4':4,'5':4})
 def blocked(self,c):
  with tempfile.TemporaryDirectory() as d:
   h=Path(d);(h/'CONFIG.json').write_text(__import__('json').dumps(c))
   with patch.object(batch,'load_multi') as m,patch.object(batch.ordering_generation,'prepare') as o:
    with self.assertRaises(ValueError):batch.prepare(h)
    m.assert_not_called();o.assert_not_called();self.assertFalse((h/'BATCH_PREPARATION.STARTED.json').exists())
 def test_held_selected_source_rejected(self):
  s=deepcopy(self.summary);s['roots'][0]['decision']='HOLD'
  with patch.object(source_admission,'verify',return_value=s):self.blocked(self.config)
 def test_source_binding_changed_rejected(self):
  s=deepcopy(self.summary);s['roots'][0]['source']['sha256']='0'*64
  with patch.object(source_admission,'verify',return_value=s):self.blocked(self.config)
 def test_root_budget_gold_and_training_mutations_rejected(self):
  changes=[lambda c:c.update(max_generation_http=13),lambda c:c.update(training_allowed=True),
   lambda c:c['multiselect_config']['slots'][0].update(target_counts=[2,3,4,6]),
   lambda c:c['ordering_slots'][0].update(root_id='cc-train-episode-231899'),
   lambda c:c['source_consistency'].update(excluded_held_roots=[]),
   lambda c:c['multiselect_config'].update(selection_rule='all_questions')]
  for change in changes:
   c=deepcopy(self.config);change(c);self.blocked(c)
 def test_upstream_pin_change_rejected_before_generation(self):
  original=source_subset.sha
  with patch.object(source_subset,'sha',side_effect=lambda p:'0'*64 if str(p).endswith('CONFIG_TEMPLATE.json') else original(p)):
   self.blocked(self.config)
if __name__=='__main__':unittest.main(verbosity=2)
