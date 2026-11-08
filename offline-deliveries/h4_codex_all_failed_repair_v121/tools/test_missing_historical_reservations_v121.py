"""Previously source-held roots have no historical matching/reservations."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import prepare_v120 as p


class MissingHistory(unittest.TestCase):
    def test_zero_retained_b_needs_no_old_reservations(self):
        self.assertEqual(p.historical_b({'unit_path': '/nonexistent/SYNTHETIC_ONLY', 'retained_scene_refs': []}), [])

    def test_retained_a_only_needs_no_old_reservations(self):
        self.assertEqual(p.historical_b({'unit_path': '/nonexistent/SYNTHETIC_ONLY', 'retained_scene_refs': list(p.SCENES[:2])}), [])

    def test_retained_b_still_requires_real_historical_reservation(self):
        with TemporaryDirectory() as d:
            unit = Path(d) / 'new'; unit.mkdir()
            (unit / 'CONSTRAINTS_PATH_ADAPTATION.json').write_text(json.dumps({'original_binding': {'path': str(Path(d) / 'old/inputs/CONTINUATION_CONSTRAINTS.json')}}))
            with self.assertRaises(FileNotFoundError):
                p.historical_b({'unit_path': str(unit), 'retained_scene_refs': list(p.SCENES[:4])})


if __name__ == '__main__': unittest.main(verbosity=2)
