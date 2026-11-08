"""Check that incomplete family reviews cannot partially complete generation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generation_driver as driver


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class ReviewLifecycleTests(unittest.TestCase):
    def fixture(self, home, *, supplement=False):
        slots = []
        for n in (1, 2):
            run = home / 'assets' / f'family{n}'
            (run / 'rounds' / 'initial').mkdir(parents=True)
            save(run / 'rounds' / 'initial' / 'SOURCE_JUDGMENTS.json', {'complete': True})
            slots.append({'slot_id': f'family{n}', 'run': str(run)})
        if supplement:
            save(Path(slots[1]['run']) / 'rounds' / 'supplement' / 'SOURCE_JUDGMENTS.json', {'complete': True})
        save(home / 'EXECUTION_PLAN.json', {})
        return {'slots': slots, 'max_http': 4}

    def execute(self, home, plan):
        events = []

        def reviewed(run, rid):
            # This fake review gate raises on the same unavailable-record case
            # as the real gate; schema/content cases have separate gate tests.
            judgment = driver.assets.read(Path(run) / 'rounds' / rid / 'SOURCE_JUDGMENTS.json')
            if judgment != {'complete': True}:
                raise ValueError('incomplete_review')
            events.append(('review', Path(run).name, rid))

        def finalize_family(run):
            run = Path(run)
            events.append(('finalize', run.name))
            value = {'slot_id': run.name, 'family_id': run.name, 'root_id': run.name,
                     'status': 'COMPLETE', 'fact_count': 6, 'counterversion_count': 4,
                     'http_attempts': 1, 'requests_prepared': 1, 'bounded_supplements': 0}
            save(run / 'GENERATION.COMPLETE.json', value)
            return value

        patches = [patch.object(driver, 'verify_plan', return_value=plan),
                   patch.object(driver.assets, 'reviewed_round', side_effect=reviewed),
                   patch.object(driver.assets, 'finalize', side_effect=finalize_family),
                   patch.object(driver.assets, 'verify', return_value=None)]
        for item in patches:
            item.start()
        try:
            result = driver.finalize(home)
        finally:
            for item in reversed(patches):
                item.stop()
        return result, events

    def assert_no_completion(self, home):
        self.assertFalse((home / 'GENERATION.COMPLETE.json').exists())
        self.assertEqual(list((home / 'assets').glob('*/GENERATION.COMPLETE.json')), [])

    def test_initial_only_and_reviewed_supplement_complete_after_all_reviews(self):
        for supplement in (False, True):
            with self.subTest(supplement=supplement), tempfile.TemporaryDirectory() as td:
                home = Path(td)
                plan = self.fixture(home, supplement=supplement)
                result, events = self.execute(home, plan)
                first_finalize = next(i for i, event in enumerate(events) if event[0] == 'finalize')
                self.assertEqual(first_finalize, 3 if supplement else 2)
                self.assertTrue(all(event[0] == 'review' for event in events[:first_finalize]))
                self.assertEqual(result['http_attempts'], 2)

    def test_later_family_missing_initial_review_creates_no_completions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            plan = self.fixture(home)
            (Path(plan['slots'][1]['run']) / 'rounds' / 'initial' / 'SOURCE_JUDGMENTS.json').unlink()
            with self.assertRaises(FileNotFoundError):
                self.execute(home, plan)
            self.assert_no_completion(home)

    def test_actual_supplement_without_review_creates_no_completions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            plan = self.fixture(home)
            (Path(plan['slots'][1]['run']) / 'rounds' / 'supplement').mkdir()
            with self.assertRaises(FileNotFoundError):
                self.execute(home, plan)
            self.assert_no_completion(home)

    def test_unexpected_actual_round_creates_no_completions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            plan = self.fixture(home)
            (Path(plan['slots'][1]['run']) / 'rounds' / 'extra').mkdir()
            with self.assertRaisesRegex(ValueError, 'unexpected_round_directory'):
                self.execute(home, plan)
            self.assert_no_completion(home)

    def test_missing_initial_round_creates_no_completions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            plan = self.fixture(home)
            initial = Path(plan['slots'][1]['run']) / 'rounds' / 'initial'
            (initial / 'SOURCE_JUDGMENTS.json').unlink()
            initial.rmdir()
            with self.assertRaisesRegex(ValueError, 'initial_round_missing'):
                self.execute(home, plan)
            self.assert_no_completion(home)

    def test_supplement_path_must_be_a_round_directory(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            plan = self.fixture(home)
            save(Path(plan['slots'][1]['run']) / 'rounds' / 'supplement', {})
            with self.assertRaisesRegex(ValueError, 'unexpected_round_directory'):
                self.execute(home, plan)
            self.assert_no_completion(home)


if __name__ == '__main__':
    unittest.main()
