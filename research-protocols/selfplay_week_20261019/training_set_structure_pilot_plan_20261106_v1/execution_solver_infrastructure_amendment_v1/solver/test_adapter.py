"""Safe static/evidence check only; never legacy fixture discovery or writes."""
import unittest
import adapter


class AmendmentStaticCheck(unittest.TestCase):
    def test_static_approval_and_prior_validation(self):
        result = adapter.safe_self_check()
        self.assertEqual(result['status'], 'SAFE_STATIC_SELF_CHECK_PASS')
        self.assertEqual((result['writes'], result['model_calls']), (0, 0))


if __name__ == '__main__':
    raise SystemExit('Use adapter.py self-check; no legacy experimental fixtures are executed.')
