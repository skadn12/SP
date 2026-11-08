import unittest
from .constants import THREE_RUN
from .dry_run import exercise
from .dependencies import forbid_network


class SharedSchedulerTests(unittest.TestCase):
    def test_original_entrypoint_four_cells_and_caught_failure_paths(self):
        with forbid_network():result=exercise(THREE_RUN)
        self.assertEqual(result['status'],'PASS',result['scenarios'])


if __name__=='__main__':unittest.main()
