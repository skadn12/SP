from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import dependencies as d
from .runtime import GuardrailStop


class DependencyTests(unittest.TestCase):
    def test_recursive_imports_include_local_and_parent_modules(self):
        files,edges,_=d.static_closure([d.CODE/'run.py'])
        for name in ('backends.py','core.py','live_backends.py','simple_append_v1/live_runtime.py','ledger.py'):
            self.assertIn(d.CODE.parent/name,files)
        self.assertTrue(edges)

    def test_unfrozen_loaded_repository_file_stops(self):
        with patch.object(d,'loaded_repository_files',return_value={d.CODE/'runtime.py'}):
            with self.assertRaises(GuardrailStop):d.check_loaded_coverage([])

    def test_execution_guard_rejects_unlisted_repository_function(self):
        with self.assertRaises(GuardrailStop):
            with d.repository_execution_guard([d.record(d.CODE/'dependencies.py')]):
                exec(compile('def forbidden(): return 1\nforbidden()\n',str(d.CODE/'unfrozen_fake.py'),'exec'),{})


if __name__=='__main__':unittest.main()
