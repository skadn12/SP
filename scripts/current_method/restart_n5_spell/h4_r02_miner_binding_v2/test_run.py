"""No-network frozen implementation/input/config integrity tests."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from ..core import atomic_json
from . import run as r
from .runtime import GuardrailStop

class FreezeTests(unittest.TestCase):
    def test_input_code_config_source_and_preflight_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp); artifacts={}
            for name in ('input','code','source','config','preflight'):
                p=run/name;p.write_text(name);artifacts[name]=p
            frozen={'version':r.VERSION,'root_order':list(r.FIRST_ROOTS),
                'code':[r.record(artifacts['code'])],'inputs':[r.record(artifacts['input'])],
                'sources':[r.record(artifacts['source'])],'config':r.record(artifacts['config']),
                'preflight':r.record(artifacts['preflight'])}
            atomic_json(run/'audit/frozen_manifest.json',frozen)
            with patch.object(r,'code_identity',return_value=frozen['code']):
                self.assertEqual(r.verify_frozen(run),frozen)
                for name,p in artifacts.items():
                    original=p.read_bytes();p.write_text('changed')
                    with self.subTest(name=name),self.assertRaises(GuardrailStop):r.verify_frozen(run)
                    p.write_bytes(original)
            with patch.object(r,'code_identity',return_value=frozen['code']+[{'path':'extra','sha256':'extra'}]):
                with self.assertRaises(GuardrailStop):r.verify_frozen(run)

    def test_root_scope_and_version_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            for version,roots in [('old',list(r.FIRST_ROOTS)),(r.VERSION,list(reversed(r.FIRST_ROOTS)))]:
                atomic_json(run/'audit/frozen_manifest.json',{'version':version,'root_order':roots})
                with self.assertRaises(GuardrailStop):r.verify_frozen(run)

if __name__=='__main__':unittest.main()
