"""Export evaluation delivery with canonical public labels; no content edits."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
B=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(B/'delivery_tools'))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--split',choices=['dev','test'],required=True);a=parser.parse_args()
 disclosure=json.loads((B/'evidence/POSTFREEZE_CANONICAL_EXPORT_IMPLEMENTATION.json').read_text())
 for item in disclosure['implementation_bindings']+[disclosure['frozen_original_exporter_binding'],disclosure['frozen_delivery_export_binding']]:
  assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
 path=B/'delivery_tools/codex_evidence_export_canonical.py'
 spec=importlib.util.spec_from_file_location('codex_evidence_export',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sys.modules['codex_evidence_export']=module
 import state_and_export
 state_and_export.export(a.split)
if __name__=='__main__':main()
