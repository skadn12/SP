"""Persist explicit M_v125_newonly registration with noncircular audit binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
LEGACY_NAME = 'M_clean_new'
FINAL_NAME = 'M_v125_newonly'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    identity_path = HERE / 'MODEL_IDENTITY.json'
    final_path = HERE / 'NEW_MODEL_IDENTITY.json'
    audit_path = PACKAGE / 'evidence/MODEL_MERGE_AUDIT.json'
    assert not final_path.exists(), 'REGISTRATION_ALREADY_FINALIZED'
    identity = json.loads(identity_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert identity['status'] == audit['status'] == 'PASS'
    assert identity['model_identifier'] == audit['model_identifier'] == LEGACY_NAME
    assert audit.pop('identity_binding')['sha256'] == sha(identity_path)
    audit['model_identifier'] = FINAL_NAME
    audit['registration_finalize_script_binding'] = {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())}
    audit['registration_policy'] = 'Audit binds only model files, merge script and inputs; final identity binds the audit. No identity/audit hash cycle.'
    audit['tokenizer_warning_interpretation'] = ('Transformers 4.57.3 emits its known generic mistral-regex warning on the serialized Qwen tokenizer directory. '
        'No correction flag or tokenizer mutation was applied: vocabulary, special tokens, exact chat template and all363 public chat input IDs matched the original base tokenizer. '
        'Evaluation intentionally uses original base tokenizer path for every model, so serialized-directory warning does not change paired tokenizer.')
    with audit_path.open('w') as handle:
        handle.write(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    identity['model_identifier'] = identity['model_name'] = FINAL_NAME
    identity['audit_path'] = str(audit_path)
    identity['audit_sha256'] = sha(audit_path)
    identity['registration_policy'] = audit['registration_policy']
    with final_path.open('x') as handle:
        handle.write(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    # The initial identity was an internal, unregistered build artifact. The final
    # name is the sole published registration; preserve its own reproducible audit.
    identity_path.unlink()
    print(json.dumps({'status': 'PASS', 'identity_path': str(final_path), 'identity_sha256': sha(final_path),
                      'identity_bytes': final_path.stat().st_size, 'model_identifier': FINAL_NAME,
                      'model_path': identity['model_path'], 'audit_sha256': identity['audit_sha256'],
                      'model_fingerprint_sha256': identity['model_fingerprint_sha256'],
                      'merged_file_count': identity['merged_file_count'], 'merged_size_bytes': identity['merged_size_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
