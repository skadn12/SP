"""Fresh-only role adapter; no historical scene or QA import is permitted."""
from pathlib import Path
import json


def expected_author(batch, unit, root, phase, scene=None):
    batch, unit = Path(batch).resolve(), Path(unit).resolve()
    if unit != batch / 'units' / ('root' + root.rsplit('-', 1)[-1]):
        raise ValueError('FRESH_UNIT_PATH_MISMATCH')
    if (unit / 'PREFIX_IMPORT.json').exists() or list(unit.glob('roots/*/HISTORICAL_QA_IMPORT.json')):
        raise ValueError('EVAL_HISTORICAL_IMPORT_FORBIDDEN')
    rows = json.loads((batch / 'ROLE_ASSIGNMENTS.json').read_text())['roots']
    matches = [r for r in rows if r['root_id'] == root]
    if len(matches) != 1 or matches[0]['author'] == matches[0]['reviewer']:
        raise ValueError('FRESH_ASSIGNED_ROLE_REQUIRED')
    return matches[0]['author']
