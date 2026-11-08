"""Native paired preflight on historical temporary fixtures; never runs inference."""
from pathlib import Path
import importlib.util
import json
import tempfile

spec = importlib.util.spec_from_file_location('v118_eval_fixture', Path(__file__).with_name('evaluate.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
old = m.read(m.B.parent / 'h4_codex_screened_ab_validation_v117/evaluation/EVALUATION_POOL.json')
questions = [next(q for q in old['questions'] if q['question_type'] == typ) for typ in ['multi_select', 'ordering']]
with tempfile.TemporaryDirectory() as temp:
    scope = Path(temp)
    text = (m.NATIVE / 'protocol.py').read_text()
    text = text.replace('if provenance.get("requested_model") != "Qwen/Qwen3.6-35B-A3B":',
        'if (provenance.get("backend") != "CURRENT_CODEX_SESSION" or provenance.get("source_batch_id") != ' + repr(m.B.name) + '):')
    text = text.replace('CONTRACT_DIR = Path(__file__).resolve().parent.parent', 'CONTRACT_DIR = Path(' + repr(str(m.NATIVE.parent)) + ')')
    (scope / 'adapter_protocol.py').write_text(text)
    m.save(scope / 'pool.json', {'provenance': {'backend': 'CURRENT_CODEX_SESSION', 'source_batch_id': m.B.name}, 'questions': questions})
    native = m.native(scope)
    native.validate_pool(scope / 'pool.json')
    for side, model in [('base', 'M0'), ('trained', 'M_clean_policy')]:
        native.preflight(scope / 'pool.json', scope / 'run' / side, solver_model=model)
    for name in ['questions.json', 'gold.json']:
        assert m.read(scope / 'run/base' / name) == m.read(scope / 'run/trained' / name)
    manifests = [m.read(scope / 'run' / side / 'manifest.json') for side in ['base', 'trained']]
    assert all(manifest['request_count'] == 30 for manifest in manifests)
    for a, b in zip(manifests[0]['requests'], manifests[1]['requests']):
        assert a['seed'] == b['seed'] and a['variant'] == b['variant'] and a['prompt_sha256'] == b['prompt_sha256']
    print(json.dumps({'status': 'PASS', 'scope': 'Two historical temporary fixtures, no inference or generation',
          'calls_prepared_per_model': 30, 'max_prompt_tokens': max(r['prompt_tokens'] for r in manifests[0]['requests'])}))
