#!/usr/bin/env python3
"""Independent, public-only audit; no model generation and no private Gold reads."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[1]
R = P / 'runtime/evidence_v1'
def read(p): return json.loads(p.read_text())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def binding(p): return {'path': str(p.resolve()), 'sha256': sha(p), 'bytes': p.stat().st_size}
checks = []
def ck(ok, name, detail=''):
    checks.append({'name': name, 'pass': bool(ok), 'detail': str(detail)})
def verify_binding(b):
    p = Path(b['path'])
    ck(p.is_file(), 'binding_exists', p)
    if p.is_file():
        ck(p.stat().st_size == b['bytes'], 'binding_bytes', p)
        ck(sha(p) == b['sha256'], 'binding_sha256', p)
frozen = read(R/'PREFLIGHT_FREEZE.json')
m = read(R/'manifest.json')
for group in [frozen['files'], m['historical_public_bindings'], m['public_input_bindings'], m['code_bindings'], m['model_bindings']]:
    for b in group: verify_binding(b)
ck(frozen['status'] == 'PREPARED_AWAITING_INDEPENDENT_PREFLIGHT' and not frozen['inference_started'], 'prepared_not_started')
ck(not (R/'STARTED.json').exists(), 'live_not_started')
spec = importlib.util.spec_from_file_location('diagnostic_public_audit_runner', P/'runtime/evidence_handoff.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
direct = read(runner.DIRECT/'manifest.json')
base_rows = [x for x in direct['requests'] if x['variant_id'] == 'perm_00' and x['slot'] == 0]
public = read(R/'PUBLIC_QUESTIONS.json')['questions']
qby = {q['candidate_id']: q for q in public}
admission = read(R/'CONTENT_ADMISSION.json')
maps = read(R/'DISPLAY_MAPPINGS.json')
oldmaps = read(runner.DIRECT/'DISPLAY_MAPPINGS.json')
ids = read(R/'PROMPT_TOKEN_IDS.json')
rows = m['requests']
ck(len(rows) == len(qby) == len(base_rows) == 23, 'all_23_questions_requests')
ck({r['candidate_id'] for r in rows} == set(qby) == set(admission['candidate_ids']), 'content_admitted_no_filter')
ck(len({r['root_id'] for r in rows}) == 4, 'four_roots')
ck(sum(q['question_type']=='multi_select' for q in public) == 12 and sum(q['question_type']=='ordering' for q in public) == 11, 'types_12_MC_11_ordering')
ck(m['request_count'] == 23 and m['automatic_retry'] is False, 'max23_no_retry')
ck(m['enable_thinking'] is False and m['max_tokens'] == 2048, 'short_auditable_explanations_budget')
ck(m['source_truncation'] is False and m['max_model_len'] == 16384, 'full_context_no_truncation')
ck(all(m[k] is False for k in ('constrained_decoding','json_grammar','finite_answer_choices')), 'no_finite_or_schema_decoding_constraint')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(m['tokenizer'], local_files_only=True)
for row, base in zip(rows, base_rows):
    q = qby[row['candidate_id']]
    allowed = {'candidate_id','root_id','question_type','question','options','items','source_path','source_sha256'}
    ck(not set(q)-allowed, 'public_whitelist_no_gold_author_evidence', row['candidate_id'])
    ck(row['variant_id'] == 'perm_00' and row['slot']==0, 'fixed_perm00_sample00', row['request_id'])
    ck(row['direct_request_id'] == base['request_id'] and row['seed'] == base['seed'], 'same_direct_seed_mapping', row['request_id'])
    variant = maps['requests'][row['request_id']]
    ck(variant == oldmaps['requests'][base['request_id']], 'mapping_byte_semantics_preserved', row['request_id'])
    sourcepath = R/'sources'/(row['root_id']+'.json')
    source = read(sourcepath)
    ck(sha(sourcepath) == row['source_sha256'] == q['source_sha256'], 'source_binding', row['request_id'])
    ck(len(source['turns']) == row['source_turns'], 'source_turn_count', row['request_id'])
    text = (R/'prompts'/(row['request_id']+'.txt')).read_text()
    ck(all(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" in text for t in source['turns']), 'every_full_turn_present', row['request_id'])
    reconstructed = runner.prompt(tokenizer, q, variant, source)
    ck(text == reconstructed, 'public_only_prompt_reconstruction', row['request_id'])
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    ck(token_ids == ids[row['request_id']] and len(token_ids) == row['prompt_tokens'], 'token_ids_recomputed', row['request_id'])
    ck(len(token_ids) + row['max_output_tokens'] <= 16384, 'no_token_truncation', row['request_id'])
    ck(hashlib.sha256(text.encode()).hexdigest() == row['prompt_sha256'], 'prompt_sha256', row['request_id'])
    params = runner.params(row)
    ck(getattr(params,'structured_outputs',None) is None and getattr(params,'guided_decoding',None) is None and not params.stop, 'real_sampling_no_constraint_or_stopbox', row['request_id'])
tree = ast.parse((P/'runtime/evidence_handoff.py').read_text())
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in {'prepare','verify','live','prompt','params','selected'}:
        calls.append(node.name)
ck(set(calls) == {'prepare','verify','live','prompt','params','selected'}, 'audited_all_execution_functions')
ck('Gold is never opened by this module.' in (P/'runtime/evidence_handoff.py').read_text(), 'gold_blind_module_declaration')
ck(m['gold_opened_by_prepare'] is False and m['use_as_final_eval'] is False and m['training_started'] is False, 'diagnostic_only_boundary')
# The parent loader helper reads code/config for imports. It performs no runtime private
# data read during import; the audited runner's preparation/live path only uses the
# public files reconstructed above. The shared filesystem is not technical isolation.
result = {'schema_version':'independent_diagnostic_preflight_v1','status':'PASS' if all(x['pass'] for x in checks) else 'FAIL',
 'created_at_utc':datetime.now(timezone.utc).isoformat(), 'reviewer':'separate_workflow_handoff_agent',
 'public_only':True,'private_gold_or_author_evidence_read':False,'inference_calls':0,
 'checks_count':len(checks),'failure_count':sum(not x['pass'] for x in checks),'checks':checks,
 'audited_files':[binding(R/'PREFLIGHT_FREEZE.json'),binding(R/'manifest.json'),binding(P/'runtime/evidence_handoff.py'),binding(Path(__file__))],
 'request_count':23,'question_count':23,'token_input_min':min(x['prompt_tokens'] for x in rows),
 'token_input_max':max(x['prompt_tokens'] for x in rows),'max_output_tokens':2048,
 'minimum_context_headroom':min(16384-x['prompt_tokens']-2048 for x in rows),
 'model_files_verified':len(m['model_bindings']),'code_files_verified':len(m['code_bindings']),
 'verified_model_bytes':sum(x['bytes'] for x in m['model_bindings']),
 'interpretation':'GO for exactly 23 fixed evidence diagnostic requests; does not certify semantic evidence quality or feedback readiness.',
 'limitations':['Procedural independent review, shared filesystem; no technical blinding claim.',
 'No hidden reasoning trace is requested or inferred; output is a short verifiable explanation.',
 'JSON schema is posthoc validation only; valid format is not guaranteed.']}
out=P/'review/INDEPENDENT_DIAGNOSTIC_PREFLIGHT.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in {'checks','audited_files'}},ensure_ascii=False))
if result['status']!='PASS': raise SystemExit(2)
