"""Additive school-API evaluation; never changes the sealed v122 experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
B = Path(__file__).resolve().parents[1]
OLD = B.parent / 'h4_codex_remaining_six_repair_v122'
REFERENCE = OLD / 'evaluations/paired_v1/run/base'
NATIVE = B.parent / 'h4_feedback_training_pool_v15_qa/evaluation'
KEY_FILE = Path('/root/.config/gxy-selfplay/tju_api_key')
ENDPOINT = 'https://ai.tju.edu.cn/api/v3/chat/completions'
MODEL = 'tju-llm'
SYSTEM = ('You are a strict structured dialogue QA solver. Use only the supplied '
          'dialogue when it is present. Return the final answer in exactly one '
          '<box>...</box> pair. Do not provide explanations, reasoning, or any '
          'text outside the box.')
PROTOCOL = {
    'schema_version': 'v123_tju_same_questions_api_v1',
    'endpoint': ENDPOINT, 'model': MODEL,
    'sampling': {'temperature': 0.8, 'top_p': 0.9, 'max_tokens': 8192, 'stream': False},
    'seed': 'Send the original per-request seed; provider support is unverified.',
    'structured_decoder': 'NONE: API support for the local finite-choice decoder is unverified.',
    'thinking_control': 'No provider-specific override; server behavior is unknown.',
    'stop': 'No stop override; preserve the complete returned message for strict native parsing.',
    'comparison': 'Same public system/user message contents, questions, permutations, slots, Gold and scoring. '
                  'Provider chat template, model/tokenizer, constrained decoding and reasoning budget differ or are unknown. '
                  'An external-solver descriptive comparison, not a strictly matched decoding experiment.',
    'retry_policy': 'One HTTP attempt per frozen evaluation request. No response repair, parameter fallback, '
                    'automatic retries or score-guided rewriting. A transport/HTTP failure stops unsent requests. '
                    'Resume skips every completed response, including failures; ambiguous attempts cannot be resent.',
    'timeout_seconds': 300, 'minimum_interval_seconds': 1.0,
    'probe': 'One separate synthetic formatting/connectivity request per explicit probe invocation; not scored as a question.',
    'credentials': {'environment_variable': 'TJU_API_KEY', 'fallback_file': str(KEY_FILE),
                    'secret_in_evidence': False},
    'training_or_pool_merge': False,
}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def verify(rows):
    for row in rows:
        need(sha(row['path']) == row['sha256'], 'BOUND_FILE_CHANGED:' + row['path'])


def native_scoring():
    spec = importlib.util.spec_from_file_location('v123_native_scoring', NATIVE / 'scoring.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def messages_for(question, request, source):
    """Public inputs only; round-trip checked against every frozen native prompt."""
    qtype = question['question_type']
    field = 'options' if qtype == 'multi_select' else 'items'
    heading = 'Options' if qtype == 'multi_select' else 'Items'
    chunks = ['Dialogue:\n' + '\n'.join(
        f"[{row['turn_ref']}] {row['speaker']}: {row['text']}" for row in source['turns']),
        f"Question ({qtype}):\n{question['question']}",
        heading + ':\n' + '\n'.join(f'{key}. {text}' for key, text in request['variant'][field].items())]
    chunks.append('Output only selected letters separated by commas, e.g. <box>A,C</box>.' if qtype == 'multi_select'
                  else 'Output only ordered item IDs separated by commas, e.g. <box>I2,I4,I1,I3</box>.')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(chunks)}]


def native_chat_prompt(messages):
    return ('<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n'
            '<|im_start|>user\n' + messages[1]['content'] + '<|im_end|>\n'
            '<|im_start|>assistant\n<think>\n\n</think>\n\n')


def payload_for(messages, seed):
    return {'model': MODEL, 'messages': messages, **PROTOCOL['sampling'], 'seed': seed}


def prepare():
    need(not (B / 'prepared').exists(), 'PREPARED_ALREADY_EXISTS_NO_OVERWRITE')
    need(not (B / 'PROTOCOL.json').exists(), 'PROTOCOL_ALREADY_EXISTS')
    sealed = read(OLD / 'COMPLETE_MANIFEST.json')
    verify(sealed['files'])
    old_files = {row['path'] for row in sealed['files']} | {str(OLD / 'COMPLETE_MANIFEST.json')}
    need({str(p) for p in OLD.rglob('*') if p.is_file()} == old_files, 'SEALED_V122_FILE_SET_CHANGED')
    base = read(REFERENCE / 'manifest.json')
    trained = read(OLD / 'evaluations/paired_v1/run/trained/manifest.json')
    need(base['requests'] == trained['requests'], 'REFERENCE_PAIRED_REQUESTS_DIFFER')
    questions = read(REFERENCE / 'questions.json')['candidates']
    qmap = {q['candidate_id']: q for q in questions}
    need(len(qmap) == len(questions) == 42 and len(base['requests']) == 684, 'FIXED_42_684_SCOPE')
    folder = B / 'prepared'
    folder.mkdir()
    for original, name in [('questions.json', 'QUESTIONS.json'), ('gold.json', 'GOLD.json'),
                           ('canonical_bindings.json', 'CANONICAL_BINDINGS.json')]:
        shutil.copyfile(REFERENCE / original, folder / name)
    save(B / 'PROTOCOL.json', PROTOCOL)
    originals = [OLD / 'COMPLETE_MANIFEST.json', REFERENCE / 'manifest.json',
                 OLD / 'evaluations/paired_v1/run/trained/manifest.json',
                 OLD / 'evaluations/paired_v1/EVALUATION_POOL.json',
                 REFERENCE / 'questions.json', REFERENCE / 'gold.json', REFERENCE / 'canonical_bindings.json']
    sources = {}
    for root in sorted({q['root_id'] for q in questions}):
        path = REFERENCE / 'sources' / (root + '.json')
        target = folder / 'sources' / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        originals.append(path)
        sources[root] = read(path)
    for row in base['requests']:
        messages = messages_for(qmap[row['candidate_id']], row, sources[row['root_id']])
        original = REFERENCE / 'prompts' / (row['request_id'] + '.txt')
        need(native_chat_prompt(messages) == original.read_text(), 'PUBLIC_PROMPT_ROUNDTRIP_FAILED:' + row['request_id'])
        need(sha(original) == row['prompt_sha256'], 'REFERENCE_PROMPT_HASH_MISMATCH')
        save(folder / 'payloads' / (row['request_id'] + '.json'), payload_for(messages, row['seed']))
        originals.append(original)
    save(folder / 'MANIFEST.json', {
        'status': 'PREPARED_NO_NETWORK_CALLS', 'created_at_utc': now(),
        'solver_model': MODEL, 'question_count': 42, 'request_count': 684,
        'requests': base['requests'], 'full_slots': base['full_slots'],
        'same_public_message_contents_as_local': True, 'native_chat_template_sent_to_api': False,
        'prompt_proxy_tokens': {'tokenizer': 'Qwen3-4B, NOT the unknown school tokenizer',
                                'total': sum(row['prompt_tokens'] for row in base['requests']),
                                'maximum': max(row['prompt_tokens'] for row in base['requests'])},
        'input_bindings': [bind(p) for p in originals],
        'training_or_pool_merge': False})
    print(json.dumps({'status': 'PREPARED', 'questions': 42, 'evaluation_http_attempts_planned': 684,
                      'key_file': str(KEY_FILE), 'network_calls': 0}))


def freeze():
    need(not (B / 'TOOLING_FREEZE.json').exists(), 'TOOLING_ALREADY_FROZEN')
    need(not (B / 'probes').exists() and not (B / 'run').exists(), 'NETWORK_SCOPE_EXISTS_BEFORE_FREEZE')
    need(read(B / 'PROTOCOL.json') == PROTOCOL, 'PROTOCOL_CODE_MISMATCH')
    manifest = read(B / 'prepared/MANIFEST.json')
    verify(manifest['input_bindings'])
    need((B / 'tools/summarize_tju.py').exists(), 'REPORT_TOOL_REQUIRED')
    paths = [B / 'PROTOCOL.json', *sorted((B / 'prepared').rglob('*')),
             *sorted((B / 'tools').glob('*.py')), NATIVE / 'scoring.py']
    paths = [p for p in paths if p.is_file()]
    save(B / 'TOOLING_FREEZE.json', {'status': 'FROZEN_BEFORE_ANY_API_CALL', 'created_at_utc': now(),
         'files': [bind(p) for p in paths], 'credentials_excluded': True,
         'reference_manifest_binding': bind(OLD / 'COMPLETE_MANIFEST.json'),
         'planned_evaluation_calls': manifest['request_count'], 'automatic_retry': False,
         'training_or_pool_merge': False})
    print(json.dumps({'status': 'FROZEN', 'files': len(paths), 'network_calls': 0}))


def frozen():
    document = read(B / 'TOOLING_FREEZE.json')
    verify(document['files'])
    verify([document['reference_manifest_binding']])
    manifest = read(B / 'prepared/MANIFEST.json')
    verify(manifest['input_bindings'])
    need(read(B / 'PROTOCOL.json') == PROTOCOL, 'PROTOCOL_CODE_MISMATCH')
    return manifest


def credential():
    # Never print, hash, bind or serialize this value.
    key = os.environ.get('TJU_API_KEY')
    if not key:
        need(KEY_FILE.is_file(), 'CREDENTIAL_FILE_MISSING:' + str(KEY_FILE))
        need(KEY_FILE.stat().st_mode & 0o077 == 0, 'CREDENTIAL_FILE_MUST_BE_PRIVATE_0600')
        key = KEY_FILE.read_text(encoding='utf-8').strip()
    need(bool(key) and key not in {'PASTE_YOUR_TJU_API_KEY_HERE', '<YOUR_API_KEY>', 'YOUR_API_KEY'},
         'CREDENTIAL_REQUIRED: fill ' + str(KEY_FILE) + ' or set TJU_API_KEY')
    need(key.isascii() and not any(c.isspace() for c in key), 'CREDENTIAL_MUST_BE_ONE_ASCII_TOKEN')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_once(payload, key):
    """Exactly one transport attempt; credentials and request headers are never saved."""
    started = time.monotonic()
    request = urllib.request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, method='POST')
    opener = urllib.request.build_opener(NoRedirect())
    body = ''
    try:
        with opener.open(request, timeout=PROTOCOL['timeout_seconds']) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            need(len(raw) <= 8 * 1024 * 1024, 'API_RESPONSE_TOO_LARGE')
            body = raw.decode('utf-8', errors='replace').replace(key, '[REDACTED]')
            status = response.status
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            return {'transport_status': 'NON_JSON_RESPONSE', 'http_status': status,
                    'raw_body': body, 'elapsed_seconds': time.monotonic() - started}
        return {'transport_status': 'HTTP_OK', 'http_status': status,
                'response_json': value, 'elapsed_seconds': time.monotonic() - started}
    except urllib.error.HTTPError as error:
        body = error.read(8 * 1024 * 1024).decode('utf-8', errors='replace').replace(key, '[REDACTED]')
        return {'transport_status': 'HTTP_ERROR', 'http_status': error.code,
                'raw_body': body, 'elapsed_seconds': time.monotonic() - started}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        # Some exception messages contain request URLs or proxy details. Keep only the type.
        return {'transport_status': 'TRANSPORT_ERROR', 'http_status': None,
                'error_type': type(error).__name__, 'elapsed_seconds': time.monotonic() - started}


def response_content(outcome):
    value = outcome.get('response_json')
    if not isinstance(value, dict):
        return None, None, None, None
    choices = value.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None, None, value.get('model'), value.get('usage')
    choice = choices[0]
    message = choice.get('message')
    content = message.get('content') if isinstance(message, dict) else None
    return content, choice.get('finish_reason'), value.get('model'), value.get('usage')


def probe():
    frozen()
    key = credential()  # Missing keys never create an attempt or send a request.
    directory = B / 'probes'
    directory.mkdir(exist_ok=True)
    serial = max([int(p.name.split('_')[1]) for p in directory.glob('probe_*') if p.is_dir()] or [0]) + 1
    target = directory / f'probe_{serial:04d}'
    target.mkdir()
    payload = payload_for([{'role': 'user', 'content': '请只输出 <box>A,C</box>，不要解释。'}], 2026100800)
    save(target / 'REQUEST.json', payload)
    save(target / 'ATTEMPT.json', {'started_at_utc': now(), 'request_binding': bind(target / 'REQUEST.json'),
         'http_attempt_number': 1, 'automatic_retry': False, 'part_of_evaluation': False})
    outcome = http_once(payload, key)
    save(target / 'RAW_RESPONSE.json', outcome)
    content, finish, model, usage = response_content(outcome)
    success = outcome['transport_status'] == 'HTTP_OK' and isinstance(content, str) and bool(content.strip()) and finish in {'stop', 'eos'}
    result = {'status': 'PROBE_PASS' if success else 'PROBE_FAILED', 'completed_at_utc': now(),
              'raw_binding': bind(target / 'RAW_RESPONSE.json'), 'request_binding': bind(target / 'REQUEST.json'),
              'tooling_freeze_binding': bind(B / 'TOOLING_FREEZE.json'),
              'returned_model': model, 'finish_reason': finish, 'usage': usage,
              'format_followed': native_scoring().parse_box(content) == (['A', 'C'], None),
              'provider_seed_parameter_acceptance_not_proof_of_effect': True,
              'part_of_evaluation': False}
    save(target / 'RESULT.json', result)
    print(json.dumps({'status': result['status'], 'returned_model': model, 'finish_reason': finish,
                      'format_followed': result['format_followed'], 'path': str(target)}, ensure_ascii=False))
    need(success, 'PROBE_FAILED_FULL_EVALUATION_NOT_STARTED')


def record_for(row, question, gold, outcome):
    raw, finish, returned_model, usage = response_content(outcome)
    scored = dict(question, gold_content_ids=gold)
    score = native_scoring().score_response(scored, row, raw)
    errors = []
    if outcome['transport_status'] != 'HTTP_OK':
        errors.append(outcome['transport_status'])
    if not isinstance(raw, str):
        errors.append('CONTENT_MISSING_OR_NOT_TEXT')
    if finish not in {'stop', 'eos'}:
        errors.append('INCOMPLETE_FINISH_REASON:' + str(finish))
    if score.get('scored') is not True:
        errors.append('NATIVE_SCORE_FAILURE:' + str(score.get('scoring_status')))
    return {'request_id': row['request_id'], 'candidate_id': row['candidate_id'], 'root_id': row['root_id'],
            'solver_model': MODEL, 'model_path': None, 'question_type': row['question_type'],
            'variant_id': row['variant']['variant_id'], 'mode': row['mode'], 'slot': row['slot'], 'seed': row['seed'],
            'raw_output': raw, 'score': score, 'evaluation_valid': not errors, 'data_errors': errors,
            'transport_status': outcome['transport_status'], 'http_status': outcome.get('http_status'),
            'returned_model': returned_model, 'finish_reason': finish, 'usage': usage,
            'elapsed_seconds': outcome['elapsed_seconds'], 'http_attempt_count': 1,
            'seed_adoption_verified': False, 'structured_decoder': 'NONE', 'thinking_control': 'UNKNOWN'}


def live():
    manifest = frozen()
    key = credential()
    probe_paths = sorted((B / 'probes').glob('probe_*/RESULT.json'))
    need(probe_paths and read(probe_paths[-1])['status'] == 'PROBE_PASS', 'LATEST_CONNECTIVITY_PROBE_MUST_PASS')
    probe_doc = read(probe_paths[-1])
    verify([probe_doc['tooling_freeze_binding'], probe_doc['request_binding'], probe_doc['raw_binding']])
    run = B / 'run'
    run.mkdir(exist_ok=True)
    with (run / 'EXECUTION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not (run / 'COMPLETE.json').exists(), 'EVALUATION_ALREADY_COMPLETE_DO_NOT_REPEAT')
        planned = {row['request_id']: row for row in manifest['requests']}
        observed = {p.stem: p for p in (run / 'responses').glob('*.json')}
        attempts = {p.stem: p for p in (run / 'attempts').glob('*.json')}
        need(set(observed) <= set(attempts) <= set(planned), 'UNPLANNED_RESPONSE_OR_ATTEMPT')
        need(set(attempts) == set(observed), 'AMBIGUOUS_PRIOR_ATTEMPT_NO_AUTOMATIC_RESEND')
        questions = {q['candidate_id']: q for q in read(B / 'prepared/QUESTIONS.json')['candidates']}
        gold = read(B / 'prepared/GOLD.json')
        # A pending attempt always blocks. Recovery may inspect its raw receipt,
        # but this runner never silently recovers or resends it.
        for rid, path in observed.items():
            record = read(path)
            need(record['request_id'] == rid and record['candidate_id'] == planned[rid]['candidate_id'], 'PRIOR_RECORD_IDENTITY')
            expected_raw = run / 'raw' / (rid + '.json')
            expected_payload = B / 'prepared/payloads' / (rid + '.json')
            need(read(attempts[rid])['request_id'] == rid, 'PRIOR_ATTEMPT_IDENTITY_CHANGED')
            need(record['raw_response_binding'] == bind(expected_raw), 'PRIOR_RAW_RECEIPT_CHANGED')
            need(read(attempts[rid])['payload_binding'] == bind(expected_payload), 'PRIOR_PAYLOAD_RECEIPT_CHANGED')
            rebuilt = record_for(planned[rid], questions[planned[rid]['candidate_id']],
                                 gold[planned[rid]['candidate_id']], read(expected_raw))
            rebuilt['raw_response_binding'] = bind(expected_raw)
            need(record == rebuilt, 'PRIOR_RESPONSE_IDENTITY_OR_SCORE_CHANGED:' + rid)
        invocations = run / 'invocations'
        invocations.mkdir(exist_ok=True)
        invocation = len(list(invocations.glob('START_*.json'))) + 1
        save(invocations / f'START_{invocation:04d}.json', {'started_at_utc': now(),
             'already_recorded': len(observed), 'probe_binding': bind(probe_paths[-1]),
             'tooling_freeze_binding': bind(B / 'TOOLING_FREEZE.json'), 'automatic_retry': False})
        for row in manifest['requests']:
            rid = row['request_id']
            if rid in observed:
                continue
            path = B / 'prepared/payloads' / (rid + '.json')
            save(run / 'attempts' / (rid + '.json'), {'started_at_utc': now(), 'request_id': rid,
                 'payload_binding': bind(path), 'http_attempt_number': 1, 'automatic_retry': False})
            outcome = http_once(read(path), key)
            raw_path = run / 'raw' / (rid + '.json')
            save(raw_path, outcome)
            record = record_for(row, questions[row['candidate_id']], gold[row['candidate_id']], outcome)
            record['raw_response_binding'] = bind(raw_path)
            save(run / 'responses' / (rid + '.json'), record)
            observed[rid] = run / 'responses' / (rid + '.json')
            if len(observed) % 12 == 0:
                print(json.dumps({'status': 'RUNNING', 'recorded': len(observed), 'planned': len(planned)}), flush=True)
            if outcome['transport_status'] != 'HTTP_OK':
                save(invocations / f'STOP_{invocation:04d}.json', {'status': 'STOPPED_UNSENT_PRESERVED',
                     'completed_at_utc': now(), 'request_id': rid, 'reason': outcome['transport_status'],
                     'http_status': outcome.get('http_status'), 'recorded': len(observed),
                     'remaining_unsent': len(planned) - len(observed), 'request_will_not_be_retried': True})
                raise RuntimeError('HTTP_OR_TRANSPORT_FAILURE_STOPPED_UNSENT_REQUESTS')
            time.sleep(PROTOCOL['minimum_interval_seconds'])
        verify(read(B / 'TOOLING_FREEZE.json')['files'])
        invalid = sum(read(path)['evaluation_valid'] is not True for path in observed.values())
        save(run / 'COMPLETE.json', {'status': 'COMPLETE', 'completed_at_utc': now(),
             'actual_call_count': len(observed), 'valid_call_count': len(observed) - invalid,
             'invalid_call_count': invalid, 'tooling_freeze_binding': bind(B / 'TOOLING_FREEZE.json'),
             'response_bindings': [bind(observed[rid]) for rid in sorted(observed)],
             'training_or_pool_merge': False})
        print(json.dumps({'status': 'COMPLETE', 'calls': len(observed), 'invalid': invalid}), flush=True)


def status():
    paths = sorted((B / 'run/responses').glob('*.json'))
    rows = [read(p) for p in paths]
    print(json.dumps({'prepared': (B / 'prepared/MANIFEST.json').exists(),
         'frozen': (B / 'TOOLING_FREEZE.json').exists(), 'key_file': str(KEY_FILE),
         'recorded': len(rows), 'valid': sum(r['evaluation_valid'] is True for r in rows),
         'complete': (B / 'run/COMPLETE.json').exists(), 'key_contents_never_printed': True}))


def run():
    """Single entry point after preparation/freeze and manual credential entry."""
    frozen()
    if not (B / 'run/COMPLETE.json').exists():
        probes = sorted((B / 'probes').glob('probe_*/RESULT.json'))
        if not probes:
            probe()
        else:
            need(read(probes[-1])['status'] == 'PROBE_PASS',
                 'LATEST_PROBE_FAILED: inspect saved result; after fixing the credential/service, invoke probe explicitly')
        live()
    target = B / 'delivery/SCHOOL_SOLVER_SUMMARY_V1.json'
    if target.exists():
        print(json.dumps({'status': 'REPORT_ALREADY_EXISTS_NO_NETWORK_OR_OVERWRITE', 'report': str(target)}))
        return
    spec = importlib.util.spec_from_file_location('v123_tju_report', B / 'tools/summarize_tju.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'freeze', 'probe', 'live', 'run', 'status'])
    args = parser.parse_args()
    try:
        globals()[args.action]()
    except (ValueError, RuntimeError, BlockingIOError) as error:
        print(json.dumps({'status': 'STOPPED', 'reason': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
