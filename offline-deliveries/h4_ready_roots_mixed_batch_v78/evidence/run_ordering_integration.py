"""Exercise real offline ordering preparation in a disposable directory."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
blocked_attempts = []


def forbid_external(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto',
                 'socket.sendmsg', 'subprocess.Popen', 'os.system', 'os.exec',
                 'os.posix_spawn', 'os.posix_spawnp'}:
        blocked_attempts.append(event)
        raise AssertionError('External operation prohibited: ' + event)


sys.addaudithook(forbid_external)


def binding(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    evidence = HERE / 'evidence'
    result = {'status': 'RUNNING', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Real ordering prepare, feedback, tokenizer capacity and verification; no generation.',
              'network_and_subprocess_forbidden': True,
              'generation_called': False, 'real_dependency_mocks': False,
              'input_config': binding(HERE / 'CONFIG.json'),
              'ordering_code': binding(HERE / 'ordering_generation.py')}
    success = False
    with (evidence / 'ORDERING_INTEGRATION.log').open('w') as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                import ordering_generation as generation
                with tempfile.TemporaryDirectory(prefix='v78_ordering_integration_') as temporary:
                    temporary_home = Path(temporary).resolve()
                    (temporary_home / 'CONFIG.json').write_bytes((HERE / 'CONFIG.json').read_bytes())
                    prepared = generation.prepare(temporary_home)
                    rows = []
                    for slot in prepared:
                        run = temporary_home / 'ordering' / slot['slot_id']
                        manifest = generation.read(run / 'MANIFEST.json')
                        payload = generation.read(run / 'REQUEST.json')
                        proof = generation.read(run / 'PROOF.json')
                        capacity = generation.read(run / 'CAPACITY.json')
                        context = json.JSONDecoder().raw_decode(
                            payload['messages'][0]['content'].split('\nINPUT_JSON\n', 1)[1])[0]
                        assert context['source_turns'] == generation.read(run / 'SOURCE.json')['turns']
                        assert manifest['solver_model'] == 'M_clean_policy'
                        assert manifest['generator_model'] == payload['model'] == 'Qwen/Qwen3.6-35B-A3B'
                        assert capacity['status'] == 'PASS'
                        assert proof['payload_sha256'] == generation.engine().base.digest(payload)
                        rows.append({'slot_id': slot['slot_id'], 'root_id': slot['root_id'],
                                     'full_source_turn_count': len(context['source_turns']),
                                     'source': manifest['source'], 'original_source': manifest['original_source'],
                                     'solver_model': manifest['solver_model'],
                                     'generator_model': manifest['generator_model'],
                                     'payload_sha256': proof['payload_sha256'],
                                     'feedback_receipt': proof['feedback_receipt'],
                                     'capacity': capacity})
                        print(json.dumps({'event': 'real_prepare_pass', 'slot_id': slot['slot_id'],
                                          'capacity': capacity}, ensure_ascii=False), flush=True)
                    # Reproduce the previously mocked-away dependency with the real builder.
                    run = temporary_home / 'ordering' / prepared[0]['slot_id']
                    original = (run / 'MANIFEST.json').read_bytes()
                    manifest = generation.read(run / 'MANIFEST.json')
                    del manifest['solver_model']
                    (run / 'MANIFEST.json').write_text(json.dumps(manifest))
                    try:
                        generation.payload_for(run)
                    except KeyError as error:
                        assert error.args == ('solver_model',)
                        result['missing_solver_model_regression'] = 'Real payload builder raises KeyError as expected.'
                    else:
                        raise AssertionError('Missing solver model was unexpectedly accepted.')
                    finally:
                        (run / 'MANIFEST.json').write_bytes(original)
                    replay = generation.verify(temporary_home)
                    assert replay == {'candidate_count': 0, 'http_attempts': 0}
                    assert not list(temporary_home.rglob('HTTP_ATTEMPT.json'))
                    assert not list(temporary_home.rglob('GENERATION.STARTED.json'))
                    assert not blocked_attempts
                    result.update(status='PASS', slots=rows, prepared_slot_count=len(prepared),
                                  verification=replay, http_attempts=0, actual_network_attempts=0,
                                  temporary_directory_removed_after_check=True)
                    success = True
            except BaseException as error:
                traceback.print_exc()
                result.update(status='FAIL', error_type=type(error).__name__, error=str(error))
            finally:
                result['blocked_external_attempts'] = blocked_attempts
                result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
                print(json.dumps({'status': result['status'], 'completed_at_utc': result['completed_at_utc']}), flush=True)
    result['log'] = binding(evidence / 'ORDERING_INTEGRATION.log')
    (evidence / 'ORDERING_INTEGRATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'report': str(evidence / 'ORDERING_INTEGRATION.json')}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
