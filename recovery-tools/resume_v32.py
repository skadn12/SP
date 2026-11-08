"""Recover one interrupted v32 run without repeating completed HTTP calls.

The original frozen runner stays unchanged. Its live body is reused with only
the three initial reservation/start writes removed. Every completed call must
replay the exact payload and proof. An unfinished call may resume only if it
has no recorded HTTP attempt. Existing replay artifacts are equality checked.
"""
from pathlib import Path
import argparse
import ast
import inspect
import json
import sys

DELIVERY = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_sequential_ab_v32')
RUN = DELIVERY / 'runs/inherited_a2_b3_01'
sys.path.insert(0, str(DELIVERY))
import runner as r


class ReplayBoundary(Exception):
    pass


def main(dry):
    r.verify(RUN, live_execution=True)
    r.base.require((RUN/'LIVE_STARTED.json').exists(), 'missing_original_start')
    r.base.require(not (RUN/'REPORT.json').exists(), 'already_complete')
    directories = sorted((RUN/'requests').iterdir())
    r.base.require(len(directories) == 17, 'unexpected_recovery_boundary')
    for i, directory in enumerate(directories, 1):
        r.base.require(directory.name.startswith(f'{i:03d}-'), 'nonsequential_requests')
        if i < 17:
            r.base.require((directory/'RESPONSE.json').exists(), 'missing_recorded_response')
        else:
            r.base.require(not list(directory.glob('HTTP_ATTEMPT*.json')),
                           'uncertain_HTTP_outcome_cannot_resume')
    original_write = r.base.write
    frozen = r.base.bindings(sorted(p for p in RUN.rglob('*') if p.is_file()
                                    and p.suffix == '.json'))
    state = {'replaying': True, 'replayed': 0}

    def checked_write(path, value):
        path = Path(path)
        if path.exists():
            r.base.require(r.base.digest(r.base.read(path)) == r.base.digest(value),
                           'existing_artifact_mismatch:'+str(path))
            return
        if state['replaying'] and not path.exists():
            raise ValueError('unexpected_new_replay_artifact:'+str(path))
        original_write(path, value)

    class RecoveryCaller(r.Caller):
        def call(self, label, payload, proof):
            number = self.request_counter + 1
            directory = self.run/'requests'/f'{number:03d}-{label}'
            if number <= 17:
                r.base.require(directory == directories[number-1], 'replay_label_mismatch')
                r.base.require(r.base.read(directory/'REQUEST.json') == payload, 'replay_payload_mismatch')
                saved = r.base.read(directory/'PROOF.json')
                r.base.require(all(r.base.digest(saved.get(k)) == r.base.digest(v)
                                   for k,v in proof.items()), 'replay_proof_mismatch:'+label)
            if number < 17:
                self.request_counter += 1
                self.http_counter += len(list(directory.glob('HTTP_ATTEMPT*.json')))
                state['replayed'] += 1
                return r.base.read(directory/'RESPONSE.json')['text'], directory
            if number == 17:
                r.base.verify_bindings(frozen)
                if dry:
                    raise ReplayBoundary()
                state['replaying'] = False
                # Earlier response was hours ago; preserve the HTTP counters
                # and start an ordinary, bounded first attempt for request 17.
                receipt = RUN/'RECOVERY_STARTED.json'
                if receipt.exists():
                    receipt = RUN/'RECOVERY_STARTED_02.json'
                original_write(receipt, {
                    'version':'v32_interruption_recovery_v1', 'started_at':r.now(),
                    'replayed_calls':state['replayed'], 'resume_request':17,
                    'prior_http_attempts':self.http_counter,
                    'pending_request_had_no_http_attempt':True,
                    'helper_sha256':r.base.digest(Path(__file__).read_bytes()),
                    'original_artifacts':frozen,
                    'no_source_reallocation':True,
                })
            return super().call(label, payload, proof)

    source = inspect.getsource(r.live)
    tree = ast.parse(source)
    body = tree.body[0].body[0].body
    expected = [
        'base.require(not (run / "LIVE_STARTED.json").exists(), "already_started_no_automatic_replay")',
        'source_inventory.reserve(run, allocation["assignments"], source_inventory.collect_prior())',
        'base.write(run / "LIVE_STARTED.json", {"version": VERSION, "started_at": now(), "http_max": manifest["http_max"]})',
    ]
    expected_dumps = {ast.dump(ast.parse(text).body[0]) for text in expected}
    removed = [node for node in body if ast.dump(node) in expected_dumps]
    r.base.require(len(removed) == 3, 'runner_startup_changed')
    tree.body[0].body[0].body = [node for node in body if node not in removed]
    namespace = dict(r.__dict__)
    namespace['Caller'] = RecoveryCaller
    exec(compile(tree, str(DELIVERY/'runner.py')+':recovery_body', 'exec'), namespace)
    r.base.write = checked_write
    try:
        result = namespace['live'](RUN)
    except ReplayBoundary:
        print(json.dumps({'status':'REPLAY_PASS','replayed_calls':state['replayed'],
                          'next_request':17,'http_calls':0}))
        return
    finally:
        r.base.write = original_write
    original_write(RUN/'RECOVERY_COMPLETED.json', {'completed_at':r.now(),
        'status':result['status'],'replayed_calls':state['replayed'],
        'http_attempts':result['http_attempts']})
    print(json.dumps({'status':result['status'],'http_attempts':result['http_attempts']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args=parser.parse_args()
    main(not args.live)
