"""Exercise the formal scheduler, shared Miner and original HTTP client offline."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import shutil
import tempfile
import urllib.error
from unittest.mock import patch

from ..core import atomic_json, atomic_text, sha256_bytes
from .constants import SCHEDULE, ROOTS, ARMS, THREE_RUN, root_path
from .adapter import build_prompt
from ..h4_qa_mining_probe_v1.intent_miner import build_intent_prompt
from .dependencies import loaded_repository_files


class Response:
    status = 200
    def __init__(self, raw):
        self.raw = raw
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def getcode(self):
        return 200
    def read(self):
        return json.dumps({'choices':[{'message':{'content':self.raw},'finish_reason':'stop'}],
                           'usage':None}).encode()


class OfflineTokens:
    def __init__(self, run):
        self.run = run
    def check(self, *args, **kwargs):
        return {'status':'PASS'}
    def verify_usage(self, *args):
        return {'status':'UNKNOWN'}


def make_inputs(target, source):
    shutil.copytree(source/'inputs/shared', target/'inputs/shared')
    atomic_json(target/'audit/preflight_report.json', {'status':'PASS'})
    atomic_json(target/'audit/frozen_manifest.json', {'dry_run':True})
    atomic_json(target/'schedule.json', {'schedule':list(SCHEDULE)})
    pairings = []
    for row in SCHEDULE:
        rid,arm = row['root_id'],row['arm_id'];folder=target/'inputs/shared'/rid
        expanded,cards,history=[json.loads((folder/name).read_text()) for name in
                                ('expanded_dialogue_clean.json','observation_cards.json','history_only.json')]
        spec=build_prompt(arm,build_intent_prompt(expanded),cards,history,expanded=expanded,root_id=rid)
        atomic_text(target/'policy_snapshot/prompts'/arm/rid/'miner_prompt.txt',spec['prompt'])
        pairings.append({**row,'prompt_sha256':spec['prompt_sha256'],'max_output_tokens':4096,'maximum_logical_calls':1})
    atomic_text(target/'pairing_manifest.jsonl',''.join(json.dumps(r)+'\n' for r in pairings))


def exercise(source):
    from . import run as runner
    from .runtime import GuardrailStop
    outcomes=[]
    for scenario in ('success','malformed_json','known_429_exhausted','timeout_unknown','caught_pre_request_error'):
        with tempfile.TemporaryDirectory(prefix='basic-miner-v3-offline-') as temp:
            target=Path(temp);make_inputs(target,Path(source));calls=[]
            def mocked_http(request, **kwargs):
                payload=json.loads(request.data)
                assert payload['model']=='Qwen/Qwen3-14B' and payload['temperature']==.1
                assert payload['enable_thinking'] is False and payload['max_tokens']==4096
                assert 'seed' not in payload and len(payload['messages'])==1
                assert kwargs['timeout']==180
                calls.append({'payload_keys':sorted(payload),'prompt_sha256':sha256_bytes(payload['messages'][0]['content'].encode())})
                if scenario=='timeout_unknown':
                    raise TimeoutError('offline simulated uncertain completion')
                if scenario=='known_429_exhausted' and len(calls)<=3:
                    raise urllib.error.HTTPError('https://example.invalid',429,'offline known rejection',{'Retry-After':'0'},io.BytesIO(b''))
                if scenario=='malformed_json':
                    return Response('{"intents": [')
                prompt=payload['messages'][0]['content']
                rid=next(rid for rid in ROOTS if rid in prompt)
                prior=THREE_RUN/'arms/A_OFF/qa/shards/shard-00/roots'/rid/'raw/responses'/f'{rid}-intent-miner.json.txt'
                return Response(prior.read_text())
            def fake_freeze(run):
                return {'code':[]}
            stopped=False;reason=None
            with patch.object(runner,'verify_frozen',side_effect=fake_freeze), \
                 patch.object(runner,'TokenPreflight',OfflineTokens), \
                 patch.object(runner,'repository_execution_guard',lambda records:nullcontext()), \
                 patch('urllib.request.urlopen',side_effect=mocked_http), \
                 patch('time.sleep'), patch('builtins.print'):
                if scenario=='caught_pre_request_error':
                    # Exception (not BaseException) is swallowed by shared _call_json.
                    with patch.object(OfflineTokens,'check',side_effect=ValueError('offline frozen-token guard failure')):
                        try:runner.live(target)
                        except (GuardrailStop,Exception) as error:stopped=True;reason=str(error)
                else:
                    try:runner.live(target)
                    except (GuardrailStop,Exception) as error:stopped=True;reason=str(error)
                first_calls=len(calls)
                if scenario in ('success','malformed_json','known_429_exhausted') and not stopped:
                    runner.live(target)
                    assert len(calls)==first_calls,'completed checkpoint resend'
            reservations=list(target.glob('**/ledger/requests/*.json'))
            completed=list(target.glob('**/miner_generation_report.json'))
            if scenario in ('success','malformed_json'):
                passed=not stopped and len(calls)==4 and len(reservations)==4 and len(completed)==4
            elif scenario=='known_429_exhausted':
                passed=not stopped and len(calls)==6 and len(reservations)==4 and len(completed)==4
            elif scenario=='timeout_unknown':
                passed=stopped and len(calls)==1 and len(reservations)==1 and len(completed)==0
            else:
                passed=stopped and not calls and not reservations and not completed
            outcomes.append({'scenario':scenario,'status':'PASS' if passed else 'FAIL',
                'mock_http_attempts':len(calls),'reservations':len(reservations),'completed_groups':len(completed),
                'stopped':stopped,'reason':reason,'actual_network_calls':0})
    return {'status':'PASS' if all(r['status']=='PASS' for r in outcomes) else 'FAIL',
            'scenarios':outcomes,'repository_modules':sorted(map(str,loaded_repository_files())),
            'method':'Actual run.live, shared generate_root_intents, original stateless client/observer/ledger, both arms; only urllib transport and pre-freeze/token dependencies stubbed. No model requests.'}
