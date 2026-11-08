"""Bounded same-input template comparison, with immutable arm inputs."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import random

import generate as generation

HERE = Path(__file__).resolve().parent
ROOT = "cc-train-episode-104814"
PAIR = "root104814_templates_01"
SOURCE = HERE.parent / "h4_sequential_ab_v43/runs/fresh_b2_grounded_01/CANDIDATE_HISTORY.json"
PRIOR = HERE.parent / "h4_sequential_ab_v44_qa/runs/root104814_prompt_repair_01_evaluation/training_pool/TRAINING_POOL.jsonl"
DIRECTORY = HERE / "runs" / PAIR
read, write, require = generation.read, generation.write, generation.require


def arm_run(arm):
    return HERE / "runs" / (PAIR + "_" + arm + "_generation")


def payload_data(payload):
    return json.JSONDecoder().raw_decode(payload["messages"][0]["content"].split("\nINPUT_JSON\n", 1)[1])[0]


def same_inputs(left, right, feedback_left, feedback_right):
    require(payload_data(left) == payload_data(right), "paired_initial_context_differs")
    require({k:v for k,v in left.items() if k != "messages"}
            == {k:v for k,v in right.items() if k != "messages"}, "paired_decoder_differs")
    require(feedback_left == feedback_right, "paired_frozen_feedback_differs")
    require(left["messages"][0]["content"].count(feedback_left) == 1
            and right["messages"][0]["content"].count(feedback_right) == 1, "paired_feedback_missing_or_repeated")


def prepare():
    require(not DIRECTORY.exists(), "immutable_pair_already_exists")
    initial = {}; manifests = {}
    for arm in generation.PROMPT_ARMS:
        run = arm_run(arm)
        generation.prepare(run, SOURCE, roots=[ROOT], slots=2, prior_pools=[PRIOR],
                           prompt_arm=arm, comparison_id=PAIR)
        manifests[arm] = read(run / "MANIFEST.json")
        initial[arm] = {}
        for kind in generation.TYPES:
            context = generation.initial_context(run, ROOT, kind)
            payload, receipt = generation.make_payload(run, ROOT, kind, context)
            proof = {"root_id":ROOT, "role":"question_generator", "question_type":kind,
                     "feedback_receipt":receipt, "payload_sha256":generation.base.digest(payload)}
            generation.verify_prompt_request(run, payload, proof)
            initial[arm][kind] = payload
            write(DIRECTORY / "initial" / arm / (kind + ".json"), {"payload":payload,"proof":proof})
    feedback = {arm:(arm_run(arm)/"feedback"/ROOT/"writer__M_clean_policy_feedback.txt").read_text()
                for arm in generation.PROMPT_ARMS}
    checks = []
    for kind in generation.TYPES:
        same_inputs(initial["old"][kind],initial["current"][kind],feedback["old"],feedback["current"])
        checks.append({"question_type":kind,"identical_input_json":True,"identical_parameters":True,
                       "input_json_sha256":generation.base.digest(payload_data(initial['old'][kind]))})
    order = list(generation.PROMPT_ARMS)
    random.Random(20261011).shuffle(order)
    files = [p for p in DIRECTORY.rglob("*.json")]
    files += [arm_run(arm)/"MANIFEST.json" for arm in generation.PROMPT_ARMS]
    files += [HERE / "BASELINE_FILES.json", HERE / "README.md"]
    write(DIRECTORY / "COMPARISON.json", {"version":"v46_paired_templates_v1", "root_id":ROOT,
        "comparison_id":PAIR,"arms":list(generation.PROMPT_ARMS), "arm_execution_order":order,
        "order_seed":20261011,"checks":checks,
        "source_body_tokens":4566,"source_turns":156,"feedback_identical":True,
        "source_sha256":manifests['old']['source_rows'][0]['source_sha256'],
        "prior_identical":read(arm_run('old')/'prior'/(ROOT+'.json')) == read(arm_run('current')/'prior'/(ROOT+'.json')),
        "reviewer_held_constant":"v45 current reviewer",
        "limits":{"slots":8,"qwen_http_max":12,"solver_full_max":120},
        "cross_arm_outputs_in_prompts":False,"training_started":False,
        "scope":"adapted_template_packages_not_exact_historical_replay_or_tail_only_ablation",
        "files":generation.base.bindings(files)})
    return verify()


def verify():
    pair = read(DIRECTORY / "COMPARISON.json")
    require(pair["prior_identical"] and pair["feedback_identical"], "paired_inputs_not_equal")
    generation.base.verify_bindings(pair["files"])
    generation.base.verify_bindings(read(HERE / "BASELINE_FILES.json")["files"])
    for arm in pair['arms']:
        manifest = generation.verify(arm_run(arm))
        require(manifest['prompt_arm']==arm and manifest['comparison_id']==PAIR, "arm_identity_changed")
        for kind in generation.TYPES:
            frozen = read(DIRECTORY/"initial"/arm/(kind+'.json'))
            payload, receipt = generation.make_payload(arm_run(arm),ROOT,kind,
                generation.initial_context(arm_run(arm),ROOT,kind))
            require(frozen['payload']==payload and frozen['proof']['feedback_receipt']==receipt,
                    "initial_payload_changed_after_freeze")
    return pair


def live():
    pair = verify()
    require(not (DIRECTORY/'STARTED.json').exists(), "comparison_already_started_no_replay")
    write(DIRECTORY/'STARTED.json',{'started_at_utc':generation.v15.now(),'order':pair['arm_execution_order']})
    reports = {}
    for arm in pair['arm_execution_order']:
        verify()
        print(json.dumps({'event':'arm_started','arm':arm}),flush=True)
        reports[arm] = generation.generate(arm_run(arm))
        for kind, code in [('multi_select','multi'),('ordering','order')]:
            matches = list((arm_run(arm)/'requests').glob('*-merged-'+code+'-generate-1/REQUEST.json'))
            require(len(matches)==1,'initial_actual_request_missing')
            frozen = read(DIRECTORY/'initial'/arm/(kind+'.json'))
            require(read(matches[0])==frozen['payload'],'actual_initial_request_not_pre_frozen')
        print(json.dumps({'event':'arm_complete','arm':arm,'ready_for_solver':reports[arm]['ready_for_solver']}),flush=True)
    verify()
    write(DIRECTORY/'COMPLETE.json',{'status':'GENERATION_COMPARISON_COMPLETE',
        'ready_for_solver':{arm:r['ready_for_solver'] for arm,r in reports.items()},
        'training_started':False})
    return read(DIRECTORY/'COMPLETE.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','live'))
    args=parser.parse_args()
    result={'prepare':prepare,'verify':verify,'live':live}[args.action]()
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2),flush=True)
