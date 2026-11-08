"""Prospective Gold-blind correction of letter-vs-I1 label instructions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def bind(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def save(name,obj):
    p=HERE/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    return bind(p)


def main():
    old=read('inputs/PUBLIC_QUESTIONS.json')
    public=json.loads(json.dumps(old))
    changes=[]
    for q in public['questions']:
        before=q['question']
        if q['question_type']=='ordering':
            q['question']=before.replace('Output the six letters.','Output the six displayed item IDs.').replace('output the six letters.','output the six displayed item IDs.')
        if before!=q['question']:
            changes.append({'candidate_id':q['candidate_id'],'before':before,'after':q['question']})
    assert len(changes)==55
    private=read('inputs_private/PRIVATE_QUESTIONS.json')
    questions={q['candidate_id']:q for q in public['questions']}
    for q in private['questions']:
        q['question']=questions[q['candidate_id']]['question']
    pub=save('inputs_v2/PUBLIC_QUESTIONS.json',public)
    priv=save('inputs_private_v2/PRIVATE_QUESTIONS.json',private)
    Path(priv['path']).chmod(0o600)
    log=save('FORMAT_NORMALIZATION_V2.json',{'status':'FROZEN_BEFORE_ANY_SOLVER_INFERENCE',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'original_public':bind(HERE/'inputs/PUBLIC_QUESTIONS.json'),
        'new_public':pub,'new_private':priv,'changed_questions':55,'semantic_content_or_gold_changed':False,
        'rule':'Only fixed letters-output instruction becomes displayed-item-IDs instruction for I1-I6 native presentation. Applied independently of arm and Gold.',
        'changes':changes,'script':bind(__file__),
        'prior_unexecuted_preparation':str(HERE/'runtime/direct/PREFLIGHT_FREEZE.json')})
    contract=read('EXECUTION_FREEZE.json')
    contract.update(schema_version='day3_solver187_diagnostic_v2',created_at_utc=datetime.now(timezone.utc).isoformat(),
        prior_unexecuted_contract=bind(HERE/'EXECUTION_FREEZE.json'),format_normalization=log,
        public_questions=pub,private_questions_binding=priv)
    save('EXECUTION_FREEZE_V2.json',contract)
    admission=read('CONTENT_ADMISSION.json')
    admission.update(public_questions=pub,private_questions=priv,format_normalization=log,
        scope='Exact accepted content/Gold with documented Gold-blind label-format correction; no new semantic content or review claim.')
    save('CONTENT_ADMISSION_V2.json',admission)
    print(json.dumps({'status':'PASS','format_normalized':55,'questions':187,'solver_calls':0}))


if __name__=='__main__':
    main()
