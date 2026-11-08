#!/usr/bin/env python3
"""Only the five never-verified F136202 final MC slots, using saved final Gold.

No planner, author, content revision or Gold call is available. Existing three
ordering judgments are preserved. Preparation and checks never dispatch calls.
"""
from __future__ import annotations
import argparse,ast,copy,fcntl,hashlib,inspect,json,os
from pathlib import Path
import sys,tempfile
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'scripts'))
import question_driver as q
import saved_revision_gold_verification as saved
import review_schema_adapter as rsa
import review_schema_adapter_v2 as v2
import f179186_saved_shape_gold_verification as guard
ROOT='cc-train-episode-136202';ARM='F'
SLOTS=('MC2','MC3','MC4','MC5','MC6');RETAIN=('ORDER1','ORDER2','ORDER3')
INPUT_WORK=BASE/'question_candidate_shape_continuation/136202/F'
PLAN_WORK=BASE/'question_plan_continuation/136202/F'
OUT=BASE/'question_saved_mc5_first_verification/136202/F'
SELECTION=BASE/'SOURCE_SELECTION_FINAL.json'
VARIANT='f136202_final_revision_exact_five_MC_first_verifier_list_keyed'
KIND='saved_final_MC_five_first_verifier'
NEW_CALL='q20_verify_f_136202_saved_mc5_first_01_revision_01'
OLD_CALL='q20_verify_f_136202_candidate_shape_01_revision_01'
INSPECTION={'path':str(BASE/'mechanical_audits/F136202_SAVED_FINAL_GOLD_EXACT_FIVE_FIRST_VERIFIER_INSPECTION.json'),'sha256':'bbe93abb0484dd56a2ab5570a68695100b2f76793247eead588a61f2a90ed5e2','bytes':61437}
CODE={'saved_revision_gold_verification.py':'d85e11b5db07befe1aab07ebdcc6fa238bc6f36dfff54a44e50c382a704ba5b0',
 'review_schema_adapter_v2.py':'3d8bf15b6e42655fb4ca5d53fc0fde00f2da7970f3dafae1660c53987b469645',
 'f179186_saved_shape_gold_verification.py':'04593b00fc5e0ccfaa36ac2e6415dd30a43b49ae7c7d70f7b07f611265278e54'}


def verify_code():
    saved.verify_code();v2.verify_code()
    for name,digest in CODE.items():
        if q.binding(BASE/'scripts'/name)['sha256']!=digest:raise RuntimeError('Frozen helper drift: '+name)
    q.verify_binding(INSPECTION)


def call_id(root,arm):
    if (root,arm)!=(ROOT,ARM):raise RuntimeError('Exact-five verifier identity drift')
    return NEW_CALL


def normalized_saved_gold(raw):
    """Invert five keyed representations and the already-saved v2 mapping."""
    if raw!=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json'):raise RuntimeError('Only exact saved final Gold supported')
    inherited,prior_changes=v2.normalize_blind_schema(raw)
    baseline=q.read(INPUT_WORK/'MECHANICAL_REVIEW_SCHEMA_INPUT_revision_01.json')
    if inherited!=baseline or len(prior_changes)!=24:raise RuntimeError('Existing frozen v2 normalization drift')
    value=copy.deepcopy(inherited);changes=[]
    for row in value['reviews']:
        reasons=row.get('item_reasons')
        if reasons is None:continue
        if not isinstance(reasons,list) or len(reasons)!=6:raise RuntimeError('Complete saved six-item lists required')
        ids=[r.get('option_id') if isinstance(r,dict) else None for r in reasons]
        if ids!=['X1','X2','X3','X4','X5','X6']:raise RuntimeError('Exact complete unique saved option order required')
        if any(set(r)!={'option_id','truth','rationale','evidence'} or type(r['truth']) is not bool
               or not isinstance(r['rationale'],str) or not r['rationale'].strip()
               or not isinstance(r['evidence'],list) or not r['evidence'] for r in reasons):
            raise RuntimeError('No missing truth, reason or evidence may be repaired')
        mapped={r['option_id']:{k:copy.deepcopy(v) for k,v in r.items() if k!='option_id'} for r in reasons}
        changes.append({'kind':'complete_list_to_option_id_keyed_dict','question_key':row['question_key'],
            'ordered_option_ids':ids,'original_value':copy.deepcopy(reasons),'normalized_value':copy.deepcopy(mapped),'all_values_changed':False})
        row['item_reasons']=mapped
    if len(changes)!=5:raise RuntimeError('Exactly five complete existing MC lists required')
    inverse=copy.deepcopy(value)
    for change in changes:
        row=next(r for r in inverse['reviews'] if r['question_key']==change['question_key'])
        row['item_reasons']=[{'option_id':key,**row['item_reasons'][key]} for key in change['ordered_option_ids']]
    if inverse!=baseline:raise RuntimeError('Whole existing normalized saved object inverse differs')
    for change in reversed(prior_changes):
        row=inverse['reviews'][change['review_index']]
        if change['kind']=='explicit_call_scope_attestation_copy':
            if row[change['field']] is not change['source_global_value']:raise RuntimeError('Explicit read declaration changed')
            del row[change['field']]
        elif change['kind']=='exact_observed_r03_key_alias':
            reverse={v:k for k,v in change['mapping'].items()}
            row['r03_necessity']={reverse.get(k,k):v for k,v in row['r03_necessity'].items()}
        else:raise RuntimeError('Unrecognized inherited transform')
    if inverse!=raw:raise RuntimeError('Whole actual authored saved Gold does not round-trip')
    return value,changes


def _source():
    _,sources=q.load_selection(SELECTION);return sources[ROOT]


def request_coverage(candidates,source):
    key_slots={q.anonymize_candidate(c)[0]['question_key']:c['slot_id'] for c in candidates}
    requests=saved.actual_requests(ROOT,ARM);records=[];per={s:{k:[] for k in requests} for s in q.SLOTS}
    for stage,items in requests.items():
        ids={r['call_id'] for r in items}
        if stage!='verify' and len(ids)!={'plan':1,'author':1,'revision':1,'gold':2}[stage]:raise RuntimeError('Actual stage budget drift: '+stage)
        if stage=='verify' and ids-{OLD_CALL,NEW_CALL}:raise RuntimeError('Unexpected verifier consumes scope')
        for key in sorted(ids):
            for path in sorted((BASE/'cli_calls'/key).glob('attempt_*/REQUEST.txt')):
                payload=saved.shape.request_payload(path.read_text());assignment=payload.get('assignment')
                if assignment:
                    if assignment!={'root_id':ROOT,'arm':ARM,'slots':list(q.SLOTS)}:raise RuntimeError('Original assignment drift')
                    slots=assignment['slots']
                else:
                    rows=payload.get('anonymous_questions',payload.get('review_cases'))
                    if payload.get('root_id')!=ROOT or not isinstance(rows,list):raise RuntimeError('Reviewer payload identity drift')
                    keys=[r.get('question_key') for r in rows]
                    if len(keys)!=len(set(keys)) or any(k not in key_slots for k in keys):raise RuntimeError('Unknown/duplicate reviewer key')
                    slots=[key_slots[k] for k in keys]
                    if payload.get('complete_source_and_warnings')!=q.source_view(source):raise RuntimeError('Complete reviewer source drift')
                if stage=='verify' and slots!=(list(RETAIN) if key==OLD_CALL else list(SLOTS)):
                    raise RuntimeError('Prior ORDER and new MC verifier scopes must be exactly disjoint')
                for slot in slots:per[slot][stage].append(key)
                records.append({'stage':stage,'call_id':key,'slot_ids':slots,'request_binding':q.binding(path)})
    for slot,stages in per.items():
        counts={k:len(set(v)) for k,v in stages.items()}
        expected={'plan':1,'author':1,'gold':2,'revision':1,'verify':int(slot in RETAIN or NEW_CALL in stages['verify'])}
        if counts!=expected:raise RuntimeError('Per-slot consumed budget drift: '+slot)
    if {r['call_id'] for r in requests['verify']}-{NEW_CALL}!={OLD_CALL}:raise RuntimeError('Actual ORDER verifier missing')
    return {'records':records,'per_slot_call_ids':per,'per_slot_counts':{s:{k:len(set(v)) for k,v in stages.items()} for s,stages in per.items()}}


def inspect_inputs(source):
    verify_code();inspection=q.read(INSPECTION['path'])
    for binding in inspection['input_bindings']:q.verify_binding(binding)
    candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json');reviews=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')['reviews']
    if (len(candidates)!=8 or candidates!=q.read(INPUT_WORK/'CANDIDATES_REVISION_01.json')
            or [c.get('slot_id') for c in candidates]!=list(q.SLOTS) or len(reviews)!=8):raise RuntimeError('Actual sole final eight drift')
    outcome=q.read(INPUT_WORK/'OUTCOME.json');ready=q.read(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json')
    if ((outcome.get('root_id'),outcome.get('arm'),outcome.get('source_binding'),outcome.get('max_content_revisions'))!=(ROOT,ARM,source['binding'],1)
            or outcome.get('status')!='COMPLETED_WITH_HOLES' or ready.get('final_work_dir')!=str(INPUT_WORK)):
        raise RuntimeError('Terminal source/revision identity drift')
    for key,value in ready.items():
        if key.endswith('_binding'):q.verify_binding(value)
    for candidate in candidates:
        row=next(r for r in reviews if r['candidate_id']==candidate['candidate_id'])
        if row.get('decision')!=('PASS' if candidate['slot_id'] in RETAIN else 'HOLD'):raise RuntimeError('Prior PASS/HOLD drift')
        if row['candidate_sha256']!=q.gate.fingerprint(candidate) or row['source_binding']!=source['binding']:raise RuntimeError('Review binding drift')
    receipts={}
    for name,path in [('initial',PLAN_WORK/'CANDIDATES_DRAFT_CALL.json'),('revision',INPUT_WORK/'CANDIDATES_REVISION_01_CALL.json'),
                      ('gold',INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json'),('old_verify',INPUT_WORK/'CONTENT_VERIFIER_CALL_revision_01.json')]:
        receipts[name],_=saved.verify_receipt(path)
    old=receipts['old_verify']
    if Path(old['parsed_response']['path']).parent.name!=OLD_CALL:raise RuntimeError('Existing ORDER verifier receipt drift')
    if len({r['backend']['thread_id'] for r in receipts.values()})!=4:raise RuntimeError('Prior threads not independent')
    revision_raw=q.read(receipts['revision']['parsed_response']['path'])
    if revision_raw!=q.read(INPUT_WORK/'CANDIDATES_REVISION_01_RESPONSE.json'):raise RuntimeError('Actual revision response drift')
    actual_final,revision_errors,_=saved.ORIGINAL_NORMALIZE_CANDIDATES(copy.deepcopy(revision_raw),ARM,ROOT,
        receipts['initial']['backend']['thread_id'],source,q.read(PLAN_WORK/'TASK_PLANS.json'),q.binding(SELECTION),
        receipts['revision']['backend']['thread_id'])
    if (revision_errors or actual_final!=candidates or any(not q.basic_solver_shape(c,source)
            or c.get('content_revision_count')!=1 or c.get('revision_author_thread_id')!=receipts['revision']['backend']['thread_id']
            for c in candidates)):raise RuntimeError('Final candidates differ from the actual sole authored revision')
    raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
    if raw!=q.read(receipts['gold']['parsed_response']['path']):raise RuntimeError('Actual saved final Gold receipt mismatch')
    actual=saved.shape.request_payload(Path(receipts['gold']['request']['path']).read_text())
    if actual!={'root_id':ROOT,'complete_source_and_warnings':q.source_view(source),'anonymous_questions':[q.anonymize_candidate(c)[0] for c in candidates]}:
        raise RuntimeError('Actual Gold request complete source/final eight drift')
    normalized,changes=normalized_saved_gold(raw);blind,errors=saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(normalized),candidates,source['source'])
    if errors or len(blind)!=8:raise RuntimeError('Saved final Gold still fails frozen evidence/answer parser')
    coverage=request_coverage(candidates,source)
    if any(NEW_CALL in row['verify'] for row in coverage['per_slot_call_ids'].values()) and not (OUT/'RECOVERY_INPUT_FREEZE.json').exists():
        raise RuntimeError('New verifier request predates exact input freeze')
    return {'schema_version':'fgr20_exact_five_saved_mc_first_verifier_inputs_v1','root_id':ROOT,'arm':ARM,
        'serialization_variant':VARIANT,'input_work_dir':str(INPUT_WORK),'source_binding':source['binding'],
        'input_bindings':inspection['input_bindings']+[q.binding(INSPECTION['path']),q.binding(SELECTION),q.binding(BASE/'SOURCE_FREEZE.json')],
        'actual_plan_path':str(PLAN_WORK/'TASK_PLANS.json'),'actual_plan_freeze_path':str(PLAN_WORK/'TASK_PLANS_FREEZE.json'),
        'saved_gold_receipt_binding':q.binding(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json'),
        'saved_gold_call_id':Path(receipts['gold']['parsed_response']['path']).parent.name,
        'actual_initial_author_thread_id':receipts['initial']['backend']['thread_id'],
        'actual_revision_author_thread_id':receipts['revision']['backend']['thread_id'],
        'actual_saved_gold_thread_id':receipts['gold']['backend']['thread_id'],
        'actual_existing_ORDER_verifier_thread_id':old['backend']['thread_id'],
        'existing_ORDER_verifier_call_id':OLD_CALL,'existing_ORDER_verifier_receipt_binding':q.binding(INPUT_WORK/'CONTENT_VERIFIER_CALL_revision_01.json'),
        'allowed_new_call_id':NEW_CALL,'new_verifier_exact_slot_ids':list(SLOTS),'retained_existing_review_slot_ids':list(RETAIN),
        'prior_blind_validation_errors':q.read(INPUT_WORK/'BLIND_GOLD_VALIDATION_revision_01.json')['errors'],
        'additional_exact_list_keyed_transformations':changes,'all_saved_final_gold_parser_errors':errors,'actual_request_coverage':coverage,
        'no_new_gold_call':True,'no_new_plan_or_author_call':True,'no_second_revision':True,'semantic_admission_asserted':False}


class AdaptReview(ast.NodeTransformer):
    def __init__(self):self.changes=[]
    def visit_Call(self,node):
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='rsa'
                and node.func.attr=='normalize_blind_aliases' and len(node.args)==1 and ast.unparse(node.args[0])=='raw_gold'):
            self.changes.append('only_saved_final_gold_complete_list_keyed_normalizer')
            return ast.copy_location(ast.Call(func=ast.Name(id='normalized_saved_gold',ctx=ast.Load()),args=node.args,keywords=[]),node)
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='q'
                and node.func.attr=='binding' and len(node.args)==1 and ast.unparse(node.args[0])=='rsa.__file__'):
            self.changes.append('honest_concrete_wrapper_normalization_binding');node.args[0]=ast.Name(id='__file__',ctx=ast.Load())
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and ast.unparse(node.targets[0])=='previous_threads':
            if not isinstance(node.value,ast.Set) or len(node.value.elts)!=3:raise RuntimeError('Independence hook drift')
            node.value.elts.append(ast.parse("frozen['actual_existing_ORDER_verifier_thread_id']",mode='eval').body)
            self.changes.append('also_exclude_existing_ORDER_verifier_thread')
        return self.generic_visit(node)


class ReadOnlyReplay(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Call(self,node):
        if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='q' and node.func.attr=='publish'):
            self.count+=1
            return ast.copy_location(ast.Constant(value=None),node)
        return self.generic_visit(node)


def build_review(read_only=False):
    verify_code();namespace=dict(vars(saved));namespace.update(__file__=__file__,normalized_saved_gold=normalized_saved_gold,call_id=call_id)
    tree=ast.parse(inspect.getsource(saved.review_batch));adapt=AdaptReview();tree=ast.fix_missing_locations(adapt.visit(tree))
    if adapt.changes!=['only_saved_final_gold_complete_list_keyed_normalizer','honest_concrete_wrapper_normalization_binding',
                         'honest_concrete_wrapper_normalization_binding','also_exclude_existing_ORDER_verifier_thread']:
        raise RuntimeError('Frozen verifier exact engineering hooks drift')
    elisions=0
    if read_only:
        transform=ReadOnlyReplay();tree=ast.fix_missing_locations(transform.visit(tree));elisions=transform.count
        if elisions!=9:raise RuntimeError('Frozen saved verifier read-only publication hook count drift')
    exec(compile(tree,'<f136202_exact_five_saved_first_verifier>','exec'),namespace);original=namespace['review_batch']
    def reviewer(root,arm,candidates,source,selection,out,frozen,caller):
        if ((root,arm)!=(ROOT,ARM) or [c['slot_id'] for c in candidates]!=list(SLOTS)
                or candidates!=[c for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id'] in SLOTS]):
            raise RuntimeError('Only exact existing five MC revisions may reach first verifier')
        return original(root,arm,candidates,source,selection,out,frozen,caller)
    return reviewer,{'changes':adapt.changes,'read_only_publish_elisions':elisions,
        'adapted_review_ast_sha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()}


class FirstOnlyCalls:
    def __init__(self):self.used=False
    def call(self,key,prompt):
        if self.used or key!=NEW_CALL:raise RuntimeError('Only exact-five first verifier invocation allowed')
        self.used=True;payload=saved.shape.request_payload(prompt)
        expected=[q.anonymize_candidate(c)[0]['question_key'] for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id'] in SLOTS]
        if [r.get('question_key') for r in payload.get('review_cases',[])]!=expected:raise RuntimeError('New request contains ORDER or lacks MC')
        directory=BASE/'cli_calls'/key
        if (directory/'RESULT.json').exists():
            receipt,actual=saved.verify_receipt(directory/'RESULT.json')
            if actual!=key or Path(receipt['request']['path']).read_text()!=prompt:raise RuntimeError('Completed verifier prompt/receipt drift')
            return self.accept(q.read(receipt['parsed_response']['path']),receipt,expected)
        if any(directory.glob('attempt_*/REQUEST.txt')) or any(directory.glob('attempt_*/START.json')):
            raise RuntimeError('Consumed/interrupted exact-five verifier lacks RESULT; no re-dispatch')
        response,receipt=q.cc.call(key,prompt)
        return self.accept(response,receipt,expected)
    @staticmethod
    def accept(response,receipt,expected):
        rows=q.extract_list(response,'reviews','answers')
        if len(rows)!=5 or {r.get('question_key') for r in rows}!=set(expected):
            raise RuntimeError('Consumed verifier response lacks exact unique five MC review rows')
        return response,receipt


def exact_copy(source,target):
    raw=source.read_bytes()
    if target.exists():
        if target.read_bytes()!=raw:raise RuntimeError('Exact archived/current bytes drift')
    else:
        with target.open('xb') as handle:handle.write(raw)


def publish_ready(out,frozen,data):
    candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json');prior=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')
    if len(data['reviews'])!=5 or {r['candidate_id'] for r in data['reviews']}!={c['candidate_id'] for c in candidates if c['slot_id'] in SLOTS}:
        raise RuntimeError('First-verifier exact-five final review coverage drift')
    newer={r['candidate_id']:r for r in data['reviews']};old={r['candidate_id']:r for r in prior['reviews']}
    reviews=[newer[c['candidate_id']] if c['slot_id'] in SLOTS else copy.deepcopy(old[c['candidate_id']]) for c in candidates]
    exact_copy(INPUT_WORK/'CANDIDATES_FINAL.json',out/'CANDIDATES_FINAL.json')
    exact_copy(INPUT_WORK/'CONTENT_REVIEW_FINAL.json',out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json')
    audits=[q.audit_candidate(c,SELECTION) for c in candidates]
    gates=[q.gate.validate_review(c,next(r for r in reviews if r['candidate_id']==c['candidate_id']),SELECTION) for c in candidates]
    if any(not g['admitted'] for c,g in zip(candidates,gates) if c['slot_id'] in RETAIN):raise RuntimeError('Existing ORDER review no longer admitted')
    q.publish(out/'CONTENT_REVIEW_FINAL.json',{'reviews':reviews,'mechanical_audits':audits,'all_reviews_ai_family_not_human':True,
        'review_stage':'revision_01','fresh_verifier_slot_ids':list(SLOTS),'retained_existing_review_slot_ids':list(RETAIN),
        'prior_review_archive_binding':q.binding(out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json')})
    failures=copy.deepcopy(data['slot_failures']);q.publish(out/'SLOT_FAILURES.json',failures)
    q.publish(out/'FINAL_REVIEW_GATE.json',{'results':gates,'semantic_admission_basis':'Existing three ORDER judgments retained; five MC judgments from their first fresh independent verifier only.'})
    retained={c['candidate_id']:old[c['candidate_id']] for c in candidates if c['slot_id'] in RETAIN}
    q.publish(out/'RETAINED_ORDER_REVIEWS.json',{'reviews':retained,'original_full_review_archive_binding':q.binding(out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json'),
        'existing_verifier_receipt_binding':frozen['existing_ORDER_verifier_receipt_binding'],'ORDER_verifier_repeated':False})
    outcome={'schema_version':'fgr20_question_root_arm_v1','root_id':ROOT,'arm':ARM,'source_binding':frozen['source_binding'],
        'status':'COMPLETED_WITH_HOLES' if failures else 'COMPLETED','candidate_ids':[c['candidate_id'] for c in candidates],
        'candidate_bindings':[{'candidate_id':c['candidate_id'],'candidate_sha256':q.gate.fingerprint(c)} for c in candidates],
        'slot_failures':failures,'calls':[data['verification_call']] if data['verification_call'] else [],'max_content_revisions':1,
        'continuation_kind':KIND,'review_stage':'revision_01','final_candidates_binding':q.binding(out/'CANDIDATES_FINAL.json'),
        'review_binding':q.binding(out/'CONTENT_REVIEW_FINAL.json'),'slot_failures_binding':q.binding(out/'SLOT_FAILURES.json'),
        'plan_freeze_binding':q.binding(frozen['actual_plan_freeze_path']),'no_new_gold_call':True,'no_new_plan_or_author_call':True,
        'no_second_revision':True,'only_new_verifier_slot_ids':list(SLOTS),'retained_existing_review_slot_ids':list(RETAIN)}
    q.publish(out/'OUTCOME.json',outcome);q.publish(out/'CONTINUATION_INTEGRATION_OUTCOME.json',outcome)
    row={'schema_version':'fgr20_question_continuation_finalizer_ready_v1','root_id':ROOT,'arm':ARM,'continuation_kind':KIND,
        'serialization_variant':VARIANT,'review_stage':'revision_01','original_work_dir':str(INPUT_WORK),'initial_work_dir':str(INPUT_WORK),
        'main_original_work_dir':str(BASE/'question_authoring/F/roots/136202'),'final_work_dir':str(out),
        'actual_plan_path':frozen['actual_plan_path'],'actual_plan_freeze_path':frozen['actual_plan_freeze_path'],
        'supersedes_ready_binding':q.binding(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json'),'original_outcome_binding':q.binding(INPUT_WORK/'OUTCOME.json'),
        'final_outcome_binding':q.binding(out/'CONTINUATION_INTEGRATION_OUTCOME.json'),'driver_outcome_binding':q.binding(out/'OUTCOME.json'),
        'final_candidates_binding':q.binding(out/'CANDIDATES_FINAL.json'),'final_review_binding':q.binding(out/'CONTENT_REVIEW_FINAL.json'),
        'slot_failures_binding':q.binding(out/'SLOT_FAILURES.json'),'input_freeze_binding':q.binding(out/'RECOVERY_INPUT_FREEZE.json'),
        'saved_gold_receipt_reference_binding':q.binding(out/'SAVED_REVISION_GOLD_REFERENCE.json'),
        'normalization_binding':q.binding(out/'MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json'),
        'gate_validation_binding':q.binding(out/'FINAL_REVIEW_GATE.json'),'driver_binding':q.binding(__file__),'adapter_binding':q.binding(rsa.__file__),
        'prior_final_reviews_archive_binding':q.binding(out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json'),
        'retained_ORDER_reviews_binding':q.binding(out/'RETAINED_ORDER_REVIEWS.json'),'existing_ORDER_verifier_receipt_binding':frozen['existing_ORDER_verifier_receipt_binding'],
        'new_verifier_exact_slot_ids':list(SLOTS),'retained_existing_review_slot_ids':list(RETAIN),
        'no_new_gold_call':True,'no_new_plan_or_author_call':True,'no_second_revision':True,'initial_candidate_and_reviews_not_overwritten':True,
        'final_revision_retained_without_initial_fallback':True,'final_candidate_bytes_unchanged':True,'existing_ORDER_review_objects_unchanged':True}
    q.publish(out/'CONTINUATION_FINALIZER_READY.json',row);return row


def validate_terminal_ready(path=None):
    """Read-only terminal interface; prior ordering judgments never rejudged."""
    path=Path(path) if path is not None else OUT/'CONTINUATION_FINALIZER_READY.json'
    row=q.read(path);out=path.parent
    if ((row.get('root_id'),row.get('arm'),row.get('continuation_kind'),row.get('serialization_variant'),
         row.get('final_work_dir'),row.get('original_work_dir'))!=(ROOT,ARM,KIND,VARIANT,str(out),str(INPUT_WORK))):
        raise RuntimeError('Exact-five terminal identity drift')
    for name,value in row.items():
        if name.endswith('_binding'):q.verify_binding(value)
    if (row['driver_binding']!=q.binding(__file__) or row['adapter_binding']!=q.binding(rsa.__file__)
            or row['supersedes_ready_binding']!=q.binding(INPUT_WORK/'CONTINUATION_FINALIZER_READY.json')
            or row['new_verifier_exact_slot_ids']!=list(SLOTS) or row['retained_existing_review_slot_ids']!=list(RETAIN)):
        raise RuntimeError('Exact-five terminal driver/supersession/scope drift')
    source=_source();frozen=q.read(out/'RECOVERY_INPUT_FREEZE.json')
    for value in frozen['input_bindings']:q.verify_binding(value)
    if frozen['source_binding']!=source['binding'] or frozen['serialization_variant']!=VARIANT:raise RuntimeError('Terminal input freeze drift')
    candidates=q.read(out/'CANDIDATES_FINAL.json');reviews=q.read(out/'CONTENT_REVIEW_FINAL.json')['reviews']
    old=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')['reviews']
    if ((out/'CANDIDATES_FINAL.json').read_bytes()!=(INPUT_WORK/'CANDIDATES_FINAL.json').read_bytes()
            or (out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json').read_bytes()!=(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').read_bytes()):
        raise RuntimeError('Candidate or original review archive byte drift')
    if len(reviews)!=8 or {r['candidate_id'] for r in reviews}!={c['candidate_id'] for c in candidates}:raise RuntimeError('Final review coverage drift')
    for c in candidates:
        review=next(r for r in reviews if r['candidate_id']==c['candidate_id'])
        if c['slot_id'] in RETAIN and review!=next(r for r in old if r['candidate_id']==c['candidate_id']):raise RuntimeError('Existing ORDER judgment changed')
        if review['candidate_sha256']!=q.gate.fingerprint(c) or review['author_id']!=c['author_id'] or review['source_binding']!=source['binding']:
            raise RuntimeError('Final review fingerprint/author/source drift')
        result=q.gate.validate_review(c,review,SELECTION)
        if review['decision']=='PASS' and not result['admitted']:raise RuntimeError('Claimed PASS fails frozen gate')
    normalization=q.read(out/'MECHANICAL_SAVED_REVISION_GOLD_NORMALIZATION.json')
    mapped,changes=normalized_saved_gold(q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json'))
    if (q.read(out/'MECHANICAL_SAVED_REVISION_GOLD_INPUT.json')!=mapped or normalization.get('changes')!=changes
            or normalization.get('adapter_binding')!=q.binding(__file__)):raise RuntimeError('Exact saved Gold normalization drift')
    reference=q.read(out/'SAVED_REVISION_GOLD_REFERENCE.json')
    if (reference.get('raw_binding')!=q.binding(INPUT_WORK/'BLIND_GOLD_revision_01.json')
            or reference.get('call_receipt_binding')!=q.binding(INPUT_WORK/'BLIND_GOLD_CALL_revision_01.json')
            or reference.get('copied_as_new_call') is not False):raise RuntimeError('Actual saved Gold reference drift')
    coverage=request_coverage(candidates,source)
    receipt=out/'CONTENT_VERIFIER_CALL_revision_01.json'
    actual_response=actual_receipt=None
    if receipt.exists():
        actual,key=saved.verify_receipt(receipt)
        if key!=NEW_CALL:raise RuntimeError('Only new exact-five first verifier receipt allowed')
        payload=saved.shape.request_payload(Path(actual['request']['path']).read_text())
        expected=saved.build_verify_rows([c for c in candidates if c['slot_id'] in SLOTS],
            saved.ORIGINAL_NORMALIZE_BLIND(copy.deepcopy(mapped),candidates,source['source'])[0],SELECTION)
        if payload!={'root_id':ROOT,'complete_source_and_warnings':q.source_view(source),'review_cases':expected}:raise RuntimeError('Actual new exact-five verifier request drift')
        if actual['backend']['thread_id'] in {frozen[k] for k in ('actual_initial_author_thread_id','actual_revision_author_thread_id','actual_saved_gold_thread_id','actual_existing_ORDER_verifier_thread_id')}:
            raise RuntimeError('New verifier is not fresh independent')
        actual_response=q.read(actual['parsed_response']['path']);actual_receipt=actual
        if q.read(out/'CONTENT_VERIFIER_revision_01.json')!=actual_response:raise RuntimeError('Saved verifier response differs from actual RESULT')
        normalized_response,response_changes=rsa.normalize_blind_aliases(copy.deepcopy(actual_response))
        if q.read(out/'MECHANICAL_VERIFIER_SCHEMA_INPUT_revision_01.json')!=normalized_response:raise RuntimeError('Verifier schema normalization values drift')
        schema_log=q.read(out/'MECHANICAL_VERIFIER_SCHEMA_NORMALIZATION_revision_01.json')
        if schema_log['changes']!=response_changes or schema_log['adapter_binding']!=q.binding(__file__):raise RuntimeError('Verifier concrete normalization log drift')
    else:
        if not (out/'CONTENT_VERIFIER_FAILURE_revision_01.json').is_file():raise RuntimeError('No verifier receipt requires explicit actual failure record')
        if any(r['decision']!='HOLD' for c in candidates if c['slot_id'] in SLOTS for r in reviews if r['candidate_id']==c['candidate_id']):
            raise RuntimeError('No receipt permits only HOLD for the five never-reviewed MC')
    replay,_=build_review(read_only=True)
    class SavedOnly:
        def call(self,key,prompt):
            if key!=NEW_CALL:raise RuntimeError('Read-only replay call identity drift')
            if actual_receipt is None:raise RuntimeError('READ_ONLY_EXISTING_FAILURE_REPLAY_NO_DISPATCH')
            if prompt!=Path(actual_receipt['request']['path']).read_text():raise RuntimeError('Read-only replay prompt drift')
            return copy.deepcopy(actual_response),copy.deepcopy(actual_receipt)
    result=replay(ROOT,ARM,[c for c in candidates if c['slot_id'] in SLOTS],source,SELECTION,out,frozen,SavedOnly())
    if result['reviews']!=[r for c in candidates if c['slot_id'] in SLOTS for r in reviews if r['candidate_id']==c['candidate_id']]:
        raise RuntimeError('Final MC judgments differ from frozen actual saved verifier normalization')
    if result['slot_failures']!=q.read(out/'SLOT_FAILURES.json'):raise RuntimeError('MC failure ledger differs from actual normalized review')
    bindings=[q.binding(p) for p in sorted(out.glob('*.json'))]
    bindings+=frozen['input_bindings']
    return {'record':{**row,'ready_binding':q.binding(path),'ready_path':str(path.resolve())},
            'input_bindings':bindings,'coverage':coverage,'semantic_admission_authored':False}


def prepare(source):
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'RECOVERY_INPUT_FREEZE.json'
    if path.exists():
        frozen=q.read(path)
        if ((frozen.get('root_id'),frozen.get('arm'),frozen.get('serialization_variant'),frozen.get('source_binding'))!=(ROOT,ARM,VARIANT,source['binding'])):
            raise RuntimeError('Exact-five input freeze identity drift')
        for binding in frozen['input_bindings']:q.verify_binding(binding)
        request_coverage(q.read(INPUT_WORK/'CANDIDATES_FINAL.json'),source);return frozen
    frozen=inspect_inputs(source);q.publish(path,frozen);return frozen


def assert_idle():
    guard.assert_idle();parents=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=guard.decode_proc_cmdline((proc/'cmdline').read_bytes())
            if not argv or 'python' not in Path(argv[0]).name or 'run' not in argv:continue
            script=next((x for x in argv[1:] if x.endswith('.py')),None)
            if script and Path(script).name==Path(__file__).name:
                actual=(Path(script) if Path(script).is_absolute() else (proc/'cwd').resolve()/script).resolve()
                if actual==Path(__file__).resolve() and (proc/'stat').read_text().split()[2]!='Z':parents.append(int(proc.name))
        except (OSError,ValueError):continue
    if parents:raise RuntimeError('Another exact-five parent is active: '+str(parents))


def run(source,do_run):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.continuation.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ready_path=OUT/'CONTINUATION_FINALIZER_READY.json'
        if ready_path.exists():return validate_terminal_ready(ready_path)['record']
        if do_run and (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content already frozen')
        frozen=prepare(source);reviewer,clone=build_review()
        q.publish(OUT/'DRIVER_ADAPTATION.json',{'schema_version':'fgr20_exact_five_first_verifier_engineering_v1','wrapper_binding':q.binding(__file__),
            'saved_helper_binding':q.binding(saved.__file__),'RAM_adapter_binding':q.binding(rsa.__file__),'v2_adapter_binding':q.binding(v2.__file__),
            'input_freeze_binding':q.binding(OUT/'RECOVERY_INPUT_FREEZE.json'),'clone_changes':clone,'only_new_call_id':NEW_CALL,
            'new_verifier_exact_slot_ids':list(SLOTS),'existing_ORDER_never_reverified':True,'no_new_gold_author_plan_or_revision':True,
            'semantic_admission_asserted':False})
        if not do_run:return {'status':'PREPARED_ENGINEERING_ONLY','input_freeze_binding':q.binding(OUT/'RECOVERY_INPUT_FREEZE.json'),'semantic_calls_started':0}
        if q.cc.stop_generation():raise RuntimeError('Generation cutoff before exact-five verifier')
        assert_idle();handoff=guard.serial7_handoff();q.publish(OUT/'SERIAL7_TERMINAL_PREREQUISITE.json',{'handoff_binding':handoff,'all_parents_and_registry_idle_verified':True})
        with (saved.ROOT/'.family_call_start.lock').open('a+') as family:
            fcntl.flock(family,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_idle()
            if q.cc.stop_generation() or (BASE/'CONTENT_FREEZE.json').exists():raise RuntimeError('Content/generation cutoff at startup')
            candidates=[c for c in q.read(INPUT_WORK/'CANDIDATES_FINAL.json') if c['slot_id'] in SLOTS]
            data=reviewer(ROOT,ARM,candidates,source,SELECTION,OUT,frozen,FirstOnlyCalls())
        request_coverage(q.read(INPUT_WORK/'CANDIDATES_FINAL.json'),source)
        publish_ready(OUT,frozen,data);return validate_terminal_ready()['record']


def self_check(source):
    before=saved.actual_requests(ROOT,ARM);frozen=inspect_inputs(source);candidates=q.read(INPUT_WORK/'CANDIDATES_FINAL.json')
    mapped,changes=normalized_saved_gold(q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json'));assert len(changes)==5
    for label in ('missing_reason','extra_reason','duplicate_id','missing_evidence','changed_truth'):
        raw=q.read(INPUT_WORK/'BLIND_GOLD_revision_01.json')
        if label=='missing_reason':raw['reviews'][0]['item_reasons'].pop()
        elif label=='extra_reason':raw['reviews'][0]['item_reasons'].append(copy.deepcopy(raw['reviews'][0]['item_reasons'][0]))
        elif label=='duplicate_id':raw['reviews'][0]['item_reasons'][1]['option_id']='X1'
        elif label=='missing_evidence':del raw['reviews'][0]['item_reasons'][0]['evidence']
        else:raw['reviews'][0]['item_reasons'][0]['truth']=not raw['reviews'][0]['item_reasons'][0]['truth']
        try:normalized_saved_gold(raw)
        except RuntimeError:pass
        else:raise AssertionError('Changed saved schema/value accepted: '+label)
    reviewer,clone=build_review();calls=[]
    class Mock:
        def call(self,key,prompt):
            assert key==NEW_CALL;payload=saved.shape.request_payload(prompt)
            assert payload['root_id']==ROOT and payload['complete_source_and_warnings']==q.source_view(source)
            assert [r['question_key'] for r in payload['review_cases']]==[q.anonymize_candidate(c)[0]['question_key'] for c in candidates if c['slot_id'] in SLOTS]
            calls.append({'call_id':key,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
            raise RuntimeError('ENGINEERING_MOCK_EXACT_FIVE_FIRST_VERIFIER_NO_DISPATCH')
    cache=BASE/'runtime_cache/f136202_mc5_selfcheck';cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mc5_',dir=cache) as directory:
        out=Path(directory);q.publish(out/'RECOVERY_INPUT_FREEZE.json',frozen)
        data=reviewer(ROOT,ARM,[c for c in candidates if c['slot_id'] in SLOTS],source,SELECTION,out,frozen,Mock())
        assert len(calls)==1 and len(data['reviews'])==5 and all(r['decision']=='HOLD' for r in data['reviews'])
        assert len(data['slot_failures'])==5 and all(not row['admitted'] for row in data['gate_validations'])
        ready=publish_ready(out,frozen,data);reviews=q.read(out/'CONTENT_REVIEW_FINAL.json')['reviews'];old=q.read(INPUT_WORK/'CONTENT_REVIEW_FINAL.json')['reviews']
        assert [r for r in reviews if r['candidate_id'].endswith(RETAIN)]==[r for r in old if r['candidate_id'].endswith(RETAIN)]
        assert (out/'CANDIDATES_FINAL.json').read_bytes()==(INPUT_WORK/'CANDIDATES_FINAL.json').read_bytes()
        assert (out/'ORIGINAL_FINAL_REVIEWS_ARCHIVE.json').read_bytes()==(INPUT_WORK/'CONTENT_REVIEW_FINAL.json').read_bytes()
        assert ready['serialization_variant']==VARIANT and q.read(out/'OUTCOME.json')['status']=='COMPLETED_WITH_HOLES'
        before_bytes={p.name:p.read_bytes() for p in out.glob('*.json')}
        assert validate_terminal_ready(out/'CONTINUATION_FINALIZER_READY.json')['record']['root_id']==ROOT
        assert before_bytes=={p.name:p.read_bytes() for p in out.glob('*.json')}
    assert saved.actual_requests(ROOT,ARM)==before
    return {'status':'PASS','semantic_calls_started':0,'root_id':ROOT,'arm':ARM,'exact_first_verifier_slots':list(SLOTS),
        'prior_ORDER_verifier_never_repeated':True,'prior_three_ORDER_review_objects_and_archive_bytes_unchanged':True,
        'final_eight_candidate_bytes_unchanged':True,'saved_final_Gold_full_inverse_verified':True,'additional_list_keyed_transformations':5,
        'frozen_final_Gold_evidence_and_answer_parser_errors':[],'negative_actual_saved_schema_fixtures_rejected':5,
        'mock_only_first_MC_verifier':calls,'mock_MC_HOLD_count':5,'prior_ORDER_PASS_count_preserved':3,
        'request_registry_unchanged':True,'new_plan_author_Gold_revision_calls':0,'semantic_admission_asserted':False,'clone':clone}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['inspect','prepare','run','self-check']);args=parser.parse_args()
    q.SOURCE_SELECTION_PATH=SELECTION;q.SOURCE_FREEZE_PATH=BASE/'SOURCE_FREEZE.json';source=_source()
    value=self_check(source) if args.command=='self-check' else inspect_inputs(source) if args.command=='inspect' else run(source,args.command=='run')
    print(json.dumps(value,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
