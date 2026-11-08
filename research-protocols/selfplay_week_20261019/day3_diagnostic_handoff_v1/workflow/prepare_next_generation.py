#!/usr/bin/env python3
"""Prepare an immutable feedback input packet after fresh runtime validation.

This program reads artifacts. It does not call an LLM, add roots or author questions.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

class GateRejected(RuntimeError): pass

def load_validator():
    path=Path(__file__).with_name('validate_handoff.py')
    spec=importlib.util.spec_from_file_location('fresh_handoff_validator',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,path

def prepare(state_path, output_dir, scope):
    state_path=Path(state_path).resolve();output_dir=Path(output_dir).resolve()
    if not state_path.is_file(): raise GateRejected('HANDOFF_STATE_MISSING')
    if output_dir.exists(): raise GateRejected('OUTPUT_EXISTS_IMMUTABLE_NO_OVERWRITE')
    if scope not in {'exploratory','formal'}: raise GateRejected('SCOPE_NOT_ALLOWED')
    validator,validator_path=load_validator()
    sha=validator.sha
    def bind(p):return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}
    def read(p):return json.loads(p.read_text())
    def verified(b):
        p=Path(b['path'])
        if not p.is_absolute():p=state_path.parent/p
        p=p.resolve()
        if not p.is_file():raise GateRejected('BOUND_FILE_MISSING:'+str(p))
        if sha(p)!=b['sha256']:raise GateRejected('BOUND_SHA_MISMATCH:'+str(p))
        if 'bytes' in b and p.stat().st_size!=b['bytes']:raise GateRejected('BOUND_BYTES_MISMATCH:'+str(p))
        return p
    state=read(state_path)
    fresh=validator.validate(state_path)
    if fresh.get('failures') or fresh.get('status')=='FAIL':raise GateRejected('FRESH_HANDOFF_VALIDATION_FAILED:'+json.dumps(fresh.get('failures'),ensure_ascii=False))
    gates=fresh.get('gates',{})
    if not gates.get('local_feedback_ready') or not gates.get('local_feedback_ready_scope'):raise GateRejected('LOCAL_FEEDBACK_NOT_READY')
    # This tool deliberately does not manufacture a formal arm out of local cards.
    if scope=='formal':
        if not gates.get('formal_feedback_B_go'):raise GateRejected('FORMAL_B_NO_GO_REPRODUCED_AND_REALIZATION_GATE_NOT_MET')
        raise GateRejected('FORMAL_SCOPE_REQUIRES_SEPARATE_MATCHED_AB_EXECUTION_TOOL')
    snapshot_path=verified(state['artifacts']['feedback_snapshot'])
    snapshot=read(snapshot_path)
    prior_validation_path=state_path.parent/'workflow/VALIDATION.json'
    if not prior_validation_path.is_file():raise GateRejected('SAVED_VALIDATION_MISSING')
    saved_validation=read(prior_validation_path)
    if saved_validation.get('state_sha256')!=sha(state_path):raise GateRejected('SAVED_VALIDATION_STATE_MISMATCH')
    if saved_validation.get('failures') or not saved_validation.get('gates',{}).get('local_feedback_ready'):raise GateRejected('SAVED_VALIDATION_NOT_READY')
    admitted=set(gates['local_feedback_ready_scope'])
    selected=[];excluded=[];actually_read=[bind(state_path),bind(snapshot_path),bind(prior_validation_path),bind(validator_path)]
    for card in snapshot.get('entries',[]):
        if card.get('intended_feedback_consumer')!='exploratory_train_challenger':
            excluded.append({'feedback_id':card['feedback_id'],'consumer':card.get('intended_feedback_consumer'),'reason':'Separate diagnostic interface quality scope; not used for dialogue or question generation.'});continue
        if card.get('source_verified_operational_feedback') is not True:raise GateRejected('CARD_NOT_SOURCE_VERIFIED:'+card['feedback_id'])
        if card.get('version')!=snapshot.get('version'):raise GateRejected('CARD_VERSION_MISMATCH:'+card['feedback_id'])
        if card.get('formal_reproduced') or card.get('qualified_for_formal_B_target_quota'):raise GateRejected('EXPLORATORY_PACKET_UNEXPECTED_FORMAL_CLAIM:'+card['feedback_id'])
        support=set(card.get('evidence_candidate_ids',[]))
        if not support or not support<=admitted:raise GateRejected('CARD_SUPPORT_OUTSIDE_READY_SCOPE:'+card['feedback_id'])
        if card.get('no_dev_test_feedback') is not True:raise GateRejected('TRAIN_BOUNDARY_UNCONFIRMED:'+card['feedback_id'])
        # Read the exact review and response artifacts, never invent model evidence.
        for b in [card['source_review_binding']]+card.get('response_bindings',[]):
            artifact=verified(b);artifact.read_bytes();actually_read.append(bind(artifact))
        selected.append(card)
    if len(selected)!=5:raise GateRejected('EXPECTED_FIVE_SOURCE_VERIFIED_GENERATION_CARDS:'+str(len(selected)))
    now=datetime.now(timezone.utc).isoformat()
    packet={'schema_version':'exploratory_train_feedback_input_packet_v1','created_at_utc':now,
        'status':'PREPARED_NO_LLM_CONSUMPTION_YET','scope':'exploratory_train_feedback',
        'source_handoff_state':bind(state_path),'source_feedback_snapshot':bind(snapshot_path),
        'source_validation':bind(prior_validation_path),'feedback_version':snapshot['version'],
        'feedback_ids':[c['feedback_id'] for c in selected],'feedback_cards':selected,
        'excluded_feedback':excluded,'admitted_evidence_candidate_ids':sorted({x for c in selected for x in c['evidence_candidate_ids']}),
        'primary_evaluation':'direct','new_roots_started':0,'new_questions_generated':0,'new_solver_calls':0,
        'matched_AB':False,'formal_feedback_B':False,'formal_reproduced_claim':False,
        'A_input_leakage_or_technical_isolation_claim':False,'faithful_internal_trace_claim':False,
        'feedback_loop_closed':False,'llm_received_packet':False,
        'next_consumer_requirements':['Verify all packet and request manifest hashes before reading.',
          'Log the actual LLM input and read manifest including these feedback IDs and packet hash.',
          'Map each proposed material/question plan to a selected feedback ID and a falsifiable observable target.',
          'Do not add dialogue for citation-format-only issues; that card is excluded.',
          'Do not claim formal B, matched A/B, reproduced weakness, causal gain, or completed feedback loop.',
          'Candidate generation requires separately frozen roots, source, slots, budget and content target realization review.'],
        'limits':['Program preparation receipt is not an LLM consumption receipt.',
          'Local source-verified observations support exploratory planning only; original reproduced/probe gates still apply.',
          'Negative/invalid diagnostics remain recorded in upstream state; original question content admission and 342 direct results remain unchanged.']}
    output_dir.mkdir(parents=True)
    def write_new(name,obj):
        p=output_dir/name
        with p.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
        return p
    fresh_path=write_new('FRESH_VALIDATION.json',fresh)
    packet_path=write_new('INPUT_PACKET.json',packet)
    unique={b['path']:b for b in actually_read}
    receipt_path=write_new('PROGRAM_READ_RECEIPT.json',{'schema_version':'feedback_program_read_receipt_v1','created_at_utc':now,
        'reader':'prepare_next_generation.py','reader_type':'program_not_LLM','source_state':bind(state_path),
        'source_feedback_snapshot':bind(snapshot_path),'source_validation':bind(prior_validation_path),
        'actually_read_and_hash_verified_artifacts':list(unique.values()),'selected_feedback_ids':packet['feedback_ids'],
        'excluded_feedback_ids':[x['feedback_id'] for x in excluded],'input_packet':bind(packet_path),
        'llm_received_packet':False,'generation_started':False,'feedback_loop_closed':False})
    request_path=write_new('REQUEST_MANIFEST.json',{'schema_version':'exploratory_generation_preparation_manifest_v1','created_at_utc':now,
        'status':'INPUTS_PREPARED_AWAITING_ACTUAL_AGENT_READ','scope':'exploratory_train_feedback',
        'source_handoff_state':bind(state_path),'source_feedback_snapshot':bind(snapshot_path),
        'source_validation':bind(prior_validation_path),'fresh_validation':bind(fresh_path),
        'input_packet':bind(packet_path),'program_read_receipt':bind(receipt_path),
        'prepare_script':bind(Path(__file__)),'validator_script':bind(validator_path),
        'feedback_ids':packet['feedback_ids'],'new_roots_started':0,'new_questions_generated':0,'new_solver_calls':0,
        'next_action':'A separate agent may read this fixed packet and record a concrete exploratory plan; no generation is authorized by this preparation alone.',
        'matched_AB':False,'formal_feedback_B':False,'feedback_loop_closed':False,'llm_received_packet':False})
    # Read back what was written; leave no unverifiable prepared outputs.
    for b in read(request_path).values():
        if isinstance(b,dict) and 'path' in b and 'sha256' in b:verified(b)
    return {'status':'PREPARED','scope':'exploratory_train_feedback','output_dir':str(output_dir),'feedback_ids':packet['feedback_ids'],
            'request_manifest':bind(request_path),'packet':bind(packet_path),'llm_consumed':False,'feedback_loop_closed':False,'new_solver_calls':0}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--scope',choices=('exploratory','formal'),required=True)
    args=ap.parse_args()
    try:result=prepare(args.state,args.output_dir,args.scope)
    except Exception as exc:print(json.dumps({'status':'REJECTED','reason':str(exc),'exception_type':type(exc).__name__}));return 2
    print(json.dumps(result,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
