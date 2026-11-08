"""Offline formal preparation and conditional outline previews; zero API calls."""
import json
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch
WORKTREE=Path('/root/autodl-tmp/gxy-selfplay/worktrees/h4-outline-evidence-id-v1')
sys.path.insert(0,str(WORKTREE))
from scripts.current_method.restart_n5_spell.core import atomic_json, atomic_text
from scripts.current_method.restart_n5_spell.h4_writer_model_comparison_v1.transport import build_payload
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_validate import prepare,run_round,sha,tokenizer_from
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_runtime import capacity_check
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.development_prepare import PreparedBoundary,ROOTS
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.profiles import SOURCE_BOUND_OUTLINE_PROFILE
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.outline_source_policy import validate_outline
from scripts.current_method.restart_n5_spell.simple_append_memorax_v1_memory_event_continuation_probe_v1.live import _sink
OUT=Path(__file__).resolve().parent
RUN=OUT/'prepared_run'
SOURCE=Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_planner_gate_source_bound_v1/prepared_run')
read=lambda p:json.loads(p.read_text())

def main():
    with patch('socket.socket.connect',side_effect=AssertionError('offline_no_network')),patch('socket.socket.connect_ex',side_effect=AssertionError('offline_no_network')):
        manifest=prepare(RUN,execution_profile=SOURCE_BOUND_OUTLINE_PROFILE)
        tokenizer=tokenizer_from(manifest['tokenizer_path']);counter=lambda text:len(tokenizer.encode(text,add_special_tokens=False))
        summaries=[];extra=[Path(__file__).resolve()]
        for root in ROOTS:
            inputs=read(RUN/'frozen'/root/'inputs.json')
            stored=SOURCE/'roots'/root
            planner=read(stored/'planner/output.json');gate=read(stored/'semantic_gate/output.json')
            sourcefiles=[stored/'planner/output.json',stored/'semantic_gate/output.json',stored/'planner/outline_semantic_output.json']
            before={str(p):sha(p) for p in sourcefiles};extra.extend(sourcefiles)
            assert read(RUN/'frozen'/root/'planner_request.json')==read(SOURCE/'frozen'/root/'planner_request.json')
            preview=RUN/'outline_preview_from_old_outputs'/root
            artifacts={};writer=_sink(preview);calls=[]
            def sink(name,value):
                artifacts[name]=deepcopy(value);writer(name,value)
            def boundary(stage,prompt,max_tokens):
                calls.append(stage)
                if stage=='planner':return deepcopy(planner)
                if stage=='gate':
                    baseline=read(SOURCE/'execution/calls'/root/'gate/request.json')['payload']['messages']
                    assert [{'role':'user','content':prompt['prompt']}]==baseline
                    return deepcopy(gate)
                if stage=='outline':raise PreparedBoundary()
                raise AssertionError('unexpected_model_boundary')
            try:run_round(inputs,boundary,sink,counter,execution_profile=SOURCE_BOUND_OUTLINE_PROFILE)
            except PreparedBoundary:pass
            assert calls==['planner','gate','outline']
            prompt=artifacts['planner/outline_semantic_prompt']
            catalog=artifacts['semantic_gate/evidence_catalog']
            profile=deepcopy(manifest['public_profile']);profile['request_parameters']['max_tokens']=8192;profile['output_limit']=8192
            request=build_payload(profile,prompt['prompt'])
            capacity=capacity_check(profile,tokenizer,request['messages'],8192)
            assert capacity['status']=='PASS','preview_exceeds_capacity'
            atomic_json(preview/'outline_request.json',request)
            atomic_json(preview/'capacity.json',capacity)
            legacy=validate_outline(read(stored/'planner/outline_semantic_output.json'),plan=artifacts['planner/compiled_plan'],catalog=catalog,expected_root_id=root,expected_catalog_sha256=catalog['sha256'])
            assert not legacy['valid'] and legacy['errors']==['source_outline_envelope_shape']
            atomic_json(preview/'unchanged_legacy_outline_validation.json',legacy)
            assert before=={str(p):sha(p) for p in sourcefiles}
            summaries.append({'root_id':root,'preview_only':True,'new_outline_model_outputs':0,
              'candidate_count':prompt['candidate_count'],'pair_count':prompt['pair_count'],
              'planner_request_matches_previous':True,'gate_request_matches_previous':True,
              'outline_full_transcript_included':True,'capacity':capacity,
              'unchanged_legacy_outline_status':'REJECTED_NEW_ENVELOPE_NO_MIGRATION',
              'old_outputs_unchanged':True,'semantic_effect':'NOT_TESTED'})
        # Freeze preview sources and products too, without touching prior manifests.
        extra.extend(p for p in (RUN/'outline_preview_from_old_outputs').rglob('*') if p.is_file())
        files={row['path']:row for row in manifest['files']}
        for p in extra:files[str(p)]={'path':str(p),'sha256':sha(p)}
        manifest['files']=list(files.values())
        atomic_json(RUN/'freeze_manifest.json',manifest)
        result={'status':'OFFLINE_READY_NOT_EXECUTED','external_calls':0,'roots':summaries,
          'real_outline_quality':'NOT_TESTED','no_legacy_output_rewrite':True,
          'future_run_maximum_http':manifest['maximum_http_attempts'],
          'future_run_requires_separate_authorization':True}
        atomic_json(RUN/'outline_preflight.json',result)
        atomic_text(RUN/'OFFLINE_STATUS.md','OFFLINE_READY_NOT_EXECUTED\n\nNew Outline model calls: 0. Previews condition on unchanged stored Planner/Gate output, not fresh generation. Prior outputs are not migrated or repaired. New profile also supplies full frozen transcript to Outline; this changes input view, so later results cannot be attributed solely to returning IDs.\n')
        print(json.dumps({'status':result['status'],'input_tokens':[r['capacity']['input_tokens'] for r in summaries],'external_calls':0}))
if __name__=='__main__':main()
