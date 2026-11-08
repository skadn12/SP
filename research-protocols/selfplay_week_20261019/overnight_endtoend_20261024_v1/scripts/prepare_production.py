#!/usr/bin/env python3
"""Immutable OET1 Train source/event preparation. This never authors model data."""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
PRIOR = BASE.parent / 'postweek_feedback_control_4root_v1'
TRAIN = REPO / 'offline-deliveries/h4_codex_train_pool_v125'
POOL = REPO / 'offline-deliveries/h4_external_event_source_refresh_v2/multiwoz_train_sample.json'
PROVENANCE = POOL.with_name('multiwoz_train_source.json')
LOCK = REPO / 'offline-deliveries/h4_r02_source_unique_v2/runs/.unique_event_live.lock'
SPLITS = REPO / 'restart-n5-runs/h4_round0_b0_k3_difficulty_v1_live_1/root_split_manifest.jsonl'
ROOTS = [f'cc-train-episode-{i}' for i in [24719,25652,29753,31457,38601,43599,45849,50175]]
SCENES = [f'OET1:{r}:S{i:02}' for r in 'AB' for i in range(1,4)]
CHECKS = ['time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity']
POOL_SHA = '895a8109bf01fa5ecf15ccdbd2dfe1628bd923f6b61dcd2e26b10ee5076a1596'
META = {'SOURCE_RESERVATIONS.json','CONFIG.json','RUN_CONFIG.json','SOURCE_CONSUMPTION_FREEZE.json','SOURCE_CONSUMPTION_SCAN.json','UNIQUE_EVENT_ALLOCATION.json','RESERVATION.json','SOURCE_BINDINGS.json','FINAL_SOURCE_BINDING.json','AUTHOR_CONSUMPTION.json'}
EXCLUDES = {'.git','__pycache__','node_modules','env','venv','.venv','.venv-local4b'}
SID_RE = re.compile(r'multiwoz22:train:([A-Za-z0-9]+\.json)')
CLOCK_RE = re.compile(r'(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?!\d)')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def bind(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

def verify(b):
    assert bind(b['path']) == b, ('BINDING_CHANGED', b['path'])
    return read(b['path'])

def write_bytes(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_bytes() == data, ('IMMUTABLE_CONTENT_CONFLICT',str(p))
    else:
        with p.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())

def write(p, obj):
    write_bytes(p, (json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))

def complete_records():
    raw = POOL.read_bytes(); assert hashlib.sha256(raw).hexdigest() == POOL_SHA
    text = raw.decode('utf-8'); decoder = json.JSONDecoder(); pos = text.index('[')+1; out = {}
    while True:
        while text[pos] in ' \t\r\n,': pos += 1
        if text[pos] == ']': break
        start = pos; row,pos = decoder.raw_decode(text,pos)
        piece = text[start:pos].encode('utf-8')
        byte_start = len(text[:start].encode('utf-8')); byte_end = byte_start+len(piece)
        assert raw[byte_start:byte_end] == piece and json.loads(piece) == row
        assert row['dialogue_id'] not in out
        assert row['turns'] and all(t.get('utterance','').strip() and t.get('speaker') in ['USER','SYSTEM'] for t in row['turns'])
        out[row['dialogue_id']] = (row,piece,byte_start,byte_end)
    assert len(out) == 512
    provenance = read(PROVENANCE)
    assert provenance['split'] == 'train' and provenance['sha256'] == POOL_SHA
    return out

def clocks(record):
    evidence = []
    for t in record['turns']:
        for match in CLOCK_RE.finditer(t['utterance']):
            evidence.append({'turn_id':t['turn_id'],'speaker':t['speaker'],'utterance':t['utterance'],
                'raw_clock':match.group(0),'normalized_clock':f'{int(match.group(1)):02}:{match.group(2)}',
                'normalization':'zero-pad hour only; no inferred date/AMPM',
                'clock_role':'REQUEST_OR_SERVICE_STATEMENT_AS_WRITTEN_NOT_ACTUAL_COMPLETION'})
    return sorted({x['normalized_clock'] for x in evidence}), evidence

def legacy_usage():
    helper = REPO / 'offline-deliveries/h4_r02_source_unique_v2/unique_events.py'
    spec = importlib.util.spec_from_file_location('oet1_legacy_unique_events',helper)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    runs = mod.prior_runs(REPO/'offline-deliveries/h4_r02_dualroute_feedback_v1/runs',REPO/'offline-deliveries/h4_r02_source_unique_v2/runs')
    result = mod.collect_prior_usage(runs)
    evidence = [bind(x['path']) for x in result['prior_evidence']]
    return result, evidence, bind(helper)

def scan_history(pool_ids):
    blocked = set(); evidence = []; parse_errors = []; parsed = 0
    for current,dirs,files in os.walk(REPO):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDES and not x.startswith('.venv') and not (Path(current)/x).is_symlink())
        here = Path(current)
        if here == BASE or BASE in here.parents:
            dirs[:] = []; continue
        for name in sorted(files):
            if name not in META: continue
            p = here/name
            try:
                text = p.read_text(encoding='utf-8'); json.loads(text)
            except (ValueError,OSError,UnicodeError) as e:
                parse_errors.append({'path':str(p),'error':repr(e)}); continue
            parsed += 1
            ids = set(SID_RE.findall(text)) & pool_ids
            # Reservation/allocations sometimes store native identity without the prefix.
            if name in {'SOURCE_RESERVATIONS.json','UNIQUE_EVENT_ALLOCATION.json','RESERVATION.json'}:
                ids |= set(re.findall(r'(?<![A-Za-z0-9])((?:P?MUL|S?SNG|WOZ)[0-9]+\.json)',text)) & pool_ids
            blocked.update(ids)
            evidence.append({**bind(p),'multiwoz_train_ids':sorted(ids)})
    selection = REPO/'offline-deliveries/h4_codex_controlled_batch_v105/evidence/SOURCE_SELECTION.json'
    explicit = read(selection)['consumed_source_ids']
    ids = {sid.split(':',2)[2] for sid in explicit if sid.startswith('multiwoz22:train:')} & pool_ids
    blocked.update(ids)
    evidence.append({**bind(selection),'multiwoz_train_ids':sorted(ids),'scope':'explicit consumed_source_ids only; unselected candidates excluded'})
    legacy, legacy_evidence, helper_binding = legacy_usage()
    legacy_ids = {x['item_id'] for x in legacy['accepted_uses'] if x['dataset']=='MultiWOZ_2.2' and x['split']=='train'} & pool_ids
    blocked.update(legacy_ids)
    evidence.extend({**x,'scope':'validated legacy v1/v2 report/card/source/result'} for x in legacy_evidence)
    assert not parse_errors, ('HISTORICAL_METADATA_PARSE_ERRORS',parse_errors)
    return blocked,evidence,{'scan_root':str(REPO),'metadata_filename_allowlist':sorted(META),
        'directory_exclusions':sorted(EXCLUDES)+['names starting .venv','symlink directories',str(BASE)],
        'metadata_files_parsed':parsed,'metadata_parse_errors':parse_errors,
        'explicit_consumed_source_ids_binding':bind(selection),'legacy_v1_v2_helper_binding':helper_binding,
        'legacy_v1_v2_validated_runs':legacy['prior_runs'],'legacy_v1_v2_accepted_uses_count':len(legacy['accepted_uses']),
        'legacy_v1_v2_multiwoz_train_ids':sorted(legacy_ids),
        'limitations':['Repo-external ledgers and unlogged bypass consumption cannot be certified.','Shared source lock serializes cooperating allocators only.']}

def root_rows():
    manifest = read(TRAIN/'data/ROOT_MANIFEST.json'); by_id = {r['root_id']:r for r in manifest}
    assert len(manifest)==68 and len(by_id)==68 and all(r['split']=='train' for r in manifest)
    authority = {r['root_id']:r for r in [json.loads(x) for x in SPLITS.read_text().splitlines() if x.strip()]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/Qwen3-4B',local_files_only=True)
    rows = []
    for root in ROOTS:
        m=by_id[root]; p=TRAIN/m['packaged_source']; src=read(p); b=bind(p)
        assert src['root_id']==root and m['split']=='train' and authority[root]['split']=='train'
        assert b['sha256']==m['source_binding']['sha256'] and b['bytes']==m['source_binding']['size_bytes']
        original=Path(m['source_binding']['path']); assert original.read_bytes()==p.read_bytes()
        turns=src['turns']; refs=[t['turn_ref'] for t in turns]
        assert turns and len(refs)==len(set(refs)) and all(t['speaker'].strip() and t['text'].strip() for t in turns)
        assert len(turns)==src['baseline_turn_count']+src['cumulative_new_turn_count']
        assert m['historical_parent_seen_by_M_clean'] is True
        text='\n'.join(f"[{t['turn_ref']}] {t['speaker']}: {t['text']}" for t in turns)
        count=len(tokenizer(text,add_special_tokens=False)['input_ids']); assert count<=8192
        local=BASE/'inputs/initial_sources'/f'{root}.json'; write_bytes(local,p.read_bytes())
        rows.append({'root_id':root,'numeric_root_id':int(root.rsplit('-',1)[1]),'source_binding':b,
            'frozen_initial_source_binding':bind(local),'original_source_binding':bind(original),
            'source_final_scene_ref':src['scene_ref'],'source_complete_untruncated':True,
            'source_structurally_loaded_in_full':True,'source_semantic_review_performed':False,
            'speaker_names':list(dict.fromkeys(t['speaker'] for t in turns)),'turn_count':len(turns),
            'historical_parent_seen_by_M_clean':True,'unseen_root_claim_permitted':False,
            'train_split_evidence':{'authority_binding':bind(SPLITS),'authority_record':authority[root],
                'packaged_root_manifest_record':{k:v for k,v in m.items() if k!='candidate_ids'}},
            'length':{'tokenizer_path':'/root/autodl-tmp/Qwen3-4B','full_dialogue_tokens_exact':count,
                'definition':'All turns rendered [turn_ref] speaker: text joined by newline; add_special_tokens=False.',
                'future_complete_prompt_retokenization_required':True},
            'selection_checks':{'train_whitelist':True,'manifest_and_original_hash_match':True,
                'complete_nonempty_unique_turns':True,'baseline_plus_extension_count_exact':True,
                'whole_source_tokens_lte8192':True}})
    return rows

def clean_packet(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            assert 'gold' not in k.lower() and k.lower() not in {'answer','answers','final_answer','answer_reason','expected_error','model_reported_explanations','semantic_review'},k
            clean_packet(v)
    elif isinstance(obj,list):
        for v in obj: clean_packet(v)

def main():
    receipt=BASE/'inputs/PRODUCTION_PREPARATION_RECEIPT.json'
    if receipt.exists():
        result=read(receipt)
        for b in result['artifacts']: verify(b)
        print(json.dumps({'status':'VERIFIED_ALREADY_PREPARED','receipt':bind(receipt),'counts':result['counts']})); return
    assert not (BASE/'FINALIZE_REQUEST').exists(), 'FINALIZE_REQUEST_STOP_NEW_PREPARATION'
    started=BASE/'inputs/PREPARATION_STARTED.json'
    if not started.exists(): write(started,{'schema_version':'oet1_production_preparation_start_v1','created_at_utc':now(),'purpose':'Train-only immutable input preparation; no model generation'})
    stamp=read(started)['created_at_utc']; rows=root_rows(); records=complete_records()
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX); acquired=now()
        blocked,evidence,coverage=scan_history(set(records))
        trace=[]; eligible=[]
        for native_id in sorted(records):
            row=records[native_id][0]; values,ev=clocks(row)
            placeholder=bool(re.search(r'\[(?:value_)?time\]', ' '.join(t['utterance'] for t in row['turns']),re.I))
            okay=native_id not in blocked and len(values)>=2 and not placeholder
            trace.append({'dialogue_id':native_id,'distinct_explicit_clock_count':len(values),
                'time_placeholder_found':placeholder,'prior_use_hit':native_id in blocked,'eligible':okay})
            if okay: eligible.append(native_id)
        if (BASE/'ROOT_SELECTION_PROPOSAL.json').exists():
            prior=read(BASE/'ROOT_SELECTION_PROPOSAL.json'); selections=[r['proposed_native_event_source_ids'] for r in prior['roots']]
            selected=[s.split(':',2)[2] for group in selections for s in group]
            assert all(n not in blocked for n in selected) and len(selected)==24
        else:
            assert len(eligible)>=24, ('INSUFFICIENT_ELIGIBLE_NATIVE_EVENTS',len(eligible))
            selected=eligible[:24]
        reservations=[]; groups=[]; events=[]
        for index,root in enumerate(rows):
            native_ids=selected[index*3:index*3+3]; union=set()
            root['proposed_native_event_source_ids']=['multiwoz22:train:'+n for n in native_ids]
            for slot,native_id in enumerate(native_ids,1):
                record,raw,begin,end=records[native_id]; values,ev=clocks(record); union.update(values)
                p=BASE/'inputs/native_events'/native_id; write_bytes(p,raw); b=bind(p); events.append(b)
                reservations.append({'dataset':'MultiWOZ_2.2','split':'train','item_id':native_id,'pool_record_id':native_id,
                    'source_id':'multiwoz22:train:'+native_id,'source_item_sha256':b['sha256'],
                    'run':str(BASE/'inputs'),'root_id':root['root_id'],'scene_id':f'B:S{slot:02}',
                    'scene_ref':f'OET1:B:S{slot:02}','pair_id':f"{root['root_id']}:B:S{slot:02}",
                    'authorized_consumers':[root['root_id']+':'+a for a in ['GENERIC','TARGETED']],
                    'authorized_use':'One logical reservation shared only by the paired conditions of this root and slot.',
                    'other_reuse_prohibited':True,'status':'PERMANENTLY_RESERVED_BEFORE_AUTHORING',
                    'reserved_before_authoring':True,'release_on_failure':False,'release_on_nonexecution':False,
                    'pool_binding':bind(POOL),'pool_record_byte_span':{'start_inclusive':begin,'end_exclusive':end},
                    'raw_pool_record_sha256':b['sha256'],'native_record_binding':b,'binding':b,
                    'native_record_canonical_json_sha256':hashlib.sha256(json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                    'native_turn_count':len(record['turns']),'distinct_explicit_clocks':values,'clock_evidence':ev})
            assert len(union)>=4, ('CLOCK_UNION_SHORTFALL',root['root_id'],union)
            groups.append({'root_id':root['root_id'],'source_ids':root['proposed_native_event_source_ids'],
                'distinct_explicit_clock_union':sorted(union),'clock_union_count':len(union)})
        proposal={'schema_version':'oet1_fixed_train_root_selection_v1','created_at_utc':stamp,'status':'FROZEN_INPUT_SELECTION',
            'root_selection_rule':'Fixed user-authorized eight Train roots; no solver scores read or used.',
            'seen_root_caution':'Different from the recent four-root pilot; all historical_parent_seen_by_M_clean=true; no unseen-root claim.',
            'source_manifest_binding':bind(TRAIN/'data/ROOT_MANIFEST.json'),'initial_source_roots':ROOTS,
            'native_event_selection_rule':'Under permanent global lock: sorted native ID, exclude all historical reservations including failed/nonexecuted, at least two explicit HH:MM values per record, first 24 eligible; group three per fixed root with union at least four.',
            'roots':rows,'root_groups':groups,'no_solver_scores_used':True,'generation_started':False}
        write(BASE/'ROOT_SELECTION_PROPOSAL.json',proposal)
        reservation={'schema_version':'h4.r02.source_reservations.v4','reservation_extension_schema':'oet1_paired_native_reservations_v1',
            'run':str(BASE/'inputs'),'created_at_utc':stamp,'status':'PERMANENT',
            'policy':'Reserve before authoring; 24 logical reservations; two authorized paired consumers each. No release on failure, cancellation, rejection or nonexecution.',
            'proposal_binding':bind(BASE/'ROOT_SELECTION_PROPOSAL.json'),'pool_binding':bind(POOL),
            'source_ids':[r['source_id'] for r in reservations],'reservations':reservations}
        write(BASE/'inputs/SOURCE_RESERVATIONS.json',reservation)
        # The permanent metadata is visible to cooperating allocators before lock release.
        assert len(set(selected))==24 and all(n not in blocked for n in selected)
        audit={'schema_version':'oet1_permanent_paired_reservation_audit_v1','created_at_utc':stamp,
            'status':'PASS_PERMANENT_PAIRED_RESERVATIONS','reservation_binding':bind(BASE/'inputs/SOURCE_RESERVATIONS.json'),
            'proposal_binding':bind(BASE/'ROOT_SELECTION_PROPOSAL.json'),'pool_binding':bind(POOL),
            'pool_provenance_binding':bind(PROVENANCE),'logical_reservations':24,'authorized_consumers':48,
            'root_groups':groups,'all_event_records_exact_raw_pool_bytes':True,'all_record_hashes_verified':True,
            'all_selected_prior_unused_within_verified_scan_scope':True,'release_on_failure':False,
            'source_lock':{'path':str(LOCK),'mechanism':'fcntl.LOCK_EX','acquired_at_utc':acquired,'held_for_inventory_and_reservation':True},
            'scan_coverage':{**coverage,'blocked_pool_count_before_reservation':len(blocked),
                'blocked_pool_count_after_this_reservation':len(blocked)+24},
            'inventory_evidence':evidence,'selection_trace':trace,'event_files':events,
            'no_dialogue_or_question_generation':True,'no_solver_data_read':True,'no_gpu_or_model_started':True,'historical_files_modified':False}
        existing_audit=BASE/'inputs/RESERVATION_AUDIT.json'
        if existing_audit.exists():
            old=read(existing_audit); assert old['reservation_binding']==audit['reservation_binding']
        else: write(existing_audit,audit)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    transfer=read(PRIOR/'TRANSFER_PLAN.json'); cards=read(PRIOR/'WEAKNESS_CARDS.json')
    for card in cards['cards']:
        for b in card['origin'].values():
            if isinstance(b,dict) and 'path' in b: verify(b)
    write_bytes(BASE/'WEAKNESS_CARDS.json',(PRIOR/'WEAKNESS_CARDS.json').read_bytes())
    generic=transfer['generic_abstract_goals']; targeted=transfer['targeted_abstract_goals']
    plan={'schema_version':'oet1_exploratory_cross_root_transfer_v1','created_at_utc':stamp,
        'origin_cards':bind(BASE/'WEAKNESS_CARDS.json'),'unchanged_origin_cards_binding':bind(PRIOR/'WEAKNESS_CARDS.json'),
        'origin_transfer_plan':bind(PRIOR/'TRANSFER_PLAN.json'),'target_roots':ROOTS,'mapping':transfer['mapping'],
        'targeted_abstract_goals':targeted,'generic_abstract_goals':generic,'abstract_goals_changed':False,
        'status':'EXPLORATORY_CROSS_ROOT_TRANSFER_NOT_PROVEN_FORMAL_WEAKNESS',
        'same_root_feedback_claim':False,'source_authors_receive_raw_reasons':False,
        'train_only_origin_verified_by_fbc1_evidence':True,'source_authors_receive_origin_question_or_private_answers':False}
    write(BASE/'TRANSFER_PLAN.json',plan)
    inputs=[]
    requirements=[
        'Maintain enough uniquely grounded factual relationships for actor binding, updated state, conclusion support, joint constraints and record audit tasks.',
        'Preserve native explicit clock values and their actual roles in reported service interactions. At least four distinct clock readings across the B material; only report times already explicit in native utterances.',
        'Native events are distinct enquiries; do not imply one cross-day journey or completion. A numeric clock comparison may compare reported clock readings with explicit scope, not real chronology.',
        'Keep enquiries with identifiable disclosure steps, without inserting numbered steps or quiz language. Do not add unsupported genders or new central people.',
        'Do not invent service replies, confirmation, payments, departures or external facts. Fictional linking dialogue may frame an exact service record as a reported enquiry, retaining requests versus responses.'
    ]
    for row in rows:
        source=verify(row['frozen_initial_source_binding']); native=[]
        for sid in row['proposed_native_event_source_ids']:
            p=BASE/'inputs/native_events'/sid.split(':',2)[2]
            native.append({'source_id':sid,'native_event_binding':bind(p),'native_record':read(p)})
        for arm in ['GENERIC','TARGETED']:
            packet={'schema_version':'oet1_source_author_input_v1','root_id':row['root_id'],'condition':arm,
                'namespace':'OET1','source_binding':row['frozen_initial_source_binding'],'full_initial_source':source,
                'material_goals':generic if arm=='GENERIC' else targeted,'scene_sequence':SCENES,
                'route_roles':{'A':'same-root continuation','B':'reserved native event continuation; cumulative after A'},
                'turns_per_scene':14,'attempts_per_scene':1,'speakers':row['speaker_names'],'native_events':native,
                'context_policy':'FULL_SOURCE_NO_TRUNCATION','review_dimensions':CHECKS,'append_only':True,
                'common_material_requirements':requirements,
                'feedback_scope':'EXPLORATORY_CROSS_ROOT_ABSTRACT_GOAL_TRANSFER; NOT_PROVEN_FORMAL_WEAKNESS' if arm=='TARGETED' else 'GENERIC_FROZEN_ABSTRACT_GOALS'}
            clean_packet(packet)
            p=BASE/'sources'/row['root_id']/arm/'SOURCE_AUTHOR_INPUT.json'; write(p,packet); inputs.append(bind(p))
    artifacts=[bind(BASE/'ROOT_SELECTION_PROPOSAL.json'),bind(BASE/'inputs/SOURCE_RESERVATIONS.json'),
        bind(BASE/'inputs/RESERVATION_AUDIT.json'),bind(BASE/'TRANSFER_PLAN.json'),bind(BASE/'WEAKNESS_CARDS.json'),
        bind(Path(__file__)),bind(BASE/'scripts/control.py')]+events+inputs+[r['frozen_initial_source_binding'] for r in rows]
    result={'schema_version':'oet1_production_preparation_receipt_v1','created_at_utc':now(),'status':'PREPARED_NOT_GENERATED',
        'counts':{'roots':8,'initial_source_copies':8,'logical_native_event_reservations':24,
            'native_event_authorized_consumers':48,'author_inputs':16,'planned_scenes':96,'planned_question_slots':112},
        'artifacts':artifacts,'old_files_modified':False,'model_data_authored':False,'generation_started':False,
        'limitations':['Preparation proves structural source and immutable material binding only; independent generated-source/content review remains required.',
            'All eight initial roots were historically seen by M-clean; cross-root feedback transfer is exploratory.',
            'Initial dialogue token counts are not final complete training or evaluation prompt lengths.']}
    write(receipt,result)
    print(json.dumps({'status':result['status'],'receipt':bind(receipt),'counts':result['counts']},ensure_ascii=False))

if __name__=='__main__': main()
