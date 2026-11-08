"""Read-only historical checks and immutable delivery assembly for configured batches."""
import argparse
import collections
import importlib.util
from datetime import datetime
from pathlib import Path
from codex_controlled_batch import read,save,digest,now,config,root_folder,copy_once

def snapshot(batch, protected):
    rows=[]
    for directory in protected:
        directory=Path(directory).resolve();manifest=directory/'COMPLETE_MANIFEST.json'
        value=read(manifest);issues=[]
        for item in value['files']:
            p=Path(item['path']);p=p if p.is_absolute() else directory/p
            if not p.is_file() or digest(p)!=item['sha256']:issues.append(str(p))
        if issues:raise ValueError('historical_manifest_mismatch:'+str(issues[:5]))
        files=[{'path':str(p),'sha256':digest(p)} for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
        rows.append({'directory':str(directory),'manifest_sha256':digest(manifest),'manifest_listed_files_verified':len(value['files']),'files':files})
    save(Path(batch)/'evidence/HISTORICAL_BEFORE.json',{'captured_at_utc':now(),'protected':rows})
    print('historical snapshot',sum(len(r['files']) for r in rows))

def history_unchanged(batch):
    old=read(Path(batch)/'evidence/HISTORICAL_BEFORE.json');result=[]
    for row in old['protected']:
        prior={r['path']:r['sha256'] for r in row['files']}
        current={str(p):digest(p) for p in Path(row['directory']).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        changed=[p for p,h in prior.items() if current.get(p)!=h]
        extra=sorted(set(current)-set(prior))
        result.append({'directory':row['directory'],'before_count':len(prior),'after_count':len(current),'changed_or_missing':changed,'extra':extra,'unchanged':not changed and not extra})
    return result

def duplicate_audit(batch):
    batch=Path(batch).resolve();deliveries=batch.parent
    # Reuse the frozen public-projection parser without executing its batch main.
    parser=deliveries/'h4_batch_codex_benchmark_v104/evidence/audit_public_duplicates_and_lengths.py'
    spec=importlib.util.spec_from_file_location('frozen_public_parser',parser)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.DELIVERIES=deliveries;module.BATCH=batch
    historical=[];scanned=[];failures=[]
    for path in sorted(deliveries.rglob('*')):
        if not path.is_file() or not module.public_candidate(path):continue
        try:
            records,excluded=module.extract_questions(module.load_objects(path),path)
            historical.extend(records);scanned.append({'path':str(path),'sha256':digest(path),'public_occurrences':len(records),'excluded':dict(excluded)})
        except (OSError,ValueError,UnicodeError,RecursionError) as exc:
            failures.append({'path':str(path),'error':str(exc)[:300]})
    current=[]
    for rid in config(batch)['roots']:
        p=root_folder(batch,rid)/'PUBLIC_QUESTIONS.json'
        if p.exists():
            records,excluded=module.extract_questions(module.load_objects(p),p)
            if excluded:raise ValueError('current_public_extraction_failed:'+str(excluded))
            current.extend(records)
    index=collections.defaultdict(list);internal=collections.defaultdict(list)
    for r in historical:index[r['key']].append(r)
    for r in current:internal[r['key']].append(r)
    desc=module.descriptor
    matches=[{'current':desc(r),'historical_matches':[desc(x) for x in index[r['key']]]} for r in current if r['key'] in index]
    repeats=[{'members':[desc(x) for x in rows]} for rows in internal.values() if len(rows)>1]
    output={'created_at_utc':now(),'parser_path':str(parser),'parser_sha256':digest(parser),'normalization':'canonical root, question type, sorted set of six casefolded whitespace-removed propositions; punctuation retained; stem ignored',
        'historical_public_occurrences':len(historical),'historical_unique_keys':len(index),'current_count':len(current),'current_vs_historical_exact_duplicates':matches,'current_internal_duplicate_groups':repeats,
        'scanned_files':scanned,'parse_failures':failures,'semantic_deduplication_performed':False,
        'limitations':['Only discovered public-shaped six-item/option records are covered.','Exact comparison does not establish semantic novelty.','Family variants and overlapping ordering slots remain correlated.']}
    save(batch/'evidence/HISTORICAL_DUPLICATE_AUDIT.json',output)
    return output

def assemble(batch):
    batch=Path(batch).resolve();cfg=config(batch)
    duplicate=read(batch/'evidence/HISTORICAL_DUPLICATE_AUDIT.json')
    duplicate_ids={x['current']['candidate_id'] for x in duplicate['current_vs_historical_exact_duplicates']}
    for group in duplicate['current_internal_duplicate_groups']:
        duplicate_ids.update(m['candidate_id'] for m in group['members'])
    new_public=[];new_gold=[];controls=[];control_gold=[];all_public=[];excluded_public=[];excluded_gold=[];all_outcomes=[];roots=[];dialogues=[];suffixes=[]
    for row in read(batch/'SOURCE_RESERVATIONS.json')['allocation']:
        rid=row['root_id'];folder=root_folder(batch,rid);entry={'root_id':rid,'source_id':row['source_id'],'source_status':'NO_SOURCE','new_multiselect':0,'ordering':0,'controls':0,'excluded':0,'unfilled_slots':[]}
        if (folder/'SOURCE_REVIEW.json').exists():entry['source_status']=read(folder/'SOURCE_REVIEW.json')['decision']
        if (folder/'CUMULATIVE_SOURCE.json').exists():
            source=read(folder/'CUMULATIVE_SOURCE.json');entry['new_turn_count']=source['new_turn_count']
            dialogues.append((rid,source['turns']));suffixes.append((rid,source['turns'][source['baseline_turn_count']:]))
        if entry['source_status']=='READY':
            for name in ('COMPOSITION_RECEIPT.json','FINAL_CROSSCHECK.json','PRIVATE_GOLD.json','PUBLIC_SOURCE_JUDGMENTS.json','AUTHOR_RECEIPT.json','QA_AUTHOR_RECEIPT.json'):
                if not (folder/name).exists():raise ValueError(rid+':incomplete:'+name)
            entry['unfilled_slots']=read(folder/'COMPOSITION_RECEIPT.json')['unfilled_slots']
            pub=read(folder/'PUBLIC_QUESTIONS.json');priv=read(folder/'PRIVATE_GOLD.json');cross=read(folder/'FINAL_CROSSCHECK.json')
            if priv['public_sha256']!=digest(folder/'PUBLIC_QUESTIONS.json') or priv['source_judgments_sha256']!=digest(folder/'PUBLIC_SOURCE_JUDGMENTS.json') or cross['private_sha256']!=digest(folder/'PRIVATE_GOLD.json'):
                raise ValueError(rid+':stale_final_hash')
            source_hash=digest(folder/'CUMULATIVE_SOURCE.json')
            public_ids=[x['candidate_id'] for x in pub['questions']]
            public_id_set=set(public_ids)
            if len(public_ids)!=len(public_id_set):
                raise ValueError(rid+':duplicate_public_candidate_ids')
            if any(x.get('source_sha256')!=source_hash for x in pub['questions']):
                raise ValueError(rid+':public_source_hash_mismatch')
            if priv['source_sha256']!=source_hash or cross.get('public_sha256')!=digest(folder/'PUBLIC_QUESTIONS.json'):
                raise ValueError(rid+':source_or_public_hash_mismatch')
            if priv['public_sha256']!=cross.get('public_sha256') or priv['source_judgments_sha256']!=cross.get('source_judgments_sha256'):
                raise ValueError(rid+':cross_review_hash_mismatch')
            if priv['source_judgments_sha256']!=digest(folder/'PUBLIC_SOURCE_JUDGMENTS.json'):
                raise ValueError(rid+':public_review_hash_mismatch')
            cross_results=cross.get('results',[])
            cross_ids=[x.get('candidate_id') for x in cross_results]
            if len(cross_ids)!=len(set(cross_ids)) or set(cross_ids)!=public_id_set:
                raise ValueError(rid+':final_crosscheck_candidate_set_mismatch')
            by_id={x['candidate_id']:x for x in pub['questions']};golds={x['candidate_id']:x for x in priv['questions']}
            if set(golds)!=public_id_set:
                raise ValueError(rid+':private_gold_candidate_set_mismatch')
            all_public.extend(pub['questions'])
            for result in cross_results:
                out=dict(result);cid=out['candidate_id']
                if cid in duplicate_ids:out={**out,'status':'EXCLUDED','issues':out['issues']+['public_exact_duplicate']}
                all_outcomes.append(out)
                if out['status']!='CONTENT_PASS':
                    entry['excluded']+=1;excluded_public.append(by_id[cid]);excluded_gold.append(golds[cid]);continue
                if out['role']=='control6':
                    controls.append(by_id[cid]);control_gold.append(golds[cid]);entry['controls']+=1
                else:
                    new_public.append(by_id[cid]);new_gold.append(golds[cid])
                    entry['new_multiselect' if out['question_type']=='multi_select' else 'ordering']+=1
        roots.append(entry)
    for name,questions in [('NEW_PUBLIC_QUESTIONS.json',new_public),('NEW_PRIVATE_GOLD.json',new_gold),('CONTROL_PUBLIC_QUESTIONS.json',controls),('CONTROL_PRIVATE_GOLD.json',control_gold),('ALL_PUBLIC_CANDIDATES.json',all_public),('EXCLUDED_PUBLIC_QUESTIONS.json',excluded_public),('EXCLUDED_PRIVATE_GOLD.json',excluded_gold)]:
        save(batch/name,{'questions':questions})
    history=history_unchanged(batch)
    if not all(r['unchanged'] for r in history):raise ValueError('protected_historical_batch_changed')
    save(batch/'evidence/HISTORICAL_AFTER.json',{'verified_at_utc':now(),'results':history})
    def render(groups):return '\n\n'.join(rid+'\n'+'\n'.join(t['turn_ref']+' '+t['speaker']+': '+t['text'] for t in turns) for rid,turns in groups)+'\n'
    for name,groups in [('CUMULATIVE_DIALOGUES.txt',dialogues),('NEW_DIALOGUES.txt',suffixes)]:
        with (batch/name).open('x',encoding='utf-8') as f:f.write(render(groups))
    measurements={'tokenizer':None,'suffix_characters':sum(len(t['text']) for _,ts in suffixes for t in ts),'suffix_turns':sum(len(ts) for _,ts in suffixes),'billing_tokens':None,'cost':None,'limitation':'Artifact text tokens are not generation usage.'}
    try:
        import tiktoken
        import tiktoken.load
        import os
        os.environ['TIKTOKEN_CACHE_DIR']='/tmp/data-gym-cache'
        def reject_download(*a,**k):raise RuntimeError('offline measurement only')
        tiktoken.load.read_file=reject_download
        enc=tiktoken.get_encoding('o200k_base')
        measurements.update(tokenizer='o200k_base',suffix_text_tokens=sum(len(enc.encode(t['text'])) for _,ts in suffixes for t in ts))
    except Exception as exc:measurements['tokenizer_failure']=str(exc)[:200]
    timing=batch/'BATCH_STARTED.json'
    started=read(timing)['started_at_utc'] if timing.exists() else cfg['created_at_utc'];finished=now()
    gold_counts=collections.Counter(str(len(g['gold'])) for g in new_gold if len(g['gold'])<6)
    result={'batch_id':cfg['batch_id'],'status':'CONTENT_REVIEWED_NOT_SOLVER_OR_TRAINING_VALIDATED','roots':roots,'new_questions':len(new_public),'new_multiselect':sum(r['new_multiselect'] for r in roots),'new_ordering':sum(r['ordering'] for r in roots),'gold2_to5_distribution':dict(gold_counts),'gold6_controls':len(controls),'public_candidates':len(all_outcomes),'excluded_final_candidates':sum(r['excluded'] for r in roots),'unfilled_slots':sum(len(r['unfilled_slots']) for r in roots),'measurements':measurements,'started_at_utc':started,'finished_at_utc':finished,'wall_seconds_including_preparation':(datetime.fromisoformat(finished)-datetime.fromisoformat(started)).total_seconds(),'source_selection':'purposive historically exposed roots; not random holdout','backend':cfg['backend'],'exact_model_id':None,'solver_run':False,'training_run':False,'limitations':['Same inherited Codex model and shared session background across author and reviewer roles.','Semantic judgments are model-authored; scripts check record consistency, exact citations, hashes and enumerated anchors.','Semantic novelty and task difficulty have not been established.']}
    save(batch/'ALL_OUTCOMES_INDEX.json',{'results':all_outcomes,'roots':roots})
    save(batch/'UNFILLED_SLOTS.json',{'roots':[{'root_id':r['root_id'],'source_status':r['source_status'],'unfilled_slots':r['unfilled_slots']} for r in roots]})
    save(batch/'FINAL_BATCH_ACCOUNTING.json',result)
    print(result)

def seal(batch):
    batch=Path(batch).resolve();read(batch/'FINAL_BATCH_ACCOUNTING.json')
    for name in ('codex_controlled_batch.py','codex_batch_questions.py','codex_batch_delivery.py','test_codex_controlled_batch.py','test_codex_batch_questions.py'):
        copy_once(Path(__file__).with_name(name),batch/'evidence/code'/name)
    files=[{'path':str(p.relative_to(batch)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(batch.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    save(batch/'COMPLETE_MANIFEST.json',{'sealed_at_utc':now(),'status':'SEALED_CONTENT_ONLY','files':files,'self_hash_excluded':True})
    print('sealed',len(files),'files')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['snapshot','duplicates','assemble','seal']);p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--protected',type=Path,action='append',default=[]);a=p.parse_args()
    if a.command=='snapshot':snapshot(a.batch_dir,a.protected)
    elif a.command=='duplicates':print({k:v for k,v in duplicate_audit(a.batch_dir).items() if k in ('current_count','historical_public_occurrences','historical_unique_keys')})
    elif a.command=='assemble':assemble(a.batch_dir)
    else:seal(a.batch_dir)

if __name__=='__main__':main()
