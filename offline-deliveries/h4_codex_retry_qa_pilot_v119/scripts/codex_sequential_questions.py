"""Cross-scene immutable question stages for the separate A3/B3 pilot.

Adapted from codex_batch_questions.py; the old single-scene controller is unchanged.
No model, Solver or training process is invoked.
"""
import argparse
import itertools
import random
import re
from pathlib import Path
from codex_controlled_batch import read, save, digest, canonical_digest, now, root_folder
from codex_sequential_ab import load_config as config, SCENES, inspect_root, scene_folder

def require(condition, message):
    if not condition:
        raise ValueError(message)

def configured_roots(cfg):
    return [row if isinstance(row, dict) else {'root_id': row} for row in cfg['roots']]

def scope_scenes(cfg):
    scenes = cfg.get('scope_scene_refs', cfg['scene_sequence'])
    require(scenes == list(SCENES), 'question_scope_must_cover_all_six_scenes')
    return scenes

def folder_root_id(folder):
    require(folder.name.startswith('root') and folder.name[4:].isdigit(), 'root_folder_name')
    return 'cc-train-episode-' + folder.name[4:]

def source_ready(folder):
    batch = folder.parent.parent
    cfg = config(batch)
    state = inspect_root(batch, cfg, folder_root_id(folder))
    require(state['full_expansion_complete'] is True, 'all_six_source_scenes_must_be_READY')
    final_scene = scene_folder(batch, folder_root_id(folder), SCENES[-1]) / 'CUMULATIVE_SOURCE.json'
    require(digest(folder / 'CUMULATIVE_SOURCE.json') == digest(final_scene), 'final_source_must_match_B3_bytes')
    source = read(folder / 'CUMULATIVE_SOURCE.json')
    review = read(folder / 'SOURCE_REVIEW.json')
    require(review['source_sha256'] == digest(folder / 'CUMULATIVE_SOURCE.json'), 'source_review_hash')
    require(review['decision'] == 'READY' and review['full_source_read'] is True, 'source_not_ready')
    require(review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']], 'source_review_coverage')
    required = {'time','location','actor_identity','plan_vs_completion','cross_scene_continuity','semantic_repetition','naturalness','event_fidelity'}
    require(required.issubset(review['checks']), 'source_checks_incomplete')
    require(all(review['checks'][k].get('status') in ('PASS','WARNING','PASS_WITH_WARNING') and review['checks'][k].get('reason') for k in required),'source_check_failed_or_unexplained')
    require(not any(f.get('severity') == 'BLOCKING' for f in review['findings']), 'blocking_source_finding')
    return source

def review_header(review, folder, public_name):
    source = read(folder / 'CUMULATIVE_SOURCE.json')
    require(review['source_sha256'] == digest(folder / 'CUMULATIVE_SOURCE.json'), 'review_source_hash')
    require(review['public_sha256'] == digest(folder / public_name), 'review_public_hash')
    require(review['full_source_read'] is True, 'full_source_read_not_attested')
    require(review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']], 'review_read_coverage')
    require(review['private_answers_read_before_save'] is False, 'source_first_not_attested')
    require(review['independent_of_question_author'] is True, 'question_author_cannot_self_review')
    return {t['turn_ref']: t for t in source['turns']}

def check_statement(judgment, proposition, turns):
    issues = []
    if judgment.get('proposition') != proposition: issues.append('proposition_mismatch')
    if judgment.get('relation') not in ('supports','contradicts','unknown'): issues.append('invalid_relation')
    if judgment.get('scope_valid') is not True: issues.append('scope_invalid')
    if not judgment.get('reason') or not judgment.get('citations'): issues.append('missing_reason_or_citations')
    for citation in judgment.get('citations', []):
        turn = turns.get(citation.get('turn_ref'))
        if not turn or not citation.get('quote') or citation['quote'] not in turn['text']:
            issues.append('citation_not_exact')
        elif citation.get('speaker') != turn['speaker']:
            issues.append('citation_speaker_mismatch')
    return issues

def anchor_key(anchor, turns, positions, scene):
    turn = turns[anchor['turn_ref']]; quote = anchor['quote']
    require(quote and turn['text'].count(quote) == 1, 'anchor_not_unique_exact_span')
    require(turn['scene_ref'] in scene, 'anchor_outside_public_scope')
    start = turn['text'].index(quote)
    return positions[anchor['turn_ref']], start, start + len(quote)

def check_ordering(question, judgment, source, scene, fixed_turns=None):
    issues = []; turns = {t['turn_ref']: t for t in source['turns']}
    positions = {t['turn_ref']: i for i,t in enumerate(source['turns'])}
    for field in ('stem_valid','scope_valid','coherent','unique_order','neutral_stem_ok'):
        if judgment.get(field) is not True: issues.append(field + '_failed')
    if set(question['items']) != {f'I{i}' for i in range(1,7)}: issues.append('six_items_required')
    if set(judgment.get('items',{})) != set(question['items']): return issues + ['item_judgments_incomplete'], []
    choices = []
    for label,text in question['items'].items():
        j = judgment['items'][label]
        if j.get('proposition') != text: issues.append(label + ':proposition_mismatch')
        for field in ('supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok'):
            if j.get(field) is not True: issues.append(label + ':' + field + '_failed')
        if not j.get('support_refs') or any(ref not in turns for ref in j['support_refs']): issues.append(label + ':support_refs')
        if not j.get('reason'): issues.append(label + ':reason_missing')
        if fixed_turns and j.get('order_anchor',{}).get('turn_ref') != fixed_turns[label]: issues.append(label + ':outside_frozen_turn')
        resolved = []
        for anchor in [j.get('order_anchor',{})] + j.get('alternative_matching_anchors',[]):
            try: resolved.append((label,anchor_key(anchor,turns,positions,scene)))
            except (KeyError, ValueError, TypeError) as exc: issues.append(label + ':bad_anchor:' + str(exc))
        choices.append(resolved)
    orders = []
    for combination in itertools.product(*choices):
        ordered = sorted(combination,key=lambda v:v[1][:2])
        if any(a[1][0] == b[1][0] and a[1][2] > b[1][1] for a,b in zip(ordered,ordered[1:])):
            issues.append('overlapping_semantic_anchors')
        orders.append(tuple(v[0] for v in ordered))
    unique = set(orders)
    if len(unique) != 1: issues.append('semantic_order_not_unique')
    derived = list(orders[0]) if orders else []
    if judgment.get('derived_order') != derived: issues.append('review_derived_order_mismatch')
    if judgment.get('alternative_orders') and any(list(v) != derived for v in judgment['alternative_orders']): issues.append('alternative_order_reported')
    return sorted(set(issues)), derived

def ordering_questions(batch, row, source, draft, cfg):
    folder = root_folder(batch,row['root_id']); plan = read(folder/'ORDERING_ANCHOR_PLAN.json')
    require(plan['source_sha256'] == digest(folder/'CUMULATIVE_SOURCE.json'), 'anchor_plan_source_hash')
    turns = {t['turn_ref']:t for t in source['turns']}
    # Saving the plan before the draft is a necessary file-sequence condition;
    # truthful author receipts remain required for semantic procedure claims.
    require((folder/'ORDERING_ANCHOR_PLAN.json').stat().st_mtime_ns <= (folder/'QA_DRAFT.json').stat().st_mtime_ns, 'anchor_plan_not_saved_first')
    out = []; mapping = []; failures = []
    require(len(plan['slots']) == 2 and len({str(slot['slot_id']) for slot in plan['slots']}) == 2, 'exactly_two_distinct_ordering_slots')
    for slot in plan['slots']:
        anchors = slot['anchors']; by_id = {a['anchor_id']:a for a in anchors}
        require(len(anchors)==6 and len({a['turn_ref'] for a in anchors})==6,'anchor_slot_six_distinct_turns')
        require({turns[a['turn_ref']]['scene_ref'] for a in anchors} == set(scope_scenes(cfg)), 'ordering_requires_one_anchor_per_scene')
        for a in anchors:
            require(turns[a['turn_ref']]['text']==a['full_turn_quote'] and turns[a['turn_ref']]['speaker']==a['speaker'],'anchor_frozen_quote')
            require(turns[a['turn_ref']]['scene_ref'] in scope_scenes(cfg),'anchor_scene')
        item = next(x for x in draft['ordering'] if str(x['slot_id'])==str(slot['slot_id']))
        cid = cfg['batch_id'] + '-root' + row['root_id'].rsplit('-',1)[-1] + '-ordering' + str(slot['slot_id'])
        if item.get('failure'):
            failures.append({'question_type':'ordering','slot_id':str(slot['slot_id']),
                'candidate_id':cid,'reason':'author_failure','failure':item['failure']})
            continue
        items = item['items']
        require({x['item_id'] for x in items}=={f'I{i}' for i in range(1,7)},'ordering_item_ids')
        require({x['anchor_id'] for x in items}==set(by_id),'ordering_anchor_coverage')
        for x in items:
            require(bool(x['support_quote']) and x['support_quote'] in by_id[x['anchor_id']]['full_turn_quote'],'ordering_support_quote')
        out.append({'candidate_id':cid,'root_id':row['root_id'],'question_type':'ordering','chronology_kind':'conversation_order',
            'question':f"Order the statements, using left-to-right clause order within a turn, as they APPEAR IN THE SUPPLIED DIALOGUE within these six scenes: {', '.join(scope_scenes(cfg))}.",
            'items':{x['item_id']:x['text'] for x in items},'source_path':str(folder/'CUMULATIVE_SOURCE.json'),
            'source_sha256':digest(folder/'CUMULATIVE_SOURCE.json')})
        mapping.append({'candidate_id':cid,'role':'ordering','slot_id':str(slot['slot_id']),
            'fixed_turns':{x['item_id']:by_id[x['anchor_id']]['turn_ref'] for x in items},
            'item_anchor_ids':{x['item_id']:x['anchor_id'] for x in items}})
    return out,mapping,failures

def publish_assets(batch):
    batch = Path(batch); cfg=config(batch)
    for row in configured_roots(cfg):
        folder=root_folder(batch,row['root_id'])
        if not (folder/'QA_DRAFT.json').exists() or (folder/'PUBLIC_ASSETS.json').exists(): continue
        source=source_ready(folder); draft=read(folder/'QA_DRAFT.json')
        require(len(draft['facts'])==6,'six_distinct_facts_required')
        require(len({f['fact_id'] for f in draft['facts']})==6,'duplicate_fact_ids')
        turns = {turn['turn_ref']: turn for turn in source['turns']}
        require(all(fact.get('source_turn_ref') in turns for fact in draft['facts']), 'fact_source_turn_ref_required')
        require({turns[fact['source_turn_ref']]['scene_ref'] for fact in draft['facts']} == set(scope_scenes(cfg)), 'facts_require_one_per_scene')
        require(all(scene in draft['stem'] for scene in scope_scenes(cfg)), 'public_multiselect_stem_must_name_all_six_scenes')
        statements=[]; facts=[]
        for fact in draft['facts']:
            mapped={'fact_id':fact['fact_id'],'source_turn_ref':fact['source_turn_ref'],'true_statement_id':None,'false_statement_id':None}
            for kind in ('true','false'):
                text=fact.get(kind+'_text')
                if text is None:continue
                require(isinstance(text,str) and bool(text.strip()),'empty_fact')
                sid='S-'+canonical_digest([cfg['batch_id'],row['root_id'],fact['fact_id'],text])[:16]
                statements.append({'statement_id':sid,'proposition':text})
                mapped[kind+'_statement_id']=sid
            facts.append(mapped)
        rng=random.Random(cfg['seed_namespace']+row['root_id']+'assets');rng.shuffle(statements)
        orderings,order_map,ordering_failures=ordering_questions(batch,row,source,draft,cfg)
        save(folder/'PUBLIC_ASSETS.json',{'root_id':row['root_id'],'source_sha256':digest(folder/'CUMULATIVE_SOURCE.json'),
            'scope_scene_refs':scope_scenes(cfg),'multiselect_stem':draft['stem'],'statements':statements,
            'ordering_questions':orderings,'ordering_failures':ordering_failures})
        save(folder/'PRIVATE_ASSET_DESIGN.json',{'root_id':row['root_id'],'qa_draft_sha256':digest(folder/'QA_DRAFT.json'),
            'public_assets_sha256':digest(folder/'PUBLIC_ASSETS.json'),'facts':facts,'ordering':order_map})
        print(row['root_id'],'PUBLISHED_ASSETS',len(statements),len(orderings))

def compose(batch):
    batch=Path(batch);cfg=config(batch)
    for row in configured_roots(cfg):
        folder=root_folder(batch,row['root_id'])
        if not (folder/'ASSET_SOURCE_JUDGMENTS.json').exists() or (folder/'PUBLIC_QUESTIONS.json').exists():continue
        source=source_ready(folder);assets=read(folder/'PUBLIC_ASSETS.json');mapping=read(folder/'PRIVATE_ASSET_DESIGN.json')
        review=read(folder/'ASSET_SOURCE_JUDGMENTS.json');turns=review_header(review,folder,'PUBLIC_ASSETS.json')
        judgments=review['statements']; issues={}; valid_false=[]; truth_ok=True
        require(set(judgments)=={x['statement_id'] for x in assets['statements']},'asset_judgments_incomplete')
        by_id={x['statement_id']:x['proposition'] for x in assets['statements']}
        for sid,text in by_id.items():issues[sid]=check_statement(judgments[sid],text,turns)
        for fact in mapping['facts']:
            true=fact['true_statement_id'];false=fact['false_statement_id']
            if true is None or issues.get(true) or judgments[true]['relation']!='supports':truth_ok=False
            if false is not None and not issues.get(false) and judgments[false]['relation']=='contradicts':valid_false.append(fact['fact_id'])
        questions=[];designs=[];unfilled=[]
        for cardinality in [6]+cfg['multiselect_target_cardinalities']:
            need=6-cardinality
            if not truth_ok or len(valid_false)<need:
                unfilled.append({'question_type':'multi_select','intended_cardinality':cardinality,
                    'reason':'truth_not_supported' if not truth_ok else 'insufficient_reviewed_contradictory_variants','valid_false_count':len(valid_false)});continue
            rng=random.Random(cfg['seed_namespace']+row['root_id']+str(cardinality))
            false_ids=set(rng.sample(sorted(valid_false),need));facts=list(mapping['facts']);rng.shuffle(facts)
            labels_to_sid={label:fact['false_statement_id' if fact['fact_id'] in false_ids else 'true_statement_id'] for label,fact in zip('ABCDEF',facts)}
            cid=cfg['batch_id']+'-ms-'+canonical_digest([row['root_id'],cardinality,cfg['seed_namespace']])[:16]
            questions.append({'candidate_id':cid,'root_id':row['root_id'],'question_type':'multi_select','question':assets['multiselect_stem'],
                'options':{label:by_id[sid] for label,sid in labels_to_sid.items()},'source_path':str(folder/'CUMULATIVE_SOURCE.json'),'source_sha256':digest(folder/'CUMULATIVE_SOURCE.json')})
            designs.append({'candidate_id':cid,'role':'control6' if cardinality==6 else 'new_child','intended_cardinality':cardinality,
                'labels_to_statement_ids':labels_to_sid,'labels_to_intended_truth':{label:fact['fact_id'] not in false_ids for label,fact in zip('ABCDEF',facts)}})
        unfilled.extend(assets.get('ordering_failures',[]))
        ordering_reviews={x['candidate_id']:x for x in review['ordering_questions']}
        for q in assets['ordering_questions']:
            d=next(x for x in mapping['ordering'] if x['candidate_id']==q['candidate_id'])
            j=ordering_reviews[q['candidate_id']]
            failures,derived=check_ordering(q,j,source,scope_scenes(cfg),d['fixed_turns'])
            if failures:
                unfilled.append({'question_type':'ordering','slot_id':d['slot_id'],'candidate_id':q['candidate_id'],'reason':'asset_review_failed','issues':failures});continue
            questions.append(q);designs.append(d)
        save(folder/'PUBLIC_QUESTIONS.json',{'questions':questions})
        save(folder/'PRIVATE_DESIGN.json',{'source_sha256':digest(folder/'CUMULATIVE_SOURCE.json'),
            'public_assets_sha256':digest(folder/'PUBLIC_ASSETS.json'),'asset_judgments_sha256':digest(folder/'ASSET_SOURCE_JUDGMENTS.json'),'design':designs})
        save(folder/'COMPOSITION_RECEIPT.json',{'created_at_utc':now(),'public_sha256':digest(folder/'PUBLIC_QUESTIONS.json'),
            'public_count':len(questions),'valid_false_fact_ids':valid_false,'all_six_truths_supported':truth_ok,
            'statement_mechanical_issues':issues,'unfilled_slots':unfilled,
            'unknown_variants_composed':False,'gold_not_constructed_yet':True})
        print(row['root_id'],'COMPOSED',len(questions),'unfilled',len(unfilled))

def run_stage(batch, command):
    {'publish-assets':publish_assets,'compose':compose,'finalize':finalize}[command](batch)

def validate_final_review(folder, cfg):
    """Validate recorded judgments before reading any answer-side design."""
    source=source_ready(folder);public=read(folder/'PUBLIC_QUESTIONS.json')
    review=read(folder/'PUBLIC_SOURCE_JUDGMENTS.json')
    turns=review_header(review,folder,'PUBLIC_QUESTIONS.json')
    questions=public['questions']; judgments=review['questions']
    require(len(questions)==len({q['candidate_id'] for q in questions}),'duplicate_candidate_id')
    require(set(judgments)=={q['candidate_id'] for q in questions},'final_judgments_incomplete')
    outcomes={}
    for q in questions:
        j=judgments[q['candidate_id']];issues=[];derived=[]
        require(q['source_sha256']==digest(folder/'CUMULATIVE_SOURCE.json'),'public_source_hash')
        if q['question_type']=='multi_select':
            for field in ('stem_valid','scope_valid','fact_distinct','no_entangled_options'):
                if j.get(field) is not True:issues.append(field+'_failed')
            if set(q['options'])!=set('ABCDEF'):issues.append('six_options_required')
            if set(j.get('options',{}))!=set(q['options']):
                raise ValueError('option_judgments_incomplete')
            else:
                for label,text in q['options'].items():
                    sj=j['options'][label]
                    require({'proposition','relation','scope_valid','fact_distinct','reason','citations'}.issubset(sj),'incomplete_option_judgment')
                    issues.extend(label+':'+v for v in check_statement(sj,text,turns))
                    if sj.get('fact_distinct') is not True:issues.append(label+':fact_distinct_failed')
                    if sj.get('relation')=='unknown':issues.append(label+':unknown')
                derived=[label for label in 'ABCDEF' if j['options'][label].get('relation')=='supports']
        elif q['question_type']=='ordering':
            require(set(j.get('items',{}))==set(q['items']),'item_judgments_incomplete')
            require(all({'proposition','supported','actor_ok','modality_ok','state_ok','neutral_stem_ok','single_event_ok','order_anchor','support_refs','alternative_matching_anchors','reason'}.issubset(v) for v in j['items'].values()),'incomplete_ordering_item_judgment')
            issues,derived=check_ordering(q,j,source,scope_scenes(cfg))
        else:issues.append('unsupported_type')
        outcomes[q['candidate_id']]={'source_issues':sorted(set(issues)),'source_derived_answer':derived}
    return source,questions,review,outcomes

def finalize(batch):
    """Create original-design Gold only after the entire public review validates."""
    batch=Path(batch);cfg=config(batch)
    for row in configured_roots(cfg):
        folder=root_folder(batch,row['root_id'])
        if not (folder/'PUBLIC_SOURCE_JUDGMENTS.json').exists() or (folder/'FINAL_CROSSCHECK.json').exists():continue
        source,questions,review,outcomes=validate_final_review(folder,cfg)
        # No design or answer file is read above this line.
        design=read(folder/'PRIVATE_DESIGN.json')
        require(design['source_sha256']==digest(folder/'CUMULATIVE_SOURCE.json'),'design_source_hash')
        require(design['asset_judgments_sha256']==digest(folder/'ASSET_SOURCE_JUDGMENTS.json'),'asset_review_hash_changed')
        require(design['public_assets_sha256']==digest(folder/'PUBLIC_ASSETS.json'),'assets_hash_changed')
        ds={d['candidate_id']:d for d in design['design']}
        require(set(ds)=={q['candidate_id'] for q in questions},'design_coverage')
        private=[];results=[];positions={t['turn_ref']:i for i,t in enumerate(source['turns'])}
        for q in questions:
            cid=q['candidate_id'];d=ds[cid];j=review['questions'][cid];out=outcomes[cid]
            issues=list(out['source_issues'])
            if q['question_type']=='multi_select':
                gold=[label for label in 'ABCDEF' if d['labels_to_intended_truth'][label]]
                require(len(gold)==d['intended_cardinality'],'original_design_cardinality_invalid')
                evidence={label:j['options'].get(label,{}).get('citations',[]) for label in 'ABCDEF'}
            else:
                gold=sorted(d['fixed_turns'],key=lambda label:positions[d['fixed_turns'][label]])
                evidence={label:j['items'].get(label,{}).get('order_anchor') for label in q['items']}
                for label,ref in d['fixed_turns'].items():
                    if j['items'].get(label,{}).get('order_anchor',{}).get('turn_ref')!=ref:
                        issues.append(label+':outside_frozen_turn')
            if out['source_derived_answer']!=gold:issues.append('original_gold_source_judgment_conflict')
            private.append({'candidate_id':cid,'role':d['role'],'gold':gold,'evidence':evidence})
            results.append({'candidate_id':cid,'root_id':row['root_id'],'question_type':q['question_type'],'role':d['role'],
                'status':'CONTENT_PASS' if not issues else 'EXCLUDED','issues':sorted(set(issues)),
                'original_gold_unchanged':True,'source_derived_answer':out['source_derived_answer']})
        save(folder/'PRIVATE_GOLD.json',{'created_at_utc':now(),'source_sha256':digest(folder/'CUMULATIVE_SOURCE.json'),
            'public_sha256':digest(folder/'PUBLIC_QUESTIONS.json'),'source_judgments_sha256':digest(folder/'PUBLIC_SOURCE_JUDGMENTS.json'),
            'original_design_sha256':digest(folder/'PRIVATE_DESIGN.json'),'gold_constructed_after_complete_public_review':True,'questions':private})
        save(folder/'FINAL_CROSSCHECK.json',{'created_at_utc':now(),'private_sha256':digest(folder/'PRIVATE_GOLD.json'),
            'public_sha256':digest(folder/'PUBLIC_QUESTIONS.json'),'source_judgments_sha256':digest(folder/'PUBLIC_SOURCE_JUDGMENTS.json'),
            'semantic_review_is_programmatic':False,'results':results})
        print(row['root_id'],'FINALIZED',sum(r['status']=='CONTENT_PASS' for r in results),'/',len(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['publish-assets', 'compose', 'finalize'])
    parser.add_argument('--batch-dir', type=Path, required=True)
    args = parser.parse_args()
    run_stage(args.batch_dir, args.stage)

if __name__ == '__main__':
    main()
