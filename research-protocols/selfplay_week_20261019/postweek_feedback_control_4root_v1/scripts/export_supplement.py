"""Export independently reviewed presentation supplement, retaining original yield."""
import json, re
from control import BASE, read, write, bind, verify, now, SCENES
from questions import SLOTS, mechanical
from normalize_presentation import PATTERN


def review_rows(path, root, arm):
    review = read(path)
    assert review['root_id'] == root and review['condition'] == arm
    assert review['independent_of_question_author'] and review['independent_of_source_author']
    assert review['full_source_read'] and review['solver_scores_read'] is False
    source = verify(review['source_binding'])
    assert review['turn_refs_read'] == [t['turn_ref'] for t in source['turns']]
    rows = {r['slot_id']: r for r in review['questions']}
    assert set(rows) == set(SLOTS) and len(review['questions']) == len(SLOTS)
    return review, rows


def export():
    protocol = read(BASE/'PROTOCOL_FREEZE.json')
    approval = read(BASE/'PRESENTATION_AMENDMENT_REVIEW.json')
    assert approval['status'] == 'SUPPORTED_AS_EXPLORATORY_SUPPLEMENT'
    assert approval['amendment_binding'] == bind(BASE/'PRESENTATION_AMENDMENT.json')
    assert approval['implementation_binding'] == bind(BASE/'scripts/normalize_presentation.py')
    assert approval['original_protocol_binding'] == bind(BASE/'PROTOCOL_FREEZE.json')
    amendment = read(BASE/'PRESENTATION_AMENDMENT.json')
    assert amendment['pattern'] == PATTERN and amendment['implementation'] == approval['implementation_binding']
    for binding in approval['triggering_locked_review_bindings']: verify(binding)
    # No raw reruns, alternative permutations, or adaptive question selection.
    assert not (BASE/'solver_runtime/execution/STARTED.json').exists()
    roots = protocol['roots']
    original_decisions, decisions, records, review_bindings = [], [], [], []
    public, private, seen = [], [], set()
    for root in roots:
        for arm in protocol['conditions']:
            unit = BASE/'sources'/root/arm
            final = unit/'FINAL_SOURCE.json'
            if not final.exists():
                failed = [unit/s.replace(':','_')/'REVIEW.json' for s in SCENES
                          if (unit/s.replace(':','_')/'REVIEW.json').exists()
                          and read(unit/s.replace(':','_')/'REVIEW.json')['decision'] != 'READY']
                assert failed, ('UNFINISHED_SOURCE_IS_NOT_FAILURE', root, arm)
                for slot in SLOTS:
                    row = {'root_id': root, 'condition': arm, 'slot_id': slot,
                           'status': 'SOURCE_GATE_NOT_READY_NO_SOLVER',
                           'source_gate_decisions': [read(p)['decision'] for p in failed],
                           'source_failure_reviews': [bind(p) for p in failed]}
                    original_decisions.append(row); decisions.append(row)
                continue
            assert all(read(unit/s.replace(':','_')/'ACCEPTED.json')['status']=='READY' for s in SCENES)
            raw_review_path = BASE/'question_reviews'/root/arm/'REVIEW.json'
            normalized_review_path = BASE/'normalized_question_reviews'/root/arm/'REVIEW.json'
            raw_review, raw_rows = review_rows(raw_review_path, root, arm)
            review, rows = review_rows(normalized_review_path, root, arm)
            assert review['source_binding'] == raw_review['source_binding'] == bind(final)
            receipt_path = BASE/'normalized_questions'/root/arm/'TRANSFORM_RECEIPT.json'
            receipt = read(receipt_path)
            assert receipt['gold_used_for_replacement'] is False and receipt['author_attempts_added'] == 0
            assert receipt['solver_requests_added'] == 0 and receipt['pattern'] == PATTERN
            receipt_rows = {r['slot_id']:r for r in receipt['records']}
            assert set(receipt_rows) == set(SLOTS) and len(receipt['records']) == 7
            review_bindings += [bind(raw_review_path), bind(normalized_review_path), bind(receipt_path)]
            for slot in SLOTS:
                raw_path = BASE/'questions'/root/arm/(slot+'.json')
                path = BASE/'normalized_questions'/root/arm/(slot+'.json')
                raw_row, row = raw_rows[slot], rows[slot]
                common = {'root_id': root, 'condition': arm, 'slot_id': slot}
                original_decisions.append({**common, 'status': raw_row['decision'], 'review': bind(raw_review_path)})
                if not raw_path.exists():
                    assert not path.exists() and raw_row['decision'] in ('SHORTFALL','FAIL') and row['decision'] in ('SHORTFALL','FAIL')
                    decisions.append({**common, 'status': row['decision'], 'review': bind(normalized_review_path)})
                    continue
                q, raw = read(path), read(raw_path)
                assert raw_row['question_binding'] == bind(raw_path)
                assert row['question_binding'] == bind(path)
                assert raw_row['reviewer_id'] != raw['author_id'] and row['reviewer_id'] != q['author_id']
                assert q['presentation_parent_binding'] == bind(raw_path)
                assert q['source_binding'] == bind(final)
                expected, replacements = raw['question'], 0
                if raw['question_type'] == 'multi_select':
                    expected, replacements = re.subn(PATTERN, lambda m:m.group(1)+' ', expected, flags=re.I)
                assert replacements <= 1 and q['question'] == expected
                transform = q['presentation_transform']
                assert transform['version'] == 'fbc1_shared_count_cue_removal_v1'
                assert transform['replacement_count'] == replacements
                assert transform['gold_used_for_replacement'] is False and transform['author_regeneration'] is False
                assert transform['source_options_items_gold_evidence_display_unchanged'] is True
                assert transform['amendment_binding'] == approval['amendment_binding']
                transformed = receipt_rows[slot]
                assert transformed['original'] == bind(raw_path) and transformed['normalized'] == bind(path)
                assert transformed['before'] == raw['question'] and transformed['after'] == expected
                assert transformed['replacements'] == replacements
                unchanged = {k:v for k,v in q.items() if k not in ('question','presentation_parent_binding','presentation_transform')}
                assert unchanged == {k:v for k,v in raw.items() if k != 'question'}
                status = row['decision']
                if status == 'ADMIT':
                    assert row['submitted_gold_matches'] and row['operation_alignment'] == 'PASS'
                    mechanical(path)
                    source = verify(q['source_binding'])
                    identity = json.dumps({'full_source_context': [{k:t[k] for k in ('turn_ref','speaker','text')} for t in source['turns']],
                                           'question': q['question'], 'items': q.get('options') or q.get('items')}, sort_keys=True)
                    if identity in seen: status = 'EXACT_WHOLE_DUPLICATE'
                    else: seen.add(identity)
                decisions.append({**common, 'candidate_id':q['candidate_id'], 'status':status,
                                  'original_status':raw_row['decision'], 'review':bind(normalized_review_path)})
                if status != 'ADMIT': continue
                labels = list('ABCDEF' if slot.startswith('MC') else 'ABCD')
                canonical = q.get('options') or q['items']
                mapping = dict(zip(labels,q['display_order'])); reverse = {v:k for k,v in mapping.items()}
                common_public = {k:q[k] for k in ('candidate_id','root_id','condition','question_type','task_operation','question','source_binding')}
                public.append({**common_public, 'display':{k:canonical[v] for k,v in mapping.items()}})
                private.append({'candidate_id':q['candidate_id'],'question_type':q['question_type'],
                                'canonical_gold':q['gold'],'display_gold':[reverse[k] for k in q['gold']],
                                'display_to_canonical':mapping,'question_binding':bind(path)})
                records.append(bind(path))
    assert len(decisions) == len(original_decisions) == 56
    original = {'schema_version':'fbc1_original_first_pass_quality_ledger_v1','created_at_utc':now(),
                'planned_slots':56,'admitted_before_presentation_supplement':sum(r['status']=='ADMIT' for r in original_decisions),
                'decisions':original_decisions,'original_protocol':bind(BASE/'PROTOCOL_FREEZE.json'),'solver_scores_read':False}
    write(BASE/'ORIGINAL_QUESTION_QUALITY_FREEZE.json',original)
    bindings = [bind(BASE/p) for p in ('PROTOCOL_FREEZE.json','PRESENTATION_AMENDMENT.json','PRESENTATION_AMENDMENT_REVIEW.json',
                                      'DUPLICATE_IDENTITY_CLARIFICATION.json','scripts/normalize_presentation.py','scripts/export_supplement.py')]
    freeze = {'schema_version':'fbc1_supplement_question_freeze_v1','status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),
              'analysis_scope':'EXPLORATORY_SHARED_PRESENTATION_SUPPLEMENT','unchanged_preregistered_primary_analysis':False,
              'planned_slots':56,'admitted_questions':len(public),'decisions':decisions,'question_bindings':records,
              'original_first_pass_quality_ledger':bind(BASE/'ORIGINAL_QUESTION_QUALITY_FREEZE.json'),
              'bindings':bindings+review_bindings,'solver_scores_read':False}
    write(BASE/'QUESTION_FREEZE.json',freeze)
    write(BASE/'inputs/PUBLIC_QUESTIONS.json',{'questions':public})
    write(BASE/'inputs/PRIVATE_SCORING.json',{'questions':private})
    result = {'schema_version':'fbc1_solver_export_v1','status':'FROZEN_BEFORE_SOLVER','created_at_utc':now(),
              'analysis_scope':freeze['analysis_scope'],'unchanged_preregistered_primary_analysis':False,
              'public_binding':bind(BASE/'inputs/PUBLIC_QUESTIONS.json'),'private_binding':bind(BASE/'inputs/PRIVATE_SCORING.json'),
              'question_freeze_binding':bind(BASE/'QUESTION_FREEZE.json'),'planned_root_ids':roots,'planned_slots':56,
              'content_review_status':'PASS','solver_score_filtering':False,'full_source_no_truncation':True,
              'bindings':bindings+review_bindings+records+[bind(BASE/'ORIGINAL_QUESTION_QUALITY_FREEZE.json')]}
    write(BASE/'inputs/EXPORT_FREEZE.json',result)
    return {'admitted':len(public),'requests':2*len(public),'export':bind(BASE/'inputs/EXPORT_FREEZE.json')}


if __name__ == '__main__':
    print(json.dumps(export(),ensure_ascii=False))
