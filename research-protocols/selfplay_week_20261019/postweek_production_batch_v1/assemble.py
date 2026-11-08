"""Assemble reviewed production slots without changing frozen drafts or answers."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GATE_PATH = HERE.parents[2] / "scripts/current_method/production_question_gate.py"
sys.path.insert(0, str(GATE_PATH.parent))
spec = importlib.util.spec_from_file_location("production_gate_assembly", GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def review_rows(doc):
    return doc if isinstance(doc, list) else doc["reviews"]


def main():
    selection_path = HERE / "SELECTION_FREEZE.json"
    selection = read(selection_path)
    expected = {f"PROD1-{root['root_id'].split('-')[-1]}-{slot['slot']}": (root, slot)
                for root in selection["roots"] for slot in root["slot_plan"]}
    candidates, reviews, slots, bindings, gate_results = [], [], [], [], []
    for group in ("a", "b"):
        draft_path = HERE / f"author_{group}/CANDIDATES_PRIVATE_DRAFT.json"
        final_path = HERE / f"author_{group}/CANDIDATES_PRIVATE.json"
        initial_path = HERE / f"review_{group}/INITIAL_CONTENT_REVIEW_AI.json"
        review_path = HERE / f"review_{group}/CONTENT_REVIEW_AI.json"
        ledger_path = HERE / f"author_{group}/SLOT_LEDGER_DRAFT.json"
        draft_rows, final_rows = read(draft_path), read(final_path)
        initial = {r["candidate_id"]: r for r in review_rows(read(initial_path))}
        final_reviews = {r["candidate_id"]: r for r in review_rows(read(review_path))}
        draft_by_id = {c["candidate_id"]: c for c in draft_rows}
        assert len(draft_by_id) == len(draft_rows), "DUPLICATE_DRAFT_ID"
        assert len({c['candidate_id'] for c in final_rows}) == len(final_rows)
        assert set(draft_by_id) == set(initial) == set(final_reviews) == {c['candidate_id'] for c in final_rows}
        bound, binding_audit = gate.bind_candidates(final_rows, selection_path)
        assert bound == final_rows, "SOURCE_BINDING_MUST_BE_DERIVED_BEFORE_REVIEW"
        submitted = set()
        for candidate in final_rows:
            cid = candidate["candidate_id"]
            root, slot = expected[cid]
            assert candidate['root_id'] == root['root_id'] and root['author_group'] == group
            assert candidate['question_type'] == slot['question_type']
            assert candidate['task_operation'] == slot['task_operation'], "FROZEN_OPERATION_DRIFT:" + cid
            if candidate['question_type'] == 'multi_select':
                assert len(candidate['gold']) == slot['gold_count'], "FROZEN_GOLD_COUNT_DRIFT:" + cid
            else:
                assert 4 <= len(candidate['items']) <= 6
            original = initial[cid]
            assert original['candidate_sha256'] == gate.fingerprint(draft_by_id[cid]), "INITIAL_REVIEW_DRIFT:" + cid
            review = dict(final_reviews[cid])
            result = gate.validate_review(candidate, review, selection_path)
            assert result['review_record_complete'], (cid, result['review_errors'])
            if review['decision'] == 'PASS':
                assert result['admitted'], (cid, result)
            review.update(initial_decision=original['decision'],
                          initial_content_decision=original.get('initial_content_decision', original['decision']),
                          initial_candidate_sha256=gate.fingerprint(draft_by_id[cid]),
                          repaired_after_initial_review=gate.fingerprint(candidate) != gate.fingerprint(draft_by_id[cid]),
                          initial_review_binding=gate.file_binding(initial_path))
            candidates.append(candidate)
            reviews.append(review)
            gate_results.append(result)
            submitted.add(cid)
            slots.append({'candidate_id': cid, 'root_id': root['root_id'], 'slot': slot['slot'],
                          'task_operation': slot['task_operation'], 'status': review['decision'],
                          'initial_decision': original['decision'],
                          'repaired': review['repaired_after_initial_review'],
                          'candidate_sha256': gate.fingerprint(candidate)})
        ledger_doc = read(ledger_path)
        ledger = ledger_doc if isinstance(ledger_doc, list) else ledger_doc['slots']
        by_slot = {r.get('candidate_id', r.get('slot_id')): r for r in ledger}
        expected_group = {cid for cid, (root, _) in expected.items() if root['author_group'] == group}
        assert set(by_slot) == expected_group, "EVERY_PLANNED_SLOT_REQUIRES_LEDGER"
        for cid in sorted(expected_group - submitted):
            row = by_slot[cid]
            assert row['status'] == 'MATERIAL_SHORTFALL', (cid, row)
            root, slot = expected[cid]
            slots.append({**row, 'candidate_id': cid, 'root_id': root['root_id'],
                          'slot': slot['slot'], 'task_operation': slot['task_operation']})
        bindings.extend(gate.file_binding(p) for p in (draft_path, final_path, initial_path, review_path,
                                                      ledger_path, HERE / f'author_{group}/AUTHOR_READ_RECEIPT.json'))
    candidates.sort(key=lambda c: c['candidate_id'])
    reviews.sort(key=lambda r: r['candidate_id'])
    slots.sort(key=lambda r: r['candidate_id'])
    assert len(slots) == 70 and len({s['candidate_id'] for s in slots}) == 70
    signatures = {}
    for c in candidates:
        choices = c['options' if c['question_type'] == 'multi_select' else 'items']
        key = gate.fingerprint({'type': c['question_type'], 'question': c['question'],
                                'choices': sorted(choices.values())})
        assert key not in signatures, 'EXACT_WHOLE_QUESTION_DUPLICATE:' + c['candidate_id']
        signatures[key] = c['candidate_id']
    save(HERE / 'CANDIDATES_PRIVATE.json', candidates)
    save(HERE / 'CONTENT_REVIEW_AI.json', {'schema_version': 'production_independent_content_review_v1',
         'reviewer_is_human': False, 'gold_blind': False, 'solver_results_read': False,
         'author_reviewer_separation': True, 'reviews': reviews, 'bindings': bindings})
    save(HERE / 'SLOT_LEDGER.json', {'planned_slots': 70, 'slots': slots,
         'counts': dict(Counter(s['status'] for s in slots)), 'unfilled_slots_retained': True,
         'solver_score_filtering': False, 'silent_operation_substitution': False})
    save(HERE / 'audit/PRODUCTION_GATE_AUDIT.json', {'schema_version': 'production_gate_collection_v1',
         'status': 'PASS_REVIEW_INTEGRITY', 'submitted_candidates': len(candidates),
         'mechanical_pass': sum(r['mechanical_pass'] for r in gate_results),
         'admitted': sum(r['admitted'] for r in gate_results), 'results': gate_results,
         'new_gate_binding': gate.file_binding(GATE_PATH)})
    first_pass = sum(r['initial_decision'] == 'PASS' for r in reviews)
    save(HERE / 'COLLECTION_AUDIT.json', {'planned_candidates': 70,
         'submitted_candidates': len(candidates), 'whole_question_exact_duplicates_in_batch': 0,
         'initial_joint_pass': first_pass, 'initial_joint_pass_rate_planned': first_pass / 70,
         'initial_joint_pass_rate_submitted': first_pass / len(candidates),
         'initial_semantic_content_pass': sum(r['initial_content_decision'] == 'PASS' for r in reviews),
         'final_pass': sum(r['decision'] == 'PASS' for r in reviews),
         'material_shortfalls': [s for s in slots if s['status'] == 'MATERIAL_SHORTFALL'],
         'initial_joint_90_percent_scaleup_threshold_met': first_pass / 70 >= .9,
         'repaired_candidates': [r['candidate_id'] for r in reviews if r['repaired_after_initial_review']],
         'new_dialogue_turns': 0, 'bindings': bindings})
    print(json.dumps({'planned_slots': 70, 'submitted_candidates': len(candidates),
                      'initial_joint_pass': first_pass, 'final_decisions': dict(Counter(r['decision'] for r in reviews)),
                      'shortfalls': 70-len(candidates)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
