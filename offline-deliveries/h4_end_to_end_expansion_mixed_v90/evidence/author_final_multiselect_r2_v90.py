"""Serialize explicit source-first whole-question decisions for the r2 public set.

Each label's semantic decision and cited source turn were authored after reading
the complete 127/204-turn sources and all current public questions. No round
judgments, writer response, child private artifact, pool or Solver is loaded.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARD = HERE.parent / 'shards/shard_01r2'
PUBLIC = SHARD / 'multiselect/PUBLIC_QUESTIONS.json'
PREFIX = 'R02:B:S01:'


# relation, independently authored explanation, citation turn suffixes.
PACKED = ('supports', 'Emily explicitly reports that everything on the checked gear list is packed.', ['T001'])
UNPACKED = ('contradicts', 'Emily explicitly says everything is packed, which refutes the same gear list being unpacked.', ['T001'])
DINNER = ('supports', 'James proposes the new Chinese place downtown for dinner after returning from camping.', ['T002', 'T004'])
HANDLING = ('supports', 'James explicitly says James is handling the restaurant reservation.', ['T004'])
BOOTH = ('supports', 'James reports the restaurant confirmed a booth for four at 9 p.m. on Saturday.', ['T006'])
CONFIRMED = ('supports', 'James explicitly confirms the reservation is already confirmed in this scene.', ['T008'])
PENDING = ('contradicts', 'James explicitly says the reservation is confirmed, not pending, at the scene outcome.', ['T006', 'T008'])
NUMBER = ('supports', 'James commits to bringing the reservation confirmation number on Saturday.', ['T008'])

ADDRESS = ('supports', 'Quinn states the office is at 42 Innovation Drive; the office referent is the internship start-up.', ['T001', 'T002'])
ADDRESS43 = ('contradicts', 'Quinn states 42 Innovation Drive for this office, which explicitly refutes the reported number 43.', ['T001', 'T002'])
POSTCODE = ('supports', 'Quinn gives TX5 8GH as the postcode for the same office area.', ['T003', 'T004'])
POSTCODEJ = ('contradicts', 'Quinn gives TX5 8GH for this area, which refutes the altered postcode TX5 8GJ.', ['T003', 'T004'])
PHONE = ('supports', 'In response to the reception-number request, Quinn gives the main line as 555-0199.', ['T005', 'T006'])
PHONE98 = ('contradicts', 'Quinn gives the same main reception line as 555-0199, which refutes the altered 555-0198.', ['T005', 'T006'])
CALL = ('supports', 'Morgan plans a call tomorrow to check office hours and whether a visit is possible.', ['T007'])
GIFT = ('supports', 'Morgan wants to bring a celebratory gift and considers a quick drop-off before the interview.', ['T001', 'T005', 'T013'])
OUT = ('supports', 'Morgan explicitly says Morgan is heading out to run errands now.', ['T013'])
IN = ('contradicts', "The current 'staying in' report conflicts with Morgan's 'heading out ... now'; later timing alone is not treated as a contradiction.", ['T013'])

# These are explicit per-question label decisions; no truth count or private
# selection record is used to author or fill them.
DECISIONS = {
    'cc-train-episode-103866-v90-fresh-parent-r1': {
        'A': PACKED, 'B': DINNER, 'C': HANDLING, 'D': BOOTH, 'E': CONFIRMED, 'F': NUMBER},
    'cc-train-episode-103866-v79-root103866_family-483672baf6f0': {
        'A': PENDING, 'B': NUMBER, 'C': DINNER, 'D': HANDLING, 'E': BOOTH, 'F': UNPACKED},
    'cc-train-episode-103866-v79-root103866_family-b4f141064efc': {
        'A': BOOTH, 'B': PENDING, 'C': DINNER, 'D': PACKED, 'E': HANDLING, 'F': NUMBER},
    'cc-train-episode-135466-v90-fresh-parent': {
        'A': ADDRESS, 'B': POSTCODE, 'C': PHONE, 'D': CALL, 'E': GIFT, 'F': OUT},
    'cc-train-episode-135466-v79-root135466_family-5e2c7e8dce9d': {
        'A': POSTCODEJ, 'B': IN, 'C': ADDRESS43, 'D': GIFT, 'E': PHONE98, 'F': CALL},
    'cc-train-episode-135466-v79-root135466_family-4f42d8018a92': {
        'A': IN, 'B': ADDRESS43, 'C': CALL, 'D': PHONE98, 'E': GIFT, 'F': POSTCODE},
    'cc-train-episode-135466-v79-root135466_family-19a73a5b8905': {
        'A': CALL, 'B': GIFT, 'C': POSTCODE, 'D': ADDRESS43, 'E': PHONE98, 'F': OUT},
    'cc-train-episode-135466-v79-root135466_family-09bfd40f4509': {
        'A': GIFT, 'B': POSTCODE, 'C': IN, 'D': ADDRESS, 'E': PHONE, 'F': CALL},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    public = read(PUBLIC)['questions']
    assert {q['candidate_id'] for q in public} == set(DECISIONS)
    helper_path = HERE / 'save_authored_final_review_v90.py'
    spec = importlib.util.spec_from_file_location('v90_save_final_multiselect_review', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reports = []
    for group, root, total_turns in [('a', 'cc-train-episode-103866', 127),
                                     ('b', 'cc-train-episode-135466', 204)]:
        subset = [q for q in public if q['root_id'] == root]
        assert subset
        source_path = Path(subset[0]['source_path'])
        source = read(source_path)
        assert len(source['turns']) == total_turns and all(q['source_sha256'] == sha(source_path) for q in subset)
        turns = {t['turn_ref']: t for t in source['turns']}
        questions = []
        for q in subset:
            authored = DECISIONS[q['candidate_id']]
            assert set(authored) == set(q['options'])
            options = {}
            for label, (relation, reason, suffixes) in authored.items():
                refs = [PREFIX + suffix for suffix in suffixes]
                assert all(ref in turns for ref in refs)
                options[label] = {'proposition': q['options'][label], 'relation': relation,
                                  'scope_valid': True, 'fact_distinct': True,
                                  'reason': reason, 'citation_turn_refs': refs}
            questions.append({'candidate_id': q['candidate_id'], 'stem_valid': True, 'options': options})
        authored_path = SHARD / 'multiselect/final_review' / ('group_' + group + '_AUTHORED_JUDGMENTS.json')
        record = {'version': 'v90_authored_whole_question_semantic_review_v1',
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'group': group, 'root_id': root, 'public_questions_sha256': sha(PUBLIC),
                  'source_sha256': sha(source_path), 'complete_source_turns_read': total_turns,
                  'reviewer': 'codex-v90-flow-audit-source-first-whole-question-review',
                  'previous_round_private_crosschecks_known': True,
                  'previous_round_exposure_note': 'Conservative historical exposure disclosure; this reviewer previously worked on historical inventories and export auditing. No fresh blind-parent claim is made.',
                  'parent_all_true_provenance_known': True, 'parent_gold_known_before_review': True,
                  'fresh_blind_parent_review': False, 'writer_raw_responses_read': False,
                  'current_solver_scores_read': False, 'new_child_private_artifacts_read': False,
                  'semantic_judgments_derived_from_round_records': False,
                  'whole_question_scope_note': 'The stem scopes all claims to R02:B:S01. In every whole question, the six items express distinct facts or distinct incompatible attributes. Related plans, confirmations and availability remain separate claims; each item was judged against the complete source.',
                  'questions': questions}
        authored_path.parent.mkdir(parents=True, exist_ok=True)
        with authored_path.open('x') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        helper.save(SHARD, group, authored_path)
        reports.append({'group': group, 'questions': len(questions), 'options': 6 * len(questions)})
    print(json.dumps({'status': 'SAVED', 'groups': reports, 'current_private_artifacts_read': False}))


if __name__ == '__main__':
    main()
