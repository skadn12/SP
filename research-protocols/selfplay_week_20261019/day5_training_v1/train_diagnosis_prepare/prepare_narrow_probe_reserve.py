"""Reserve the actual narrow-card quote/positions contrast; do not run it."""
from __future__ import annotations
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_prepare as b


def main():
    week = b.WEEK
    old = week / 'day2_frozen_evidence_probe_v2/run'
    manifest = b.read(old / 'manifest.json')
    frozen = b.read(old / 'FROZEN_EVIDENCE.json')['records']
    byroot = collections.defaultdict(list)
    for row in frozen:
        byroot[row['root_id']].append(row)
    roots = sorted(byroot, key=lambda root: b.hashkey('narrow-quote-position|' + root))[:3]
    chosen = [min(byroot[root], key=lambda row: b.hashkey('narrow-record|' + row['frozen_record_id'])) for root in roots]
    fids = {row['frozen_record_id'] for row in chosen}
    rows = [row for row in manifest['requests'] if row['frozen_record_id'] in fids and row['repeat_id'] in ('r00', 'r01')]
    assert len(rows) == 12 and len(roots) == 3
    allow = set(b.read(week / 'day1/TRAIN_ROOTS.json')['root_ids'])
    forbidden = set()
    for c in b.read(week / 'day4_schedule_v1/EVALUATION_LOCK.json')['cohorts']:
        forbidden.update(c['root_ids'])
    assert set(roots) <= allow and not set(roots) & forbidden
    public_bindings = []
    for row in rows:
        p = old / 'prompts' / (row['request_id'] + '.txt')
        assert b.sha(p) == row['prompt_sha256']
        row['prompt_path'] = str(p)
        public_bindings.append(b.binding(p))
    # Actual submitted decoder ordering is retained, as for the main direct packet.
    ids = {r['request_id'] for r in rows}
    arrays, keys, batch_bindings = {}, {}, []
    for p in sorted((old / 'batches').glob('*_SUBMITTED.json')):
        block = b.read(p)
        subset = [(rid, v) for rid, v in zip(block['request_ids'], block['sampling_params']) if rid in ids]
        if not subset:
            continue
        batch_bindings.append(b.binding(p))
        for rid, params in subset:
            if isinstance(params, str):
                match = re.search(r'choice=(\[[^\]]*\])', params)
                assert match
                values = ast.literal_eval(match.group(1))
            else:
                values = params['structured_outputs']['choice']
            assert len(values) == len(set(values)) == 720
            key = hashlib.sha256(json.dumps(values, separators=(',', ':')).encode()).hexdigest()
            arrays[key] = values
            keys[rid] = key
    assert set(keys) == ids
    cards = b.read(week / 'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json')['cards']
    card = next(x for x in cards if x['card_id'] == 'D2-ORDER-FIXED-CITATIONS')
    out = HERE / 'NARROW_PROBE_RESERVE_V2.json'
    b.save(out, {'schema_version': 'day5_narrow_quote_positions_probe_reserve_v2', 'status': 'FROZEN_RESERVE_NOT_RELEASED_NOT_EXECUTED',
        'created_at_utc': b.datetime.now(b.timezone.utc).isoformat(), 'hypothesis': card['operational_claim'],
        'models': ['A1', 'B1'], 'roots': roots, 'frozen_records': chosen, 'requests_per_model': rows,
        'requests_per_model_count': 12, 'requests_total_max': 24, 'new_solver_calls': 0,
        'contrast': {'control': 'freeze_quotes', 'treatment': 'explicit_positions'},
        'selection': 'Hash-sort the five old source-qualified frozen-citation roots; take first3. Per root, take lowest hash frozen_record_id; use original r00/r01. No S0/new-model scores or response answers read for selection.',
        'source_protocol': b.binding(week / 'day2_frozen_evidence_probe_v2/PROTOCOL.json'),
        'source_manifest': b.binding(old / 'manifest.json'), 'source_evidence': b.binding(old / 'FROZEN_EVIDENCE.json'),
        'source_semantic_review': b.binding(week / 'day2_frozen_evidence_probe_v2/SEMANTIC_REVIEW.json'),
        'source_operational_card_collection': b.binding(week / 'day2_feedback_readiness_v1/OPERATIONAL_CARDS.json'),
        'public_prompt_bindings': public_bindings, 'finite_choices': arrays, 'request_choice_keys': keys, 'S0_submitted_batch_bindings': batch_bindings,
        'S0_response_bindings': [b.binding(old / 'responses' / (r['request_id'] + '.json')) for r in rows],
        'release_requirements': ['Review actual B1 diagnostic records against source; identify concrete residual narrow ordering hypothesis.',
            'At least3 distinct Train roots and3 source-reviewed fact families with relevant observable failures; counts alone insufficient.',
            'Separate release freeze binds both actual A1/B1 endpoints, complete public request bytes, exact decoder order, source/semantic provenance and independent audit.',
            'Explicitly enforce shared Day5/Day6 max24 new probe requests; no general instruction reserve execution.'],
        'prospective_controlled_support_gate': {'all12_per_model_strict_valid_required': True,
            'primary': 'Root-equal citation-order pair-accuracy change, explicit_positions minus freeze_quotes; all fixed denominators.',
            'minimum_positive_roots': 3, 'root_equal_pair_accuracy_difference_positive': True, 'root_equal_exact_difference_nonnegative': True,
            'not_statistical_significance': True, 'not_internal_cause_proof': True,
            'new_gate_disclosure': 'Conservative prospective operational support gate for this24-request reserve; the old S0 study had no preregistered positive threshold. This new gate is not attributed to the old protocol.'},
        'same_model_required': 'Evaluate B1 support using actual B1 requests; S0 history is retained but not inherited. A1 contrast stays private.',
        'if_not_released_or_gate_fails': 'Record NOT_EXECUTED or NO_GO_WITH_RECORDS; retain B1 hypothesis/improving/recurred observations without formal reproduced/second-wave-targeted GO.',
        'limitations': ['Historical Train roots and old S0-selected frozen citations; not a new-family/new-root generalization test.',
            'Two repeats per root are repeated measurements, not extra independent families.', 'Explicit positions provide a scaffold; no inference about faithful internal COT or unique latent cause.',
            'Even a support gate PASS does not replace source-reviewed B1 failure admission or prove feedback training utility.'],
        'preparation_code': b.binding(Path(__file__))})
    b.save(HERE / 'PROBE_RESERVE_ADDENDUM_V2.json', {'status': 'OLD_GENERAL_INSTRUCTION_RESERVE_RETIRED_WITHOUT_EXECUTION',
        'created_at_utc': b.datetime.now(b.timezone.utc).isoformat(), 'old_reserve_preserved': b.binding(HERE / 'CONTROLLED_PROBE_RESERVE.json'),
        'new_narrow_reserve': b.binding(out), 'old_reserve_current_disposition': 'RETIRED_DO_NOT_RELEASE_OR_EXECUTE',
        'mutually_exclusive': True, 'combined_new_probe_cap_day5_day6': 24, 'actual_new_probe_calls': 0,
        'main_direct_and_brief_evidence_freeze_unchanged': b.binding(HERE / 'EXECUTION_ADDENDUM_V2.json'),
        'new_reserve_is_not_release_or_second_wave_GO': True})
    print(json.dumps({'status': 'OLD_RESERVE_RETIRED_NARROW_RESERVE_FROZEN_NOT_RELEASED', 'roots': roots, 'requests_per_model': 12, 'combined_cap': 24, 'new_calls': 0}))


if __name__ == '__main__':
    main()
