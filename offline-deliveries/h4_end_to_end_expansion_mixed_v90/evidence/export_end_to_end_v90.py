"""Independent v90 input/raw-result audit, typed exports, and exact-fileset seal.

Run inputs after prepare-evaluation and before either live Solver. Run export
after both live stages/report. Every write is append-only; seal writes COMPLETE
last. Neither this auditor nor its imports call production score/selection code.
"""
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True
VERSION = 'v90_end_to_end_independent_export_v1'
SEED = 2026100800
MODELS = {'trained': 'M_clean_policy', 'base': 'M0'}
LABELS = {'multi_select': tuple('ABCDEF'), 'ordering': tuple('I'+str(i) for i in range(1, 7))}
CONSISTENCY = {'time', 'location', 'actor_identity', 'plan_vs_completion', 'cross_scene_continuity', 'semantic_repetition'}
DEFAULT_HOME = Path(__file__).resolve().parents[1]
TOKENIZER = Path('/root/autodl-tmp/Qwen3-4B')
BASE_AUDIT = Path('/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_multiselect_partial_admission_v1/followup_model_identity.json')
BASE_AUDIT_SHA = '59e23710e2373a00aef3b7d50d117d4089eea47cd17fd16a00eebcc83fba0035'
TRAINED_AUDIT = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_trained_solver_train_reeval_v1_live_1/model_identity_report.json')
TRAINED_AUDIT_SHA = 'add007d4e4f937f2ea1b7e124ccac3c53e5eda5b8bbed1beab3dd4b0855cadd0'
MODEL_PATHS = {'M0': str(TOKENIZER), 'M_clean_policy': '/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_common_dev_eval_v1_live_3/clean_merged_model'}


def need(value, message):
    if not value:
        raise ValueError('v90_audit:' + message)


def now():
    return datetime.now(timezone.utc).isoformat()


def dt(value):
    result = datetime.fromisoformat(value)
    need(result.utcoffset() is not None, 'timezone_required')
    return result


def read(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            need(key not in out, 'duplicate_json_key:' + key); out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: need(False, 'nonfinite_json:' + value))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        if isinstance(value, str):
            stream.write(value)
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')


def files_under(home):
    paths = list(Path(home).rglob('*'))
    need(not any(p.is_symlink() for p in paths), 'symlink_in_inventory:' + str(home))
    return {str(p.resolve()) for p in paths if p.is_file()}


def preservation(home):
    snapshot = read(home / 'PRESERVATION_BEFORE.json'); expected = snapshot['files']
    roots = [Path(p).resolve() for p in snapshot['directories']]
    need(roots and expected and all(p != home and home not in p.parents for p in roots), 'historical_roots_required')
    actual = set().union(*(files_under(p) for p in roots))
    need(actual == set(expected), 'historical_inventory_changed:' + json.dumps({
        'added': sorted(actual-set(expected)), 'missing': sorted(set(expected)-actual)}))
    for path, digest in expected.items():
        need(sha(path) == digest, 'historical_bytes_changed:' + path)
    need(snapshot.get('protected_count', len(expected)) == len(expected), 'historical_count')
    return {'status': 'PASS', 'files': len(expected), 'exact_inventory': True,
            'all_sha256_recomputed': True, 'snapshot': bind(home / 'PRESERVATION_BEFORE.json')}


class Audit:
    def __init__(self):
        self.files = {}; self.checks = 0; self.weights = {}

    def check(self, value, message):
        self.checks += 1; need(value, message)

    def bind(self, path):
        row = bind(path); self.files[row['path']] = row; return row

    def read(self, path):
        self.bind(path); return read(path)

    def binding(self, row):
        self.check(isinstance(row, dict) and set(row) == {'path', 'sha256'}
                   and Path(row['path']).is_absolute(), 'binding_schema')
        self.check(self.bind(row['path']) == row, 'binding_changed:' + row['path'])
        # Receipts also bind Python, prompts and other opaque artifacts. Their
        # exact bytes are verified above; only JSON records require decoding.
        return self.read(row['path']) if Path(row['path']).suffix == '.json' else None

    def receipts(self, path):
        value = self.read(path)
        for row in value.get('files', []):
            self.binding(row)
        for row in value.get('source_files', []):
            self.binding(row)
        return value


def exact_citation(a, cite, turns, speaker=True):
    turn = turns.get(cite.get('turn_ref'))
    a.check(turn is not None and isinstance(cite.get('quote'), str) and bool(cite['quote'])
            and cite['quote'] in turn['text'], 'exact_source_citation')
    if speaker:
        a.check(cite.get('speaker') == turn['speaker'], 'citation_speaker')


def source_review(a, source, review, source_binding):
    turns = source['turns']; lookup = {t['turn_ref']: t for t in turns}; groups = OrderedDict()
    a.check(len(lookup) == len(turns) and bool(turns), 'source_unique_turns')
    a.check(review['schema_version'] == 'dialogue_consistency_v1' and review['root_id'] == source['root_id']
            and review['source'] == source_binding and review['review_scope'] == 'source_consistency_only', 'source_review_identity')
    a.check(review['full_source_read'] is True and review['current_question_gold_read'] is False
            and review['current_solver_scores_read'] is False, 'source_review_exposure')
    for key in ('historical_question_gold_known', 'fresh_blind_review'):
        a.check(type(review[key]) is bool, 'source_review_exposure_bool:' + key)
    a.check(not (review['fresh_blind_review'] and review['historical_question_gold_known']), 'false_fresh_blind_claim')
    a.check(review['turn_refs_read'] == list(lookup) and set(review['checks_completed']) == CONSISTENCY
            and all(v is True for v in review['checks_completed'].values()), 'all_turns_six_checks')
    for turn in turns:
        a.check(turn['origin'] in ('source_grounded', 'generated_extension', 'provisional_candidate')
                and all(isinstance(turn[k], str) and turn[k].strip() for k in ('turn_ref', 'speaker', 'text')),
                'source_turn_schema')
        unit = turn['session_ref'] if turn['origin'] == 'source_grounded' else turn['scene_ref']
        groups.setdefault(unit, []).append(turn)
        a.check(isinstance(unit, str) and bool(unit.strip())
                and len({t['origin'] for t in groups[unit]}) == 1, 'source_unit_origin')
    a.check([t['turn_ref'] for rows in groups.values() for t in rows] == list(lookup), 'contiguous_source_units')
    a.check([u['unit_id'] for u in review['units']] == list(groups), 'source_review_unit_coverage')
    findings = review['findings']; fids = [f['finding_id'] for f in findings]
    a.check(len(fids) == len(set(fids)), 'unique_source_findings')
    unit_by_ref = {t['turn_ref']: uid for uid, rows in groups.items() for t in rows}
    for finding in findings:
        a.check(finding['severity'] in ('BLOCKING', 'WARNING') and finding['kind'] in CONSISTENCY | {'naturalness'}
                and bool(finding['reason'].strip()) and bool(finding['proposed_action'].strip()), 'source_finding_schema')
        a.check(finding['unit_ids'] and len(set(finding['unit_ids'])) == len(finding['unit_ids'])
                and set(finding['unit_ids']) <= set(groups) and finding['evidence'], 'finding_units_evidence')
        for cite in finding['evidence']:
            a.check(set(cite) == {'turn_ref', 'quote'}, 'source_finding_citation_schema')
            exact_citation(a, cite, lookup, False)
        a.check({unit_by_ref[cite['turn_ref']] for cite in finding['evidence']} == set(finding['unit_ids']),
                'source_finding_evidence_unit_coverage')
    for unit in review['units']:
        rows = groups[unit['unit_id']]; relevant = [f for f in findings if unit['unit_id'] in f['unit_ids']]
        decision = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in relevant) else 'WARN' if relevant else 'PASS'
        a.check(unit['origin'] == rows[0]['origin'] and bool(unit['reason'].strip())
                and unit['turn_refs'] == [t['turn_ref'] for t in rows] and unit['decision'] == decision
                and unit['finding_ids'] == [f['finding_id'] for f in relevant], 'source_unit_decision')
    expected = 'HOLD' if any(f['severity'] == 'BLOCKING' for f in findings) else 'READY'
    a.check(review['decision'] == expected, 'source_decision_replayed')
    return expected


def stages(a, home, index_path):
    plan = a.read(home / 'PILOT_PLAN.json'); index = a.read(index_path); rows = index['roots']
    a.check(plan['training_allowed'] is False and plan['formal_training_admission'] is False, 'plan_no_training')
    a.check([r['root_id'] for r in rows] == plan['candidate_roots'] and len(rows) == plan['candidate_root_budget'], 'stage_root_coverage')
    expansion = a.binding(index['expansion_audit'])
    a.check(expansion['status'] == 'PASS' and expansion['failures'] == []
            and expansion['roots'] == plan['candidate_roots']
            and 0 <= expansion['http_attempts'] <= plan['expansion']['maximum_http_attempts']
            and expansion['question_calls'] == expansion['solver_calls'] == 0
            and expansion['training_started'] is False, 'expansion_audit_budget_and_identity')
    expansion_run = Path(expansion['run']).resolve()
    a.check(expansion_run.is_relative_to(home/'expansion_engine/runs'), 'expansion_run_local')
    for path in sorted(files_under(expansion_run)): a.bind(path)
    result = {}
    keys = ('input_source', 'expansion_result', 'expanded_source', 'expanded_source_provenance',
            'source_review', 'public_parent', 'parent_judgments')
    for row in rows:
        root = row['root_id']; actual = {}
        for key in keys:
            if row.get(key) is not None:
                actual[key] = a.binding(row[key])
        fresh_refs = set(); source_decision = None
        if 'expanded_source' in actual:
            a.check('input_source' in actual and 'expanded_source_provenance' in actual, 'expanded_source_requires_provenance')
            old, new = actual['input_source'], actual['expanded_source']; prefix = old['turns']
            a.check(old['root_id'] == new['root_id'] == root and len(new['turns']) > len(prefix)
                    and new['turns'][:len(prefix)] == prefix, 'unchanged_complete_history_and_actual_extension:' + root)
            refs = [t['turn_ref'] for t in new['turns']]
            a.check(len(refs) == len(set(refs)), 'expanded_source_unique_refs')
            fresh_refs = {t['turn_ref'] for t in new['turns'][len(prefix):]}
            provenance = actual['expanded_source_provenance']
            a.check(provenance['root_id'] == root and provenance['output_source'] == row['expanded_source'], 'expansion_source_provenance')
            a.check(provenance['version'] == 'v90_parent_expansion_binding_wrapper_v1'
                    and provenance['input_source'] == row['input_source'] and provenance['original_provenance_modified'] is False,
                    'expansion_wrapper_identity')
            original_provenance = a.binding(provenance['original_expansion_provenance'])
            accepted = a.binding(provenance['original_accepted_output'])
            a.check(original_provenance['root_id'] == root and original_provenance['source'] == row['expanded_source']
                    and original_provenance['accepted_output'] == provenance['original_accepted_output']
                    and accepted == new, 'materialized_source_exact_accepted_expansion')
            if 'source_review' in actual:
                source_decision = source_review(a, new, actual['source_review'], row['expanded_source'])
        if 'parent_judgments' in actual:
            a.check(source_decision == 'READY' and 'public_parent' in actual, 'parent_requires_ready_expanded_source')
            public, review = actual['public_parent'], actual['parent_judgments']
            a.check('gold' not in public and 'option_evidence' not in public and set(public['options']) == set('ABCDEF'), 'public_parent_no_private')
            a.check(review['version'] == 'v90_fresh_parent_source_judgments_v1'
                    and review['public_question'] == public and review['public_question_sha256'] == row['public_parent']['sha256']
                    and review['source_sha256'] == row['expanded_source']['sha256'], 'fresh_parent_review_binding')
            a.check(review['full_source_read'] is True and review['turn_refs_read'] == [t['turn_ref'] for t in actual['expanded_source']['turns']]
                    and review['full_source_turn_count'] == len(actual['expanded_source']['turns'])
                    and review['stem_valid'] is True and review['content_status'] == 'CORE_USABLE', 'fresh_parent_complete_source')
            for key in ('fresh_blind_review', 'public_parent_authored_by_reviewer',
                        'proposed_all_supported_intent_known', 'historical_question_gold_known'):
                a.check(type(review[key]) is bool, 'fresh_parent_exposure_bool:' + key)
            a.check(not review['fresh_blind_review'] or (not review['public_parent_authored_by_reviewer']
                    and not review['historical_question_gold_known']), 'fresh_parent_no_false_blind_claim')
            for key in ('known_old_parent_gold', 'current_parent_private_gold_read_before_judgments_saved',
                        'current_parent_private_evidence_read_before_judgments_saved', 'current_solver_results_read'):
                a.check(review[key] is False, 'fresh_parent_disclosure:' + key)
            lookup = {t['turn_ref']: t for t in actual['expanded_source']['turns']}
            a.check([j['option'] for j in review['judgments']] == list('ABCDEF'), 'parent_all_six_reviewed')
            for item in review['judgments']:
                a.check(item['proposition'] == public['options'][item['option']] and item['relation'] == 'supports'
                        and item['source_judgment'] == 'supported'
                        and item['scope_valid'] is True and item['fact_distinct'] is True and item['entailed'] is True
                        and bool(item['reason'].strip()) and item['required_evidence'], 'parent_supported_distinct')
                for cite in item['required_evidence']:
                    exact_citation(a, cite, lookup)
                    a.check(cite['turn_ref'] in fresh_refs, 'parent_fact_must_cite_new_scene')
        result[root] = {'row': row, 'actual': actual, 'fresh_refs': fresh_refs, 'source_decision': source_decision,
                        'new_turns': len(fresh_refs), 'input_turns': len(actual['input_source']['turns']) if 'input_source' in actual else None}
    return result


def source_token_counts(a, source_context):
    """Count complete, historical and new turn bodies without prompt wrappers."""
    from transformers import AutoTokenizer
    tokenizer_paths = [TOKENIZER/name for name in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')]
    bindings = [a.bind(p) for p in tokenizer_paths]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    rows = []
    for root, ctx in source_context.items():
        old = ctx['actual'].get('input_source'); expanded = ctx['actual'].get('expanded_source')
        if old is None: continue
        complete_turns = expanded['turns'] if expanded is not None else old['turns']
        prefix_count = len(old['turns'])
        old_tokens = sum(len(tokenizer.encode(t['text'], add_special_tokens=False)) for t in complete_turns[:prefix_count])
        new_tokens = sum(len(tokenizer.encode(t['text'], add_special_tokens=False)) for t in complete_turns[prefix_count:])
        rows.append({'root_id': root, 'accepted_new_scene': expanded is not None,
            'historical_turns': prefix_count, 'new_turns': len(complete_turns)-prefix_count,
            'historical_body_tokens': old_tokens, 'new_body_tokens': new_tokens,
            'complete_body_tokens': old_tokens+new_tokens})
    return {'tokenizer_path': str(TOKENIZER), 'tokenizer_files': bindings,
        'counting_definition': 'Sum of len(Qwen3-4B tokenizer.encode(turn.text, add_special_tokens=False)) separately for every turn; excludes speaker/ref labels, chat/prompt wrappers and question text.',
        'qa_stage_new_expansion_tokens': 0,
        'qa_stage_field_scope': 'Frozen v79 QA stage creates questions from upstream expanded sources and performs no dialogue expansion.',
        'accepted_new_scene_roots': sum(row['accepted_new_scene'] for row in rows),
        'new_turns': sum(row['new_turns'] for row in rows),
        'new_body_tokens': sum(row['new_body_tokens'] for row in rows),
        'roots': rows, 'new_model_or_network_calls': 0}


def question_key(q):
    kind = q['question_type']; need(kind in LABELS, 'unknown_type')
    field = 'options' if kind == 'multi_select' else 'items'
    need(tuple(q[field]) == LABELS[kind], 'six_labels_in_native_order')
    texts = tuple(sorted(' '.join(v.split()).casefold() for v in q[field].values()))
    need(len(set(texts)) == 6 and all(texts), 'distinct_statements')
    need(len(q['gold']) == len(set(q['gold'])) and set(q['gold']) <= set(LABELS[kind]), 'gold_labels')
    need(2 <= len(q['gold']) <= 6 if kind == 'multi_select' else len(q['gold']) == 6, 'gold_cardinality')
    return q['root_id'], kind, texts


def control(q):
    return q['question_type'] == 'multi_select' and len(q['gold']) == 6


def content_ids(q):
    field = 'options' if q['question_type'] == 'multi_select' else 'items'
    return {label: 'content_'+hashlib.sha256(text.strip().encode()).hexdigest()[:16] for label, text in q[field].items()}


def variants(q):
    """Independent replay of the native Gold-blind seeded permutation contract."""
    kind = q['question_type']; field = 'options' if kind == 'multi_select' else 'items'; ids = content_ids(q)
    def seed_for(number):
        return int.from_bytes(hashlib.sha256(f'{SEED}|{q["candidate_id"]}|{number}'.encode()).digest()[:8], 'big')
    canonical = list(ids.values()); random.Random(seed_for(-1)).shuffle(canonical)
    mapping_text = {ids[k]: v for k, v in q[field].items()}; seen = set(); out = []
    for number in range(6 if kind == 'multi_select' else 12):
        rng = random.Random(seed_for(number))
        while True:
            order = canonical[:]; rng.shuffle(order)
            if tuple(order) not in seen: break
        seen.add(tuple(order)); mapping = dict(zip(LABELS[kind], order))
        out.append({'variant_id': f'perm_{number:02d}', 'canonical_order_content_ids': canonical,
                    'display_order_content_ids': order, field: {k: mapping_text[v] for k, v in mapping.items()},
                    'display_to_content': mapping, 'content_to_text': mapping_text})
    return out


def raw_score(q, request, response):
    mapping = request['variant']['display_to_content']; ids = content_ids(q)
    need(set(mapping) == set(ids) and set(mapping.values()) == set(ids.values()), 'score_content_mapping')
    raw = response['raw_output']; match = re.fullmatch(r'\s*<box>\s*(.*?)\s*</box>\s*', raw, re.S) if isinstance(raw, str) else None
    need(match is not None, 'invalid_box')
    labels = [x.strip().upper() for x in match.group(1).split(',')]
    need(labels and len(labels) == len(set(labels)) and set(labels) <= set(mapping), 'invalid_labels')
    prediction = [mapping[k] for k in labels]; gold = [ids[k] for k in q['gold']]
    if q['question_type'] == 'multi_select':
        tp, fp, fn = len(set(prediction)&set(gold)), len(set(prediction)-set(gold)), len(set(gold)-set(prediction))
        value = Fraction(tp, len(gold)) - (Fraction(fp, 6-len(gold)) if len(gold)<6 else 0)
        exact = set(prediction) == set(gold)
    else:
        need(len(prediction) == 6 and set(prediction) == set(gold), 'incomplete_order')
        rank = {v: i for i, v in enumerate(prediction)}
        fp = sum(rank[x] > rank[y] for i, x in enumerate(gold) for y in gold[i+1:]); tp, fn = 15-fp, 0
        value, exact = Fraction(tp-fp, 15), prediction == gold
    need(response['score']['partial_score_fraction'] == str(value) and response['score']['exact_match'] is exact
         and all(response['score'][k] == v for k, v in {'tp': tp, 'fp': fp, 'fn': fn}.items()), 'saved_score_disagrees')
    return value, exact


def span(a, anchor, turns, positions):
    exact_citation(a, anchor, turns, 'speaker' in anchor)
    text = turns[anchor['turn_ref']]['text']; quote = anchor['quote']
    a.check(text.count(quote) == 1, 'anchor_quote_must_be_unambiguous')
    start = text.index(quote)
    return (positions[anchor['turn_ref']], start, start+len(quote))


def matching_refs(q, source):
    """Resolve the public terminal scene restriction without private evidence."""
    scene_ids = list(dict.fromkeys(t['turn_ref'].rsplit(':', 1)[0] for t in source['turns']))
    found = re.search(r'\bAPPEAR IN THE SUPPLIED DIALOGUE within the (?P<scenes>.+?) scenes?\.?\s*$', q['question'], re.I)
    selected = []
    if found:
        entries = re.split(r'\s*,\s*(?:and\s+)?|\s+and\s+', found.group('scenes'))
        if entries and all(entry in scene_ids for entry in entries):
            selected = [s for s in scene_ids if s in entries]
    return {t['turn_ref'] for t in source['turns'] if not selected or t['turn_ref'].rsplit(':', 1)[0] in selected}


def content_review(a, shard, questions, source_context):
    multi_public = a.read(shard/'multiselect/PUBLIC_QUESTIONS.json')['questions']
    groups = a.read(shard/'multiselect/CONFIG.json')['final_review_groups']; rows = {}; review_bindings = {}
    public_by_id = {q['candidate_id']: q for q in multi_public}
    for group in groups:
        path = shard/'multiselect/final_review'/f'group_{group}_SOURCE_JUDGMENTS.json'; record = a.read(path)
        a.check(record['review_schema_version'] == 'source_judgment_v1' and record['current_solver_scores_read'] is False
                and record['solver_scores_used'] is False and record['child_gold_read_before_save'] is False
                and record['current_pool_read_before_save'] is False, 'multiselect_source_first')
        a.check(record['public_questions'] == a.bind(shard/'multiselect/PUBLIC_QUESTIONS.json'), 'multiselect_public_binding')
        for item in record['questions']:
            qid = item['candidate_id']; a.check(qid not in rows, 'duplicate_whole_question_review'); rows[qid] = item; review_bindings[qid] = a.bind(path)
    a.check(set(rows) == set(public_by_id), 'whole_multiselect_review_coverage')
    ordering_public = a.read(shard/'ordering/PUBLIC_QUESTIONS.json')['questions']
    ordering_judgment = a.read(shard/'ordering/SOURCE_JUDGMENTS.json')
    a.check(ordering_judgment['review_schema_version'] == 'source_judgment_v1'
            and ordering_judgment['current_solver_scores_read'] is False and ordering_judgment['solver_scores_used'] is False
            and ordering_judgment['child_gold_read_before_save'] is False and ordering_judgment['new_variant_private_artifacts_read_before_save'] is False,
            'ordering_source_first')
    a.check(ordering_judgment['public_questions'] == a.bind(shard/'ordering/PUBLIC_QUESTIONS.json'), 'ordering_public_binding')
    ordering_rows = {q['candidate_id']: q for q in ordering_judgment['questions']}
    a.check(set(ordering_rows) == {q['candidate_id'] for q in ordering_public}, 'whole_ordering_review_coverage')
    ordering_by_id = {q['candidate_id']: q for q in ordering_public}
    for q in questions:
        question_key(q); root = q['root_id']; ctx = source_context[root]
        source = a.read(q['source_path']); a.check(a.bind(q['source_path'])['sha256'] == q['source_sha256'], 'question_source_hash')
        a.check(source == ctx['actual']['expanded_source'] and ctx['source_decision'] == 'READY', 'question_complete_ready_source')
        turns = {t['turn_ref']: t for t in source['turns']}; positions = {t['turn_ref']: i for i, t in enumerate(source['turns'])}
        fresh = ctx['fresh_refs']; kind = q['question_type']; field = 'options' if kind == 'multi_select' else 'items'
        pub = public_by_id[q['candidate_id']] if kind == 'multi_select' else ordering_by_id[q['candidate_id']]
        a.check(all(q[k] == pub[k] for k in ('candidate_id', 'root_id', 'question_type', 'question', field, 'source_path', 'source_sha256')), 'public_private_identity')
        if kind == 'multi_select':
            judged = rows[q['candidate_id']]
            a.check(judged['stem_valid'] is True and judged['complete_source_turns_read'] == len(source['turns']), 'multiselect_stem_full_source')
            a.check(set(judged['options']) == set(q['options']), 'multiselect_all_options')
            supported = []
            for label, item in judged['options'].items():
                a.check(item['proposition'] == q['options'][label] and item['relation'] in ('supports', 'contradicts')
                        and item['scope_valid'] is True and item['fact_distinct'] is True and item['citations'], 'accepted_option_definite_relation')
                if item['relation'] == 'supports': supported.append(label)
                for cite in item['citations']:
                    exact_citation(a, cite, turns)
                a.check(any(cite['turn_ref'] in fresh for cite in item['citations']), 'accepted_option_cites_new_scene')
                evidence = q['option_evidence'][label]; exact_citation(a, evidence, turns)
                a.check(evidence['relation'] == item['relation'] and evidence['turn_ref'] in fresh, 'private_evidence_matches_saved_relation')
            a.check(set(supported) == set(q['gold']), 'whole_question_gold_matches_saved_supports')
        else:
            judged = ordering_rows[q['candidate_id']]
            a.check(judged['complete_source_turns_read'] == len(source['turns'])
                    and all(judged[k] is True for k in ('coherent', 'unique_order', 'neutral_stem_ok')), 'ordering_full_source_and_unique_order')
            anchors, alternatives = {}, {}; matching = matching_refs(pub, source)
            a.check(set(judged['items']) == set(q['items']), 'ordering_all_six_judgments')
            for label, item in judged['items'].items():
                a.check(item['proposition'] == q['items'][label]
                        and all(item[k] is True for k in ('supported', 'actor_ok', 'modality_ok', 'state_ok', 'neutral_stem_ok')), 'ordering_item_semantics')
                primary = item['order_anchor']; anchors[label] = span(a, primary, turns, positions)
                a.check(primary['turn_ref'] in fresh and primary['turn_ref'] in matching
                        and item['support_refs'] and set(item['support_refs']) <= set(turns), 'ordering_new_anchor_and_support')
                actual = q['order_anchor_spans'][label]
                a.check((actual['turn_index'], actual['char_start'], actual['char_end']) == anchors[label]
                        and actual['turn_ref'] == primary['turn_ref'] and actual['quote'] == primary['quote'], 'final_anchor_exact')
                alternatives[label] = [anchors[label]]
                for entry in item['alternative_matching_anchors']:
                    resolved = span(a, entry, turns, positions)
                    if entry['turn_ref'] in matching and resolved not in alternatives[label]: alternatives[label].append(resolved)
                exact_citation(a, q['item_evidence'][label], turns)
            order = sorted(anchors, key=lambda k: anchors[k]); a.check(order == q['gold'] == q['original_gold'], 'ordering_gold_source_order')
            for left, right in zip(order, order[1:]):
                a.check(all(x[0] < y[0] or (x[0] == y[0] and x[2] <= y[1]) for x in alternatives[left] for y in alternatives[right]), 'all_alternative_anchor_orders')
            a.check(all(v == order for v in judged['alternative_orders']), 'declared_alternative_orders')
            program = a.binding(q['program_gold_plan']); frozen_q = next(v for v in program['questions'] if v['candidate_id'] == q['candidate_id'])
            a.check(frozen_q['gold'] == q['gold'] and program['gold_supplied_to_writer'] is False, 'program_gold_unchanged')
            a.check(dt(program['created_at_utc']) <= dt(a.read(Path(q['source_path']).parent/'GENERATION.STARTED.json')['started_at_utc']), 'gold_before_writer')


def manifest_pair(a, shard, questions):
    manifests = {}; expected_requests = []
    for q in questions:
        for variant in variants(q):
            for slot in range(3 if q['question_type'] == 'multi_select' else 1):
                expected_requests.append({'request_id': q['candidate_id']+'__'+variant['variant_id']+f'__full_context__s{slot:02d}',
                    'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
                    'variant': variant, 'mode': 'full_context', 'slot': slot, 'seed': SEED+len(expected_requests)})
    if not questions:
        for side, model in MODELS.items():
            skip = a.read(shard/'solver'/side/'SKIPPED.json')
            a.check(skip['solver_model'] == model and skip['candidate_count'] == skip['actual_call_count'] == 0, 'empty_solver_skip')
        return manifests
    for side, model in MODELS.items():
        run = shard/'solver'/side; m = a.read(run/'manifest.json'); manifests[side] = m
        a.check(m['status'] == 'PREFLIGHT_PASS' and m['solver_model'] == model and m['training_started'] is False
                and m['training_promotion'] is False and m['execution'] == 'full_only_no_ground_v1'
                and m['model'] == MODEL_PATHS[model] and m['tokenizer'] == str(TOKENIZER)
                and m['policy_version'] == 'v15_full_only_random_mcq6x3_order12x1_v1', 'solver_manifest_contract')
        a.check(m['random_display'] == {'version':'independent_seeded_permutation_v2', 'seed':SEED,
                'gold_blind':True, 'cyclic_rebinding':False, 'shape_filter':False}, 'manifest_gold_blind_permutation_contract')
        a.check(m['pool_sha256'] == a.bind(shard/'EVALUATION_POOL.json')['sha256']
                and Path(m['pool_path']).resolve() == shard/'EVALUATION_POOL.json', 'solver_pool_binding')
        a.check(m['candidate_count'] == len(questions) and m['request_count'] == len(expected_requests) == len(m['requests']), 'exact_request_count')
        for key, value in {'temperature': .8, 'top_p': .9, 'max_tokens': 32, 'enable_thinking': False,
                           'structured_output_version': 'finite_choice_set_v1'}.items():
            a.check(m['decoder_contract'][key] == value, 'decoder:' + key)
        for key, digest in m['code_sha256'].items():
            a.check(a.bind(key)['sha256'] == digest, 'native_code_hash')
        for key, digest in m['input_sha256'].items():
            path = shard/'EVALUATION_POOL.json' if key == 'pool' else run/key
            a.check(a.bind(path)['sha256'] == digest, 'native_input_hash')
        a.check(a.read(run/'gold.json') == {q['candidate_id']: [content_ids(q)[k] for k in q['gold']] for q in questions}, 'gold_content_ids')
        a.check(a.read(run/'questions.json')['candidates'] == [{k: v for k, v in q.items() if k not in ('gold', 'gold_content_ids')} for q in questions], 'native_public_questions')
        for actual, expected in zip(m['requests'], expected_requests):
            a.check(set(actual) == set(expected)|{'prompt_sha256', 'prompt_tokens'} and all(actual[k] == v for k, v in expected.items()), 'gold_blind_request_plan')
            a.check(type(actual['prompt_tokens']) is int and 0 < actual['prompt_tokens'] <= 16352, 'local_context_capacity')
            a.check(a.bind(run/'prompts'/(actual['request_id']+'.txt'))['sha256'] == actual['prompt_sha256'], 'prompt_hash')
        identity = a.read(run/'solver_identity.json'); a.check(identity == m['solver_identity'], 'solver_identity_binding')
        a.check(identity['model_identifier'] == model and identity['model_path'] == m['model'], 'model_identity')
        expected_audit = TRAINED_AUDIT if model == 'M_clean_policy' else BASE_AUDIT
        expected_digest = TRAINED_AUDIT_SHA if model == 'M_clean_policy' else BASE_AUDIT_SHA
        a.check(Path(identity['audit_path']) == expected_audit and identity['audit_sha256'] == expected_digest,
                'fixed_model_audit_identity')
        a.check(a.bind(identity['audit_path'])['sha256'] == identity['audit_sha256'], 'model_audit_hash')
        a.check(a.bind(BASE_AUDIT)['sha256'] == BASE_AUDIT_SHA
                and identity['tokenizer_audit_path'] == str(BASE_AUDIT)
                and identity['tokenizer_audit_sha256'] == BASE_AUDIT_SHA, 'fixed_tokenizer_audit')
        base_rows = a.read(BASE_AUDIT)['files']
        if model == 'M0': expected_weights = base_rows
        else:
            trained_rows = a.read(TRAINED_AUDIT)['models'][model]['merged_model_identity']['files']
            expected_weights = [{'path':str(Path(MODEL_PATHS[model])/r['name']), 'name':r['name'],
                                 'bytes':r['size_bytes'], 'sha256':r['sha256']} for r in trained_rows]
            expected_weights += [r for r in base_rows if r['name'] in {'tokenizer.json','tokenizer_config.json','merges.txt','vocab.json'}]
        a.check(identity['checkpoint_files'] == expected_weights, 'complete_fixed_audit_checkpoint_set')
        for item in identity['checkpoint_files']:
            path = Path(item['path']); a.check(path.is_file() and path.stat().st_size == item['bytes'], 'weight_presence_and_size')
            if str(path) not in a.weights:
                digest = sha(path); a.check(digest == item['sha256'], 'weight_sha256'); a.weights[str(path)] = digest
            else:
                a.check(a.weights[str(path)] == item['sha256'], 'shared_weight_sha256')
    allowed = {'prepared_at_utc', 'model', 'solver_model', 'solver_identity'}
    a.check({k:v for k,v in manifests['trained'].items() if k not in allowed} ==
            {k:v for k,v in manifests['base'].items() if k not in allowed}, 'identical_paired_inputs')
    return manifests


def shard_inputs(a, shard, source_context):
    config = a.read(shard/'CONFIG.json'); installation = a.receipts(shard/'SHARD_ENGINE_INSTALLATION.json')
    a.check(installation['training_allowed'] is False and installation['shard_home'] == str(shard), 'installed_shard')
    a.check(installation['version'] == 'v90_shard_engine_installation_v1'
            and {Path(b['path']).name: b['sha256'] for b in installation['source_files']} ==
            {Path(b['path']).name: b['sha256'] for b in installation['files']}, 'installed_code_exact_copy')
    for name in ('SHARD_INPUTS.COMPLETE.json', 'SOURCE_ADMISSION.COMPLETE.json', 'BATCH_PREPARATION.COMPLETE.json',
                 'BATCH_GENERATION.COMPLETE.json', 'BATCH_REVIEW.COMPLETE.json', 'EVALUATION_PREPARATION.COMPLETE.json'):
        a.receipts(shard/name)
    freeze = a.read(shard/'BATCH_REVIEW.COMPLETE.json')
    a.check(freeze['before_any_solver'] is True and freeze['solver_scores_used'] is False and freeze['training_allowed'] is False, 'content_frozen_before_solver')
    slots = config['multiselect_config']['slots']; roots = [s['root_id'] for s in slots]
    a.check(2 <= len(roots) <= 4 and len(set(roots)) == len(roots)
            and roots == [s['root_id'] for s in config['ordering_slots']], 'same_roots_joint_generation')
    a.check(config['max_generation_http'] == 3*len(roots) and config['max_solver_calls_per_model'] == 114*len(roots)
            and config['multiselect_config']['max_supplemental_requests_per_family'] == 1, 'joint_generation_budget')
    attempts = list((shard/'multiselect/assets').glob('*/rounds/*/requests/*/HTTP_ATTEMPT.json'))
    ordering_attempts = list((shard/'ordering').glob('*/requests/*/HTTP_ATTEMPT.json'))
    a.check(len(attempts) <= 2*len(roots) and len(ordering_attempts) <= len(roots)
            and len(attempts)+len(ordering_attempts) <= config['max_generation_http'], 'actual_question_http_budget')
    for path in attempts+ordering_attempts:
        receipt = a.read(path)
        a.check(receipt['http_attempt_number'] == 1 and receipt['transport_retries'] == 0
                and receipt['actual_payload_equals_frozen'] is True, 'qa_no_retries_or_payload_mutation')
    for slot in slots:
        ctx = source_context[slot['root_id']]; a.check(ctx['source_decision'] == 'READY' and 'parent_judgments' in ctx['actual'], 'production_root_ready_with_parent')
        provenance = a.read(slot['parent_provenance_path']); public = a.binding(provenance['public_parent']); review = a.binding(provenance['source_judgments'])
        a.check(public == ctx['actual']['public_parent'] and review == ctx['actual']['parent_judgments'], 'parent_stage_binding')
        a.check(provenance['historical_parent_reused'] is False and provenance['current_private_parent_created_after_saved_source_judgments'] is True
                and provenance['current_solver_scores_read'] is False and provenance['training_allowed'] is False
                and dt(review['created_at_utc']) <= dt(provenance['created_at_utc']), 'parent_before_private_construction')
        parents = a.read(slot['parent_pool_path'])['questions']; parent = next(q for q in parents if q['candidate_id'] == slot['parent_candidate_id'])
        a.check(parent['gold'] == list('ABCDEF') and parent['options'] == public['options']
                and parent['source_sha256'] == public['source_sha256'] and parent['training_allowed'] is False, 'private_six_true_parent')
        for item in review['judgments']:
            expected = {**item['required_evidence'][0], 'relation': 'supports', 'reason': item['reason'], 'rationale': item['reason']}
            a.check(parent['option_evidence'][item['option']] == expected, 'parent_evidence_saved_judgment_projection')
    pool = a.read(shard/'EVALUATION_POOL.json'); questions = pool['questions']
    a.check(pool['training_allowed'] is False and pool['formal_training_admission'] is False, 'evaluation_pool_no_training')
    a.check(len({q['candidate_id'] for q in questions}) == len(questions), 'unique_candidate_ids')
    content_review(a, shard, questions, source_context)
    manifests = manifest_pair(a, shard, questions)
    return {'config': config, 'pool': pool, 'questions': questions, 'manifests': manifests}


def metric(rows):
    n = len(rows); correct = sum(r['exact'] for r in rows)
    return {'calls': n, 'exact_correct': correct, 'exact_accuracy': str(Fraction(correct, n)) if n else None,
            'partial_mean': str(sum((Fraction(r['partial']) for r in rows), Fraction())/n) if n else None}


def replay(a, shard, context):
    questions = context['questions']; byid = {q['candidate_id']: q for q in questions}; sides = {}
    for side, model in MODELS.items():
        run = shard/'solver'/side
        if not questions:
            a.check(not list((run/'responses').glob('*.json')), 'empty_no_solver_responses'); sides[side] = {'rows': [], 'questions': [], 'selected_ids': []}; continue
        m = context['manifests'][side]; complete = a.read(run/'COMPLETE.json'); start = a.read(run/'STARTED.json')
        a.check(not (run/'FAILED.json').exists() and complete['actual_call_count'] == m['request_count'], 'native_solver_complete')
        a.check(start['manifest_sha256'] == a.bind(run/'manifest.json')['sha256']
                and dt(a.read(shard/'BATCH_REVIEW.COMPLETE.json')['completed_at_utc']) < dt(start['started_at_utc']), 'solver_after_freeze')
        planned = {r['request_id']: r for r in m['requests']}; actual = list((run/'responses').glob('*.json'))
        a.check(len(planned) == len(m['requests']) and {p.stem for p in actual} == set(planned), 'exact_valid_call_inventory')
        rows = []
        for path in sorted(actual):
            r = a.read(path); request = planned[path.stem]; q = byid[request['candidate_id']]
            a.check(all(r[k] == request[k] for k in ('request_id', 'candidate_id', 'root_id', 'question_type', 'slot', 'seed', 'mode'))
                    and r['variant_id'] == request['variant']['variant_id'], 'response_plan_identity')
            a.check(r['evaluation_valid'] is True and r['data_errors'] == [] and r['solver_model'] == model
                    and r['model_path'] == m['model'] and r['metadata']['finish_reason'] in ('stop', 'eos'), 'valid_response')
            meta = r['metadata']; a.check(meta['sampling_seed'] == request['seed'] and meta['prompt_token_count'] == request['prompt_tokens'], 'sampling_identity')
            for key in ('temperature', 'top_p', 'max_tokens', 'structured_output_api', 'structured_output_version'):
                a.check(meta[key] == m['decoder_contract'][key], 'response_decoder')
            value, exact = raw_score(q, request, r)
            rows.append({'request_id': path.stem, 'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                         'question_type': q['question_type'], 'variant_id': r['variant_id'], 'slot': r['slot'],
                         'seed': r['seed'], 'partial': str(value), 'exact': exact, 'response': a.bind(path)})
        summaries = []; chosen = []; saved = a.read(run/'RESULTS.json')
        a.check(saved['call_set_errors'] == [] and saved['solver_model'] == model, 'saved_native_results_valid')
        for q in questions:
            qrows = [r for r in rows if r['candidate_id'] == q['candidate_id']]; count = 18 if q['question_type'] == 'multi_select' else 12
            positions = {(f'perm_{i:02d}', j) for i in range(6 if count == 18 else 12) for j in range(3 if count == 18 else 1)}
            a.check(len(qrows) == count and {(r['variant_id'], r['slot']) for r in qrows} == positions, 'all_candidate_permutations_rollouts')
            stats = metric(qrows); selected = 0 < Fraction(stats['partial_mean']) < 1
            original = next(v for v in saved['questions'] if v['candidate_id'] == q['candidate_id'])
            a.check(original['full_partial_mean'] == stats['partial_mean'] and original['full_exact_rate'] == stats['exact_accuracy']
                    and original['selected_by_policy'] is selected and original['data_errors'] == [], 'independent_selection_replay')
            if selected: chosen.append(q['candidate_id'])
            summaries.append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'], 'question_type': q['question_type'],
                              'gold_count': len(q['gold']), 'control': control(q), 'selected': selected, **stats})
        a.check(saved['selected_questions'] == chosen, 'native_selected_ids')
        parent_complete = a.read(shard/(side.upper()+'_EVALUATION.COMPLETE.json'))
        a.check(parent_complete['status'] == 'COMPLETE' and parent_complete['calls'] == len(rows)
                and parent_complete['training_started'] is False, 'parent_solver_complete')
        sides[side] = {'rows': rows, 'questions': summaries, 'selected_ids': chosen}
    a.check([r['request_id'] for r in sides['trained']['rows']] == [r['request_id'] for r in sides['base']['rows']], 'paired_call_ids')
    selected = [q for q in questions if q['candidate_id'] in sides['trained']['selected_ids']]
    a.check(a.read(shard/'SELECTED_CANDIDATES.json')['questions'] == selected, 'selected_pool_projection')
    report = a.read(shard/'MIXED_EVALUATION_REPORT.json')
    a.check(report['selected_candidate_ids'] == sides['trained']['selected_ids'] and report['training_started'] is False
            and report['formal_training_admission'] is False, 'mixed_report_projection')
    return {'home': str(shard), 'models': sides, 'questions': questions,
            'selected_new': [q for q in selected if not control(q)], 'controls': [q['candidate_id'] for q in questions if control(q)]}


def deduplicate(questions):
    seen = {}; ids = {}; out = []; removed = []
    for q in questions:
        qid = q['candidate_id']; key = question_key(q)
        need(qid not in ids or ids[qid] == q, 'candidate_id_collision'); ids[qid] = q
        if key in seen:
            removed.append({'candidate_id': qid, 'retained_candidate_id': seen[key], 'reason': 'same_root_type_normalized_all_six_statements'})
        else:
            seen[key] = qid; out.append(q)
    return out, removed


def typed_pool(questions, status):
    need(not any(control(q) for q in questions), 'gold6_control_leaked_into_training_candidates')
    types = {k: sum(q['question_type'] == k for q in questions) for k in LABELS}
    gold = {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in questions) for g in range(2, 6)}
    roots = dict(sorted(Counter(q['root_id'] for q in questions).items()))
    return {'version': VERSION, 'created_at_utc': now(), 'status': status, 'questions': questions,
            'candidate_count': len(questions), 'by_type': {k: sum(q['question_type'] == k for q in questions) for k in LABELS},
            'type_distribution': types, 'gold_distribution': gold, 'multiselect_gold_distribution': gold,
            'root_distribution': roots, 'source_root_count': len(roots),
            'source_split': 'train', 'independent_test': False, 'training_allowed': False, 'training_started': False,
            'formal_training_admission': False, 'merged_into_baseline': False,
            'deduplication': 'root_id + question_type + sorted whitespace-normalized casefolded all six statements'}


def next_audit(home, stem):
    path = home/'evidence'/(stem+'.json'); attempt = 1
    while path.exists():
        attempt += 1; path = home/'evidence'/f'{stem}.attempt{attempt:02d}.json'
    return path


def run_inputs(home, shards, index):
    a = Audit(); target = next_audit(home, 'END_TO_END_INPUT_AUDIT'); report = {'version': VERSION, 'phase': 'inputs', 'created_at_utc': now()}
    try:
        a.bind(__file__); histories = preservation(home); ctx = stages(a, home, index)
        expansion_counts = source_token_counts(a, ctx)
        for shard in shards:
            a.check(not any((shard/(side+'_EVALUATION.STARTED.json')).exists() for side in ('TRAINED', 'BASE')), 'input_audit_before_live_solver')
            shard_inputs(a, shard, ctx)
        report.update(status='PASS', historical_preservation=histories, shards=[str(s) for s in shards],
                      expansion_metrics=expansion_counts)
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
    report.update(check_count=a.checks, files=list(a.files.values()), weights=a.weights,
                  new_network_or_model_calls=0, training_performed=False, completed_at_utc=now())
    write(target, report); print(json.dumps({'audit': str(target), 'status': report['status'], 'checks': a.checks}))
    need(report['status'] == 'PASS', 'input_audit_failed')
    return target


def latest_input(home, shards):
    candidates = []
    for path in (home/'evidence').glob('END_TO_END_INPUT_AUDIT*.json'):
        value = read(path)
        if value.get('status') == 'PASS' and value.get('shards') == [str(s) for s in shards]: candidates.append((value['completed_at_utc'], path, value))
    need(candidates, 'matching_input_audit_required')
    return max(candidates, key=lambda v: v[0])[1:]


def run_export(home, baseline, shards, index):
    a = Audit(); target = next_audit(home, 'END_TO_END_RESULT_AUDIT'); report = {'version': VERSION, 'phase': 'results', 'created_at_utc': now()}
    try:
        a.bind(__file__); prior_path, prior = latest_input(home, shards); a.bind(prior_path)
        for row in prior['files']: a.binding(row)
        # Weight SHA were checked before Solver; detect size/mtime changes by a
        # fresh hash in the result audit rather than trusting prior status alone.
        histories = preservation(home); source_context = stages(a, home, index); runs = []
        for shard in shards:
            for side in ('TRAINED', 'BASE'):
                a.check(dt(prior['completed_at_utc']) < dt(a.read(shard/(side+'_EVALUATION.STARTED.json'))['time']), 'input_audit_before_solver_stage')
            context = shard_inputs(a, shard, source_context); runs.append(replay(a, shard, context))
        a.check(sum(len(run['models']['trained']['rows']) for run in runs) <= a.read(home/'PILOT_PLAN.json')['max_solver_calls_per_model_total'],
                'actual_solver_budget')
        snapshot = a.read(home/'PRESERVATION_BEFORE.json')['files']
        a.check(str(baseline) in snapshot and a.bind(baseline)['sha256'] == snapshot[str(baseline)], 'protected_baseline_binding')
        old = a.read(baseline); a.check(old['candidate_count'] == len(old['questions']) and not any(control(q) for q in old['questions']), 'baseline_count_and_types')
        raw_new = [q for run in runs for q in run['selected_new']]
        new, removed = deduplicate(raw_new); cumulative, union_removed = deduplicate(old['questions']+new)
        new_pool = typed_pool(new, 'AUDITED_NEW_END_TO_END_CANDIDATES'); new_pool.update(shards=[str(s) for s in shards],
            selected_new_ids_before_deduplication=[q['candidate_id'] for q in raw_new], duplicates_removed=removed)
        cumulative_pool = typed_pool(cumulative, 'AUDITED_CUMULATIVE_CANDIDATES'); cumulative_pool.update(baseline=a.bind(baseline), duplicates_removed=union_removed)
        per_root = []; failures = []; attempts = []
        all_q = [q for run in runs for q in run['questions']]
        for root, ctx in source_context.items():
            content = [q for q in all_q if q['root_id'] == root and not control(q)]; chosen = [q for q in new if q['root_id'] == root]
            parent = 'parent_judgments' in ctx['actual']; entry = {'root_id': root, 'input_turns': ctx['input_turns'], 'new_turns': ctx['new_turns'],
                'source_decision': ctx['source_decision'], 'parent_prepared': parent,
                'content_usable': {kind: sum(q['question_type'] == kind for q in content) for kind in LABELS},
                'selected': {kind: sum(q['question_type'] == kind for q in chosen) for kind in LABELS},
                'selected_multiselect_gold': {str(g): sum(q['question_type'] == 'multi_select' and len(q['gold']) == g for q in chosen) for g in range(2, 6)},
                'stage_record': ctx['row']}; per_root.append(entry)
            if not ctx['new_turns'] or ctx['source_decision'] != 'READY' or not parent or not chosen:
                failures.append({'root_id': root, 'kind': 'pipeline_shortfall', 'actual_stages': entry,
                                 'reported_failure': ctx['row'].get('failure')})
        for run in runs:
            shard = Path(run['home'])
            for folder in (shard/'multiselect/assets', shard/'ordering'):
                for path in sorted(folder.rglob('*')):
                    if not path.is_file(): continue
                    if path.parent.parent.name == 'requests':
                        attempts.append({'kind': 'qa_request_artifact', 'artifact': a.bind(path)})
                    if path.name in ('ERROR.json', 'HTTP_ERROR.json', 'TIMEOUT.json', 'REJECTIONS.json'):
                        failures.append({'kind': 'raw_generation_failure', 'artifact': a.bind(path), 'record': a.read(path)})
            for name in ('multiselect/ALLOCATION.json', 'multiselect/FINAL_CONTENT_REVIEW.json', 'ordering/CONTENT_REVIEW.json'):
                failures.append({'kind': 'review_or_allocation_record', 'artifact': a.bind(shard/name), 'record': a.read(shard/name)})
            for row in run['models']['trained']['questions']:
                if not row['selected'] and not row['control']: failures.append({'kind': 'difficulty_exclusion', **row})
        expansion_run = Path(a.read(a.read(index)['expansion_audit']['path'])['run'])
        for path in sorted(expansion_run.rglob('*')):
            if not path.is_file(): continue
            if path.parent.parent.name == 'requests': attempts.append({'kind': 'expansion_request_artifact', 'artifact': a.bind(path)})
            if path.name in ('ERROR.json', 'HTTP_ERROR.json', 'TIMEOUT.json', 'REJECTIONS.json'):
                failures.append({'kind': 'raw_expansion_failure', 'artifact': a.bind(path), 'record': a.read(path)})
        rows = {side: [r for run in runs for r in run['models'][side]['rows']] for side in MODELS}
        byid = {q['candidate_id']: q for q in all_q}; stats = {}
        for kind in LABELS:
            stats[kind] = {side: metric([r for r in rows[side] if r['question_type'] == kind and not control(byid[r['candidate_id']])]) for side in MODELS}
        report.update(status='PASS', independent_raw_scores_and_selection=True, input_audit=a.bind(prior_path),
            historical_preservation=histories, stage_index=a.bind(index), baseline=a.bind(baseline),
            expansion_metrics=prior['expansion_metrics'],
            shards=[str(s) for s in shards],
            replayed_models={side: [q for run in runs for q in run['models'][side]['questions']] for side in MODELS},
            control_candidate_ids=[qid for run in runs for qid in run['controls']],
            per_root=per_root, by_type=stats, selected_new_ids=[q['candidate_id'] for q in new],
            new_count=len(new), cumulative_count=len(cumulative),
            source_semantics_recertified_by_this_audit=False, source_ready_is_answer_certification=False,
            limitations=['Saved independent semantic judgments are replayed, not independently recertified by this numeric auditor.',
                         'Roots are historical TRAIN sources with known feedback, not unseen roots or an independent test set.',
                         'Sibling questions and repeated Solver calls are dependent samples.'])
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
    report.update(check_count=a.checks, files=list(a.files.values()), weights=a.weights,
                  new_network_or_model_calls=0, training_performed=False, completed_at_utc=now())
    write(target, report)
    need(report['status'] == 'PASS', 'result_audit_failed:' + repr(report.get('error')))
    new_pool['result_audit'] = bind(target); cumulative_pool['result_audit'] = bind(target)
    write(home/'AUDITED_NEW_CANDIDATES.json', new_pool); write(home/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json', cumulative_pool)
    write(home/'FAILURE_INDEX.json', {'version': VERSION, 'created_at_utc': now(), 'failures': failures,
          'all_request_artifacts': attempts, 'result_audit': bind(target), 'training_allowed': False})
    text = ['# v90 四 root 扩充＋多选排序结果', '', f'本轮入选 {len(new)} 题，累计 {len(cumulative)} 题。完整 source 新增 {sum(v["new_turns"] for v in source_context.values())} turns。',
        '', f'本轮上游扩充成功 {prior["expansion_metrics"]["accepted_new_scene_roots"]}/{len(source_context)} root，新增正文 {prior["expansion_metrics"]["new_body_tokens"]} Qwen3-4B tokens（逐轮text独立encode、无speaker/ref/chat包装）。QA内部new_expansion_tokens=0仅表示QA阶段不再扩写对话。',
        '', '这些是历史 TRAIN root 上本轮新增场景及新题；不构成跨新 root 稳定性证据。未训练、未正式训练入库。', '',
        '| Root | 新增 turns | Source | 多选合格/入选 | 排序合格/入选 |', '|---|---:|---|---:|---:|']
    for row in per_root:
        text.append(f'| {row["root_id"].rsplit("-",1)[-1]} | {row["new_turns"]} | {row["source_decision"]} | {row["content_usable"]["multi_select"]}/{row["selected"]["multi_select"]} | {row["content_usable"]["ordering"]}/{row["selected"]["ordering"]} |')
    text += ['', 'Gold2/3/4/5 本轮入选数：' + json.dumps(new_pool['gold_distribution'], ensure_ascii=False),
        '', '内容审核通过与难度入选分别计数。入选仍为所有作答有效且 trained 平均 partial 严格在 (0,1) 内；完全正确率仅描述。',
        '', '完整题目及答案见 AUDITED_NEW_CANDIDATES.json；本轮所有失败及审核分配见 FAILURE_INDEX.json；按题型配对指标、逐 root 阶段及 raw 回放审计见 ' + str(target.relative_to(home)) + '。',
        '', 'Saved judgments 和精确引文支持内容审核，数值审计不重新认证语义；同一 root/父题的兄弟题不能作为独立样本。', '']
    write(home/'RESULTS_ZH.md', '\n'.join(text))
    export_check = audit_saved_exports(home)
    write(home/'evidence/SAVED_EXPORT_PROJECTION_AUDIT.json', export_check)
    print(json.dumps({'audit': str(target), 'status': 'PASS', 'new': len(new), 'cumulative': len(cumulative), 'by_type': new_pool['by_type']}))


def audit_saved_exports(home):
    """Rebuild the stored new/cumulative questions from raw calls after writing."""
    a = Audit(); new = a.read(home/'AUDITED_NEW_CANDIDATES.json'); cumulative = a.read(home/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json')
    a.check(new['result_audit'] == cumulative['result_audit'], 'saved_exports_same_result_audit')
    report = a.binding(new['result_audit']); a.check(report['status'] == 'PASS', 'saved_exports_result_audit_pass')
    for row in report['files']: a.binding(row)
    raw_selected = []; question_summaries = {side: [] for side in MODELS}
    for shard_path in report['shards']:
        shard = Path(shard_path); qs = a.read(shard/'EVALUATION_POOL.json')['questions']
        for side in MODELS:
            if not qs: continue
            manifest = a.read(shard/'solver'/side/'manifest.json')
            planned = {r['request_id']: r for r in manifest['requests']}
            paths = list((shard/'solver'/side/'responses').glob('*.json'))
            a.check({p.stem for p in paths} == set(planned) and len(paths) == len(planned), 'saved_export_raw_inventory')
            by_id = {q['candidate_id']: q for q in qs}; rows = []
            for path in sorted(paths):
                request = planned[path.stem]; q = by_id[request['candidate_id']]
                value, exact = raw_score(q, request, a.read(path))
                rows.append({'candidate_id': q['candidate_id'], 'partial': str(value), 'exact': exact})
            for q in qs:
                stats = metric([r for r in rows if r['candidate_id'] == q['candidate_id']])
                expected_calls = 18 if q['question_type'] == 'multi_select' else 12
                a.check(stats['calls'] == expected_calls, 'saved_export_all_candidate_calls')
                selected = 0 < Fraction(stats['partial_mean']) < 1
                question_summaries[side].append({'candidate_id': q['candidate_id'], 'root_id': q['root_id'],
                    'question_type': q['question_type'], 'gold_count': len(q['gold']), 'control': control(q), 'selected': selected, **stats})
                if side == 'trained' and selected and not control(q): raw_selected.append(q)
    a.check(question_summaries == report['replayed_models'], 'saved_export_replays_match_result_audit')
    expected_new, removed = deduplicate(raw_selected); old = a.binding(report['baseline'])
    expected_cumulative, union_removed = deduplicate(old['questions']+expected_new)
    a.check(new['questions'] == expected_new and cumulative['questions'] == expected_cumulative, 'saved_export_exact_question_projection')
    a.check(new['selected_new_ids_before_deduplication'] == [q['candidate_id'] for q in raw_selected]
            and new['duplicates_removed'] == removed and cumulative['duplicates_removed'] == union_removed
            and report['selected_new_ids'] == [q['candidate_id'] for q in expected_new], 'saved_export_deduplication_projection')
    a.check(cumulative['baseline'] == report['baseline'] and new['shards'] == report['shards'], 'saved_export_baseline_shard_binding')
    for pool, expected in ((new, expected_new), (cumulative, expected_cumulative)):
        recomputed = typed_pool(expected, pool['status'])
        for field in ('candidate_count', 'by_type', 'type_distribution', 'gold_distribution',
                      'multiselect_gold_distribution', 'root_distribution', 'source_root_count',
                      'training_allowed', 'training_started', 'formal_training_admission'):
            a.check(pool[field] == recomputed[field], 'saved_export_typed_counts_and_permissions:' + field)
        a.check(len({question_key(q) for q in expected}) == len(expected), 'saved_export_unique_typed_fingerprints')
    a.check(report['new_count'] == len(expected_new) and report['cumulative_count'] == len(expected_cumulative), 'saved_export_result_counts')
    return {'version': VERSION, 'status': 'PASS', 'created_at_utc': now(), 'independent_raw_selection_recomputed': True,
            'new_count': len(expected_new), 'cumulative_count': len(expected_cumulative), 'check_count': a.checks,
            'files': list(a.files.values()), 'new_network_or_model_calls': 0, 'training_performed': False}


def seal(home):
    need(not (home/'PILOT_COMPLETE.json').exists(), 'already_sealed')
    preservation_report = preservation(home)
    replayed_export = audit_saved_exports(home)
    saved_export = read(home/'evidence/SAVED_EXPORT_PROJECTION_AUDIT.json')
    need(saved_export['status'] == 'PASS' and saved_export['new_count'] == replayed_export['new_count']
         and saved_export['cumulative_count'] == replayed_export['cumulative_count'], 'saved_export_audit_before_seal')
    for row in saved_export['files']: need(bind(row['path']) == row, 'saved_export_binding_changed_before_seal')
    for name in ('AUDITED_NEW_CANDIDATES.json', 'CUMULATIVE_AUDITED_NEW_CANDIDATES.json'):
        pool = read(home/name); question_key_all = [question_key(q) for q in pool['questions']]
        need(len(question_key_all) == len(set(question_key_all)) == pool['candidate_count'] and not any(control(q) for q in pool['questions']), 'final_pool_unique_types')
        audit = read(pool['result_audit']['path']); need(bind(pool['result_audit']['path']) == pool['result_audit'] and audit['status'] == 'PASS', 'final_result_audit_pass')
        for row in audit['files']: need(bind(row['path']) == row, 'audited_file_changed_before_seal')
    write(home/'evidence/HISTORICAL_PRESERVATION_FINAL.json', preservation_report)
    listing = ['# v90 artifact index', '', 'All original generation attempts, saved judgments, failures, paired responses and exports are retained.', '']
    listing += [f'- [{Path(p).relative_to(home)}]({Path(p).relative_to(home)})' for p in sorted(files_under(home))]
    write(home/'ARTIFACT_INDEX.md', '\n'.join(listing)+'\n')
    files = {path: sha(path) for path in sorted(files_under(home))}
    complete = {'version': VERSION, 'status': 'COMPLETE', 'completed_at_utc': now(), 'home': str(home),
                'files': files, 'file_count_excluding_this_manifest': len(files), 'historical_preservation': preservation_report,
                'training_allowed': False, 'training_started': False, 'formal_training_admission': False,
                'no_further_writes_permitted': True}
    write(home/'PILOT_COMPLETE.json', complete)
    return verify_complete(home)


def verify_complete(home):
    path = home/'PILOT_COMPLETE.json'; complete = read(path); actual = files_under(home)-{str(path)}
    need(complete['status'] == 'COMPLETE' and complete['home'] == str(home) and actual == set(complete['files']), 'complete_exact_inventory')
    for item, digest in complete['files'].items(): need(sha(item) == digest, 'complete_file_changed:' + item)
    need(all(complete[k] is False for k in ('training_allowed', 'training_started', 'formal_training_admission')), 'complete_no_training')
    preservation(home)
    return {'status': 'PASS', 'file_count': len(actual)+1, 'complete_manifest': str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=DEFAULT_HOME)
    parser.add_argument('--action', choices=('inputs', 'export', 'seal', 'verify-complete'), required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--stage-index', type=Path)
    parser.add_argument('--shard', type=Path, action='append', default=[])
    args = parser.parse_args(); home = args.home.resolve(); shards = [p.resolve() for p in args.shard]
    need(len(shards) == len(set(shards)) and all(s.parent == home/'shards' for s in shards), 'unique_shards_under_home')
    if args.action == 'verify-complete': print(json.dumps(verify_complete(home))); return
    need(not (home/'PILOT_COMPLETE.json').exists(), 'sealed_directory_read_only')
    if args.action == 'seal': print(json.dumps(seal(home))); return
    index = (args.stage_index or home/'evidence/END_TO_END_STAGE_INDEX.json').resolve()
    if args.action == 'inputs': run_inputs(home, shards, index)
    else:
        baseline = (args.baseline or home.parent/'h4_scale_pilot_v89/CUMULATIVE_AUDITED_NEW_CANDIDATES.json').resolve()
        run_export(home, baseline, shards, index)


if __name__ == '__main__':
    main()
