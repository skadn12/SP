"""Provenance and review-completeness gate; entailment remains a reviewer judgment."""
from __future__ import annotations
import copy, hashlib, importlib.util, json, random, sys
from pathlib import Path

sys.dont_write_bytecode = True
LEGACY_PATH = Path('/root/autodl-tmp/gxy-selfplay/scripts/current_method/production_question_gate.py')
spec = importlib.util.spec_from_file_location('single_order_legacy_production_gate', LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(LEGACY_PATH.parent))
spec.loader.exec_module(legacy)

def fingerprint(value):
    return legacy.fingerprint(value)

def binding(path):
    return legacy.file_binding(path)

def scene(ref):
    return ref.rsplit(':', 1)[0]

def text(value):
    return isinstance(value, str) and bool(value.strip())

def validate_candidate(candidate):
    errors = []
    def reject(condition, message):
        if not condition:
            errors.append(message)
    if not isinstance(candidate, dict):
        return {'candidate_id': None, 'mechanical_pass': False,
                'errors': ['candidate must be an object'], 'semantic_necessity_proved_by_code': False}
    options, gold = candidate.get('options'), candidate.get('gold')
    reject(all(text(candidate.get(k)) for k in ('candidate_id', 'root_id', 'author_id', 'question')),
           'nonempty candidate/root/author IDs and question required')
    kind = candidate.get('question_type')
    reject(kind in ('single_choice', 'ordering'), 'single_choice or ordering required')
    good_options = (isinstance(options, dict) and ((kind == 'single_choice' and len(options) == 4) or (kind == 'ordering' and 4 <= len(options) <= 6))
                    and all(text(k) and text(v) for k, v in options.items())
                    and len(set(options.values())) == len(options))
    reject(good_options, 'four single-choice or four-to-six ordering distinct options with stable IDs required')
    options = options if isinstance(options, dict) else {}
    good_gold = (isinstance(gold, list) and ((kind == 'single_choice' and len(gold) == 1) or (kind == 'ordering' and len(gold) == len(options)))
                 and all(isinstance(x, str) for x in gold) and len(set(gold)) == len(gold)
                 and set(gold) <= set(options))
    reject(good_gold, 'Gold must contain exactly one single-choice ID or every ordering ID once in order')
    gold = gold if good_gold else []
    source, by_ref, positions = {}, {}, {}
    try:
        source_binding = candidate['source_binding']
        reject(binding(source_binding['path']) == source_binding, 'source binding drift')
        source = json.loads(Path(source_binding['path']).read_text())
        reject(source.get('root_id') == candidate.get('root_id'), 'source root drift')
        turns = source['turns']
        reject(isinstance(turns, list) and bool(turns), 'nonempty source turns required')
        for index, turn in enumerate(turns):
            ref = turn['turn_ref']
            reject(text(ref) and ':' in ref and ref not in by_ref, 'source turn refs must be distinct coordinates')
            reject(text(turn.get('speaker')) and text(turn.get('text')), 'source speaker/text required')
            by_ref[ref], positions[ref] = turn, index
    except (KeyError, TypeError, ValueError, OSError) as error:
        errors.append('invalid source binding/document: ' + str(error))
    analyses = candidate.get('item_analysis')
    reject(isinstance(analyses, dict) and set(analyses) == set(options), 'one analysis per option required')
    for key, analysis in (analyses.items() if isinstance(analyses, dict) else []):
        if not isinstance(analysis, dict):
            errors.append(f'{key}: analysis must be an object')
            continue
        reject(analysis.get('should_select') is (key in gold), f'{key}: truth/Gold mismatch')
        evidence = analysis.get('evidence')
        reject(text(analysis.get('rationale')) and isinstance(evidence, list) and bool(evidence),
               f'{key}: rationale and evidence required')
        for ev in evidence if isinstance(evidence, list) else []:
            ev = ev if isinstance(ev, dict) else {}
            ref = ev.get('turn_ref')
            turn = by_ref.get(ref) if isinstance(ref, str) else None
            reject(bool(turn and turn['speaker'] == ev.get('speaker') and text(ev.get('quote'))
                        and ev['quote'] in turn['text']), f'{key}: invalid exact evidence')
    cert = candidate.get('dependency_certificate')
    cert = cert if isinstance(cert, dict) else {}
    groups = cert.get('premise_groups')
    reject(isinstance(groups, list) and len(groups) >= 2, 'at least two premise groups required')
    if kind == 'single_choice':
        reject(cert.get('target_option') in gold and len(gold) == 1, 'target_option must be the unique correct option')
    elif kind == 'ordering':
        pair = cert.get('target_pair')
        good_pair = isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) and x in gold for x in pair) and pair[0] != pair[1]
        reject(good_pair and gold.index(pair[0]) < gold.index(pair[1]), 'target_pair must be distinct Gold-ordered stable IDs')
    reject(text(cert.get('joint_inference')) and cert.get('single_group_insufficient') is True,
           'joint inference explanation and single-group limitation required')
    ids, all_refs = [], []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            errors.append('premise group must be an object')
            continue
        gid, refs = group.get('group_id'), group.get('turn_refs')
        reject(text(gid) and gid not in ids, 'unique nonempty group IDs required')
        ids.append(gid)
        good_refs = (isinstance(refs, list) and bool(refs) and all(isinstance(r, str) and r in by_ref for r in refs))
        reject(good_refs, 'unknown or empty premise group')
        if good_refs:
            reject(len(refs) == len(set(refs)) and refs == sorted(refs, key=positions.get),
                   'premise refs must be distinct and in original source order')
            reject(not set(refs).intersection(all_refs), 'premise groups must not overlap; merge shared premises')
            all_refs.extend(refs)
        reject(all(text(group.get(k)) for k in ('premise', 'why_necessary', 'removal_effect')),
               'premise group needs substantive dependency rationale')
    redundant = cert.get('redundant_turn_refs')
    good_redundant = (isinstance(redundant, list) and all(isinstance(r, str) and r in by_ref for r in redundant))
    reject(good_redundant, 'redundant_turn_refs must be an explicit list of known source refs')
    if good_redundant:
        reject(len(redundant) == len(set(redundant)) and redundant == sorted(redundant, key=positions.get),
               'redundant refs must be distinct and in source order')
        reject(set(redundant) <= set(all_refs), 'every redundant ref must belong to its full premise deletion group')
    reject(len({scene(r) for r in all_refs}) >= 2, 'cross-scene pilot target requires >=2 scenes')
    return {'candidate_id': candidate.get('candidate_id'), 'mechanical_pass': not errors,
            'errors': errors, 'semantic_necessity_proved_by_code': False}

REVIEW_CHECKS = ('gold_truth', 'source_support', 'joint_necessity',
                 'redundancy_bypass', 'question_options_shortcut',
                 'distractor_style', 'not_independent_fact_bundle')

def validate_review(candidate, review):
    audit = validate_candidate(candidate)
    errors = list(audit['errors'])
    if not isinstance(candidate, dict) or not isinstance(review, dict):
        return audit | {'admitted': False, 'errors': errors + ['review must be an object']}
    if errors:
        return audit | {'admitted': False, 'errors': errors,
                        'admission_basis': 'independent semantic judgments, not mechanical proof'}
    if review.get('candidate_id') != candidate.get('candidate_id'):
        errors.append('review candidate ID drift')
    if review.get('candidate_sha256') != fingerprint(candidate):
        errors.append('review hash drift')
    if (not text(candidate.get('author_id')) or not text(review.get('reviewer_id'))
            or review['reviewer_id'] == candidate.get('author_id')):
        errors.append('independent reviewer identity required')
    if 'source_binding' in review and review['source_binding'] != candidate.get('source_binding'):
        errors.append('review source binding drift')
    if 'author_id' in review and review['author_id'] != candidate.get('author_id'):
        errors.append('review author ID drift')
    if review.get('gold_blind') is not True:
        errors.append('source/question/options-only independent Gold stage required')
    if review.get('decision') != 'PASS':
        errors.append('review not PASS')
    independent = review.get('independent_gold')
    if (not isinstance(independent, list) or not all(isinstance(x, str) for x in independent)
            or len(independent) != len(set(independent)) or independent != candidate.get('gold', [])):
        errors.append('independent Gold mismatch')
    checks = review.get('checks', {})
    required_checks = list(REVIEW_CHECKS)
    if candidate.get('question_type') == 'ordering':
        required_checks.remove('distractor_style')
        required_checks += ['ordering_total_order', 'item_style' if 'item_style' in checks else 'distractor_style']
    for key in required_checks:
        row = checks.get(key, {}) if isinstance(checks, dict) else {}
        if not isinstance(row, dict) or row.get('decision') != 'PASS' or not text(row.get('rationale')):
            errors.append(f'unresolved/missing review check: {key}')
    cert = candidate.get('dependency_certificate', {})
    groups = {g.get('group_id'): g.get('turn_refs', []) for g in cert.get('premise_groups', []) if isinstance(g, dict)}
    rows = review.get('ablation_review', [])
    if (not isinstance(rows, list) or len(rows) != len(groups)
            or any(not isinstance(r, dict) for r in rows)
            or {r.get('group_id') for r in rows if isinstance(r, dict)} != set(groups)):
        errors.append('independent removal review must cover every group exactly once')
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        if row.get('removed_turn_refs') != groups.get(row.get('group_id')):
            errors.append('review must confirm the full ordered deletion group including repeated evidence')
        if row.get('target_underdetermined_after_removal') is not True or not text(row.get('rationale')):
            errors.append('removal does not establish target underdetermination')
    return audit | {'admitted': not errors, 'errors': errors,
                    'admission_basis': 'independent semantic judgments, not mechanical proof'}

def prepare_display(candidate, seed):
    # Stable ID order and candidate/seed only: Gold never influences the shuffle.
    options = candidate['options']
    original = list(options)
    material = json.dumps(['single_order_dependency_display_v1', seed, candidate['candidate_id'], original],
                          ensure_ascii=False, separators=(',', ':')).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), 'big'))
    order = original[:]
    rng.shuffle(order)
    mapping = {chr(65+i): key for i, key in enumerate(order)}
    public = {k: candidate[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')}
    public['options'] = {k: options[v] for k, v in mapping.items()}
    private = {'display_to_canonical': mapping, 'canonical_gold': copy.deepcopy(candidate['gold']),
               'display_gold': [{v: k for k, v in mapping.items()}[key] for key in candidate['gold']], 'seed': seed}
    return public, private

def prepare_displays(candidate, seeds):
    """Resolve display collisions without inspecting Gold or option semantics."""
    import itertools
    orders = list(itertools.permutations(candidate['options']))
    used, result = set(), []
    for seed in seeds:
        material = json.dumps(['single_order_unique_displays_v1', candidate['candidate_id'], seed,
                               list(candidate['options'])], ensure_ascii=False, separators=(',', ':')).encode()
        rank = int.from_bytes(hashlib.sha256(material).digest(), 'big') % len(orders)
        skipped = 0
        while orders[rank] in used:
            rank, skipped = (rank + 1) % len(orders), skipped + 1
            if skipped >= len(orders):
                raise ValueError('too many display seeds for distinct permutations')
        order = orders[rank]
        used.add(order)
        mapping = {chr(65+i): key for i, key in enumerate(order)}
        public = {k: candidate[k] for k in ('candidate_id', 'root_id', 'question_type', 'question')}
        public['options'] = {label: candidate['options'][key] for label, key in mapping.items()}
        reverse = {key: label for label, key in mapping.items()}
        private = {'display_to_canonical': mapping, 'canonical_gold': copy.deepcopy(candidate['gold']),
                   'display_gold': [reverse[key] for key in candidate['gold']], 'seed': seed,
                   'collision_steps': skipped}
        result.append((public, private))
    return result
