"""Reproducible measurements, deliberately separate from semantic judgments."""
from pathlib import Path
import collections, json, re, statistics

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'natural_question_forms_fg_15root_extension_20261026_v1'

def read(p): return json.loads(Path(p).read_text())
def avg(xs): return statistics.mean(xs) if xs else None
def norm(s): return re.sub(r'\s+', ' ', s).strip().strip('“”"').strip()

def main():
    candidates = read(OLD / 'CANDIDATES_PRIVATE.json')
    rows = [c for c in candidates if 'question' in c]
    scores = read(OLD / 'execution/SCORES.json')['records']
    sources = {c['root_id']: read(c['source_binding']['path']) for c in rows}
    audits = []
    features = collections.defaultdict(list)
    for c in rows:
        by_ref = {t['turn_ref']: t for t in sources[c['root_id']]['turns']}
        option_rows = []
        for key, text in c['options'].items():
            analysis = c['item_analysis'][key]
            refs = list(dict.fromkeys(e['turn_ref'] for e in analysis['evidence']))
            exact_matches = [r for r, t in by_ref.items() if norm(text) == norm(t['text'])]
            substring_matches = [r for r, t in by_ref.items() if norm(text) in norm(t['text'])]
            truth = key in c['gold']
            negative = bool(re.search(r"\b(?:no|not|never|only|already|always|must|cannot|can't|won't|without)\b", text, re.I))
            record = {'option_id': key, 'gold': truth, 'chars': len(text),
                      'evidence_ref_count': len(refs),
                      'evidence_scene_count': len({r.rsplit(':', 1)[0] for r in refs}),
                      'evidence_refs': refs, 'exact_source_turn_match': exact_matches,
                      'source_substring_match': substring_matches,
                      'surface_marker': negative}
            option_rows.append(record)
            features[('true' if truth else 'false')].append(record)
        audits.append({'candidate_id': c['candidate_id'], 'root_id': c['root_id'],
                       'arm': c['arm'], 'slot_id': c['slot_id'], 'gold': c['gold'],
                       'options': option_rows, 'semantic_necessity': 'NOT_ESTABLISHED_BY_REFERENCE_COUNTS'})
    summary = {}
    for slot in ('MC2', 'MC3', 'MC4', 'MC5', 'MC6'):
        xs = [r for r in audits if r['slot_id'] == slot]
        options = [o for r in xs for o in r['options']]
        gold_options = [o for o in options if o['gold']]
        summary[slot] = {
            'questions': len(xs), 'options': len(options),
            'questions_with_any_exact_source_turn_option': sum(any(o['exact_source_turn_match'] for o in r['options']) for r in xs),
            'questions_all_options_exact_source_turns': sum(all(o['exact_source_turn_match'] for o in r['options']) for r in xs),
            'questions_all_gold_options_cite_single_turn': sum(all(o['evidence_ref_count'] == 1 for o in r['options'] if o['gold']) for r in xs),
            'gold_option_single_evidence_ref_count': sum(o['evidence_ref_count'] == 1 for o in gold_options),
            'gold_options': len(gold_options),
            'questions_any_gold_option_cites_multiple_scenes': sum(any(o['evidence_scene_count'] >= 2 for o in r['options'] if o['gold']) for r in xs),
            'caution': 'Citation counts and exact text overlap do not prove inference complexity or context independence.'}
    ablations = []
    for slot in ('MC2', 'MC3', 'MC4', 'MC5', 'MC6'):
        for arm in ('F', 'G'):
            for cond in ('FULL', 'QUESTION_ONLY'):
                xs = [r for r in scores if (r['slot_id'], r['arm'], r['condition']) == (slot, arm, cond)]
                ablations.append({'slot_id': slot, 'arm': arm, 'condition': cond, 'n': len(xs),
                                  'unique_questions': len({r['candidate_id'] for r in xs}),
                                  'valid': sum(r['valid'] for r in xs),
                                  'exact_count': sum(r['exact'] for r in xs),
                                  'exact_accuracy': avg([int(r['exact']) for r in xs]),
                                  'J_n': sum(r['J'] is not None for r in xs),
                                  'J': avg([r['J'] for r in xs if r['J'] is not None])})
    surface = {key: {'options': len(xs), 'mean_chars': avg([o['chars'] for o in xs]),
                     'marker_rate': avg([int(o['surface_marker']) for o in xs])}
               for key, xs in features.items()}
    out = {'planned_records': len(candidates), 'authored': len(rows),
           'failed': len(candidates)-len(rows), 'by_slot': summary,
           'surface_features': surface, 'surface_features_are_risk_indicators_not_proof_of_answerability': True,
           'existing_full_question_only': ablations, 'per_question': audits,
           'limits': 'The existing FULL and QUESTION_ONLY use different instructions and evidence-reference schemas. No parent-only, extension-only, single-premise, or removal test was run in the original experiment. A correct no-context prediction can be a guess; a low no-context score cannot establish logical necessity.'}
    p = HERE / 'audit/measurements.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'by_slot': summary, 'surface_features': surface}, ensure_ascii=False))

if __name__ == '__main__': main()
