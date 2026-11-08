"""Record which v98 gates were actually exercised, with no semantic inference."""
from pathlib import Path
from collections import Counter
import hashlib
import json

HOME = Path(__file__).resolve().parents[1]
RUN = HOME / 'expansion_engine/runs/four_roots_B_scene01'
def read(p): return json.loads(p.read_text())
def bind(p): return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    assert (RUN / 'REPORT.json').exists()
    rows = []
    roles = Counter(); corrections = Counter(); nested = Counter()
    for directory in sorted((RUN / 'requests').iterdir()):
        proof = read(directory / 'PROOF.json'); role = proof['role']; roles[role] += 1
        row = {'request': str(directory), 'role': role, 'root_id': proof.get('root_id')}
        request = read(directory / 'REQUEST.json')
        row['request_max_tokens'] = request['max_tokens']
        response = directory / 'RESPONSE.json'
        if response.exists():
            raw = read(response)['text']
            try: parsed = json.loads(raw)
            except (TypeError, ValueError): parsed = None
            if isinstance(parsed, dict):
                if role == 'planner':
                    row['scene_identity_present'] = [isinstance(s.get('event_identity'), dict)
                        for s in parsed.get('scenes', []) if isinstance(s, dict)]
                functions = []
                for card in parsed.get('card_checks', []):
                    if not isinstance(card, dict): continue
                    function = card.get('function_check') if role == 'event_state_extractor' else (
                        card if card.get('kind') == 'event_function' else None)
                    if not isinstance(function, dict): continue
                    comparison = function.get('comparison')
                    functions.append({'comparison_present': isinstance(comparison, dict),
                                      'outer_verdict': function.get('verdict'),
                                      'realized': function.get('realized')})
                    if isinstance(comparison, dict):
                        for key in ('request_acceptance', 'experience_state'):
                            child = comparison.get(key)
                            if isinstance(child, dict): nested[key + ':' + str(child.get('verdict'))] += 1
                if functions: row['whole_event_checks'] = functions
        progress = directory / 'CORRECTION_PROGRESS.json'
        if progress.exists():
            obj = read(progress)
            corrections[obj['status']] += 1
            row['correction_progress'] = {'binding': bind(progress), 'status': obj['status'],
                                         'target_count': len(obj.get('targets', [])),
                                         'error_codes': [e.get('code') for e in obj.get('errors', [])]}
        row['saved_records'] = [bind(p) for p in directory.glob('*.json') if p.name in (
            'VALIDATION.json', 'REVIEW.json', 'ACCEPTANCE.json', 'EVENT_STATE_DECISION.json',
            'FULL_VALIDATION_BEFORE_CORRECTION_GATE.json')]
        rows.append(row)
    result = {'version': 'v98_actual_minfix_observations_v1', 'status': 'RECORDED',
              'roles': dict(roles), 'correction_progress_counts': dict(corrections),
              'nested_model_verdict_counts': dict(nested), 'requests': rows,
              'limits': 'Presence and mechanical validity do not certify purpose, entailment, applicability, history completeness or generation reliability. Regression fixtures are not independent new yield.',
              'semantic_certified': False, 'new_model_calls': 0, 'script': bind(Path(__file__))}
    with (HOME / 'evidence/ACTUAL_MINFIX_OBSERVATIONS.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'roles': dict(roles), 'correction': dict(corrections), 'nested': dict(nested)}))
if __name__ == '__main__': main()
