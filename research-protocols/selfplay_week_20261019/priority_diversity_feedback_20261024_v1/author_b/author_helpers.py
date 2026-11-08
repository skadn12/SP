import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
AUTHOR = '/root/followup_production_pipeline_audit/historical_fixture_checks'
sys.path.insert(0, '/root/autodl-tmp/gxy-selfplay/scripts/current_method')
from production_question_gate import bind_candidates, audit_candidate

SELECTION = json.loads((BASE / 'SELECTION_FREEZE.json').read_text())
PLANS = {p['candidate_id']: p for p in json.loads((OUT / 'TASK_PLANS.json').read_text())['plans']}


class Author:
    def __init__(self, number):
        self.number = str(number)
        self.root = next(r for r in SELECTION['roots'] if r['root_id'].endswith('-' + self.number))
        self.source = json.loads(Path(self.root['path']).read_text())
        self.turns = {t['turn_ref']: t for t in self.source['turns']}
        self.candidates = []
        self.shortfalls = {}

    def ev(self, refs):
        return [{k: self.turns[r][k] for k in ('turn_ref', 'speaker')} | {'quote': self.turns[r]['text']} for r in refs]

    def pair(self, ref, claim):
        t = self.turns[ref]
        return f'Evidence: {t["speaker"]} says, "{t["text"]}" Claim: {claim}'

    def base(self, slot, question):
        plan = PLANS[f'PDF1-{self.number}-{slot}']
        return {
            'schema_version': 'production_author_candidate_v1',
            'candidate_id': plan['candidate_id'], 'parent_candidate_id': None,
            'author_id': AUTHOR, 'author_type': 'codex_ai_agent',
            'root_id': self.root['root_id'],
            'question_type': next(s['question_type'] for s in self.root['slot_plan'] if s['slot'] == slot),
            'task_operation': plan['task_operation'], 'question': question,
            'semantic_admission': 'AUTHOR_DRAFT_PENDING_INDEPENDENT_REVIEW',
            'source_extension_used': False, 'solver_evaluated': False,
            'formal_weakness_admission': False, 'difficulty_increase_claim': False,
            'feedback_target': {
                'card_id': plan['card_id'], 'dimension': plan['dimension'],
                'predicted_error': plan['predicted_error'],
                'competing_explanation': plan['competing_explanations'],
                'source_specific_realization': plan['target_relation'],
                'positive_counterexample': plan['positive_counterexample'],
                'status': 'EXPLORATORY_TASK_REALIZATION_PENDING_INDEPENDENT_REVIEW',
                'internal_cause_claim': False,
            },
            'inspiration': 'Operation-level adaptation of MultiRC, TimeDial, DREAM and RecipeQA; no borrowed dataset question.',
        }

    def mc(self, slot, question, rows):
        c = self.base(slot, question)
        c.update(options={}, gold=[], item_analysis={})
        for label, (text, truth, refs, rationale, changed) in zip('ABCDEF', rows):
            c['options'][label] = text
            if truth:
                c['gold'].append(label)
            c['item_analysis'][label] = {
                'evidence': self.ev(refs), 'should_select': truth,
                'author_judgment': 'supported_for_question' if truth else 'not_supported_for_question',
                'rationale': rationale, 'changed_relation': changed,
                'independent_fact_key': f'{self.number}_{slot}_{label}',
                'review_status': 'AUTHOR_DRAFT_PENDING_INDEPENDENT_REVIEW',
            }
        c['all_select_control'] = len(c['gold']) == 6
        self.candidates.append(c)

    def order(self, slot, question, operation, rows, sequence, edge_reasons, dependence, values=None):
        c = self.base(slot, question)
        c.update(items={}, gold=sequence, item_analysis={})
        for label, text, refs, rationale in rows:
            c['items'][label] = text
            c['item_analysis'][label] = {
                'evidence': self.ev(refs), 'rationale': rationale,
                'review_status': 'AUTHOR_DRAFT_PENDING_INDEPENDENT_REVIEW',
            }
            if values:
                c['item_analysis'][label]['clock_value_minutes'] = values[label]
        edges = []
        for (a, b), reason in zip(zip(sequence, sequence[1:]), edge_reasons):
            refs = list(dict.fromkeys([e['turn_ref'] for e in c['item_analysis'][a]['evidence'] + c['item_analysis'][b]['evidence']]))
            edges.append({'before': a, 'after': b, 'relation': operation, 'rationale': reason, 'evidence': self.ev(refs)})
        c['ordering_analysis'] = {
            'operation_class': operation, 'edges': edges,
            'source_dependence_rationale': dependence,
            'common_sense_sufficiency': 'NOT_CLAIMED; independent reviewer must check the source-specific comparisons.',
            'editorial_ordinal_cues_added': False,
            'public_canonical_label_order_is_source_order': sequence == list(c['items']),
        }
        self.candidates.append(c)

    def shortfall(self, slot, reason, refs):
        self.shortfalls[slot] = {'status': 'MATERIAL_SHORTFALL', 'reason': reason, 'evidence': self.ev(refs)}

    def export(self):
        bound, binding_audit = bind_candidates(self.candidates, BASE / 'SELECTION_FREEZE.json')
        audits = [audit_candidate(c, BASE / 'SELECTION_FREEZE.json') for c in bound]
        ledger = []
        for s in self.root['slot_plan']:
            row = {'root_id': self.root['root_id'], 'slot': s['slot'], 'candidate_id': f'PDF1-{self.number}-{s["slot"]}', 'task_operation': s['task_operation'], 'planned_gold_count': s.get('gold_count')}
            if s['slot'] in self.shortfalls:
                row.update(self.shortfalls[s['slot']])
            else:
                c = next(c for c in bound if c['candidate_id'] == row['candidate_id'])
                row.update(status='AUTHOR_DRAFT_PENDING_INDEPENDENT_REVIEW', actual_gold_count=len(c['gold']) if c['question_type'] == 'multi_select' else None)
            ledger.append(row)
        for name, value in [
            (f'CANDIDATES_{self.number}_PRIVATE_DRAFT.json', bound),
            (f'SLOT_LEDGER_{self.number}_DRAFT.json', ledger),
            (f'MECHANICAL_GATE_{self.number}.json', {'candidate_count': len(bound), 'binding_audit': binding_audit, 'audits': audits, 'independent_semantic_review': False}),
        ]:
            with (OUT / name).open('x') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
                f.write('\n')
        print(json.dumps({'root': self.number, 'candidates': len(bound), 'mechanical_pass': sum(a['mechanical_pass'] for a in audits), 'shortfalls': self.shortfalls}, ensure_ascii=False))

