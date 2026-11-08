"""Offline V2 authoring: event discovery -> answer freeze -> question -> blind review.

Codex supplies semantic extraction; this module never calls a model or network.
Path results are exact only relative to the submitted, reviewed proposition ledger.
They are not a logical proof of unrestricted natural-language entailment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
try:
    from . import base_qa_pipeline_v1 as v1
except ImportError:
    import base_qa_pipeline_v1 as v1

VERSION = 'base_qa_generation_v2'
PROMPT_DIR = Path(__file__).with_name('base_qa_prompts_v2')
CHECKS = (*v1.CHECKS, 'event_identity', 'source_scope', 'alternative_paths', 'extraction_completeness')
POLICY = dict(version=VERSION, challenger='Codex direct semantic work',
              reviewer='isolated same-model blind review; not independent-model verification',
              model_calls_enabled=False, solver_enabled=False, training_enabled=False)
fp = v1.legacy.fingerprint
read = v1.read
write_new = v1.write_new

def instruction(stage):
    return (PROMPT_DIR / (stage + '.txt')).read_text()

def source_manifest(source, binding=None):
    """Unknown admission is false; provenance remains visible in every packet."""
    version = (binding or {}).get('sha256') or fp(source)
    rows = []
    for t in source['turns']:
        admission = t.get('material_admission', {})
        reviewed = admission.get('review_status') == 'PASS'
        explicit = admission.get('formal_qa_allowed') is True
        traceable = all(isinstance(admission.get(k), str) and admission[k].strip()
                        for k in ('review_record', 'reviewer_id'))
        provenance = t.get('provenance_type', t.get('origin', 'unknown'))
        eligible = reviewed and explicit and traceable and isinstance(provenance, str) and bool(provenance.strip()) and provenance.strip().lower() not in ('unknown', 'unspecified', 'unverified')
        rows.append(dict(source_id=t.get('source_id', source.get('source_id', source['root_id'])),
                         turn_id=t['turn_ref'], provenance_type=provenance,
                         source_version=t.get('source_version', version),
                         material_review_status=admission.get('review_status', 'UNVERIFIED'),
                         material_review_record=admission.get('review_record'),
                         material_reviewer_id=admission.get('reviewer_id'),
                         formal_qa_allowed=eligible))
    return dict(root_id=source['root_id'], source_binding=binding,
                source_status=source.get('source_status', 'UNSPECIFIED'),
                source_version=version, warnings=copy.deepcopy(source.get('warnings', [])),
                warning_policy='source limits visible to reviewer; not factual premises; no Solver views generated',
                provenance_lineage={k: copy.deepcopy(source[k]) for k in ('baseline_sha256', 'input_source_sha256', 'source_item_sha256', 'schema_version') if k in source},
                turns=rows, eligibility_policy='explicit traceable material admission only; unknown is research-only')

def source_view(source, manifest):
    by_ref = {r['turn_id']: r for r in manifest['turns']}
    return dict(root_id=source['root_id'], turns=[dict(
        **{k: t[k] for k in ('turn_ref', 'speaker', 'text', 'scene_ref', 'session_ref') if k in t},
        provenance=by_ref[t['turn_ref']]) for t in source['turns']])

def evidence(source, refs):
    return [{k: t[k] for k in ('turn_ref', 'speaker')} | {'quote': t['text']}
            for t in source['turns'] if t['turn_ref'] in refs]

def _refs(evs, source, errors, label):
    return v1._evidence(evs, {t['turn_ref']: t for t in source['turns']}, errors, label)

def validate_discovery(d, source):
    errors = []
    if d.get('status') == 'NO_SAFE_RELATION':
        return [] if v1.text(d.get('reason')) else ['NO_SAFE_RELATION requires a reason']
    events = d.get('events', [])
    ids = [e.get('event_id') for e in events]
    if not events or not all(v1.text(i) for i in ids) or len(ids) != len(set(ids)):
        errors.append('event IDs must be unique and nonempty')
    source_refs = {t['turn_ref'] for t in source['turns']}
    for event in events:
        for k in ('actors', 'action', 'objects', 'time', 'quantities', 'status_history', 'mentions'):
            if k not in event: errors.append(event.get('event_id', '?') + ': missing ' + k)
        mentions = event.get('mentions', [])
        if not mentions: errors.append('event requires source-bound mentions')
        for i, mention in enumerate(mentions):
            refs = _refs(mention.get('evidence'), source, errors, 'event mention')
            kind = mention.get('identity_basis')
            if not i and kind != 'introduction': errors.append('first event mention must introduce its identity')
            if kind not in ('introduction', 'explicit_reference', 'local_continuation'):
                errors.append('unknown event identity basis')
            if i and kind == 'introduction': errors.append('new introduction cannot silently merge an event')
            if kind == 'explicit_reference' and not v1.text(mention.get('identity_rationale')):
                errors.append('explicit identity reference requires rationale')
            if not refs: continue
            by_ref = {t['turn_ref']: t for t in source['turns']}
            first = mentions[0].get('evidence', [])
            first_refs = [e.get('turn_ref') for e in first]
            scenes = {by_ref[r].get('scene_ref') or by_ref[r].get('session_ref') for r in first_refs if r in by_ref}
            current = {by_ref[r].get('scene_ref') or by_ref[r].get('session_ref') for r in refs}
            if i and current != scenes and kind != 'explicit_reference':
                errors.append('cross-scene event identity requires explicit reference evidence')
    candidates = d.get('relations', [])
    if not isinstance(candidates, list): return errors + ['relations must be a list']
    rids = [r.get('relation_id') for r in candidates]
    if not all(v1.text(i) for i in rids) or len(rids) != len(set(rids)):
        errors.append('relation IDs must be unique')
    for r in candidates:
        for k in ('endpoints', 'relation_type', 'derivation_basis', 'information_need',
                  'natural_value', 'uncertainties', 'decision', 'evidence'):
            if k not in r: errors.append('relation discovery missing ' + k)
        if not set(r.get('endpoints', [])) <= set(ids): errors.append('unknown relation endpoint')
        _refs(r.get('evidence'), source, errors, 'discovery relation')
        if r.get('decision') not in ('SELECT', 'REJECT', 'HOLD'): errors.append('invalid discovery decision')
        if r.get('decision') == 'SELECT' and (r.get('natural_value') != 'PASS' or r.get('uncertainties') != []):
            errors.append('selected relation must have natural value and no unresolved uncertainty')
    coverage = [ref for row in d.get('source_scan', []) for ref in row.get('turn_refs', [])]
    if set(coverage) != source_refs or len(coverage) != len(set(coverage)):
        errors.append('source_scan must account for every source turn exactly once')
    if any(not v1.text(row.get('rationale')) for row in d.get('source_scan', [])):
        errors.append('source_scan needs relevance/exclusion rationale')
    return errors

def _minimal(sets, limit=512):
    unique = set(map(frozenset, sets))
    if len(unique) > limit: raise ValueError('support path limit exceeded; HOLD without truncating')
    return sorted([s for s in unique if not any(t < s for t in unique)], key=lambda s: (len(s), sorted(s)))

def proof_paths(relation):
    """Enumerate AND/OR support antichains; never choose just the author path."""
    paths = {}
    for f in relation['facts']:
        paths[f['fact_id']] = _minimal([s['turn_refs'] for s in f['supports']])
    for s in relation['inference_steps']:
        alternatives = s.get('input_alternatives', [s['inputs']])
        merged = []
        for inputs in alternatives:
            count = 1
            for i in inputs: count *= len(paths[i])
            if count > 4096: raise ValueError('support product limit exceeded; HOLD')
            for combination in itertools.product(*(paths[i] for i in inputs)):
                merged.append(frozenset().union(*combination))
        paths[s['step_id']] = _minimal(merged)
    return paths[relation['inference_steps'][-1]['step_id']]

def evidence_audit(relation, source, manifest):
    errors = []
    for f in relation.get('facts', []):
        supports = f.get('supports', [])
        if not supports: errors.append('each fact requires alternative source support sets')
        for support in supports:
            refs = _refs(support.get('evidence'), source, errors, 'support')
            if support.get('turn_refs') != refs: errors.append('support references must bind exact evidence')
            if not v1.text(support.get('entailment_rationale')): errors.append('support requires entailment rationale')
        if f.get('evidence_turn_ids') not in [support.get('turn_refs') for support in supports]:
            errors.append('primary fact evidence must match one declared support set')
    if errors: return dict(errors=errors, mechanical_pass=False)
    try:
        paths = proof_paths(relation)
    except (KeyError, TypeError, ValueError) as e:
        return dict(errors=['invalid proof paths: ' + str(e)], mechanical_pass=False)
    cert = relation.get('evidence_certificate', {})
    primary = frozenset(cert.get('primary_turn_refs', []))
    if primary not in paths: errors.append('primary evidence is not a minimal sufficient set in the ledger')
    source_refs = {t['turn_ref'] for t in source['turns']}
    scan = cert.get('full_source_scan', [])
    covered = [r for row in scan for r in row.get('turn_refs', [])]
    if set(covered) != source_refs or len(covered) != len(set(covered)):
        errors.append('evidence alternative search must cover entire source exactly once')
    for row in scan:
        if row.get('classification') not in ('support', 'possible_alternative', 'irrelevant') or not v1.text(row.get('rationale')):
            errors.append('every source scan row needs relevance and rationale')
        if row.get('classification') == 'possible_alternative' and row.get('resolution') != 'excluded_with_evidence':
            errors.append('unresolved alternative path or event identity')
        if row.get('classification') == 'possible_alternative':
            _refs(row.get('resolution_evidence'), source, errors, 'alternative exclusion')
    for s in relation['inference_steps']:
        alternatives = s.get('input_alternatives', [s['inputs']])
        known = {f['fact_id'] for f in relation['facts']}
        for earlier in relation['inference_steps']:
            if earlier['step_id'] == s['step_id']: break
            known.add(earlier['step_id'])
        if not alternatives or any(not a or not set(a) <= known for a in alternatives):
            errors.append('alternative inputs must use preceding known nodes')
    single_removals = [dict(removed_turn_ref=r, still_sufficient_paths=[sorted(p) for p in paths if r not in p],
                           necessary_in_ledger=not any(r not in p for p in paths)) for r in sorted(primary)]
    groups = []
    for g in relation['necessity_groups']:
        removed = set(g['turn_refs'])
        surviving = [sorted(p) for p in paths if not p & removed]
        alone = [sorted(p) for p in paths if p <= removed]
        groups.append(dict(group_id=g['group_id'], removed_turn_refs=g['turn_refs'],
                           surviving_paths=surviving, sufficient_alone_paths=alone,
                           necessary_in_ledger=not surviving, insufficient_alone_in_ledger=not alone))
        if surviving or alone: errors.append('joint premise necessity fails: ' + g['group_id'])
    claim = cert.get('necessity_claim')
    if claim not in ('individual_turns', 'premise_groups'): errors.append('explicit necessity claim required')
    if claim == 'individual_turns' and any(not r['necessary_in_ledger'] for r in single_removals):
        errors.append('individual evidence necessity contradicted by alternative sufficient path')
    declared = {frozenset(p) for p in cert.get('all_minimal_support_sets', [])}
    if declared != set(paths): errors.append('certificate must disclose every computed minimal support set')
    used = set().union(*paths) if paths else set()
    used.update(e['turn_ref'] for link in relation.get('identity_links', []) for e in link.get('evidence', []))
    used.update(e['turn_ref'] for row in scan for e in row.get('resolution_evidence', []))
    by_ref = {r['turn_id']: r for r in manifest['turns']}
    ineligible = sorted(r for r in used if not by_ref.get(r, {}).get('formal_qa_allowed'))
    return dict(mechanical_pass=not errors, errors=errors, minimal_support_sets=[sorted(p) for p in paths],
                alternative_sufficient_path_count=max(0, len(paths)-1),
                individual_deletions=single_removals, premise_group_deletions=groups,
                formal_evidence_eligible=not ineligible, ineligible_turn_refs=ineligible,
                semantic_completeness_proved=False,
                basis='exact relative to author-extracted ledger; full-source independent semantic review required',
                solver_ablation_is_not_logical_proof=True)

def relation_for_v1(relation):
    graph = copy.deepcopy(relation)
    for step in graph.get('inference_steps', []):
        step['inputs'] = list(dict.fromkeys(i for branch in step.get('input_alternatives', [step['inputs']]) for i in branch))
    return graph

def validate_candidate(candidate, relation, source):
    audit = v1.validate_candidate(candidate, relation, source)
    graph_error = 'all facts must feed the final inference; independent fact bundles rejected'
    if graph_error in audit['errors'] and graph_error not in v1.validate_relation(relation_for_v1(relation), source):
        audit['errors'].remove(graph_error)
    refs = {t['turn_ref'] for t in source['turns']}
    for key, row in candidate.get('item_analysis', {}).items():
        if key in candidate.get('gold', []) or candidate.get('question_type') == 'ordering': continue
        path = row.get('error_path_turn_refs')
        if not isinstance(path, list) or not path or not all(isinstance(r, str) and r in refs for r in path):
            audit['errors'].append('distractor error path must reference known source turns: ' + key)
        if not v1.text(row.get('error_path_type')):
            audit['errors'].append('distractor must identify a source error path: ' + key)
    audit['mechanical_pass'] = not audit['errors']
    return audit

def validate_target(relation, discovery, source, manifest):
    errors = validate_discovery(discovery, source) + v1.validate_relation(relation_for_v1(relation), source)
    selected = next((r for r in discovery.get('relations', []) if r['relation_id'] == relation.get('relation_id')), None)
    if not selected or selected.get('decision') != 'SELECT': errors.append('target must bind a selected discovered relation')
    if relation.get('discovery_sha256') != fp(discovery): errors.append('discovery binding drift')
    if selected and relation.get('information_need') != selected.get('information_need'):
        errors.append('target information need drift from selected discovery')
    if not v1.text(relation.get('information_need')): errors.append('freeze natural information need before type/options')
    if any(k in relation for k in ('options', 'gold', 'question_type', 'question')):
        errors.append('target freeze precedes wording, question type and option construction')
    if relation.get('uncertainties') != []: errors.append('unresolved target uncertainty')
    ids = {e['event_id'] for e in discovery.get('events', [])}
    endpoint_ids = set(selected.get('endpoints', [])) if selected else set()
    if relation.get('identity_requirement') not in ('same_event', 'explicit_cross_event_relation'):
        errors.append('explicit event identity requirement required')
    for fact in relation.get('facts', []):
        if not fact.get('event_ids') or not set(fact['event_ids']) <= ids: errors.append('fact needs known event identity binding')
        if not set(fact.get('event_ids', [])) <= endpoint_ids: errors.append('fact event drifts from selected relation endpoints')
    if relation.get('identity_requirement') == 'same_event':
        if len({eid for f in relation.get('facts', []) for eid in f.get('event_ids', [])}) != 1:
            errors.append('same_event facts must bind one explicit event ID')
        bindings = relation.get('identity_links', [])
        if not bindings: errors.append('same-event derivation requires an explicit identity link')
        for link in bindings:
            if link.get('status') != 'confirmed': errors.append('unresolved event identity link')
            if link.get('basis') != 'explicit_reference': errors.append('similar actor/action/time is not event identity')
            if not set(link.get('event_ids', [])) <= ids or not link.get('event_ids'): errors.append('unknown identity endpoint')
            _refs(link.get('evidence'), source, errors, 'identity link')
            if not v1.text(link.get('rationale')): errors.append('identity link needs a source-based rationale')
    audit = evidence_audit(relation, source, manifest) if not errors else {'errors': []}
    errors += audit['errors']
    return dict(mechanical_pass=not errors, errors=errors, evidence_audit=audit, semantic_quality_proved_by_code=False)

def style_audit(candidate):
    counts = {k: len(v.split()) for k, v in candidate['options'].items()}
    flags = []
    if candidate['question_type'] != 'ordering' and min(counts.values()) and max(counts.values()) > 2 * min(counts.values()):
        flags.append('OPTION_LENGTH_IMBALANCE_REQUIRES_REVIEW')
    return dict(option_word_counts=counts, flags=flags, semantic_shortcuts_proved_absent=False)

def blind_packet(candidate, source, manifest):
    public, private = v1.prepare_display(candidate, 'v2-independent-blind')
    return dict(stage='INDEPENDENT_SOLVE', instruction=instruction('blind'),
                candidate_id=candidate['candidate_id'], reviewer_mode=POLICY['reviewer'],
                source=source_view(source, manifest), source_scope=manifest,
                question_type=candidate['question_type'], question=candidate['question'],
                options={private['display_to_canonical'][k]: value for k, value in public['options'].items()})

def _guarded_validator(function, kind):
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError) as exc:
            errors = ['malformed input; HOLD: ' + str(exc)]
            return errors if kind == 'list' else dict(mechanical_pass=False, errors=errors, semantic_quality_proved_by_code=False)
    return guarded

validate_discovery = _guarded_validator(validate_discovery, 'list')
evidence_audit = _guarded_validator(evidence_audit, 'dict')
validate_target = _guarded_validator(validate_target, 'dict')
validate_candidate = _guarded_validator(validate_candidate, 'dict')

def implementation_bindings():
    return [v1.legacy.file_binding(p) for p in (Path(__file__), Path(v1.__file__), Path(v1.legacy.__file__), *sorted(PROMPT_DIR.glob('*.txt')))]

def freeze(run, name, value):
    write_new(run / (name + '.json'), value)
    write_new(run / (name + '_FREEZE.json'), dict(sha256=fp(value), version=VERSION))

def frozen(run, name):
    value = read(run / (name + '.json'))
    if read(run / (name + '_FREEZE.json'))['sha256'] != fp(value): raise ValueError('frozen artifact drift: ' + name)
    return value

def context(run):
    ctx = frozen(run, 'INPUT')
    if ctx['implementation_bindings'] != implementation_bindings(): raise ValueError('implementation drift; create new versioned run')
    source = v1.source_from(ctx['source_binding'], ctx['root_id'])
    if ctx['source_scope'] != source_manifest(source, ctx['source_binding']): raise ValueError('source manifest drift')
    return ctx, source

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--source', type=Path, required=True); prep.add_argument('--run-dir', type=Path, required=True)
    for name in ('freeze-discovery', 'freeze-target', 'freeze-candidate', 'freeze-blind', 'finalize'):
        cmd = sub.add_parser(name); cmd.add_argument('--run-dir', type=Path, required=True); cmd.add_argument('--input', type=Path, required=True)
    a = p.parse_args(argv); run = a.run_dir
    if a.command == 'prepare':
        source = read(a.source); binding = v1.legacy.file_binding(a.source); manifest = source_manifest(source, binding)
        v1.source_from(binding, source['root_id'])
        run.mkdir(parents=True, exist_ok=False)
        freeze(run, 'INPUT', dict(policy=POLICY, root_id=source['root_id'], source_binding=binding, source_scope=manifest, implementation_bindings=implementation_bindings()))
        write_new(run/'DISCOVERY_PACKET.json', dict(instruction=instruction('discovery'), source=source_view(source, manifest), source_scope=manifest))
    else:
        ctx, source = context(run); manifest = ctx['source_scope']; value = read(a.input)
        if a.command == 'freeze-discovery':
            errors = validate_discovery(value, source)
            if errors: raise ValueError(errors)
            freeze(run, 'DISCOVERY', value)
            if value.get('status') == 'NO_SAFE_RELATION':
                write_new(run/'DECISION.json', dict(decision='NO_SAFE_RELATION', reason=value['reason'], solver_run=False))
                return 0
            write_new(run/'TARGET_PACKET.json', dict(instruction=instruction('target'), source=source_view(source, manifest), source_scope=manifest, discovery=value, discovery_sha256=fp(value)))
        elif a.command == 'freeze-target':
            d = frozen(run, 'DISCOVERY')
            if d.get('status') == 'NO_SAFE_RELATION': raise ValueError('NO_SAFE_RELATION has no target')
            if value.get('source_scope_sha256') != fp(manifest): raise ValueError('target source scope binding missing')
            audit = validate_target(value, d, source, manifest)
            if not audit['mechanical_pass']: raise ValueError(audit['errors'])
            freeze(run, 'TARGET', value); freeze(run, 'LOGICAL_AUDIT', audit)
            write_new(run/'AUTHOR_PACKET.json', dict(instruction=instruction('author'), source=source_view(source, manifest), source_scope=manifest, relation=value, relation_sha256=fp(value)))
        elif a.command == 'freeze-candidate':
            target = frozen(run, 'TARGET'); audit = validate_candidate(value, target, source)
            if not audit['mechanical_pass']: raise ValueError(audit['errors'])
            if value.get('target_sha256') != fp(target): raise ValueError('target freeze binding missing')
            if value.get('source_scope_sha256') != fp(manifest): raise ValueError('candidate source scope binding missing')
            for key, row in value['item_analysis'].items():
                if value['question_type'] != 'ordering' and key not in value['gold'] and (not v1.text(row.get('error_path_type')) or not row.get('error_path_turn_refs')):
                    raise ValueError('distractor must bind a real source error path')
            freeze(run, 'CANDIDATE', value); write_new(run/'CANDIDATE_AUDIT.json', dict(validation=audit, style=style_audit(value)))
            freeze(run, 'BLIND_PACKET', blind_packet(value, source, manifest))
        elif a.command == 'freeze-blind':
            c = frozen(run, 'CANDIDATE'); packet = frozen(run, 'BLIND_PACKET')
            errors = v1.validate_blind(c, source, packet, value)
            if value.get('reviewer_mode') != 'same_model_blind': errors.append('same-model review must be honestly labeled')
            if value.get('source_scope_sha256') != fp(manifest): errors.append('reviewer must acknowledge complete source scope')
            receipt = value.get('isolation_receipt', {})
            if receipt.get('context_mode') != 'fresh_context_no_history' or receipt.get('author_material_visible') is not False or not v1.text(receipt.get('context_id')):
                errors.append('external isolation receipt missing or unresolved')
            freeze(run, 'BLIND_SOLUTION', value)
            freeze(run,'COMPARISON_PACKET', dict(instruction=instruction('comparison'), source=source_view(source, manifest), source_scope=manifest, candidate=c, relation=frozen(run,'TARGET'), discovery=frozen(run,'DISCOVERY'), blind_solution=value, blind_errors=errors, required_checks=CHECKS, blind_packet=packet, blind_packet_sha256=fp(packet), display_order=list(packet['options']), logical_audit=frozen(run,'LOGICAL_AUDIT')))
        else:
            c=frozen(run,'CANDIDATE'); target=frozen(run,'TARGET'); blind=frozen(run,'BLIND_SOLUTION'); packet=frozen(run,'BLIND_PACKET')
            result=v1.validate_review(c,target,source,packet,blind,value); errors=result['errors'][:]
            if packet != blind_packet(c,source,manifest): errors.append('blind packet differs from canonical public reconstruction')
            graph_error = 'all facts must feed the final inference; independent fact bundles rejected'
            if graph_error in errors and graph_error not in validate_candidate(c,target,source)['errors']: errors.remove(graph_error)
            errors += frozen(run,'COMPARISON_PACKET')['blind_errors']
            repeated = validate_target(target,frozen(run,'DISCOVERY'),source,manifest)
            errors += repeated['errors']
            if repeated != frozen(run,'LOGICAL_AUDIT'): errors.append('logical audit drift')
            result['admitted'] = not errors
            for check in CHECKS:
                row=value.get('checks',{}).get(check,{})
                if not isinstance(row,dict): row={}
                if row.get('decision')!='PASS' or not v1.text(row.get('rationale')): errors.append('unresolved semantic check: '+check)
            if blind.get('reviewer_mode')!='same_model_blind' or blind.get('source_scope_sha256')!=fp(manifest): errors.append('review scope/mode unresolved')
            logical=repeated['evidence_audit']
            if not logical['formal_evidence_eligible']: errors.append('source material not formally admitted; research-only HOLD')
            all_candidate_refs = {e['turn_ref'] for row in c['item_analysis'].values() for e in row.get('evidence', [])}
            all_candidate_refs.update(r for row in c['item_analysis'].values() for r in row.get('error_path_turn_refs', []))
            all_candidate_refs.update(e['turn_ref'] for edge in c.get('ordering_edges', []) for e in edge.get('evidence', []))
            all_candidate_refs.update(e['turn_ref'] for e in blind.get('evidence', []))
            eligible = {r['turn_id'] for r in manifest['turns'] if r['formal_qa_allowed']}
            if not all_candidate_refs <= eligible: errors.append('candidate evidence or distractor path not formally admitted')
            decision=dict(decision='HOLD' if errors else 'PASS_STATIC_REVIEW', errors=errors,
                          research_semantic_pass=result['admitted'] and all(value.get('checks',{}).get(k,{}).get('decision')=='PASS' for k in CHECKS),
                          formal_qa_admitted=not errors, reviewer_mode='same_model_blind',
                          independent_model_cross_validation=False, solver_run=False, empirical_quality_established=False)
            freeze(run,'COMPARISON_REVIEW',value); write_new(run/'DECISION.json',decision)
            print(json.dumps(decision,ensure_ascii=False)); return 0 if not errors else 2
    print(json.dumps(dict(stage=a.command,run_dir=str(run),model_calls=0,solver_calls=0))); return 0

if __name__ == '__main__':
    raise SystemExit(main())
