#!/usr/bin/env python3
"""RAM-only reversible serialization adapters for future frozen question drivers.

No prompts, author data, decisions, keys, evidence or semantic budgets change.
The frozen driver files stay byte-identical. All actual calls remain their own
fresh Codex sessions. The parent must schedule wrapper families separately.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import question_driver as q
import question_candidate_shape_continuation as shape

EXPECTED_DRIVER = '33ab6b7350bbaa1c9003d66f9c15a7c0c87fd855cf373ae3810664ab4701886d'
EXPECTED_SHAPE = '70b44b835a314fc9a89c2fb88533f7e03dbb0c6d2ca047b22bd58b9613ad3796'
DRIVERS = {
    'v3': ('question_plan_schema_continuation_v3', '083c968fad2c28cde34efd1b90f87ba97945020d9bc020d65b2e5f387c339b66'),
    'shape': ('question_candidate_shape_continuation', EXPECTED_SHAPE),
    'unused': ('question_unfilled_slot_continuation', '0bf0fedd04e85d23edc1df0d013ef0f8f82c2303d2ffd885a6fafa315621ea34'),
}
ALIASES = {'removal_reason': 'rationale', 'r03_turn_refs': 'turn_refs'}


def normalize_g_verifier_validation_payload(payload: Any, role: str):
    """Return a validation-only copy plus exact false-attestation paths."""
    copied = copy.deepcopy(payload)
    if role != 'content_verifier' or not isinstance(copied, dict):
        return copied, []
    changes = []
    cases = copied.get('review_cases')
    if isinstance(cases, list):
        for index, case in enumerate(cases):
            saved = case.get('saved_independent_gold') if isinstance(case, dict) else None
            if isinstance(saved, dict) and saved.get('solver_results_read') is False:
                del saved['solver_results_read']
                changes.append({'path': ['review_cases', index, 'saved_independent_gold', 'solver_results_read'],
                                'value': False, 'purpose': 'explicit required non-access attestation'})
    restored = copy.deepcopy(copied)
    for change in changes:
        restored['review_cases'][change['path'][1]]['saved_independent_gold']['solver_results_read'] = False
    if restored != payload:
        raise RuntimeError('G validation allowance changed an unrelated payload value')
    return copied, changes


def isolated_g_validation_copy(prompt: str, role: str):
    """Permit only the contract's explicit false isolation attestation.

    The actual prompt is never changed. This creates a validation-only copy so
    the existing recursive material-key ban does not mistake an assurance of
    non-access for solver observations. True or other values remain rejected.
    """
    if role != 'content_verifier':
        return prompt, []
    marker = next((m for m in ('FROZEN_INPUTS:\n', 'AUTHORIZED_INPUTS:\n', 'REVIEW_INPUT:\n') if m in prompt), None)
    if marker is None: return prompt, []
    start = prompt.index(marker) + len(marker)
    payload, end = json.JSONDecoder().raw_decode(prompt[start:])
    if not isinstance(payload, dict) or prompt[start + end:].strip(): return prompt, []
    copied, changes = normalize_g_verifier_validation_payload(payload, role)
    if not changes: return prompt, []
    return prompt[:start] + q.prompt_json(copied), changes


def normalize_blind_aliases(raw: Any) -> tuple[Any, list[dict[str, Any]]]:
    value = copy.deepcopy(raw)
    changes = []
    rows = q.extract_list(value, 'reviews', 'answers')
    for index, row in enumerate(rows):
        r03 = row.get('r03_necessity')
        # Deliberately do not infer missing values, merge mixed schemas, discard
        # extra keys, or resolve even equal-valued alias/canonical duplicates.
        if not isinstance(r03, dict) or set(r03) != {'decision', 'removal_reason', 'r03_turn_refs'}:
            continue
        if (not isinstance(r03['removal_reason'], str) or not r03['removal_reason'].strip()
                or not isinstance(r03['r03_turn_refs'], list) or not r03['r03_turn_refs']
                or not all(isinstance(ref, str) and ref for ref in r03['r03_turn_refs'])):
            continue
        normalized = {ALIASES.get(key, key): copy.deepcopy(v) for key, v in r03.items()}
        row['r03_necessity'] = normalized
        changes.append({'review_index': index, 'question_key': row.get('question_key'),
            'field': 'r03_necessity', 'mapping': dict(ALIASES),
            'authored_original_value': copy.deepcopy(r03), 'normalized_value': copy.deepcopy(normalized),
            'decision_and_all_values_changed': False})
    reversed_value = copy.deepcopy(value)
    reversed_rows = q.extract_list(reversed_value, 'reviews', 'answers')
    reverse = {v: k for k, v in ALIASES.items()}
    for change in reversed(changes):
        row = reversed_rows[change['review_index']]
        restored = {reverse.get(key, key): copy.deepcopy(v) for key, v in row['r03_necessity'].items()}
        if restored != change['authored_original_value']:
            raise RuntimeError('Blind schema normalization was not exactly reversible')
        row['r03_necessity'] = restored
    if reversed_value != raw:
        raise RuntimeError('Blind schema adapter changed an unrelated authored field')
    return value, changes


def _context(allowed_names: set[str]):
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        while frame:
            if frame.f_code.co_name in allowed_names:
                local = frame.f_locals
                directory = local.get('work', local.get('out'))
                if isinstance(directory, Path):
                    directory.resolve().relative_to(BASE)
                    return directory, dict(local)
            frame = frame.f_back
        return None
    finally:
        del frame


def _publish_normalization(directory: Path, kind: str, suffix: str, raw: Any,
                           normalized: Any, changes: list[dict[str, Any]],
                           original_path: Path | None, launch_binding: dict[str, Any] | None):
    tag = ('_' + suffix) if suffix else ''
    prefix = 'MECHANICAL_REVIEW_SCHEMA' if kind == 'blind' else 'MECHANICAL_CANDIDATE_SCHEMA'
    snapshot = directory / f'{prefix}_INPUT{tag}.json'
    q.publish(snapshot, normalized)
    original_binding = q.binding(original_path) if original_path and original_path.exists() else None
    raw_matches_saved = q.read(original_path) == raw if original_binding else None
    report = {'schema_version': 'fgr20_reversible_runtime_schema_normalization_v1',
        'kind': kind, 'stage_suffix': suffix or 'initial', 'adapter_binding': q.binding(__file__),
        'launcher_receipt_binding': launch_binding, 'authored_input_binding': original_binding,
        'input_object_matches_saved_authored_response': raw_matches_saved,
        'input_object_canonical_sha256': q.digest(raw),
        'normalized_input_binding': q.binding(snapshot), 'normalized_object_canonical_sha256': q.digest(normalized),
        'changes': changes, 'exact_round_trip_verified': True,
        'question_text_gold_evidence_decisions_and_all_values_changed': False,
        'semantic_calls_or_revision_budget_consumed_by_adapter': 0,
        'semantic_admission_asserted': False,
        'earlier_raw_validation_and_review_files_overwritten': False}
    q.publish(directory / f'{prefix}_NORMALIZATION{tag}.json', report)
    return report


def install(launch_binding: dict[str, Any] | None = None):
    if hashlib.sha256(Path(q.__file__).read_bytes()).hexdigest() != EXPECTED_DRIVER:
        raise RuntimeError('Frozen question driver file drift')
    if hashlib.sha256(Path(shape.__file__).read_bytes()).hexdigest() != EXPECTED_SHAPE:
        raise RuntimeError('Frozen candidate-shape normalizer file drift')
    if getattr(q.normalize_blind, '_fgr20_review_schema_adapter', False):
        raise RuntimeError('RAM review schema adapter is already installed')
    old_blind, old_candidates, old_g_validator = q.normalize_blind, q.normalize_candidates, q.validate_g_prompt
    def adapted_blind(response, candidates, source):
        context = _context({'review_batch'})
        if context is None:
            raise RuntimeError('Blind runtime normalization requires a concrete review-stage output directory')
        directory, local = context
        suffix = local.get('suffix', '')
        if not isinstance(suffix, str): raise RuntimeError('Invalid review suffix')
        normalized, changes = normalize_blind_aliases(response)
        raw_path = directory / f"BLIND_GOLD{('_' + suffix) if suffix else ''}.json"
        _publish_normalization(directory, 'blind', suffix, response, normalized, changes, raw_path, launch_binding)
        # Frozen normalizer still checks every rationale, ref, Gold, edge and
        # attestation, and can HOLD/FAIL. Alias conversion grants no admission.
        return old_blind(copy.deepcopy(normalized), candidates, source)
    def adapted_candidates(response, *args, **kwargs):
        # Do not change saved-lineage checks or inspection calls. Only the actual
        # frozen production process/revision path receives this future adapter.
        context = _context({'process_root_arm', 'run_one', 'normalize'})
        if context is None:
            return old_candidates(response, *args, **kwargs)
        directory, local = context
        revision_thread = kwargs.get('revision_thread') or (args[6] if len(args) > 6 else None)
        suffix = 'revision_01' if revision_thread else 'initial'
        normalized, changes = shape.normalizer(response)
        raw_path = directory / ('CANDIDATES_REVISION_01_RESPONSE.json' if revision_thread else 'CANDIDATES_DRAFT_RESPONSE.json')
        _publish_normalization(directory, 'candidates', suffix, response, normalized, changes, raw_path, launch_binding)
        return old_candidates(normalized, *args, **kwargs)
    def adapted_g_validator(prompt, role, root):
        validation_copy, changes = isolated_g_validation_copy(prompt, role)
        # All original top-level allowlists and every other nested material-key
        # rejection still run. The original prompt remains the call argument.
        old_g_validator(validation_copy, role, root)
        if changes:
            context = _context({'review_batch'})
            if context is None:
                raise RuntimeError('G verifier attestation allowance lacks a concrete review stage')
            directory, local = context
            suffix = local.get('suffix', '')
            tag = ('_' + suffix) if suffix else ''
            q.publish(directory / f'G_VERIFIER_ISOLATION_ATTESTATION_VALIDATION{tag}.json', {
                'schema_version': 'fgr20_g_false_nonaccess_attestation_validation_v1',
                'adapter_binding': q.binding(__file__), 'launcher_receipt_binding': launch_binding,
                'root_id': root, 'role': role, 'stage_suffix': suffix or 'initial',
                'actual_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'validation_only_copy_sha256': hashlib.sha256(validation_copy.encode()).hexdigest(),
                'exact_allowed_false_attestation_paths': changes, 'actual_prompt_modified': False,
                'other_g_whitelists_and_nested_material_bans_unchanged': True,
                'solver_observations_or_feedback_exposed': False})
    adapted_blind._fgr20_review_schema_adapter = True
    adapted_candidates._fgr20_review_schema_adapter = True
    q.normalize_blind, q.normalize_candidates, q.validate_g_prompt = adapted_blind, adapted_candidates, adapted_g_validator
    return {'original_blind_parser_source_sha256': hashlib.sha256(inspect.getsource(old_blind).encode()).hexdigest(),
        'original_candidate_normalizer_source_sha256': hashlib.sha256(inspect.getsource(old_candidates).encode()).hexdigest(),
        'original_g_validator_source_sha256': hashlib.sha256(inspect.getsource(old_g_validator).encode()).hexdigest(),
        'frozen_question_driver_binding': q.binding(q.__file__),
        'frozen_shape_normalizer_binding': q.binding(shape.__file__),
        'RAM_only': True, 'semantic_policy_changed': False}


def self_check():
    work = BASE / 'question_plan_continuation/179186/G'
    raw = q.read(work / 'BLIND_GOLD.json')
    normalized, changes = normalize_blind_aliases(raw)
    assert len(changes) == 8
    assert raw == q.read(work / 'BLIND_GOLD.json')
    for defect in ('extra', 'conflict', 'missing', 'empty'):
        bad = copy.deepcopy(raw)
        r03 = bad['reviews'][0]['r03_necessity']
        if defect == 'extra': r03['unapproved'] = True
        elif defect == 'conflict': r03['rationale'] = 'UNTRUSTED_CONFLICT_FIXTURE'
        elif defect == 'missing': del r03['removal_reason']
        else: r03['removal_reason'] = ''
        out, logs = normalize_blind_aliases(bad)
        assert out['reviews'][0]['r03_necessity'] == r03
        assert all(c['review_index'] != 0 for c in logs)
    sel = BASE / 'SOURCE_SELECTION_FINAL.json'
    q.SOURCE_FREEZE_PATH = BASE / 'SOURCE_FREEZE.json'
    _, sources = q.load_selection(sel)
    source_row = sources['cc-train-episode-179186']
    candidates = q.read(work / 'CANDIDATES_DRAFT.json')
    prior, prior_errors = q.normalize_blind(copy.deepcopy(raw), candidates, source_row['source'])
    after, after_errors = q.normalize_blind(copy.deepcopy(normalized), candidates, source_row['source'])
    assert len(prior_errors) == 8 and all(e.startswith('blind_r03_invalid:') for e in prior_errors), prior_errors
    assert not after_errors, after_errors
    assert {k: v['independent_gold'] for k, v in prior.items()} == {k: v['independent_gold'] for k, v in after.items()}
    before_requests = sorted(str(x) for x in (BASE / 'cli_calls').glob('q20_*/attempt_*/REQUEST.txt'))
    cache = BASE / 'runtime_cache/review_schema_adapter_selfcheck';cache.mkdir(parents=True, exist_ok=True)
    old_blind, old_candidates, old_g_validator = q.normalize_blind, q.normalize_candidates, q.validate_g_prompt
    installation = install()
    try:
        with tempfile.TemporaryDirectory(prefix='ram_',dir=cache) as tmp:
            directory=Path(tmp)
            def review_batch():
                work=directory;suffix='mock_only'
                q.publish(work/'BLIND_GOLD_mock_only.json',raw)
                return q.normalize_blind(copy.deepcopy(raw),candidates,source_row['source'])
            mapped, errors = review_batch()
            assert not errors and len(mapped) == 8
            assert (directory/'MECHANICAL_REVIEW_SCHEMA_NORMALIZATION_mock_only.json').exists()
            # A real saved initial response tests future candidate serialization;
            # this is strictly a temporary engineering process, not an author.
            candidate_raw=q.read(BASE/'question_plan_continuation/179186/F/CANDIDATES_DRAFT_RESPONSE.json')
            plans=q.read(BASE/'question_plan_continuation/179186/F/TASK_PLANS.json')
            def process_root_arm():
                work=directory
                q.publish(work/'CANDIDATES_DRAFT_RESPONSE.json',candidate_raw)
                return q.normalize_candidates(candidate_raw,'F','cc-train-episode-179186','ENGINEERING_MOCK_ONLY',source_row,plans,q.binding(sel))
            restored, failures, logs=process_root_arm()
            assert len(restored)==8 and all(q.basic_solver_shape(c,source_row) for c in restored)
            assert len(q.read(directory/'MECHANICAL_CANDIDATE_SCHEMA_NORMALIZATION_initial.json')['changes'])==16
            clone_checks=[]
            receipt=q.read(work/'BLIND_GOLD_CALL.json')
            for module_name in ('question_plan_schema_continuation_v3','question_plan_schema_continuation'):
                module=importlib.import_module(module_name)
                process,_=module.build_driver()
                namespace=process.__globals__
                assert namespace['normalize_blind'] is q.normalize_blind
                assert namespace['normalize_candidates'] is q.normalize_candidates
                trace=[]
                class MockCalls:
                    def call(self,key,prompt,**kwargs):
                        trace.append(key)
                        if key.startswith('q20_gold_'):
                            return copy.deepcopy(raw),copy.deepcopy(receipt)
                        raise RuntimeError('ENGINEERING_MOCK_VERIFIER_REACHED_WITHOUT_DISPATCH')
                namespace['cc']=MockCalls()
                reviewer=namespace['continuation_review_batch']
                suffix='engineering_'+module_name
                blind_map,_,_,_=reviewer('cc-train-episode-179186',candidates,source_row,sel,directory,suffix)
                assert len(blind_map)==8 and len(trace)==2 and trace[1].startswith('q20_verify_'),trace
                assert q.read(directory/f'BLIND_GOLD_VALIDATION_{suffix}.json')['errors']==[]
                assert q.read(directory/f'BLIND_GOLD_{suffix}.json')==raw
                validation=q.read(directory/f'G_VERIFIER_ISOLATION_ATTESTATION_VALIDATION_{suffix}.json')
                assert len(validation['exact_allowed_false_attestation_paths'])==8
                assert validation['actual_prompt_modified'] is False
                clone_checks.append(module_name)
            for denied in (True, 'false', None):
                payload={'root_id':'cc-train-episode-179186','complete_source_and_warnings':q.source_view(source_row),
                         'review_cases':[{'saved_independent_gold':{'solver_results_read':denied}}]}
                prompt='ENGINEERING_VALIDATION_FIXTURE\nREVIEW_INPUT:\n'+q.prompt_json(payload)
                validation_copy,changes=isolated_g_validation_copy(prompt,'content_verifier')
                assert validation_copy==prompt and not changes
                try:
                    old_g_validator(validation_copy,'content_verifier','cc-train-episode-179186')
                    raise AssertionError('Non-false solver-access value was allowed')
                except ValueError:
                    pass
    finally:
        q.normalize_blind,q.normalize_candidates,q.validate_g_prompt=old_blind,old_candidates,old_g_validator
    after_requests = sorted(str(x) for x in (BASE / 'cli_calls').glob('q20_*/attempt_*/REQUEST.txt'))
    assert all(x in after_requests for x in before_requests)
    return {'status':'PASS','semantic_calls_started':0,'real_blind_records_restored':8,
        'real_candidate_records_restored':8,'real_candidate_reversible_field_changes':16,
        'gold_and_decision_and_evidence_values_changed':False,'raw_and_old_validation_unchanged':True,
        'conflict_extra_missing_empty_aliases_not_repaired':True,'original_errors':prior_errors,
        'normalized_parser_errors':after_errors,'semantic_admission_asserted':False,
        'real_frozen_review_clones_reached_mock_verifier':clone_checks,
        'nonfalse_solver_access_attestations_still_rejected':True,
        'installation':installation}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    launch=sub.add_parser('launch')
    launch.add_argument('--driver',choices=DRIVERS,required=True)
    launch.add_argument('driver_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.command=='self-check':
        print(json.dumps(self_check(),ensure_ascii=False));return 0
    argv=args.driver_args[1:] if args.driver_args[:1]==['--'] else args.driver_args
    if not argv or argv[0]!='run':
        raise RuntimeError('RAM launcher is restricted to future driver run commands')
    if (BASE/'CONTENT_FREEZE.json').exists():
        raise RuntimeError('Content already frozen; no future review or revision may be launched')
    name, expected = DRIVERS[args.driver]
    module=importlib.import_module(name)
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen target driver binding drift')
    key=q.digest({'driver':args.driver,'argv':argv})[:24]
    receipt_path=BASE/'review_schema_adapter/launches'/f'{key}.json'
    receipt={'schema_version':'fgr20_ram_review_schema_launch_v1','driver':args.driver,
        'driver_binding':q.binding(module.__file__),'adapter_binding':q.binding(__file__),
        'original_question_driver_binding':q.binding(q.__file__),'driver_argv':argv,
        'authorized_mutations':'RAM parser/serialization functions only; frozen files unchanged',
        'old_raw_validation_and_review_artifacts_preserved':True,'extra_semantic_budget':0,
        'parent_must_schedule_wrapper_families_as_separate_stages':True}
    q.publish(receipt_path,receipt)
    install(q.binding(receipt_path))
    old_argv=sys.argv
    try:
        sys.argv=[str(Path(module.__file__)),*argv]
        return module.main()
    finally:
        sys.argv=old_argv


if __name__=='__main__':
    raise SystemExit(main())
