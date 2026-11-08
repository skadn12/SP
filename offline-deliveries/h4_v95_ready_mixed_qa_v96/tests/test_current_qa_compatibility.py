"""Offline current-scope/lineage checks; temporary builder only, no QA prepare."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HOME = Path(__file__).resolve().parents[1]
ENGINE = HOME / 'engine'
PYTHON = '/root/autodl-tmp/gxy-selfplay/.venv-local4b/bin/python'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ENGINE))
import batch_scope as scope
import expansion_binding
import fresh_parent
import ordering_plan

spec = importlib.util.spec_from_file_location('v96_compatibility_builder', HOME / 'evidence/build_fresh_shard.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def current_spec():
    rows = []
    for root in scope.EXPECTED_ROOTS:
        short = root.rsplit('-', 1)[-1]
        rows.append({'root_id': root,
                     'public_parent': str(HOME / 'inputs' / f'root{short}_PUBLIC_PARENT.json'),
                     'parent_judgments': str(HOME / 'inputs' / f'root{short}_PARENT_JUDGMENTS.json'),
                     'source_review': str(HOME / 'inputs' / f'root{short}_SOURCE_REVIEW.json'),
                     'expanded_source_provenance': str(HOME / 'evidence/expansion_provenance' / f'root{short}_PROVENANCE.json'),
                     'public_prior_paths': [], 'public_ordering_prior_paths': []})
    return {'version': 'v96_current_ready_shard_spec_v1', 'shard_id': 'offline_compatibility_fixture',
            'roots': rows, 'scene_plans': str(HOME / 'inputs/SCENE_PLANS.json'),
            'plan_path': str(HOME / 'PILOT_PLAN_TEMPLATE.json')}


class CurrentScopeTests(unittest.TestCase):
    def test_exact_two_root_scope_rejects_missing_duplicate_reordered_and_held(self):
        scope.validate_roots(list(scope.EXPECTED_ROOTS))
        for roots in ([], [scope.EXPECTED_ROOTS[0]], list(reversed(scope.EXPECTED_ROOTS)),
                      [scope.EXPECTED_ROOTS[0]] * 2,
                      [scope.EXPECTED_ROOTS[0], 'cc-train-episode-24719']):
            with self.assertRaisesRegex(ValueError, 'exact_two_ready_roots'):
                scope.validate_roots(roots)

    def test_plan_requires_current_scope_budget_and_exact_upstream_bindings(self):
        original = fresh_parent.read(HOME / 'PILOT_PLAN_TEMPLATE.json')
        scope.validate_plan(original)
        patches = [{'version': 'v90_four_root_end_to_end_expansion_mixed_v1'},
                   {'candidate_root_budget': 4}, {'max_generation_http_total': 7},
                   {'max_solver_calls_per_model_total': 229}, {'training_allowed': True},
                   {'upstream_expansion_run': {}}]
        for values in patches:
            with self.assertRaises(ValueError):
                scope.validate_plan({**original, **values})

    def test_official_source_parent_and_whole_turn_plan_inputs_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'SPEC.json'
            save(path, current_spec())
            before = set(Path(tmp).rglob('*'))
            self.assertEqual(builder.check(path), current_spec())
            self.assertEqual(set(Path(tmp).rglob('*')), before)

    def test_builder_check_rejects_unready_source_without_creating_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); value = current_spec()
            review = fresh_parent.read(value['roots'][0]['source_review'])
            review['decision'] = 'HOLD'
            save(home / 'HOLD.json', review)
            value['roots'][0]['source_review'] = str(home / 'HOLD.json')
            save(home / 'SPEC.json', value)
            with patch.object(builder, 'PILOT', home), self.assertRaises(ValueError):
                builder.check(home / 'SPEC.json')
            self.assertFalse((home / 'shards').exists())


class MaterializedLineageTests(unittest.TestCase):
    def test_actual_materialized_sources_are_exact_upstream_bytes(self):
        for row in current_spec()['roots']:
            public = fresh_parent.read(row['public_parent'])
            record = expansion_binding.validate(row['root_id'], scope.bind(public['source_path']),
                                                  row['expanded_source_provenance'])
            self.assertEqual(Path(public['source_path']).read_bytes(), Path(record['accepted_output']['path']).read_bytes())
            self.assertEqual(record['new_turn_count'], 14 if row['root_id'].endswith('25652') else 16)

    def test_repointed_and_incompletely_bound_provenance_rejected(self):
        row = current_spec()['roots'][0]
        public = fresh_parent.read(row['public_parent'])
        original = fresh_parent.read(row['expanded_source_provenance'])
        changes = [lambda r: r.update(root_id=scope.EXPECTED_ROOTS[1]),
                   lambda r: r.update(run='/tmp/not_the_accepted_run'),
                   lambda r: r.update(new_turn_count=999),
                   lambda r: r.update(bound_files=[b for b in r['bound_files'] if not b['path'].endswith('/REPORT.json')]),
                   lambda r: r.update(bound_files=[*r['bound_files'], r['bound_files'][0]])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'PROVENANCE.json'
            for change in changes:
                record = deepcopy(original); change(record); save(path, record)
                with self.assertRaises(ValueError):
                    expansion_binding.validate(row['root_id'], scope.bind(public['source_path']), path)

    def test_bound_source_bytes_cannot_be_reformatted(self):
        row = current_spec()['roots'][0]
        public = fresh_parent.read(row['public_parent'])
        record = fresh_parent.read(row['expanded_source_provenance'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'SOURCE.json'
            path.write_bytes(Path(public['source_path']).read_bytes() + b' ')
            changed_binding = scope.bind(path)
            record['source'] = record['output_source'] = changed_binding
            proof = Path(tmp) / 'PROVENANCE.json'; save(proof, record)
            with self.assertRaisesRegex(ValueError, 'not_exact_accepted_output'):
                expansion_binding.validate(row['root_id'], changed_binding, proof)


class FrozenAlgorithmCompatibilityTests(unittest.TestCase):
    def test_semantic_ordering_parent_prior_and_solver_code_only_changes_version_labels(self):
        original = HOME.parent / 'h4_end_to_end_expansion_mixed_v90/engine'
        for name in ('fresh_parent.py', 'ordering_generation.py', 'ordering_plan.py',
                     'ordering_review.py', 'actor_reference.py', 'mixed_evaluation.py', 'prior_rendering.py'):
            self.assertEqual((ENGINE / name).read_text().replace('v96', 'v90').replace('V96', 'V90'),
                             (original / name).read_text(), name)

    def test_temporary_builder_freezes_current_two_roots_and_v79_accepts_fresh_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp); shutil.copytree(ENGINE, home / 'engine')
            value = current_spec(); path = home / 'SPEC.json'; save(path, value)
            with patch.object(builder, 'PILOT', home):
                builder.build(path)
            shard = home / 'shards' / value['shard_id']
            config = fresh_parent.read(shard / 'CONFIG.json')
            self.assertEqual([s['root_id'] for s in config['ordering_slots']], list(scope.EXPECTED_ROOTS))
            self.assertEqual((config['max_generation_http'], config['max_candidates'], config['max_solver_calls_per_model']),
                             (6, 14, 228))
            self.assertFalse((shard / 'BATCH_PREPARATION.STARTED.json').exists())
            self.assertFalse((shard / 'solver').exists())
            proof = fresh_parent.read(shard / 'SHARD_INPUTS.COMPLETE.json')
            bound = {r['path'] for r in proof['files']}
            self.assertIn(str(path), bound)
            for row in value['roots']:
                self.assertIn(row['expanded_source_provenance'], bound)
            code = '''import batch
from common import load_multi
value=batch.config()
assets=load_multi('asset_generation')
for slot in value['multiselect_config']['slots']:
    assets.get_slot(value['multiselect_config'],slot['slot_id'])
    parent,source,path,inputs=assets.parent_source(slot)
    assert parent['gold']==list('ABCDEF')
print('V96_TEMP_BUILDER_AND_SEALED_V79_PASS')
'''
            result = subprocess.run([PYTHON, '-B', '-c', code], cwd=shard, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('V96_TEMP_BUILDER_AND_SEALED_V79_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
