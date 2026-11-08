"""Offline v79 exact output-tail and historical statement-set regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import asset_generation as a
import compose as c
import test_asset_generation as asset_tests
import test_compose as compose_tests


def tail_contract(payload):
    content = payload['messages'][0]['content']
    tail = content[content.rindex(a.START):]
    assert tail.endswith(a.END)
    targets, _ = json.JSONDecoder().raw_decode(tail.split('EXACT_TARGET_FACT_IDS_JSON\n', 1)[1])
    rendered = tail.split('OUTPUT_JSON_SKELETON\n', 1)[1]
    skeleton, end = json.JSONDecoder().raw_decode(rendered)
    assert rendered[end:].strip() == a.END
    return tail, targets, skeleton


class OutputContractTests(unittest.TestCase):
    # The inherited fixture provides only synthetic feedback, payload builder,
    # and transport. Production engine functions themselves are not patched.
    setUp = asset_tests.ContractTests.setUp
    fake_generate = asset_tests.ContractTests.fake_generate
    judgment = asset_tests.ContractTests.judgment
    review = asset_tests.ContractTests.review

    def test_initial_root_and_round_end_in_exact_full_skeleton(self):
        for path in [self.run/'INITIAL_REQUEST.json', self.run/'rounds/initial/INITIAL_REQUEST.json']:
            _, targets, skeleton = tail_contract(a.read(path))
            self.assertEqual(targets, a.FACT_IDS)
            self.assertEqual(skeleton, {'root_id':'root1', 'family_id':'family1',
                'replacements':[{'fact_id':fid, 'replacement_proposition':None} for fid in a.FACT_IDS]})
        a.verify(self.run)

    def supplement(self):
        self.fake_generate()
        self.review(approved=['F01'])
        result = a.prepare_supplement(self.run)
        self.assertEqual(result['target_fact_ids'], a.FACT_IDS[1:])
        return a.read(self.run/'rounds/supplement/INITIAL_REQUEST.json')

    def test_supplement_tail_has_only_F02_to_F06_and_replays(self):
        payload = self.supplement()
        tail, targets, skeleton = tail_contract(payload)
        self.assertEqual(targets, a.FACT_IDS[1:])
        self.assertEqual([r['fact_id'] for r in skeleton['replacements']], targets)
        self.assertNotIn('F01', tail)
        manifest, replayed, proof = a.verify_round_request(self.run, 'supplement')
        self.assertEqual(replayed, payload)
        self.assertEqual(manifest['target_fact_ids'], targets)
        self.assertEqual(proof['target_fact_ids'], targets)
        self.assertEqual(proof['payload_sha256'], a.base.digest(payload))

    def test_all_null_supplement_is_valid_and_preserves_initial_acceptance(self):
        _, targets, skeleton = tail_contract(self.supplement())
        data, errors = a.validate_assets(json.dumps(skeleton), self.m, self.source, self.parent, targets)
        self.assertEqual(errors, [])
        self.assertEqual(len(data['replacements']), 5)
        self.assertTrue(all(r['contradicted'] is None for r in data['replacements']))
        self.fake_generate('supplement', skeleton)
        self.review('supplement', approved=[])
        complete = a.finalize(self.run)
        self.assertEqual(complete['counterversion_count'], 1)
        a.verify(self.run, complete=True)
        accepted = a.read(self.run/'ROUND_PROVENANCE.json')['accepted_versions']
        self.assertEqual([(r['fact_id'],r['round_id']) for r in accepted], [('F01','initial')])

    def test_supplement_added_F01_missing_or_reordered_target_rejected_whole(self):
        _, targets, skeleton = tail_contract(self.supplement())
        variants = []
        extra = deepcopy(skeleton)
        extra['replacements'].insert(0, {'fact_id':'F01','replacement_proposition':None})
        variants.append(extra)
        missing = deepcopy(skeleton); missing['replacements'].pop(); variants.append(missing)
        reordered = deepcopy(skeleton); reordered['replacements'].reverse(); variants.append(reordered)
        for response in variants:
            data, errors = a.validate_assets(json.dumps(response), self.m, self.source, self.parent, targets)
            self.assertIsNone(data)
            self.assertEqual(errors, [{'fact_id':None,'error':'target_fact_order'}])

    def test_all_source_turns_and_metadata_precede_exact_tail_unchanged(self):
        source = deepcopy(self.source)
        source['turns'] += [{'turn_ref':f'SOURCE:EXACT:T{i:03d}', 'speaker':'Jordan',
            'text':f'Complete source continuation {i}, including qualifiers and later outcomes.'}
            for i in range(2, 92)]
        source['candidate_metadata']['audit_marker'] = 'preserve all source metadata'
        path = self.home/'complete_source.json'; asset_tests.dump(path, source)
        parent = deepcopy(self.parent); parent.update(source_path=str(path), source_sha256=a.sha(path))
        pool = self.home/'complete_parent.json'; asset_tests.dump(pool, {'questions':[parent]})
        config = deepcopy(self.cfg); config['slots'][0].update(parent_pool_path=str(pool),
            source_path=str(path), history_path=str(path), source_sha256=a.sha(path), history_sha256=a.sha(path))
        config_path = self.home/'complete_CONFIG.json'; asset_tests.dump(config_path, config)
        run = self.home/'complete_run'; a.prepare(config_path, 'family1', run=run)
        for payload in [a.read(run/'INITIAL_REQUEST.json'),a.read(run/'rounds/initial/INITIAL_REQUEST.json'),
                        a.round_payload(run,'supplement',a.FACT_IDS[1:])[0]]:
            content = payload['messages'][0]['content']
            context, _ = json.JSONDecoder().raw_decode(content.split('INPUT_JSON\n',1)[1])
            self.assertEqual(context['source_turns'], source['turns'])
            self.assertEqual(context['source_metadata'], {k:v for k,v in source.items() if k!='turns'})
            self.assertLess(content.index('FROZEN_FEEDBACK'), content.rindex(a.START))
            tail_contract(payload)
        self.assertEqual((run/'SOURCE.json').read_bytes(), path.read_bytes())
        a.verify(run)

    def test_changed_output_tail_rejected_by_frozen_round_replay(self):
        self.supplement()
        path = self.run/'rounds/supplement/INITIAL_REQUEST.json'
        original = path.read_bytes(); payload = a.read(path)
        payload['messages'][0]['content'] = payload['messages'][0]['content'].replace(
            'EXACT_TARGET_FACT_IDS_JSON\n["F02"', 'EXACT_TARGET_FACT_IDS_JSON\n["F01"')
        asset_tests.dump(path, payload)
        with self.assertRaisesRegex(ValueError, 'round_request_changed'):
            a.verify_round_request(self.run, 'supplement')
        path.write_bytes(original)
        a.verify_round_request(self.run, 'supplement')

    def test_renderer_rejects_empty_duplicate_unknown_or_reordered_targets(self):
        for targets in [[], ['F01','F01'], ['F07'], ['F03','F02'], ('F02','F03')]:
            with self.assertRaisesRegex(ValueError, 'task_tail_targets_invalid'):
                a.render_task_tail(self.m, targets)


class HistoricalDedupTests(unittest.TestCase):
    setUp = compose_tests.ComposeTests.setUp

    def test_slot_prior_excludes_all_old_sets_keeps_controls_and_binds_replay(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td); parents = compose_tests.fixture(home)
            config = c.read(home/'CONFIG.json'); previous = []
            for slot in config['slots']:
                family = c.load_family(home, slot, parents, {})
                for candidates in c.enumerate_children(family).values():
                    for candidate in candidates:
                        q, _ = c.assemble_child(family, candidate)
                        q['candidate_id'] = 'old-' + q['candidate_id']
                        q['question'] = 'An old stem whose wording does not define novelty.'
                        q['options'] = {label:'  '+text.upper()+'  ' for label,text
                                        in zip('FEDCBA',q['options'].values())}
                        q.pop('gold'); q.pop('option_evidence'); previous.append(q)
            prior_path = home/'previous_public_pool.json'
            compose_tests.save(prior_path, {'questions':previous})
            for slot in config['slots']: slot['prior_pool_paths'] = [str(prior_path)]
            compose_tests.save(home/'CONFIG.json', config)
            outputs, inputs, _ = c.build(home)
            self.assertEqual(inputs[str(prior_path.resolve())], c.sha(prior_path))
            self.assertEqual(outputs['POOL.json']['questions'], parents)
            self.assertEqual(outputs['ALLOCATION.json']['assembled_child_count'], 0)
            self.assertTrue(all(not m['new_candidate'] for m in outputs['FAMILY_SIDECAR.json']['members']))
            self.assertTrue(all(s['reason']=='NO_NOVEL_EXACT_STATEMENT_SET'
                for f in outputs['ALLOCATION.json']['families'] for s in f['shortfalls']))
            c.compose(home); c.verify_assembly(home)
            compose_tests.save(prior_path, {'questions':[]})
            with self.assertRaisesRegex(ValueError, 'assembly_bound_file_changed'):
                c.verify_assembly(home)

    def test_same_parent_duplicate_alternative_does_not_increase_new_questions(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td); parents = compose_tests.fixture(home)
            slot = c.read(home/'CONFIG.json')['slots'][0]
            family = c.load_family(home, slot, parents, {})
            expected = c.enumerate_children(family)
            family['variants']['F01'].append(deepcopy(family['variants']['F01'][-1]))
            self.assertEqual(c.enumerate_children(family), expected)


if __name__ == '__main__':
    unittest.main()
