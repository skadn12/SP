"""Offline adversarial checks for the independent v90 result exporter."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


audit = load(HERE/'export_end_to_end_v90.py', 'v90_export_test_module')
native = load(HERE.parents[1]/'h4_feedback_training_pool_v15_qa/evaluation/protocol.py', 'v90_export_native_reference')


def question(kind='multi_select', count=2, number=1):
    field = 'options' if kind == 'multi_select' else 'items'
    return {'candidate_id': 'q'+str(number), 'root_id': 'root1', 'question_type': kind,
            'question': 'Which statements are supported?', field: {k: 'Distinct proposition '+k for k in audit.LABELS[kind]},
            'gold': list(audit.LABELS[kind])[:count if kind == 'multi_select' else 6]}


def response(q, values):
    ids = audit.content_ids(q); mapping = ids
    pred, gold = {ids[k] for k in values}, {ids[k] for k in q['gold']}
    if q['question_type'] == 'multi_select':
        tp, fp, fn = len(pred&gold), len(pred-gold), len(gold-pred)
        partial = Fraction(tp,len(gold))-(Fraction(fp,6-len(gold)) if len(gold)<6 else 0)
        exact = pred == gold
    else:
        rank = {k:i for i,k in enumerate(values)}
        fp = sum(rank[x]>rank[y] for i,x in enumerate(q['gold']) for y in q['gold'][i+1:]); tp, fn = 15-fp,0
        partial,exact = Fraction(tp-fp,15), list(values)==q['gold']
    return {'variant': {'display_to_content': mapping}}, {'raw_output':'<box>'+','.join(values)+'</box>',
        'score': {'partial_score_fraction':str(partial),'exact_match':exact,'tp':tp,'fp':fp,'fn':fn}}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


class ExportChecks(unittest.TestCase):
    def test_receipt_checks_mixed_installed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);bindings=[]
            for name,body in [('common.py','import json\n'),('RULES.md','# Frozen rules\n'),
                              ('prompt.txt','Full dialogue prompt\n')]:
                path=home/name;path.write_text(body);bindings.append(audit.bind(path))
            record=home/'CONFIG.json';save(record,{'training_allowed':False})
            bindings.append(audit.bind(record))
            receipt=home/'INSTALLATION.json';save(receipt,{'files':bindings,'source_files':bindings})
            checker=audit.Audit();self.assertEqual(checker.receipts(receipt)['files'],bindings)
            self.assertEqual(checker.binding(audit.bind(record)),{'training_allowed':False})
            self.assertTrue(all(b['path'] in checker.files for b in bindings))

    def test_opaque_artifact_tamper_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'common.py';path.write_text('import json\n');binding=audit.bind(path)
            path.write_text('import os\n')
            with self.assertRaisesRegex(ValueError,'binding_changed'):audit.Audit().binding(binding)

    def test_invalid_bound_json_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'CONFIG.json';path.write_text('not a JSON record')
            with self.assertRaises(json.JSONDecodeError):audit.Audit().binding(audit.bind(path))

    def test_gold2_all_select_zero_and_wrong_negative(self):
        q=question()
        for values, expected in [(list('AB'),Fraction(1)), (list('ABCDEF'),Fraction(0)), (list('CDEF'),Fraction(-1))]:
            request, raw=response(q,values); self.assertEqual(audit.raw_score(q,request,raw)[0],expected)

    def test_ordering_pair_score(self):
        q=question('ordering')
        for values, expected in [(q['gold'],Fraction(1)), (q['gold'][::-1],Fraction(-1)),
                                 (['I2','I1','I3','I4','I5','I6'],Fraction(13,15))]:
            request,raw=response(q,values);self.assertEqual(audit.raw_score(q,request,raw)[0],expected)

    def test_invalid_and_stale_score_rejected(self):
        q=question();request,raw=response(q,['A','B']);raw['score']['partial_score_fraction']='0'
        with self.assertRaisesRegex(ValueError,'saved_score_disagrees'):audit.raw_score(q,request,raw)
        request,raw=response(q,['A','B']);raw['raw_output']='<box>A,A</box>'
        with self.assertRaisesRegex(ValueError,'invalid_labels'):audit.raw_score(q,request,raw)

    def test_typed_control_keeps_ordering(self):
        multi=question(count=6);ordering=question('ordering',number=2)
        self.assertTrue(audit.control(multi));self.assertFalse(audit.control(ordering))
        self.assertEqual(audit.typed_pool([ordering],'TEST')['by_type']['ordering'],1)
        with self.assertRaisesRegex(ValueError,'control_leaked'):audit.typed_pool([multi],'TEST')

    def test_gold_blind_native_variants_exact(self):
        for kind,count in [('multi_select',6),('ordering',12)]:
            q=question(kind);self.assertEqual(audit.variants(q),native.make_variants(q,seed=audit.SEED,count=count))
            if kind=='multi_select':
                q['gold']=list('CDEF');self.assertEqual(audit.variants(q),native.make_variants(q,seed=audit.SEED,count=count))

    def test_public_scope_retains_context_but_excludes_old_occurrence(self):
        src={'turns':[{'turn_ref':'old:T001'},{'turn_ref':'new:T001'},{'turn_ref':'new:T002'}]}
        q={'question':'Order the statements as they APPEAR IN THE SUPPLIED DIALOGUE within the new scene.'}
        self.assertEqual(audit.matching_refs(q,src),{'new:T001','new:T002'})
        q['question']='Order all statements chronologically.';self.assertEqual(len(audit.matching_refs(q,src)),3)
        q['question']='Order as they APPEAR IN THE SUPPLIED DIALOGUE within the old and new scenes.'
        self.assertEqual(len(audit.matching_refs(q,src)),3)

    def test_typed_dedup_and_candidate_collision(self):
        q=question();duplicate=deepcopy(q);duplicate['candidate_id']='dup'
        duplicate['options']['A']='  DISTINCT   proposition A  '
        rows,removed=audit.deduplicate([q,duplicate]);self.assertEqual(rows,[q]);self.assertEqual(len(removed),1)
        changed=deepcopy(q);changed['gold']=list('AC')
        with self.assertRaisesRegex(ValueError,'candidate_id_collision'):audit.deduplicate([q,changed])

    def test_historical_inventory_and_bytes_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);home=root/'pilot';old=root/'old';home.mkdir();old.mkdir();p=old/'keep.txt';p.write_text('original')
            save(home/'PRESERVATION_BEFORE.json',{'files':{str(p):audit.sha(p)},'directories':[str(old)],'protected_count':1})
            self.assertEqual(audit.preservation(home)['status'],'PASS')
            extra=old/'extra.txt';extra.write_text('added')
            with self.assertRaisesRegex(ValueError,'historical_inventory_changed'):audit.preservation(home)
            extra.unlink();p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'historical_bytes_changed'):audit.preservation(home)

    def test_saved_exports_rebuilt_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);shard=home/'shards/shard_01';qs=[question(),question('ordering',number=2)]
            baseline=home/'baseline.json';save(baseline,{'questions':[],'candidate_count':0})
            save(shard/'EVALUATION_POOL.json',{'questions':qs})
            summaries={side:[] for side in audit.MODELS}
            for side in audit.MODELS:
                run=shard/'solver'/side;requests=[]
                for q in qs:
                    n=18 if q['question_type']=='multi_select' else 12
                    metrics=[]
                    for i in range(n):
                        values=['A'] if q['question_type']=='multi_select' else ['I2','I1','I3','I4','I5','I6']
                        req,raw=response(q,values);req.update(request_id=q['candidate_id']+'_'+str(i),candidate_id=q['candidate_id'])
                        requests.append(req);save(run/'responses'/(req['request_id']+'.json'),raw)
                        value,exact=audit.raw_score(q,req,raw);metrics.append({'partial':str(value),'exact':exact})
                    stats=audit.metric(metrics);summaries[side].append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],
                        'question_type':q['question_type'],'gold_count':len(q['gold']),'control':False,'selected':True,**stats})
                save(run/'manifest.json',{'requests':requests})
            report={'status':'PASS','files':[],'shards':[str(shard)],'replayed_models':summaries,'baseline':audit.bind(baseline),
                    'selected_new_ids':[q['candidate_id'] for q in qs],'new_count':2,'cumulative_count':2}
            report_path=home/'result_audit.json';save(report_path,report)
            new=audit.typed_pool(qs,'NEW');new.update(result_audit=audit.bind(report_path),shards=[str(shard)],duplicates_removed=[],selected_new_ids_before_deduplication=[q['candidate_id'] for q in qs])
            cumulative=audit.typed_pool(qs,'CUMULATIVE');cumulative.update(result_audit=audit.bind(report_path),baseline=audit.bind(baseline),duplicates_removed=[])
            save(home/'AUDITED_NEW_CANDIDATES.json',new);save(home/'CUMULATIVE_AUDITED_NEW_CANDIDATES.json',cumulative)
            self.assertEqual(audit.audit_saved_exports(home)['new_count'],2)
            new['gold_distribution']['2']=99;save(home/'AUDITED_NEW_CANDIDATES.json',new)
            with self.assertRaisesRegex(ValueError,'typed_counts_and_permissions'):audit.audit_saved_exports(home)
            new['gold_distribution']['2']=1;new['questions']=new['questions'][:1];save(home/'AUDITED_NEW_CANDIDATES.json',new)
            with self.assertRaisesRegex(ValueError,'exact_question_projection'):audit.audit_saved_exports(home)


if __name__=='__main__':unittest.main(verbosity=2)
