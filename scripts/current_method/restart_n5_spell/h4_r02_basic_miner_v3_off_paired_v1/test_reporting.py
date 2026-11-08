import unittest
from copy import deepcopy
from .reporting import DIMENSIONS,REQUIRED,PASS,FAIL,UNDETERMINED,aggregate_dimensions,summarize,NOT_EVALUATED
from .constants import SCHEDULE


def dimensions():return {d:{'judgment':'PASS','reason':'offline fixture'} for d in DIMENSIONS}


class ReviewTests(unittest.TestCase):
    def test_NA_is_permitted_only_on_optional_dimensions(self):
        d=dimensions()
        for name in set(DIMENSIONS)-REQUIRED:d[name]['judgment']='NOT_APPLICABLE'
        self.assertEqual(aggregate_dimensions(d),PASS)
        for name in REQUIRED:
            broken=deepcopy(d);broken[name]['judgment']='NOT_APPLICABLE'
            with self.assertRaises(ValueError):aggregate_dimensions(broken)

    def test_specificity_fail_does_not_modify_evidence_verdict(self):
        d=dimensions();d['DISTINCTION_SPECIFICITY']['judgment']='FAIL'
        self.assertEqual(aggregate_dimensions(d),FAIL)
        self.assertEqual(d['EVIDENCE_SUPPORT']['judgment'],'PASS')
        d['DISTINCTION_SPECIFICITY']['judgment']='UNDETERMINED'
        self.assertEqual(aggregate_dimensions(d),UNDETERMINED)
        d['EVIDENCE_GROUPING']['judgment']='FAIL'
        self.assertEqual(aggregate_dimensions(d),FAIL)

    def fixture(self):
        items=[];reviews=[];groups=[]
        for row in SCHEDULE:
            groups.append({**row,'full_response_schema_valid':True,'full_item_schema_valid_count':3})
            for i in range(3):
                item={**row,'raw_index':i,'returned':True,'planned_slot':True,'structurally_valid':True}
                items.append(item);reviews.append({**item,'judgment':PASS,'dimensions':dimensions(),'tags':[]})
        return items,reviews,groups

    def test_strict_instance_acceptance_and_fixed_denominators(self):
        items,reviews,groups=self.fixture()
        out=summarize(items,reviews,groups)
        self.assertEqual(out['V3_INSTANCE_ACCEPTANCE'],'PASS')
        self.assertEqual(out['equal_weight_mean_delta'],0)
        reviews[3]['judgment']=FAIL
        self.assertEqual(summarize(items,reviews,groups)['V3_INSTANCE_ACCEPTANCE'],'FAIL')
        reviews[3]['judgment']=NOT_EVALUATED;items[3]['structurally_valid']=False
        out=summarize(items,reviews,groups)
        self.assertEqual(out['V3_INSTANCE_ACCEPTANCE'],'INCONCLUSIVE')
        self.assertEqual(out['groups'][1]['planned_slots'],3)
        self.assertEqual(out['groups'][1]['pass_rate'],2/3)

    def test_full_response_schema_failure_does_not_erase_original_valid_items(self):
        items,reviews,groups=self.fixture();groups[1]['full_response_schema_valid']=False
        out=summarize(items,reviews,groups)
        self.assertEqual(out['V3_INSTANCE_ACCEPTANCE'],'INCONCLUSIVE')
        self.assertEqual(out['groups'][1]['original_valid_intents'],3)
        self.assertEqual(out['groups'][1]['content_pass'],3)

    def test_extra_pass_cannot_replace_failed_planned_raw_slot(self):
        items,reviews,groups=self.fixture();reviews[3]['judgment']=FAIL
        extra={**items[3],'raw_index':3,'planned_slot':False}
        items.append(extra);reviews.append({**extra,'judgment':PASS,'dimensions':dimensions(),'tags':[]})
        out=summarize(items,reviews,groups)
        self.assertEqual(out['groups'][1]['content_pass'],2)
        self.assertEqual(out['groups'][1]['extras'],1)
        self.assertEqual(out['V3_INSTANCE_ACCEPTANCE'],'FAIL')


if __name__=='__main__':unittest.main()
