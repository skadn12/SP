"""Shared supplemental presentation transform; preserves original failed drafts."""
import argparse, re
from control import BASE, read, write, bind, now
from questions import SLOTS, mechanical

# Deliberately limited to selection questions, never factual option/source text.
PATTERN = r'\b(which)\s+(?:two|three|four|five|six)\s+(?=(?:bindings|combinations|reported actions|proposed matches|statements|descriptions|details|proposed entries|options|claims|pieces of evidence)\b)'
def normalize(root,arm):
    approval=read(BASE/'PRESENTATION_AMENDMENT_REVIEW.json')
    assert approval['status']=='SUPPORTED_AS_EXPLORATORY_SUPPLEMENT', 'Independent amendment review required'
    manifest=read(BASE/'PRESENTATION_AMENDMENT.json')
    assert manifest['pattern']==PATTERN
    records=[]
    for slot in SLOTS:
        original=BASE/'questions'/root/arm/(slot+'.json')
        if not original.exists():
            records.append({'slot_id':slot,'status':'NO_ORIGINAL_CANDIDATE_NO_REPLACEMENT'});continue
        q=read(original); before=q['question']; after=before; count=0
        if q['question_type']=='multi_select':
            after,count=re.subn(PATTERN,lambda m:m.group(1)+' ',before,flags=re.I)
        assert count<=1, 'Ambiguous multi-match remains failure; no adaptive rewrite'
        nq={**q,'question':after,'presentation_parent_binding':bind(original),
            'presentation_transform':{'version':'fbc1_shared_count_cue_removal_v1','created_at_utc':now(),
             'editor_id':'/root/program','replacement_count':count,'gold_used_for_replacement':False,
             'source_options_items_gold_evidence_display_unchanged':True,'author_regeneration':False,
             'amendment_binding':bind(BASE/'PRESENTATION_AMENDMENT.json')}}
        assert {k:v for k,v in nq.items() if k not in ['question','presentation_parent_binding','presentation_transform']}=={k:v for k,v in q.items() if k!='question'}
        dest=BASE/'normalized_questions'/root/arm/(slot+'.json');write(dest,nq);mechanical(dest)
        records.append({'slot_id':slot,'original':bind(original),'normalized':bind(dest),'replacements':count,'before':before,'after':after})
    receipt={'schema_version':'fbc1_shared_presentation_transform_receipt_v1','created_at_utc':now(),'root_id':root,'condition':arm,
             'original_first_pass_reviews_preserved':True,'independent_normalized_review_required':True,'records':records,
             'author_attempts_added':0,'solver_requests_added':0,'gold_used_for_replacement':False,'pattern':PATTERN}
    write(BASE/'normalized_questions'/root/arm/'TRANSFORM_RECEIPT.json',receipt)
    return {'root_id':root,'condition':arm,'candidates':len([r for r in records if 'normalized' in r]),'replacements':sum(r.get('replacements',0) for r in records)}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--arm',required=True);a=ap.parse_args();print(normalize(a.root,a.arm))
