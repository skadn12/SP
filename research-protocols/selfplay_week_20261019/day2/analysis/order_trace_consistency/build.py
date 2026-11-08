"""Check declared ordering citations against source appearance, no inference/cause edits."""
from __future__ import annotations
import csv,hashlib,itertools,json,pathlib,re
from datetime import datetime,timezone
HERE=pathlib.Path(__file__).resolve().parent
DAY2=HERE.parents[1]
RUN=DAY2/'solver/evidence_v2'

def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bind(p):p=pathlib.Path(p).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def require(c,m):
    if not c:raise RuntimeError(m)
def save(p,v):
    with pathlib.Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def scene(ref):return ref.rsplit(':T',1)[0]
def anchor(q,label):
    a=(q.get('order_anchor_spans') or {}).get(label) or (q.get('item_evidence') or {}).get(label,{})
    return a.get('order_anchor') or a.get('primary_citation') or a

def compute():
    manifest=read(RUN/'manifest.json');maps=read(RUN/'private/DISPLAY_MAPPINGS.json');scores=read(RUN/'SCORES.json')
    require(scores['status']=='COMPLETE','EVIDENCE_RUN_NOT_COMPLETE')
    freeze=read(RUN/'RAW_RESPONSE_FREEZE.json')
    for b in freeze['files']+[freeze['manifest_binding'],manifest['private_answer_binding']]:require(sha(b['path'])==b['sha256'],'BOUND_FILE_CHANGED')
    private={q['candidate_id']:q for q in read(manifest['private_answer_binding']['path'])['questions']}
    result=[];inputs=[bind(RUN/'manifest.json'),bind(RUN/'SCORES.json'),bind(RUN/'RAW_RESPONSE_FREEZE.json'),bind(RUN/'private/DISPLAY_MAPPINGS.json'),manifest['private_answer_binding']]
    for row in manifest['requests']:
        if row['question_type']!='ordering':continue
        q=private[row['candidate_id']];path=RUN/'responses'/f"{row['request_id']}.json";raw=read(path);inputs.append(bind(path))
        srcpath=RUN/'sources'/f"{row['root_id']}.json";require(sha(srcpath)==q['source_sha256'],'SOURCE_CHANGED');source=read(srcpath);inputs.append(bind(srcpath))
        turns={t['turn_ref']:t for t in source['turns']};indices={t['turn_ref']:i for i,t in enumerate(source['turns'])}
        canonical=maps['public_to_canonical'][q['candidate_id']];inv={v:k for k,v in canonical.items()};display=maps['requests'][row['request_id']]['display_to_content'];labelmap={d:inv[c] for d,c in display.items()}
        obj=json.loads(raw['raw_output']);evidence=obj['evidence'];answer_display=obj['answer'];answer=[labelmap[x] for x in answer_display]
        labels=[];mechanical=True;unique=True;position_ambiguity=[];authloc=True
        for entry in evidence:
            d=entry['label'];label=labelmap[d];ref=entry['turn_ref'];quote=entry['quote'];turn=turns.get(ref)
            offsets=[m.start() for m in re.finditer(re.escape(quote),turn['text'])] if turn and quote else []
            quote_valid=bool(turn and quote and offsets);mechanical &=quote_valid
            if len(offsets)!=1:position_ambiguity.append({'canonical_label':label,'reason':'NO_EXACT_OFFSET' if not offsets else 'MULTIPLE_EXACT_OFFSETS_IN_TURN'})
            primary=anchor(q,label);authref=primary.get('turn_ref');authquote=primary.get('quote','');authturn=turns.get(authref)
            authoffsets=[m.start() for m in re.finditer(re.escape(authquote),authturn['text'])] if authturn and authquote else []
            info={'display_label':d,'canonical_label':label,'item_text':q['items'][label],'model_status':entry.get('status'),
              'cited_turn_ref':ref,'cited_speaker':turn['speaker'] if turn else None,'cited_scene':scene(ref) if turn else None,
              'quote':quote,'source_full_turn':turn['text'] if turn else None,'quote_exact':quote_valid,'quote_offsets':offsets,
              'source_turn_index':indices.get(ref),'position':[indices[ref],offsets[0]] if len(offsets)==1 else None,
              'author_anchor_turn_ref':authref,'author_anchor_quote':authquote,'author_anchor_speaker':authturn['speaker'] if authturn else None,
              'author_anchor_quote_exact':bool(authoffsets),'same_turn_as_author':ref==authref,
              'location_only_semantic_judgment':'REQUIRES_SOURCE_REVIEW; different turn is not automatically an error'}
            labels.append(info)
        unique=len(labels)==len(q['items']) and {x['canonical_label'] for x in labels}==set(q['items'])
        if not unique:position_ambiguity.append({'reason':'LABEL_SET_INCOMPLETE_OR_DUPLICATE'})
        # A unique verbatim quote only locates the quoted span, not necessarily the proposition.
        computable=mechanical and unique and not position_ambiguity
        if computable:
            positions=[tuple(x['position']) for x in labels]
            if len(positions)!=len(set(positions)):position_ambiguity.append({'reason':'TIED_CITED_SPAN_POSITIONS'});computable=False
        citation_order=[x['canonical_label'] for x in sorted(labels,key=lambda x:x['position'])] if computable else None
        citation_display=[next(d for d,l in labelmap.items() if l==c) for c in citation_order] if citation_order else None
        gold=q['gold'];gold_display=[next(d for d,l in labelmap.items() if l==c) for c in gold]
        # Crosscheck frozen score after independent display->canonical conversion.
        score=next(x for x in scores['requests'] if x['request_id']==row['request_id'])
        require(bool(answer==gold)==score['exact'],'GOLD_SCORE_DISAGREEMENT')
        gold_positions=[]
        for label in gold:
            a=anchor(q,label);ref=a.get('turn_ref');qt=a.get('quote','');t=turns.get(ref);off=t['text'].find(qt) if t and qt else -1
            gold_positions.append([indices.get(ref),off] if ref in indices and off>=0 else None)
        auth_gold_valid=all(p is not None for p in gold_positions) and gold_positions==sorted(gold_positions)
        group='EXACT_CITATIONS_UNIQUE_POSITIONS' if computable else 'NONVERBATIM_OR_AMBIGUOUS_CITATION_POSITIONS'
        result.append({'candidate_id':q['candidate_id'],'root_id':q['root_id'],'request_id':row['request_id'],'group':group,
          'source_path':str(srcpath),'source_sha256':q['source_sha256'],'question':q['question'],'labels':labels,
          'all_item_quotes_verbatim_in_cited_turn':mechanical,'complete_unique_item_labels':unique,
          'position_ambiguity':position_ambiguity,'unique_citation_positions':computable,
          'answer_display':answer_display,'answer_canonical':answer,'citation_sorted_display':citation_display,'citation_sorted_canonical':citation_order,
          'gold_display':gold_display,'gold_canonical':gold,'answer_consistent_with_own_citation_positions':answer==citation_order if computable else None,
          'citation_position_order_matches_gold':citation_order==gold if computable else None,'answer_matches_gold':answer==gold,
          'author_gold_anchors_verbatim_and_monotone':auth_gold_valid,'author_gold_anchor_positions':gold_positions,
          'strict_runtime_valid':score['valid'],'source_review_status':'PENDING_MANUAL_CHECK',
          'interpretation':'Observable citation/answer consistency only. Self-reported evidence may not reflect internal computation. No mechanism proved.'})
    require(len(result)==20,'ORDERING_EXPECTED_20')
    out={'status':'MECHANICAL_CHECK_COMPLETE_MANUAL_SOURCE_REVIEW_PENDING','created_at_utc':datetime.now(timezone.utc).isoformat(),
      'namespace':'train.feedback','ordering_questions':20,'results':result,'input_bindings':inputs,'tool_binding':bind(pathlib.Path(__file__)),
      'counts':{'all_quotes_exact_unique_positions':sum(r['unique_citation_positions'] for r in result),
        'quote_or_position_issues':sum(not r['unique_citation_positions'] for r in result),
        'final_answer_matches_own_citation_order':sum(r['answer_consistent_with_own_citation_positions'] is True for r in result),
        'all_quotes_exact_but_answer_inconsistent_with_own_citation_order':sum(r['unique_citation_positions'] and not r['answer_consistent_with_own_citation_positions'] for r in result),
        'own_citation_order_matches_gold':sum(r['citation_position_order_matches_gold'] is True for r in result),
        'final_answer_matches_gold':sum(r['answer_matches_gold'] for r in result),
        'author_gold_anchors_verbatim_and_monotone':sum(r['author_gold_anchors_verbatim_and_monotone'] for r in result)},
      'limits':['Verbatim quote only establishes span position, not semantic support.','Different cited turns are not automatically errors.','Source review may inspect and qualify these observations; no Gold changes.','Self-reported evidence is not a faithful internal computation trace by assumption.','No expansion, API calls or model inference.']}
    save(HERE/'MECHANICAL_TRACE_CHECK.json',out)
    print(json.dumps(out['counts'],ensure_ascii=False,indent=2))
    for r in result:
        print('\nROOT',r['root_id'],'group',r['group'],'answer',r['answer_canonical'],'cited',r['citation_sorted_canonical'],'gold',r['gold_canonical'])
        for l in r['labels']:print(l['canonical_label'],l['item_text'],'\n CITE',l['cited_turn_ref'],l['cited_speaker'],repr(l['quote']),'\n FULL',l['source_full_turn'],'\n AUTHOR',l['author_anchor_turn_ref'],repr(l['author_anchor_quote']))
if __name__=='__main__':compute()
