"""Deterministic, lossless fact ledger from the ORIGINAL saved draft.

Canonical facts are attributed utterances, never inferred world facts. Whitespace
is normalized; clocks, quantities, negations and modality remain attributable
to exact source spans. JSON lexical byte spans and decoded character spans are
separately labeled. Native mappings are verified against reserved utterances.
"""
import copy,json,pathlib,re
from common import BASE,atomic,bind,checked,inputs,load,source_input,sha
from events import classify
def tokens(raw):
    for m in re.finditer(r'"(?:\\.|[^"\\])*"',raw):
        try:yield m,json.loads(m.group())
        except ValueError:continue
def text_locations(raw):
    ts=list(tokens(raw));found=[]
    for i,(m,value) in enumerate(ts[:-1]):
        if value=='text':
            n,v=ts[i+1]
            if raw[m.end():n.start()].strip()==':':
                found.append((v,dict(start_inclusive=len(raw[:n.start()].encode()),end_exclusive=len(raw[:n.end()].encode()),coordinate='UTF8_JSON_STRING_TOKEN')))
    return found
def canonical(turn):
    text=turn['text']
    return dict(fact_id=turn['turn_ref'],kind='attributed_utterance',speaker=turn['speaker'],
                canonical_fact=' '.join(text.split()),quote=text,turn_ref=turn['turn_ref'],
                char_span=dict(start_inclusive=0,end_exclusive=len(text)),
                clocks=re.findall(r'\b\d{1,2}:\s?\d{2}\b',text),
                quantities=re.findall(r'\b\d+(?:\.\d+)?\b',text),
                modality_markers=sorted(set(re.findall(r"\b(?:propose|planned|plan|requested|offered|agreed|confirmed|completed|haven't|hasn't|not|never|tentative|will|I'll|may|because)\b",text,re.I))))
def source_turns(candidate):return [t for s in candidate['scenes'] for t in s['turns']]
def native_data(inp):
    return [dict(source_id=n['source_id'],scene_ref=n['scene_ref'],binding=n['binding'],
                 turns=[dict(turn_id=str(t['turn_id']),speaker=t['speaker'],utterance=t['utterance']) for t in checked(n['binding'])['turns']]) for n in inp['native_events']]
def mappings(candidate,natives):
    turns={t['turn_ref']:t for t in source_turns(candidate)}
    bynative={(n['source_id'],t['turn_id']):t for n in natives for t in n['turns']}
    sidscene={n['source_id']:n['scene_ref'] for n in natives};out=[];errors=[];seen=set()
    for i,m in enumerate(candidate.get('native_evidence',[])):
        key=(m.get('source_id'),str(m.get('native_turn_id')));scene=sidscene.get(key[0]);errs=[];native=bynative.get(key)
        if key in seen:errs.append('duplicate_native_mapping')
        seen.add(key)
        q=m.get('native_quote','')
        if not native or not q or q not in native['utterance']:errs.append('native_quote_mismatch')
        if m.get('disposition') not in ['PRESERVED','OMITTED_NONMATERIAL']:errs.append('invalid_disposition')
        if not isinstance(m.get('rationale'),str) or not m['rationale'].strip():errs.append('missing_disposition_rationale')
        evidence=[]
        for e in m.get('source_evidence',[]):
            t=turns.get(e.get('turn_ref'));quote=e.get('quote','')
            if not t or not quote or quote not in t['text'] or not e.get('turn_ref','').startswith(str(scene)+':'):errs.append('source_mapping_quote_mismatch')
            else:
                start=t['text'].index(quote);evidence.append(dict(turn_ref=t['turn_ref'],quote=quote,char_span=dict(start_inclusive=start,end_exclusive=start+len(quote))))
        if m.get('disposition')=='PRESERVED' and not evidence:errs.append('preserved_without_source_span')
        if m.get('disposition')=='OMITTED_NONMATERIAL' and m.get('source_evidence'):errs.append('omitted_has_source_evidence')
        row=dict(source_id=key[0],native_turn_id=key[1],scene_ref=scene,disposition=m.get('disposition'),native_quote=q,rationale=m.get('rationale'),source_spans=evidence)
        if native and q in native['utterance']:row['native_char_span']={'start_inclusive':native['utterance'].index(q),'end_exclusive':native['utterance'].index(q)+len(q)}
        out.append(row)
        errors.extend(dict(scene_ref=scene,code=e,mapping_index=i,native_turn_id=key[1]) for e in errs)
    for key in bynative.keys()-seen:errors.append(dict(scene_ref=sidscene[key[0]],code='missing_native_mapping',native_turn_id=key[1]))
    for key in seen-bynative.keys():errors.append(dict(scene_ref=None,code='unknown_native_mapping',native_turn_id=key[1]))
    return out,errors
def mechanical(candidate,inp,parent):
    errors=[]
    def error(scene,code):errors.append(dict(scene_ref=scene,code=code))
    if candidate.get('root_id')!=inp['root_id']:error(None,'root_mismatch')
    if candidate.get('source_status')!=inp['source_status']:error(None,'source_status_mismatch')
    scenes=candidate.get('scenes',[])
    if [s.get('scene_ref') for s in scenes]!=inp['scene_sequence']:error(None,'scene_sequence_mismatch')
    refs=[];texts=[];speakers={t['speaker'] for t in parent['turns']}
    for s in scenes:
        sr=s.get('scene_ref');ts=s.get('turns',[])
        if s.get('route')!=sr.split(':')[1]:error(sr,'route_mismatch')
        if len(ts)!=8:error(sr,'requires_eight_turns')
        for i,t in enumerate(ts,1):
            if t.get('turn_ref')!=f'{sr}:T{i:03d}':error(sr,'turn_ref_mismatch')
            if t.get('speaker') not in speakers:error(sr,'new_speaker')
            if t.get('origin')!='codex_synthetic_extension':error(sr,'origin_mismatch')
            if not isinstance(t.get('text'),str) or not t['text'].strip():error(sr,'empty_text')
            else:texts.append(t['text'])
            refs.append(t.get('turn_ref'))
    if len(refs)!=len(set(refs)):error(None,'duplicate_turn_ref')
    if len(texts)!=48:error(None,'requires_48_new_turns')
    if texts and sum(len(t.split()) for t in texts)>18*len(texts):
        for s in scenes:
            if sum(len(t['text'].split()) for t in s['turns'])>18*len(s['turns']):error(s['scene_ref'],'root_mean_words_above_18_scene_contributor')
    return errors
def ledger(candidate,inp,parent,natives,initial_binding,raw=None):
    facts=[canonical(t) for t in source_turns(candidate)];locations=text_locations(raw) if raw else []
    if raw:
        if [v for v,_ in locations]!=[f['quote'] for f in facts]:raise ValueError('JSON text token mapping differs from draft')
        for f,(_,span) in zip(facts,locations):f['saved_draft_byte_span']=span
    maps,errors=mappings(candidate,natives)
    errors=mechanical(candidate,inp,parent)+errors
    return dict(schema_version='compact_fact_ledger_v2',root_id=inp['root_id'],initial_draft_binding=initial_binding,
                facts=facts,parent_facts=[canonical(t) for t in parent['turns']],native_mappings=maps,
                native_bindings=[n['binding'] for n in natives],parent_binding=inp['parent_source_binding'],
                warnings=candidate.get('warnings',[]),mechanical_errors=errors,
                canonicalization='Attributed utterance; whitespace only. Markers are lexical indexes, not inferred truth or state.')
def build(root):
    d=BASE/'sources'/root;d.mkdir(parents=True,exist_ok=True)
    old=inputs()[root]['fixed_initial_draft'];candidate=checked(old);raw=pathlib.Path(old['path']).read_text();od=pathlib.Path(old['path']).parent
    terminal=load(od/'TERMINAL.json');receipt=classify((od/'EVENTS.jsonl').read_text(),raw,terminal['returncode'],terminal.get('timed_out',False))
    inp=source_input(root);parent=checked(inp['parent_source_binding']);native=native_data(inp)
    atomic(d/'INITIAL_EVENT_RECEIPT.json',receipt)
    atomic(d/'INITIAL_PROVENANCE.json',dict(draft=old,events=bind(od/'EVENTS.jsonl'),terminal=bind(od/'TERMINAL.json'),prompt=bind(od/'REQUEST.txt'),backend=terminal.get('backend'),initial_only=True))
    atomic(d/'CANDIDATE_0.json',candidate)
    result=ledger(candidate,inp,parent,native,old,raw);atomic(d/'LEDGER_0.json',result)
    return candidate,inp,parent,native,result,receipt
STOP=set('the a an to in on and or is it i you we my your our for of at with that this as have has had will would be been not'.split())
def words(text):return set(re.findall(r'[a-z]{3,}',text.lower()))-STOP
def related_parent(scene,parent,limit=18):
    terms=words(' '.join(t['text'] for t in scene['turns']))
    ranked=sorted(enumerate(parent['turns']),key=lambda it:(-len(terms & words(it[1]['text'])),it[0]))[:limit]
    return [dict(turn_ref=t['turn_ref'],speaker=t['speaker'],quote=t['text']) for _,t in sorted(ranked)]
if __name__=='__main__':
    from common import protocol
    for root in protocol()['targets']['new_train_roots']:
        *_,led,receipt=build(root);print(root,'events',receipt['accepted'],'mechanical_errors',len(led['mechanical_errors']),flush=True)
