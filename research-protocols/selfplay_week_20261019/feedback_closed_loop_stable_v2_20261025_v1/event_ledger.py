"""Fail-closed event reconstruction and deterministic, quote-backed source facts.

Canonical facts are normalized REPORTED ASSERTIONS, not inferred world facts.
The complete text, speaker, scope, negations and modality are retained. Native
utterances and author mappings are separate so a mapping is never its own audit.
"""
from __future__ import annotations
import json, re, statistics
from common import *

WARNING = ('Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n'
 '  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n'
 '  user (/root/.codex/config.toml): `network_access` is ignored.\n'
 '  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored.')
RECONNECT = re.compile(r'Reconnecting\.\.\. ([1-5])/5 \(stream disconnected before completion: stream closed before response\.completed\)\Z')

def classify(events, final, *, returncode=0, timed_out=False):
    reasons=[]; classes=[]; parsed=[]; obj=None
    try:
        obj=json.loads(final)
        if not isinstance(obj,(dict,list)): reasons.append('final_json_not_object_or_list')
    except (ValueError,TypeError): reasons.append('missing_or_invalid_final_json')
    if isinstance(events,str):
        for i,line in enumerate(events.splitlines()):
            try:
                e=json.loads(line)
                if not isinstance(e,dict): raise ValueError('not object')
                parsed.append(e)
            except ValueError: reasons.append('malformed_jsonl:%d'%i)
    else: parsed=list(events or [])
    if not parsed: reasons.append('missing_events')
    messages=[]; starts=[]; turnstarts=[]; completes=[]; reconnects=[]; warnings=[]
    for i,e in enumerate(parsed):
        kind='REJECT';typ=e.get('type') if isinstance(e,dict) else None
        if isinstance(e,dict) and 'item' in e and typ!='item.completed':
            reasons.append('item_on_unknown_lifecycle:%s'%i)
        if typ=='thread.started' and isinstance(e.get('thread_id'),str): starts.append(i);kind='thread_start'
        elif typ=='turn.started': turnstarts.append(i);kind='turn_start'
        elif typ=='turn.completed':
            usage=e.get('usage',{})
            if (isinstance(usage,dict) and all(isinstance(usage.get(k),int) and not isinstance(usage[k],bool) and usage[k]>=0 for k in ('input_tokens','output_tokens')) and usage['output_tokens']>0):
                completes.append(i);kind='turn_complete_with_usage'
            else: reasons.append('invalid_completion_usage:%d'%i)
        elif typ=='error' and isinstance(e.get('message'),str) and RECONNECT.fullmatch(e['message']):
            reconnects.append(i);kind='conditional_prefinal_reconnect'
        elif typ=='item.completed' and isinstance(e.get('item'),dict):
            item=e['item'];it=item.get('type')
            if it=='error' and item.get('message')==WARNING: warnings.append(i);kind='conditional_exact_config_warning'
            elif it in ('reasoning','agent_message') and isinstance(item.get('text'),str):
                kind=it
                if it=='agent_message': messages.append(i)
            else: reasons.append('forbidden_or_unknown_item:%s:%s'%(i,it))
        else: reasons.append('unknown_event_or_error:%s:%s'%(i,typ))
        classes.append({'index':i,'class':kind,'event_sha256':sha(json_compact(e).encode())})
    if len(starts)!=1 or len(turnstarts)!=1 or len(completes)!=1: reasons.append('require_exactly_one_thread_turn_and_completion')
    if not messages: reasons.append('missing_final_agent_message')
    if messages:
        mi=messages[-1]
        if parsed[mi]['item']['text'].strip()!=str(final).strip(): reasons.append('saved_final_agent_message_mismatch')
        if any(i>=mi for i in reconnects): reasons.append('reconnect_after_final')
        if turnstarts and any(i<=turnstarts[0] for i in reconnects):reasons.append('reconnect_outside_active_turn')
        if any(i>=mi for i in warnings): reasons.append('warning_after_final')
        if completes and not (turnstarts and starts and starts[0]<turnstarts[0]<mi<completes[0]==len(parsed)-1): reasons.append('invalid_event_lifecycle_order')
    if returncode: reasons.append('nonzero_exit:%s'%returncode)
    if timed_out: reasons.append('timed_out')
    reasons=sorted(set(reasons))
    return {'schema_version':'stable_v2_strict_events','accepted':not reasons,'failure_reasons':reasons,
            'event_classes':classes,'exact_warnings':warnings,'prefinal_reconnects':reconnects,
            'thread_id':parsed[starts[0]]['thread_id'] if len(starts)==1 else None,
            'usage':parsed[completes[0]]['usage'] if len(completes)==1 else None,
            'final_sha256':sha(str(final).encode()),'events_sha256':sha(json_compact(parsed).encode()),
            'complete_final_json':isinstance(obj,(dict,list)),'matching_agent_message':bool(messages) and 'saved_final_agent_message_mismatch' not in reasons}

def turn_fact(t,scope):
    text=t['text'];norm=' '.join(text.split())
    return {'fact_id':t['turn_ref'],'turn_id':t['turn_ref'],'scope':scope,'speaker':t['speaker'],
            'span':{'char_start':0,'char_end':len(text),'quote':text},
            'canonical_fact':{'kind':'reported_assertion','reporting_speaker':t['speaker'],'proposition':norm},
            'explicit_clocks':re.findall(r'\b\d{1,2}:\s*\d{2}\b',text),
            'explicit_state_tokens':re.findall(r"\b(?:proposed|propose|planned|plan|agreed|agree|confirmed|completed|cancelled|declined|unsent|unconfirmed|haven't|hasn't|not|never|before|after|because)\b",text,re.I)}

def mechanical(candidate,inp,parent,native):
    errors=[];root=inp['root_id']
    def err(scene,code,fact_id=None): errors.append({'scene_ref':scene,'code':code,'fact_id':fact_id})
    if candidate.get('root_id')!=root: err(None,'root_id_mismatch')
    if candidate.get('source_status')!=inp['source_status']: err(None,'source_status_mismatch')
    scenes=candidate.get('scenes',[])
    if [s.get('scene_ref') for s in scenes]!=inp['scene_sequence']:err(None,'scene_sequence_mismatch')
    refs={};texts=[];speakers={t['speaker'] for t in parent['turns']}
    for s in scenes:
        sr=s.get('scene_ref');ts=s.get('turns',[])
        if len(ts)!=8:err(sr,'not_eight_turns')
        if s.get('route')!=sr.split(':')[1]:err(sr,'route_mismatch')
        for i,t in enumerate(ts,1):
            ref=t.get('turn_ref')
            if ref!=f'{sr}:T{i:03d}':err(sr,'turn_id_mismatch',ref)
            if ref in refs:err(sr,'duplicate_turn_id',ref)
            refs[ref]=t
            if t.get('speaker') not in speakers:err(sr,'new_speaker',ref)
            if t.get('origin')!='codex_synthetic_extension':err(sr,'origin_mismatch',ref)
            if not isinstance(t.get('text'),str) or not t['text'].strip():err(sr,'empty_text',ref)
            else:texts.append(t['text'])
    if len(texts)!=48:err(None,'not_48_turns')
    if texts and statistics.mean(len(t.split()) for t in texts)>18:err(max(scenes,key=lambda s:sum(len(t['text'].split()) for t in s['turns']))['scene_ref'],'mean_words_above_18')
    nts={(n['source_id'],str(t['turn_id'])):(n['scene_ref'],t['utterance']) for n in native for t in n['turns']}
    seen=set()
    for m in candidate.get('native_evidence',[]):
        key=(m.get('source_id'),str(m.get('native_turn_id')));src=nts.get(key);sr=src[0] if src else None
        if key in seen:err(sr,'duplicate_native_mapping',str(key))
        seen.add(key)
        if not src or not m.get('native_quote') or m['native_quote'] not in src[1]:err(sr,'native_quote_mismatch',str(key))
        if m.get('disposition') not in ('PRESERVED','OMITTED_NONMATERIAL'):err(sr,'invalid_native_disposition',str(key))
        if not m.get('rationale'):err(sr,'missing_native_rationale',str(key))
        if m.get('disposition')=='PRESERVED' and not m.get('source_evidence'):err(sr,'missing_source_mapping',str(key))
        for q in m.get('source_evidence',[]):
            t=refs.get(q.get('turn_ref'))
            if not t or not q.get('quote') or q['quote'] not in t['text'] or not q['turn_ref'].startswith(str(sr)+':'):
                err(sr,'source_span_mapping_mismatch',q.get('turn_ref'))
    for key in nts.keys()-seen:err(nts[key][0],'missing_native_mapping',str(key))
    return errors

def inputs(root):
    p=PRE/'inputs'/f'{root}_SOURCE_AUTHOR_INPUT.json';inp=load(p)
    verify(inp['parent_source_binding']);parent=load(inp['parent_source_binding']['path']);native=[]
    for n in inp['native_events']:
        verify(n['binding']);x=load(n['binding']['path'])
        native.append({'source_id':n['source_id'],'scene_ref':n['scene_ref'],'turns':[{k:t[k] for k in ('turn_id','speaker','utterance')} for t in x['turns']]})
    return inp,parent,native

def build(root,candidate,revision):
    inp,parent,native=inputs(root)
    facts=[turn_fact(t,'parent') for t in parent['turns']]
    facts += [turn_fact(t,s['scene_ref']) for s in candidate['scenes'] for t in s['turns']]
    return {'schema_version':'stable_v2_fact_ledger','root_id':root,'revision':revision,
      'canonicalization':'whitespace-normalized reported assertions; no inferred status, role, date or causal links',
      'facts':facts,'native_utterances':native,'native_mappings':candidate.get('native_evidence',[]),
      'mechanical_failures':mechanical(candidate,inp,parent,native),
      'parent_binding':inp['parent_source_binding'],'source_status':inp['source_status'],'warnings':candidate.get('warnings',[])}

def initial_ledger_with_byte_spans(root):
    """Immutable-draft token offsets are UTF-8 bytes, separate from decoded chars."""
    old=PRE/'cli_calls'/(root.replace('-','_')+'_source_draft')/'attempt_01/LAST_MESSAGE.txt'
    raw=old.read_text();candidate=json.loads(raw);ledger=build(root,candidate,0)
    toks=[]
    for m in re.finditer(r'"(?:\\.|[^"\\])*"',raw):
        toks.append((m,json.loads(m.group())))
    locations=[]
    for i,(m,value) in enumerate(toks[:-1]):
        n,v=toks[i+1]
        if value=='text' and raw[m.end():n.start()].strip()==':':
            locations.append((v,{'start_inclusive':len(raw[:n.start()].encode()),'end_exclusive':len(raw[:n.end()].encode()),'coordinate':'UTF8_JSON_STRING_TOKEN_IN_IMMUTABLE_DRAFT'}))
    fs=[f for f in ledger['facts'] if f['scope']!='parent']
    if [f['span']['quote'] for f in fs]!=[v for v,_ in locations]:raise ValueError('draft_lexical_text_mapping_mismatch')
    for f,(_,span) in zip(fs,locations):f['immutable_draft_byte_span']=span
    ledger['immutable_draft_binding']=bind(old)
    byref={f['turn_id']:f for f in fs}
    native={(n['source_id'],str(t['turn_id'])):t['utterance'] for n in ledger['native_utterances'] for t in n['turns']}
    expanded=[]
    for m in ledger['native_mappings']:
        row=dict(m);text=native.get((m['source_id'],str(m['native_turn_id'])),'');q=m['native_quote']
        if q and q in text:
            start=text.index(q);row['native_character_span']={'start_inclusive':start,'end_exclusive':start+len(q)}
        row['source_spans']=[]
        for ev in m.get('source_evidence',[]):
            f=byref.get(ev['turn_ref']);q=ev['quote']
            if f and q in f['span']['quote']:
                start=f['span']['quote'].index(q);row['source_spans'].append({'turn_id':ev['turn_ref'],'quote':q,'char_start':start,'char_end':start+len(q)})
        expanded.append(row)
    ledger['native_mapping_spans']=expanded
    return ledger

def recover(root):
    d=BASE/'sources'/root;old=PRE/'cli_calls'/(root.replace('-','_')+'_source_draft')/'attempt_01'
    frozen=load(next(b['path'] for b in load(BASE/'INPUT_BINDINGS.json')['bindings'] if b['path'].endswith('/SOURCE_FREEZE.json')))
    binding=next(x['fixed_initial_draft'] for x in frozen['outcomes'] if x['root_id']==root)
    verify(binding)
    if bind(old/'LAST_MESSAGE.txt')!=binding:raise ValueError('fixed_initial_draft_selection_mismatch')
    raw=(old/'LAST_MESSAGE.txt').read_text();term=load(old/'TERMINAL.json')
    c=classify((old/'EVENTS.jsonl').read_text(),raw,returncode=term.get('returncode',1),timed_out=term.get('timed_out',False))
    c['immutable_evidence']={n:bind(old/n) for n in ('REQUEST.txt','EVENTS.jsonl','LAST_MESSAGE.txt','TERMINAL.json')}
    c['predecessor_backend']=term.get('backend')
    atomic(d/'INITIAL_EVENT_RECEIPT.json',c)
    if not c['accepted']:return c
    candidate=json.loads(raw);atomic(d/'CANDIDATE_0.json',candidate);atomic(d/'LEDGER_0.json',build(root,candidate,0))
    return c

def compact(ledger,scope=None):
    return [[f['turn_id'],f['speaker'],f['canonical_fact']['proposition']] for f in ledger['facts'] if scope is None or f['scope']==scope]

if __name__=='__main__':
    for root in ROOTS:
        c=recover(root);print(root,c['accepted'],c['failure_reasons'],flush=True)
