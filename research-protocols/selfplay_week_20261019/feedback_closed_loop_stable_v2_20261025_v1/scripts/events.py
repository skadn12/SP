"""Fail-closed Codex event receipt. No semantic verdicts are manufactured."""
import json,re
from common import sha
EXACT_WARNING = ('Codex is ignoring 3 unrecognized configuration settings. Check for typos or deprecated settings.\n'
 '  user (/root/.codex/config.toml): `disable_response_storage` is ignored.\n'
 '  user (/root/.codex/config.toml): `network_access` is ignored.\n'
 '  user (/root/.codex/config.toml): `windows_wsl_setup_acknowledged` is ignored.')
RECONNECT = re.compile(r'Reconnecting\.\.\. [1-5]/5 \(stream disconnected before completion: stream closed before response\.completed\)')
def classify(raw_events, final, returncode=0, timed_out=False):
    errors=[]; rows=[]; events=[]; messages=[]; completions=[]; starts=[]; threads=[]; semantic=False
    if returncode: errors.append('nonzero_exit:'+str(returncode))
    if timed_out: errors.append('timeout')
    try:
        for line in raw_events.splitlines():
            e=json.loads(line)
            if not isinstance(e,dict): raise ValueError('nonobject_event')
            events.append(e)
        if not events: raise ValueError('empty_stream')
    except (ValueError,TypeError) as exc: errors.append('invalid_event_json:'+str(exc))
    for i,e in enumerate(events):
        kind=e.get('type'); cls='REJECT'; why=None
        if kind=='thread.started' and isinstance(e.get('thread_id'),str):
            threads.append(i); cls='thread_start'
        elif kind=='turn.started': starts.append(i); cls='turn_start'
        elif kind=='turn.completed':
            completions.append(i); cls='turn_completion'
            u=e.get('usage')
            if not isinstance(u,dict) or not all(type(u.get(k)) is int and u[k]>=0 for k in ['input_tokens','output_tokens']): why='missing_or_invalid_completion_usage'
        elif kind=='error' and isinstance(e.get('message'),str) and RECONNECT.fullmatch(e['message']): cls='conditional_prefinal_reconnect'
        elif kind=='item.completed':
            item=e.get('item',{}); it=item.get('type') if isinstance(item,dict) else None
            if it=='error' and item.get('message')==EXACT_WARNING: cls='conditional_exact_local_warning'
            elif it in ['reasoning','agent_message'] and isinstance(item.get('text'),str):
                cls=it
                if it=='agent_message': messages.append(i); semantic=True
            else: why='forbidden_or_unknown_item:'+str(it)
        else:
            item=e.get('item',{})
            if isinstance(item,dict) and item.get('type')=='agent_message': semantic=True
            why='forbidden_or_unknown_event:'+str(kind)
        if why: errors.append(f'event_{i}:{why}'); cls='REJECT'
        rows.append(dict(index=i,classification=cls,event_sha256=sha(json.dumps(e,sort_keys=True).encode())))
    if len(threads)!=1: errors.append('requires_exactly_one_thread')
    if len(starts)!=1: errors.append('requires_exactly_one_turn_start')
    if len(messages)!=1: errors.append('requires_exactly_one_final_agent_message')
    if len(completions)!=1: errors.append('requires_exactly_one_completion')
    obj=None
    try:
        obj=json.loads(final)
        if not isinstance(obj,dict): errors.append('final_not_json_object')
    except (ValueError,TypeError): errors.append('missing_or_invalid_final_json')
    if final.strip(): semantic=True
    if len(messages)==len(completions)==len(starts)==len(threads)==1:
        m,c,s,t=messages[0],completions[0],starts[0],threads[0]
        if not t<s<m<c or c!=len(events)-1: errors.append('invalid_lifecycle_order')
        if events[m]['item']['text'].strip()!=final.strip(): errors.append('final_agent_message_mismatch')
        for row in rows:
            if row['classification']=='conditional_prefinal_reconnect' and not s<row['index']<m: errors.append('reconnect_not_prefinal')
            if row['classification']=='conditional_exact_local_warning' and row['index']>=m: errors.append('warning_not_prefinal')
    return dict(schema_version='strict_event_receipt_v2',accepted=not errors, failure_reasons=sorted(set(errors)),
                semantic_response_present=semantic, response=obj, classes=rows, event_count=len(events),
                events_sha256=sha(raw_events.encode()), final_sha256=sha(final.encode()),
                thread_id=events[threads[0]]['thread_id'] if len(threads)==1 else None,
                usage=events[completions[0]].get('usage') if len(completions)==1 else None)
