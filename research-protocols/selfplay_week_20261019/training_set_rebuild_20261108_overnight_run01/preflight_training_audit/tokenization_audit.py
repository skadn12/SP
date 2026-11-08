import json, pathlib, hashlib, collections, sys
from transformers import AutoTokenizer
root=pathlib.Path('/root/autodl-tmp')
data=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl')
model=root/'Qwen3-4B'; max_length=12288
out=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261108_overnight_run01/preflight_training_audit/tokenization_audit.json')
tok=AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

def render(msgs, add_generation_prompt):
    if hasattr(tok,'apply_chat_template') and tok.chat_template:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
    system=msgs[0]['content'] if msgs and msgs[0].get('role')=='system' else ''
    user=next((m['content'] for m in msgs if m.get('role')=='user'),'')
    assistant=next((m['content'] for m in msgs if m.get('role')=='assistant'),'')
    prompt=f'System: {system}\n\nUser: {user}\n\nAssistant:'
    return prompt if add_generation_prompt else f'{prompt} {assistant}'
recs=[json.loads(x) for x in data.read_text().splitlines() if x.strip()]
rows=[]; failures=[]
for i,r in enumerate(recs):
    msgs=r['messages']; pm=[m for m in msgs if m.get('role')!='assistant']
    full_text=render(msgs,False); prompt_text=render(pm,True)
    full=tok(full_text,add_special_tokens=False).input_ids
    prompt=tok(prompt_text,add_special_tokens=False).input_ids
    appended=False
    if tok.eos_token_id is not None and (not full or full[-1]!=tok.eos_token_id): full.append(tok.eos_token_id); appended=True
    overlap=min(len(full),len(prompt)); prefix=(full[:overlap]==prompt[:overlap])
    # same exact trainer masking
    labels=[-100]*overlap+full[overlap:]
    overflow=max(0,len(full)-max_length)
    if overflow: full=full[overflow:]; labels=labels[overflow:]
    label_count=sum(x!=-100 for x in labels)
    row={'index':i,'candidate_id':r.get('candidate_id'),'root_id':r.get('root_id'),'full_tokens_pretrim':len(full)+overflow,'prompt_tokens':len(prompt),'full_tokens':len(full),'overflow':overflow,'gold_tokens':label_count,'prompt_prefix_match':prefix,'eos_appended':appended}
    rows.append(row)
    if not prefix or overflow or label_count==0: failures.append(row)
summary={'data':str(data),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'n':len(recs),'max_length':max_length,'tokenizer_path':str(model),'tokenizer_chat_template':bool(tok.chat_template),'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,'min_full_tokens':min(x['full_tokens_pretrim'] for x in rows),'max_full_tokens':max(x['full_tokens_pretrim'] for x in rows),'min_prompt_tokens':min(x['prompt_tokens'] for x in rows),'max_prompt_tokens':max(x['prompt_tokens'] for x in rows),'min_gold_tokens':min(x['gold_tokens'] for x in rows),'max_gold_tokens':max(x['gold_tokens'] for x in rows),'overflow_count':sum(x['overflow']>0 for x in rows),'prefix_mismatch_count':sum(not x['prompt_prefix_match'] for x in rows),'all_labels_masked_count':sum(x['gold_tokens']==0 for x in rows),'failures':failures,'rows':rows}
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:summary[k] for k in ['n','data_sha256','tokenizer_chat_template','pad_token_id','eos_token_id','min_full_tokens','max_full_tokens','max_prompt_tokens','min_gold_tokens','max_gold_tokens','overflow_count','prefix_mismatch_count','all_labels_masked_count']},ensure_ascii=False,indent=2))
if failures: print('FAILURES',json.dumps(failures[:10],ensure_ascii=False,indent=2)); sys.exit(1)
