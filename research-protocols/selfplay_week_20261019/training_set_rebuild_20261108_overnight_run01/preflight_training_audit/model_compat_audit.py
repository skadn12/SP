import json, pathlib, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
p=pathlib.Path('/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261108_overnight_run01/preflight_training_audit/model_compat_audit.json')
trainer_path='/root/autodl-tmp/gxy-selfplay/current_method_github'
sys.path.insert(0,trainer_path)
import train_qwen3_lora_sft as t
model_path='/root/autodl-tmp/Qwen3-4B'; adapter='/root/autodl-tmp/gxy-selfplay/restart-n5-runs/h4_round0_b0_formal_training_v1_live_2/clean_adapter'; data='/root/autodl-tmp/gxy-selfplay/research-protocols/selfplay_week_20261019/training_set_rebuild_20261105_F_supplement_run01/repaired_training_export_v1/TRAIN.jsonl'
# CPU-safe compatibility load; no optimizer/backward/training
out={'torch_version':torch.__version__,'cuda_available':torch.cuda.is_available(),'cuda_device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
tok=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
record=t.load_jsonl(data)[0]
feat=t.tokenize_record(tok,record,12288)
out.update({'sample_input_tokens':len(feat['input_ids']),'sample_gold_tokens':sum(x!=-100 for x in feat['labels']),'tokenizer_vocab_size':len(tok)})
dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=dtype,trust_remote_code=True)
t.freeze_base_model(model); targets=set(t.DEFAULT_TARGET_MODULES.split(',')); replaced=t.inject_lora(model,targets,16,32,.05)
# create namespace for adapter check
args=type('A',(),{'target_modules':t.DEFAULT_TARGET_MODULES,'lora_rank':16,'lora_alpha':32})()
ainfo=t.load_initial_adapter(model,adapter,args)
out.update({'target_modules_replaced':replaced,'adapter_loaded':True,'adapter_info':ainfo,'base_model_config':{'model_type':model.config.model_type,'architectures':model.config.architectures,'hidden_size':model.config.hidden_size,'num_layers':model.config.num_hidden_layers,'vocab_size':model.config.vocab_size}})
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); model.eval()
with torch.no_grad():
    batch=t.collate([feat],tok.pad_token_id,device); y=model(**batch)
out.update({'forward_ok':True,'forward_loss_finite':bool(torch.isfinite(y.loss).item()),'forward_loss':float(y.loss.detach().cpu()),'device':str(device)})
p.write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
