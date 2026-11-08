"""Read-only inherited constructor API inspection; never instantiate LLM."""
import inspect
import json
import sys
import importlib.metadata as metadata
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs

signature = inspect.signature(LLM)
names = ('model', 'tokenizer', 'trust_remote_code', 'dtype', 'max_model_len', 'gpu_memory_utilization', 'seed', 'enable_prefix_caching')
fields = EngineArgs.__dataclass_fields__
found = {name: {'LLM_parameter': name in signature.parameters, 'EngineArgs_field': name in fields} for name in names}
print(json.dumps({'executable': sys.executable, 'vllm_version': metadata.version('vllm'),
                  'LLM_module': inspect.getfile(LLM),
                  'LLM_has_kwargs': any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()),
                  'historical_LLM_keywords': found, 'LLM_instantiated': False,
                  'model_loaded': False, 'tokenizer_loaded': False, 'model_calls': 0}))
