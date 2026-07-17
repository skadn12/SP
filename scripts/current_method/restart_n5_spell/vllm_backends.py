from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .backends import BackendResponse, SolverRequest


class VLLMSolverBackend:
    """Raw local generation only; parsing, retry, reward and audits stay in the orchestrator."""

    def __init__(self, *, model_path: Path, tokenizer_path: Path, max_model_len: int, temperature: float = 0.7,
                 top_p: float = 0.9, max_tokens: int = 32,
                 gpu_memory_utilization: float = 0.80, dtype: str = "auto"):
        from vllm import LLM

        self.model_path = model_path.resolve()
        self.tokenizer_path = tokenizer_path.resolve()
        self.model_identifier = str(self.model_path)
        self.max_model_len = int(max_model_len)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        started = time.perf_counter()
        self.engine = LLM(model=str(self.model_path), tokenizer=str(self.tokenizer_path), tensor_parallel_size=1,
            max_model_len=self.max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype, trust_remote_code=True, enable_prefix_caching=False)
        self.load_latency_seconds = time.perf_counter() - started

    def generate(self, request: SolverRequest) -> BackendResponse:
        from vllm import SamplingParams

        params = SamplingParams(temperature=self.temperature, top_p=self.top_p,
                                max_tokens=self.max_tokens, seed=request.sampling_seed)
        started = time.perf_counter()
        outputs = self.engine.generate([request.prompt], params, use_tqdm=False)
        latency = time.perf_counter() - started
        output = outputs[0]
        completion = output.outputs[0]
        metadata: dict[str, Any] = {"sampling_seed":request.sampling_seed,
            "latency_seconds":latency, "model_load_latency_seconds":self.load_latency_seconds,
            "prompt_token_count":len(output.prompt_token_ids),
            "completion_token_count":len(completion.token_ids),
            "finish_reason":completion.finish_reason, "max_model_len":self.max_model_len,
            "temperature":self.temperature, "top_p":self.top_p, "max_tokens":self.max_tokens}
        metadata.update({"tokenizer_path":str(self.tokenizer_path), "enable_thinking":False,
                         "add_generation_prompt":True, "stop_sequences":[]})
        return BackendResponse(completion.text, self.model_identifier, metadata)
