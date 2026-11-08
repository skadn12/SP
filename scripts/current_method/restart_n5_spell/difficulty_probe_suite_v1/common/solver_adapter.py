from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .. import SUITE_VERSION


class LocalSolver:
    """Thin wrapper around the existing Qwen3-4B/vLLM Solver contract."""

    def __init__(self, run_root: Path, *, max_model_len: int = 16384):
        from transformers import AutoTokenizer
        from scripts.current_method.restart_n5_spell.vllm_backends import VLLMSolverBackend
        self.model_path = Path("/root/autodl-tmp/Qwen3-4B")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.backend = VLLMSolverBackend(model_path=self.model_path, tokenizer_path=self.model_path,
            max_model_len=max_model_len, temperature=0.7, top_p=0.9, max_tokens=32,
            gpu_memory_utilization=0.80)
        self.run_root = run_root
        self.calls = 0

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def _prompt(self, question: str, options: Mapping[str, str], turns: list[Mapping[str, Any]], retry: bool = False) -> str:
        from scripts.current_method.restart_n5_spell.prompts import (
            SOLVER_NOTHINK_RESPONDER_PROMPT, SOLVER_PARSE_RETRY_SUFFIX,
            SOLVER_TRAILING_FORMAT_INSTRUCTION,
        )
        context = "\n".join(f"[{row.get('turn_ref') or row.get('turn_id','')}] {row.get('speaker','')}: {row.get('text','')}" for row in turns)
        user = ((f"Context:\n{context}\n\n" if context else "") + question + "\n" +
                "\n".join(f"{key}. {value}" for key, value in options.items()) +
                "\n\n" + SOLVER_TRAILING_FORMAT_INSTRUCTION +
                (("\n\n" + SOLVER_PARSE_RETRY_SUFFIX) if retry else ""))
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": SOLVER_NOTHINK_RESPONDER_PROMPT},
            {"role": "user", "content": user}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def run(self, qa: Mapping[str, Any], turns: list[Mapping[str, Any]], *, mode: str,
            requested: int = 5, seed_base: int = 0) -> list[dict[str, Any]]:
        from scripts.current_method.restart_n5_spell.core import parse_boxed_mcq_answer, local_reward
        rows = []
        for slot in range(requested):
            seed = int(hashlib.sha256(f"{qa.get('qa_id')}:{mode}:{slot}:{seed_base}".encode()).hexdigest()[:8], 16)
            prompt = self._prompt(str(qa.get("question")), dict(qa.get("options") or {}), turns)
            from scripts.current_method.restart_n5_spell.backends import SolverRequest
            request = SolverRequest(f"difficulty-{qa.get('qa_id')}-{mode}-s{slot}", str(qa.get("qa_id")), mode,
                "original", slot, 0, seed, dict(qa.get("options") or {}), str(qa.get("gold_text")),
                hashlib.sha256(prompt.encode()).hexdigest(), prompt)
            response = self.backend.generate(request)
            self.calls += 1
            parsed = parse_boxed_mcq_answer(response.raw_text, dict(qa.get("options") or {}))
            reward = local_reward(parsed, str(qa.get("gold_text")))
            rows.append({"qa_id": qa.get("qa_id"), "mode": mode, "slot": slot,
                         "raw_response": response.raw_text, **parsed, "local_reward": reward,
                         "sampling_seed": seed, "backend_metadata": response.metadata})
        return rows


def maybe_solver(run_root: Path) -> tuple[LocalSolver | None, str | None]:
    try:
        return LocalSolver(run_root), None
    except Exception as exc:  # A report is still useful on CPU-only hosts.
        return None, f"solver_unavailable:{type(exc).__name__}:{exc}"

