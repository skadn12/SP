"""Local, final-message counting; role limits do not change data/solver policy."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

VERSION = "incremental_run_readiness_v2"
MODEL = "Qwen/Qwen3-14B"
TOKENIZER_PATH = "/root/autodl-tmp/Qwen3-4B"
TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MODEL_TOKENIZER_SOURCE = "https://huggingface.co/Qwen/Qwen3-14B/blob/d17b97ca036704c156b50686b3f4551e224bbe03/tokenizer.json"
API_ROLES = ("planner", "supplement", "gate", "realizer", "adherence")


class LocalCounter:
    def __init__(self, path=TOKENIZER_PATH, expected_sha=TOKENIZER_SHA):
        from transformers import AutoTokenizer
        import transformers
        from tokenizers.pre_tokenizers import ByteLevel
        path = Path(path)
        actual = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        if actual != expected_sha:
            raise ValueError("local_tokenizer_sha_mismatch")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        vocab = json.loads((path / "tokenizer.json").read_text())["model"]["vocab"]
        # ByteLevel vocabulary characters encode bytes, including 128-space tokens.
        alphabet = set(ByteLevel.alphabet())
        if any(not set(token) <= alphabet for token in vocab):
            raise ValueError("unsupported_non_bytelevel_vocabulary")
        self.max_token_bytes = max(map(len, vocab))
        self.identity = {"path": str(path), "tokenizer_sha256": actual,
                         "tokenizer_config_sha256": hashlib.sha256((path / "tokenizer_config.json").read_bytes()).hexdigest(),
                         "chat_template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                         "transformers_version": transformers.__version__,
                         "method": "apply_chat_template(tokenize=True, add_generation_prompt=True, enable_thinking=False)",
                         "model_tokenizer_source": MODEL_TOKENIZER_SOURCE,
                         "model_vocabulary_identity": "official_14B_SHA_matches_local_vocabulary",
                         "provider_chat_template_identity": "NOT_TESTED", "max_vocabulary_token_bytes": self.max_token_bytes}

    def count_tokens(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False))

    def count_text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def dialogue_tokens(self, turns):
        return self.count_text(json.dumps(turns, ensure_ascii=False, sort_keys=True))


@lru_cache(maxsize=4)
def local_counter(path=TOKENIZER_PATH, expected_sha=TOKENIZER_SHA):
    return LocalCounter(path, expected_sha)


def role_capacity(budget, role):
    if "role_capacity" in budget:
        return budget["role_capacity"][role]
    # Historical offline fixtures remain reproducible; never used as a live default.
    return {"model": MODEL, "context_window_tokens": budget["context_window_tokens"],
            "max_input_tokens": budget["max_input_tokens"], "thinking_reserve_tokens": 0,
            "safety_margin_tokens": 0, "capacity_source": "legacy_offline_fixture_only"}


def measure(messages, *, counter, budget, role, future_tokens=0):
    capacity = role_capacity(budget, role)
    known = counter.count_tokens(messages)
    output = budget["max_output_tokens"][role]
    total = known + future_tokens + output + capacity["thinking_reserve_tokens"] + capacity["safety_margin_tokens"]
    over = max(0, total - capacity["context_window_tokens"],
               known + future_tokens - capacity["max_input_tokens"])
    return {"role": role, "known_final_messages_tokens": known,
            "future_content_bound_tokens": future_tokens, "output_reserve_tokens": output,
            "thinking_reserve_tokens": capacity["thinking_reserve_tokens"],
            "safety_margin_tokens": capacity["safety_margin_tokens"],
            "chat_wrapper_overhead": "already_counted_once_in_final_messages",
            "context_window_tokens": capacity["context_window_tokens"],
            "max_input_tokens": capacity["max_input_tokens"], "total_reserved_tokens": total,
            "excess_tokens": over, "fits": over == 0,
            "http_json_bytes_DIAGNOSTIC_ONLY": len(json.dumps(messages, ensure_ascii=False).encode()),
            "counting_identity": counter.identity}


def proposed_budget():
    outputs = {"planner": 8192, "supplement": 8192, "gate": 2048,
               "realizer": 8192, "adherence": 4096, "closure": 1}
    roles = {"planner": 0, "supplement": 1, "gate": 8, "realizer": 4, "adherence": 4, "closure": 1}
    capacity = {role: {"model": MODEL, "context_window_tokens": 128000,
                      "max_input_tokens": 128000 - outputs[role] - 256,
                      "thinking_reserve_tokens": 0, "safety_margin_tokens": 256,
                      "capacity_source": "user_supplier_page_128K_conservative_local_128000_not_server_verified"}
                for role in API_ROLES}
    return {"total_requests": 18, "role_requests": roles,
            "max_input_tokens": 128000 - 8192 - 256,
            "context_window_tokens": 128000, "max_output_tokens": outputs,
            "max_generated_tokens": sum(roles[r] * outputs[r] for r in roles),
            "max_cumulative_tokens": 100000, "scene_output_reserve_tokens": 4096,
            "role_capacity": capacity, "readiness_version": VERSION,
            "cumulative_length_measure": "local_tokenizer_canonical_dialogue_json",
            "solver_training_limits": "UNCHANGED_NOT_CONFIGURED_BY_THIS_BUDGET"}


def future_audit_reserve(*, parent, plan, conditions, budget, counter, prompt_version=None):
    from .retention import render_adherence, ADHERENCE_VERSION, LEGACY_ADHERENCE_VERSION
    prompt_version = ADHERENCE_VERSION if prompt_version is None else prompt_version
    child = {"root_id": parent["root_id"], "round_index": 3,
             "schema_version": "memory_event_continuation_child_v1", "scenes": []}
    for scene in plan["scenes"]:
        ref = scene["scene_ref"]
        child["scenes"].append({"scene_ref": ref, "session_ref": ref,
            "memory_event_id": scene["memory_event_id"],
            "dialogue_turns": [{"turn_ref": f"{ref}:T{i:03d}", "speaker": "", "text": ""} for i in range(1, 49)]})
    if prompt_version == LEGACY_ADHERENCE_VERSION:
        child["schema_version"] = "expansion_child_v1"
        prompt = render_adherence(parent=parent, plan=plan, child=child,
                                  requested_conditions=conditions, generated_response="",
                                  prompt_version=prompt_version)
        boundary = 2 * counter.max_token_bytes
        measured = measure(prompt["messages"], counter=counter, budget=budget, role="adherence",
                           future_tokens=budget["max_output_tokens"]["realizer"] + boundary)
        measured.update(future_response_output_cap=budget["max_output_tokens"]["realizer"],
                        raw_message_boundary_reserve=boundary, maximum_turns_per_scene=48,
                        future_content_method="exact_raw_response_once_same_tokenizer_plus_boundary_reserve",
                        unknown="actual response content and provider chat-template overhead; actual messages rechecked before audit")
        return measured
    if set(conditions) - {scene["scene_ref"] for scene in child["scenes"]}:
        raise ValueError("future_audit_conditions_unknown_scene")
    raw_skeleton = {"scenes": [{"dialogue_turns": [{} for _ in scene["dialogue_turns"]]} for scene in child["scenes"]]}
    prompt = render_adherence(parent=parent, plan=plan, child=child,
        requested_conditions={scene["scene_ref"]: conditions.get(scene["scene_ref"], []) for scene in child["scenes"]},
        generated_response=raw_skeleton, prompt_version=prompt_version)
    # Each raw turn remains one contiguous, unescaped JSON segment. Reserve
    # both ends when extracting it and when placing it beside its assigned ID.
    # All per-turn wrappers and the maximum 48 IDs/scene are already counted.
    segments = 48 * len(child["scenes"])
    boundary = (4 * segments + 2) * counter.max_token_bytes
    measured = measure(prompt["messages"], counter=counter, budget=budget, role="adherence",
                       future_tokens=budget["max_output_tokens"]["realizer"] + boundary)
    measured.update(future_response_output_cap=budget["max_output_tokens"]["realizer"],
                    raw_segment_boundary_reserve=boundary, maximum_turns_per_scene=48,
                    maximum_raw_turn_segments=segments,
                    adherence_prompt_version=prompt["prompt_version"],
                    future_content_method="raw_turn_segments_once_same_tokenizer_plus_per_segment_boundaries",
                    unknown="actual response content and provider chat-template overhead; actual messages rechecked before audit")
    return measured
