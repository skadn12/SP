"""Blind, fail-closed model audit of the frozen v28 multi-select drafts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "h4_crossroot_multiselect_batch_v28_newroots_quota4to8"
CONFIG = Path("/root/.config/gxy-selfplay/round1_strong_judge.env")
ROOTS = ("194259", "210046", "231899", "96544")
MODEL = "tju-llm"
ENDPOINT = "https://ai.tju.edu.cn/api/v3/chat/completions"
MAX_TOKENS = 6144
CONTRADICTION_CUE = re.compile(
    r"\b(?:not|no|never|none|without|only|instead|rather|less|cannot|can't|"
    r"didn't|doesn't|won't|wouldn't|isn't|wasn't|other than)\b", re.I)
REFERENT_CUE = re.compile(r"\b(?:it|this|that|her|him|them|they|these|those)\b", re.I)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        if isinstance(value, str):
            handle.write(value)
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def source_dir(suffix: str) -> Path:
    if suffix not in ROOTS:
        raise ValueError("unknown_root")
    return SOURCE / f"cc-train-episode-{suffix}" / "batch-01"


def input_for(suffix: str) -> tuple[dict, dict]:
    directory = source_dir(suffix)
    draft = read(directory / "MODEL_DRAFT.json")
    checks = read(directory / "MECHANICAL_CHECKS.json")
    source = read(directory / "source.json")
    dedup = read(directory / "DEDUP_SUMMARY.json")
    response = read(directory / "MODEL_RESPONSE.json")
    if (checks["fatal_errors"] or checks["draft_sha256"] != digest(canonical(draft))
            or checks["response_sha256"] != digest(canonical(response))
            or draft["root_id"] != source["root_id"]):
        raise ValueError(f"frozen_input_binding_invalid:{suffix}")
    allowed = {x["candidate_id"] for x in checks["question_checks"] if x["mechanically_valid"]}
    rows = [q for q in draft["questions"] if q["candidate_id"] in allowed]
    if len(rows) != checks["mechanically_valid_count"]:
        raise ValueError(f"mechanical_count_mismatch:{suffix}")
    turns = source["turns"]
    by_ref = {turn["turn_ref"]: index for index, turn in enumerate(turns)}
    evidence_turns = {}
    for row in rows:
        for option in row["options"]:
            ref = option["turn_ref"]
            index = by_ref[ref]
            evidence_turns[ref] = {
                "turn_ref": ref, "speaker": turns[index]["speaker"],
                "full_turn": turns[index]["text"],
                "previous_turn": (turns[index - 1]["text"] if index else None),
            }
    compact_rows = [{
        "candidate_id": row["candidate_id"], "topic": row["topic"],
        "question": row["question"],
        "options": [{k: option[k] for k in ("proposition", "relation", "turn_ref", "speaker", "quote")}
                    for option in row["options"]],
    } for row in rows]
    old = [{"candidate_id": row["candidate_id"], "question_type": row["question_type"],
            "question": row["question"], "fact_texts": row["fact_texts"]}
           for row in dedup["prior_candidates_all_types_dedup_only_not_training_selection"]]
    payload = {"root_id": source["root_id"], "candidate_questions": compact_rows,
               "cited_source_turns": list(evidence_turns.values()),
               "prior_candidates_for_semantic_dedup": old}
    hashes = {name: digest((directory / name).read_bytes()) for name in
              ("MODEL_DRAFT.json", "MECHANICAL_CHECKS.json", "source.json",
               "DEDUP_SUMMARY.json", "MODEL_RESPONSE.json")}
    return payload, hashes


INSTRUCTIONS = """You are an independent content auditor, not the question writer. Return one JSON object only.
Review every candidate question in the input. Each of its six options has one quoted source turn.
Do not repair, rewrite, or infer the writer's intent. A quote is sufficient only if its own text,
with the named speaker, directly entails the COMPLETE proposition, including referent, timing,
modality (suggested/planned/done), actor, and negation. The previous turn is diagnostic context:
if it is necessary to identify an `it`, `that`, or `her`, mark quote_sufficient false.
For relation `supports`, verdict must be `supports`; for `contradicts`, the quote must directly
exclude the proposition, not merely mention an alternative. Otherwise verdict `unsupported`.
Reject a question if the stem mixes unrelated scopes, any option is unsupported, its stated
relations do not yield a trustworthy complete Gold, or its tested facts materially repeat
prior candidates (including other question types) or another draft in this batch.
For each question output exactly:
candidate_id, stem_scope_ok (boolean), novelty_ok (boolean), batch_distinct_ok (boolean),
gold_consistent (boolean), overlap_candidate_ids (array of strings), reason (brief string),
options (array of six objects in original order, each with index 1..6, verdict one of
`supports`,`contradicts`,`unsupported`, quote_sufficient boolean, reason brief string).
Top-level object: {"reviews": [...]} and no other keys. Use false/unsupported when uncertain.
Do not treat all-six-true Gold as an error by itself. Do not use question-writer rationales.
"""


def prepare(suffix: str) -> dict:
    payload, hashes = input_for(suffix)
    directory = HERE / payload["root_id"]
    if directory.exists():
        raise FileExistsError(directory)
    prompt = INSTRUCTIONS + "\nINPUT JSON:\n" + json.dumps(payload, ensure_ascii=False)
    request = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False}
    write_new(directory / "INPUT.json", payload)
    write_new(directory / "PROMPT.txt", prompt)
    write_new(directory / "REQUEST.json", request)
    write_new(directory / "PREPARED.json", {
        "root_id": payload["root_id"], "source_hashes": hashes,
        "input_sha256": digest(canonical(payload)), "prompt_sha256": digest(prompt.encode()),
        "request_sha256": digest(canonical(request)),
        "review_provider": "TJU AI" if MODEL == "tju-llm" else "SiliconFlow",
        "review_model": MODEL, "endpoint": ENDPOINT, "maximum_http_attempts": 1,
        "source_candidate_count": 4, "mechanically_valid_count": len(payload["candidate_questions"]),
        "semantic_judge_calls": 0, "solver_calls": 0, "training_started": False,
    })
    return {"root_id": payload["root_id"], "mechanically_valid": len(payload["candidate_questions"])}


def config_key() -> str:
    fields = {}
    for line in CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip().strip("\"'")
    for key, value in (("provider", "TJU AI"), ("model", MODEL),
                       ("base_url", "https://ai.tju.edu.cn/api/v3"),
                       ("api_key_env", "ROUND1_STRONG_JUDGE_API_KEY")):
        if fields.get(key) != value:
            raise ValueError("judge_config_identity_invalid")
    credential = fields.get("ROUND1_STRONG_JUDGE_API_KEY")
    if not credential or any(char.isspace() for char in credential):
        raise ValueError("judge_credential_missing")
    return credential


def execute(suffix: str) -> dict:
    directory = HERE / f"cc-train-episode-{suffix}"
    prepared = read(directory / "PREPARED.json")
    request = read(directory / "REQUEST.json")
    if (prepared["source_hashes"] != input_for(suffix)[1]
            or prepared["request_sha256"] != digest(canonical(request))
            or prepared["prompt_sha256"] != digest(request["messages"][0]["content"].encode())):
        raise ValueError("prepared_binding_changed")
    if not request["messages"][0]["content"] == (directory / "PROMPT.txt").read_text():
        raise ValueError("prompt_changed")
    if (directory / "RESPONSE.json").exists() or (directory / "ATTEMPT.json").exists():
        raise FileExistsError("attempt_already_reserved")
    if prepared["mechanically_valid_count"] == 0:
        return {"root_id": prepared["root_id"], "status": "NO_MECHANICAL_CANDIDATES"}
    key = config_key()
    write_new(directory / "ATTEMPT.json", {"request_sha256": prepared["request_sha256"],
                                           "model": MODEL, "attempt_number": 1})
    http_request = urllib.request.Request(ENDPOINT, data=canonical(request), method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(http_request, timeout=300) as result:
            status, raw = result.status, result.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except Exception as error:
        write_new(directory / "ERROR.json", {"error_type": type(error).__name__,
                                               "status": "TRANSPORT_ERROR_NO_RETRY"})
        return {"root_id": prepared["root_id"], "status": "TRANSPORT_ERROR_NO_RETRY"}
    write_new(directory / "RESPONSE.json", {"http_status": status, "raw_body": raw.decode("utf-8", "replace")})
    return {"root_id": prepared["root_id"], "http_status": status}


def review(suffix: str, *, output_name: str = "REVIEW_RESULT.json") -> dict:
    directory = HERE / f"cc-train-episode-{suffix}"
    prepared = read(directory / "PREPARED.json")
    data = read(directory / "INPUT.json")
    original = {x["candidate_id"]: x for x in data["candidate_questions"]}
    response_path = directory / "RESPONSE.json"
    if not response_path.exists():
        result = {"root_id": prepared["root_id"], "status": "NO_RESPONSE",
                  "approved_ids": [], "rejected_ids": [], "unresolved_ids": list(original)}
    else:
        response = read(response_path)
        result = {"root_id": prepared["root_id"], "status": "REVIEWED_FAIL_CLOSED",
                  "approved_ids": [], "rejected_ids": [], "unresolved_ids": [],
                  "candidate_decisions": {},
                  "response_sha256": digest(response_path.read_bytes())}
        try:
            if response["http_status"] != 200:
                raise ValueError("http_non_200")
            envelope = json.loads(response["raw_body"])
            choice = envelope["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("incomplete_finish")
            content = choice["message"]["content"].strip()
            if content.startswith("```json\n") and content.endswith("\n```"):
                content = content[len("```json\n"):-len("\n```")]
            parsed = json.loads(content)
            rows = parsed["reviews"]
            if set(parsed) != {"reviews"} or not isinstance(rows, list) or len(rows) != len(original):
                raise ValueError("review_schema_invalid")
            if {x["candidate_id"] for x in rows} != set(original):
                raise ValueError("candidate_set_invalid")
            for item in rows:
                cid = item["candidate_id"]
                fields = {"candidate_id", "stem_scope_ok", "novelty_ok", "batch_distinct_ok",
                          "gold_consistent", "overlap_candidate_ids", "reason", "options"}
                if set(item) != fields or any(type(item[k]) is not bool for k in
                    ("stem_scope_ok", "novelty_ok", "batch_distinct_ok", "gold_consistent")):
                    raise ValueError("candidate_review_fields_invalid")
                if (not isinstance(item["reason"], str) or not item["reason"].strip()
                        or not isinstance(item["overlap_candidate_ids"], list)
                        or any(not isinstance(x, str) for x in item["overlap_candidate_ids"])):
                    raise ValueError("candidate_review_reason_invalid")
                options = item["options"]
                if not isinstance(options, list) or len(options) != 6:
                    raise ValueError("option_review_count_invalid")
                for index, option in enumerate(options):
                    if (set(option) != {"index", "verdict", "quote_sufficient", "reason"}
                            or type(option["index"]) is not int or option["index"] != index + 1
                            or option["verdict"] not in ("supports", "contradicts", "unsupported")
                            or type(option["quote_sufficient"]) is not bool
                            or not isinstance(option["reason"], str) or not option["reason"].strip()):
                        raise ValueError("option_review_fields_invalid")
                relation_ok = all(option["quote_sufficient"] and
                                  option["verdict"] == expected["relation"]
                                  for option, expected in zip(options, original[cid]["options"]))
                contradiction_cue_ok = all(
                    expected["relation"] != "contradicts" or
                    CONTRADICTION_CUE.search(expected["quote"]) is not None
                    for expected in original[cid]["options"])
                self_contained_quote_ok = all(
                    REFERENT_CUE.search(expected["quote"]) is None
                    for expected in original[cid]["options"])
                approved = (all(item[k] for k in ("stem_scope_ok", "novelty_ok",
                             "batch_distinct_ok", "gold_consistent")) and relation_ok
                            and contradiction_cue_ok and self_contained_quote_ok)
                result["candidate_decisions"][cid] = {"approved": approved,
                                                        "relation_ok": relation_ok,
                                                        "contradiction_cue_ok": contradiction_cue_ok,
                                                        "self_contained_quote_ok": self_contained_quote_ok,
                                                        "judge": item}
                (result["approved_ids"] if approved else result["rejected_ids"]).append(cid)
            result["usage"] = envelope.get("usage")
            result["returned_model"] = envelope.get("model")
            if result["returned_model"] != MODEL:
                raise ValueError("returned_model_mismatch")
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result["status"] = "INVALID_JUDGE_RESPONSE"
            result["invalid_reason"] = str(error)
            result["approved_ids"] = []
            result["rejected_ids"] = []
            result["unresolved_ids"] = list(original)
            result.pop("candidate_decisions", None)
    write_new(directory / output_name, result)
    return {"root_id": prepared["root_id"], "status": result["status"],
            "approved": len(result["approved_ids"]), "rejected": len(result["rejected_ids"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "execute", "review", "reparse", "policy_recheck", "policy_v3"))
    parser.add_argument("root", choices=ROOTS)
    args = parser.parse_args()
    if args.action == "reparse":
        value = review(args.root, output_name="REVIEW_RESULT_REPARSED.json")
    elif args.action == "policy_recheck":
        value = review(args.root, output_name="REVIEW_RESULT_POLICY_V2.json")
    elif args.action == "policy_v3":
        value = review(args.root, output_name="REVIEW_RESULT_POLICY_V3.json")
    else:
        value = globals()[args.action](args.root)
    print(json.dumps(value, ensure_ascii=False))
