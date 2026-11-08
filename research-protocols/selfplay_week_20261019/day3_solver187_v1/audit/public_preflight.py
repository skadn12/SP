"""CPU-only public execution preflight. Never opens private QA or starts a backend."""
from __future__ import annotations
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "runtime"))
import common as c


def independently_rebind(question, seed, n):
    """Separate implementation of the frozen Gold-blind v15 rebind algorithm."""
    field = "options" if question["question_type"] == "multi_select" else "items"
    labels = list(question[field])
    canonical = {label: "content_" + hashlib.sha256(text.strip().encode()).hexdigest()[:16]
                 for label, text in question[field].items()}
    content = list(canonical.values())
    texts = {canonical[label]: question[field][label] for label in labels}
    def derived_seed(index):
        material = f"{seed}|{question['candidate_id']}|{index}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.Random(derived_seed(-1)).shuffle(content)
    seen, variants = set(), []
    for index in range(n):
        generator = random.Random(derived_seed(index))
        while True:
            order = list(content)
            generator.shuffle(order)
            if tuple(order) not in seen:
                break
        seen.add(tuple(order))
        variants.append({"variant_id": f"perm_{index:02d}",
                         field: dict(zip(labels, (texts[value] for value in order))),
                         "display_to_content": dict(zip(labels, order)),
                         "canonical_order_content_ids": list(content),
                         "display_order_content_ids": list(order),
                         "content_to_text": dict(texts)})
    return canonical, variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE.parent / "runtime/direct")
    parser.add_argument("--output", type=Path, default=HERE / "PUBLIC_PREFLIGHT_PASS.json")
    args = parser.parse_args()
    run = args.run.resolve()
    started = time.perf_counter()
    c.environment()
    if args.output.exists():
        raise RuntimeError("NO_OVERWRITE_AUDIT_REPORT")
    if not (run / "PREFLIGHT_FREEZE.json").exists():
        raise RuntimeError("PREFLIGHT_FREEZE_NOT_READY")
    if (run / "STARTED.json").exists():
        raise RuntimeError("AUDIT_MUST_PRECEDE_GPU_START")
    checks = collections.Counter()
    def check(condition, name):
        if not condition:
            raise RuntimeError(name)
        checks[name] += 1

    # This public-only helper reads code, model/tokenizer and public execution
    # bindings. It does not resolve private-sidecar metadata or QA.
    manifest = c.verify(run, check_models=True)
    freeze = c.read(run / "PREFLIGHT_FREEZE.json")
    public = c.read(run / "PUBLIC_QUESTIONS.json")
    maps = c.read(run / "DISPLAY_MAPPINGS.json")
    token_ids = c.read(run / "PROMPT_TOKEN_IDS.json")
    protocol = c.read(run / "PROTOCOL.json")
    admission = c.read(run / "CONTENT_ADMISSION.json")
    check(set(public) == {"questions"}, "PUBLIC_WRAPPER_WHITELIST")
    questions = public["questions"]
    qids = {q["candidate_id"] for q in questions}
    counts = dict(collections.Counter(q["question_type"] for q in questions))
    check(len(questions) == len(qids) == 187, "EXACT_187_UNIQUE_QUESTIONS")
    check(counts == {"multi_select": 74, "ordering": 113}, "EXACT_TYPE_COUNTS")
    check(len({q["root_id"] for q in questions}) == manifest["root_count"] == 20,
          "EXACT_20_TRAIN_ROOTS")
    check(all(q["root_id"].startswith("cc-train-episode-") for q in questions),
          "TRAIN_ROOT_NAMESPACE")
    check(freeze["questions"] == 187 and freeze["requests"] == 2688 and
          freeze["inference_started"] is False and freeze["public_only"] is True,
          "PREFLIGHT_COUNTS_AND_STATE")
    check(manifest["use_as_final_eval"] is False and manifest["training_started"] is False,
          "DIAGNOSTIC_ONLY_NOT_TRAINING")
    check(manifest["production_denominator"] == 240 and
          manifest["production_admitted_numerator"] == 187 and
          manifest["production_unadmitted_slots"] == 53, "PRODUCTION_DENOMINATOR_RETAINED")
    check(manifest["batch_size"] == 8 and manifest["max_model_len"] == 16384 and
          manifest["prefix_caching"] is False and manifest["automatic_retry"] is False and
          manifest["source_truncation"] is False, "NATIVE_EXECUTION_LIMITS")
    expected_solver = {"name": "S0_M_clean", "model": str(c.MODEL), "tokenizer": str(c.TOKENIZER),
                       "MC_mappings": 6, "MC_samples_each_mapping": 3,
                       "ordering_mappings": 12, "ordering_samples_each_mapping": 1,
                       "temperature": .8, "top_p": .9, "max_tokens": 32,
                       "enable_thinking": False, "max_model_len": 16384,
                       "source_truncation": False, "automatic_retry": False,
                       "display_seed": 2026101903, "sampling_seed": 2026101904}
    check(all(protocol["solver"][key] == value for key, value in expected_solver.items()),
          "FROZEN_SOLVER_CONTRACT")
    check({slot["candidate_id"] for slot in protocol["slots"]} == qids,
          "PUBLIC_PROTOCOL_EXACT_QUESTION_SET")
    check(admission["status"] == "PASS" and set(admission["candidate_ids"]) == qids and
          admission["public_questions_sha256"] == c.sha(run / "PUBLIC_QUESTIONS.json"),
          "CONTENT_ADMISSION_BINDS_PUBLIC_SET")
    rows = manifest["requests"]
    rids = {row["request_id"] for row in rows}
    check(len(rows) == len(rids) == 2688, "EXACT_2688_UNIQUE_REQUESTS")
    check(set(token_ids) == set(maps["requests"]) == rids, "EXACT_TOKEN_AND_MAP_REQUEST_SET")
    check(set(maps["public_to_canonical"]) == qids, "EXACT_CANONICAL_MAP_QUESTION_SET")
    check({path.stem for path in (run / "prompts").glob("*.txt")} == rids,
          "EXACT_PROMPT_FILE_SET")
    check([row["seed"] for row in rows] == list(range(2026101904, 2026101904 + 2688)),
          "PROSPECTIVE_UNIQUE_SAMPLING_SEEDS")
    by_question = collections.defaultdict(list)
    for row in rows:
        by_question[row["candidate_id"]].append(row)
    check(set(by_question) == qids, "EXACT_REQUEST_QUESTION_SET")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(c.TOKENIZER), local_files_only=True)
    _, solver = c.a.native()
    native_decoder = solver.generation_config("multi_select")
    check(manifest["decoder_contract"] == native_decoder, "UNCHANGED_NATIVE_DECODER")
    check(all(native_decoder[key] == value for key, value in
              {"temperature": .8, "top_p": .9, "max_tokens": 32,
               "enable_thinking": False,
               "structured_output_version": "finite_choice_set_v1"}.items()),
          "DIRECT_FINITE_CHOICE_NO_THINKING")

    source_cache, unique_variant_count = {}, 0
    prompt_min, prompt_max, roundtrip_sequences = 10**9, 0, 0
    param_types_checked = set()
    for index, question in enumerate(questions):
        c.a.public_question(question)
        check(set(question) <= c.a.PUBLIC_KEYS, "NO_GOLD_OR_DESIGN_IN_PUBLIC_QUESTION")
        typ, qid, root = question["question_type"], question["candidate_id"], question["root_id"]
        field = "options" if typ == "multi_select" else "items"
        labels = list(question[field])
        check(labels == (list("ABCDEF") if typ == "multi_select" else
                         [f"I{i}" for i in range(1, 7)]), "SIX_NATIVE_CANONICAL_LABELS")
        source_path = run / "sources" / (root + ".json")
        check(c.below(source_path, run), "PUBLIC_STAGED_SOURCE_LOCATION")
        if root not in source_cache:
            source = c.read(source_path)
            check(source["root_id"] == root and isinstance(source["turns"], list) and
                  bool(source["turns"]), "SOURCE_IDENTITY_AND_NONEMPTY_TURNS")
            check(len({turn["turn_ref"] for turn in source["turns"]}) == len(source["turns"]),
                  "UNIQUE_SOURCE_TURN_REFS")
            source_cache[root] = (source, c.sha(source_path))
        source, source_sha = source_cache[root]
        check(question["source_sha256"] == source_sha, "STAGED_SOURCE_EXACT_DECLARED_HASH")
        canonical, variants = independently_rebind(question, 2026101903,
                                                   6 if typ == "multi_select" else 12)
        check(maps["public_to_canonical"][qid] == canonical and
              c.a.content_ids(question) == canonical, "INDEPENDENT_CONTENT_ID_RECOMPUTATION")
        check(c.a.variants(question, 2026101903, len(variants)) == variants,
              "INDEPENDENT_NATIVE_PERMUTATION_RECOMPUTATION")
        qrows = by_question[qid]
        check(len(qrows) == (18 if typ == "multi_select" else 12), "PER_QUESTION_CALL_BUDGET")
        expected_variants = {variant["variant_id"]: variant for variant in variants}
        groups = collections.defaultdict(list)
        for row in qrows:
            groups[row["variant_id"]].append(row)
        check(set(groups) == set(expected_variants), "EXACT_6_OR_12_UNIQUE_VARIANTS")
        unique_variant_count += len(variants)
        for variant_id, variant in expected_variants.items():
            group = groups[variant_id]
            check(sorted(row["slot"] for row in group) == ([0, 1, 2] if typ == "multi_select" else [0]),
                  "EXACT_3_OR_1_INDEPENDENT_SAMPLES")
            content_map = variant["display_to_content"]
            inverse = {value: label for label, value in content_map.items()}
            check(len(inverse) == 6 and set(content_map) == set(labels) and
                  set(inverse) == set(canonical.values()), "BIJECTIVE_DISPLAY_CONTENT_MAPPING")
            # Exhaustive public label sequence roundtrips need no Gold.
            sequences = (itertools.chain.from_iterable(itertools.combinations(labels, k)
                         for k in range(1, 7)) if typ == "multi_select" else itertools.permutations(labels))
            n_sequences = 0
            for sequence in sequences:
                check(tuple(inverse[content_map[label]] for label in sequence) == sequence,
                      "EXHAUSTIVE_LABEL_SEQUENCE_ROUNDTRIP")
                n_sequences += 1
            check(n_sequences == (63 if typ == "multi_select" else 720), "FINITE_CHOICE_CARDINALITY")
            roundtrip_sequences += n_sequences
            prompt = c.a.direct_prompt(tokenizer, question, variant, source)
            check(all(f"[{turn['turn_ref']}] {turn['speaker']}: {turn['text']}" in prompt
                      for turn in source["turns"]), "ALL_SOURCE_TURNS_PRESENT_VERBATIM")
            check(qid not in prompt, "GENERATOR_CANDIDATE_ID_ABSENT_FROM_PROMPT")
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            check(len(ids) + 32 <= 16384, "FULL_CONTEXT_FITS_WITHOUT_TRUNCATION")
            prompt_min, prompt_max = min(prompt_min, len(ids)), max(prompt_max, len(ids))
            for row in group:
                rid = row["request_id"]
                check(row["root_id"] == root and row["question_type"] == typ and
                      row["branch"] == "direct" and row["max_output_tokens"] == 32 and
                      row["display"] == {field: variant[field]}, "REQUEST_PUBLIC_IDENTITY_AND_DISPLAY")
                check(rid == f"{qid}__{variant_id}__direct__s{row['slot']:02d}",
                      "REQUEST_ID_STRUCTURE")
                check(maps["requests"][rid] == variant, "STAGED_MAP_EXACT_RECOMPUTATION")
                check((run / "prompts" / (rid + ".txt")).read_text() == prompt and
                      row["prompt_sha256"] == prompt_sha, "PROMPT_EXACT_NATIVE_REBUILD_AND_HASH")
                check(token_ids[rid] == ids and row["prompt_tokens"] == len(ids),
                      "TOKEN_IDS_EXACT_RECOMPUTATION")
                check(row["source_sha256"] == source_sha and row["source_turns"] == len(source["turns"]),
                      "REQUEST_FULL_SOURCE_BINDING")
            if typ not in param_types_checked:
                params = c.a.sampling_params(group[0], native_decoder["structured_output_api"])
                check(params.temperature == .8 and params.top_p == .9 and params.max_tokens == 32 and
                      params.seed == group[0]["seed"] and params.stop == ["</box>"] and
                      params.include_stop_str_in_output is True, "ACTUAL_NATIVE_SAMPLING_PARAMS")
                actual_choices = set(c.a.choices(typ, variant[field]))
                raw_sequences = (itertools.chain.from_iterable(itertools.combinations(labels, k)
                                 for k in range(1, 7)) if typ == "multi_select" else itertools.permutations(labels))
                expected_choices = {"<box>" + ",".join(sequence) + "</box>" for sequence in raw_sequences}
                check(actual_choices == expected_choices, "EXACT_NATIVE_FINITE_CHOICE_SET")
                param_types_checked.add(typ)
        if (index + 1) % 25 == 0:
            print(json.dumps({"status": "AUDITING", "questions": index + 1, "total": 187}), flush=True)
    check(unique_variant_count == 1800, "EXACT_1800_UNIQUE_QUESTION_VARIANTS")
    check(set(source_cache) == {path.stem for path in (run / "sources").glob("*.json")},
          "EXACT_STAGED_SOURCE_SET")
    check(manifest["max_prompt_tokens"] == prompt_max and
          manifest["minimum_context_headroom"] == 16384 - prompt_max - 32,
          "MANIFEST_TOKEN_RANGE_EXACT")
    # All requested audited_files bind only the public execution directory;
    # the report/script stay outside it, with separate metadata below.
    audited_files = [c.bind(run / "PREFLIGHT_FREEZE.json")] + freeze["files"]
    report = {"status": "PASS", "created_at_utc": c.now(),
              "role": "separate Codex public-preflight task; shared filesystem/model, not external independent audit",
              "public_only": True, "private_gold_opened": False, "backend_created": False,
              "new_solver_calls": 0, "model_hashes_verified": True,
              "questions": 187, "question_type_counts": counts, "roots": 20,
              "requests": 2688, "question_variants": unique_variant_count,
              "display_seed": 2026101903, "sampling_seed_start": 2026101904,
              "production_denominator": 240, "production_admitted": 187,
              "prompt_token_range": [prompt_min, prompt_max],
              "minimum_context_headroom": 16384 - prompt_max - 32,
              "exhaustive_public_label_sequence_roundtrips": roundtrip_sequences,
              "check_counts": dict(checks), "wall_seconds": time.perf_counter() - started,
              "audit_script_binding": c.bind(Path(__file__).resolve()),
              "audited_files": audited_files}
    c.save(args.output.resolve(), report)
    print(json.dumps({key: report[key] for key in ("status", "questions", "requests", "roots",
          "prompt_token_range", "minimum_context_headroom", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
