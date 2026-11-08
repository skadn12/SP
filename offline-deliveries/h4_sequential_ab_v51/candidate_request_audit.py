"""Replay candidate request inputs and account for every returned dialogue.

This module checks artifact relationships only. It never interprets reviewer
prose or emits semantic acceptance. Imports are delayed to avoid runner cycles.
"""
from __future__ import annotations

import json
from pathlib import Path


ROLES = {"planner", "realizer", "reviewer"}


def _context(payload):
    try:
        prompt = payload["messages"][0]["content"]
        encoded = prompt.split("\nCANDIDATE_INPUT_JSON\n", 1)[1]
        value, _end = json.JSONDecoder().raw_decode(encoded)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("candidate_request_context_unreadable") from error
    if not isinstance(value, dict):
        raise ValueError("candidate_request_context_not_object")
    return value


def audit_requests(run, records, baseline):
    import candidate_runner as c

    run = Path(run).resolve()
    read, digest, require = c.r.base.read, c.r.base.digest, c.r.base.require
    manifest_root = read(run / "MANIFEST.json")["root_id"]
    request_root = (run / "requests").resolve()
    record_by_request, record_by_review = {}, {}
    for row in records:
        path = Path(row["request"]).resolve()
        require(path.parent == request_root, "candidate_request_outside_current_run")
        require(path not in record_by_request, "multiple_records_for_one_realizer_response")
        record_by_request[path] = row
        if row.get("review_request"):
            review = Path(row["review_request"]).resolve()
            require(review.parent == request_root, "candidate_review_outside_current_run")
            require(review not in record_by_review, "review_response_reused_for_multiple_candidates")
            record_by_review[review] = row

    successes = {"planner": set(), "realizer": set(), "reviewer": set()}
    calls, actuals = [], {}
    for directory in sorted(request_root.iterdir() if request_root.exists() else []):
        if not directory.is_dir():
            continue
        request_path, proof_path = directory / "REQUEST.json", directory / "PROOF.json"
        require(request_path.exists(), "request_directory_missing_payload")
        if not proof_path.exists():
            local_path = directory / "LOCAL_ERROR.json"
            require(local_path.exists() and read(local_path).get("http_attempted") is False
                    and not list(directory.glob("HTTP_ATTEMPT*.json"))
                    and not (directory / "RESPONSE.json").exists(), "request_missing_proof")
            continue
        proof, actual = read(proof_path), read(request_path)
        if proof.get("role") not in ROLES:
            continue
        role = proof["role"]
        require(proof.get("root_id") == manifest_root and proof.get("route") in c.r.ROUTES,
                "request_identity_invalid")
        scene_id = proof.get("scene_id")
        require(isinstance(scene_id, str) and scene_id.startswith("S") and scene_id[1:].isdigit(),
                "request_scene_id_invalid")
        context = _context(actual)
        require(all(context.get(key) == proof.get(key) for key in c.IDENTITY_KEYS),
                "request_context_identity_differs_from_proof")
        response_path = directory / "RESPONSE.json"
        if response_path.exists():
            response = read(response_path)
            http_paths = sorted(directory.glob("HTTP_RESPONSE_*.json"))
            require(http_paths, "successful_response_missing_http_body")
            http = read(http_paths[-1])
            require(response["text"] == http["text"], "response_differs_from_raw_http_body")
            successes[role].add(directory.resolve())
            if role == "realizer":
                row = record_by_request.get(directory.resolve())
                require(row is not None, "successful_realizer_response_missing_record")
                try:
                    obj = c.r.parse_object(response["text"])
                except (TypeError, ValueError):
                    obj = None
                require(obj == row["realization"], "record_differs_from_raw_realizer_response")
            if role == "reviewer":
                row = record_by_review.get(directory.resolve())
                require(row is not None, "successful_reviewer_response_missing_record")
                require(response["text"] == row["raw_review"], "record_differs_from_raw_review_response")
        reservation = read(directory / "RESERVATION.json")
        calls.append((reservation["request_number"], directory, proof, actual, context))
        actuals[directory.resolve()] = context

    require(successes["realizer"] == set(record_by_request), "candidate_records_do_not_cover_realizer_responses")
    require(successes["reviewer"] == set(record_by_review), "candidate_records_do_not_cover_reviewer_responses")
    require(len({entry[0] for entry in calls}) == len(calls), "duplicate_candidate_request_number")
    latest_realizer, plan_checked = {}, set()
    roles = {role: 0 for role in ROLES}
    for _number, directory, proof, actual, context in sorted(calls, key=lambda row: row[0]):
        root, route, scene_id, role = proof["root_id"], proof["route"], proof["scene_id"], proof["role"]
        slot_id = (root, route, scene_id)
        folder = run / "candidates" / root / ("A" if route == c.r.ROUTES[0] else "B") / scene_id
        slot = read(folder / "INPUT.json")
        require(all(slot.get(key) == proof[key] for key in c.IDENTITY_KEYS), "slot_identity_changed")
        history, card = slot["history"], slot["card"]
        require(history[:len(baseline)] == baseline and slot["baseline_sha256"] == digest(baseline),
                "request_formal_baseline_changed")
        advisory = read(folder / "PLAN_ADVISORY.json")
        require(advisory["blocks_generation"] is False, "advisory_reintroduced_generation_gate")
        plan = None if role == "planner" else advisory["plan"]
        realization, repair = None, None
        if role == "planner":
            require(slot_id not in plan_checked, "multiple_planner_calls_for_candidate_slot")
            plan_checked.add(slot_id)
            parsed_plan = None
            if (directory / "RESPONSE.json").exists():
                try:
                    parsed_plan = c.validate_plan(read(directory / "RESPONSE.json")["text"],
                                                 c.identity(root, route, scene_id), history, card)
                except (ValueError, TypeError, KeyError):
                    pass
            require(parsed_plan == advisory["plan"], "advisory_plan_not_derived_from_raw_planner_response")
        elif role == "realizer":
            previous = latest_realizer.get(slot_id)
            if previous is not None:
                require(previous["disposition"] == "hard_reject", "realizer_retry_after_retained_or_quarantined_candidate")
                require(not (isinstance(previous["realization"], dict) and
                             previous["realization"].get("status") == "no_scene"), "realizer_retry_after_no_scene")
                repair = {"instruction": "Fix only the mechanical/duplicate issues, preserving source and history.",
                          "failures": previous["mechanical_failures"],
                          "duplicate_reasons": (previous["duplicate_result"] or {}).get("reasons", []),
                          "previous_response": read(Path(previous["request"]) / "RESPONSE.json")["text"]}
            row = record_by_request.get(directory.resolve())
            if row is not None:
                latest_realizer[slot_id] = row
        else:
            row = latest_realizer.get(slot_id)
            if row is None:
                # A successful reviewer is paired by its immutable record; a
                # failed reviewer still follows a returned realizer response.
                row = record_by_review.get(directory.resolve())
            require(row is not None, "reviewer_missing_preceding_realizer")
            realization = row["realization"]
            require(row.get("review_request") in (None, str(directory.resolve())),
                    "reviewer_paired_with_wrong_realizer")
        expected_request, expected_proof = c.request(run, root, route, scene_id, role,
            history, baseline, card, plan, realization, repair)
        require(context == _context(expected_request), "candidate_request_full_context_changed")
        require(actual == expected_request, "candidate_request_payload_or_feedback_changed")
        require(proof.get("payload_sha256") == digest(actual), "candidate_request_payload_hash_changed")
        require(all(proof.get(key) == value for key, value in expected_proof.items()),
                "candidate_request_proof_changed")
        roles[role] += 1
    return {"status": "REQUEST_ARTIFACTS_REPLAYED", "roles": roles,
            "successful_realizer_responses": len(successes["realizer"]),
            "successful_reviewer_responses": len(successes["reviewer"]),
            "semantic_certified": False, "training_pool_eligible": False}
