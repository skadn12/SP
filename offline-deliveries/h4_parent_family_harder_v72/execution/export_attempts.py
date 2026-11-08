"""Export every raw rewrite attempt and its validation, review and adoption trail."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FACT_IDS = [f"F{i:02d}" for i in range(1, 7)]


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite_json_number: " + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def export(root):
    root = Path(root).resolve()
    bindings = {}

    def read(path, default=None):
        if not path.exists():
            return default
        raw = path.read_bytes()
        bindings[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    families = []
    for manifest_path in sorted((root / "assets").glob("*/MANIFEST.json")):
        family = manifest_path.parent
        manifest = read(manifest_path)
        provenance = read(family / "ROUND_PROVENANCE.json", {})
        adopted = {r["fact_id"]: r for r in provenance.get("accepted_versions", [])}
        rounds = []
        for folder in sorted((family / "rounds").glob("*"), key=lambda p: (p.name != "initial", p.name)):
            if not folder.is_dir():
                continue
            rm = read(folder / "MANIFEST.json", {})
            targets = rm.get("target_fact_ids", FACT_IDS if folder.name == "initial" else [])
            public = read(folder / "PUBLIC_ASSETS.json", {})
            judgments = read(folder / "SOURCE_JUDGMENTS.json", {})
            review = read(folder / "ASSET_REVIEW.json", {})
            completion = read(folder / "GENERATION.COMPLETE.json")
            receipt = read(folder / "REVIEW_RECEIPT.json")
            versions = {v["fact_id"]: v for v in public.get("versions", [])}
            parents = {p["fact_id"]: p for p in public.get("parent_facts", [])}
            jparents = {p["fact_id"]: p for p in judgments.get("parents", [])}
            rparents = {p["fact_id"]: p for p in review.get("parents", [])}
            jversions = {v["version_id"]: v for v in judgments.get("versions", [])}
            rversions = {v["version_id"]: v for v in review.get("versions", [])}
            requests = []
            for directory in sorted((folder / "requests").glob("*")):
                if not directory.is_dir():
                    continue
                response = read(directory / "RESPONSE.json")
                validation = read(directory / "VALIDATION.json")
                error = read(directory / "ERROR.json")
                parsed, parse_error = None, None
                if response is not None:
                    try:
                        parsed = strict_json(response.get("text"))
                    except (ValueError, TypeError, AttributeError) as exc:
                        parse_error = {"type": type(exc).__name__, "message": str(exc)}
                raw_rows = parsed.get("replacements", []) if isinstance(parsed, dict) else []
                raw_rows = raw_rows if isinstance(raw_rows, list) else []
                errors = (validation or {}).get("mechanical_errors", [])
                fact_rows = []
                for fid in targets:
                    version = versions.get(fid)
                    vid = version["version_id"] if version else None
                    source_judgment = jversions.get(vid)
                    private_review = rversions.get(vid)
                    matches = [(i, row) for i, row in enumerate(raw_rows)
                               if isinstance(row, dict) and row.get("fact_id") == fid]
                    origin = adopted.get(fid)
                    selected = bool(origin and origin.get("round_id") == folder.name
                                    and origin.get("version_id") == vid)
                    for index, row in matches or [(None, None)]:
                        contradicted = row.get("contradicted") if isinstance(row, dict) else None
                        fact_rows.append({"fact_id": fid, "raw_row_index": index,
                            "raw_row": row,
                            "raw_proposition": contradicted.get("proposition") if isinstance(contradicted, dict) else None,
                            "raw_rationale": contradicted.get("rationale") if isinstance(contradicted, dict) else None,
                            "contradiction_unavailable_reason": row.get("contradiction_unavailable_reason") if isinstance(row, dict) else None,
                            "raw_row_missing": not matches,
                            "parent_public": parents.get(fid),
                            "parent_source_judgment": jparents.get(fid),
                            "parent_private_review": rparents.get(fid),
                            "validation_errors": [e for e in errors if e.get("fact_id") in (None, fid)],
                            "mechanically_exported_version": version,
                            "source_judgment": source_judgment,
                            "private_review": private_review,
                            "usable": private_review.get("usable") if private_review else None,
                            "adopted_from_this_round": selected,
                            "final_adoption_origin": origin})
                requests.append({"request_directory": str(directory.resolve()),
                    "request_binding": read(directory / "PROOF.json"),
                    "http_attempt": read(directory / "HTTP_ATTEMPT.json"),
                    "http_result": read(directory / "HTTP_RESULT.json"),
                    "response_present": response is not None,
                    "response": response, "parsed_response": parsed,
                    "response_json_parse_error": parse_error,
                    "transport_error": error, "validation": validation,
                    "all_raw_replacement_rows": raw_rows,
                    "unexpected_raw_rows": [{"raw_row_index": i, "raw_row": row}
                        for i, row in enumerate(raw_rows)
                        if not isinstance(row, dict) or row.get("fact_id") not in targets],
                    "facts": fact_rows})
            rounds.append({"round_id": folder.name, "target_fact_ids": targets,
                "completion": completion, "review_receipt": receipt,
                "requests": requests, "requests_absent": not requests})
        families.append({"root_id": manifest.get("root_id"),
            "family_id": manifest.get("family_id"),
            "source_sha256": manifest.get("source_sha256"),
            "final_adopted_versions": list(adopted.values()), "rounds": rounds})
    if not families:
        raise ValueError("no_family_manifests")
    script = Path(__file__).resolve()
    bindings[str(script)] = hashlib.sha256(script.read_bytes()).hexdigest()
    return {"version": "v72_all_rewrite_attempts_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "report_is_new_independent_review": False, "solver_scores_used": False,
        "missing_judgment_or_usability_is_null_not_false": True,
        "families": families,
        "files": [{"path": p, "sha256": digest} for p, digest in sorted(bindings.items())]}


def markdown(result):
    def block(value):
        text = json.dumps(value, ensure_ascii=False, indent=2)
        fence = "```"
        while fence in text:
            fence += "`"
        return fence + "json\n" + text + "\n" + fence + "\n"

    lines = ["# All rewrite attempts\n",
        "Original responses and rows are retained, including rejected and unparseable responses. "
        "Null judgment/usability means no saved judgment is available. This export is not a new review.\n"]
    for family in result["families"]:
        lines.append("## " + str(family["family_id"]) + "\n")
        lines.append(block({"final_adopted_versions": family["final_adopted_versions"]}))
        for round_ in family["rounds"]:
            lines.append("### " + round_["round_id"] + "\n")
            lines.append(block({k: v for k, v in round_.items() if k != "requests"}))
            for request in round_["requests"]:
                lines.append("#### " + Path(request["request_directory"]).name + "\n")
                lines.append(block({k: v for k, v in request.items() if k != "facts"}))
                for fact in request["facts"]:
                    lines.append("##### " + fact["fact_id"] + "\n")
                    lines.append(block(fact))
    lines.append("## Input bindings\n")
    lines.append(block(result["files"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out = (args.output_dir or args.root).resolve()
    paths = [out / "ALL_REWRITE_ATTEMPTS.json", out / "ALL_REWRITE_ATTEMPTS.md"]
    if any(path.exists() for path in paths):
        raise FileExistsError("rewrite_attempt_export_already_exists")
    result = export(args.root)
    out.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, [json.dumps(result, ensure_ascii=False, indent=2) + "\n", markdown(result)]):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(json.dumps({"status": "EXPORTED", "families": len(result["families"]),
                      "outputs": [str(path) for path in paths]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
