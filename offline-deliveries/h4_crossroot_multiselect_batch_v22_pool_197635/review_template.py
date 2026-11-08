"""Create a conservative, response-bound review form after pool validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pool_pipeline_for_review", HERE / "pipeline.py")
assert spec and spec.loader
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def main() -> None:
    _, _, prompt_input, _, _, _ = pool.preflight()
    response, draft, checks = [pool.read(pool.RUN / name) for name in
                               ("POOL_RESPONSE.json", "POOL_DRAFT.json", "POOL_CHECKS.json")]
    if checks["fatal_errors"] or checks["draft_sha256"] != pool.sha(pool.encoded(draft)):
        raise ValueError("validate_pool_first")
    claims = prompt_input["reviewed_reusable_claims_candidates_only"] + draft["claims"]
    review = {"reviewer": "", "draft_sha256": checks["draft_sha256"],
              "response_sha256": pool.sha(pool.encoded(response)),
              "claim_decisions": [{"claim_id": item["claim_id"], "approved": False,
                                   "reason": "REVIEW_PENDING", "semantic_key": "",
                                   "selected_fact_duplicate": False} for item in claims]}
    pool.write_new(pool.RUN / "CONTENT_REVIEW.json", review)
    print(json.dumps({"status": "REVIEW_TEMPLATE_PENDING_MANUAL_AUDIT",
                      "claim_count": len(claims)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
