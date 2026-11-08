from __future__ import annotations

import hashlib
import json

from .freeze_scope import canonical, require_unique, schema_sha


def test_canonical_is_order_invariant() -> None:
    left = canonical({"b": 2, "a": [1, 3]})
    right = canonical({"a": [1, 3], "b": 2})
    assert left == right
    assert hashlib.sha256(left.encode()).hexdigest() == hashlib.sha256(right.encode()).hexdigest()


def test_schema_sha_uses_union_of_keys() -> None:
    assert schema_sha([{"a": 1}, {"b": 2}]) == schema_sha([{"b": 3, "a": 4}])


def test_require_unique_rejects_duplicate_ids() -> None:
    try:
        require_unique([{"candidate_id": "q1"}, {"candidate_id": "q1"}], "candidate_id", "test")
    except RuntimeError as exc:
        assert str(exc) == "duplicate_candidate_id:test:1"
    else:
        raise AssertionError("duplicate candidate ids must fail closed")

