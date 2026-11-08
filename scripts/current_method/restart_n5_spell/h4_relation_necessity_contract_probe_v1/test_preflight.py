from __future__ import annotations

from .preflight import ANSWER_COMPONENTS, ROLE_REFS


def test_minimal_role_windows_are_disjoint():
    refs = [set(ROLE_REFS[name]) for name in ("fact_A", "trigger", "fact_B")]
    assert not refs[0] & refs[1]
    assert not refs[0] & refs[2]
    assert not refs[1] & refs[2]


def test_each_role_has_explicit_answer_component():
    assert all(ANSWER_COMPONENTS.values())
