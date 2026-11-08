from __future__ import annotations

from ..canary import semantic_audit


def _scene(index: int, texts: list[str]) -> dict:
    return {"scene_ref": f"R01:S{index:02d}", "dialogue_turns": [
        {"turn_ref": f"R01:S{index:02d}:T{position:03d}", "speaker": "Classmates A", "text": text}
        for position, text in enumerate(texts, 1)
    ]}


def test_ordered_chain_passes() -> None:
    child = {"scenes": [
        _scene(1, ["Use two parts honey and one part mustard.", "Add vinegar, brown sugar, garlic, and paprika."]),
        _scene(2, ["This is too thick.", "Add water so it is thinner and spreadable."]),
        _scene(3, ["Brush the glaze over the ham.", "Put it in the oven and set a timer before we baste it."]),
    ]}
    assert semantic_audit(child)["overall_pass"]


def test_premature_oven_fails() -> None:
    child = {"scenes": [
        _scene(1, ["Use two parts honey and one part mustard in the oven.", "Add vinegar, brown sugar, garlic, and paprika."]),
        _scene(2, ["This is too thick.", "Add water so it is thinner and spreadable."]),
        _scene(3, ["Brush the glaze over the ham.", "Put it in the oven and set a timer before we baste it."]),
    ]}
    assert not semantic_audit(child)["overall_pass"]


def test_water_before_trigger_fails() -> None:
    child = {"scenes": [
        _scene(1, ["Use two parts honey and one part mustard.", "Add vinegar, brown sugar, garlic, and paprika."]),
        _scene(2, ["Add water for a thinner consistency.", "It had been too thick."]),
        _scene(3, ["Brush the glaze over the ham.", "Put it in the oven and set a timer before we baste it."]),
    ]}
    assert not semantic_audit(child)["overall_pass"]


def test_explicit_finality_in_scene_one_fails() -> None:
    child = {"scenes": [
        _scene(1, ["Use two parts honey and one part mustard.", "Add vinegar, brown sugar, garlic, and paprika. This is the final version."]),
        _scene(2, ["This is too thick.", "Add water so it is thinner and spreadable."]),
        _scene(3, ["Brush the glaze over the ham.", "Put it in the oven and set a timer before we baste it."]),
    ]}
    assert not semantic_audit(child)["overall_pass"]


def test_application_in_revision_scene_fails() -> None:
    child = {"scenes": [
        _scene(1, ["Use two parts honey and one part mustard.", "Add vinegar, brown sugar, garlic, and paprika."]),
        _scene(2, ["This is too thick.", "Add water, brush it on, and check the thinner consistency."]),
        _scene(3, ["Brush the glaze over the ham.", "Put it in the oven and set a timer before we baste it."]),
    ]}
    assert not semantic_audit(child)["overall_pass"]
