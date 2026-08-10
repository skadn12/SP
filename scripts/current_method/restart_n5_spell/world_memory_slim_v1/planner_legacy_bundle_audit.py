"""Read-only classification of pre-v3.3 Planner bundle-shaped fixtures."""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_BUNDLE_AUDIT_VERSION = "planner_legacy_bundle_audit_v1"


def classify_legacy_bundle_plan(
    plan: Mapping[str, Any], *, allowed_bundles: Mapping[str, Any],
    planner_action_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify old proposal arrays without migrating or accepting them."""
    bundle_by_id = {
        str(row.get("bundle_id")): row
        for row in allowed_bundles.get("bundles") or []
    }
    projected = {
        str(row.get("action_id")): row
        for row in planner_action_view.get("actions") or []
    }
    errors: list[str] = []
    bundle_scenes: dict[str, list[int]] = {}
    bundle_action_scenes: dict[str, dict[str, list[int]]] = {}

    for scene_index, scene in enumerate(plan.get("scene_goals") or [], 1):
        proposals = scene.get("extension_proposals")
        bundle_id_value = scene.get("proposal_bundle_id")
        bundle_id = str(bundle_id_value) if bundle_id_value else None
        if not isinstance(proposals, list):
            errors.append(f"invalid_legacy_proposal_array:scene={scene_index}")
            continue
        action_ids = [str(row.get("action_id") or "")
                      for row in proposals if isinstance(row, Mapping)]

        if len(action_ids) > 1:
            bundle = bundle_by_id.get(bundle_id or "")
            required = ([str(value) for value in bundle.get("required_action_ids") or []]
                        if bundle else [])
            if bundle is None or set(action_ids) != set(required):
                errors.append(
                    f"invalid_unregistered_action_pair:scene={scene_index}:actions=" +
                    ",".join(action_ids))

        if bundle_id:
            bundle_scenes.setdefault(bundle_id, []).append(scene_index)
            bundle = bundle_by_id.get(bundle_id)
            required = ([str(value) for value in bundle.get("required_action_ids") or []]
                        if bundle else [])
            if len(action_ids) == 1:
                errors.append(
                    f"single_action_illegally_carries_bundle:scene={scene_index}:"
                    f"action={action_ids[0]}:bundle={bundle_id}")
            if bundle is None:
                errors.append(
                    f"unknown_bundle_id:scene={scene_index}:bundle={bundle_id}")
            else:
                missing = [value for value in required if value not in action_ids]
                extra = [value for value in action_ids if value not in required]
                if missing:
                    errors.append(
                        f"incomplete_bundle_actions:scene={scene_index}:bundle={bundle_id}:"
                        "missing=" + ",".join(missing))
                if extra:
                    errors.append(
                        f"extra_bundle_actions:scene={scene_index}:bundle={bundle_id}:"
                        "extra=" + ",".join(extra))
                action_scenes = bundle_action_scenes.setdefault(bundle_id, {})
                for action_id in action_ids:
                    if action_id in required:
                        action_scenes.setdefault(action_id, []).append(scene_index)

        for action_id in action_ids:
            action = projected.get(action_id) or {}
            required_bundle = str(action.get("required_bundle_id") or "")
            if action.get("selection_policy") == "bundle_only":
                bundle = bundle_by_id.get(required_bundle) or {}
                required = [str(value) for value in bundle.get("required_action_ids") or []]
                complete_here = (bundle_id == required_bundle and
                                 set(action_ids) == set(required))
                if not complete_here:
                    errors.append(
                        f"bundle_only_action_selected_singly:scene={scene_index}:"
                        f"action={action_id}:required_bundle={required_bundle}")
                    errors.append(
                        f"required_bundle_not_selected:scene={scene_index}:"
                        f"action={action_id}:bundle={required_bundle}")

    for bundle_id, scenes in bundle_scenes.items():
        if len(scenes) > 1:
            errors.append(
                f"duplicate_bundle_id:bundle={bundle_id}:scenes=" +
                ",".join(str(value) for value in scenes))
        bundle = bundle_by_id.get(bundle_id) or {}
        required = [str(value) for value in bundle.get("required_action_ids") or []]
        action_scenes = bundle_action_scenes.get(bundle_id) or {}
        if (required and all(action_scenes.get(action_id) for action_id in required) and
                len({scene for action_id in required
                     for scene in action_scenes[action_id]}) > 1):
            errors.append(
                f"split_bundle_across_scenes:bundle={bundle_id}:actions=" +
                ",".join(
                    f"{action_id}@{','.join(str(value) for value in action_scenes[action_id])}"
                    for action_id in required))

    return {
        "valid_live_v3_3": False,
        "classification": "planner_repairable",
        "errors": list(dict.fromkeys(errors)),
        "audit_version": LEGACY_BUNDLE_AUDIT_VERSION,
        "migration_performed": False,
    }
