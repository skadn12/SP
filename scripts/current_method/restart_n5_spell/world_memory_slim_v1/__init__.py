"""World Memory Slim package with a business-semantics-free initializer.

Version strings are duplicated intentionally: importing a v7 submodule must
not eagerly import either v2 or v7 orchestration.  Concrete modules remain the
authority for their contracts.
"""

WORLD_MEMORY_SCHEMA_VERSION = "world_memory_v1"
PLANNER_SCHEMA_VERSION = "slim_world_planner_v2"
REALIZER_SCHEMA_VERSION = "natural_realizer_v2"
EVIDENCE_COMMITTER_SCHEMA_VERSION = "evidence_committer_v1"
AFFORDANCE_SCHEMA_VERSION = "delta_affordance_v1"
COMPILED_DELTA_SCHEMA_VERSION = "compiled_graph_delta_v1"
AFFORDANCE_SCHEMA_VERSION_V3 = "delta_affordance_compiler_v3"
CONTROLLED_WORLD_EXPANSION_VERSION = "controlled_world_expansion_v1"
PLANNER_SCHEMA_VERSION_V3 = "slim_world_planner_v3"
NATURAL_REALIZER_VERSION_V3 = "natural_realizer_v3"
PLANNER_CONTRACT_VERSION_V3_1 = "slim_world_planner_contract_v3_1"
PLANNER_ACTION_VIEW_VERSION = "planner_action_view_v1"

__all__ = [
    "AFFORDANCE_SCHEMA_VERSION", "COMPILED_DELTA_SCHEMA_VERSION",
    "EVIDENCE_COMMITTER_SCHEMA_VERSION", "PLANNER_SCHEMA_VERSION",
    "REALIZER_SCHEMA_VERSION", "WORLD_MEMORY_SCHEMA_VERSION",
    "AFFORDANCE_SCHEMA_VERSION_V3", "CONTROLLED_WORLD_EXPANSION_VERSION",
    "PLANNER_SCHEMA_VERSION_V3", "NATURAL_REALIZER_VERSION_V3",
    "PLANNER_CONTRACT_VERSION_V3_1", "PLANNER_ACTION_VIEW_VERSION",
]
