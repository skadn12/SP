"""Isolated QA-only Memory-only Writer probe.

This sibling is deliberately not the production QA route.  Its Writer sees
only a frozen Memorax overview; transcript validation is a separate phase.
"""

BRANCH_VERSION = "qa_memory_only_probe_v1"
SCOPE = "qa-only/memory-only-writer/no-expansion/no-memory-write"
