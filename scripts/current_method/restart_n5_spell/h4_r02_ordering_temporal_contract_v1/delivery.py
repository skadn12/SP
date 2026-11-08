from pathlib import Path
from ..core import atomic_text
from .constants import DEFAULT_RUN, REPO, CONFIG


def write_delivery(run=DEFAULT_RUN):
    run=Path(run)
    prefix=f'env PYTHONPATH={REPO} /root/miniconda3/bin/python -B -m scripts.current_method.restart_n5_spell.h4_r02_ordering_temporal_contract_v1.run'
    atomic_text(run/'EXECUTE_AFTER_AUTHORIZATION.md',f'''# Future execution commands

These commands identify the frozen experiment. Preparation and preflight are offline. The live command requires a later explicit authorization.

```bash
{prefix} prepare --run-dir {run} --config {CONFIG}
{prefix} preflight --run-dir {run} --config {CONFIG}
{prefix} live --run-dir {run} --config {CONFIG} --execute-after-authorization
{prefix} report --run-dir {run} --config {CONFIG}
```

The live scheduler stops on unknown completion, frozen drift, budget anomaly, or full actual repair prompt capacity failure. It does not resend under a new ID or directory.
''')
    atomic_text(run/'ARTIFACT_INDEX.md','''# Artifact Index

- `RUN_STATUS.md`: terminal offline status.
- `reports/preparation_report.md`: implementation boundary and outcome.
- `execution_contract.json`, `schedule.json`: future identity and 12+<=12 budget.
- `policy_snapshot/primary_requests/`: twelve full primary requests.
- `policy_snapshot/schema/`, `policy_snapshot/examples/`: 4/5/6 contracts and valid synthetic examples.
- `audit/offline_integration_report.json`: Guided/Free primary/repair integration.
- `audit/test_report.json`: regression results.
- `audit/dependency_coverage_report.json`, `audit/frozen_manifest.json`: transitive closure and freeze.
- `audit/token_budget_summary.json`: exact local primary capacity and dynamic repair rule.
- `review_policy.json`: future content, dedup, coverage and cost diagnosis.
- `EXECUTE_AFTER_AUTHORIZATION.md`: exact commands.
''')
