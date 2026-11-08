from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.current_method.restart_n5_spell.difficulty_probe_suite_v1.common.io_utils import prepare_empty, sha256_file, write_json
from scripts.current_method.restart_n5_spell.planner_realizer_temporal_boundary_probe_v1.stage_transition_reanalysis_v2 import strict_stage_audit

RUN_BASE = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')

def run(*, source_run: Path, case_id: str, run_root: Path) -> dict:
    prepare_empty(run_root)
    child_path = source_run / 'live_child_dialogue.json'
    metrics_path = source_run / 'live_metrics.json'
    child = json.loads(child_path.read_text())
    audit = strict_stage_audit(child, case_id)
    metrics = {
        'probe_version': 'planner_realizer_temporal_single_strict_reanalysis_v1',
        'source_run': str(source_run.resolve()),
        'source_child_sha256': sha256_file(child_path),
        'source_metrics_sha256': sha256_file(metrics_path),
        'case_id': case_id,
        'strict_stage_audit_pass': bool(audit['overall_pass']),
        'violations': audit['violations'],
        'conclusion': 'STRICT_STAGE_PASS' if audit['overall_pass'] else 'STRICT_STAGE_FAIL',
        'analysis_execution_api_calls': 0,
        'planner_calls': 0, 'qa_writer_calls': 0, 'solver_calls': 0,
        'writer_calls': 0, 'auditor_calls': 0,
        'memorax_search_calls': 0, 'memorax_add_calls': 0,
    }
    write_json(run_root / 'strict_audit.json', audit)
    write_json(run_root / 'metrics.json', metrics)
    write_json(run_root / 'manifest.json', {'isolated': True, 'source_run_read_only': str(source_run.resolve()), 'api_calls': 0})
    (run_root / 'report.md').write_text(
        '# Realizer Temporal Single Strict Reanalysis v1\n\n'
        + f"Verdict: **{metrics['conclusion']}**\n\n"
        + f"- Case: `{case_id}`\n- Strict pass: `{metrics['strict_stage_audit_pass']}`\n"
        + f"- Violations: `{metrics['violations']}`\n- API calls: `0`\n\n"
        + 'This reanalysis does not modify or rerun the source generation.\n',
        encoding='utf-8',
    )
    return metrics

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_root = (args.run_root or RUN_BASE / f'planner_realizer_temporal_single_strict_reanalysis_v1_{stamp}').resolve()
    print(json.dumps({'run_root': str(run_root), 'metrics': run(source_run=args.source_run, case_id=args.case_id, run_root=run_root)}, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
