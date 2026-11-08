from pathlib import Path

VERSION = 'h4_r02_basic_miner_v3_off_paired_v1'
ROOTS = ('cc-train-episode-198480', 'cc-train-episode-122711')
ARMS = ('BASE_OFF', 'V3_OFF')
RUNS = Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
P1_RUN = RUNS / 'h4_r02_feedback_ablation_p0_p1_v1_live_1'
V1_RUN = RUNS / 'h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1'
V2_RUN = RUNS / 'h4_r02_miner_binding_v2_acceptance_v1_live_1'
THREE_RUN = RUNS / 'h4_r02_miner_three_arm_diagnostic_v1_live_1'
DEFAULT_RUN = RUNS / (VERSION + '_live_1')
WORK_ORDER = RUNS / (VERSION + '_handoff/CODEX_WORK_ORDER.md')
CONFIG = Path('/root/autodl-tmp/gxy-selfplay/config.json')


def global_request_id(arm_id, root_id):
    return f'{arm_id}::{root_id}-intent-miner'


def root_path(run, arm_id, root_id):
    return Path(run) / 'arms' / arm_id / 'qa/shards/shard-00/roots' / root_id


SCHEDULE = tuple({'order': i, 'root_id': rid, 'arm_id': arm,
                  'request_id': rid + '-intent-miner',
                  'global_request_id': global_request_id(arm, rid)}
                 for i, (rid, arm) in enumerate(((ROOTS[0], ARMS[0]), (ROOTS[0], ARMS[1]),
                                                (ROOTS[1], ARMS[1]), (ROOTS[1], ARMS[0])), 1))
