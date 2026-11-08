"""Fixed experimental scope; model-visible prompts never include arm identities."""
from pathlib import Path
VERSION='h4_r02_miner_three_arm_diagnostic_v1'
ROOTS=tuple(f'cc-train-episode-{n}' for n in (198480,122711))
ARMS=('A_OFF','B_OBSERVATION_ONLY','C_P2_2_TRANSFER')
SCHEDULE=tuple({'order':i+1,'root_id':r,'arm_id':a,'request_id':r+'-intent-miner',
               'global_request_id':a+'::'+r+'-intent-miner'} for i,(r,a) in enumerate(
    [(ROOTS[0],a) for a in ARMS]+[(ROOTS[1],a) for a in reversed(ARMS)]))
RUNS=Path('/root/autodl-tmp/gxy-selfplay/restart-n5-runs')
P1=RUNS/'h4_r02_feedback_ablation_p0_p1_v1_live_1'
V1_RUN=RUNS/'h4_r02_feedback_transfer_p2_2_acceptance_v1_live_1'
V2_RUN=RUNS/'h4_r02_miner_binding_v2_acceptance_v1_live_1'
DEFAULT_RUN=RUNS/'h4_r02_miner_three_arm_diagnostic_v1_live_1'
WORK_ORDER=RUNS/'h4_r02_miner_three_arm_diagnostic_v1_handoff/CODEX_WORK_ORDER.md'
CONFIG=Path('/root/autodl-tmp/gxy-selfplay/config.json')

def root_path(run,arm_id,root_id):
    return Path(run)/'arms'/arm_id/'qa/shards/shard-00/roots'/root_id

def global_request_id(arm_id,root_id):
    return arm_id+'::'+root_id+'-intent-miner'
