"""Validate an explicit immutable new-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = ('cc-train-episode-103866', 'cc-train-episode-109164', 'cc-train-episode-112812', 'cc-train-episode-135466')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/CONFIG_TEMPLATE.json': '98167deac5f7d4fea255eb74cdb4ced095e10471551788486312a814c4053d21', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/SOURCE_ADMISSION.COMPLETE.json': '90afacf345c1a99fb101716162c29c26e442c3db638772547b4dfc1e7f7f1717', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/source_admission.py': '99600e51fdd5df56eabf676a7003d831387c37987673afad068b1850f39493b1', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/SCENE_PLANS.json': 'b7a57b621420d0916f478ab681b48a1ef8c49933252661072a77844dc9adce59', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/ACTOR_REFERENCE_POLICY.json': '1bc200d45d1a32677063abba93935a5e3b5f96450d8d25f2de84c105c343f60b', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/CHARACTER_REGISTRY.json': '83c1e84f2da1c8904f44b7dd6478aa1d6004c02e7e6a29f81afbe6a04260ed36', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_reason_before_gold_mixed_batch_v86/PARENT_POOL.json': 'dfb875f674774acdd0afb4f5dc6231786db9b4b8e33b3911e7ed154210aca005', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'}

def verified_summary():
    for path, expected in PINS.items():
        need(sha(path)==expected,'upstream_binding_changed:'+path)
    summary=source_admission.verify()
    by_root={r['root_id']:r for r in summary['roots']}
    need(all(r in by_root and by_root[r]['decision']=='READY' for r in ROOTS),'selected_source_not_ready')
    return summary

def validate(value):
    summary=verified_summary();expected=read(HERE/'CONFIG_TEMPLATE.json')
    need(value==expected,'ready_subset_config_changed')
    need(tuple(s['root_id'] for s in value['multiselect_config']['slots'])==ROOTS,'root_selection_changed')
    by_root={r['root_id']:r for r in summary['roots']}
    for s in value['multiselect_config']['slots']:
        need(by_root[s['root_id']]['source']=={'path':s['source_path'],'sha256':s['source_sha256']},'selected_source_binding_changed')
    return value
