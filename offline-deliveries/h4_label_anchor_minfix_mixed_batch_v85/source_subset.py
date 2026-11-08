"""Validate an explicit immutable new-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = ('cc-train-episode-103866', 'cc-train-episode-109164', 'cc-train-episode-112812', 'cc-train-episode-135466')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/CONFIG_TEMPLATE.json': 'a6345ebedaf4bd12ad9e939e78a81a056e19e40d8fff816c8d0bcab056954171', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/SOURCE_ADMISSION.COMPLETE.json': 'dbd1319c653119ddc6094a0dad864f8f4e0bc31cd1bf2138ab9d57140833fcaa', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/source_admission.py': '31d70e0fea2a0468c9b08a12f56c7fa9537dd76a7f7e03e58d27de38cb161848', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/SCENE_PLANS.json': 'b7a57b621420d0916f478ab681b48a1ef8c49933252661072a77844dc9adce59', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/ACTOR_REFERENCE_POLICY.json': '3e1a215e4efed888b61d77048a4a057923a83fed7af291ba19f287679e230e57', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/CHARACTER_REGISTRY.json': '83c1e84f2da1c8904f44b7dd6478aa1d6004c02e7e6a29f81afbe6a04260ed36', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_label_anchor_minfix_mixed_batch_v85/PARENT_POOL.json': 'a2b9156427a7e6e8fe18978e62f1cfd20664c6283729b68e3f882899d6753955', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'}

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
