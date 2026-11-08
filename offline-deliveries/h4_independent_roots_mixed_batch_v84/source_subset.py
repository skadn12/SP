"""Validate an explicit immutable new-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = ('cc-train-episode-103866', 'cc-train-episode-109164', 'cc-train-episode-112812', 'cc-train-episode-135466')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/CONFIG_TEMPLATE.json': '4b4d071a29532a41e4eb997699a922c5fb5b722473b786d4c7e7c0e7e56d15c2', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/SOURCE_ADMISSION.COMPLETE.json': '79615ee28656dbe5196df7f8fa0f29c907e0ebd09f71c3f872b951f3817d6ca5', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/source_admission.py': '06b51b2193631271921dbd3ee9b92cd782b5faa2eb61defbbe5870bba3c4a6ac', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/SCENE_PLANS.json': 'b7a57b621420d0916f478ab681b48a1ef8c49933252661072a77844dc9adce59', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/ACTOR_REFERENCE_POLICY.json': 'da92f88c65cb29f49179b4138da129bb79772fd555d02bbadb669d37ecc8d058', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/CHARACTER_REGISTRY.json': '83c1e84f2da1c8904f44b7dd6478aa1d6004c02e7e6a29f81afbe6a04260ed36', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_independent_roots_mixed_batch_v84/PARENT_POOL.json': '805c1f752ce36293c2680c634a4fd3d6c75dc93827272f22cf4536d3c78b3968', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'}

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
