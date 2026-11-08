"""Validate an explicit immutable same-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = ('cc-train-episode-103866', 'cc-train-episode-109164', 'cc-train-episode-112812', 'cc-train-episode-135466')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/CONFIG_TEMPLATE.json': '9f4e1a07b3f7da498dd68118156c05d73884b6fb53eab44fe437b43b4b311785', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/SOURCE_ADMISSION.COMPLETE.json': '56e5b4c4e5ff5b114fcf2f3126b463eb23532dce8c05e41597b7e5f6f30b5b57', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/source_admission.py': '52ce3df7558cc9f8ffbb90b7d79afcad3b37eae0ad8e53670cdb10ed9254f2bd', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/SCENE_PLANS.json': '72ddce740ef80d551d1b57f28ea9b25e6bef911f4b9f0acf4d57b6feadac0b22', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/ACTOR_REFERENCE_POLICY.json': '80a6ab403c95febf41f6d188cf16d043bb3a54db5e36a3cb11caa3d8ac3f3b12', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/CHARACTER_REGISTRY.json': '83c1e84f2da1c8904f44b7dd6478aa1d6004c02e7e6a29f81afbe6a04260ed36', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_anchor_records_mixed_batch_v88/PARENT_POOL.json': 'b6b8c5438d20388153a76e0fe44c99ee55e7e34dea22c26fcd63c55b8acd8144', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'}

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
