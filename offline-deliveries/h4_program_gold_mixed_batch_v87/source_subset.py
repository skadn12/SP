"""Validate an explicit immutable new-root plan with unchanged full-source gate."""
from common import HERE, read, need, sha
import source_admission
ROOTS = ('cc-train-episode-103866', 'cc-train-episode-109164', 'cc-train-episode-112812', 'cc-train-episode-135466')
PINS = {'/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/CONFIG_TEMPLATE.json': '756962139ea827eed74f93edc2a4bb35367dd54eb3ce53c37db9e00bae5963a8', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/SOURCE_ADMISSION.COMPLETE.json': '24679211c6554df22494576d8d56944c947686c2f29607db1cc5362d9e621914', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/source_admission.py': 'bf65667d2b0f881ec79f3417c8d5c2883bfd44a5a72a71ecf83dc49177453a1d', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/SCENE_PLANS.json': 'c70e4d59bc199329b5aca4c3c1953a50f5e720a8b553b6f11aa5f3ae16792823', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/ACTOR_REFERENCE_POLICY.json': 'f0c443cd92d54e9c4da232d7869f19068009d6b7a8b3f87d70821b8fcd992fba', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/CHARACTER_REGISTRY.json': '83c1e84f2da1c8904f44b7dd6478aa1d6004c02e7e6a29f81afbe6a04260ed36', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_program_gold_mixed_batch_v87/PARENT_POOL.json': '887c4c070af0642b50705b993255ebd02d1de8461187fe9bc43719ea2f0af2ff', '/root/autodl-tmp/gxy-selfplay/offline-deliveries/h4_source_consistency_minfix_v77/source_gate.py': '2c5d6b8bd0eeb50de96bb37800f0599ae6db9b5764a7a3e181b542e89c40e58e'}

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
