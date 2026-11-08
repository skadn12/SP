"""Only arm adaptation: unchanged Guided prompt plus the literal frozen suffix."""
from pathlib import Path
from ..core import sha256_bytes
from ..h4_qa_mining_hybrid_generation_v1.contract_guided_generator import build_prompt, build_repair_prompt
from ..h4_qa_multiroot_k3_t08_probe_v1.generation import _append_feedback_context
from .constants import ARMS
GUIDANCE_SHA256 = '4dfa2c6d97c1e0cc56f7d39bae7ac9216cb368635d8928056e5f0b91bd2f6fc9'
GUIDANCE_FILENAME = 'policy_snapshot/scms_quality_guidance_v1.txt'

def guidance_text(run):
    data = (Path(run) / GUIDANCE_FILENAME).read_bytes()
    if sha256_bytes(data) != GUIDANCE_SHA256:
        raise ValueError('SCMS_GUIDANCE_BYTE_IDENTITY_DRIFT')
    return data.decode('utf-8')

def quality_suffix(run, arm_id):
    if arm_id not in ARMS:
        raise ValueError('UNKNOWN_SCMS_ARM')
    return guidance_text(run) if arm_id == ARMS[1] else ''

def prompt_spec(expanded, contract, arm_id, run):
    return _append_feedback_context(build_prompt(expanded, contract, intent_guidance=True), quality_suffix(run, arm_id))

def repair_spec(expanded, contract, raw, arm_id, run):
    return _append_feedback_context(build_repair_prompt(expanded, contract, raw, 'contract_guided', intent_guidance=True), quality_suffix(run, arm_id))
