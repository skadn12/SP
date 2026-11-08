"""Offline token accounting against frozen provider and operational evidence.

No completion or metadata network calls occur in this module. Service catalog and
immutable upstream tokenizer snapshots are gathered before prepare. The exact
single-user/no-thinking wrapper is verified against 92 prior same-service calls.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from ..core import atomic_json
from ..h4_r02_feedback_ablation_p0_p1_v1.qa_run import GuardrailStop

MODEL = 'Qwen/Qwen3-14B'
SERVICE_BASE_URL = 'https://api.siliconflow.cn/v1'
TOKENIZER_REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
TOKEN_RESERVE = 512


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _catalog_row(html: str, model: str) -> dict[str, Any]:
    chunks = []
    for raw in re.findall(r'self\.__next_f\.push\((\[.*?\])\)</script>', html):
        value = json.loads(raw)
        if len(value) > 1 and isinstance(value[1], str):
            chunks.append(value[1])
    content = ''.join(chunks)
    at = content.index('"modelName":' + json.dumps(model))
    start = content.rfind('{', 0, at)
    row, _ = json.JSONDecoder().raw_decode(content[start:])
    if row.get('modelName') != model:
        raise GuardrailStop('TOKEN_CONTEXT_MODEL_SELECTOR_MISMATCH')
    return row


def _load_tokenizer(run: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(run / 'policy_snapshot/tokenizer', local_files_only=True)


def prepare_tokenizer(run: Path, config: Any) -> dict[str, Any]:
    """Verify frozen evidence and produce identity; exclusively local reads."""
    run = Path(run)
    evidence = run / 'audit/service_evidence'
    tokenizer_dir = run / 'policy_snapshot/tokenizer'
    if config.model != MODEL or config.base_url.rstrip('/') != SERVICE_BASE_URL:
        raise GuardrailStop('TOKEN_SERVICE_CONFIG_DRIFT')
    manifest = _json(evidence / 'evidence_manifest.json')
    for rel, expected in manifest['files'].items():
        path = run / rel
        if not path.is_file() or _sha(path) != expected['sha256']:
            raise GuardrailStop('TOKEN_SERVICE_EVIDENCE_DRIFT:' + rel)
    download = _json(tokenizer_dir / 'download_manifest.json')
    if download.get('model') != MODEL or download.get('revision') != TOKENIZER_REVISION:
        raise GuardrailStop('TOKENIZER_IDENTITY_DRIFT')
    for name, expected in download['files'].items():
        if _sha(tokenizer_dir / name) != expected['sha256']:
            raise GuardrailStop('TOKENIZER_FILE_DRIFT:' + name)
    discovery = _json(evidence / 'authenticated_model_discovery.json')
    if discovery.get('endpoint') != SERVICE_BASE_URL + '/models' or not any(
            row.get('id') == MODEL for row in discovery.get('selected_rows', [])):
        raise GuardrailStop('TOKEN_SERVICE_MODEL_NOT_DISCOVERED')
    row = _catalog_row((evidence / 'official_models.html').read_text(), MODEL)
    if row != _json(evidence / 'service_model_metadata.json')['model']:
        raise GuardrailStop('TOKEN_CONTEXT_SOURCE_PROJECTION_DRIFT')
    context_limit = row.get('contextLen')
    if not isinstance(context_limit, int) or context_limit <= 0 or row.get('status') != 'normal':
        raise GuardrailStop('TOKEN_CONTEXT_LIMIT_UNVERIFIED')
    tokenizer = _load_tokenizer(run)
    historical = _json(evidence / 'historical_token_verification.json')
    verified = []
    no_usage = []
    for record in historical['rows']:
        ledger_path, prompt_path = Path(record['ledger_path']), Path(record['prompt_path'])
        if (_sha(ledger_path) != record['ledger_sha256'] or
                _sha(prompt_path) != record['prompt_sha256']):
            raise GuardrailStop('TOKEN_OPERATIONAL_SOURCE_DRIFT:' + record['request_id'])
        ledger = _json(ledger_path)
        backend = ledger.get('backend_identity') or {}
        if backend.get('model') != MODEL or backend.get('base_url') != SERVICE_BASE_URL:
            raise GuardrailStop('TOKEN_OPERATIONAL_BACKEND_DRIFT')
        reported = (ledger.get('response_metadata', {}).get('usage') or {}).get('prompt_tokens')
        if reported is None:
            no_usage.append(record['request_id'])
            continue
        count = len(tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt_path.read_text()}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False))
        if count != reported:
            raise GuardrailStop('TOKEN_OPERATIONAL_WRAPPER_MISMATCH:' + record['request_id'])
        verified.append(record)
    kinds = {'miner' if '-intent-miner' in r['request_id'] else
             'repair' if '-repair-' in r['request_id'] else 'writer' for r in verified}
    if kinds != {'miner', 'writer', 'repair'} or not verified:
        raise GuardrailStop('TOKEN_OPERATIONAL_COVERAGE_INSUFFICIENT')
    output_limit = max(r['max_tokens'] for r in verified)
    if output_limit < 8192:
        raise GuardrailStop('TOKEN_OUTPUT_8192_NOT_VERIFIED')
    identity = {
        'version': 'p2_2_token_preflight_v1', 'status': 'PASS', 'offline_prepare': True,
        'model': MODEL, 'service_base_url': SERVICE_BASE_URL,
        'service_model_revision': None,
        'service_revision_note': 'Hosted provider does not expose checkpoint/tokenizer revision.',
        'context_limit': context_limit,
        'context_limit_source': {'url': 'https://www.siliconflow.cn/models',
            'path': str(evidence / 'official_models.html'),
            'sha256': _sha(evidence / 'official_models.html'),
            'selector': 'modelName == Qwen/Qwen3-14B; field contextLen'},
        'service_model_discovery': {'path': str(evidence / 'authenticated_model_discovery.json'),
            'sha256': _sha(evidence / 'authenticated_model_discovery.json')},
        'tokenizer': {'model': MODEL, 'revision': TOKENIZER_REVISION,
            'directory': str(tokenizer_dir), 'files': download['files']},
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'chat_rendering': {'messages': [{'role': 'user', 'content': 'exact prompt'}],
            'add_generation_prompt': True, 'enable_thinking': False, 'tools': None},
        'operational_wrapper_verification': {
            'source_path': str(evidence / 'historical_token_verification.json'),
            'source_sha256': _sha(evidence / 'historical_token_verification.json'),
            'verified_completed_count': len(verified), 'mismatch_count': 0,
            'without_usage_count': len(no_usage), 'covered_request_kinds': sorted(kinds),
            'qualification': 'Exact rendered count matches every prior completed same-service QA request; current live usage is checked for drift.'},
        'verified_supported_max_output_tokens': output_limit,
        'output_limit_source': 'Prior same-service completed Miner/Writer/repair request max_tokens; global service output ceiling is not advertised.',
        'global_service_output_ceiling': None,
        'fixed_request_max_outputs': {'miner': 4096, 'writer': 8192, 'repair': 8192},
        'reserve_tokens': TOKEN_RESERVE,
        'evidence_manifest_sha256': _sha(evidence / 'evidence_manifest.json'),
        'download_manifest_sha256': _sha(tokenizer_dir / 'download_manifest.json'),
    }
    atomic_json(run / 'audit/token_identity.json', identity)
    return identity


def measure_request(tokenizer: Any, *, prompt: str, max_output_tokens: int,
                    request_id: str, context_limit: int,
                    supported_max_output: int, components: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Measure the final message, including chat wrapper, without truncation."""
    if not isinstance(prompt, str):
        raise GuardrailStop('TOKEN_PROMPT_TYPE_INVALID')
    if max_output_tokens not in (4096, 8192) or max_output_tokens > supported_max_output:
        raise GuardrailStop('TOKEN_OUTPUT_BUDGET_DRIFT')
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    rendered_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    counts = {name: len(tokenizer.encode(str((components or {}).get(name, '')), add_special_tokens=False))
              for name in ('transcript', 'feedback', 'binding', 'brief', 'template')}
    total = rendered_tokens + max_output_tokens + TOKEN_RESERVE
    return {
        'request_id': request_id, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'rendered_message_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
        'prompt_content_tokens': prompt_tokens, 'rendered_message_tokens': rendered_tokens,
        'chat_wrapper_increment_tokens': rendered_tokens - prompt_tokens,
        'max_output_tokens': max_output_tokens, 'reserve_tokens': TOKEN_RESERVE,
        'context_limit': context_limit, 'required_context_tokens': total,
        'remaining_context_tokens': context_limit - total,
        'status': 'PASS' if total <= context_limit else 'BLOCKED',
        'component_tokens': counts,
        'components_overlap': True,
        'component_count_note': 'Diagnostic independent tokenizations overlap: feedback/binding may be nested in brief; template may include rendered sections. BPE boundaries are nonadditive. Only rendered_message_tokens determines admission.',
    }


class TokenPreflight:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.identity = _json(self.run / 'audit/token_identity.json')
        if self.identity.get('status') != 'PASS':
            raise GuardrailStop('TOKEN_IDENTITY_NOT_VERIFIED')
        for name, expected in self.identity['tokenizer']['files'].items():
            if _sha(self.run / 'policy_snapshot/tokenizer' / name) != expected['sha256']:
                raise GuardrailStop('TOKENIZER_FILE_DRIFT:' + name)
        self.tokenizer = _load_tokenizer(self.run)
        self.report_path = self.run / 'audit/token_report.json'

    def _save(self, record: Mapping[str, Any]) -> dict[str, Any]:
        report = _json(self.report_path) if self.report_path.exists() else {
            'version': 'p2_2_token_preflight_v1', 'identity_path': str(self.run / 'audit/token_identity.json'),
            'requests': []}
        rows = report['requests']
        old = next((row for row in rows if row['request_id'] == record['request_id']), None)
        if old is not None:
            expected_keys = ('prompt_sha256', 'rendered_message_sha256', 'max_output_tokens', 'required_context_tokens')
            if any(old.get(k) != record.get(k) for k in expected_keys):
                raise GuardrailStop('TOKEN_REQUEST_REPLAY_DRIFT:' + record['request_id'])
            # Preserve post-response usage and the richer live components.
            merged = {**old, **record}
            rows[rows.index(old)] = merged
        else:
            rows.append(dict(record))
        report['status'] = 'PASS' if all(r['status'] == 'PASS' for r in rows) else 'BLOCKED'
        report['request_count'] = len(rows)
        atomic_json(self.report_path, report)
        return dict(record)

    def check(self, prompt: str, max_output_tokens: int, request_id: str,
              components: Mapping[str, str] | None = None) -> dict[str, Any]:
        record = measure_request(self.tokenizer, prompt=prompt, max_output_tokens=max_output_tokens,
            request_id=request_id, context_limit=self.identity['context_limit'],
            supported_max_output=self.identity['verified_supported_max_output_tokens'], components=components)
        self._save(record)
        if record['status'] != 'PASS':
            raise GuardrailStop('TOKEN_CONTEXT_EXCEEDED:' + request_id)
        return record

    def verify_usage(self, request_id: str, usage: Mapping[str, Any] | None) -> dict[str, Any]:
        """Check a completed response's known input count; never invent usage."""
        report = _json(self.report_path)
        record = next(row for row in report['requests'] if row['request_id'] == request_id)
        observed = (usage or {}).get('prompt_tokens')
        record['service_prompt_tokens'] = observed
        record['service_usage_count_status'] = ('UNKNOWN' if observed is None else
            'MATCH' if observed == record['rendered_message_tokens'] else 'MISMATCH')
        if record['service_usage_count_status'] == 'MISMATCH':
            record['status'] = 'BLOCKED'
        self._save(record)
        if record['service_usage_count_status'] == 'MISMATCH':
            raise GuardrailStop('TOKEN_CURRENT_SERVICE_WRAPPER_DRIFT:' + request_id)
        return record
