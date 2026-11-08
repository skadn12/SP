"""Normalize public service settings, without reading or copying credentials."""
from __future__ import annotations
import copy
import json
import hashlib
from pathlib import Path
from .constants import ARMS, ROOTS, SOURCE_ARM, SOURCE_RUN, STRONG_CONFIG, QWEN_CONFIG, WORKSPACE


def _service_exception(raw, config_path, endpoint, parameters):
    """Load the specific user-approved documentation exception, never infer support."""
    reference = raw.get('user_exception_record')
    if reference is None:
        return None
    path = Path(reference)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    path = path.resolve()
    value = read(path)
    assert_no_secrets(value)
    expected = {
        'record_id': '2026-09-18-user-authorized-tju-service-unknowns-v1',
        'user_explicitly_authorized': True,
        'scope': {'roots': list(ROOTS), 'arms': list(ARMS), 'slots_per_root': 3},
        'budgets': {'primary_calls': 12, 'maximum_repair_calls': 12,
                    'maximum_logical_calls': 24, 'maximum_http_attempts': 24},
        'strong_service': {'endpoint': 'https://ai.tju.edu.cn/api/v3/chat/completions',
                           'model': 'tju-llm'},
        'requested_parameters': {'temperature': 0.1, 'max_tokens': 8192,
                                 'response_format': {'type': 'json_object'}},
        'strong_reasoning': {'mode': 'service_default', 'verified_status': 'UNKNOWN'},
        'allow_unknown_service_capacity': True,
        'allow_unknown_parameter_support': True,
        'stop_on_any_technical_failure': True,
        'automatic_retries': 0, 'ssl_failure_fallback': False,
        'extra_paid_connectivity_tests': 0,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('USER_SERVICE_EXCEPTION_SCOPE_OR_POLICY_MISMATCH')
    if endpoint != expected['strong_service']['endpoint'] or raw.get('api', {}).get('model') != 'tju-llm':
        raise ValueError('USER_SERVICE_EXCEPTION_ENDPOINT_OR_MODEL_MISMATCH')
    if parameters != expected['requested_parameters'] or raw.get('generation', {}).get('reasoning_settings') is not None:
        raise ValueError('USER_SERVICE_EXCEPTION_REQUIRES_EXACT_REQUEST_AND_DEFAULT_REASONING')
    return {**value, 'record_path': str(path),
            'record_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_no_secrets(value, path='config'):
    """Fail closed on credential values; references to environment/file are public."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {'api_key', 'apikey', 'authorization', 'bearer', 'access_token', 'secret', 'password', 'headers'}:
                raise ValueError('SECRET_FIELD_FORBIDDEN:' + path + '.' + key)
            assert_no_secrets(item, path + '.' + key)
    elif isinstance(value, list):
        for item in value:
            assert_no_secrets(item, path)


def _qwen_profile(source_run):
    source_run = Path(source_run)
    requests = [read(source_run/'policy_snapshot/primary_requests'/SOURCE_ARM/root/f'intent_{i}'/'request.json')
                for root in ROOTS for i in range(1, 4)]
    payload = requests[0]['payload']
    parameters = {k: v for k, v in payload.items() if k not in ('model', 'messages')}
    if any({k: v for k, v in r['payload'].items() if k not in ('model', 'messages')} != parameters
           or r['payload']['model'] != payload['model'] for r in requests):
        raise ValueError('SOURCE_QWEN_PARAMETER_DRIFT')
    identity = read(source_run/'audit/token_identity.json')
    execution = read(source_run/'execution_contract.json')
    settings = read(QWEN_CONFIG)
    assert_no_secrets(settings)
    if settings.get('request_parameters') != parameters:
        raise ValueError('QWEN_CONFIG_DIFFERS_FROM_REAL_FROZEN_WRITER_REQUEST')
    base = execution['generation']['base_url'].rstrip('/')
    return {
        'arm_id': ARMS[0], 'provider': 'siliconflow', 'model': payload['model'],
        'base_url': base, 'endpoint': base + '/chat/completions',
        'interface_format': 'openai_compatible_chat_completions', 'authentication_scheme': 'bearer',
        'api_key_env': settings.get('api_key_env'),
        'credential_file': str(WORKSPACE/'config.json'), 'credential_format': 'challenger_json',
        'timeout_seconds': execution['generation']['timeout_seconds'],
        'request_parameters': parameters, 'output_limit': parameters['max_tokens'],
        'seed_policy': 'not_sent_by_original_client',
        'reasoning_policy': {'enable_thinking': False},
        'limits': {'context_window_tokens': identity['context_limit'],
                   'max_output_tokens': identity['verified_supported_max_output_tokens'],
                   'reasoning_budget_tokens': 0, 'reasoning_in_output_budget': True,
                   'reserve_tokens': identity['reserve_tokens']},
        'token_accounting': {'method': 'huggingface_chat_template',
            'local_tokenizer_path': str(source_run/'policy_snapshot/tokenizer'),
            'model': identity['model'], 'revision': identity['tokenizer']['revision'],
            'chat_template_kwargs': {'add_generation_prompt': True, 'enable_thinking': False},
            'evidence_path': str(source_run/'audit/token_identity.json')},
        'service_evidence': [{'path': str(source_run/'audit/token_identity.json'),
                              'kind': 'historical_service_and_tokenizer_identity'}],
        'pricing': 'UNKNOWN', 'unresolved': [], 'status': 'READY_OFFLINE',
        'http_attempts_per_logical_call': 1, 'transport_retries': 0,
        'sdk_retries': 0, 'ssl_fallback': False,
        'source_configuration_note': 'Actual TEMPORAL Writer payload; no Realizer configuration reused.',
        'requested_parameter_adoption': {key: 'UNKNOWN_FOR_THIS_RUN' for key in parameters},
    }


def _strong_profile(path):
    raw = read(path)
    assert_no_secrets(raw)
    api, generation = raw.get('api', {}), raw.get('generation', {})
    unresolved = []
    endpoint = api.get('endpoint')
    base = api.get('base_url')
    if not endpoint and base and api.get('request_path'):
        endpoint = base.rstrip('/') + '/' + api['request_path'].lstrip('/')
    for key in ('model', 'interface_format', 'authentication_scheme', 'api_key_env', 'timeout_seconds'):
        if api.get(key) is None:
            unresolved.append('api.' + key)
    if not endpoint:
        unresolved.append('api.endpoint_or_base_url_and_request_path')
    if api.get('interface_format') != 'openai_compatible_chat_completions':
        unresolved.append('api.interface_format_supported_adapter')
    if api.get('authentication_scheme') != 'bearer':
        unresolved.append('api.authentication_scheme_supported_adapter')
    parameters = copy.deepcopy(raw.get('request_parameters') or {})
    support = raw.get('parameter_support') or {}
    supported = set(support.get('supported') or [])
    unsupported = set(support.get('unsupported') or [])
    mapping = {'temperature': 'temperature', 'max_output_tokens': generation.get('output_parameter', 'max_tokens'),
               'response_format': 'response_format', 'stop': 'stop'}
    for key, parameter in mapping.items():
        value = generation.get(key)
        if value is not None and parameter not in parameters:
            parameters[parameter] = value
        if key != 'stop' and value is None and parameter not in parameters and parameter not in unsupported:
            unresolved.append('generation.' + key + '_or_explicit_unsupported')
    seed_policy = generation.get('seed_policy')
    if seed_policy is None:
        unresolved.append('generation.seed_policy')
    elif seed_policy not in ('not_sent', 'not_sent_by_original_client', 'fixed'):
        raise ValueError('SEED_POLICY_UNSUPPORTED')
    if seed_policy in ('not_sent', 'not_sent_by_original_client') and ('seed' in parameters or generation.get('seed') is not None):
        raise ValueError('SEED_SENT_DESPITE_OMISSION_POLICY')
    if generation.get('seed_policy') == 'fixed':
        if generation.get('seed') is None:
            unresolved.append('generation.seed')
        else:
            parameters['seed'] = generation['seed']
    reasoning = generation.get('reasoning_settings')
    if reasoning is None:
        unresolved.append('generation.reasoning_settings')
    elif not isinstance(reasoning, dict):
        raise ValueError('REASONING_SETTINGS_MUST_BE_EXPLICIT_PARAMETER_OBJECT')
    else:
        parameters.update(reasoning)
    if any(k in parameters for k in ('model', 'messages', 'stream', 'tools', 'tool_choice')):
        raise ValueError('REQUEST_PARAMETERS_MUST_NOT_CHANGE_MESSAGES_OR_ADD_STAGES')
    if parameters.get('response_format') is not None and parameters['response_format'] != {'type':'json_object'}:
        raise ValueError('STRONG_ONLY_SCHEMA_CHANGE_FORBIDDEN')
    if parameters.get('n', 1) != 1:
        raise ValueError('MULTIPLE_COMPLETIONS_FORBIDDEN')
    for key in parameters:
        if key not in supported:
            unresolved.append('parameter_support.' + key)
        if key in unsupported:
            raise ValueError('PARAMETER_DECLARED_BOTH_USED_AND_UNSUPPORTED:' + key)
    output = raw.get('output_limit') or parameters.get('max_tokens') or parameters.get('max_completion_tokens')
    if not isinstance(output, int) or isinstance(output, bool) or output <= 0:
        unresolved.append('output_limit')
        output = None
    output_fields = [parameters[k] for k in ('max_tokens', 'max_completion_tokens') if parameters.get(k) is not None]
    if not output_fields:
        unresolved.append('request_parameters.explicit_output_budget_field')
    elif len(output_fields) != 1 or output_fields[0] != output:
        raise ValueError('OUTPUT_LIMIT_MUST_MATCH_ONE_SENT_TOKEN_BUDGET')
    limits = copy.deepcopy(raw.get('service_limits') or {})
    for key in ('context_window_tokens', 'max_output_tokens', 'reasoning_budget_tokens'):
        if limits.get(key) is None:
            unresolved.append('service_limits.' + key)
    if limits.get('reasoning_in_output_budget') is None:
        unresolved.append('service_limits.reasoning_in_output_budget')
    limits.setdefault('reserve_tokens', 512)
    for key in ('context_window_tokens','max_output_tokens','reasoning_budget_tokens','reserve_tokens'):
        value = limits.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < (0 if key in ('reasoning_budget_tokens','reserve_tokens') else 1)):
            raise ValueError('INVALID_SERVICE_LIMIT:' + key)
    if limits.get('reasoning_in_output_budget') is not None and not isinstance(limits['reasoning_in_output_budget'], bool):
        raise ValueError('REASONING_OUTPUT_ACCOUNTING_MUST_BE_BOOLEAN')
    if output is not None and limits.get('max_output_tokens') is not None and output > limits['max_output_tokens']:
        raise ValueError('REQUESTED_OUTPUT_EXCEEDS_SERVICE_LIMIT')
    accounting = copy.deepcopy(raw.get('token_accounting') or {})
    for key in ('method', 'local_tokenizer_path', 'chat_template_kwargs', 'evidence_path'):
        if accounting.get(key) is None:
            unresolved.append('token_accounting.' + key)
    if accounting.get('method') not in (None, 'huggingface_chat_template'):
        unresolved.append('token_accounting.supported_method')
    if not raw.get('service_evidence'):
        unresolved.append('service_evidence')
    exception = _service_exception(raw, path, endpoint, parameters)
    unknown_configuration = sorted(set(unresolved))
    waived = []
    if exception:
        permitted_unknowns = {
            'generation.reasoning_settings',
            'parameter_support.temperature', 'parameter_support.max_tokens', 'parameter_support.response_format',
            'service_limits.context_window_tokens', 'service_limits.max_output_tokens',
            'service_limits.reasoning_budget_tokens', 'service_limits.reasoning_in_output_budget',
            'token_accounting.method', 'token_accounting.local_tokenizer_path',
            'token_accounting.chat_template_kwargs', 'token_accounting.evidence_path',
        }
        waived = sorted(set(unresolved) & permitted_unknowns)
        unresolved = sorted(set(unresolved) - permitted_unknowns)
        reasoning = {'mode': 'service_default', 'verified_status': 'UNKNOWN', 'request_parameters': {}}
    return {
        'arm_id': ARMS[1], 'provider': api.get('provider'), 'model': api.get('model'),
        'base_url': base, 'endpoint': endpoint, 'interface_format': api.get('interface_format'),
        'authentication_scheme': api.get('authentication_scheme'), 'api_key_env': api.get('api_key_env'),
        'credential_file': api.get('credential_file'), 'credential_format': api.get('credential_format', 'raw_key'),
        'timeout_seconds': api.get('timeout_seconds'), 'request_parameters': parameters,
        'output_limit': output, 'seed_policy': generation.get('seed_policy'), 'reasoning_policy': reasoning,
        'limits': limits, 'token_accounting': accounting,
        'service_evidence': copy.deepcopy(raw.get('service_evidence') or []),
        'user_reported_backend_model': raw.get('user_reported_backend_model'),
        'backend_model_identity_status': 'USER_REPORTED_NOT_SERVICE_VERIFIED',
        'pricing': copy.deepcopy(raw.get('pricing') or 'UNKNOWN'),
        'unknown_configuration': unknown_configuration, 'waived_configuration': waived,
        'user_service_exception': exception,
        'requested_parameter_adoption': {key: 'UNKNOWN' for key in parameters},
        'parameter_support': copy.deepcopy(support),
        'unresolved': sorted(set(unresolved)),
        'status': 'WAITING_FOR_STRONG_MODEL_CONFIG' if unresolved else ('READY_WITH_USER_SERVICE_EXCEPTION' if exception else 'READY_OFFLINE'),
        'http_attempts_per_logical_call': 1, 'transport_retries': 0,
        'sdk_retries': 0, 'ssl_fallback': False,
    }


def load_profiles(strong_config_path=STRONG_CONFIG, source_run=SOURCE_RUN):
    profiles = {ARMS[0]: _qwen_profile(source_run), ARMS[1]: _strong_profile(strong_config_path)}
    assert_no_secrets(profiles)
    return profiles
