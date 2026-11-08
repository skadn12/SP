"""Narrow offline delivery for endpoint and credential handoff corrections."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import difflib
import io
import os
import subprocess
import unittest

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[4]
FROZEN = PROJECT / "restart-n5-runs/h4_round1_r03_incremental_readiness_v2_offline_preflight_live_1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(output, name, value):
    path = Path(output) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2))


def capture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    expected = json.loads((FROZEN / "implementation_sha256.json").read_text())
    if any(digest(path) != value for path, value in expected.items()):
        raise ValueError("source_changed_before_connection_fix")
    save(output, "implementation_before_sha256.json", expected)
    for name in ("run.py", "readiness.py", "test_readiness.py"):
        destination = output / "implementation_before" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(PACKAGE / name, destination)
    protected = {name: value for name, value in expected.items()
                 if Path(name).parent != PACKAGE or Path(name).name not in {"run.py", "readiness.py", "test_readiness.py"}}
    protected.update(json.loads((FROZEN / "protected_after.json").read_text()))
    for run in (FROZEN, PROJECT / "restart-n5-runs/h4_round1_r03_incremental_retention_authorized_single_root_live_1"):
        protected.update({str(path): digest(path) for path in run.rglob("*") if path.is_file()})
    save(output, "protected_before_sha256.json", protected)


def run(output):
    from .transport_config import VERSION, credential_environment
    from .test_transport_config import corrected_production_model
    from .test_readiness import exercise_http_cli
    from .run import implementation_manifest, offline_network_guard, production_registry, SingleShotTransport
    from .bundle import validate_bundle_feedback
    from .capacity import local_counter
    output = Path(output)
    if not (output / "protected_before_sha256.json").exists() or (output / "final_report.json").exists():
        raise ValueError("new_captured_output_required")
    before = json.loads((output / "protected_before_sha256.json").read_text())
    if any(digest(path) != value for path, value in before.items()):
        raise ValueError("protected_files_changed")
    if production_registry().exists():
        raise ValueError("production_state_changed_do_not_probe")
    old_model = json.loads((FROZEN / "model_config_PENDING_AUTHORIZATION.json").read_text())
    model = corrected_production_model()
    changed_keys = [key for key in old_model if model[key] != old_model[key]]
    assert changed_keys == ["endpoint"]
    save(output, "model_config_CORRECTED_V1.json", model)
    save(output, "model_config_diff.json", {"changed_keys": changed_keys,
        "before_endpoint": old_model["endpoint"], "after_endpoint": model["endpoint"],
        "old_frozen_config_sha256": digest(FROZEN / "model_config_PENDING_AUTHORIZATION.json")})
    historical = PROJECT / "restart-n5-runs/h4_round1_r03_expanded_single_plan_fullflow_debug_live_3/run_freeze_manifest.json"
    freeze = json.loads(historical.read_text())
    source = {"path": str(Path(freeze["config_path"]).resolve()), "sha256": freeze["config_sha256"]}
    env_before = os.environ.get(model["api_key_env"])
    with offline_network_guard():
        with credential_environment(model, source) as metadata:
            # Constructor verifies frozen config and local tokenizer. No request
            # method or workflow is invoked with the real project credential.
            transport = SingleShotTransport(model, json.loads((FROZEN / "budget_proposal_NOT_AUTHORIZED.json").read_text()))
            assert transport.last_response_envelope is None
            credential_check = dict(metadata)
        assert os.environ.get(model["api_key_env"]) == env_before
        bundle = json.loads((FROZEN / "bundle_PENDING_AUTHORIZATION.json").read_text())
        feedback = validate_bundle_feedback(bundle)
        messages = json.loads((FROZEN / "final_supplement_messages_PENDING_AUTHORIZATION.json").read_text())
        tokens = local_counter().count_tokens(messages)
        for case in ("normal", "400", "timeout"):
            exercise_http_cli(output / "formal_url_mock_cases" / case, case, model_config=model, use_credential_file=True)
        modules = [__package__ + "." + name for name in (
            "test_transport_config", "test_readiness", "test_http_evidence", "test_policy", "test_bundle")]
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    save(output, "credential_source_identity.json", {**source, "source_manifest_path": str(historical),
        "source_manifest_sha256": digest(historical), "credential_value_in_artifacts": False})
    save(output, "credential_handoff_check.json", {**credential_check,
        "source_provider_model_and_sha_match": True, "environment_restored": os.environ.get(model["api_key_env"]) == env_before,
        "real_credential_used_for_request": False, "transport_constructor_only": True,
        "codex_auth_files_accessed": False})
    save(output, "tests.log", stream.getvalue())
    save(output, "feedback_unchanged_check.json", feedback)
    changes = []
    for name in ("run.py", "readiness.py", "test_readiness.py", "transport_config.py", "test_transport_config.py", "connection_preflight.py"):
        old = output / "implementation_before" / name
        old_text = old.read_text() if old.exists() else ""
        current = PACKAGE / name
        changes.append({"path": str(current), "kind": "modified" if old.exists() else "new", "sha256": digest(current)})
        save(output, "diffs/" + name + ".diff", "".join(difflib.unified_diff(old_text.splitlines(True), current.read_text().splitlines(True),
            fromfile="before/" + name, tofile="after/" + name)))
        compile(current.read_text(), str(current), "exec")
    save(output, "changed_files.json", changes)
    after = {path: digest(path) for path in before}
    changed_protected = [path for path in before if before[path] != after[path]]
    save(output, "protected_after_sha256.json", after)
    save(output, "implementation_sha256.json", implementation_manifest())
    diff = subprocess.run(["git", "diff", "--check"], cwd=PROJECT / "current_method_github", capture_output=True, text=True)
    save(output, "git_diff_check.txt", diff.stdout + diff.stderr)
    ready = result.wasSuccessful() and not changed_protected and diff.returncode == 0 and not production_registry().exists()
    report = {"version": VERSION, "status": "CONNECTION_FIX_OFFLINE_READY" if ready else "CONNECTION_FIX_WITH_GAPS",
        "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "protected_files": len(before), "protected_changes": changed_protected, "current_model_config_changes": changed_keys,
        "endpoint": model["endpoint"], "credential_handoff_offline_verified": True,
        "unchanged_frozen_messages_tokens": tokens, "prompt_or_policy_changed": False,
        "budget_previously_authorized": True, "budget_changed": False, "real_execution_performed_this_task": False,
        "real_external_calls": 0, "production_registry_created": False, "incremental_opportunity_consumed": False,
        "credentials_written_to_artifacts": False, "NOT_TESTED": ["real endpoint response", "provider 8192/128000 acceptance"],
        "next_step": "Use corrected model config and current implementation SHA when preparing a real run; no additional budget requested"}
    save(output, "final_report.json", report)
    command = "PYTHONDONTWRITEBYTECODE=1 python -m " + __package__ + ".run --execute \\\n  --bundle " + str(FROZEN / "bundle_PENDING_AUTHORIZATION.json") + " \\\n  --budget " + str(FROZEN / "budget_proposal_NOT_AUTHORIZED.json") + " \\\n  --model-config " + str(output / "model_config_CORRECTED_V1.json") + " \\\n  --credential-config " + source["path"] + " \\\n  --authorization /ABS/AUTHORIZATION_WITH_CURRENT_SHA_AND_CREDENTIAL_SOURCE.json \\\n  --run-dir /ABS/NEW_REAL_RUN_DIRECTORY\n"
    save(output, "UNEXECUTED_COMMAND.md", "# 未执行\n\n本任务只修复和离线验证；此前17次预算不变。正式授权文件须绑定本目录当前实现SHA、新model config SHA，"
         "并包含credential_source_identity.json中的path/sha256作为credential_source。不能套用旧实现身份。\n\n```bash\n" + command + "```\n")
    save(output, "final_report.md", "# Endpoint与凭据接入最小修复\n\n" +
        f"状态：`{report['status']}`。本轮测试{result.testsRun}项，失败{len(result.failures)}，错误{len(result.errors)}；保护文件{len(before)}项、变化{len(changed_protected)}。\n\n" +
        "新配置明确使用完整`https://api.siliconflow.cn/v1/chat/completions`。配置构造只补一次路径，运行时拒绝旧base地址，不静默修写冻结配置。"
        "新增正式配置经CLI到Request.full_url的断言，发送端全部mock。\n\n" +
        "显式--credential-config通过授权内固定path/SHA读取原项目凭据，核对model/provider；仅进程内临时设置环境变量，结束或异常后恢复。"
        "真实来源SHA/模型/地址验证及transport构造通过，没有发送；密钥不复制、不日志、不进入产物。未访问Codex auth文件。\n\n" +
        f"共享反馈、原方案、143条历史、模板/policy和已批准预算不变，原冻结messages仍为{tokens} tokens。旧离线交付及上次阻塞报告全部保留。\n\n" +
        "只有3个必要既有文件修改、3个新模块/测试/交付脚本，详见changed_files和diffs。正式调用需使用新配置及新实现身份；未自动创建运行授权或生产账本。\n\n"
        "真实模型、Gate/adherence、QA、Solver、training、Memory调用全部0，唯一补充机会未消费。服务端响应和8192/128000接受情况NOT_TESTED。\n\n"
        "本轮修复完成后停止，未自动恢复真实生成。\n")
    save(output, "artifact_sha256.json", {str(path.relative_to(output)): digest(path) for path in output.rglob("*") if path.is_file()})
    print(json.dumps({"status": report["status"], "tests": result.testsRun, "real_external_calls": 0}))
    return 0 if ready else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if args.capture:
        capture(args.output)
    else:
        raise SystemExit(run(args.output))
