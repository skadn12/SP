#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_batch_v11.sh BATCH_ROOT ROOT_ID ROOT_ID [ROOT_ID ...]

The batch root must not already exist. At least two distinct Conversation
Chronicles root IDs are required. The script always runs the zero-API hard
preflight first and starts the live batch only when it passes.

Optional environment variables:
  API_CONFIG       Challenger API configuration path
  TOKENIZER_PATH   Local tokenizer path
  SOLVER_MODEL     Local SPELL Solver model path
  PYTHON_BIN       Python executable (default: python)
  PREFLIGHT_ONLY   Set to 1 to stop after zero-API preflight

Example:
  PREFLIGHT_ONLY=1 ./scripts/current_method/restart_n5_spell/world_memory_slim_v1/run_batch_v11.sh \
    /root/autodl-tmp/gxy-selfplay/restart-n5-runs/my_slim_v11_batch \
    cc-train-episode-61555 cc-train-episode-216742
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../../../.." && pwd)"

batch_root="$1"
shift
root_ids=("$@")

python_bin="${PYTHON_BIN:-python}"
api_config="${API_CONFIG:-/root/autodl-tmp/gxy-selfplay/config.json}"
tokenizer_path="${TOKENIZER_PATH:-/root/autodl-tmp/Qwen3-4B}"
solver_model="${SOLVER_MODEL:-/root/autodl-tmp/gxy-selfplay/base_restart_current_method_validation/accepted_models/solver_base_restart_stage2_merged}"

if [[ -e "${batch_root}" ]]; then
  echo "ERROR: batch root already exists; refusing overwrite or resume: ${batch_root}" >&2
  exit 2
fi

declare -A seen_roots=()
for root_id in "${root_ids[@]}"; do
  if [[ -n "${seen_roots[${root_id}]+present}" ]]; then
    echo "ERROR: duplicate root ID: ${root_id}" >&2
    exit 2
  fi
  seen_roots["${root_id}"]=1
done

module="scripts.current_method.restart_n5_spell.world_memory_slim_v1.production_batch_v11"
args=(
  --batch-root "${batch_root}"
  --api-config "${api_config}"
  --tokenizer "${tokenizer_path}"
  --solver-model "${solver_model}"
)
for root_id in "${root_ids[@]}"; do
  args+=(--root-id "${root_id}")
done

cd "${project_root}"

echo "[1/2] Running zero-API batch preflight"
"${python_bin}" -m "${module}" "${args[@]}" --preflight-only --no-api

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed. PREFLIGHT_ONLY=1; no live roles were called."
  exit 0
fi

if [[ -e "${batch_root}" ]]; then
  echo "ERROR: preflight unexpectedly created the batch root: ${batch_root}" >&2
  exit 2
fi

echo "[2/2] Starting fresh live batch"
echo "Batch root: ${batch_root}"
echo "Roots: ${root_ids[*]}"
"${python_bin}" -m "${module}" "${args[@]}"

echo "Batch finished. Summary:"
echo "${batch_root}/reports/END_TO_END_BATCH_PILOT_SUMMARY.md"
