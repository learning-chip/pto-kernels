#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cannsim}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
# shellcheck source=/dev/null
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
cd "${SCRIPT_DIR}"

usage() {
  echo "usage: $0 {cannsim|cannsim-report} [--gen-report] [--output DIR] [--kernel c2v|v2c|both] [--num-iters N]" >&2
  exit 2
}

case "${MODE}" in
  cannsim|cannsim-report)
    export PTO_SIMULATOR=1
    SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
    export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
    GEN_REPORT=""
    OUT_DIR="${SCRIPT_DIR}/outputs/cannsim"
    USER_OPTS="--sim"

    while [[ $# -gt 0 ]]; do
      case "$1" in
        --gen-report)
          GEN_REPORT="--gen-report"
          shift
          ;;
        --output)
          OUT_DIR="$2"
          shift 2
          ;;
        --kernel)
          USER_OPTS="${USER_OPTS} --kernel $2"
          shift 2
          ;;
        --num-iters)
          USER_OPTS="${USER_OPTS} --num-iters $2"
          shift 2
          ;;
        --block-dim)
          USER_OPTS="${USER_OPTS} --block-dim $2"
          shift 2
          ;;
        *)
          USER_OPTS="${USER_OPTS} $1"
          shift
          ;;
      esac
    done

    mkdir -p "${OUT_DIR}"
    run_log="${OUT_DIR}/run_stream.stdout"
    set +e
    cannsim record -s "${CANNSIM_SOC:-Ascend950}" \
      ${GEN_REPORT} \
      -o "${OUT_DIR}" \
      "${SCRIPT_DIR}/run_sim_entry.sh" \
      -u "${USER_OPTS}" 2>&1 | tee "${run_log}"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ "${rc}" -ne 0 ]]; then
      if grep -q "SIM_SUMMARY" "${run_log}" 2>/dev/null; then
        echo "cannsim exited ${rc} after SIM_SUMMARY; treating as success"
        exit 0
      fi
    fi
    exit "${rc}"
    ;;
  *)
    usage
    ;;
esac
