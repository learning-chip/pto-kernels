#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_with_timeout.sh" python3 "${SCRIPT_DIR}/run_stream.py" "$@"
