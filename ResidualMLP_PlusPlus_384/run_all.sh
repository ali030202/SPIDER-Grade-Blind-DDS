#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SPIDER_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
python "${SCRIPT_DIR}/residual_mlp_plusplus_dds.py" train --root "$ROOT" "$@"
