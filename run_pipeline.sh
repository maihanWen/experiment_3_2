#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pipeline.sh /path/to/toy_transformer_pkg
#
# Assumptions:
#   - cfg_torch_paths.py, instrument.py, join_profile.py are in the current directory

SRC="${1:-}"
if [[ -z "${SRC}" ]]; then
  echo "Usage: bash run_pipeline.sh /path/to/toy_transformer_pkg"
  exit 2
fi

SRC="$(cd "${SRC}" && pwd)"
SRC_NAME="$(basename "${SRC}")"
CWD="$(pwd)"

ANALYZER="${CWD}/cfg_torch_paths.py"
INSTRUMENTER="${CWD}/instrument.py"
JOINER="${CWD}/join_profile.py"

PLAN_JSON="${CWD}/torch_boundaries.json"
ANALYSIS_TXT="${CWD}/cfg_torch_paths.txt"
DST_DIR="${CWD}/instrumented_${SRC_NAME}"

TRACE_JSON="${DST_DIR}/trace.json"
TB_REPORT="${CWD}/tb_report.txt"

# ---------
# 1) Static analysis -> plan JSON
# ---------
echo "==> [1/4] Static analysis"
python "${ANALYZER}" "${SRC}" \
  --out "${ANALYSIS_TXT}" \
  --export-boundaries-json "${PLAN_JSON}"

# ---------
# 2) Instrument -> instrumented copy in current directory
# ---------
echo "==> [2/4] Instrument source"
if [[ -d "${DST_DIR}" ]]; then
  echo "Destination exists: ${DST_DIR}"
  echo "Removing it to re-generate..."
  rm -rf "${DST_DIR}"
fi

python "${INSTRUMENTER}" \
  --plan "${PLAN_JSON}" \
  --src "${SRC}" \
  --dst "${DST_DIR}"

# ---------
# 3) Run instrumented code under profiler -> ${DST_DIR}/trace.json
# ---------
echo "==> [3/4] Run instrumented profiler runner"
python "${DST_DIR}/run_profile.py"

if [[ ! -f "${TRACE_JSON}" ]]; then
  echo "Expected trace not found: ${TRACE_JSON}"
  exit 1
fi

# ---------
# 4) Join static+runtime -> report
# ---------
echo "==> [4/4] Join trace with plan -> tb_report.txt"
python "${JOINER}" --plan "${PLAN_JSON}" --trace "${TRACE_JSON}" > "${TB_REPORT}"

echo ""
echo "Done ✅"
echo "  Plan:        ${PLAN_JSON}"
echo "  Analysis:    ${ANALYSIS_TXT}"
echo "  Instrumented:${DST_DIR}"
echo "  Trace:       ${TRACE_JSON}"
echo "  Report:      ${TB_REPORT}"