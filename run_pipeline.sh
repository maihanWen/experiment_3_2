#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pipeline.sh /path/to/code_folder /path/to/report_folder
#
# Example:
#   bash run_pipeline.sh ./toy_transformer_pkg ./reports/run1
#
# Assumptions:
#   - cfg_torch_paths.py, instrument.py, join_profile_timeline.py (or join_profile.py) are in the same dir as this script

SRC="${1:-}"
REPORT_DIR="${2:-}"

if [[ -z "${SRC}" || -z "${REPORT_DIR}" ]]; then
  echo "Usage: bash run_pipeline.sh /path/to/code_folder /path/to/report_folder"
  exit 2
fi

SRC="$(cd "${SRC}" && pwd)"
REPORT_DIR="$(mkdir -p "${REPORT_DIR}" && cd "${REPORT_DIR}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ANALYZER="${SCRIPT_DIR}/cfg_torch_paths.py"
INSTRUMENTER="${SCRIPT_DIR}/instrument.py"

# Choose ONE joiner:
# 1) Timeline joiner that prints + writes timeline.json:
JOINER="${SCRIPT_DIR}/join_profile_timeline.py"
# 2) Or your old ranking joiner:
# JOINER="${SCRIPT_DIR}/join_profile.py"

PLAN_JSON="${REPORT_DIR}/torch_boundaries.json"
ANALYSIS_TXT="${REPORT_DIR}/cfg_torch_paths.txt"

SRC_NAME="$(basename "${SRC}")"
DST_DIR="${REPORT_DIR}/instrumented_${SRC_NAME}"
TRACE_JSON="${DST_DIR}/trace.json"

TB_REPORT_TXT="${REPORT_DIR}/tb_report.txt"
TIMELINE_JSON="${REPORT_DIR}/timeline.json"

echo "========================================"
echo "Pipeline"
echo "  Source:  ${SRC}"
echo "  Report:  ${REPORT_DIR}"
echo "========================================"
echo ""

# 1) Static analysis -> plan JSON
echo "==> [1/4] Static analysis -> plan"
python "${ANALYZER}" "${SRC}" \
  --out "${ANALYSIS_TXT}" \
  --export-boundaries-json "${PLAN_JSON}"

# 2) Instrument -> instrumented copy in report folder
echo "==> [2/4] Instrument -> ${DST_DIR}"
rm -rf "${DST_DIR}"
python "${INSTRUMENTER}" \
  --plan "${PLAN_JSON}" \
  --src "${SRC}" \
  --dst "${DST_DIR}" \
  --quiet

# 3) Run instrumented code under profiler -> trace.json
echo "==> [3/4] Run profiler -> trace.json"
python "${DST_DIR}/run_profile.py"

if [[ ! -f "${TRACE_JSON}" ]]; then
  echo "ERROR: Expected trace not found: ${TRACE_JSON}"
  exit 1
fi

# 4) Join -> tb_report.txt and timeline.json (if using timeline joiner)
echo "==> [4/4] Join -> reports"

# Always capture a readable text report
python "${JOINER}" --trace "${TRACE_JSON}" > "${TB_REPORT_TXT}"

# If JOINER is the timeline joiner, also write timeline.json
# (It supports --out; if your joiner doesn't, you can remove this block.)
if python "${JOINER}" --help 2>/dev/null | grep -q -- "--out"; then
  python "${JOINER}" --trace "${TRACE_JSON}" --out "${TIMELINE_JSON}" >/dev/null
fi

echo ""
echo "Done ✅"
echo "Report folder contains:"
echo "  - ${PLAN_JSON}"
echo "  - ${ANALYSIS_TXT}"
echo "  - ${TB_REPORT_TXT}"
if [[ -f "${TIMELINE_JSON}" ]]; then
  echo "  - ${TIMELINE_JSON}"
fi
echo "  - ${TRACE_JSON}"