#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/handover_audit}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-5000}"
WINDOW_SEC="${WINDOW_SEC:-1 5 10}"

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/samples" "$OUTPUT_ROOT/figures"

python3 tools/audit_handover_dataset.py \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --sample-limit "$SAMPLE_LIMIT" \
  --window-sec $WINDOW_SEC

cat <<MSG

Audit complete.
Main report: $OUTPUT_ROOT/HANDOVER_FEASIBILITY_REPORT.md

Large CSV outputs are intentionally ignored by Git and can be regenerated with:
  scripts/run_handover_audit.sh
MSG
