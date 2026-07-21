#!/usr/bin/env bash
# Read-only join-integrity audit runner (Phase 9M-C).
# Submits an OAR besteffort batch job (single core, 30 min, no GPU,
# no inference, no model loading) that runs the audit script and
# writes a structured JSON report to a persistent path.

set -euo pipefail

PERSISTENT_ROOT="${PERSISTENT_ROOT:-/home/nflandre/g5k-smoke}"
INPUT_ROOT="${INPUT_ROOT:-${PERSISTENT_ROOT}/input}"
REPORT_DIR="${REPORT_DIR:-${PERSISTENT_ROOT}/bench_audit}"
OAR_WALLTIME="${OAR_WALLTIME:-00:30:00}"
JOB_NAME="${JOB_NAME:-audit-9mc}"

REPORT_DIR="${REPORT_DIR}"
mkdir -p "${REPORT_DIR}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPORT_DIR}/join_audit_${TS}.json"
LOG="${REPORT_DIR}/join_audit_${TS}.log"

# Resolve repo root for the compute node: it must be on the same
# filesystem as INPUT_ROOT for parquet reads to be cheap. We submit
# from ${PERSISTENT_ROOT}/repo and pass its absolute path as a
# resource.
REPO_ROOT="${PERSISTENT_ROOT}/repo"

cat > "${REPORT_DIR}/run_audit_oar.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="\${PYTHONPATH:-}:${REPO_ROOT}/scripts"
exec .venv/bin/python scripts/audit/join_integrity_audit.py \\
  --input-root "${INPUT_ROOT}" \\
  --report "${REPORT}" \\
  --target-shard italy-latest
EOF
chmod 0700 "${REPORT_DIR}/run_audit_oar.sh"

oarsub \
  -l host=1/core=1,walltime="${OAR_WALLTIME}" \
  -t besteffort \
  -n "${JOB_NAME}" \
  "${REPORT_DIR}/run_audit_oar.sh" \
  > "${LOG}" 2>&1

cat "${LOG}"
