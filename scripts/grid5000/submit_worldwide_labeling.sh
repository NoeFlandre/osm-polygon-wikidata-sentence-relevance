#!/usr/bin/env bash
# Frontend submission adapter for one V2 worldwide label allocation.

set -euo pipefail
umask 077
if [ "$#" -ne 22 ]; then echo "submit_worldwide_labeling: exactly twenty-two arguments are required" >&2; exit 2; fi
REPO_ROOT="$1"; readonly REPO_ROOT
for path in "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"; do
    case "${path}" in /*) ;; *) echo "submit_worldwide_labeling: paths must be absolute" >&2; exit 2;; esac
done
for revision in "$9" "${10}" "${11}"; do
    [[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || { echo "submit_worldwide_labeling: revisions must be immutable" >&2; exit 2; }
done
[[ "${12}" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] || { echo "submit_worldwide_labeling: invalid dataset ID" >&2; exit 2; }
[[ "${13}" =~ ^[1-9][0-9]*$ && "${14}" =~ ^(0|[1-9][0-9]*)$ && "${15}" =~ ^(1|2|4|8|16|32)$ ]] || { echo "submit_worldwide_labeling: invalid runtime arguments" >&2; exit 2; }
WRAPPER="${REPO_ROOT}/scripts/grid5000/run_worldwide_labeling_job.sh"; HELPER="${REPO_ROOT}/scripts/grid5000/_submit_gpu_job.sh"
[ -x "${WRAPPER}" ] && [ -x "${HELPER}" ] || { echo "submit_worldwide_labeling: launcher is missing" >&2; exit 1; }
case "${21}" in smoke|production) ;; *) echo "submit_worldwide_labeling: label lane is invalid" >&2; exit 2;; esac
[[ "${22}" =~ ^[0-9a-f]{40}$ ]] || { echo "submit_worldwide_labeling: checkout revision is invalid" >&2; exit 2; }
shell_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
command_string="exec $(shell_quote "${WRAPPER}")"
for value in "$@"; do command_string="${command_string} $(shell_quote "${value}")"; done
read -r weekday hour < <(TZ=Europe/Paris date '+%u %H')
policy_type=night
if [ "${weekday}" -le 5 ] && [ "$((10#${hour}))" -ge 9 ] && [ "$((10#${hour}))" -lt 19 ]; then policy_type=day; fi
exec "${HELPER}" "40000" "00:55:00" "${policy_type}" "${command_string}"
