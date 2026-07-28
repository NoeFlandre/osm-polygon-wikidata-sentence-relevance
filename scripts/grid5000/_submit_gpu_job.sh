#!/usr/bin/env bash
# Submit one CUDA payload using the live site's matching OAR resource type.

set -euo pipefail
umask 077

if [ "$#" -ne 4 ]; then
    echo "submit_gpu_job: exactly four arguments are required" >&2
    exit 2
fi

GPU_MIN_MEMORY_MB="$1"; readonly GPU_MIN_MEMORY_MB
WALLTIME="$2"; readonly WALLTIME
POLICY_TYPE="$3"; readonly POLICY_TYPE
PAYLOAD="$4"; readonly PAYLOAD

if ! [[ "${GPU_MIN_MEMORY_MB}" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "${WALLTIME}" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || \
   [ -z "${PAYLOAD}" ]; then
    echo "submit_gpu_job: invalid resource or payload argument" >&2
    exit 2
fi
case "${POLICY_TYPE}" in
    day|night) ;;
    *) echo "submit_gpu_job: policy type must be day or night" >&2; exit 2;;
esac
for required_command in jq oarnodes oarsub; do
    command -v "${required_command}" >/dev/null || {
        echo "submit_gpu_job: required scheduler command is unavailable" >&2
        exit 1
    }
done

inventory="$(oarnodes -J)"
eligible_filter='
  [.[] | select(
    .state == "Alive"
    and (.gpu_count // 0) > 0
    and (.gpu_mem // 0) >= $minimum
    and (.gpu_compute_capability_major // 0) >= 7
  )]'
gpu_resource_type=standard
if jq -e --argjson minimum "${GPU_MIN_MEMORY_MB}" \
    "${eligible_filter} | any((.exotic // \"NO\") != \"YES\")" \
    >/dev/null <<<"${inventory}"; then
    :
elif jq -e --argjson minimum "${GPU_MIN_MEMORY_MB}" \
    "${eligible_filter} | any(.exotic == \"YES\")" \
    >/dev/null <<<"${inventory}"; then
    gpu_resource_type=exotic
else
    echo "submit_gpu_job: no compatible live GPU resource" >&2
    exit 1
fi

if [ "${gpu_resource_type}" = exotic ]; then
    exec oarsub -q default -p "gpu_mem>=${GPU_MIN_MEMORY_MB}" \
        -t exotic -t "${POLICY_TYPE}" \
        -l "gpu=1,walltime=${WALLTIME}" "${PAYLOAD}"
else
    exec oarsub -q default -p "gpu_mem>=${GPU_MIN_MEMORY_MB}" \
        -t "${POLICY_TYPE}" \
        -l "gpu=1,walltime=${WALLTIME}" "${PAYLOAD}"
fi
