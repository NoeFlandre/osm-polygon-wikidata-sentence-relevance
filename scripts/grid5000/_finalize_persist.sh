#!/usr/bin/env bash
# Persistent-storage copy helper for bounded one-shard OAR finalization.
#
# When the OAR scheduler terminates a compute allocation it removes the
# allocation-local scratch directory.  The finalization payload writes
# ``sentences.parquet``, ``manifest.json`` and ``README.md`` into
# ``${WORK_DIR}/out`` on the compute node; this helper copies those
# three artifacts to ``${LOG_ROOT}/${OAR_JOB_ID}/output`` on the
# operator's persistent NFS mount so they survive the cleanup.
#
# Contract (public)
# -----------------
# ``finalize_persist_artifacts SCRATCH_OUT_DIR LOG_ROOT OAR_JOB_ID``:
#   * Requires SCRATCH_OUT_DIR to be a real directory containing
#     exactly the three required artifacts (regular files, no
#     symlinks, no extra entries).
#   * Creates a fresh mode-0700 staging directory under
#     ``${LOG_ROOT}/${OAR_JOB_ID}.persist.XXXXXX``.
#   * Copies each artifact (preserving bytes, dropping the source
#     mode), sets mode 0600 on the copies, and verifies the staging
#     directory contains exactly three regular files.
#   * Atomically renames the staging directory to
#     ``${LOG_ROOT}/${OAR_JOB_ID}/output``.
#   * Refuses overwrite/reuse when the target already exists.
#   * Returns 0 on success and prints the persistent path on stdout.
#   * Returns non-zero on any failure with a diagnostic on stderr.
#
# The helper is sourced by ``run_streaming_finalization_job.sh`` and
# tested directly by ``tests/unit/scripts/test_finalize_persist_sh.py``.

set -euo pipefail
umask 077

_REQUIRED_ARTIFACTS=(sentences.parquet manifest.json README.md)

finalize_persist_artifacts() {
    local scratch_out_dir="$1"
    local log_root="$2"
    local oar_job_id="$3"

    if [ -z "${scratch_out_dir}" ] || [ -z "${log_root}" ] || [ -z "${oar_job_id}" ]; then
        echo "finalize_persist: scratch_out_dir, log_root, and oar_job_id are required" >&2
        return 2
    fi
    if [[ ! "${scratch_out_dir}" =~ ^/ ]] || [[ ! "${log_root}" =~ ^/ ]]; then
        echo "finalize_persist: scratch_out_dir and log_root must be absolute" >&2
        return 2
    fi
    if [ ! -d "${scratch_out_dir}" ] || [ -L "${scratch_out_dir}" ]; then
        echo "finalize_persist: scratch_out_dir must be a real directory: ${scratch_out_dir}" >&2
        return 2
    fi

    local target="${log_root}/${oar_job_id}/output"
    if [ -e "${target}" ] || [ -L "${target}" ]; then
        echo "finalize_persist: refusing to overwrite existing target: ${target}" >&2
        return 1
    fi

    for required in "${_REQUIRED_ARTIFACTS[@]}"; do
        if [ ! -f "${scratch_out_dir}/${required}" ]; then
            echo "finalize_persist: missing required artifact: ${required}" >&2
            return 1
        fi
        if [ -L "${scratch_out_dir}/${required}" ]; then
            echo "finalize_persist: refused symlink artifact: ${required}" >&2
            return 1
        fi
    done

    local scratch_file_count
    scratch_file_count=$(find "${scratch_out_dir}" -mindepth 1 -maxdepth 1 -type f -o -type l -o -type d | wc -l | tr -d '[:space:]')
    if [ "${scratch_file_count}" != "3" ]; then
        echo "finalize_persist: scratch has ${scratch_file_count} entries, expected exactly 3" >&2
        return 1
    fi

    local staging
    staging="$(mktemp -d "${log_root}/${oar_job_id}.persist.XXXXXX")"
    chmod 0700 "${staging}"

    for required in "${_REQUIRED_ARTIFACTS[@]}"; do
        install -m 0600 /dev/null "${staging}/${required}"
        cat "${scratch_out_dir}/${required}" > "${staging}/${required}"
    done

    local file_count
    file_count=$(find "${staging}" -mindepth 1 -maxdepth 1 -type f | wc -l | tr -d '[:space:]')
    if [ "${file_count}" != "3" ]; then
        rm -rf "${staging}"
        echo "finalize_persist: staging has ${file_count} regular files, expected 3" >&2
        return 1
    fi

    local link_count
    link_count=$(find "${staging}" -mindepth 1 -maxdepth 1 -type l | wc -l | tr -d '[:space:]')
    if [ "${link_count}" != "0" ]; then
        rm -rf "${staging}"
        echo "finalize_persist: staging has ${link_count} symlinks, expected 0" >&2
        return 1
    fi

    mkdir -m 0700 -p "${log_root}/${oar_job_id}"
    if ! mv "${staging}" "${target}"; then
        rm -rf "${staging}"
        echo "finalize_persist: atomic rename failed for ${target}" >&2
        return 1
    fi

    printf '%s\n' "${target}"
    return 0
}
