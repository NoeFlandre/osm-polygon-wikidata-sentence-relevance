#!/usr/bin/env bash
# Strict clean-checkout guard for one OAR Afghanistan labeling allocation.
#
# This helper is sourced (not executed) by the scheduler-owned job wrapper.
# It exposes ``validate_clean_checkout`` which inspects the repository
# working tree for any tracked, staged, or untracked entry and rejects
# everything except the single explicitly approved ``.venv`` deployment
# entry.
#
# The ``.venv`` allowance is only valid when:
#   - the path's basename is exactly ``.venv``;
#   - the entry is owned by the current user;
#   - the entry is either a directory or a symlink (not a regular file);
#   - for a symlink, its physical resolution remains inside the
#     approved persistent run root and the file ownership matches;
#   - the resolved path contains ``.venv/bin/python`` and the installed
#     ``osm-polygon-label-sentences`` CLI binary.
#
# The function uses only path-free, public error messages so that no
# deployment-specific paths leak into the guarded OAR job log.
#
# Public surface:
#   validate_clean_checkout <repo_root> <approved_run_root> [current_user]
#   Returns 0 when the checkout is safe to run, non-zero otherwise.
#   Sets ``CHECKOUT_GUARD_ERROR`` to a single-line description on failure.
#
# The caller is responsible for sourcing this file before invocation.

set -euo pipefail

# Paths / names recognised by the guard. Using literal names keeps the
# guard immune to symlink traversal tricks.
_DEPLOYMENT_ENTRY_NAME=".venv"
_VENV_PYTHON_RELATIVE_PATH=".venv/bin/python"
_VENV_CLI_RELATIVE_PATH=".venv/bin/osm-polygon-label-sentences"


# Validate and synchronize the repository environment on the compute node.
# Reusing a compatible environment avoids downloading the same locked wheels
# during every short allocation.  A frontend-created or damaged environment
# is never trusted blindly: a failed synchronization removes it and performs
# one clean rebuild on the worker architecture.
prepare_compute_environment() {
    local repo_root="$1"
    local scratch_base="$2"
    local job_log_dir="$3"
    local label="${4:-compute}"
    local uv_bin="${UV_BIN:-}"

    if [ -z "${uv_bin}" ]; then
        uv_bin="$(command -v uv || true)"
    fi
    if [ -z "${uv_bin}" ]; then
        uv_bin="${HOME}/.local/bin/uv"
    fi
    if [ ! -x "${uv_bin}" ]; then
        echo "${label}: uv is missing on the compute node" >&2
        return 1
    fi
    if [ -L "${repo_root}/.venv" ]; then
        echo "${label}: .venv must not be a symlink" >&2
        return 1
    fi
    if [ -e "${scratch_base}/uv-cache" ] && [ -L "${scratch_base}/uv-cache" ]; then
        echo "${label}: uv cache must not be a symlink" >&2
        return 1
    fi
    mkdir -p -m 0700 -- "${scratch_base}/uv-cache"
    export UV_CACHE_DIR="${scratch_base}/uv-cache"

    : >"${job_log_dir}/environment.stdout.log"
    : >"${job_log_dir}/environment.stderr.log"
    if [ -d "${repo_root}/.venv" ]; then
        if "${uv_bin}" sync --locked --no-dev \
            --extra hub --extra segmentation --extra operator \
            --project "${repo_root}" \
            >>"${job_log_dir}/environment.stdout.log" \
            2>>"${job_log_dir}/environment.stderr.log" && \
            [ -x "${repo_root}/.venv/bin/python" ] && \
            "${repo_root}/.venv/bin/python" -c \
                'import h3, huggingface_hub, pyarrow, torch, typer, wtpsplit' \
                >>"${job_log_dir}/environment.stdout.log" \
                2>>"${job_log_dir}/environment.stderr.log"; then
            printf 'COMPUTE_ENVIRONMENT_REUSED\n' \
                >>"${job_log_dir}/environment.stdout.log"
            return 0
        fi
        printf 'Existing compute environment failed validation; rebuilding.\n' \
            >>"${job_log_dir}/environment.stderr.log"
    fi

    rm -rf -- "${repo_root}/.venv"
    if ! "${uv_bin}" sync --locked --no-dev \
        --extra hub --extra segmentation --extra operator \
        --project "${repo_root}" \
        >>"${job_log_dir}/environment.stdout.log" \
        2>>"${job_log_dir}/environment.stderr.log"; then
        echo "${label}: compute-node environment preparation failed" >&2
        return 1
    fi
    if [ ! -x "${repo_root}/.venv/bin/python" ] || \
        ! "${repo_root}/.venv/bin/python" -c \
            'import h3, huggingface_hub, pyarrow, torch, typer, wtpsplit' \
            >>"${job_log_dir}/environment.stdout.log" \
            2>>"${job_log_dir}/environment.stderr.log"; then
        echo "${label}: compute-node environment validation failed" >&2
        return 1
    fi
}


# Install a non-destructive failure marker for a managed run. The marker is
# intentionally written only below the operator root, allowing the Mac-side
# cleanup to reclaim failed roots without ever scanning unrelated home data.
mark_managed_run_failed() {
    local run_root="$1"
    local job_id="$2"
    local return_code="$3"
    if [ "${return_code}" -eq 0 ]; then
        return 0
    fi
    case "${run_root}" in
        "${HOME}/osm-polygon-operator/"*) ;;
        *) return 0 ;;
    esac
    local marker="${run_root}/.operator-managed.json"
    if [ ! -f "${marker}" ] || [ -L "${marker}" ]; then
        return 0
    fi
    local marker_tmp="${marker}.tmp.${job_id}"
    if printf '%s\n' '{"schema_version":1,"status":"failed"}' >"${marker_tmp}"; then
        chmod 0600 "${marker_tmp}" 2>/dev/null || true
        mv -f -- "${marker_tmp}" "${marker}" 2>/dev/null || true
    fi
}


checkout_guard_error() {
    local message="$1"
    CHECKOUT_GUARD_ERROR="${message}"
    printf '%s\n' "${message}" >&2
}


# Resolve ``$path`` to a physical, canonical absolute path that follows
# every symlink in the chain. Uses ``readlink -f`` when available; falls
# back to a portable loop that walks the path component-by-component.
checkout_guard_physical_path() {
    local path="$1"
    if [ ! -e "${path}" ]; then
        return 1
    fi
    if command -v readlink >/dev/null 2>&1; then
        local resolved
        if resolved="$(readlink -f "${path}" 2>/dev/null)"; then
            printf '%s\n' "${resolved}"
            return 0
        fi
    fi
    local target="${path}"
    while [ -L "${target}" ]; do
        local link
        link="$(readlink "${target}")"
        case "${link}" in
            /*) target="${link}" ;;
            *) target="$(dirname "${target}")/${link}" ;;
        esac
    done
    local parent
    parent="$(cd "$(dirname "${target}")" && pwd -P)"
    printf '%s/%s\n' "${parent}" "$(basename "${target}")"
}


# Is ``$target`` physically inside ``$root``? Empty or self-rooted roots
# are always considered to contain their target.
checkout_guard_is_inside() {
    local target="$1"
    local root="$2"
    if [ -z "${root}" ]; then
        return 0
    fi
    case "${target}" in
        "${root}/"*) return 0 ;;
        "${root}") return 0 ;;
        *) return 1 ;;
    esac
}


# Validate one ``.venv`` deployment entry. Returns 0 when the entry is
# trustworthy, non-zero otherwise. ``$1`` is the absolute ``.venv`` path
# inside the repository; ``$2`` is the approved persistent run root
# (which may be the run root or empty for tests); ``$3`` is the current
# user name.
validate_venv_entry() {
    local venv_path="$1"
    local approved_run_root="$2"
    local current_user="$3"

    if [ "$(basename "${venv_path}")" != "${_DEPLOYMENT_ENTRY_NAME}" ]; then
        checkout_guard_error ".venv entry has an unexpected basename"
        return 1
    fi

    if [ ! -e "${venv_path}" ]; then
        checkout_guard_error ".venv entry does not exist"
        return 1
    fi

    local entry_owner
    entry_owner="$(stat -c '%U' "${venv_path}" 2>/dev/null || stat -f '%Su' "${venv_path}")"
    if [ "${entry_owner}" != "${current_user}" ]; then
        checkout_guard_error ".venv entry is not owned by the current user"
        return 1
    fi

    if [ -L "${venv_path}" ]; then
        local resolved
        if ! resolved="$(checkout_guard_physical_path "${venv_path}")"; then
            checkout_guard_error ".venv symlink target cannot be resolved"
            return 1
        fi
        if [ ! -d "${resolved}" ]; then
            checkout_guard_error ".venv symlink target is not a directory"
            return 1
        fi
        if ! checkout_guard_is_inside "${resolved}" "${approved_run_root}"; then
            checkout_guard_error ".venv symlink target escapes the approved run root"
            return 1
        fi
        local resolved_owner
        resolved_owner="$(stat -c '%U' "${resolved}" 2>/dev/null || stat -f '%Su' "${resolved}")"
        if [ "${resolved_owner}" != "${current_user}" ]; then
            checkout_guard_error ".venv symlink target is not owned by the current user"
            return 1
        fi
        venv_path="${resolved}"
    elif [ ! -d "${venv_path}" ]; then
        checkout_guard_error ".venv entry is neither a directory nor a symlink"
        return 1
    fi

    if [ ! -x "${venv_path}/bin/python" ]; then
        checkout_guard_error ".venv/bin/python is missing or not executable"
        return 1
    fi
    if [ ! -x "${venv_path}/bin/osm-polygon-label-sentences" ]; then
        checkout_guard_error ".venv installed CLI is missing"
        return 1
    fi

    return 0
}


# Determine whether ``$1`` (a path printed by ``git status --porcelain``)
# is an untracked ``.venv``-only entry.
checkout_guard_is_untracked_venv() {
    local status_code="$1"
    local path="$2"
    case "${status_code}" in
        "??")
            case "${path}" in
                "${_DEPLOYMENT_ENTRY_NAME}"|"${_DEPLOYMENT_ENTRY_NAME}"/*)
                    return 0
                    ;;
            esac
            ;;
        "!!")
            case "${path}" in
                "${_DEPLOYMENT_ENTRY_NAME}"|"${_DEPLOYMENT_ENTRY_NAME}"/*)
                    # The ``!!`` marker is git's "ignored" indicator; the
                    # Grid'5000 venv lives inside ``.venv/.gitignore`` via
                    # the standard ``.venv`` pattern.
                    return 0
                    ;;
            esac
            ;;
    esac
    return 1
}


# Public entry point. ``$1`` is the repository root; ``$2`` is the
# approved persistent run root used to validate ``.venv`` symlinks.
validate_clean_checkout() {
    local repo_root="$1"
    local approved_run_root="$2"
    local current_user="${3:-$(id -un)}"

    if [ -z "${repo_root}" ] || [ ! -d "${repo_root}" ]; then
        checkout_guard_error "repository root is unavailable"
        return 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        checkout_guard_error "git is unavailable on the host"
        return 1
    fi

    local dirty_output
    if ! dirty_output="$(git -C "${repo_root}" status --porcelain --untracked-files=normal --ignored=traditional 2>&1)"; then
        checkout_guard_error "git status failed: ${dirty_output}"
        return 1
    fi

    if [ -z "${dirty_output}" ]; then
        return 0
    fi

    # Resolve the approved run root to a physical path so the
    # ``is_inside`` check survives transient ``/tmp`` -> ``/private/tmp``
    # style mountpoint links on shared hosts.
    local physical_run_root=""
    if [ -n "${approved_run_root}" ]; then
        if [ -d "${approved_run_root}" ]; then
            physical_run_root="$(cd "${approved_run_root}" && pwd -P)"
        else
            checkout_guard_error "approved run root does not exist"
            return 1
        fi
    fi

    local saw_venv=0
    local line
    while IFS= read -r line || [ -n "${line}" ]; do
        if [ -z "${line}" ]; then
            continue
        fi
        local status_code path
        status_code="${line:0:2}"
        path="${line:3}"
        # ``git status --porcelain`` prints ``.venv/`` for an ignored or
        # untracked directory; strip the trailing slash so the basename
        # matches the literal deployment entry name.
        path="${path%/}"

        if checkout_guard_is_untracked_venv "${status_code}" "${path}"; then
            if [ "${saw_venv}" -ne 0 ]; then
                checkout_guard_error ".venv entry appears more than once"
                return 1
            fi
            saw_venv=1
            if ! validate_venv_entry "${repo_root}/${path}" "${physical_run_root}" "${current_user}"; then
                return 1
            fi
            continue
        fi

        if checkout_guard_is_allowed_ignored_entry \
            "${status_code}" "${path}" "${repo_root}" "${current_user}"; then
            continue
        fi

        case "${status_code}" in
            "??")
                checkout_guard_error "untracked entry is not allowed"
                return 1
                ;;
            "!!")
                checkout_guard_error "ignored entry is not allowed"
                return 1
                ;;
            *)
                checkout_guard_error "checkout is dirty"
                return 1
                ;;
        esac
    done <<< "${dirty_output}"

    return 0
}
# Some scheduler output files are deliberately ignored under Grid'5000
# launching conventions. These are safe to ignore because they are rewritten
# on each submission and are explicitly filtered by .gitignore to avoid
# accidental leaks.
checkout_guard_is_allowed_ignored_entry() {
    local status_code="$1"
    local path="$2"
    local repo_root="${3:-}"
    local current_user="${4:-$(id -un)}"

    if [ "${status_code}" != "!!" ]; then
        return 1
    fi

    case "${path}" in
        OAR.*.stderr|OAR.*.stdout)
            return 0
            ;;
        __pycache__|*/__pycache__)
            # Python creates these ignored directories during every job.
            # They are safe only when they are a regular, user-owned
            # directory directly below this checkout; do not admit a
            # symlink or any path that could escape the repository.
            case "${path}" in
                /*|../*|*/../*|*/..)
                    return 1
                    ;;
            esac
            local cache_path="${repo_root}/${path}"
            if [ ! -d "${cache_path}" ] || [ -L "${cache_path}" ]; then
                return 1
            fi
            local cache_owner
            cache_owner="$(stat -c '%U' "${cache_path}" 2>/dev/null || stat -f '%Su' "${cache_path}")"
            [ "${cache_owner}" = "${current_user}" ]
            return
            ;;
    esac
    return 1
}
