#!/usr/bin/env bash
# Graceful pre-walltime checkpointing for OAR Afghanistan labeling.
#
# A 12-hour OAR allocation gives the labeling CLI at most 11 hours 40
# minutes before the wrapper sends SIGINT. The CLI has up to 10 additional
# minutes to checkpoint before the wrapper escalates to SIGKILL. The
# ``interrupted=true`` branch in the CLI is expected to exit 0 and
# write a resumable checkpoint, so the helper must then propagate 0.
#
# Public surface:
#   deadline_helper_run <duration> <grace> <child> [args...]
#   <duration>  -- internal deadline (e.g. ``11h40m``, ``700s``)
#   <grace>     -- SIGINT→SIGKILL grace window (e.g. ``10m``)
#   <child>     -- executable to invoke
#   [args...]   -- forwarded to the child
#
# Implementation detail: GNU ``timeout --foreground --preserve-status
# --signal=INT --kill-after=<grace> <duration> <child> [args...]``.
# ``--preserve-status`` makes the wrapper return the child's exit status
# when the child exits on its own within the duration; ``--kill-after``
# escalates to SIGKILL only if SIGINT is ignored for the grace window.
#
# The helper exposes the duration parser for unit tests and refuses
# zero/negative durations before any subprocess is launched so a
# misconfiguration fails closed with a clear error.

set -euo pipefail

deadline_helper_error() {
    printf '%s\n' "$1" >&2
}


# Parse one human-readable duration into seconds. Supports the GNU
# ``timeout`` grammar: an optional integer followed by an optional unit
# letter (``s``, ``m``, ``h``). Compound units such as ``1h30m`` are not
# supported; callers should pick the largest single unit. Whitespace is
# not allowed. Zero or negative durations raise an error.
_deadline_helper_parse_duration() {
    local raw="$1"
    if [ -z "${raw}" ]; then
        printf '%s\n' "duration must not be empty" >&2
        return 1
    fi
    local unit=""
    local value="${raw}"
    case "${raw}" in
        *s|*m|*h)
            unit="${raw: -1}"
            value="${raw%?}"
            ;;
    esac
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "duration must be a non-negative integer: ${raw}" >&2
        return 1
    fi
    local seconds=0
    case "${unit}" in
        s|"") seconds="${value}" ;;
        m) seconds=$((value * 60)) ;;
        h) seconds=$((value * 3600)) ;;
    esac
    if [ "${seconds}" -le 0 ]; then
        printf '%s\n' "duration must be positive: ${raw}" >&2
        return 1
    fi
    printf '%d\n' "${seconds}"
}


deadline_helper_run() {
    if [ "$#" -lt 3 ]; then
        deadline_helper_error "deadline_helper_run requires at least duration, grace, and child"
        return 2
    fi
    local duration_raw="$1"
    local grace_raw="$2"
    shift 2

    local duration_seconds grace_seconds
    if ! duration_seconds="$(_deadline_helper_parse_duration "${duration_raw}")"; then
        return 2
    fi
    if ! grace_seconds="$(_deadline_helper_parse_duration "${grace_raw}")"; then
        return 2
    fi

    if ! command -v timeout >/dev/null 2>&1 && [ -z "${TIMEOUT_BIN:-}" ]; then
        deadline_helper_error "deadline_helper requires GNU timeout on PATH or TIMEOUT_BIN"
        return 1
    fi

    local timeout_bin="${TIMEOUT_BIN:-timeout}"
    if [ "${timeout_bin}" != "timeout" ] && [ ! -x "${timeout_bin}" ]; then
        deadline_helper_error "TIMEOUT_BIN is not executable: ${timeout_bin}"
        return 1
    fi

    exec ${timeout_bin} --foreground --preserve-status \
        --signal=INT --kill-after="${grace_seconds}" \
        "${duration_seconds}" "$@"
}
