#!/usr/bin/env bash
# Minimal GNU ``timeout`` shim used to exercise the deadline helper on
# hosts where the real binary is unavailable.
#
# Supports only the flags the helper uses:
# --foreground --preserve-status --signal=<NAME> --kill-after=<seconds>
# <duration> <child> [args...].
# The shim writes its exit code to ``$TIMEOUT_EXIT_FILE`` if set so the
# Python driver can read it back without a subshell.

set -u

exit_file="${TIMEOUT_EXIT_FILE:-/dev/null}"

duration=""
signal_name="INT"
kill_after=""
preserve_status=0
foreground=0
child=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --foreground)
            foreground=1
            shift
            ;;
        --preserve-status)
            preserve_status=1
            shift
            ;;
        --signal)
            signal_name="$2"
            shift 2
            ;;
        --signal=*)
            signal_name="${1#--signal=}"
            shift
            ;;
        --kill-after)
            kill_after="$2"
            shift 2
            ;;
        --kill-after=*)
            kill_after="${1#--kill-after=}"
            shift
            ;;
        -h|--help)
            echo "fake_timeout --foreground --preserve-status --signal=INT --kill-after=<s> <duration> <child> [args...]"
            exit 0
            ;;
        *)
            if [ -z "${duration}" ]; then
                duration="$1"
                shift
            else
                child=("$@")
                break
            fi
            ;;
    esac
done

if [ -z "${duration}" ] || [ "${#child[@]}" -eq 0 ]; then
    echo "fake_timeout: missing duration or child" >&2
    echo "124" > "${exit_file}"
    exit 124
fi

# Launch the child.
"${child[@]}" &
child_pid=$!

deadline_unix=$(( $(date +%s) + duration ))

while kill -0 "${child_pid}" 2>/dev/null; do
    now=$(date +%s)
    if [ "${now}" -ge "${deadline_unix}" ]; then
        break
    fi
    sleep 0.05
done

if kill -0 "${child_pid}" 2>/dev/null; then
    # Send the configured signal.
    sig_num=$(kill -l "SIG${signal_name}" 2>/dev/null || echo "")
    if [ -n "${sig_num}" ]; then
        kill -"${sig_num}" "${child_pid}" 2>/dev/null
    fi
    if [ -n "${kill_after}" ]; then
        kill_at=$(( $(date +%s) + kill_after ))
        while kill -0 "${child_pid}" 2>/dev/null; do
            now=$(date +%s)
            if [ "${now}" -ge "${kill_at}" ]; then
                break
            fi
            sleep 0.05
        done
        if kill -0 "${child_pid}" 2>/dev/null; then
            kill -9 "${child_pid}" 2>/dev/null
        fi
    fi
    # Poll until child exits instead of using ``wait`` which can hang on
    # macOS bash when the child has already exited and was not waited
    # for inline.
    while kill -0 "${child_pid}" 2>/dev/null; do
        sleep 0.05
    done
    rc=$?
    if [ "${preserve_status}" -eq 1 ]; then
        if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ] || [ "${rc}" -eq -2 ]; then
            rc=124
        fi
    else
        rc=124
    fi
    echo "${rc}" > "${exit_file}"
    exit "${rc}"
else
    rc=0
    echo "${rc}" > "${exit_file}"
    exit "${rc}"
fi
