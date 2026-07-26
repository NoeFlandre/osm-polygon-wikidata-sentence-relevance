"""Minimal GNU ``timeout`` shim used to exercise the deadline helper on
hosts where the real binary is unavailable.

The shim implements only the flags the helper uses:
``--foreground --preserve-status --signal=INT --kill-after=<seconds>``.

It is not a complete GNU ``timeout`` re-implementation; the production
helper runs on Grid'5000 Linux where the real ``timeout`` is available.
The shim exists solely so the deadline tests run on developer machines
that do not ship GNU ``coreutils`` by default.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if "--help" in args:
        print(
            "fake-timeout --foreground --preserve-status "
            "--signal=INT --kill-after=<seconds> <duration> <child> [args...]"
        )
        return 0
    preserve_status = "--preserve-status" in args
    kill_after: float | None = None
    duration: float | None = None
    signal_name = "INT"
    child: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--foreground":
            i += 1
            continue
        if arg == "--preserve-status":
            i += 1
            continue
        if arg == "--signal":
            i += 1
            signal_name = args[i]
            i += 1
            continue
        if arg.startswith("--signal="):
            signal_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--kill-after":
            i += 1
            kill_after = float(args[i])
            i += 1
            continue
        if arg.startswith("--kill-after="):
            kill_after = float(arg.split("=", 1)[1])
            i += 1
            continue
        if duration is None:
            duration = float(arg)
            i += 1
            continue
        child = args[i:]
        break
    if duration is None or not child:
        print("fake-timeout: missing duration or child", file=sys.stderr)
        return 2
    sigint_signal = getattr(signal, f"SIG{signal_name}", signal.SIGINT)
    proc = subprocess.Popen(child)
    deadline = time.monotonic() + duration
    while True:
        if proc.poll() is not None:
            rc = proc.returncode
            return 124 if (rc == -signal.SIGINT and not preserve_status) else rc
        if time.monotonic() >= deadline:
            proc.send_signal(sigint_signal)
            if kill_after is not None:
                kill_deadline = time.monotonic() + kill_after
                while proc.poll() is None and time.monotonic() < kill_deadline:
                    time.sleep(0.05)
                if proc.poll() is None:
                    proc.send_signal(signal.SIGKILL)
                    proc.wait()
            else:
                proc.wait()
            rc = proc.returncode
            return 124 if (rc == -signal.SIGINT and not preserve_status) else rc
        time.sleep(0.05)


if __name__ == "__main__":
    sys.exit(main())
