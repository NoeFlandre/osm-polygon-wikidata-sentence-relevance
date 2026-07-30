"""Terminal-safe presentation for the autonomous operator."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import IO, Any, Self

from rich.console import Console
from rich.markup import escape
from tqdm import tqdm


@dataclass(slots=True)
class ProgressView:
    """One optional bounded terminal progress display."""

    _bar: tqdm[Any] | None

    def __enter__(self) -> Self:
        return self

    def advance(self) -> None:
        """Advance by one completed item when interactive."""

        if self._bar is not None:
            self._bar.update(1)

    def close(self) -> None:
        """Finish the display and leave subsequent output on a clean line."""

        if self._bar is not None:
            self._bar.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class OperatorConsole:
    """Render human output without polluting redirected or machine output."""

    def __init__(
        self,
        *,
        stdout: IO[str] | None = None,
        stderr: IO[str] | None = None,
        interactive: bool | None = None,
    ) -> None:
        self._stdout_override = stdout
        self._stderr_override = stderr
        self._interactive_override = interactive

    @property
    def _stdout(self) -> IO[str]:
        return (
            self._stdout_override if self._stdout_override is not None else sys.stdout
        )

    @property
    def _stderr(self) -> IO[str]:
        return (
            self._stderr_override if self._stderr_override is not None else sys.stderr
        )

    def _interactive(self, stream: IO[str]) -> bool:
        requested = (
            stream.isatty()
            if self._interactive_override is None
            else self._interactive_override
        )
        return requested and "NO_COLOR" not in os.environ

    @staticmethod
    def _rich(stream: IO[str]) -> Console:
        return Console(
            file=stream,
            force_terminal=True,
            color_system="standard",
            no_color=False,
            markup=True,
            highlight=False,
            soft_wrap=True,
        )

    def milestone(self, message: str) -> None:
        """Write one immediately visible operator milestone."""

        stream = self._stdout
        if self._interactive(stream):
            self._rich(stream).print(
                f"[bold cyan]{escape('[operator]')}[/bold cyan] {escape(message)}"
            )
        else:
            print(f"[operator] {message}", file=stream, flush=True)

    def error(self, message: str) -> None:
        """Write one concise operational error."""

        stream = self._stderr
        if self._interactive(stream):
            self._rich(stream).print(f"[bold red]Error:[/bold red] {escape(message)}")
        else:
            print(f"Error: {message}", file=stream, flush=True)

    def plain(self, message: str, *, error: bool = False) -> None:
        """Write stable literal text to the selected stream."""

        print(message, file=self._stderr if error else self._stdout, flush=True)

    def job_lines(self, job_id: int, text: str) -> None:
        """Write remote log text literally, never as terminal markup."""

        stream = self._stdout
        for line in text.splitlines():
            print(f"[job {job_id}] {line}", file=stream, flush=True)

    def json(self, payload: Mapping[str, object]) -> None:
        """Write deterministic machine-readable JSON without decoration."""

        print(
            json.dumps(payload, indent=2, sort_keys=True),
            file=self._stdout,
            flush=True,
        )

    def progress(self, *, description: str, total: int) -> ProgressView:
        """Create a bounded interactive progress display or a silent no-op."""

        stream = self._stdout
        bar = (
            tqdm(
                total=total,
                desc=description,
                file=stream,
                disable=False,
                dynamic_ncols=True,
                unit="sites",
            )
            if self._interactive(stream)
            else None
        )
        return ProgressView(bar)


__all__ = ["OperatorConsole", "ProgressView"]
