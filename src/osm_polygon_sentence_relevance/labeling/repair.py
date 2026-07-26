"""Bounded repair attempt for invalid model responses.

The production labeling pipeline uses a closed JSON schema and an exact-substring
evidence rule. When a model response fails validation, the runner offers the
model one bounded repair attempt before failing the batch. The repair message
retains the original prompt and tells the model the exact rule that failed so
it can produce a corrected response. The replacement is validated against the
same strict contract; no silent fallback to an empty evidence or a relaxed
schema is permitted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import SentenceLabel
from .validation import LabelValidationError, parse_label_response

_MAX_REPAIR_ATTEMPTS = 1

RepairEngine = Callable[[Sequence[list[dict[str, str]]]], list[str]]
Messages = list[dict[str, str]]


class RepairExhausted(RuntimeError):
    """Raised when a bounded repair attempt fails to produce a valid response."""


def _failure_reason(error: LabelValidationError) -> str:
    """Return a stable, log-safe description of the validation failure."""

    message = str(error)
    if "exact substring" in message:
        return "evidence is not an exact substring of target sentence"
    if "inconsistent" in message:
        return "reason is inconsistent with the relevance label"
    if "evidence" in message and "240" in message:
        return "evidence exceeds the 240-character limit"
    return "response violates the structured label contract"


def _build_repair_messages(
    messages: Messages,
    target_sentence: str,
    reason: str,
) -> Messages:
    """Return a copy of ``messages`` with a one-shot repair instruction appended."""

    if len(messages) >= 2 and messages[0].get("role") == "system":
        return [
            messages[0],
            *messages[1:],
            {
                "role": "user",
                "content": (
                    "Your previous response was rejected because "
                    f"{reason}. "
                    "Re-emit the JSON object with the corrections:\n"
                    f"- TARGET SENTENCE for substring matching: {target_sentence!r}\n"
                    "- The JSON object must contain exactly five fields: "
                    "landuse_relevance, polygon_relevance, landuse_reason, "
                    "polygon_reason, evidence.\n"
                    "- 'evidence' must be a short exact excerpt of the TARGET "
                    "SENTENCE above, or an empty string when no useful excerpt exists.\n"
                    "- 'landuse_reason' and 'polygon_reason' must be consistent with "
                    "their 'yes'/'no'/'uncertain' relevance labels."
                ),
            },
        ]
    return messages


def _invoke_engine(engine: RepairEngine, messages: Messages) -> str:
    """Invoke the engine and return the single response for these messages."""

    outputs = engine([messages])
    if len(outputs) != 1:
        raise RepairExhausted("engine returned an unexpected response count")
    return outputs[0]


@dataclass(frozen=True, slots=True)
class RepairStats:
    """Factual, content-free accounting of repair attempts."""

    initial_failures: int = 0
    repaired: int = 0
    exhausted: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "initial_failures": self.initial_failures,
            "repaired": self.repaired,
            "exhausted": self.exhausted,
            "reasons": dict(self.reasons),
        }


def sanitize_for_log(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a log-safe copy of ``payload`` with prompt/response content redacted."""

    redacted: dict[str, object] = {}
    for key, value in payload.items():
        if key in {"prompt", "response", "raw_response", "raw_request"}:
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


class BoundedRepair:
    """Apply one bounded repair attempt and validate the replacement strictly."""

    def __init__(self, *, max_attempts: int = _MAX_REPAIR_ATTEMPTS) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self._stats = RepairStats()

    @property
    def stats(self) -> RepairStats:
        return self._stats

    def log_redacted_failure(
        self,
        *,
        sentence_id: str,
        reason: str,
        attempt: int,
    ) -> dict[str, object]:
        """Return a redacted log entry describing one failed repair attempt."""

        return sanitize_for_log(
            {
                "sentence_id": sentence_id,
                "reason": reason,
                "attempt": attempt,
                "event": "label_repair",
            }
        )

    def call(
        self,
        *,
        engine: RepairEngine,
        messages: Messages,
        target_sentence: str,
    ) -> SentenceLabel:
        """Run one attempt plus up to ``max_attempts`` repair attempts."""

        messages_copy = list(messages)
        raw = _invoke_engine(engine, messages_copy)
        last_error: LabelValidationError | None
        try:
            return parse_label_response(raw, target_sentence=target_sentence)
        except LabelValidationError as initial:
            initial_reason = _failure_reason(initial)
            self._stats = RepairStats(
                initial_failures=self._stats.initial_failures + 1,
                repaired=self._stats.repaired,
                exhausted=self._stats.exhausted,
                reasons={
                    **self._stats.reasons,
                    initial_reason: self._stats.reasons.get(initial_reason, 0) + 1,
                },
            )
            last_error = initial
        for _attempt in range(1, self.max_attempts + 1):
            reason = _failure_reason(last_error)
            repair_messages = _build_repair_messages(
                messages_copy, target_sentence, reason
            )
            raw = _invoke_engine(engine, repair_messages)
            try:
                label = parse_label_response(raw, target_sentence=target_sentence)
            except LabelValidationError as exc:
                last_error = exc
                self._stats = RepairStats(
                    initial_failures=self._stats.initial_failures,
                    repaired=self._stats.repaired,
                    exhausted=self._stats.exhausted + 1,
                    reasons={
                        **self._stats.reasons,
                        _failure_reason(exc): self._stats.reasons.get(
                            _failure_reason(exc), 0
                        )
                        + 1,
                    },
                )
                continue
            self._stats = RepairStats(
                initial_failures=self._stats.initial_failures,
                repaired=self._stats.repaired + 1,
                exhausted=self._stats.exhausted,
                reasons=self._stats.reasons,
            )
            return label
        if last_error is not None:
            raise RepairExhausted(
                "model failed to produce a valid response after "
                f"{self.max_attempts} repair attempt(s)"
            ) from last_error
        raise RepairExhausted("model failed to produce a valid response")


def repair_response(
    raw: str,
    *,
    target_sentence: str,
    engine: RepairEngine,
    messages: Messages,
    max_attempts: int = _MAX_REPAIR_ATTEMPTS,
) -> tuple[SentenceLabel, RepairStats]:
    """Convenience wrapper that returns the label and the repair stats."""

    repair = BoundedRepair(max_attempts=max_attempts)
    label = repair.call(
        engine=engine, messages=messages, target_sentence=target_sentence
    )
    return label, repair.stats


def repair_response_payload_to_dict(
    label: SentenceLabel,
    stats: RepairStats,
) -> dict[str, object]:
    """Return a JSON-safe mapping of one repair outcome."""

    return {
        "label": {
            "landuse_relevance": label.landuse_relevance.value,
            "polygon_relevance": label.polygon_relevance.value,
            "landuse_reason": label.landuse_reason,
            "polygon_reason": label.polygon_reason,
            "evidence": label.evidence,
        },
        "stats": stats.to_dict(),
    }


def _coerce_for_json(value: Any) -> Any:
    """Best-effort conversion to a JSON-serialisable value."""

    if isinstance(value, dict):
        return {str(key): _coerce_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_for_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def repair_response_json(
    label: SentenceLabel,
    stats: RepairStats,
) -> str:
    """Render a JSON-safe string for diagnostics; never includes raw content."""

    return json.dumps(
        _coerce_for_json(repair_response_payload_to_dict(label, stats)),
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "BoundedRepair",
    "RepairEngine",
    "RepairExhausted",
    "RepairStats",
    "Messages",
    "repair_response",
    "repair_response_json",
    "repair_response_payload_to_dict",
    "sanitize_for_log",
]
