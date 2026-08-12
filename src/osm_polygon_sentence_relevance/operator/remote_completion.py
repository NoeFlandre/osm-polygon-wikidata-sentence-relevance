"""Remote completion evidence, publication results, and status marking."""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from osm_polygon_sentence_relevance.labeling.v2_finalization import (
    V2_PUBLICATION_FILES,
    validate_v2_publication,
)
from osm_polygon_sentence_relevance.operator.config import DATA_ROOT
from osm_polygon_sentence_relevance.operator.relay_transport import RemoteTransfer
from osm_polygon_sentence_relevance.operator.result_text import result_text
from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout

_LABEL_RELEASE_FILES: tuple[str, ...] = (
    "sentences.parquet",
    "manifest.json",
    "README.md",
    "assets/label_distribution.png",
    "assets/positive_languages.png",
    "assets/joint_label_heatmap.png",
    "assets/polygon_coverage_funnel.png",
    "assets/reason_code_distribution.png",
)

_V2_MANUAL_EVAL_FIELDS = frozenset(
    {
        "sentence_id",
        "page_title",
        "section_title",
        "previous_sentence",
        "sentence_text",
        "next_sentence",
        "model_label",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
        "human_label",
        "notes",
    }
)


def _publish_local_label_output(directory: Path, dataset_id: str) -> str:
    """Publish a local finalized label directory through the normal validator."""

    from osm_polygon_sentence_relevance.labeling.publication import (
        publish_labeled_dataset,
    )

    return publish_labeled_dataset(  # type: ignore[no-any-return]
        directory,
        dataset_id,
    ).commit_id


def remote_exit_code(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
    filename: str,
) -> int:
    """Read and parse the payload exit-code file for a remote job."""

    path = layout.logs / str(job_id) / filename
    result = ssh.run(f"test -f {path!s} && cat {path!s}")
    try:
        return int(result_text(result).strip())
    except ValueError as exc:
        raise RuntimeError("remote payload exit status is invalid") from exc


def assert_remote_exit_zero(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
    filename: str,
) -> None:
    """Require that the remote payload exit status is zero."""

    if remote_exit_code(ssh, layout, job_id, filename) != 0:
        raise RuntimeError("remote payload returned non-zero status")


def publish_split(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
    dataset_id: str,
) -> str:
    """Invoke export publication remotely and return the Hub commit SHA."""

    code = (
        "from osm_polygon_sentence_relevance.publishing import "
        "publish_export_directory; "
        f"r=publish_export_directory({str(output_dir)!r},{dataset_id!r},"
        "target_revision='main'); print(r.commit_id)"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (str(layout.repo / ".venv/bin/python"), "-c", code)
    )
    token_file = shlex.quote(str(layout.hf_token))
    result = ssh.run(
        "set -euo pipefail; "
        f"[ -f {token_file} ] && [ ! -L {token_file} ] && "
        f'[ "$(stat -c %a -- {token_file})" = 600 ]; '
        f'export HF_TOKEN="$(cat -- {token_file})"; '
        f'[ -n "$HF_TOKEN" ]; exec {command}'
    )
    commit_id = result_text(result).strip()
    if len(commit_id) < 7:
        raise RuntimeError("Hugging Face publication did not return a commit")
    return commit_id


def publish_label(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
    dataset_id: str,
    *,
    v2: bool = False,
) -> str:
    """Fetch the finalized release locally and publish from the authenticated Mac.

    Grid'5000 frontends do not carry the operator's Hub credentials and may
    not be able to import the package's NumPy/PyArrow stack. The release is
    small and already complete, so transfer its closed validated file set to
    the Seagate-backed run directory and use the normal local publisher. A
    completed local relay is retained so a retry never refetches it.
    """

    relay_root = _retrieve_label_output(
        ssh,
        layout,
        output_dir,
        relay_name="label-publication",
        release_files=V2_PUBLICATION_FILES if v2 else _LABEL_RELEASE_FILES,
    )

    try:
        commit_id = _publish_local_label_output(relay_root, dataset_id)
    except Exception as exc:
        raise RuntimeError("local Hugging Face label publication failed") from exc
    return commit_id


def _retrieve_label_output(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
    *,
    relay_name: str,
    release_files: tuple[str, ...] = _LABEL_RELEASE_FILES,
) -> Path:
    """Fetch one completed label output into an idempotent Seagate relay."""

    run_dir = DATA_ROOT / "runs" / layout.root.name
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    relay_root = run_dir / relay_name
    if relay_root.is_symlink():
        raise RuntimeError("local label publication relay is a symlink")

    if not relay_root.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{relay_name}-", dir=run_dir))
        try:
            transfer = RemoteTransfer(ssh_target=ssh.target)
            for relative in release_files:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                transfer.fetch(str(output_dir / relative), destination)
            os.replace(staging, relay_root)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError("failed to retrieve completed label output") from exc
    return relay_root


def preserve_label(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
) -> Path:
    """Durably preserve a completed smoke output without publishing it."""

    relay_root = _retrieve_label_output(
        ssh,
        layout,
        output_dir,
        relay_name="label-smoke",
        release_files=V2_PUBLICATION_FILES,
    )
    validate_v2_publication(relay_root)
    return relay_root


def _validate_manual_eval(path: Path) -> None:
    """Reject malformed or incomplete V2 review samples before preservation."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("manual evaluation artifact is not a regular file")
    seen: set[str] = set()
    rows = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if not isinstance(payload, dict) or set(payload) != _V2_MANUAL_EVAL_FIELDS:
                raise ValueError("manual evaluation fields are invalid")
            sentence_id = payload["sentence_id"]
            if (
                not isinstance(sentence_id, str)
                or not sentence_id
                or sentence_id in seen
            ):
                raise ValueError("manual evaluation sentence IDs are invalid")
            seen.add(sentence_id)
            if payload["model_label"] not in {"yes", "no"}:
                raise ValueError("manual evaluation model label is invalid")
            for name in (
                "yes_logprob",
                "no_logprob",
                "logit_margin",
                "two_class_probability",
            ):
                value = payload[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("manual evaluation score is invalid")
                if not math.isfinite(float(value)):
                    raise ValueError("manual evaluation score is invalid")
            probability = float(payload["two_class_probability"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("manual evaluation probability is invalid")
            if not isinstance(payload["human_label"], str) or not isinstance(
                payload["notes"], str
            ):
                raise ValueError("manual evaluation review fields are invalid")
            rows += 1
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("manual evaluation artifact is invalid") from exc
    if rows < 1 or rows > 100:
        raise RuntimeError("manual evaluation row count is invalid")


def preserve_manual_eval(
    ssh: SshClient,
    layout: RemoteLayout,
    work_dir: PurePosixPath,
    *,
    lane: str,
) -> Path:
    """Preserve one editable V2 manual-review sample before remote cleanup."""

    if lane not in {"smoke", "production"}:
        raise ValueError("manual evaluation lane is invalid")
    run_dir = DATA_ROOT / "runs" / layout.root.name
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = run_dir / f"manual-eval-{lane}.jsonl"
    if destination.is_symlink():
        raise RuntimeError("manual evaluation destination is a symlink")
    if destination.exists():
        _validate_manual_eval(destination)
        return destination

    staging = Path(tempfile.mkdtemp(prefix=f".manual-eval-{lane}-", dir=run_dir))
    artifact = staging / "manual_eval.jsonl"
    try:
        RemoteTransfer(ssh_target=ssh.target).fetch(
            str(work_dir / "manual_eval.jsonl"), artifact
        )
        _validate_manual_eval(artifact)
        os.replace(artifact, destination)
    except Exception as exc:
        raise RuntimeError("failed to preserve manual evaluation artifact") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def label_publication_commit(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
) -> str:
    """Read the verified label publisher's immutable Hub commit from its log."""

    path = layout.logs / str(job_id) / "labeling.stdout.log"
    text = result_text(ssh.run(f"test -f {path!s} && cat {path!s}"))
    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        commit_id = payload.get("commit_id") if isinstance(payload, dict) else None
        if (
            isinstance(commit_id, str)
            and len(commit_id) == 40
            and all(character in "0123456789abcdef" for character in commit_id)
        ):
            return commit_id
    raise RuntimeError("label publication did not report an immutable Hub commit")


def mark_remote_status(ssh: SshClient, layout: RemoteLayout, status: str) -> None:
    """Write the pipeline-managed status marker file on the remote site."""

    if status not in {"active", "complete", "failed"}:
        raise ValueError("invalid managed status")
    marker = layout.root / ".operator-managed.json"
    ssh.run(
        "printf '%s\\n' "
        + shlex.quote(
            json.dumps(
                {"schema_version": 1, "status": status},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        + f" > {shlex.quote(str(marker))} && chmod 0600 {shlex.quote(str(marker))}"
    )


__all__ = [
    "assert_remote_exit_zero",
    "label_publication_commit",
    "mark_remote_status",
    "preserve_label",
    "preserve_manual_eval",
    "publish_label",
    "publish_split",
    "remote_exit_code",
]
