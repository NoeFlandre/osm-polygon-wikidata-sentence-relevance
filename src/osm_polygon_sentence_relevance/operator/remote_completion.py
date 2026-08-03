"""Remote completion evidence, publication results, and status marking."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from osm_polygon_sentence_relevance.operator.config import DATA_ROOT
from osm_polygon_sentence_relevance.operator.relay_transport import RemoteTransfer
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
    "assets/slice_yield.html",
)


def _result_text(result: object) -> str:
    """Return ``result.text`` if available, else ``result.stdout``."""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return str(text_attr)
    return str(getattr(result, "stdout", ""))


def _publish_local_label_output(directory: Path, dataset_id: str) -> str:
    """Publish a local finalized label directory through the normal validator."""

    from osm_polygon_sentence_relevance.labeling.publication import (
        publish_labeled_dataset,
    )

    return publish_labeled_dataset(  # type: ignore[no-any-return]
        directory,
        dataset_id,
        target_revision="main",
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
        return int(_result_text(result).strip())
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
    result = ssh.run(command)
    commit_id = _result_text(result).strip()
    if len(commit_id) < 7:
        raise RuntimeError("Hugging Face publication did not return a commit")
    return commit_id


def publish_label(
    ssh: SshClient,
    layout: RemoteLayout,
    output_dir: PurePosixPath,
    dataset_id: str,
) -> str:
    """Fetch the finalized release locally and publish from the authenticated Mac.

    Grid'5000 frontends do not carry the operator's Hub credentials and may
    not be able to import the package's NumPy/PyArrow stack. The release is
    small and already complete, so transfer exactly its five validated files
    to the Seagate-backed run directory and use the normal local publisher.
    A completed local relay is retained so a retry never refetches it.
    """

    run_dir = DATA_ROOT / "runs" / layout.root.name
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    relay_root = run_dir / "label-publication"
    if relay_root.is_symlink():
        raise RuntimeError("local label publication relay is a symlink")

    if not relay_root.exists():
        staging = Path(tempfile.mkdtemp(prefix=".label-publication-", dir=run_dir))
        try:
            transfer = RemoteTransfer(ssh_target=ssh.target)
            for relative in _LABEL_RELEASE_FILES:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                transfer.fetch(str(output_dir / relative), destination)
            os.replace(staging, relay_root)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError("failed to retrieve completed label output") from exc

    try:
        commit_id = _publish_local_label_output(relay_root, dataset_id)
    except Exception as exc:
        raise RuntimeError("local Hugging Face label publication failed") from exc
    return commit_id


def label_publication_commit(
    ssh: SshClient,
    layout: RemoteLayout,
    job_id: int,
) -> str:
    """Read the verified label publisher's immutable Hub commit from its log."""

    path = layout.logs / str(job_id) / "labeling.stdout.log"
    text = _result_text(ssh.run(f"test -f {path!s} && cat {path!s}"))
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
    "publish_label",
    "publish_split",
    "remote_exit_code",
]
