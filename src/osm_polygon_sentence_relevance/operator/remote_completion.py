"""Remote completion evidence, publication results, and status marking."""

from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath

from osm_polygon_sentence_relevance.operator.ssh import SshClient
from osm_polygon_sentence_relevance.operator.workflows import RemoteLayout


def _result_text(result: object) -> str:
    """Return ``result.text`` if available, else ``result.stdout``."""

    text_attr = getattr(result, "text", None)
    if text_attr is not None:
        return str(text_attr)
    return str(getattr(result, "stdout", ""))


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
    """Publish finalized labels with a frontend-compatible Hub-only check.

    This recovery path intentionally avoids importing the package's Parquet
    validator: some Grid'5000 frontends cannot import the remote NumPy build.
    It validates the closed output file set, manifest-recorded SHA-256 values,
    and immutable Hub readback before returning the commit.
    """

    script = f"""
import hashlib
import json
import tempfile
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

root = Path({str(output_dir)!r})
dataset_id = {dataset_id!r}
target_revision = 'main'
expected = (
    'sentences.parquet',
    'manifest.json',
    'README.md',
    'assets/label_distribution.png',
    'assets/positive_languages.png',
)
files = {{
    str(path.relative_to(root))
    for path in root.rglob('*')
    if path.is_file() and not path.is_symlink()
}}
if files != set(expected):
    raise RuntimeError('label output file set is invalid')
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
if manifest.get('dataset_repo_id') != dataset_id:
    raise RuntimeError('label output dataset ID mismatch')
digests = manifest.get('artifact_sha256')
if not isinstance(digests, dict):
    raise RuntimeError('label output artifact hashes are missing')
for relative in expected:
    if relative in {{'manifest.json', 'README.md'}}:
        continue
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != digests.get(relative):
        raise RuntimeError('label output hash mismatch')

api = HfApi()
remote = set(
    api.list_repo_files(
        repo_id=dataset_id,
        repo_type='dataset',
        revision=target_revision,
    )
)
allowed = set(expected) | {{
    '.gitattributes',
    'assets/geographic_coverage.png',
    'assets/language_distribution.png',
}}
if remote - allowed:
    raise RuntimeError('remote label tree contains unexpected files')
operations = [
    CommitOperationAdd(
        path_in_repo=relative,
        path_or_fileobj=str(root / relative),
    )
    for relative in expected
]
operations.extend(
    CommitOperationDelete(path_in_repo=relative)
    for relative in remote & {{
        'assets/geographic_coverage.png',
        'assets/language_distribution.png',
    }}
)
info = api.create_commit(
    repo_id=dataset_id,
    repo_type='dataset',
    operations=operations,
    commit_message=(
        f"Publish {{manifest.get('statistics', {{}}).get('row_count', 0)}} "
        'Afghanistan relevance labels'
    ),
    revision=target_revision,
)
commit_id = info.oid
if not isinstance(commit_id, str) or len(commit_id) != 40 or any(
    character not in '0123456789abcdef' for character in commit_id
):
    raise RuntimeError('Hub returned an invalid commit')
with tempfile.TemporaryDirectory(prefix='label-readback-') as temporary:
    for relative in expected:
        downloaded = Path(
            hf_hub_download(
                repo_id=dataset_id,
                repo_type='dataset',
                revision=commit_id,
                filename=relative,
                local_dir=temporary,
            )
        )
        if relative not in {{'manifest.json', 'README.md'}}:
            actual = hashlib.sha256(downloaded.read_bytes()).hexdigest()
            if actual != digests.get(relative):
                raise RuntimeError('Hub readback hash mismatch')
print(commit_id)
"""
    code = (
        "from huggingface_hub import CommitOperationAdd, CommitOperationDelete, "
        "HfApi, hf_hub_download; "
        f"output_dir={str(output_dir)!r}; "
        f"dataset_id={dataset_id!r}; target_revision='main'; "
        f"exec({script!r})"
    )
    command = " ".join(
        shlex.quote(value)
        for value in (str(layout.repo / ".venv/bin/python"), "-c", code)
    )
    result = ssh.run(command)
    commit_id = _result_text(result).strip()
    if len(commit_id) != 40 or any(
        character not in "0123456789abcdef" for character in commit_id
    ):
        raise RuntimeError("Hugging Face label publication did not return a commit")
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
