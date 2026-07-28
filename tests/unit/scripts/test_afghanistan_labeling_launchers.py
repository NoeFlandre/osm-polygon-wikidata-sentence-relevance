from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "scripts" / "grid5000"
GUIDES = ROOT / "docs" / "guides" / "grid5000.md"
SCRIPT = ROOT / "scripts" / "grid5000" / "submit_afghanistan_labeling.sh"
COMMIT = "a" * 40
REVISION = "b" * 40


def _text(name: str) -> str:
    return (GRID / name).read_text()


def _run_submit(
    root: Path,
    work_dir: Path,
    output_dir: Path | None = None,
    *,
    gpu_min_memory_mb: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if output_dir is None:
        output_dir = root / "output"

    fake_bin = root / "bin"
    fake_bin.mkdir(parents=True)
    fake_call = root / "oarsub.calls"
    call_count = root / "oarsub.call_count"

    fake_bin.joinpath("oarsub").write_text(
        f"#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$#" >> "{call_count}"\n'
        f'printf "%s\\n" "$*" >> "{fake_call}"\n'
        "echo 123456\n",
    )
    fake_bin.joinpath("oarsub").chmod(0o700)

    run_root = root
    repo_root = run_root / "repo"
    hf_home = run_root / "hf_home"
    log_root = run_root / "logs"
    input_parquet = run_root / "input.parquet"
    model_file = run_root / "model.gguf"
    tokenizer_dir = run_root / "tokenizer"
    repo_target = repo_root / "scripts" / "grid5000"
    repo_target.mkdir(parents=True)
    for item in (
        GRID / "run_afghanistan_labeling_job.sh",
        GRID / "run_afghanistan_labeling.sh",
    ):
        destination = repo_target / item.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.read_bytes())
        destination.chmod(item.stat().st_mode)
    for required_dir in (
        repo_root / "scripts" / "grid5000",
        hf_home,
        log_root,
        tokenizer_dir,
        work_dir.parent,
        output_dir.parent,
    ):
        required_dir.mkdir(parents=True, exist_ok=True)
    for required_file in (
        input_parquet,
        model_file,
        tokenizer_dir / "README.md",
        repo_root / "scripts" / "grid5000" / "_deadline_helper.sh",
        repo_root / "scripts" / "grid5000" / "_checkout_guard.sh",
    ):
        required_file.parent.mkdir(parents=True, exist_ok=True)
        required_file.write_text("ok")

    env = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    if gpu_min_memory_mb is not None:
        env["LABEL_GPU_MIN_MEMORY_MB"] = gpu_min_memory_mb

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(repo_root),
            str(hf_home),
            str(log_root),
            str(input_parquet),
            str(work_dir),
            str(output_dir),
            str(model_file),
            str(tokenizer_dir),
            REVISION,
            REVISION,
            COMMIT,
            "NoeFlandre/osm-polygon-wikidata-sentence-relevance",
            "128",
            "0",
            "16",
            "8192",
            "16",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result


def test_labeling_launchers_are_executable() -> None:
    for name in (
        "submit_afghanistan_labeling.sh",
        "run_afghanistan_labeling_job.sh",
        "run_afghanistan_labeling.sh",
    ):
        assert os.stat(GRID / name).st_mode & 0o111


def test_submitter_requests_one_fast_large_cuda_gpu_once() -> None:
    text = _text("submit_afghanistan_labeling.sh")
    assert text.count("exec oarsub ") == 1
    assert "gpu=1,walltime=12:00:00" in text
    assert 'GPU_MIN_MEMORY_MB="${LABEL_GPU_MIN_MEMORY_MB:-40000}"' in text
    assert "gpu_mem>=${GPU_MIN_MEMORY_MB}" in text
    assert " -q default" in text
    assert " -t exotic" in text
    assert " -t besteffort" not in text
    assert " -I" not in text


def test_submitter_accepts_validated_gpu_memory_override(tmp_path: Path) -> None:
    result = _run_submit(
        tmp_path,
        tmp_path / "work",
        gpu_min_memory_mb="40000",
    )

    assert result.returncode == 0, result.stderr
    assert "gpu_mem>=40000" in tmp_path.joinpath("oarsub.calls").read_text()


def test_submitter_rejects_invalid_gpu_memory_override(tmp_path: Path) -> None:
    result = _run_submit(
        tmp_path,
        tmp_path / "work",
        gpu_min_memory_mb="forty-gigabytes",
    )

    assert result.returncode == 2
    assert "GPU memory minimum must be a positive integer" in result.stderr
    assert not tmp_path.joinpath("oarsub.calls").exists()


def test_job_wrapper_verifies_checkout_gpu_and_persists_logs() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    assert "gpu_preflight.py" in text
    assert "git -C" in text
    assert "validate_clean_checkout" in text
    assert "_checkout_guard.sh" in text
    assert 'JOB_LOG_DIR="${LOG_ROOT}/${OAR_JOB_ID}"' in text
    assert "labeling.exit_code" in text
    assert "run_afghanistan_labeling.sh" in text


def test_job_wrapper_uses_run_root_llama_server_binary() -> None:
    text = _text("run_afghanistan_labeling_job.sh")

    assert 'LLAMA_SERVER_DIR="${RUN_ROOT}/llama-server-bin"' in text
    assert '[ ! -x "${LLAMA_SERVER_DIR}/llama-server" ]' in text
    assert 'export PATH="${LLAMA_SERVER_DIR}:${PATH}"' in text


def test_job_wrapper_invokes_deadline_helper() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    assert "_deadline_helper.sh" in text
    assert "deadline_helper_run" in text
    # The wrapper must give the labeling CLI 700m and a 10m grace window.
    # so it can checkpoint before OAR's final TERM/KILL window.
    assert "700m 10m" in text


def test_job_wrapper_translates_submit_arguments_to_payload_contract() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    normalized = " ".join(text.replace("\\", "").split())
    expected = (
        '"${PAYLOAD}" "${REPO_ROOT}" "$4" "$5" "$6" "$7" "$8" "$9" '
        '"${10}" "${11}" "${12}" "${13}" "${14}" "${LLAMA_PARALLEL}" '
        '"${LLAMA_PER_SLOT_CONTEXT}" "${REQUEST_CONCURRENCY}"'
    )
    assert expected in normalized
    assert '"${PAYLOAD}" "$@"' not in text


def test_submitter_propagates_llama_parallel_positional() -> None:
    text = _text("submit_afghanistan_labeling.sh")
    assert 'LLAMA_PARALLEL="${15}"' in text
    assert "exactly seventeen arguments" in text
    assert "1|2|4|8|16|32" in text


def test_job_wrapper_validates_llama_parallel_set() -> None:
    text = _text("run_afghanistan_labeling_job.sh")
    assert 'LLAMA_PARALLEL="${15}"' in text
    assert "1|2|4|8|16|32" in text


def test_canary_launch_contract_never_publishes() -> None:
    text = _text("run_afghanistan_labeling.sh")
    assert 'if [ "${ROW_LIMIT}" -eq 0 ]; then' in text
    assert '"${LABEL_CLI}" publish' in text


def test_payload_reads_engine_version_without_sigpipe() -> None:
    text = _text("run_afghanistan_labeling.sh")

    assert "llama-server --version 2>&1 | head" not in text
    assert "llama-server --version 2>&1 | sed -n '1p'" in text


def test_docs_include_canonical_full_run_args() -> None:
    text = GUIDES.read_text()
    assert '"NoeFlandre/osm-polygon-wikidata-sentence-relevance"' in text
    assert '"128" "0" "8" "8192" "8"' in text
    assert '"0" "0" "16"' not in text


def _is_under_root(root: Path, candidate: Path) -> bool:
    env = {
        **os.environ,
        "SCRIPT_PATH": str(SCRIPT),
        "ROOT_PATH": str(root),
        "CANDIDATE_PATH": str(candidate),
    }
    completed = subprocess.run(
        [
            "bash",
            "-lc",
            'source "$SCRIPT_PATH"; '
            'root="$(canonicalize_path "$ROOT_PATH")"; '
            'candidate="$(canonicalize_path "$CANDIDATE_PATH")"; '
            'is_under_root "$candidate" "$root";',
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def test_submit_root_accepts_root_and_descendants(tmp_path: Path) -> None:
    root = tmp_path / "run" / "root"
    root.mkdir(parents=True)
    nested = root / "nested" / "deep"
    nested.mkdir(parents=True)
    assert _is_under_root(root, root)
    assert _is_under_root(root, root / "child")
    assert _is_under_root(root, nested)


def test_submit_root_rejects_sibling_like_prefix_or_traversal() -> None:
    root = Path("/run/root")
    sibling = Path("/run/root2")
    traversal = Path("/run/root/../escape")
    assert not _is_under_root(root, sibling)
    assert not _is_under_root(root, traversal)


def test_submit_root_rejects_symlink_or_broken_entries(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    escape_link = root / "link-escape"
    escape_link.symlink_to(outside)
    broken = root / "link-broken"
    broken.symlink_to(root / "does-not-exist")
    assert not _is_under_root(root, escape_link)
    assert _is_under_root(root, broken)


def test_submit_root_accepts_paths_with_spaces_and_glob_chars(tmp_path: Path) -> None:
    root = tmp_path / "run root"
    root.mkdir(parents=True)
    special = root / "with space [x] * ? file"
    special.mkdir()
    assert _is_under_root(root, special)


def test_submit_helper_hits_oarsub_once_with_valid_paths(tmp_path: Path) -> None:
    root = tmp_path / "runroot"
    work_dir = root / "work"
    output_dir = root / "output"
    result = _run_submit(root, work_dir, output_dir)
    call_count = root / "oarsub.call_count"
    assert result.returncode == 0
    assert result.stdout.strip() == "123456"
    assert call_count.exists()
    assert call_count.read_text().strip() == "9"
    assert "submit_afghanistan_labeling: required" not in result.stderr


def test_submit_helper_rejects_escaped_path_before_oarsub(tmp_path: Path) -> None:
    root = tmp_path / "runroot"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    work_dir = outside / "work"
    output_dir = root / "output"
    result = _run_submit(root, work_dir, output_dir)
    assert result.returncode != 0
    call_count = root / "oarsub.call_count"
    assert not call_count.exists()
    assert "path is outside the approved run root" in result.stderr
    assert str(root) not in result.stderr


def test_submit_helper_rejects_broken_symlink_before_oarsub(tmp_path: Path) -> None:
    root = tmp_path / "runroot"
    root.mkdir()
    work_dir = root / "broken-work"
    work_dir.symlink_to(root / "missing-work")
    result = _run_submit(root, work_dir)
    assert result.returncode != 0
    assert not (root / "oarsub.call_count").exists()
    assert "work directory must not be a symlink" in result.stderr


def test_submit_helper_stable_error_output_on_rejected_path(tmp_path: Path) -> None:
    root = tmp_path / "runroot"
    work_dir = root / "../outside"
    output_dir = root / "output"
    result = _run_submit(root, work_dir, output_dir)
    assert result.returncode != 0
    assert (
        "submit_afghanistan_labeling: path is outside the approved run root"
        in result.stderr
    )
    assert "/runroot" not in result.stderr
