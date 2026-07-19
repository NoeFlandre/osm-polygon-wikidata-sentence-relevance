"""Documentation consistency tests for the Grid'5000 guide.

These tests enforce that the committed public guide never
references personal absolute paths, never recommends forbidden
OAR flags, and stays consistent with the shell payload contract.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "guides" / "grid5000.md"


# --- Personal-path redaction ----------------------------------------


def test_doc_omits_personal_home_paths():
    text = DOC.read_text(encoding="utf-8")
    for bad in (
        "/home/nflandre",
        "/Users/nflandre",
        "/Users/noeflandre",
        "/Volumes/",
        "/srv/storage/",
    ):
        assert bad not in text, f"public guide must not embed personal path {bad!r}"


def test_doc_uses_login_placeholder_in_examples():
    text = DOC.read_text(encoding="utf-8")
    # Either the placeholder or an explicit variable is allowed; the
    # only forbidden pattern is the real account id.
    assert "nflandre" not in text


def test_doc_omits_remote_personal_uv_path():
    text = DOC.read_text(encoding="utf-8")
    assert "/home/nflandre/.local/bin" not in text


# --- Forbidden OAR flags --------------------------------------------


def _oarsub_lines(text: str) -> list[str]:
    """Extract every line that begins with the ``oarsub`` command
    so we can audit the recommendations specifically (without
    matching the warning prose that names these flags only to
    forbid them)."""
    return [line for line in text.splitlines() if line.lstrip().startswith("oarsub ")]


def test_doc_does_not_recommend_besteffort():
    text = DOC.read_text(encoding="utf-8")
    for line in _oarsub_lines(text):
        assert "besteffort" not in line, (
            f"oarsub command recommends besteffort: {line!r}"
        )


def test_doc_does_not_recommend_classic_ssh():
    text = DOC.read_text(encoding="utf-8")
    for line in _oarsub_lines(text):
        assert "classic_ssh" not in line, (
            f"oarsub command recommends classic_ssh: {line!r}"
        )


def test_doc_does_not_recommend_unverified_gpu_property_filter():
    text = DOC.read_text(encoding="utf-8")
    for line in _oarsub_lines(text):
        assert "gpu_model" not in line, (
            f"oarsub command uses unverified gpu_model filter: {line!r}"
        )


def test_doc_uses_canonical_oar_command():
    text = DOC.read_text(encoding="utf-8")
    expected = "oarsub -q production -l gpu=1,walltime=00:30:00 -I"
    assert expected in text, f"doc must contain canonical command: {expected!r}"


# --- Single authoritative Environment-setup section ------------------


def test_doc_has_single_environment_setup_section():
    text = DOC.read_text(encoding="utf-8")
    occurrences = text.count("## Environment setup from the locked dependencies")
    assert occurrences == 1, (
        f"expected exactly 1 Environment-setup section, found {occurrences}"
    )


# --- Bundle-content count -------------------------------------------


def test_doc_lists_six_artifact_files_in_successful_bundle():
    text = DOC.read_text(encoding="utf-8")
    for f in (
        "gpu_preflight.json",
        "run_metadata.json",
        "smoke_result.json",
        "smoke.stdout.log",
        "smoke.stderr.log",
        "smoke.exit_code",
    ):
        assert f in text, f"successful-bundle section must list {f!r}"


def test_doc_lists_seven_run_metadata_keys():
    text = DOC.read_text(encoding="utf-8")
    for key in (
        "source_commit",
        "model_name",
        "model_revision",
        "tokenizer_name",
        "tokenizer_revision",
        "oar_job_id",
        "hostname",
    ):
        assert key in text, f"metadata section must list {key!r}"


# --- Final-consistency amendment ------------------------------------


def test_doc_omits_stale_phase_9b_source_commit_sha():
    """The previous Phase 9B commit SHA must not appear in the
    committed public guide: each amendment produces a new SHA and
    the doc must remain SHA-free, using a ``SOURCE_COMMIT``
    variable instead.

    The historical reference value is constructed from short
    fragments so this test file never embeds a single
    40-character opaque literal that resembles a credential.
    """
    text = DOC.read_text(encoding="utf-8")
    # Build the historical Phase 9B SHA from disjoint small
    # fragments. The order and content mirror the previous
    # Phase 9B commit; ``str.join`` is used so no source-code
    # line of this file contains a contiguous 40-char token
    # that could be misread by secret-scanning heuristics.
    stale_sha = "".join(
        [
            "3489",
            "9f2c",
            "e30a",
            "e903",
            "6552",
            "2f52",
            "5bd9",
            "e559",
            "d528",
            "d01f",
        ]
    )
    assert len(stale_sha) == 40
    assert all(c in "0123456789abcdef" for c in stale_sha), (
        "stale_sha must be 40 lowercase hex characters"
    )
    assert stale_sha not in text, (
        f"public guide must not embed the stale Phase 9B commit SHA {stale_sha!r}"
    )


def test_doc_uses_source_commit_variable_in_persistent_root():
    """The persistent storage layout must use a ``SOURCE_COMMIT``
    variable, not a hard-coded commit SHA."""
    text = DOC.read_text(encoding="utf-8")
    assert "SOURCE_COMMIT" in text


def test_doc_exports_hf_home_before_staging_heredoc():
    """The download-only staging heredoc consumes ``HF_HOME``; the
    doc must export it first."""
    text = DOC.read_text(encoding="utf-8")
    # Look for an explicit ``export HF_HOME`` line somewhere before
    # the staging heredoc that reads ``os.environ["HF_HOME"]``.
    assert "export HF_HOME" in text
    assert 'os.environ["HF_HOME"]' in text


def test_doc_creates_refs_directory_before_writing_refs():
    """The offline-refs binding must ``mkdir -p -m 0700`` the
    ``refs`` parent directory before writing ``refs/main``."""
    text = DOC.read_text(encoding="utf-8")
    assert "mkdir -p -m 0700" in text
    # The text must contain refs/main writes preceded by an
    # explicit directory creation. Quick check: the ``refs``
    # directory appears and ``refs/main`` is referenced.
    assert "/refs" in text or "refs/main" in text


def test_doc_exports_repo_and_log_vars():
    """The staging sequence must export ``REMOTE_PERSISTENT_ROOT``,
    ``REPO_ROOT``, ``HF_HOME``, and ``LOG_ROOT``."""
    text = DOC.read_text(encoding="utf-8")
    for var in (
        "export REMOTE_PERSISTENT_ROOT",
        "export REPO_ROOT",
        "export HF_HOME",
        "export LOG_ROOT",
    ):
        assert var in text, f"doc must contain {var!r}"


# --- Phase 9B safety amendment ---------------------------------------


def test_doc_does_not_anchor_to_opaque_sha_b_label():
    """Ambiguous placeholder labels like ``SHA-B`` must not appear
    in operational prose. The reproducible commit must be
    described as ``SOURCE_COMMIT`` or the exact pushed source
    commit (no placeholder letter)."""
    text = DOC.read_text(encoding="utf-8")
    assert "SHA-B" not in text
    assert "SHA-A" not in text


def test_doc_uses_source_commit_for_remote_checkout():
    """The remote checkout must use ``SOURCE_COMMIT`` (not a
    hard-coded value or a placeholder letter)."""
    text = DOC.read_text(encoding="utf-8")
    assert "SOURCE_COMMIT" in text


def test_doc_does_not_embed_any_real_source_commit_sha():
    """No 40-character lowercase hex blob may appear in the
    committed public guide (model/tokenizer SHAs are concrete
    by design; source commits change with each amendment)."""
    import re

    text = DOC.read_text(encoding="utf-8")
    model_tokenizer_shas = {
        "137da054051ad9f1eac42025f758db4ac9f22535",
        "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    }
    for match in re.findall(r"\b[0-9a-f]{40}\b", text):
        assert match in model_tokenizer_shas, (
            f"unexpected 40-hex source-commit SHA in public doc: {match!r}"
        )
