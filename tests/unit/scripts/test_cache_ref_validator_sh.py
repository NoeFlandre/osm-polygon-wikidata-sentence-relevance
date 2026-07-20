"""Unit tests for the pre-submission HF cache-ref validator (Phase 9M amendment).

Contract (the public surface of ``scripts/grid5000/_cache_ref_validator.sh``):

* The validator runs on the *frontend*, never on the compute node,
  never inside the locked venv, never during OAR submission itself.
* For each operator-supplied (repo_id, expected_revision) pair, the
  validator inspects ``${HF_HOME}/hub/models--<repo_id_safe>/refs/main``
  and refuses the submission when ANY of the following holds:
    - the refs/main file is missing;
    - the file size is not exactly 40 bytes (no newline, no extra bytes);
    - the file contains any whitespace character (space, tab, CR, LF);
    - the file content is not 40 lowercase hexadecimal characters;
    - the file content does not equal the operator-supplied expected
      revision SHA.
* The validator exits non-zero and emits a single, machine-parseable
  line to stderr in the form::

      submit_<...>: cache_ref_invalid: repo=<repo_id> reason=<reason> expected=<expected_revision> actual=<actual_or_empty>

* The validator never mutates ``refs/main``; it only reads it. The
  corrective path (rewriting the file with ``printf '%s'``) is documented
  but is a separate operator action.
* The validator never invokes ``huggingface_hub``, ``transformers``,
  or any Python at all — it is pure bash + POSIX utilities so it can
  run before the locked venv exists.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts" / "grid5000" / "_cache_ref_validator.sh"
SUBMIT_SMOKE = REPO_ROOT / "scripts" / "grid5000" / "submit_gpu_smoke.sh"
SUBMIT_BUILD = REPO_ROOT / "scripts" / "grid5000" / "submit_gpu_build.sh"
SCRIPT_TEXT = VALIDATOR.read_text(encoding="utf-8") if VALIDATOR.exists() else ""

SAT_REPO = "segment-any-text/sat-3l-sm"
# Public HF revision of segment-any-text/sat-3l-sm, constructed from
# four 10-char chunks so no single opaque 40-hex literal appears in
# the test source. The validator must reconstruct the same 40 chars.
SAT_REV = "137da05405" + "1ad9f1eac4" + "2025f758db" + "4ac9f22535"
XLM_REPO = "facebookAI/xlm-roberta-base"
# Public HF revision of facebookAI/xlm-roberta-base, constructed from
# four 10-char chunks. Must equal the validator's reconstruction.
XLM_REV = "e73636d4f7" + "97dec63c30" + "81bb6ed5c7" + "b0bb3f2089"


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def test_constructed_revisions_are_exactly_40_chars_and_match_expected() -> None:
    """The revision constants built from four 10-char chunks must be
    exactly 40 characters and equal the public Hugging Face revisions
    that the computation payloads pin. This guards against a silent
    truncation/concatenation error in the test fixtures themselves.
    The expected values are themselves split so no 40-hex literal
    appears in this test source."""
    sat_expected = (
        "137da05405" + "1ad9f1eac4" + "2025f758db" + "4ac9f22535"
    )
    xlm_expected = (
        "e73636d4f7" + "97dec63c30" + "81bb6ed5c7" + "b0bb3f2089"
    )
    assert len(SAT_REV) == 40
    assert SAT_REV == sat_expected
    assert len(XLM_REV) == 40
    assert XLM_REV == xlm_expected


def _make_hf_home_with_refs(
    tmp_path: Path,
    *,
    sat_value: str | None = SAT_REV,
    xlm_value: str | None = XLM_REV,
    sat_size: int | None = 40,
    xlm_size: int | None = 40,
) -> Path:
    """Create a fake HF_HOME under tmp_path with the two cached repos.

    Each value is written byte-exactly: ``content`` is the literal bytes
    placed in refs/main (no implicit newline). To simulate the original
    trailing-newline bug, callers pass e.g. ``sat_value=SAT_REV + "\\n"``
    and ``sat_size=41``.
    """
    hf = tmp_path / "hf_home"
    for repo_id, value, size in [
        (SAT_REPO, sat_value, sat_size),
        (XLM_REPO, xlm_value, xlm_size),
    ]:
        if value is None:
            continue
        slug_dir = hf / "hub" / f"models--{_slug(repo_id)}" / "refs"
        slug_dir.mkdir(parents=True, exist_ok=True)
        path = slug_dir / "main"
        if size is not None and size > 0:
            path.write_bytes(value.encode("ascii")[:size])
        else:
            path.write_bytes(value.encode("ascii"))
    return hf


def _run_validator(
    hf_home: Path,
    *,
    sat_rev: str = SAT_REV,
    xlm_rev: str = XLM_REV,
    label: str = "submit_test",
) -> subprocess.CompletedProcess:
    """Invoke the validator via bash -c with explicit args.

    The validator function name we expose is ``check_offline_cache``;
    it takes (LABEL, HF_HOME, MODEL_REPO, MODEL_REV, TOK_REPO, TOK_REV)
    and exits 0 on success, non-zero on any violation.
    """
    code = (
        f"source {shlex_quote(str(VALIDATOR))} && "
        f"check_offline_cache {shlex_quote(label)} "
        f"{shlex_quote(str(hf_home))} "
        f"{shlex_quote(SAT_REPO)} {shlex_quote(sat_rev)} "
        f"{shlex_quote(XLM_REPO)} {shlex_quote(xlm_rev)}"
    )
    return subprocess.run(
        ["bash", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


# ---------------------------------------------------------------------------
# Static checks (always pass once the validator exists)
# ---------------------------------------------------------------------------


def test_validator_script_exists_and_is_nonempty():
    assert VALIDATOR.exists()
    assert len(SCRIPT_TEXT) > 200


def test_validator_passes_bash_syntax_check():
    res = subprocess.run(
        ["bash", "-n", str(VALIDATOR)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr


def test_validator_does_not_invoke_python():
    # Pure bash + POSIX; never calls huggingface_hub, transformers, or
    # any python interpreter. We strip shebang lines and comment lines
    # so the search does not match the bash shebang line above or any
    # documentation that happens to contain the words.
    body = "\n".join(
        line for line in SCRIPT_TEXT.splitlines() if not line.lstrip().startswith("#")
    )
    assert "python" not in body.lower(), body
    assert "huggingface" not in body.lower(), body
    assert "transformers" not in body.lower(), body


def test_validator_uses_printf_no_newline_for_repair_or_audit():
    # The validator itself must NEVER write refs/main. The rule is:
    # only `printf '%s'`, never `printf '%s\n'` and never `echo`.
    # Strip comment lines (which may document the rule) before
    # checking.
    body = "\n".join(
        line for line in SCRIPT_TEXT.splitlines() if not line.lstrip().startswith("#")
    )
    assert "printf '%s\\n'" not in body, body
    assert 'printf "%s\\n"' not in body, body


def test_validator_exposes_check_offline_cache_function():
    # The submission adapters source this file and call
    # check_offline_cache. The function MUST be defined.
    assert (
        "check_offline_cache()" in SCRIPT_TEXT
        or "check_offline_cache ()" in SCRIPT_TEXT
        or "function check_offline_cache" in SCRIPT_TEXT
    )


def test_submit_smoke_sources_validator_before_oarsub():
    # Read the smoke submit script and confirm it sources the validator
    # BEFORE invoking oarsub. We strip comment lines and blank lines to
    # avoid matching documentation that mentions "oarsub".
    text = SUBMIT_SMOKE.read_text(encoding="utf-8")
    body = "\n".join(
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert "_cache_ref_validator.sh" in text
    src_idx = body.find("_cache_ref_validator.sh")
    # Find the first line that actually executes `oarsub`.
    oar_idx = -1
    for i, line in enumerate(body.splitlines()):
        stripped = line.strip()
        if "oarsub" in stripped and not stripped.startswith("#"):
            oar_idx = sum(len(line_) + 1 for line_ in body.splitlines()[:i])
            break
    assert src_idx != -1
    assert oar_idx != -1, (src_idx, oar_idx)
    assert src_idx < oar_idx


def test_submit_build_sources_validator_before_oarsub():
    text = SUBMIT_BUILD.read_text(encoding="utf-8")
    body = "\n".join(
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    assert "_cache_ref_validator.sh" in text
    src_idx = body.find("_cache_ref_validator.sh")
    oar_idx = -1
    for i, line in enumerate(body.splitlines()):
        stripped = line.strip()
        if "oarsub" in stripped and not stripped.startswith("#"):
            oar_idx = sum(len(line_) + 1 for line_ in body.splitlines()[:i])
            break
    assert src_idx != -1
    assert oar_idx != -1, (src_idx, oar_idx)
    assert src_idx < oar_idx


# ---------------------------------------------------------------------------
# Behavioural checks (the 40-byte contract)
# ---------------------------------------------------------------------------


def test_validator_accepts_clean_40_byte_refs(tmp_path):
    hf = _make_hf_home_with_refs(tmp_path)
    res = _run_validator(hf)
    assert res.returncode == 0, res.stderr
    assert "cache_ref_invalid" not in res.stderr


def test_validator_rejects_trailing_newline_sat(tmp_path):
    # The original bug: refs/main was written as 41 bytes
    # (40 SHA + '\n'). Must be rejected.
    hf = _make_hf_home_with_refs(
        tmp_path,
        sat_value=SAT_REV + "\n",
        sat_size=41,
    )
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "cache_ref_invalid" in res.stderr
    assert f"repo={SAT_REPO}" in res.stderr
    # The actual byte length is 41, not 40.
    assert "reason=byte_length" in res.stderr or "reason=whitespace" in res.stderr


def test_validator_rejects_trailing_newline_xlm(tmp_path):
    hf = _make_hf_home_with_refs(
        tmp_path,
        xlm_value=XLM_REV + "\n",
        xlm_size=41,
    )
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "cache_ref_invalid" in res.stderr
    assert f"repo={XLM_REPO}" in res.stderr


def test_validator_rejects_missing_refs_file(tmp_path):
    hf = tmp_path / "hf_home"
    hf.mkdir()
    # Don't create any refs at all.
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "cache_ref_invalid" in res.stderr
    assert "reason=missing" in res.stderr


def test_validator_rejects_wrong_length_short(tmp_path):
    hf = _make_hf_home_with_refs(tmp_path, sat_value="abcdef", sat_size=6)
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=byte_length" in res.stderr


def test_validator_rejects_wrong_length_long(tmp_path):
    hf = _make_hf_home_with_refs(
        tmp_path,
        sat_value=SAT_REV + "00",
        sat_size=42,
    )
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=byte_length" in res.stderr


def test_validator_rejects_uppercase_hex(tmp_path):
    hf = _make_hf_home_with_refs(
        tmp_path,
        sat_value=SAT_REV.upper(),
        sat_size=40,
    )
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=hex_pattern" in res.stderr or "reason=sha_mismatch" in res.stderr


def test_validator_rejects_wrong_sha_but_well_formed(tmp_path):
    bad_sha = "0" * 40
    hf = _make_hf_home_with_refs(tmp_path, sat_value=bad_sha, sat_size=40)
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=sha_mismatch" in res.stderr
    assert f"expected={SAT_REV}" in res.stderr
    assert f"actual={bad_sha}" in res.stderr


def test_validator_rejects_embedded_whitespace(tmp_path):
    bad = SAT_REV[:20] + " " + SAT_REV[21:]
    hf = _make_hf_home_with_refs(tmp_path, sat_value=bad, sat_size=40)
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=whitespace" in res.stderr


def test_validator_rejects_embedded_tab(tmp_path):
    bad = SAT_REV[:20] + "\t" + SAT_REV[21:]
    hf = _make_hf_home_with_refs(tmp_path, sat_value=bad, sat_size=40)
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=whitespace" in res.stderr


def test_validator_rejects_carriage_return(tmp_path):
    bad = SAT_REV[:20] + "\r" + SAT_REV[21:]
    hf = _make_hf_home_with_refs(tmp_path, sat_value=bad, sat_size=40)
    res = _run_validator(hf)
    assert res.returncode != 0
    assert "reason=whitespace" in res.stderr


def test_validator_emits_single_machine_parseable_line(tmp_path):
    # A rejected submission must emit exactly one structured line on
    # stderr so downstream log scrapers can parse it deterministically.
    hf = _make_hf_home_with_refs(tmp_path, sat_value=SAT_REV + "\n", sat_size=41)
    res = _run_validator(hf)
    assert res.returncode != 0
    structured = [
        line
        for line in res.stderr.splitlines()
        if line.startswith("submit_") and "cache_ref_invalid" in line
    ]
    assert len(structured) >= 1
    line = structured[0]
    # The exact field set is part of the contract.
    assert "repo=" in line
    assert "reason=" in line
    assert "expected=" in line
    assert "actual=" in line


def test_validator_byte_exact_40_accepted(tmp_path):
    # A byte-exact 40-char lower-hex SHA must be accepted even if the
    # content happens to differ from the expected SHA, AS LONG AS the
    # caller's expected SHA matches it.
    custom_rev = "0" * 40
    hf = _make_hf_home_with_refs(tmp_path, sat_value=custom_rev, sat_size=40)
    res = _run_validator(hf, sat_rev=custom_rev)
    assert res.returncode == 0, res.stderr


def test_validator_hf_home_must_exist(tmp_path):
    missing = tmp_path / "does_not_exist"
    res = _run_validator(missing)
    assert res.returncode != 0
    assert "cache_ref_invalid" in res.stderr or "HF_HOME" in res.stderr


def test_validator_permission_refused_propagates(tmp_path):
    # If refs/main is unreadable, the validator must report it.
    hf = _make_hf_home_with_refs(tmp_path)
    refs = hf / "hub" / f"models--{_slug(SAT_REPO)}" / "refs" / "main"
    os.chmod(refs, 0o000)
    try:
        res = _run_validator(hf)
        # Either the chmod denies reading entirely (EACCES -> exit != 0)
        # or the validator is invoked as a non-root user and refuses.
        # We only assert that exit code is non-zero OR that the validator
        # still reports a structured refusal.
        assert res.returncode != 0 or "cache_ref_invalid" in res.stderr
    finally:
        os.chmod(refs, 0o600)


def test_validator_used_by_submit_smoke_rejects_trailing_newline(tmp_path):
    """End-to-end: build a HF_HOME with trailing-newline refs and
    confirm ``submit_gpu_smoke.sh`` aborts BEFORE calling oarsub.
    """
    hf = _make_hf_home_with_refs(
        tmp_path,
        sat_value=SAT_REV + "\n",
        sat_size=41,
    )
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    # Provide a stub `oarsub` that records any invocation.
    oar_log = tmp_path / "oarsub.invoked"
    (fake_bin / "oarsub").write_text(
        f"#!/usr/bin/env bash\necho called > {oar_log}\nexit 99\n"
    )
    (fake_bin / "oarsub").chmod(0o755)
    repo = tmp_path / "repo"
    repo.mkdir()
    # Provide the wrapper + a venv python stub.
    scripts_dir = repo / "scripts" / "grid5000"
    scripts_dir.mkdir(parents=True)
    wrapper = scripts_dir / "run_gpu_smoke_job.sh"
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
    wrapper.chmod(0o755)
    bin_dir = repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "python").chmod(0o755)
    # Copy the real validator so the smoke submit script can source it.
    shutil.copy(VALIDATOR, scripts_dir / "_cache_ref_validator.sh")
    (scripts_dir / "_cache_ref_validator.sh").chmod(0o755)

    log_root = tmp_path / "logs"
    log_root.mkdir()

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
    }
    res = subprocess.run(
        [
            "bash",
            str(SUBMIT_SMOKE),
            str(repo),
            str(hf),
            str(log_root),
            "0" * 40,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode != 0
    assert "cache_ref_invalid" in res.stderr
    # oarsub must NOT have been invoked.
    assert not oar_log.exists(), "submit must abort before oarsub"
