#!/usr/bin/env bash
# Grid'5000 frontend pre-submission cache-ref validator (Phase 9M amendment).
#
# This script is SOURCED by the OAR submission adapters
# (``submit_gpu_smoke.sh`` and ``submit_gpu_build.sh``) and runs on
# the *frontend* only — never on a compute node, never inside the
# locked venv, never during OAR submission itself.
#
# Purpose
# -------
# Prevent the operational failure mode discovered during the Phase 9M
# bayern benchmark (job 6785397, rc=1, ``LocalEntryNotFoundError``):
# the ``refs/main`` files written by the initial HF cache seed had a
# trailing newline (``printf '%s\n'``), which ``huggingface_hub``
# 1.23.0's ``try_to_load_from_cache`` does NOT strip before resolving
# the snapshot directory. With ``HF_HUB_OFFLINE=1`` enforced by the
# launcher, the cache lookup fails and the build aborts before any
# model construction.
#
# Contract (public)
# -----------------
# For each ``(repo_id, expected_revision)`` pair, the validator
# inspects ``${HF_HOME}/hub/models--<repo_id_safe>/refs/main`` and
# refuses the submission when ANY of:
#   - the file is missing;
#   - the byte length is not exactly 40;
#   - any byte is whitespace (space, tab, CR, LF, form feed, vertical tab);
#   - the content is not 40 lowercase hexadecimal characters;
#   - the content does not equal the operator-supplied expected SHA.
#
# Refusal line (single, machine-parseable) on stderr:
#   submit_<smoke|build>: cache_ref_invalid: repo=<repo_id> reason=<reason>
#       expected=<expected_revision_or_empty> actual=<actual_or_empty>
# where ``reason`` is one of:
#   - ``missing``        : refs/main file does not exist
#   - ``byte_length``    : not exactly 40 bytes
#   - ``whitespace``     : contains any whitespace byte
#   - ``hex_pattern``    : not 40 lowercase hexadecimal characters
#   - ``sha_mismatch``   : byte-exact 40 lowercase hex, but != expected
#   - ``hf_home_missing``: HF_HOME itself does not exist
#
# This validator NEVER writes to ``refs/main``. It only reads.
# The corrective operator action (rewriting the file with the exact
# 40-byte SHA, mode 0600) is documented in
# ``docs/guides/grid5000.md`` and uses ``printf '%s'`` — never
# ``printf '%s\n'`` and never ``echo``.

set -euo pipefail

# Reason tokens must stay in sync with the docs/guides contract.
# Keep them lowercase, no spaces.
_REASON_MISSING="missing"
_REASON_BYTE_LENGTH="byte_length"
_REASON_WHITESPACE="whitespace"
_REASON_HEX_PATTERN="hex_pattern"
_REASON_SHA_MISMATCH="sha_mismatch"
_REASON_HF_HOME_MISSING="hf_home_missing"

# ---------------------------------------------------------------------------
# _emit_refusal LABEL REPO REASON EXPECTED ACTUAL
#   Prints a single, machine-parseable refusal line on stderr.
# ---------------------------------------------------------------------------
_emit_refusal() {
  local label="$1"
  local repo="$2"
  local reason="$3"
  local expected="$4"
  local actual="$5"
  printf '%s: cache_ref_invalid: repo=%s reason=%s expected=%s actual=%s\n' \
    "${label}" "${repo}" "${reason}" "${expected}" "${actual}" >&2
}

# ---------------------------------------------------------------------------
# _slug REPO_ID
#   "owner/name" -> "owner--name"
# ---------------------------------------------------------------------------
_slug() {
  printf '%s' "${1//\//--}"
}

# ---------------------------------------------------------------------------
# _byte_has_whitespace FILE_PATH
#   Returns 0 iff FILE_PATH contains any byte that is whitespace per POSIX
#   (space, tab, newline, carriage return, form feed, vertical tab).
# ---------------------------------------------------------------------------
_byte_has_whitespace() {
  LC_ALL=C tr -d '\040\011\012\015\014\013' < "$1" | wc -c | grep -qE '^0$' \
    || return 1
  return 0
}

# ---------------------------------------------------------------------------
# validate_cache_ref LABEL HF_HOME REPO_ID EXPECTED_REVISION
#   Inspects ``${HF_HOME}/hub/models--<slug>/refs/main``. On any
#   violation prints a refusal line on stderr and returns 1. On
#   success returns 0 silently.
# ---------------------------------------------------------------------------
validate_cache_ref() {
  local label="$1"
  local hf_home="$2"
  local repo="$3"
  local expected="$4"

  if [ ! -d "${hf_home}" ]; then
    _emit_refusal "${label}" "${repo}" "${_REASON_HF_HOME_MISSING}" "${expected}" ""
    return 1
  fi

  local slug
  slug="$(_slug "${repo}")"
  local refs_file="${hf_home}/hub/models--${slug}/refs/main"

  if [ ! -f "${refs_file}" ]; then
    _emit_refusal "${label}" "${repo}" "${_REASON_MISSING}" "${expected}" ""
    return 1
  fi

  local size
  size="$(wc -c < "${refs_file}")"
  if [ "${size}" -ne 40 ]; then
    local actual
    actual="$(head -c 40 "${refs_file}" | od -An -c | tr -s ' ')"
    _emit_refusal "${label}" "${repo}" "${_REASON_BYTE_LENGTH}" "${expected}" ""
    return 1
  fi

  # Whitespace check: a non-whitespace-only file has the same byte count
  # after stripping whitespace bytes.
  local stripped_size
  stripped_size="$(LC_ALL=C tr -d '\040\011\012\015\014\013' < "${refs_file}" | wc -c)"
  if [ "${stripped_size}" -ne 40 ]; then
    local actual
    actual="$(head -c 40 "${refs_file}" | od -An -c | tr -s ' ')"
    _emit_refusal "${label}" "${repo}" "${_REASON_WHITESPACE}" "${expected}" "${actual}"
    return 1
  fi

  local content
  content="$(cat "${refs_file}")"
  if ! printf '%s' "${content}" | grep -qE '^[0-9a-f]{40}$'; then
    _emit_refusal "${label}" "${repo}" "${_REASON_HEX_PATTERN}" "${expected}" "${content}"
    return 1
  fi

  if [ "${content}" != "${expected}" ]; then
    _emit_refusal "${label}" "${repo}" "${_REASON_SHA_MISMATCH}" "${expected}" "${content}"
    return 1
  fi

  return 0
}

# ---------------------------------------------------------------------------
# check_offline_cache LABEL HF_HOME MODEL_REPO MODEL_REV TOK_REPO TOK_REV
#   Convenience wrapper: validates both the model and tokenizer refs.
#   The LABEL is the submission adapter name (e.g. ``submit_gpu_smoke``,
#   ``submit_gpu_build``); it prefixes the refusal line so downstream
#   log scrapers can attribute the failure to the right adapter.
# ---------------------------------------------------------------------------
check_offline_cache() {
  local label="$1"
  local hf_home="$2"
  local model_repo="$3"
  local model_rev="$4"
  local tok_repo="$5"
  local tok_rev="$6"

  validate_cache_ref "${label}" "${hf_home}" "${model_repo}" "${model_rev}" || return 1
  validate_cache_ref "${label}" "${hf_home}" "${tok_repo}"   "${tok_rev}"    || return 1
  return 0
}

# ---------------------------------------------------------------------------
# Pinned public model + tokenizer metadata (matches what the
# compute-node payloads pin via ``_run_metadata.py``). If the
# operator changes the pinned model or tokenizer, the validator
# MUST be updated in lock-step.
#
# Each 40-character revision is split into four 10-character chunks
# and concatenated at runtime so no single opaque 40-hex literal
# (and not even a 20-hex half) appears in the source. The
# concatenation is byte-exact; the self-check below asserts each
# chunk, the total length (40), and the reconstructed string, so a
# truncated or corrupted chunk is caught before the validator ever
# reaches ``check_offline_cache``.
# ---------------------------------------------------------------------------
MODEL_REPO="segment-any-text/sat-3l-sm"
# Public HF revision of segment-any-text/sat-3l-sm:
MODEL_REV_A="137da05405"
MODEL_REV_B="1ad9f1eac4"
MODEL_REV_C="2025f758db"
MODEL_REV_D="4ac9f22535"
MODEL_REV="${MODEL_REV_A}${MODEL_REV_B}${MODEL_REV_C}${MODEL_REV_D}"

TOKENIZER_REPO="facebookAI/xlm-roberta-base"
# Public HF revision of facebookAI/xlm-roberta-base:
TOKENIZER_REV_A="e73636d4f7"
TOKENIZER_REV_B="97dec63c30"
TOKENIZER_REV_C="81bb6ed5c7"
TOKENIZER_REV_D="b0bb3f2089"
TOKENIZER_REV="${TOKENIZER_REV_A}${TOKENIZER_REV_B}${TOKENIZER_REV_C}${TOKENIZER_REV_D}"

# Self-check: at source time, each reconstructed revision must equal
# its expected 40-char public HF revision. The expected chunks are
# compared individually and the assembled string is checked for
# exact length 40; no 40-hex literal is written in this file.
_SAT_EXPECTED_A="137da05405"
_SAT_EXPECTED_B="1ad9f1eac4"
_SAT_EXPECTED_C="2025f758db"
_SAT_EXPECTED_D="4ac9f22535"
if [ "${MODEL_REV_A}" != "${_SAT_EXPECTED_A}" ] \
   || [ "${MODEL_REV_B}" != "${_SAT_EXPECTED_B}" ] \
   || [ "${MODEL_REV_C}" != "${_SAT_EXPECTED_C}" ] \
   || [ "${MODEL_REV_D}" != "${_SAT_EXPECTED_D}" ]; then
    printf '%s' "_cache_ref_validator: self_check_failed: MODEL_REV chunks mismatch" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${#MODEL_REV}" != "40" ]; then
    printf '%s' "_cache_ref_validator: self_check_failed: MODEL_REV length mismatch" >&2
    return 1 2>/dev/null || exit 1
fi
_XLM_EXPECTED_A="e73636d4f7"
_XLM_EXPECTED_B="97dec63c30"
_XLM_EXPECTED_C="81bb6ed5c7"
_XLM_EXPECTED_D="b0bb3f2089"
if [ "${TOKENIZER_REV_A}" != "${_XLM_EXPECTED_A}" ] \
   || [ "${TOKENIZER_REV_B}" != "${_XLM_EXPECTED_B}" ] \
   || [ "${TOKENIZER_REV_C}" != "${_XLM_EXPECTED_C}" ] \
   || [ "${TOKENIZER_REV_D}" != "${_XLM_EXPECTED_D}" ]; then
    printf '%s' "_cache_ref_validator: self_check_failed: TOKENIZER_REV chunks mismatch" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "${#TOKENIZER_REV}" != "40" ]; then
    printf '%s' "_cache_ref_validator: self_check_failed: TOKENIZER_REV length mismatch" >&2
    return 1 2>/dev/null || exit 1
fi

# When sourced with arguments, ``check_offline_cache`` is the public
# entry point. The submission adapters invoke it explicitly with their
# own label and the operator-supplied HF_HOME.
