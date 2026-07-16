# Security Policy

## Supported versions

This project is pre-1.0 and in active development. Only the current `main`
branch is supported. Security fixes, when needed, are applied to `main`.

| Version | Supported |
|---------|-----------|
| `main`  | ✅ Yes    |
| `< 0.1` | ❌ No     |

## Reporting a vulnerability

Please report security issues privately using
[GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
for this repository. Do not open a public issue for security
vulnerabilities.

Concrete details to include:
- A description of the affected behavior.
- Steps to reproduce.
- The potential impact.

There is no bug-bounty program at this time.

## Prohibited behavior

Do **not** post credentials, API tokens, Hugging Face tokens, or other
secrets in public issues, pull requests, or discussions. If you accidentally
share a secret, rotate it immediately and report the disclosure privately.

## Notes on data

This pipeline only reads OSM/Wikipedia/Wikivoyage-derived data and, in Hub
mode, performs a read-only snapshot acquisition. It does not publish or
upload any dataset. Source-content licensing (Wikipedia/Wikivoyage) and
attribution remain the responsibility of downstream consumers, as recorded
in the produced dataset metadata.
