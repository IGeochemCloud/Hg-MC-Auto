#!/usr/bin/env bash
# One-command release to GitHub (tag + Release) and PyPI.
#
# NOTE: The recommended publishing path is now the GitHub Actions workflow
# (.github/workflows/pypi-publish.yml), which uploads to PyPI automatically
# via Trusted Publishing when a GitHub Release is published. Do NOT run this
# script for the same version you publish through the workflow, or the package
# will be uploaded to PyPI twice.
#
# This script remains available for local validation and manual publishing.
#
# Usage:
#   ./release.sh 0.1.1 "release 0.1.1"
#
# The PyPI token is read from the local .env file (gitignored).
# Create it by copying .env.example to .env and filling in your token.
set -euo pipefail

VERSION="${1:-}"
MESSAGE="${2:-release ${VERSION:-unknown}}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version> [message]" >&2
  echo "Example: $0 0.1.1 \"release 0.1.1\"" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load PYPI_TOKEN from .env if present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "PYPI_TOKEN is not set. Copy .env.example to .env and fill in your token." >&2
  exit 1
fi

# Activate the project virtual environment if it exists.
if [[ -d hg-mc-auto/bin ]]; then
  # shellcheck disable=SC1091
  source hg-mc-auto/bin/activate
fi

echo "==> Dry-run validation"
hgmc-release sync --version "$VERSION" --dry-run

echo "==> Publishing $VERSION to GitHub and PyPI"
hgmc-release sync --version "$VERSION" --message "$MESSAGE"

echo "==> Done. Check GitHub Releases and PyPI for $VERSION."