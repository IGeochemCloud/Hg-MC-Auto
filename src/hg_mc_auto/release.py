from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def run_cmd(cmd: str | list[str], *, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    if isinstance(cmd, str):
        shell = True
        final_cmd = cmd
    else:
        shell = False
        final_cmd = cmd

    result = subprocess.run(
        final_cmd,
        cwd=str(cwd or PROJECT_ROOT),
        shell=shell,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {final_cmd}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result


def check_environment() -> dict:
    info = {
        "project_root": str(PROJECT_ROOT),
        "python": sys.executable,
        "git": shutil.which("git"),
        "twine": shutil.which("twine"),
        "python_build": shutil.which("python -m build"),
        "github_remote": "",
    }
    try:
        result = run_cmd("git remote -v", check=False)
        if result.returncode == 0:
            info["github_remote"] = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    except Exception:
        pass
    return info


def read_project_version() -> str:
    """Read the version declared in pyproject.toml."""
    if not PYPROJECT.exists():
        raise RuntimeError(f"pyproject.toml not found: {PYPROJECT}")
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find a version field in pyproject.toml")
    return match.group(1).strip()


def git_has_uncommitted_changes() -> bool:
    result = run_cmd(["git", "status", "--porcelain"], check=False)
    return bool(result.stdout.strip())


def git_tag_exists(tag: str) -> bool:
    result = run_cmd(["git", "tag", "-l", tag], check=False)
    return tag in result.stdout.split()


def require_pypi_credentials() -> None:
    if not os.environ.get("PYPI_TOKEN"):
        raise RuntimeError(
            "PYPI_TOKEN is not set. Create a token at https://pypi.org/manage/account/token/ "
            "and export it before publishing."
        )
    if shutil.which("twine") is None:
        raise RuntimeError("twine is not installed. Run: pip install twine")


def build_distribution() -> dict:
    run_cmd("python -m build")
    dist_dir = PROJECT_ROOT / "dist"
    files = [str(p.name) for p in sorted(dist_dir.iterdir())] if dist_dir.exists() else []
    return {"dist_dir": str(dist_dir), "files": files}


def publish_to_pypi() -> dict:
    require_pypi_credentials()
    run_cmd("python -m twine upload dist/* --repository pypi")
    return {"status": "published", "repository": "pypi"}


def create_github_release(version: str, message: str) -> dict:
    """Create a GitHub Release with the built artifacts using the gh CLI."""
    if shutil.which("gh") is None:
        return {"status": "skipped", "reason": "gh CLI not installed"}
    dist_files = " ".join(str(p) for p in sorted((PROJECT_ROOT / "dist").glob("*")))
    run_cmd(
        f"gh release create {version} {dist_files} --title v{version} --notes {message!r}",
        check=False,
    )
    return {"status": "created", "tag": version}


def sync_github_and_pypi(version: str, message: str, *, dry_run: bool = False) -> dict:
    if not (PROJECT_ROOT / ".git").exists():
        raise RuntimeError("This project is not a git repository. Initialize git first.")

    declared = read_project_version()
    if version != declared:
        raise RuntimeError(
            f"Version mismatch: requested '{version}' but pyproject.toml declares '{declared}'. "
            "Update pyproject.toml first."
        )
    if git_tag_exists(version):
        raise RuntimeError(f"Git tag '{version}' already exists. Bump the version first.")
    if git_has_uncommitted_changes():
        raise RuntimeError(
            "Working tree has uncommitted changes. Commit or stash them before a release sync."
        )
    require_pypi_credentials()

    steps = [
        ("build", ["python", "-m", "build"]),
        ("tag", ["git", "tag", version]),
        ("push", ["git", "push", "origin", "HEAD", "--tags"]),
        ("publish", ["python", "-m", "twine", "upload", "dist/*", "--repository", "pypi"]),
    ]
    if dry_run:
        return {
            "status": "dry_run",
            "version": version,
            "tag": version,
            "steps": [name for name, _ in steps],
        }

    for name, cmd in steps:
        run_cmd(cmd)

    release = create_github_release(version, message)
    return {"status": "synced", "version": version, "tag": version, "github_release": release}


def app(action: str | list[str] | None = None) -> None:
    if action is None:
        action = "help"

    if isinstance(action, str):
        args = action.split()
    else:
        args = action

    parser = argparse.ArgumentParser(description="Release helpers for hg-mc-auto")
    parser.add_argument("action", nargs="?", choices=["check", "build", "publish", "sync", "help"], default="help")
    parser.add_argument("--version", help="Version tag to sync, e.g. 0.1.1")
    parser.add_argument("--message", default="chore: release", help="Git commit message for the release sync")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the planned steps without executing them.")
    parsed = parser.parse_args(args)

    if parsed.action == "help":
        print("Release actions: check | build | publish | sync")
        print("Examples:")
        print("  hgmc-release check")
        print("  hgmc-release build")
        print("  hgmc-release publish")
        print("  hgmc-release sync --version 0.1.1 --message 'release 0.1.1'")
        print("  hgmc-release sync --version 0.1.1 --dry-run")
        return

    if parsed.action == "check":
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
        return

    if parsed.action == "build":
        print(json.dumps(build_distribution(), ensure_ascii=False, indent=2))
        return

    if parsed.action == "publish":
        print(json.dumps(publish_to_pypi(), ensure_ascii=False, indent=2))
        return

    if parsed.action == "sync":
        if not parsed.version:
            raise SystemExit("--version is required for sync")
        print(json.dumps(
            sync_github_and_pypi(parsed.version, parsed.message, dry_run=parsed.dry_run),
            ensure_ascii=False,
            indent=2,
        ))
        return

    raise SystemExit(f"Unsupported action: {parsed.action}")


def main() -> None:
    app(sys.argv[1:])


if __name__ == "__main__":
    main()
