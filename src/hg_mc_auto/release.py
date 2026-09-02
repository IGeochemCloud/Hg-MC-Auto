from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def build_distribution() -> dict:
    run_cmd("python -m build")
    dist_dir = PROJECT_ROOT / "dist"
    files = [str(p.name) for p in sorted(dist_dir.iterdir())] if dist_dir.exists() else []
    return {"dist_dir": str(dist_dir), "files": files}


def publish_to_pypi() -> dict:
    if not os.environ.get("PYPI_TOKEN"):
        raise RuntimeError("Set PYPI_TOKEN before publishing to PyPI.")
    run_cmd("python -m twine upload dist/* --repository pypi")
    return {"status": "published", "repository": "pypi"}


def sync_github_and_pypi(version: str, message: str) -> dict:
    if not (PROJECT_ROOT / ".git").exists():
        raise RuntimeError("This project is not a git repository. Initialize git first.")

    run_cmd(["git", "add", "."])
    run_cmd(["git", "commit", "-m", message], check=False)
    run_cmd(["git", "tag", version])
    run_cmd(["git", "push", "origin", "HEAD", "--tags"])
    run_cmd("python -m build")
    run_cmd("python -m twine upload dist/* --repository pypi")
    return {"status": "synced", "version": version, "tag": version}


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
    parsed = parser.parse_args(args)

    if parsed.action == "help":
        print("Release actions: check | build | publish | sync")
        print("Examples:")
        print("  hgmc-release check")
        print("  hgmc-release build")
        print("  hgmc-release publish")
        print("  hgmc-release sync --version 0.1.1 --message 'release 0.1.1'")
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
        print(json.dumps(sync_github_and_pypi(parsed.version, parsed.message), ensure_ascii=False, indent=2))
        return

    raise SystemExit(f"Unsupported action: {parsed.action}")


def main() -> None:
    app(sys.argv[1:])


if __name__ == "__main__":
    main()
