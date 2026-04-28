#!/usr/bin/env python3
"""Commit only source-like repository changes.

The helper stages and commits code, documentation, tests, notebooks, and small
metadata while excluding local environments, generated data, media, archives,
and model artifacts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import PurePosixPath


BLOCKED_DIR_PARTS = {
    ".git",
    ".ipynb_checkpoints",
    ".venv",
    "__pycache__",
    "env",
    "venv",
}

BLOCKED_SUFFIXES = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pth",
    ".pt",
    ".pyc",
    ".pyo",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".webm",
    ".zip",
}

BLOCKED_PREFIXES = (
    "Data/Countix/video/",
    "Data/LLSP/annotation_cleaned/pose_features/",
    "Data/LLSP/annotation_cleaned/pose_sequences/",
    "Data/LLSP/annotation_cleaned/rgb_resnet18_features/",
    "Data/LLSP/annotation_cleaned/rgb_resnet50_features/",
    "Data/LLSP/annotation_cleaned/squat_features/",
    "artifacts/3_Modeling/T/",
    "artifacts/3_Modeling/training_outputs/",
)


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a Git command and return the completed process."""
    return subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stage and commit source-like changes while excluding generated "
            "or local artifacts."
        )
    )
    parser.add_argument("-m", "--message", help="commit message to use")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be staged and committed, but do not commit",
    )
    return parser.parse_args()


def repo_root() -> str:
    """Return the absolute repository root."""
    return run_git(["rev-parse", "--show-toplevel"]).stdout.strip()


def has_staged_changes() -> bool:
    """Return whether the index already contains staged changes."""
    result = run_git(["diff", "--cached", "--quiet"], check=False)
    return result.returncode != 0


def changed_files() -> list[str]:
    """Return modified, deleted, and untracked files not ignored by Git."""
    result = run_git(
        ["ls-files", "--modified", "--deleted", "--others", "--exclude-standard", "-z"]
    )
    return [path for path in result.stdout.split("\0") if path]


def is_data_video(path: str) -> bool:
    """Return whether a path is under a dataset video directory."""
    parts = PurePosixPath(path).parts
    return len(parts) >= 3 and parts[0] == "Data" and "video" in parts[1:]


def is_blocked_path(path: str) -> bool:
    """Return whether a path should not be committed by this helper."""
    posix_path = PurePosixPath(path)

    if posix_path.name == ".DS_Store":
        return True

    if any(part in BLOCKED_DIR_PARTS for part in posix_path.parts):
        return True

    if is_data_video(path):
        return True

    if any(path.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        return True

    return any(path.endswith(suffix) for suffix in BLOCKED_SUFFIXES)


def allowed_files(paths: list[str]) -> list[str]:
    """Filter changed files down to paths this helper may stage."""
    return [path for path in paths if not is_blocked_path(path)]


def staged_files() -> list[str]:
    """Return names of currently staged files."""
    result = run_git(["diff", "--cached", "--name-only"])
    return [line for line in result.stdout.splitlines() if line]


def stage_files(paths: list[str]) -> None:
    """Stage the provided paths."""
    if paths:
        run_git(["add", "-A", "--", *paths])


def unstage_all() -> None:
    """Clear the index after a dry run or failed guard check."""
    run_git(["restore", "--staged", ":/"])


def print_staged_summary() -> None:
    """Print a concise summary of staged changes."""
    result = run_git(["diff", "--cached", "--name-status"])
    print("Staged files:")
    print(result.stdout, end="")


def commit(message: str) -> None:
    """Create the Git commit."""
    result = run_git(["commit", "-m", message])
    print(result.stdout, end="")


def main() -> int:
    """Run the commit helper."""
    args = parse_args()
    os.chdir(repo_root())

    if not args.message and not args.dry_run:
        print(
            "error: commit message is required unless --dry-run is used",
            file=sys.stderr,
        )
        return 2

    if has_staged_changes():
        print("error: staged changes already exist.", file=sys.stderr)
        print(
            "Commit or unstage them before using this helper, so it can guard "
            "the staged set.",
            file=sys.stderr,
        )
        return 1

    stage_files(allowed_files(changed_files()))

    blocked_files = [path for path in staged_files() if is_blocked_path(path)]
    if blocked_files:
        print("error: blocked generated/local files are staged:", file=sys.stderr)
        print("\n".join(blocked_files), file=sys.stderr)
        print(
            "\nUnstage them or commit them manually only if they are "
            "intentional tiny samples.",
            file=sys.stderr,
        )
        unstage_all()
        return 1

    if not has_staged_changes():
        print("No allowed changes staged.")
        return 0

    print_staged_summary()

    if args.dry_run:
        print("\nDry run only. No commit created.")
        unstage_all()
        return 0

    commit(args.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
