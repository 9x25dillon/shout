#!/usr/bin/env python3
"""
patch_apply.py — safely apply unified diffs to a working tree.

Features
- Reads diff from a file or stdin
- Dry-run check (no backups created during dry run)
- Optional new git branch (safe workspace)
- Auto backup of touched files before applying (non-dry-run only)
- Uses `git apply` (with --check / --index / --reject / --3way)
- Fallback to `patch`(1) when no git repo is found (tries -p1 then -p0)
- Creates *.rej files on conflicts and preserves backups

Usage
  python patch_apply.py --diff changes.diff                # apply into current repo
  python patch_apply.py --diff -                           # read diff from stdin
  python patch_apply.py --diff changes.diff --dry-run
  python patch_apply.py --diff changes.diff --new-branch qwen-opt-001
  python patch_apply.py --diff changes.diff --root /path/to/repo
  python patch_apply.py --diff changes.diff --no-index     # apply without staging

Exit codes
  0 success, 1 failure (see stderr for details)
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def sh(cmd: List[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def is_git_repo(root: Path) -> bool:
    try:
        out = sh(["git", "rev-parse", "--is-inside-work-tree"], cwd=root)
        return out.returncode == 0 and out.stdout.strip() == "true"
    except FileNotFoundError:
        return False


def git_root(start: Path) -> Path | None:
    try:
        out = sh(["git", "rev-parse", "--show-toplevel"], cwd=start)
        if out.returncode == 0:
            return Path(out.stdout.strip())
    except FileNotFoundError:
        return None
    return None


def parse_touched_paths(diff_text: str) -> List[str]:
    """
    Extract file paths from unified diff headers.
    Looks for lines: '+++ b/path' and '--- a/path' and returns likely targets.
    """
    targets: List[str] = []
    plus_re = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
    minus_re = re.compile(r"^---\s+(?:a/)?(.+)$")
    for line in diff_text.splitlines():
        m = plus_re.match(line) or minus_re.match(line)
        if m:
            p = m.group(1).strip()
            if p not in ("dev/null", "/dev/null"):
                targets.append(p)
    # de-dup while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for t in targets:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def safe_under_root(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve()) + os.sep) and p != root.resolve():
        raise ValueError(f"Refusing path outside root: {rel}")
    return p


def backup_files(root: Path, files: List[str], backup_dir: Path) -> List[Tuple[Path, Path]]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    copies: List[Tuple[Path, Path]] = []
    for f in files:
        src = safe_under_root(root, f)
        if src.exists():
            rel = Path(f)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copies.append((src, dst))
    return copies


def ensure_branch(root: Path, new_branch: str | None) -> str | None:
    if not new_branch:
        return None
    # create and switch
    r1 = sh(["git", "checkout", "-b", new_branch], cwd=root)
    if r1.returncode != 0:
        # maybe exists -> switch to it
        r2 = sh(["git", "checkout", new_branch], cwd=root)
        if r2.returncode != 0:
            sys.stderr.write(r1.stderr or r1.stdout)
            sys.stderr.write(r2.stderr or r2.stdout)
            raise SystemExit(1)
    return new_branch


def git_apply(root: Path, diff_path: Path, dry_run: bool, index: bool) -> Tuple[bool, str]:
    # First: --check (dry-run validation)
    chk = sh(["git", "apply", "--check", str(diff_path)], cwd=root)
    if chk.returncode != 0:
        return False, chk.stderr or chk.stdout

    if dry_run:
        return True, "Dry-run OK"

    # Try with index + reject
    args: List[str] = ["git", "apply"]
    if index:
        args.append("--index")
    args += ["--reject", "--whitespace=fix", str(diff_path)]
    r = sh(args, cwd=root)
    if r.returncode == 0:
        return True, r.stdout or "Applied with git apply --reject"

    # Try 3-way merge
    r2 = sh(["git", "apply", "--reject", "--whitespace=fix", "--3way", str(diff_path)], cwd=root)
    if r2.returncode == 0:
        return True, r2.stdout or "Applied with git apply --3way"
    return False, (r.stderr or "") + "\n" + (r2.stderr or "")


def patch_fallback(root: Path, diff_path: Path, dry_run: bool) -> Tuple[bool, str]:
    # Use patch(1) utility if available
    if shutil.which("patch") is None:
        return False, "Neither git repo nor 'patch' tool available."

    reject_summary_file = diff_path.with_suffix(".rej.txt")

    def run_patch(strip_components: int) -> subprocess.CompletedProcess:
        args = ["patch", f"-p{strip_components}", "-t", "-r", str(reject_summary_file)]
        if dry_run:
            args.append("--dry-run")
        with open(diff_path, "r", encoding="utf-8") as f:
            return subprocess.run(
                args,
                cwd=root,
                stdin=f,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    # Try -p1 (common for git-format diffs) first, then -p0
    proc = run_patch(1)
    if proc.returncode != 0:
        proc = run_patch(0)

    ok = proc.returncode == 0
    combined_output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return ok, combined_output


def main() -> None:
    ap = argparse.ArgumentParser(description="Safely apply unified diffs.")
    ap.add_argument("--diff", required=True, help="Path to unified diff, or '-' for stdin")
    ap.add_argument("--root", default=".", help="Repo/work tree root (default: .)")
    ap.add_argument("--dry-run", action="store_true", help="Check only; do not change files")
    ap.add_argument("--new-branch", help="Create/switch to a new git branch before applying")
    ap.add_argument("--no-index", action="store_true", help="When using git, do not stage changes")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        raise SystemExit(1)

    # Load diff content
    if args.diff == "-":
        diff_text = sys.stdin.read()
        if not diff_text.strip():
            print("Empty diff on stdin.", file=sys.stderr)
            raise SystemExit(1)
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".diff") as tf:
            tf.write(diff_text)
            tf.flush()
            diff_path = Path(tf.name)
    else:
        diff_path = Path(args.diff).resolve()
        try:
            diff_text = diff_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"Failed to read diff: {exc}", file=sys.stderr)
            raise SystemExit(1)

    # Determine touched files
    touched = parse_touched_paths(diff_text)

    # Prepare backups directory (but only perform backups when not dry-running)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = root / f".patch_backups/backup-{ts}"
    backups_made = False
    if not args.dry_run and touched:
        try:
            backup_files(root, touched, backup_dir)
            backups_made = True
        except Exception as e:
            print(f"Backup error: {e}", file=sys.stderr)
            raise SystemExit(1)

    # Git route?
    repo_root = git_root(root) if is_git_repo(root) else None
    if repo_root:
        try:
            if args.new_branch:
                ensure_branch(repo_root, args.new_branch)
            ok, msg = git_apply(repo_root, diff_path, dry_run=args.dry_run, index=not args.no_index)
        except FileNotFoundError:
            ok, msg = False, "git not found on PATH."
    else:
        ok, msg = patch_fallback(root, diff_path, dry_run=args.dry_run)

    if args.diff == "-":
        try:
            diff_path.unlink(missing_ok=True)  # cleanup tmp
        except Exception:
            pass

    if ok:
        print("✅ Patch applied." if not args.dry_run else "✅ Dry-run succeeded.")
        if msg.strip():
            print(msg.strip())
        if backups_made:
            print(f"Backups in: {backup_dir}")
        raise SystemExit(0)
    else:
        print("❌ Patch failed.", file=sys.stderr)
        if msg.strip():
            print(msg.strip(), file=sys.stderr)
        if backups_made:
            print(f"(Your files were backed up in: {backup_dir})", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

