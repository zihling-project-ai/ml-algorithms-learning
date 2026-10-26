"""Heuristic review of tracked/staged files and reachable history; never prints matched values.

This is not a guarantee that all secrets or confidential content have been detected.
Run before the first commit with --staged, and after committing with --history.
"""
from pathlib import Path
import argparse
import re
import struct
import subprocess
import zlib

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "local-absolute-path": re.compile(r"/(?:Users|home|Volumes)/[^\s`<>]+|/(?:private/)?var/folders/[^\s`<>]+"),
    "credential-candidate": re.compile(
        r"sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|AKIA[A-Z0-9]{16}|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|"
        r"(?i:(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*[\"'][^\"']{6,})"
    ),
    "email-address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
EXCLUDED_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", ".review-local"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".cfg", ".yml", ".yaml"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def png_text(data):
    """Inspect PNG text/EXIF chunks instead of interpreting image pixels as text."""
    offset = 8
    texts = []
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset:offset+4])[0]
        kind = data[offset+4:offset+8]
        chunk = data[offset+8:offset+8+size]
        if kind in {b"tEXt", b"iTXt", b"eXIf"}:
            texts.append(chunk.decode("utf-8", errors="replace"))
        elif kind == b"zTXt":
            try:
                keyword, rest = chunk.split(b"\0", 1)
                texts.append(keyword.decode(errors="replace") + zlib.decompress(rest[1:]).decode(errors="replace"))
            except (ValueError, zlib.error):
                texts.append("unreadable PNG text metadata")
        offset += size + 12
    return "\n".join(texts)


def scan(name, data):
    issues = []
    path = Path(name)
    if EXCLUDED_PARTS.intersection(path.parts) or path.name.endswith(".bak") or path.name == ".DS_Store":
        issues.append((name, 0, "excluded-file"))
    if path.name == ".env" or path.name.startswith(".env.") or path.suffix in {".pem", ".key"}:
        issues.append((name, 0, "sensitive-file-type"))
    if len(data) > 5 * 1024 * 1024:
        issues.append((name, 0, "large-file-over-5-MiB"))
    if path.suffix == ".png":
        text = png_text(data)
    elif path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}:
        text = data.decode("utf-8", errors="replace")
    else:
        text = ""
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS.items():
            if pattern.search(line):
                issues.append((name, number, label))
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    if args.staged:
        paths = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
    else:
        paths = git("ls-files", "-z").decode().split("\0")
    issues = []
    count = 0
    for name in filter(None, paths):
        data = git("show", ":" + name) if args.staged else (ROOT / name).read_bytes()
        issues.extend(scan(name, data)); count += 1
    blobs = 0
    commits = []
    if args.history:
        commits = git("rev-list", "--all").decode().splitlines()
        for commit in commits:
            # Check messages/identity for accidental credentials or absolute paths.
            meta = git("show", "-s", "--format=%B%n%an%n%ae", commit).decode()
            for label, pattern in PATTERNS.items():
                if label != "email-address" and pattern.search(meta):
                    issues.append(("commit:" + commit[:12], 0, label))
        for line in git("rev-list", "--objects", "--all").decode().splitlines():
            fields = line.split(" ", 1)
            if len(fields) != 2:
                continue
            oid, name = fields
            if git("cat-file", "-t", oid).strip() != b"blob":
                continue
            issues.extend(scan("history/" + name, git("cat-file", "blob", oid)))
            blobs += 1
    for name, number, label in sorted(set(issues)):
        print(f"{name}:{number}: {label}")
    print(f"Scanned {count} files; {len(commits)} commits; {blobs} historical blobs; {len(set(issues))} findings")
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
