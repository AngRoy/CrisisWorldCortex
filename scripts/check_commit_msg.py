"""Validate commit messages against the repository convention."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADER_RE = re.compile(
    r"^(build|chore|ci|deps|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([a-z0-9][a-z0-9_.-]*\))?"
    r"(!)?: [^\s].{0,99}$"
)

EXEMPT_PREFIXES = (
    "Merge ",
    "Revert ",
    "fixup! ",
    "squash! ",
)


def _is_comment_or_blank(line: str) -> bool:
    return not line.strip() or line.lstrip().startswith("#")


def validate(message: str) -> list[str]:
    lines = [line.rstrip() for line in message.splitlines()]
    body = [line for line in lines if not line.lstrip().startswith("#")]
    if not body or all(not line.strip() for line in body):
        return ["commit message is empty"]

    header = next((line for line in body if line.strip()), "")
    if header.startswith(EXEMPT_PREFIXES):
        return []

    errors: list[str] = []
    if not HEADER_RE.match(header):
        errors.append(
            "header must match: type(scope): subject "
            "(types: build, chore, ci, deps, docs, feat, fix, perf, "
            "refactor, revert, style, test; max 100 chars)"
        )

    if len(body) > 1 and body[1].strip():
        errors.append("body must be separated from the header by a blank line")

    for line_number, line in enumerate(body[2:], start=3):
        if _is_comment_or_blank(line):
            continue
        if len(line) > 100 and "://" not in line:
            errors.append(f"body line {line_number} is longer than 100 characters")

    return errors


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_commit_msg.py <commit-msg-file>", file=sys.stderr)
        return 2

    message_path = Path(argv[1])
    errors = validate(message_path.read_text(encoding="utf-8"))
    if not errors:
        return 0

    print("Invalid commit message:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
