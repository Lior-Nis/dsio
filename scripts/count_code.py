"""Count executable code lines per package: total minus docstrings, comments and blanks.

The spec's line target counts code only. This repository writes its reasoning into
docstrings by house style, so a total-line count measures how much was explained as much
as how much was built -- and moves the wrong way every time a decision is recorded well.

Usage: ``uv run --extra cpu python scripts/count_code.py [rev]`` (default ``HEAD``).
Reads from git rather than the working tree, so the number is reproducible at any commit.
"""

from __future__ import annotations

import ast
import collections
import subprocess
import sys


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def _doc_lines(source: str) -> set[int]:
    """Line numbers spanned by bare string-expression statements, i.e. docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    spans: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
        ):
            spans.update(range(node.lineno, node.end_lineno + 1))
    return spans


def main() -> None:
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    tracked = _git("ls-tree", "-r", "--name-only", rev, "src/").split()
    paths = [p for p in tracked if p.endswith(".py")]

    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for path in paths:
        source = _git("show", f"{rev}:{path}")
        docs = _doc_lines(source)
        parts = path.split("/")
        package = parts[2] if len(parts) > 3 else "(top-level)"
        for lineno, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                per[package][2] += 1
            elif lineno in docs or stripped.startswith("#"):
                per[package][1] += 1
            else:
                per[package][0] += 1

    total = [0, 0, 0]
    print(f"{'package':<16}{'code':>8}{'doc':>8}{'blank':>8}{'total':>8}")
    print("-" * 48)
    for package in sorted(per):
        code, doc, blank = per[package]
        total = [a + b for a, b in zip(total, (code, doc, blank), strict=True)]
        print(f"{package:<16}{code:>8}{doc:>8}{blank:>8}{code + doc + blank:>8}")
    print("-" * 48)
    print(f"{'TOTAL':<16}{total[0]:>8}{total[1]:>8}{total[2]:>8}{sum(total):>8}")


if __name__ == "__main__":
    main()
