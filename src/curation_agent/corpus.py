"""Load a knowledge corpus of evidence snippets from a file or a folder.

A corpus is plain text: one self-contained snippet per line, with blank lines and
`#` comments ignored. `load_corpus` accepts either a single `.txt` file or a
directory of `.txt` files (read in sorted filename order, concatenated). The same
loader feeds both the benchmark generator (as generation seeds) and the retriever
(as the searchable knowledge store).
"""

from __future__ import annotations

from pathlib import Path


def _read_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s for line in lines if (s := line.strip()) and not s.startswith("#")]


def load_corpus(path: str | Path) -> list[str]:
    """Return the list of snippets from a corpus file or directory.

    - A file: its non-comment, non-blank lines.
    - A directory: the same, across every `*.txt` file in sorted name order.
    Raises FileNotFoundError if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"corpus path not found: {p}")
    if p.is_dir():
        snippets: list[str] = []
        for txt in sorted(p.glob("*.txt")):
            snippets.extend(_read_file(txt))
        return snippets
    return _read_file(p)
