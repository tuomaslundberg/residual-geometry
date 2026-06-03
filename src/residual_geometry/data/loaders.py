"""Corpus loaders for Europarl and infopankki parallel data."""

from __future__ import annotations

from pathlib import Path


def load_europarl(
    fi_path: str | Path,
    en_path: str | Path,
    max_pairs: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load parallel FI/EN sentence pairs from Europarl text files.

    Args:
        fi_path: Path to Finnish sentence file (one sentence per line).
        en_path: Path to English sentence file (one sentence per line).
        max_pairs: Truncate to this many pairs if given.

    Returns:
        (fi_sentences, en_sentences) as aligned lists of strings.
    """
    fi_lines = Path(fi_path).read_text(encoding="utf-8").splitlines()
    en_lines = Path(en_path).read_text(encoding="utf-8").splitlines()

    fi_lines = [l.strip() for l in fi_lines if l.strip()]
    en_lines = [l.strip() for l in en_lines if l.strip()]

    pairs = list(zip(fi_lines, en_lines))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    fi_out, en_out = zip(*pairs) if pairs else ([], [])
    return list(fi_out), list(en_out)


def load_infopankki(
    fi_path: str | Path,
    en_path: str | Path,
    max_pairs: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load parallel FI/EN sentence pairs from infopankki text files.

    Format assumed identical to Europarl (one sentence per line).
    """
    return load_europarl(fi_path, en_path, max_pairs=max_pairs)
