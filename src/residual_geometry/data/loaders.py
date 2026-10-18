"""Corpus loaders for Europarl, infopankki, and HPLT data."""

from __future__ import annotations

import json
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

    # Pair first, then drop pairs with an empty side: filtering each file on its
    # own would shift every later pair after the first unmatched empty line.
    pairs = [(f.strip(), e.strip()) for f, e in zip(fi_lines, en_lines)
             if f.strip() and e.strip()]
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


_CORE_MAIN = frozenset({"HI", "LY", "SP", "ID", "NA", "IN", "OP", "IP"})


def load_hplt(
    path: str | Path,
    max_docs: int | None = None,
    return_labels: bool = False,
    min_confidence: float = 0.0,
    exclude_mt: bool = False,
    single_register: bool = False,
) -> "list[str] | tuple[list[str], list[str]]":
    """Load texts (and optionally register labels) from an HPLT v3 exploded JSONL file.

    Args:
        path: Path to the exploded JSONL file (one document per line).
        max_docs: Truncate to this many accepted documents if given.
        return_labels: If True, also return hard register labels (argmax over the
            8 main CORE classes: HI, LY, SP, ID, NA, IN, OP, IP). Requires the
            ``web-register`` field to be present in each record.
        min_confidence: Minimum probability of the argmax CORE class to include a
            document. Documents below this threshold are silently skipped.
            Recommended value: 0.4 (CORE-optimised threshold for EN and FI).
        exclude_mt: Skip documents whose ``MT`` (machine-translated) probability
            reaches ``min_confidence``. Only applies with ``return_labels=True``.
        single_register: Keep only documents where exactly one main CORE class
            reaches ``min_confidence`` (no multi-register documents).
            Only applies with ``return_labels=True``.

    Returns:
        ``list[str]`` when ``return_labels=False`` (default, backwards-compatible).
        ``(texts, labels)`` tuple of equal-length lists when ``return_labels=True``.
    """
    texts: list[str] = []
    labels: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_docs is not None and len(texts) >= max_docs:
                break
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            if return_labels:
                reg_probs = d.get("web-register", {})
                core = {k: v for k, v in reg_probs.items() if k in _CORE_MAIN}
                if not core:
                    continue
                if exclude_mt and reg_probs.get("MT", 0.0) >= min_confidence:
                    continue
                if single_register and sum(v >= min_confidence for v in core.values()) != 1:
                    continue
                top_label = max(core, key=core.get)  # type: ignore[arg-type]
                if core[top_label] < min_confidence:
                    continue
                labels.append(top_label)
            texts.append(text)

    return (texts, labels) if return_labels else texts


def load_hplt_records(path: str | Path, min_confidence: float = 0.4) -> list[dict]:
    """Every non-empty document of an HPLT exploded JSONL file, with reading metadata.

    Texts are stripped exactly as in ``load_hplt``, so they share embedding-cache keys.
    Fields: ``text``, ``url``, ``host``, ``tld``, ``wds`` (document quality score,
    ``doc_scores[0]``), ``mt`` (MT probability >= min_confidence), ``registers`` (main
    CORE classes >= min_confidence) and ``register`` (the single main class, ``"hybrid"``
    when several reach the threshold, ``"none"`` when none does).
    """
    from urllib.parse import urlsplit

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            probs = d.get("web-register", {}) or {}
            regs = sorted(k for k, v in probs.items() if k in _CORE_MAIN and v >= min_confidence)
            url = d.get("u", "")
            host = (urlsplit(url).hostname or "").lower()
            scores = d.get("doc_scores") or [None]
            records.append({
                "text": text,
                "url": url,
                "host": host,
                "tld": host.rsplit(".", 1)[-1] if "." in host else "",
                "wds": scores[0],
                "mt": probs.get("MT", 0.0) >= min_confidence,
                "registers": regs,
                "register": regs[0] if len(regs) == 1 else ("hybrid" if regs else "none"),
            })
    return records
