"""Core data structures and corpus utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

CORE_REGISTERS = ("HI", "ID", "IN", "IP", "LY", "NA", "OP", "SP")


@dataclass(frozen=True)
class Document:
    text: str
    language: str
    register: str


def load_jsonl(path: str | Path) -> list[Document]:
    """Load documents from a JSONL file with keys: text, language, register."""
    docs: list[Document] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(obj["text"], obj["language"], obj["register"]))
    return docs


def make_toy_data(
    n_per_class: int = 15,
    languages: tuple[str, ...] = ("fi", "en"),
    registers: tuple[str, ...] = CORE_REGISTERS,
    seed: int = 42,
) -> list[Document]:
    """Synthetic toy documents for pipeline smoke tests (no real text)."""
    rng = np.random.default_rng(seed)
    docs: list[Document] = []
    for lang in languages:
        for reg in registers:
            for i in range(n_per_class):
                text = f"[TOY] lang={lang} register={reg} doc={i}"
                docs.append(Document(text, lang, reg))
    rng.shuffle(docs)  # type: ignore[arg-type]
    return list(docs)


def split_by_language(docs: list[Document]) -> dict[str, list[Document]]:
    result: dict[str, list[Document]] = {}
    for doc in docs:
        result.setdefault(doc.language, []).append(doc)
    return result


def iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
