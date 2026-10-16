"""Tests for load_hplt's register filters and the Phase 2 balanced draw."""

import json

import numpy as np

from residual_geometry.data.loaders import load_hplt


def _write(tmp_path, records):
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


RECORDS = [
    {"text": "single", "web-register": {"NA": 0.9, "IN": 0.1, "MT": 0.0}},
    {"text": "hybrid", "web-register": {"NA": 0.6, "IN": 0.5, "MT": 0.0}},
    {"text": "translated", "web-register": {"IN": 0.8, "MT": 0.7}},
    {"text": "weak", "web-register": {"OP": 0.2}},
]


def test_default_keeps_mt_and_hybrids(tmp_path):
    texts, labels = load_hplt(_write(tmp_path, RECORDS), return_labels=True, min_confidence=0.4)
    assert texts == ["single", "hybrid", "translated"]
    assert labels == ["NA", "NA", "IN"]


def test_exclude_mt(tmp_path):
    texts, _ = load_hplt(_write(tmp_path, RECORDS), return_labels=True,
                         min_confidence=0.4, exclude_mt=True)
    assert texts == ["single", "hybrid"]


def test_single_register(tmp_path):
    texts, labels = load_hplt(_write(tmp_path, RECORDS), return_labels=True,
                              min_confidence=0.4, exclude_mt=True, single_register=True)
    assert texts == ["single"] and labels == ["NA"]


def test_balanced_indices_equal_per_class():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "experiments" / "pilot_phase2.py"
    spec = importlib.util.spec_from_file_location("pilot_phase2", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    labels = np.array(["A"] * 50 + ["B"] * 7 + ["C"] * 20)
    idx = pilot._balanced_indices(labels, ["A", "B", "C"], 7, np.random.default_rng(1))
    assert len(set(idx.tolist())) == 21
    assert {c: int(np.sum(labels[idx] == c)) for c in "ABC"} == {"A": 7, "B": 7, "C": 7}
    again = pilot._balanced_indices(labels, ["A", "B", "C"], 7, np.random.default_rng(1))
    assert np.array_equal(idx, again)
