#!/usr/bin/env python3
"""Main pipeline entry point.

Local dev:
    python scripts/run_pipeline.py --toy

Full run:
    python scripts/run_pipeline.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root without editable install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from register_geometry.data import CORE_REGISTERS, load_jsonl, make_toy_data, split_by_language
from register_geometry.embed import Embedder
from register_geometry.evaluate import evaluate_geometry
from register_geometry.project import build_projector
from register_geometry.transfer import cross_lingual_transfer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Register geometry pipeline")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument(
        "--toy",
        action="store_true",
        help="Run on synthetic toy data (no GPU, no corpus needed).",
    )
    p.add_argument("--output-dir", default=None, help="Override config output.dir")
    return p.parse_args()


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str, toy: bool) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if toy:
        toy_path = Path(config_path).parent / "toy.yaml"
        with open(toy_path) as f:
            toy_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, toy_cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, toy=args.toy)
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    languages: list[str] = cfg["data"]["languages"]
    registers: list[str] = cfg["data"].get("registers", list(CORE_REGISTERS))

    # ------------------------------------------------------------------
    # Stage 1: Load or generate data
    # ------------------------------------------------------------------
    print("=== Stage 1: Data ===")
    if args.toy:
        toy_cfg = cfg.get("toy", {})
        docs = make_toy_data(
            n_per_class=toy_cfg.get("n_docs_per_class", 15),
            languages=tuple(languages),
            registers=tuple(registers),
            seed=toy_cfg.get("seed", 42),
        )
        print(f"  Generated {len(docs)} toy documents.")
    else:
        data_path = Path(cfg["data"]["path"])
        docs = load_jsonl(data_path)
        print(f"  Loaded {len(docs)} documents from {data_path}.")

    by_lang = split_by_language(docs)
    assert set(languages).issubset(by_lang), (
        f"Missing languages: {set(languages) - set(by_lang)}"
    )

    # ------------------------------------------------------------------
    # Stage 2: Embed
    # ------------------------------------------------------------------
    print("=== Stage 2: Embed ===")
    embedder = Embedder(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        batch_size=cfg["model"]["batch_size"],
    )

    embeddings: dict[str, np.ndarray] = {}
    label_arrays: dict[str, np.ndarray] = {}
    for lang in languages:
        lang_docs = by_lang[lang]
        texts = [d.text for d in lang_docs]
        print(f"  Embedding {len(texts)} {lang} documents...")
        if args.toy:
            # Skip actual model load for toy run; use random vectors.
            rng = np.random.default_rng(cfg.get("toy", {}).get("seed", 42))
            embeddings[lang] = rng.standard_normal((len(texts), 768)).astype(np.float32)
        else:
            embeddings[lang] = embedder.embed(texts, show_progress=True)
        label_arrays[lang] = np.array([d.register for d in lang_docs])

    # ------------------------------------------------------------------
    # Stage 3: Project (pre-projection baseline first, then post)
    # ------------------------------------------------------------------
    print("=== Stage 3: Project ===")
    all_X = np.concatenate([embeddings[l] for l in languages])
    all_langs = np.concatenate([np.full(len(embeddings[l]), l) for l in languages])

    projector = build_projector(cfg["projection"])
    projector.fit(all_X, all_langs)

    projected: dict[str, np.ndarray] = {
        lang: projector.transform(embeddings[lang]) for lang in languages
    }
    print(f"  Method: {cfg['projection']['method']}")

    # ------------------------------------------------------------------
    # Stage 4: Evaluate geometry (pre and post)
    # ------------------------------------------------------------------
    print("=== Stage 4: Geometry ===")
    eval_cfg = cfg["evaluation"]
    lang_a, lang_b = languages[0], languages[1]

    result_pre = evaluate_geometry(
        embeddings[lang_a], label_arrays[lang_a],
        embeddings[lang_b], label_arrays[lang_b],
        registers=registers,
        n_permutations=eval_cfg["n_permutations"],
        seed=eval_cfg["seed"],
    )
    result_post = evaluate_geometry(
        projected[lang_a], label_arrays[lang_a],
        projected[lang_b], label_arrays[lang_b],
        registers=registers,
        n_permutations=eval_cfg["n_permutations"],
        seed=eval_cfg["seed"],
    )

    print(f"  Pre-projection:  {result_pre.summary()}")
    print(f"  Post-projection: {result_post.summary()}")

    # ------------------------------------------------------------------
    # Stage 5: Diagnostic classifier transfer
    # ------------------------------------------------------------------
    print("=== Stage 5: Classifier Transfer ===")
    fwd = cross_lingual_transfer(
        projected[lang_a], label_arrays[lang_a],
        projected[lang_b], label_arrays[lang_b],
    )
    rev = cross_lingual_transfer(
        projected[lang_b], label_arrays[lang_b],
        projected[lang_a], label_arrays[lang_a],
    )
    print(f"  {lang_a}→{lang_b} macro-F1: {fwd.macro_f1:.4f}")
    print(f"  {lang_b}→{lang_a} macro-F1: {rev.macro_f1:.4f}")

    # ------------------------------------------------------------------
    # Stage 6: Save results
    # ------------------------------------------------------------------
    results = {
        "config": cfg,
        "geometry_pre": result_pre.summary(),
        "geometry_post": result_post.summary(),
        "transfer_fwd": fwd.summary(),
        "transfer_rev": rev.summary(),
    }
    out_file = output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
