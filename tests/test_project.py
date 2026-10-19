"""Tests for project.py."""

import numpy as np
import pytest

from residual_geometry.erase import (
    IdentityProjector,
    INLPProjector,
    LanguageProjector,
    build_projector,
)


def test_identity_is_noop(synthetic_embeddings):
    X_fi, _, _, _, _, X_all = synthetic_embeddings
    proj = IdentityProjector()
    proj.fit(X_fi, np.array(["fi"] * len(X_fi)))
    out = proj.transform(X_fi)
    np.testing.assert_array_equal(out, X_fi)


def test_identity_returns_copy(synthetic_embeddings):
    X_fi, _, _, _, _, _ = synthetic_embeddings
    proj = IdentityProjector().fit(X_fi, np.array(["fi"] * len(X_fi)))
    out = proj.transform(X_fi)
    assert out is not X_fi


def test_inlp_output_shape(synthetic_embeddings):
    X_fi, labels_fi, X_en, labels_en, langs_all, X_all = synthetic_embeddings
    proj = INLPProjector(n_iterations=2)
    proj.fit(X_all, langs_all)
    out = proj.transform(X_fi)
    assert out.shape == X_fi.shape


def test_inlp_reduces_language_separability(synthetic_embeddings):
    """After INLP, a language-ID classifier should have lower accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_fi, labels_fi, X_en, labels_en, langs_all, X_all = synthetic_embeddings
    proj = INLPProjector(n_iterations=5)
    proj.fit(X_all, langs_all)
    X_proj = proj.transform(X_all)

    clf = LogisticRegression(max_iter=500)
    acc_before = cross_val_score(clf, X_all, langs_all, cv=3, scoring="accuracy").mean()
    acc_after = cross_val_score(clf, X_proj, langs_all, cv=3, scoring="accuracy").mean()

    # Language separability should strictly decrease (not necessarily to chance).
    assert acc_after < acc_before, (
        f"Expected language accuracy to drop after INLP; got {acc_before:.3f} → {acc_after:.3f}"
    )


def test_build_projector_none():
    proj = build_projector({"method": "none"})
    assert isinstance(proj, IdentityProjector)


def test_build_projector_inlp():
    proj = build_projector({"method": "inlp", "inlp": {"n_iterations": 3}})
    assert isinstance(proj, INLPProjector)
    assert proj.n_iterations == 3


def test_build_projector_unknown():
    with pytest.raises(ValueError, match="Unknown projection method"):
        build_projector({"method": "bogus"})


def test_inlp_transform_before_fit_raises():
    proj = INLPProjector()
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        proj.transform(np.zeros((5, 32)))


def test_inlp_directions_unit_and_counted(synthetic_embeddings):
    from residual_geometry.erase import INLPEraser

    *_, langs, X = synthetic_embeddings
    eraser = INLPEraser(n_iterations=3).fit(X, langs)
    D = eraser.directions
    assert D.shape == (eraser.n_directions_removed, X.shape[1])
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)


def test_language_probe_in_fold_noise_is_at_guessing():
    """Mean centering fitted on all data pushes a noise probe below 0.5; in-fold does not."""
    from residual_geometry.erase import MeanCenteringEraser
    from residual_geometry.eval.probing import apply_eraser, language_probe

    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 20))
    langs = np.array(["fi", "en"] * 200)
    leaky = MeanCenteringEraser().fit(X, langs)
    below = language_probe(apply_eraser(leaky, X, langs), langs, seed=1)
    fair = language_probe(X, langs, seed=1, eraser_factory=MeanCenteringEraser)
    assert below.accuracy < 0.45
    assert abs(fair.accuracy - 0.5) < 0.08
    assert fair.chance_accuracy == 0.5


def test_inlp_stops_at_chance(synthetic_embeddings):
    from residual_geometry.erase import INLPEraser

    *_, langs, X = synthetic_embeddings
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(X.shape)
    assert INLPEraser().fit(noise, langs).n_directions_removed <= 1
    eraser = INLPEraser().fit(X, langs)
    assert 1 <= eraser.n_directions_removed < 50
    assert eraser.dev_accuracies[-1] <= eraser.stop_threshold


def test_global_centering_keeps_difference_and_direction_eraser_removes_it(synthetic_embeddings):
    from residual_geometry.erase import DirectionEraser, GlobalCenteringEraser

    X_fi, _, X_en, _, langs, X = synthetic_embeddings
    gap = X_fi.mean(0) - X_en.mean(0)
    G = GlobalCenteringEraser().fit(X, langs).transform(X)
    assert np.allclose(G[langs == "fi"].mean(0) - G[langs == "en"].mean(0), gap, atol=1e-5)
    assert np.allclose(G.mean(0), 0, atol=1e-5)
    D = DirectionEraser.from_pairs(X_fi, X_en).transform(X)
    new_gap = D[langs == "fi"].mean(0) - D[langs == "en"].mean(0)
    assert np.linalg.norm(new_gap) < 1e-4 * np.linalg.norm(gap) + 1e-5


def test_leace_equalises_language_means(synthetic_embeddings):
    pytest.importorskip("concept_erasure")
    from residual_geometry.erase import LEACEEraser

    *_, langs, X = synthetic_embeddings
    Z = LEACEEraser().fit(X, langs).transform(X)
    assert np.linalg.norm(Z[langs == "fi"].mean(0) - Z[langs == "en"].mean(0)) < 1e-3
