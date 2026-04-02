"""Tests for project.py."""

import numpy as np
import pytest

from register_geometry.project import (
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
