"""Language-signal erasure methods."""

from .inlp import INLPEraser
from .leace import LEACEEraser
from .mean_centering import MeanCenteringEraser
from .mikolov import MikolovProjection
from ._base import IdentityProjector, LanguageProjector, build_projector

# Backward-compatible alias used in tests and experiments.
INLPProjector = INLPEraser

__all__ = [
    "LanguageProjector",
    "IdentityProjector",
    "INLPEraser",
    "INLPProjector",
    "LEACEEraser",
    "MeanCenteringEraser",
    "MikolovProjection",
    "build_projector",
]
