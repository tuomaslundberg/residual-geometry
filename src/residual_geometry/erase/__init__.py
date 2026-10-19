"""Language-signal erasure methods."""

from .direction import DirectionEraser, GlobalCenteringEraser
from .inlp import INLPEraser
from .leace import LEACEEraser
from .mean_centering import MeanCenteringEraser
from .mikolov import MikolovProjection
from ._base import IdentityProjector, LanguageProjector, build_projector

# Backward-compatible alias used in tests and experiments.
INLPProjector = INLPEraser

__all__ = [
    "LanguageProjector",
    "DirectionEraser",
    "GlobalCenteringEraser",
    "IdentityProjector",
    "INLPEraser",
    "INLPProjector",
    "LEACEEraser",
    "MeanCenteringEraser",
    "MikolovProjection",
    "build_projector",
]
