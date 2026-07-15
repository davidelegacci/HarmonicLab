"""Finite-game decomposition with a lightweight NumPy core."""

from .decomposition import (
    Decomposition,
    FlowDecomposition,
    GameGeometry,
    ResponseEdge,
    decompose,
)

__all__ = [
    "Decomposition",
    "FlowDecomposition",
    "GameGeometry",
    "ResponseEdge",
    "decompose",
]
