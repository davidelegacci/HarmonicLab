"""Exact SymPy operators for small-game research.

These routines intentionally favor transparent symbolic expressions over the
performance of :mod:`CandoganDecomposition.decomposition`.
"""

from __future__ import annotations

from itertools import product
from math import prod
from numbers import Integral
from typing import Iterable, Sequence

__all__ = [
    "harmonic_operator",
    "normalization_operator",
    "normalized_harmonic_basis",
]


def _sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "symbolic research helpers require SymPy; install it with "
            "`python -m pip install sympy`"
        ) from exc
    return sp


def _validate(
    n_actions: Sequence[int], mu: Sequence[Iterable[object]] | None
):
    sp = _sympy()
    raw_actions = tuple(n_actions)
    if not raw_actions or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in raw_actions
    ):
        raise ValueError("n_actions must contain positive integers")
    actions = tuple(int(value) for value in raw_actions)
    if mu is None:
        measures = tuple(tuple(sp.Integer(1) for _ in range(size)) for size in actions)
    else:
        if len(mu) != len(actions):
            raise ValueError("mu must provide one action measure per player")
        measures = []
        for player, (weights, size) in enumerate(zip(mu, actions)):
            values = tuple(sp.sympify(value) for value in weights)
            if len(values) != size:
                raise ValueError(f"mu[{player}] must contain {size} entries")
            if any(value.is_positive is not True for value in values):
                raise ValueError("symbolic action measures must be provably positive")
            measures.append(values)
        measures = tuple(measures)
    return sp, actions, measures


def _column(player: int, profile: tuple[int, ...], n_profiles: int, actions) -> int:
    index = 0
    stride = 1
    for action, size in zip(reversed(profile), reversed(actions)):
        index += action * stride
        stride *= size
    return player * n_profiles + index


def harmonic_operator(
    n_actions: Sequence[int],
    mu: Sequence[Iterable[object]] | None = None,
):
    """Matrix of the weighted harmonic equation from Abdou Definition 2(d)."""

    sp, actions, measures = _validate(n_actions, mu)
    profiles = tuple(product(*(range(size) for size in actions)))
    n_profiles = prod(actions)
    operator = sp.zeros(n_profiles, len(actions) * n_profiles)

    for row, profile in enumerate(profiles):
        for player, weights in enumerate(measures):
            current_column = _column(player, profile, n_profiles, actions)
            for alternative, weight in enumerate(weights):
                alternative_profile = list(profile)
                alternative_profile[player] = alternative
                alternative_column = _column(
                    player, tuple(alternative_profile), n_profiles, actions
                )
                operator[row, current_column] += weight
                operator[row, alternative_column] -= weight
    return operator


def normalization_operator(
    n_actions: Sequence[int],
    mu: Sequence[Iterable[object]] | None = None,
):
    """Matrix whose kernel is the space of mu-normalized games."""

    sp, actions, measures = _validate(n_actions, mu)
    n_profiles = prod(actions)
    rows = sum(prod(actions[:i] + actions[i + 1 :]) for i in range(len(actions)))
    operator = sp.zeros(rows, len(actions) * n_profiles)
    row = 0

    for player, weights in enumerate(measures):
        opponents_shape = actions[:player] + actions[player + 1 :]
        for opponents in product(*(range(size) for size in opponents_shape)):
            for action, weight in enumerate(weights):
                profile = list(opponents)
                profile.insert(player, action)
                column = _column(player, tuple(profile), n_profiles, actions)
                operator[row, column] = weight
            row += 1
    return operator


def normalized_harmonic_basis(
    n_actions: Sequence[int],
    mu: Sequence[Iterable[object]] | None = None,
):
    """Exact basis of normalized weighted harmonic games as column vectors."""

    harmonic = harmonic_operator(n_actions, mu)
    normalization = normalization_operator(n_actions, mu)
    return harmonic.col_join(normalization).nullspace()
