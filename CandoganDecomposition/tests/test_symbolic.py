from math import prod

import numpy as np
import pytest

pytest.importorskip("sympy")

from CandoganDecomposition import GameGeometry
from CandoganDecomposition.research import (
    harmonic_operator,
    normalization_operator,
    normalized_harmonic_basis,
)


@pytest.mark.parametrize("n_actions", ([2, 2], [2, 3], [2, 2, 2]))
def test_normalized_symbolic_harmonic_dimension(n_actions):
    dimension = int(
        prod(n_actions)
        * (len(n_actions) - 1 - sum(1 / count for count in n_actions))
        + 1
    )
    basis = normalized_harmonic_basis(n_actions)
    assert len(basis) == dimension


def test_symbolic_operators_match_numpy_weighted_conditions():
    n_actions = [2, 3]
    mu = [[1, 2], [2, 5, 1]]
    geometry = GameGeometry(n_actions, mu=mu)
    payoffs = np.random.default_rng(31).integers(
        -4, 5, size=geometry.payoff_shape
    )
    flat_payoffs = payoffs.reshape(-1)

    symbolic_harmonic = np.asarray(
        harmonic_operator(n_actions, mu).tolist(), dtype=float
    )
    symbolic_normalization = np.asarray(
        normalization_operator(n_actions, mu).tolist(), dtype=float
    )

    assert np.allclose(
        symbolic_harmonic @ flat_payoffs,
        geometry.codifferential(geometry.game_flow(payoffs)).reshape(-1),
    )
    normalized = geometry.normalize_game(payoffs)
    assert np.allclose(symbolic_normalization @ normalized.reshape(-1), 0.0)
