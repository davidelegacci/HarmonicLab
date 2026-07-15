import numpy as np
import pytest

from CandoganDecomposition import GameGeometry, decompose


def legacy_uniform_decomposition(geometry, payoffs):
    """Reference implementation of the original dense Candogan projection."""

    flat_payoffs = np.asarray(payoffs, dtype=float).reshape(-1)
    n_payoffs = geometry.n_players * geometry.n_profiles
    comparison = np.zeros((len(geometry.edges), n_payoffs))
    rows = np.arange(len(geometry.edges))
    comparison[
        rows, geometry.edge_players * geometry.n_profiles + geometry.edge_tails
    ] = -1.0
    comparison[
        rows, geometry.edge_players * geometry.n_profiles + geometry.edge_heads
    ] = 1.0

    exact_projection = (
        geometry.incidence_matrix @ np.linalg.pinv(geometry.incidence_matrix)
    )
    comparison_pinv = np.linalg.pinv(comparison)
    nonstrategic = flat_payoffs - comparison_pinv @ comparison @ flat_payoffs
    potential = (
        comparison_pinv @ exact_projection @ comparison @ flat_payoffs
    )
    harmonic = flat_payoffs - nonstrategic - potential
    return tuple(
        component.reshape(geometry.payoff_shape)
        for component in (nonstrategic, potential, harmonic)
    )


@pytest.mark.parametrize("n_actions", ([2, 2], [2, 3], [3, 3], [2, 2, 2]))
def test_counting_measure_matches_uniform_candogan_projection(n_actions):
    geometry = GameGeometry(n_actions)
    rng = np.random.default_rng(sum(n_actions))
    payoffs = rng.normal(size=geometry.payoff_shape)

    expected = legacy_uniform_decomposition(geometry, payoffs)
    actual = geometry.decompose(payoffs)

    assert np.allclose(actual.nonstrategic, expected[0])
    assert np.allclose(actual.potential, expected[1])
    assert np.allclose(actual.harmonic, expected[2])


def test_weighted_decomposition_satisfies_geometric_characterization():
    geometry = GameGeometry([2, 3, 2], mu=[[1, 4], [2, 1, 3], [5, 2]])
    payoffs = np.random.default_rng(11).normal(size=geometry.payoff_shape)
    result = geometry.decompose(payoffs)

    assert np.allclose(
        payoffs, result.nonstrategic + result.potential + result.harmonic
    )
    assert np.allclose(geometry.game_flow(result.nonstrategic), 0.0)
    assert np.allclose(
        geometry.game_flow(result.potential), result.potential_flow
    )
    assert np.allclose(
        geometry.game_flow(result.harmonic), result.harmonic_flow
    )
    assert np.allclose(geometry.codifferential(result.harmonic_flow), 0.0)

    for player, weights in enumerate(geometry.mu):
        assert np.allclose(
            np.tensordot(
                weights, result.potential[player], axes=((0,), (player,))
            ),
            0.0,
        )
        assert np.allclose(
            np.tensordot(
                weights, result.harmonic[player], axes=((0,), (player,))
            ),
            0.0,
        )

    assert np.isclose(
        geometry.game_inner_product(result.potential, result.harmonic), 0.0
    )
    assert np.isclose(
        geometry.game_inner_product(result.potential, result.nonstrategic), 0.0
    )
    assert np.isclose(
        geometry.game_inner_product(result.harmonic, result.nonstrategic), 0.0
    )


def test_gradient_and_codifferential_are_weighted_adjoints():
    geometry = GameGeometry([2, 3], mu=[[1, 5], [2, 3, 4]])
    rng = np.random.default_rng(19)
    function = rng.normal(size=geometry.n_actions)
    flow = rng.normal(size=len(geometry.edges))

    gradient = geometry.gradient_matrix @ function.reshape(-1)
    left = geometry.flow_inner_product(gradient, flow)
    right = geometry.profile_inner_product(function, geometry.codifferential(flow))

    assert np.isclose(left, right)


def test_direct_flow_decomposition_is_weighted_orthogonal():
    geometry = GameGeometry([2, 3], mu=[[1, 5], [2, 3, 4]])
    flow = np.random.default_rng(21).normal(size=len(geometry.edges))
    result = geometry.decompose_flow(flow)

    assert np.allclose(flow, result.exact + result.co_closed)
    assert np.allclose(geometry.codifferential(result.co_closed), 0.0)
    assert np.isclose(
        geometry.flow_inner_product(result.exact, result.co_closed), 0.0
    )


def test_normalized_harmonic_condition_uses_action_masses():
    geometry = GameGeometry([2, 3], mu=[[1, 2], [2, 5, 1]])
    payoffs = np.random.default_rng(23).normal(size=geometry.payoff_shape)
    harmonic = geometry.decompose(payoffs).harmonic

    balance = sum(
        geometry.action_masses[player] * harmonic[player]
        for player in range(geometry.n_players)
    )
    assert np.allclose(balance, 0.0)


def test_game_embedding_is_an_isometry_on_normalized_games():
    geometry = GameGeometry([2, 3], mu=[[1, 2], [2, 5, 1]])
    payoffs = np.random.default_rng(24).normal(size=geometry.payoff_shape)
    normalized = geometry.normalize_game(payoffs)

    flow = geometry.game_flow(normalized)
    assert np.isclose(geometry.game_norm(normalized), geometry.flow_norm(flow))

    game_result = geometry.decompose(payoffs)
    flow_result = geometry.decompose_flow(flow)
    assert np.isclose(game_result.potentialness, flow_result.potentialness)


def test_potential_component_has_returned_potential_function():
    geometry = GameGeometry([2, 3], mu=[[1, 2], [4, 1, 3]])
    source_potential = np.arange(6, dtype=float).reshape(2, 3)
    potential_game = geometry.potential_game(source_potential)
    nonstrategic = np.array(
        [
            [[10, 20, 30], [10, 20, 30]],
            [[7, 7, 7], [-2, -2, -2]],
        ],
        dtype=float,
    )

    result = geometry.decompose(potential_game + nonstrategic)

    assert np.allclose(result.potential, potential_game)
    assert np.allclose(result.harmonic, 0.0)
    assert np.allclose(result.nonstrategic, nonstrategic)
    assert np.allclose(
        geometry.gradient_matrix @ result.potential_function.reshape(-1),
        geometry.game_flow(potential_game),
    )
    assert np.isclose(result.potentialness, 1.0)


def test_matching_pennies_is_uniform_normalized_harmonic():
    first = np.array([[1.0, -1.0], [-1.0, 1.0]])
    payoffs = np.stack([first, -first])
    result = GameGeometry([2, 2]).decompose(payoffs)

    assert np.allclose(result.nonstrategic, 0.0)
    assert np.allclose(result.potential, 0.0)
    assert np.allclose(result.harmonic, payoffs)
    assert np.isclose(result.potentialness, 0.0)


def test_common_measure_rescaling_does_not_change_components():
    mu = [np.array([1.0, 3.0]), np.array([2.0, 5.0, 4.0])]
    payoffs = np.random.default_rng(29).normal(size=(2, 2, 3))

    original = GameGeometry([2, 3], mu=mu).decompose(payoffs)
    rescaled = GameGeometry([2, 3], mu=[7 * weights for weights in mu]).decompose(
        payoffs
    )

    assert np.allclose(original.nonstrategic, rescaled.nonstrategic)
    assert np.allclose(original.potential, rescaled.potential)
    assert np.allclose(original.harmonic, rescaled.harmonic)


def test_convenience_api_infers_action_counts():
    payoffs = np.arange(16, dtype=float).reshape(2, 2, 4)
    result = decompose(payoffs, mu=[[1, 2], [1, 3, 2, 4]])
    assert result.potential.shape == payoffs.shape


def test_geometry_accepts_player_major_flat_payoffs():
    geometry = GameGeometry([2, 3])
    payoffs = np.arange(12, dtype=float)
    result = geometry.decompose(payoffs)
    assert result.potential.shape == geometry.payoff_shape


def test_purely_nonstrategic_game_has_undefined_potentialness():
    geometry = GameGeometry([2, 2])
    payoffs = np.array(
        [
            [[1, 2], [1, 2]],
            [[3, 3], [4, 4]],
        ],
        dtype=float,
    )
    result = geometry.decompose(payoffs)

    assert np.allclose(result.strategic, 0.0)
    assert np.isnan(result.potentialness)


@pytest.mark.parametrize(
    ("n_actions", "mu", "message"),
    [
        ([], None, "at least one player"),
        ([2, 0], None, "positive integer"),
        ([2, 2], [[1, 1]], "one action measure"),
        ([2, 2], [[1, 0], [1, 1]], "strictly positive"),
        ([2, 2], [[1, 1, 1], [1, 1]], "shape"),
    ],
)
def test_invalid_geometry_is_rejected(n_actions, mu, message):
    with pytest.raises(ValueError, match=message):
        GameGeometry(n_actions, mu=mu)


def test_invalid_payoff_shape_is_rejected():
    with pytest.raises(ValueError, match="payoffs must have shape"):
        GameGeometry([2, 3]).decompose(np.zeros((2, 3, 2)))
