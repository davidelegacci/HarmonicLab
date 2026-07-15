"""Fast potential-harmonic decomposition of finite normal-form games.

The implementation follows Candogan et al. for counting measures and Abdou
et al. for arbitrary positive action measures ``mu``.  Comeasures are not part
of this API: the chosen convention absorbs them into a strategically
equivalent payoff representative, whereas action measures change the harmonic
component itself.

Only NumPy is imported.  Symbolic and exploratory helpers live under
``CandoganDecomposition.research``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import prod
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Decomposition",
    "FlowDecomposition",
    "GameGeometry",
    "ResponseEdge",
    "decompose",
]


@dataclass(frozen=True)
class ResponseEdge:
    """One oriented representative of an undirected response-graph edge."""

    player: int
    tail: tuple[int, ...]
    head: tuple[int, ...]


@dataclass(frozen=True)
class FlowDecomposition:
    """Weighted orthogonal decomposition of a response-graph flow."""

    exact: np.ndarray
    co_closed: np.ndarray
    potential_function: np.ndarray
    potentialness: float


@dataclass(frozen=True)
class Decomposition:
    """Potential, harmonic, and nonstrategic components of a game."""

    nonstrategic: np.ndarray
    potential: np.ndarray
    harmonic: np.ndarray
    potential_function: np.ndarray
    flow: np.ndarray
    potential_flow: np.ndarray
    harmonic_flow: np.ndarray
    potentialness: float

    @property
    def strategic(self) -> np.ndarray:
        return self.potential + self.harmonic

    # Short aliases ease comparison with the original research code.
    @property
    def uN(self) -> np.ndarray:
        return self.nonstrategic

    @property
    def uP(self) -> np.ndarray:
        return self.potential

    @property
    def uH(self) -> np.ndarray:
        return self.harmonic


class GameGeometry:
    """Response-graph geometry shared by games with the same action sets.

    Parameters
    ----------
    n_actions:
        Number of actions available to each player.
    mu:
        Strictly positive action measures.  ``mu[i][a]`` is the measure of
        action ``a`` of player ``i``.  The default is counting measure (all
        ones), which recovers the Candogan decomposition.

    Notes
    -----
    Payoffs use shape ``(n_players, *n_actions)`` and zero-based action
    indices.  Measures are deliberately not normalized: player-specific
    rescalings can change the decomposition, while a common global rescaling
    of all measures does not.
    """

    def __init__(
        self,
        n_actions: Sequence[int],
        mu: Sequence[Iterable[float]] | None = None,
    ) -> None:
        self.n_actions = self._validate_n_actions(n_actions)
        self.n_players = len(self.n_actions)
        self.n_profiles = prod(self.n_actions)
        self.payoff_shape = (self.n_players, *self.n_actions)
        self.mu = self._validate_mu(mu)
        self.action_masses = np.array([weights.sum() for weights in self.mu])

        self.profiles = tuple(np.ndindex(self.n_actions))
        self.profile_weights = np.array(
            [
                prod(self.mu[i][profile[i]] for i in range(self.n_players))
                for profile in self.profiles
            ],
            dtype=float,
        )

        self.edges = self._make_edges()
        self.edge_players = np.array(
            [edge.player for edge in self.edges], dtype=int
        )
        self.edge_tails = np.array(
            [
                np.ravel_multi_index(edge.tail, self.n_actions)
                for edge in self.edges
            ],
            dtype=int,
        )
        self.edge_heads = np.array(
            [
                np.ravel_multi_index(edge.head, self.n_actions)
                for edge in self.edges
            ],
            dtype=int,
        )

        self.incidence_matrix = self._make_incidence_matrix()
        self.gradient_scales = self._make_gradient_scales()
        self.flow_weights = (
            self.profile_weights[self.edge_tails]
            * self.profile_weights[self.edge_heads]
        )
        self.gradient_matrix = self.gradient_scales[:, None] * self.incidence_matrix
        weighted_laplacian = self.gradient_matrix.T @ (
            self.flow_weights[:, None] * self.gradient_matrix
        )
        if len(self.edges):
            self._weighted_laplacian_pinv = np.linalg.pinv(
                weighted_laplacian, hermitian=True
            )
        else:
            self._weighted_laplacian_pinv = np.zeros(
                (self.n_profiles, self.n_profiles)
            )

    @staticmethod
    def _validate_n_actions(n_actions: Sequence[int]) -> tuple[int, ...]:
        try:
            values = tuple(n_actions)
        except TypeError as exc:
            raise TypeError("n_actions must be a nonempty sequence of integers") from exc
        if not values:
            raise ValueError("n_actions must contain at least one player")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in values
        ):
            raise ValueError("each action count must be a positive integer")
        return tuple(int(value) for value in values)

    def _validate_mu(
        self, mu: Sequence[Iterable[float]] | None
    ) -> tuple[np.ndarray, ...]:
        if mu is None:
            return tuple(np.ones(size, dtype=float) for size in self.n_actions)
        if len(mu) != self.n_players:
            raise ValueError("mu must provide one action measure per player")

        measures = []
        for player, (weights, size) in enumerate(zip(mu, self.n_actions)):
            array = np.asarray(weights, dtype=float)
            if array.shape != (size,):
                raise ValueError(
                    f"mu[{player}] must have shape ({size},), got {array.shape}"
                )
            if not np.all(np.isfinite(array)) or np.any(array <= 0):
                raise ValueError("action measures must be finite and strictly positive")
            measures.append(array.copy())
        return tuple(measures)

    def _make_edges(self) -> tuple[ResponseEdge, ...]:
        edges = []
        for player, size in enumerate(self.n_actions):
            opponents_shape = self.n_actions[:player] + self.n_actions[player + 1 :]
            opponent_profiles = product(
                *(range(count) for count in opponents_shape)
            )
            for opponents in opponent_profiles:
                for tail_action, head_action in combinations(range(size), 2):
                    tail = list(opponents)
                    head = list(opponents)
                    tail.insert(player, tail_action)
                    head.insert(player, head_action)
                    edges.append(
                        ResponseEdge(player, tuple(tail), tuple(head))
                    )
        return tuple(edges)

    def _make_incidence_matrix(self) -> np.ndarray:
        incidence = np.zeros((len(self.edges), self.n_profiles), dtype=float)
        rows = np.arange(len(self.edges))
        incidence[rows, self.edge_tails] = -1.0
        incidence[rows, self.edge_heads] = 1.0
        return incidence

    def _make_gradient_scales(self) -> np.ndarray:
        scales = np.empty(len(self.edges), dtype=float)
        for row, edge in enumerate(self.edges):
            opponent_weight = prod(
                self.mu[player][edge.tail[player]]
                for player in range(self.n_players)
                if player != edge.player
            )
            scales[row] = 1.0 / np.sqrt(opponent_weight)
        return scales

    def _as_game(self, payoffs: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(payoffs, dtype=float)
        if array.shape == (self.n_players * self.n_profiles,):
            array = array.reshape(self.payoff_shape)
        if array.shape != self.payoff_shape:
            raise ValueError(
                f"payoffs must have shape {self.payoff_shape}, got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("payoffs must be finite")
        return array

    def _as_flow(self, flow: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(flow, dtype=float)
        if array.shape != (len(self.edges),):
            raise ValueError(
                f"flow must have shape ({len(self.edges)},), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("flow must be finite")
        return array

    def normalize_profile(self, function: np.ndarray, player: int) -> np.ndarray:
        """Project a profile function onto player ``i``'s mu-normalized space."""

        array = np.asarray(function, dtype=float)
        if array.shape != self.n_actions:
            raise ValueError(
                f"profile function must have shape {self.n_actions}, got {array.shape}"
            )
        weighted_mean = np.tensordot(
            self.mu[player], array, axes=((0,), (player,))
        ) / self.action_masses[player]
        return array - np.expand_dims(weighted_mean, axis=player)

    def normalize_game(
        self, payoffs: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        """Return the mu-normalized strategic representative of ``payoffs``."""

        game = self._as_game(payoffs)
        return np.stack(
            [
                self.normalize_profile(game[player], player)
                for player in range(self.n_players)
            ]
        )

    def nonstrategic_component(
        self, payoffs: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        """Return the own-action-independent component of ``payoffs``."""

        game = self._as_game(payoffs)
        return game - self.normalize_game(game)

    def potential_game(self, potential_function: np.ndarray) -> np.ndarray:
        """Build the unique mu-normalized potential game induced by a function."""

        return np.stack(
            [
                self.normalize_profile(potential_function, player)
                for player in range(self.n_players)
            ]
        )

    def game_flow(self, payoffs: np.ndarray | Sequence[float]) -> np.ndarray:
        """Embed a game into Abdou's weighted response-graph flow space."""

        game = self._as_game(payoffs).reshape(self.n_players, self.n_profiles)
        payoff_differences = (
            game[self.edge_players, self.edge_heads]
            - game[self.edge_players, self.edge_tails]
        )
        return self.gradient_scales * payoff_differences

    def flow_inner_product(
        self,
        first: np.ndarray | Sequence[float],
        second: np.ndarray | Sequence[float],
    ) -> float:
        """Inner product from Abdou Eq. (37), using one orientation per edge."""

        first_flow = self._as_flow(first)
        second_flow = self._as_flow(second)
        return float(np.dot(self.flow_weights * first_flow, second_flow))

    def flow_norm(self, flow: np.ndarray | Sequence[float]) -> float:
        return float(np.sqrt(max(0.0, self.flow_inner_product(flow, flow))))

    def profile_inner_product(
        self, first: np.ndarray, second: np.ndarray
    ) -> float:
        first_array = np.asarray(first, dtype=float).reshape(self.n_profiles)
        second_array = np.asarray(second, dtype=float).reshape(self.n_profiles)
        return float(np.dot(self.profile_weights * first_array, second_array))

    def game_inner_product(
        self,
        first: np.ndarray | Sequence[float],
        second: np.ndarray | Sequence[float],
    ) -> float:
        """Game-space inner product of Abdou Eq. (15) with gamma equal to one."""

        first_game = self._as_game(first).reshape(self.n_players, self.n_profiles)
        second_game = self._as_game(second).reshape(self.n_players, self.n_profiles)
        value = 0.0
        for player in range(self.n_players):
            value += self.action_masses[player] * np.dot(
                self.profile_weights * first_game[player], second_game[player]
            )
        return float(value)

    def game_norm(self, payoffs: np.ndarray | Sequence[float]) -> float:
        return float(np.sqrt(max(0.0, self.game_inner_product(payoffs, payoffs))))

    def potential_function(self, flow: np.ndarray | Sequence[float]) -> np.ndarray:
        """Return ``delta^dagger flow`` with a mu-weighted zero-mean gauge."""

        weighted_flow = self.flow_weights * self._as_flow(flow)
        right_hand_side = self.gradient_matrix.T @ weighted_flow
        function = self._weighted_laplacian_pinv @ right_hand_side

        function -= np.dot(self.profile_weights, function) / self.profile_weights.sum()
        return function.reshape(self.n_actions)

    def decompose_flow(self, flow: np.ndarray | Sequence[float]) -> FlowDecomposition:
        """Project a flow onto exact and co-closed subspaces.

        For game-induced flows the co-closed component is the harmonic flow.
        For an arbitrary flow it can also contain locally inconsistent
        (coexact) circulation.
        """

        input_flow = self._as_flow(flow)
        potential_function = self.potential_function(input_flow)
        exact = self.gradient_matrix @ potential_function.reshape(-1)
        co_closed = input_flow - exact
        exact_norm = self.flow_norm(exact)
        co_closed_norm = self.flow_norm(co_closed)
        norm_sum = exact_norm + co_closed_norm
        potentialness = exact_norm / norm_sum if norm_sum else float("nan")
        return FlowDecomposition(
            exact=exact,
            co_closed=co_closed,
            potential_function=potential_function,
            potentialness=potentialness,
        )

    def codifferential(self, flow: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply the adjoint ``delta*`` associated with the weighted inner products."""

        weighted_flow = self.flow_weights * self._as_flow(flow)
        result = self.gradient_matrix.T @ weighted_flow
        result /= self.profile_weights
        return result.reshape(self.n_actions)

    def decompose(self, payoffs: np.ndarray | Sequence[float]) -> Decomposition:
        """Compute the orthogonal potential-harmonic-nonstrategic decomposition."""

        game = self._as_game(payoffs)
        strategic = self.normalize_game(game)
        nonstrategic = game - strategic

        flow = self.game_flow(strategic)
        flow_decomposition = self.decompose_flow(flow)
        potential_function = flow_decomposition.potential_function
        potential = self.potential_game(potential_function)
        harmonic = strategic - potential

        potential_flow = flow_decomposition.exact
        harmonic_flow = flow_decomposition.co_closed

        potential_norm = self.game_norm(potential)
        harmonic_norm = self.game_norm(harmonic)
        norm_sum = potential_norm + harmonic_norm
        potentialness = potential_norm / norm_sum if norm_sum else float("nan")

        return Decomposition(
            nonstrategic=nonstrategic,
            potential=potential,
            harmonic=harmonic,
            potential_function=potential_function,
            flow=flow,
            potential_flow=potential_flow,
            harmonic_flow=harmonic_flow,
            potentialness=potentialness,
        )


def decompose(
    payoffs: np.ndarray,
    mu: Sequence[Iterable[float]] | None = None,
) -> Decomposition:
    """Convenience API that infers action counts from a payoff tensor."""

    array = np.asarray(payoffs, dtype=float)
    if array.ndim < 2 or array.shape[0] != array.ndim - 1:
        raise ValueError(
            "payoffs must have shape (n_players, action_1, ..., action_n)"
        )
    return GameGeometry(array.shape[1:], mu=mu).decompose(array)
