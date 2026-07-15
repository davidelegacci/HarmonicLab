# Finite-Game Potential and Harmonic Decomposition

This directory implements the orthogonal decomposition of finite normal-form
games into

1. a nonstrategic component;
2. a normalized potential component;
3. a normalized harmonic component.

The maintained numerical implementation supports both:

- the uniform decomposition of Candogan et al. (2011); and
- the weighted decomposition of Abdou et al. (2022), parameterized by a
  strictly positive action measure `mu`.

The default measure is counting measure: every action has weight one. This is
the convention that reproduces the Candogan decomposition.

Opponent comeasures `gamma` are intentionally not part of the maintained API.
The convention in this project absorbs them into the chosen strategically
equivalent payoff representative. Action measures `mu` remain explicit because
they change normalization, the flow geometry, and the harmonic component.

## What Changed

The original research code works, but combines numerical decomposition,
symbolic algebra, graph construction, equilibrium computation, plotting,
printing, and experiment-specific behavior in large classes. It also builds
several dense generalized pseudoinverses and projection matrices.

The refactor introduces three distinct layers.

### Maintained NumPy core

`decomposition.py` contains the quiet numerical implementation. It:

- imports only NumPy;
- supports arbitrary positive action measures `mu`;
- constructs the weighted profile, flow, and game inner products explicitly;
- exposes the weighted gradient, game-flow embedding, and codifferential;
- computes normalization directly along each player's action axis;
- projects flows using a profile-sized weighted graph Laplacian;
- avoids the large payoff-comparison pseudoinverse;
- avoids constructing a dense edge-by-edge exact-flow projector;
- caches geometry so it can efficiently decompose many payoff tensors with
  the same action sets and measures;
- performs no printing, plotting, filesystem writes, or symbolic computation.

### Symbolic research layer

`research/symbolic.py` contains exact SymPy routines for small games. It builds:

- the payoff-level weighted harmonic operator;
- the weighted normalization operator; and
- an exact basis of normalized weighted harmonic games.

SymPy is imported only when this research layer is used.

### Retained legacy research code

The following files remain available under `legacy/` for reproducing earlier
experiments:

- `normal_game_full.py`;
- `normal_game_minimal_euclidean.py`;
- `metric.py`;
- `solve_linear_system.py`;
- the notebooks and `run_*.py` scripts.

They are not imported by the maintained API. Their package-local imports were
adjusted so that the principal modules remain importable through
`CandoganDecomposition.legacy`, while direct execution from the legacy
directory remains possible.

## Requirements

Activate the existing virtual environment:

```bash
source ~/venv/bin/activate
```

The numerical core requires NumPy. Tests require pytest. The symbolic research
layer additionally requires SymPy.

The current environment can be checked with:

```bash
python -c "import numpy; print(numpy.__version__)"
python -m pytest --version
python -c "import sympy; print(sympy.__version__)"
```

Run examples from the `HarmonicLab` root so that the package is importable:

```bash
cd /Users/davidelegacci/RESEARCH/phd/phd-research/phd-code/game_theory_geometry/HarmonicDecomposition/HarmonicLab
```

## Payoff Representation

For an `n`-player game with action counts

```text
(A_1, A_2, ..., A_n)
```

the preferred payoff representation is a NumPy array with shape

```text
(n, A_1, A_2, ..., A_n).
```

The entry

```python
payoffs[i, a_1, ..., a_n]
```

is player `i`'s payoff at pure profile `(a_1, ..., a_n)`. Player and action
indices are zero-based.

For example, a two-player `2 x 3` game has payoff shape `(2, 2, 3)`:

```python
import numpy as np

payoffs = np.array(
    [
        # Player 0
        [
            [3, 0, 1],
            [2, 4, -1],
        ],
        # Player 1
        [
            [1, 2, 0],
            [5, -2, 3],
        ],
    ],
    dtype=float,
)
```

`GameGeometry.decompose` also accepts a flat player-major vector. For a game
with `A = A_1 * ... * A_n` profiles, the flat vector has length `n * A` and is
reshaped in NumPy C order. Tensor input is preferred because its axes make the
game structure explicit.

## Uniform Candogan Decomposition

Use the default `mu=None` or explicitly provide all-one measures:

```python
from CandoganDecomposition import GameGeometry

geometry = GameGeometry([2, 3])
result = geometry.decompose(payoffs)
```

This is equivalent to:

```python
geometry = GameGeometry(
    [2, 3],
    mu=[
        [1, 1],
        [1, 1, 1],
    ],
)
```

The result contains:

```python
result.nonstrategic
result.potential
result.harmonic
result.potential_function
result.potentialness
```

The short aliases used by the old code are also available:

```python
result.uN  # result.nonstrategic
result.uP  # result.potential
result.uH  # result.harmonic
```

Reconstruction satisfies:

```python
np.allclose(
    payoffs,
    result.nonstrategic + result.potential + result.harmonic,
)
```

## Weighted Abdou Decomposition

Pass one strictly positive action-measure vector per player:

```python
geometry = GameGeometry(
    n_actions=[2, 3],
    mu=[
        [1, 2],
        [1, 3, 2],
    ],
)

result = geometry.decompose(payoffs)
```

Here:

```text
mu[0][0] = 1
mu[0][1] = 2

mu[1][0] = 1
mu[1][1] = 3
mu[1][2] = 2
```

Measures must be finite and strictly positive. They are deliberately not
normalized internally.

This distinction matters:

- multiplying every measure of every player by the same positive constant
  leaves the decomposition unchanged;
- independently normalizing each player's measure can change the relative
  weighting of player-specific deviations and therefore change the
  potential/harmonic projection;
- to reproduce Candogan, use all-one counting measures rather than separately
  normalized uniform probability vectors.

## One-Off Convenience Function

When payoff tensors already carry all action dimensions, use `decompose` to
infer the game skeleton:

```python
from CandoganDecomposition import decompose

result = decompose(
    payoffs,
    mu=[
        [1, 2],
        [1, 3, 2],
    ],
)
```

For repeated decompositions, construct `GameGeometry` once. Geometry setup is
the expensive step; reusing it avoids rebuilding the graph and weighted
Laplacian.

```python
geometry = GameGeometry([2, 3], mu=[[1, 2], [1, 3, 2]])

first_result = geometry.decompose(first_payoffs)
second_result = geometry.decompose(second_payoffs)
third_result = geometry.decompose(third_payoffs)
```

## Decomposition Result

`GameGeometry.decompose` returns an immutable `Decomposition` container. Its
arrays are:

| Attribute | Meaning |
| --- | --- |
| `nonstrategic` | Own-action-independent component selected by weighted normalization |
| `potential` | Weighted-normalized potential component |
| `harmonic` | Weighted-normalized harmonic component |
| `strategic` | Convenience property equal to `potential + harmonic` |
| `potential_function` | Profile function generating the exact component |
| `flow` | Weighted game-induced flow of the strategic game |
| `potential_flow` | Exact projection of `flow` |
| `harmonic_flow` | Co-closed residual flow |
| `potentialness` | Weighted norm ratio described below |

The returned potential function is fixed to the gauge

```text
sum_a mu(a) * potential_function(a) = 0.
```

Adding a constant would generate the same potential game.

### Potentialness

Potentialness uses the weighted game-space norm associated with `mu`:

```text
                  ||uP||
potentialness = -------------.
                ||uP|| + ||uH||
```

It is:

- zero for a purely harmonic strategic game;
- one for a purely potential strategic game;
- `nan` for a purely nonstrategic game, because both strategic norms vanish.

This is not a naive Euclidean norm applied after a weighted decomposition. The
same inner-product data used to define the decomposition is used to measure its
components.

## Mathematical Conventions

Let

```text
A = A_1 x ... x A_n
```

be the pure-profile space, and let the positive action measures be `mu_i`.
Their product measure is

```text
mu(a) = product_i mu_i(a_i).
```

### Weighted normalization

For player `i` and fixed opponent profile `a_-i`, the nonstrategic component is
the normalized weighted own-action average

```text
Lambda_i u_i(a_i, a_-i)

      sum_bi mu_i(b_i) u_i(b_i, a_-i)
    = ---------------------------------.
               sum_bi mu_i(b_i)
```

The strategic representative is

```text
Pi_i u_i = u_i - Lambda_i u_i.
```

Consequently, both maintained strategic components satisfy

```text
sum_ai mu_i(a_i) u_i^P(a_i, a_-i) = 0,
sum_ai mu_i(a_i) u_i^H(a_i, a_-i) = 0.
```

### Profile inner product

For profile functions `f` and `h`,

```text
<f, h>_0 = sum_a mu(a) f(a) h(a).
```

### Response-graph flow inner product

The response graph connects profiles differing in exactly one player's action.
The implementation stores one orientation for each undirected edge. For stored
edge `(a, b)`,

```text
<X, Y>_1 = sum_edges mu(a) mu(b) X(a,b) Y(a,b).
```

This equals Abdou's one-half sum over both orientations because flows are
skew-symmetric.

The corresponding diagonal weights are exposed as:

```python
geometry.flow_weights
```

### Weighted gradient

If edge `(a, b)` belongs to player `i`, define

```text
W_i(a,b) = 1 / sqrt(mu_-i(a_-i)).
```

The weighted gradient is

```text
delta f(a,b) = W_i(a,b) [f(b) - f(a)].
```

Its matrix is:

```python
geometry.gradient_matrix
```

### Weighted game-flow embedding

With uniform comeasure, a game is embedded as

```text
D(u)(a,b) = W_i(a,b) [u_i(b) - u_i(a)]
```

on a player-`i` edge. Compute it with:

```python
flow = geometry.game_flow(payoffs)
```

### Codifferential

The codifferential is the adjoint of the weighted gradient for the profile and
flow inner products above:

```python
divergence = geometry.codifferential(flow)
```

The harmonic component satisfies, up to floating-point tolerance:

```python
np.allclose(
    geometry.codifferential(result.harmonic_flow),
    0.0,
)
```

### Weighted game inner product

With uniform comeasure, the game-space inner product is

```text
<u, v>_game

  = sum_i mu_i(A_i) sum_a mu(a) u_i(a) v_i(a),
```

where

```text
mu_i(A_i) = sum_ai mu_i(a_i).
```

It is available as:

```python
geometry.game_inner_product(first_game, second_game)
geometry.game_norm(payoffs)
```

The three returned components are mutually orthogonal in this inner product.

## Flow-Space API

The graph geometry can be used independently of payoff decomposition.

```python
flow = geometry.game_flow(payoffs)
flow_result = geometry.decompose_flow(flow)

flow_result.exact
flow_result.co_closed
flow_result.potential_function
flow_result.potentialness
```

The result satisfies:

```python
np.allclose(
    flow,
    flow_result.exact + flow_result.co_closed,
)

np.allclose(
    geometry.codifferential(flow_result.co_closed),
    0.0,
)

np.isclose(
    geometry.flow_inner_product(
        flow_result.exact,
        flow_result.co_closed,
    ),
    0.0,
)
```

For game-induced flows, the co-closed residual is the harmonic flow. For an
arbitrary response-graph flow, the co-closed residual can also contain
locally inconsistent, coexact circulation.

Additional exposed objects include:

```python
geometry.profiles
geometry.edges
geometry.incidence_matrix
geometry.gradient_scales
geometry.gradient_matrix
geometry.profile_weights
geometry.flow_weights
geometry.action_masses
```

Each item in `geometry.edges` is a `ResponseEdge` with:

```python
edge.player
edge.tail
edge.head
```

## Symbolic Research API

The symbolic layer is intended for exact small-game calculations, not for
large numerical experiments.

```python
from CandoganDecomposition.research import (
    harmonic_operator,
    normalization_operator,
    normalized_harmonic_basis,
)
```

### Harmonic operator

```python
H = harmonic_operator(
    [2, 3],
    mu=[
        [1, 2],
        [1, 3, 2],
    ],
)
```

For a player-major payoff column `u`, the equation

```python
H * u == 0
```

is the weighted harmonic condition.

### Normalization operator

```python
N = normalization_operator(
    [2, 3],
    mu=[
        [1, 2],
        [1, 3, 2],
    ],
)
```

The kernel of `N` is the space of `mu`-normalized games.

### Exact normalized harmonic basis

```python
basis = normalized_harmonic_basis(
    [2, 3],
    mu=[
        [1, 2],
        [1, 3, 2],
    ],
)

for vector in basis:
    print(vector)
```

The return value is a list of SymPy column vectors. Measures are converted with
`sympy.sympify`, so rational and integer weights remain exact.

## Migrating from the Legacy Classes

The old workflow typically constructed a large game class and then a payoff
class:

```python
from CandoganDecomposition.legacy import normal_game_full as ng

game = ng.GameFull([2, 3], **config)
payoff = ng.PayoffFull(game, payoff_vector, **config)

payoff.uN
payoff.uP
payoff.uH
payoff.potential
```

The maintained numerical replacement is:

```python
from CandoganDecomposition import GameGeometry

geometry = GameGeometry([2, 3], mu=[[1, 1], [1, 1, 1]])
result = geometry.decompose(payoff_vector)

result.uN
result.uP
result.uH
result.potential_function
```

For a weighted decomposition:

```python
geometry = GameGeometry(
    [2, 3],
    mu=[[1, 2], [1, 3, 2]],
)
result = geometry.decompose(payoff_vector)
```

Features such as Nash equilibrium enumeration, plotting, LaTeX graph output,
auction experiments, and arbitrary experimental metric matrices remain in the
legacy research files. They are deliberately not dependencies of the clean
decomposition core.

## Tests

Run the complete repository test suite from `HarmonicLab`:

```bash
source ~/venv/bin/activate
python -m pytest -q
```

Or run only these tests:

```bash
python -m pytest CandoganDecomposition/tests -q
```

The current suite covers:

- regression against the original dense uniform Candogan projection for
  `2 x 2`, `2 x 3`, `3 x 3`, and `2 x 2 x 2` games;
- exact reconstruction of arbitrary games;
- weighted normalization of potential and harmonic components;
- nonstrategic-flow invariance;
- exactness of the potential flow;
- co-closedness of the harmonic flow;
- adjointness of the weighted gradient and codifferential;
- mutual orthogonality in the weighted game inner product;
- the isometry between normalized games and their embedded flows;
- the weighted harmonic balance equation;
- invariance under a common global rescaling of all action measures;
- canonical pure potential and matching-pennies harmonic examples;
- payoff and measure validation;
- agreement between exact symbolic and numerical weighted operators;
- theoretical dimensions of normalized harmonic spaces.

At the time of this refactor, all 25 tests pass.

## Numerical Design and Complexity

Let:

```text
A = product_i A_i
```

be the number of pure profiles, and let `E` be the number of undirected
response-graph edges:

```text
E = sum_i [product_(j != i) A_j] * binomial(A_i, 2).
```

`GameGeometry` stores:

- an `E x A` incidence matrix;
- an `E x A` weighted gradient matrix;
- an `A x A` weighted Laplacian pseudoinverse;
- edge and measure metadata.

It does not store:

- the legacy `E x (nA)` payoff-comparison pseudoinverse; or
- an `E x E` exact-flow projection matrix.

The implementation is therefore substantially smaller and faster for repeated
decompositions, while keeping all metric choices visible.

The current implementation uses dense NumPy arrays. Very large action spaces
will eventually require sparse incidence matrices and sparse Laplacian solves.

## Validation and Common Errors

### Wrong payoff shape

If a `2 x 3` game is supplied as `(2, 3, 2)` instead of `(2, 2, 3)`, the code
raises a shape error. The leading axis is always the player axis; subsequent
axes follow player order.

```python
geometry = GameGeometry([2, 3])
assert geometry.payoff_shape == (2, 2, 3)
```

### Invalid measures

The following are rejected:

- missing player measures;
- a measure vector with the wrong number of actions;
- zero or negative weights;
- infinite or `nan` weights.

### Unexpected weighted result

Check whether measures were normalized independently. For example,

```python
mu=[[1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3]]
```

is not the same parameter choice as all-one counting measure for purposes of
the full decomposition. Use:

```python
mu=[[1, 1], [1, 1, 1]]
```

to reproduce Candogan.

### Purely nonstrategic potentialness

If the strategic component is zero, `potentialness` is `nan`, not zero. There
is no potential-versus-harmonic ratio when both norms vanish.

### Floating-point residuals

Harmonic and orthogonality checks should use `np.allclose` or `np.isclose`.
The decomposition is computed through a floating-point pseudoinverse, so exact
mathematical zeros generally appear at machine precision.

## Current Scope

The maintained core intentionally does not provide:

- opponent comeasure `gamma` parameters;
- symbolic payoff decomposition for large games;
- Nash equilibrium computation;
- plotting or graph visualization;
- experiment configuration loading;
- persistence of cached matrices;
- sparse large-game solvers.

These boundaries keep the numerical decomposition small, deterministic, and
reusable. Research-only capabilities can be added under `research/` without
adding dependencies or side effects to the core.

## Directory Map

```text
CandoganDecomposition/
|-- decomposition.py                 maintained NumPy core
|-- __init__.py                      public numerical exports
|-- research/
|   |-- __init__.py                  public symbolic exports
|   |-- symbolic.py                  exact weighted operators and bases
|   `-- README.md                    research-layer notes
|-- tests/
|   |-- test_decomposition.py        numerical and regression tests
|   `-- test_symbolic.py             exact symbolic tests
|-- legacy/
|   |-- normal_game_full.py              exploratory implementation
|   |-- normal_game_minimal_euclidean.py uniform implementation
|   |-- metric.py                        generalized metric helpers
|   |-- solve_linear_system.py           SymPy helpers
|   |-- run_*.py                         historical experiments
|   |-- *.ipynb                          exploratory notebooks
|   `-- README.md                        archive status and contents
`-- README.md                        this guide
```

## References

- Candogan, O., Menache, I., Ozdaglar, A., and Parrilo, P. A. (2011),
  "Flows and Decompositions of Games: Harmonic and Potential Games",
  Mathematics of Operations Research.
- Abdou, J., Pnevmatikos, N., Scarsini, M., and Venel, X. (2022),
  "Decomposition of Games: Some Strategic Considerations",
  Mathematics of Operations Research.
- M. Oberlechner's cleanup is used as a broad behavioral and architectural
  reference rather than copied as a tight implementation template:
  <https://github.com/MOberlechner/games_decomposition>.
