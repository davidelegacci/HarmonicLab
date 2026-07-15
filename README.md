# Harmonic Lab

## Plug and Play

From the repository root:

```python
import numpy as np
from CandoganDecomposition import GameGeometry

skeleton = [2, 3]  # two players with 2 and 3 actions
payoffs = np.array(
    [
        [[3, 0, 1], [2, 4, -1]],  # player 0
        [[1, 2, 0], [5, -2, 3]],  # player 1
    ],
    dtype=float,
)  # shape: (players, *skeleton); payoffs[i, a0, a1] = u_i(a0, a1)

mu = [[1, 2], [1, 3, 2]]  # one positive weight per action; use None for uniform
result = GameGeometry(skeleton, mu=mu).decompose(payoffs)

result.nonstrategic
result.potential
result.harmonic
result.potential_function
result.potentialness

assert np.allclose(
    payoffs,
    result.nonstrategic + result.potential + result.harmonic,
)
```

Harmonic Lab is a research repository for finite normal-form games. It studies
the decomposition of a game into nonstrategic, potential, and harmonic
components, together with symbolic constructions of weighted harmonic games
and their relation to strategically equivalent weighted zero-sum games.

## Start Here

The repository has two main areas:

- [`CandoganDecomposition/`](CandoganDecomposition/) contains the maintained
  numerical decomposition, an exact symbolic research layer, tests, and an
  archive of the earlier exploratory implementation. Its
  [README](CandoganDecomposition/README.md) is the main technical guide.
- [`HarmonicAndZeroSum/`](HarmonicAndZeroSum/) contains symbolic notebooks and
  scripts used to generate harmonic games, solve for harmonic measures, and
  study strategic equivalence with weighted zero-sum games.

For new numerical decomposition work, use `CandoganDecomposition.GameGeometry`.

## Mathematical Scope

For a finite game with payoff tensor `u`, the maintained implementation
computes the orthogonal decomposition

```text
u = nonstrategic + potential + harmonic.
```

The default geometry is the uniform decomposition of Candogan et al. Positive
action measures `mu` select the weighted decomposition of Abdou et al.

The notebook laboratory also works directly with the weighted harmonic
conservation equation. For every pure profile `a`,

```text
sum_i sum_(b_i in A_i) mu_(i,b_i)
    [u_i(a) - u_i(b_i, a_-i)] = 0.
```

These are complementary workflows: one decomposes an arbitrary game, while
the other symbolically describes or generates games satisfying harmonic
conditions.

## Repository Map

```text
HarmonicLab/
|-- CandoganDecomposition/
|   |-- decomposition.py       maintained NumPy core
|   |-- research/              maintained exact SymPy operators
|   |-- tests/                 numerical and symbolic tests
|   |-- legacy/                archived exploratory code and experiments
|   `-- README.md              decomposition API and mathematics
|-- HarmonicAndZeroSum/
|   |-- generate_harmonic_codifferential/
|   `-- generate_harmonic_master_equation/
|-- CITATION.cff
`-- README.md
```

## Quick Start

From the repository root, activate a Python environment containing NumPy and
pytest. SymPy is additionally required for the exact research layer.

```bash
source ~/venv/bin/activate
python -m pytest CandoganDecomposition/tests -q
```

A minimal decomposition is:

```python
import numpy as np

from CandoganDecomposition import GameGeometry

payoffs = np.array(
    [
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    ],
    dtype=float,
)

result = GameGeometry([2, 2]).decompose(payoffs)

assert np.allclose(
    payoffs,
    result.nonstrategic + result.potential + result.harmonic,
)
```

See the decomposition guide for weighted measures, flow-space operations,
mathematical conventions, complexity, and the complete API.

## Maintained and Exploratory Code

The maintained interface is deliberately quiet and depends only on NumPy.
Exact small-game calculations live under `CandoganDecomposition.research`.

Older all-in-one classes, plotting routines, equilibrium experiments, and
notebooks are retained under `CandoganDecomposition/legacy` for provenance.
They are not dependencies of the maintained API. The `HarmonicAndZeroSum`
notebooks are active research documents rather than a packaged library.

## References

- Candogan, O., Menache, I., Ozdaglar, A., and Parrilo, P. A. (2011),
  "Flows and Decompositions of Games: Harmonic and Potential Games",
  *Mathematics of Operations Research*.
  <https://doi.org/10.1287/moor.1110.0500>
- Abdou, J., Pnevmatikos, N., Scarsini, M., and Venel, X. (2022),
  "Decomposition of Games: Some Strategic Considerations",
  *Mathematics of Operations Research*.
- The original decomposition routine was informed by M. Oberlechner's
  [`games_decomposition`](https://github.com/MOberlechner/games_decomposition)
  cleanup.

Citation metadata for this repository is available in [`CITATION.cff`](CITATION.cff).
