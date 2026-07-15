# Harmonic Master-Equation Notebooks

This directory studies harmonic and strategically equivalent weighted
zero-sum games directly through the harmonic conservation equation, without
constructing the response-graph Hodge complex.

For every pure action profile `a`, the notebooks impose

```text
sum_i sum_(b_i in A_i) mu_(i,b_i)
    [u_i(a) - u_i(b_i, a_-i)] = 0,
```

where each action measure `mu_(i,b_i)` is strictly positive. Depending on the
notebook, the unknowns are the payoff entries, the measures, or a strategically
equivalent zero-sum representation.

## Notebook Guide

1. `1_harmonic_and_zerosum_N_player.ipynb` asks whether a given game is
   strategically equivalent to a weighted zero-sum game and solves for the
   zero-sum and nonstrategic terms.
2. `2_find_harmonic_measure.ipynb` solves for measures that make a given game
   harmonic. Its parametric solutions also describe candidate equilibrium
   weights.
3. `3_generate_harmonic_game.ipynb` fixes positive measures and solves the
   linear payoff system to generate harmonic games.
4. `4_is_harmonic_given_measure.ipynb` evaluates the conservation equations
   for a supplied payoff and measure.
5. `5_find_zerosum_given_harmonic.ipynb` starts from a generalized harmonic
   game and searches for a strategically equivalent weighted zero-sum game.
6. `6_generate_harmonic_game_symbolic_measure.ipynb` keeps measures symbolic,
   solves the payoff family, and permits numerical measures to be substituted
   afterward.

Additional material:

- `generate_harmonic_game_clean.ipynb` is a compact fixed-measure generator.
- Files containing `PLAYGROUND` and `playground.ipynb` retain exploratory
  variants and examples.
- `deprecated/harmonic_and_zerosum_2_player.ipynb` is the superseded
  two-player-specific workflow.
- [`harmonic_variety.md`](harmonic_variety.md) records symbolic payoff-measure
  families for low-dimensional skeletons.

## Typical Workflow

Within a notebook:

1. Set the game skeleton, for example `[2, 3]` or `[2, 2, 2]`.
2. Construct players, pure actions, and pure profiles.
3. Create player-major payoff symbols and action-measure symbols.
4. Build one conservation equation per pure profile.
5. Use SymPy to solve for the selected unknowns.
6. Substitute positive numerical measures or free payoff parameters.
7. Verify the resulting equations exactly or numerically.

The symbolic systems grow with both the number of profiles and the number of
payoff variables. Start with small skeletons and preserve exact rational values
where possible.

## Conventions

- Skeleton entry `A_i` is the number of actions of player `i`.
- Pure profiles follow `itertools.product` order.
- A flat payoff vector is player-major and has
  `number_of_players * number_of_profiles` entries.
- Measures are stored as one positive vector per player.
- SymPy symbols named like `h03` or `mu12` encode a player index followed by a
  profile or action index; see `harmonic_variety.md` for the precise notation
  used in recorded symbolic families.

## Environment

Use a Python 3 Jupyter kernel with NumPy, SymPy, and Matplotlib. The notebooks
are self-contained rather than imported as a package.

Some research cells depend on local modules outside this repository:

- `1_harmonic_and_zerosum_N_player.ipynb` references `aspera.utils`;
- `5_find_zerosum_given_harmonic.ipynb` references `gamelab.finitegames`;
- `playground.ipynb` also imports `gamelab.finitegames`.

Those notebooks require the corresponding external project or must be adapted
to local equivalents. Stored notebook output remains useful even when an
external module is unavailable.

## Relationship to CandoganDecomposition

These notebooks construct and characterize harmonic games directly. The
maintained [`../../CandoganDecomposition/`](../../CandoganDecomposition/)
package instead decomposes an arbitrary payoff tensor. Its
`CandoganDecomposition.research` layer exposes exact harmonic and normalization
operators for small games and is the preferred reusable symbolic API.
