# Harmonic and Weighted Zero-Sum Games

This directory is a symbolic research laboratory for harmonic games and their
relationship with strategically equivalent weighted zero-sum games. It
contains two independent approaches: one derived from differential forms and
the codifferential, and one based directly on the harmonic master equation.

This code is exploratory and notebook-oriented. The maintained decomposition
of arbitrary games lives in [`../CandoganDecomposition/`](../CandoganDecomposition/).

## Harmonic Condition

Let player `i` have finite action set `A_i`, let `a` be a pure action profile,
and let every action have a positive weight `mu_(i,b_i)`. The notebooks use the
condition

```text
sum_i sum_(b_i in A_i) mu_(i,b_i)
    [u_i(a) - u_i(b_i, a_-i)] = 0
```

at every pure profile `a`. Depending on the task, payoffs or measures are
treated as symbolic unknowns.

## Directory Map

```text
HarmonicAndZeroSum/
|-- generate_harmonic_codifferential/
|   |-- generate_harmonic.py
|   |-- utils.py
|   `-- README.md
|-- generate_harmonic_master_equation/
|   |-- 1_harmonic_and_zerosum_N_player.ipynb
|   |-- 2_find_harmonic_measure.ipynb
|   |-- 3_generate_harmonic_game.ipynb
|   |-- 4_is_harmonic_given_measure.ipynb
|   |-- 5_find_zerosum_given_harmonic.ipynb
|   |-- 6_generate_harmonic_game_symbolic_measure.ipynb
|   |-- harmonic_variety.md
|   `-- README.md
`-- README.md
```

## Two Approaches

### Codifferential construction

[`generate_harmonic_codifferential/`](generate_harmonic_codifferential/)
builds symbolic payoff fields on mixed-strategy spaces. It derives exact and
coexact conditions through differential-form calculations and can generate
small symbolic examples. The implementation is self-contained but verbose and
intended for theoretical exploration.

### Master-equation notebooks

[`generate_harmonic_master_equation/`](generate_harmonic_master_equation/)
works directly with the conservation equation. Its notebooks cover four main
questions:

1. Is a given game strategically equivalent to a weighted zero-sum game?
2. Does a game admit positive measures for which it is harmonic?
3. Given measures, how can one generate a harmonic game?
4. Given a game and measures, does the harmonic equation hold?

The later notebooks also generate payoff families with symbolic measures and
record low-dimensional harmonic varieties.

## Conventions

- A game skeleton is a list such as `[2, 3]`, giving the number of actions of
  each player.
- Pure profiles are Cartesian products ordered in Python/NumPy product order.
- Flat payoff vectors are player-major: all payoffs of player 0, followed by
  all payoffs of player 1, and so on.
- Harmonic action measures must be strictly positive.
- Notebook computations use exact SymPy expressions when possible and NumPy
  arrays for numerical instances.

## Running the Research Notebooks

Select a Python 3 Jupyter kernel containing NumPy, SymPy, Matplotlib, and the
Jupyter/IPython stack. Open the notebooks in numerical order, but treat each as
an independent research document: parameters and example payoffs are defined
inside the notebook rather than through a common package.

Some exploratory cells reference external local modules named `aspera` or
`gamelab`. Those modules are not part of this repository; notebooks containing
such imports require the corresponding external project or local adaptation.

The notebooks contain stored outputs. Re-execution may be expensive for larger
game skeletons because symbolic linear systems grow quickly with the number of
players and profiles.

## Relationship to the Decomposition Package

The notebook equations characterize or construct harmonic games directly.
`CandoganDecomposition`, by contrast, accepts any finite game and projects it
onto nonstrategic, potential, and harmonic components. The exact symbolic
operators in `CandoganDecomposition.research` provide a maintained bridge
between the two viewpoints for small games.
