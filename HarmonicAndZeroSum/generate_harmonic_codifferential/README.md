# Symbolic Harmonic Games from the Codifferential

This directory contains an exploratory symbolic implementation of finite
normal-form games based on payoff fields, differential forms, and the
codifferential. It was used to derive exact and coexact conditions and to
generate small examples before the direct master-equation notebooks became the
main symbolic workflow.

For the maintained numerical decomposition, use
[`../../CandoganDecomposition/`](../../CandoganDecomposition/). For the direct
harmonic-equation workflow, use
[`../generate_harmonic_master_equation/`](../generate_harmonic_master_equation/).

## Files

- `generate_harmonic.py` defines symbolic `Player`, `Game`, and `Payoff`
  classes, constructs mixed-strategy payoff fields, derives exact and coexact
  systems, and prints symbolic or randomly instantiated solutions.
- `utils.py` provides list reshaping, symbolic-system solving, mixed-strategy
  generation, and LaTeX formatting helpers.

## Model and Representation

The module starts from a game skeleton such as `[2, 2]`. For `N` players and
`A` pure profiles, a payoff vector has `N * A` entries in player-major order.
The code constructs:

- pure and mixed strategy profiles;
- multilinear payoff extensions;
- payoff vector fields in full and reduced simplex coordinates;
- symbolic exactness equations;
- a codifferential polynomial whose coefficients determine the coexact
  system.

The script uses zero-based actions internally but formats much of its symbolic
output for mathematical notation.

## Running the Script

The game skeleton is the `SKELETON` constant near the top of
`generate_harmonic.py`. Because the script imports `utils.py` directly and
executes a symbolic example at module load time, run it from this directory:

```bash
cd HarmonicAndZeroSum/generate_harmonic_codifferential
python generate_harmonic.py
```

Required third-party packages are NumPy, SymPy, SciPy, and Matplotlib. Symbolic
construction and simplification can become slow as the game skeleton grows.

## Scope and Status

This is research code, not a reusable package or supported command-line tool.
It prints extensive LaTeX-oriented diagnostics and contains optional plotting
and replicator-dynamics experiments. Some experimental paths are marked as not
working in the source.

No automated tests currently cover this directory. Preserve it as a record of
the codifferential derivation, and use the maintained decomposition or
master-equation notebooks for new computations.
