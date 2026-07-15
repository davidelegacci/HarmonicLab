# Legacy Candogan Research Code

This directory preserves the exploratory implementation, experiments, and
research artifacts that preceded the maintained NumPy and SymPy interfaces in
the parent package. Nothing here is imported by `CandoganDecomposition`'s
public API.

Use the maintained implementation for new work:

```python
from CandoganDecomposition import GameGeometry, decompose
```

Use `CandoganDecomposition.research` for exact symbolic harmonic operators and
bases.

## Contents

- `normal_game_full.py` is the feature-rich historical implementation. It
  combines decomposition, generalized metrics, symbolic calculations, Nash
  equilibria, plotting, and verbose diagnostics.
- `normal_game_minimal_euclidean.py` is the earlier uniform/Euclidean variant.
- `metric.py`, `solve_linear_system.py`, and `utils.py` support those classes.
- `run_*.py`, `config.yml`, `first_price_auction.py`, and
  `playground_make_alpha_game.py` are experiment-specific entry points.
- `Untitled.ipynb` and `computational_complexity.ipynb` are exploratory
  notebooks.
- `cliques.py`, `number_3_cliques.py`, and `mp-inv.py` are standalone research
  utilities.
- `artifacts/` contains ignored local experiment output retained on this
  workstation. It is not part of the tracked source tree.
- `references/` contains ignored local reference material.

## Compatibility Status

The two `normal_game_*.py` modules retain package-aware imports and can be
imported, when their optional dependencies are installed, as:

```python
from CandoganDecomposition.legacy import normal_game_full
from CandoganDecomposition.legacy import normal_game_minimal_euclidean
```

The experiment scripts were written to run with this directory as their
working directory and use direct local imports. Several scripts import a
historical `normal_game` module that is no longer present; those scripts were
already non-runnable before archival and are retained as research records.

The legacy stack is not covered by the maintained numerical test suite. Its
location records provenance without implying current support.
