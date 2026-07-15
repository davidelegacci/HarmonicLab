# Research Layer

`symbolic.py` contains the maintained SymPy interface for exact small-game
calculations. It constructs the weighted harmonic and normalization operators
directly from Abdou's payoff-level equations, without importing the numerical
core or converting through floating point.

The older exploratory stack remains one directory above for reproducibility:

- `normal_game_full.py` and `normal_game_minimal_euclidean.py`;
- `metric.py` and `solve_linear_system.py`;
- notebooks, plotting utilities, and `run_*.py` experiments.

Those modules mix symbolic algebra, plotting, equilibrium computation, and
verbose diagnostics. They are optional and are not imported when using
`CandoganDecomposition.GameGeometry`.
