# Decomposition Benchmarks

This directory contains the reproducible experiments used in Section 3 of the
computational implementation note.

From the repository root, run:

```bash
python CandoganDecomposition/Documentation/experiments/benchmark_decomposition.py
python CandoganDecomposition/Documentation/experiments/plot_benchmarks.py
```

`benchmark_decomposition.py` uses deterministic random weighted games. It
records geometry-construction time separately from the time for one
decomposition after the geometry and Laplacian pseudoinverse have been cached.
The CSV also records numerical reconstruction and co-closedness residuals.

`plot_benchmarks.py` reads the CSV and regenerates all PDF/PNG figures and the
LaTeX table included in the preprint.

Timing results depend on hardware, Python, NumPy, BLAS, and system load. The
metadata JSON records the software environment used for the committed data.
