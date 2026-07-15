# Computational Decomposition Note

`computational_decomposition.tex` is a self-contained computational companion
to the theoretical papers by Candogan et al. and Abdou et al. It documents the
exact bridge from the abstract finite-game operators to the maintained NumPy
implementation, derives dense time and storage bounds, and reports reproducible
scaling experiments.

Compile from this directory:

```bash
pdflatex computational_decomposition.tex
pdflatex computational_decomposition.tex
```

The `experiments/` directory contains the benchmark program, raw CSV data,
software metadata, generated table, and publication figures. See its README
for reproduction commands.
