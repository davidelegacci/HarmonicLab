# Catalogue of Symbolic Harmonic Varieties

This file records the symbolic payoff-measure families produced by
`6_generate_harmonic_game_symbolic_measure.ipynb` before any numeric
substitution.

Notation follows the notebook:

- `h0k` is player 0's payoff at pure profile `k`.
- `h1k` is player 1's payoff at pure profile `k`.
- `h2k` is player 2's payoff at pure profile `k`, when present.
- `mu0j` is player 0's harmonic weight on action `a0j`.
- `mu1j` is player 1's harmonic weight on action `a1j`.
- `mu2j` is player 2's harmonic weight on action `a2j`, when present.
- All measures are assumed positive, so the displayed divisions by measures are
  well-defined.

Equivalently, each section describes the algebraic variety of payoff functions
and positive measures satisfying the harmonic conservation equations.

## Skeleton: 2x2

Pure profiles are ordered as:

$$
\begin{aligned}
1 &\leftrightarrow (a_{01}, a_{11}),\\
2 &\leftrightarrow (a_{01}, a_{12}),\\
3 &\leftrightarrow (a_{02}, a_{11}),\\
4 &\leftrightarrow (a_{02}, a_{12}).
\end{aligned}
$$

Measures:

$$
\mu_{01}, \mu_{02}, \mu_{11}, \mu_{12} > 0.
$$

Free payoff parameters:

$$
h_{03}, h_{04}, h_{12}, h_{13}, h_{14}.
$$

The symbolic solution dictionary is:

$$
\begin{aligned}
h_{01} &= h_{03} + \frac{h_{13} \mu_{12}}{\mu_{01}} - \frac{h_{14} \mu_{12}}{\mu_{01}},\\
h_{02} &= h_{04} - \frac{h_{13} \mu_{11}}{\mu_{01}} + \frac{h_{14} \mu_{11}}{\mu_{01}},\\
h_{03} &= h_{03},\\
h_{04} &= h_{04},\\
h_{11} &= h_{12} - \frac{h_{13} \mu_{02}}{\mu_{01}} + \frac{h_{14} \mu_{02}}{\mu_{01}},\\
h_{12} &= h_{12},\\
h_{13} &= h_{13},\\
h_{14} &= h_{14}.
\end{aligned}
$$

In payoff-function form:

$$
\begin{aligned}
u_0(a_{01}, a_{11}) &= h_{01},&
u_0(a_{01}, a_{12}) &= h_{02},&
u_0(a_{02}, a_{11}) &= h_{03},\\
u_0(a_{02}, a_{12}) &= h_{04},&
u_1(a_{01}, a_{11}) &= h_{11},&
u_1(a_{01}, a_{12}) &= h_{12},\\
u_1(a_{02}, a_{11}) &= h_{13},&
u_1(a_{02}, a_{12}) &= h_{14}.
\end{aligned}
$$

## Skeleton: 2x3

Pure profiles are ordered as:

$$
\begin{aligned}
1 &\leftrightarrow (a_{01}, a_{11}),\\
2 &\leftrightarrow (a_{01}, a_{12}),\\
3 &\leftrightarrow (a_{01}, a_{13}),\\
4 &\leftrightarrow (a_{02}, a_{11}),\\
5 &\leftrightarrow (a_{02}, a_{12}),\\
6 &\leftrightarrow (a_{02}, a_{13}).
\end{aligned}
$$

Measures:

$$
\mu_{01}, \mu_{02}, \mu_{11}, \mu_{12}, \mu_{13} > 0.
$$

Free payoff parameters:

$$
h_{04}, h_{05}, h_{06}, h_{13}, h_{14}, h_{15}, h_{16}.
$$

The symbolic solution dictionary is:

$$
\begin{aligned}
h_{01} &= h_{04} + \frac{h_{14} \left(\mu_{12} + \mu_{13}\right)}{\mu_{01}} - \frac{h_{15} \mu_{12}}{\mu_{01}} - \frac{h_{16} \mu_{13}}{\mu_{01}},\\
h_{02} &= h_{05} - \frac{h_{14} \mu_{11}}{\mu_{01}} + \frac{h_{15} \left(\mu_{11} + \mu_{13}\right)}{\mu_{01}} - \frac{h_{16} \mu_{13}}{\mu_{01}},\\
h_{03} &= h_{06} - \frac{h_{14} \mu_{11}}{\mu_{01}} - \frac{h_{15} \mu_{12}}{\mu_{01}} + \frac{h_{16} \left(\mu_{11} + \mu_{12}\right)}{\mu_{01}},\\
h_{04} &= h_{04},\\
h_{05} &= h_{05},\\
h_{06} &= h_{06},\\
h_{11} &= h_{13} - \frac{h_{14} \mu_{02}}{\mu_{01}} + \frac{h_{16} \mu_{02}}{\mu_{01}},\\
h_{12} &= h_{13} - \frac{h_{15} \mu_{02}}{\mu_{01}} + \frac{h_{16} \mu_{02}}{\mu_{01}},\\
h_{13} &= h_{13},\\
h_{14} &= h_{14},\\
h_{15} &= h_{15},\\
h_{16} &= h_{16}.
\end{aligned}
$$

In payoff-function form:

$$
\begin{aligned}
u_0(a_{01}, a_{11}) &= h_{01},&
u_0(a_{01}, a_{12}) &= h_{02},&
u_0(a_{01}, a_{13}) &= h_{03},\\
u_0(a_{02}, a_{11}) &= h_{04},&
u_0(a_{02}, a_{12}) &= h_{05},&
u_0(a_{02}, a_{13}) &= h_{06},\\
u_1(a_{01}, a_{11}) &= h_{11},&
u_1(a_{01}, a_{12}) &= h_{12},&
u_1(a_{01}, a_{13}) &= h_{13},\\
u_1(a_{02}, a_{11}) &= h_{14},&
u_1(a_{02}, a_{12}) &= h_{15},&
u_1(a_{02}, a_{13}) &= h_{16}.
\end{aligned}
$$

## Skeleton: 2x2x2

Pure profiles are ordered as:

$$
\begin{aligned}
1 &\leftrightarrow (a_{01}, a_{11}, a_{21}),\\
2 &\leftrightarrow (a_{01}, a_{11}, a_{22}),\\
3 &\leftrightarrow (a_{01}, a_{12}, a_{21}),\\
4 &\leftrightarrow (a_{01}, a_{12}, a_{22}),\\
5 &\leftrightarrow (a_{02}, a_{11}, a_{21}),\\
6 &\leftrightarrow (a_{02}, a_{11}, a_{22}),\\
7 &\leftrightarrow (a_{02}, a_{12}, a_{21}),\\
8 &\leftrightarrow (a_{02}, a_{12}, a_{22}).
\end{aligned}
$$

Measures:

$$
\mu_{01}, \mu_{02}, \mu_{11}, \mu_{12}, \mu_{21}, \mu_{22} > 0.
$$

Free payoff parameters:

$$
h_{05}, h_{06}, h_{07}, h_{08}, h_{13}, h_{14}, h_{15}, h_{16}, h_{17}, h_{18}, h_{22}, h_{23}, h_{24}, h_{25}, h_{26}, h_{27}, h_{28}.
$$

The symbolic solution dictionary is:

$$
\begin{aligned}
h_{01} &= h_{05} + \frac{h_{15} \mu_{12}}{\mu_{01}} - \frac{h_{17} \mu_{12}}{\mu_{01}} + \frac{h_{25} \mu_{22}}{\mu_{01}} - \frac{h_{26} \mu_{22}}{\mu_{01}},\\
h_{02} &= h_{06} + \frac{h_{16} \mu_{12}}{\mu_{01}} - \frac{h_{18} \mu_{12}}{\mu_{01}} - \frac{h_{25} \mu_{21}}{\mu_{01}} + \frac{h_{26} \mu_{21}}{\mu_{01}},\\
h_{03} &= h_{07} - \frac{h_{15} \mu_{11}}{\mu_{01}} + \frac{h_{17} \mu_{11}}{\mu_{01}} + \frac{h_{27} \mu_{22}}{\mu_{01}} - \frac{h_{28} \mu_{22}}{\mu_{01}},\\
h_{04} &= h_{08} - \frac{h_{16} \mu_{11}}{\mu_{01}} + \frac{h_{18} \mu_{11}}{\mu_{01}} - \frac{h_{27} \mu_{21}}{\mu_{01}} + \frac{h_{28} \mu_{21}}{\mu_{01}},\\
h_{05} &= h_{05},\\
h_{06} &= h_{06},\\
h_{07} &= h_{07},\\
h_{08} &= h_{08},\\
h_{11} &= h_{13} - \frac{h_{15} \mu_{02}}{\mu_{01}} + \frac{h_{17} \mu_{02}}{\mu_{01}} + \frac{h_{23} \mu_{22}}{\mu_{11}} - \frac{h_{24} \mu_{22}}{\mu_{11}} + \frac{h_{27} \mu_{02} \mu_{22}}{\mu_{01} \mu_{11}} - \frac{h_{28} \mu_{02} \mu_{22}}{\mu_{01} \mu_{11}},\\
h_{12} &= h_{14} - \frac{h_{16} \mu_{02}}{\mu_{01}} + \frac{h_{18} \mu_{02}}{\mu_{01}} - \frac{h_{23} \mu_{21}}{\mu_{11}} + \frac{h_{24} \mu_{21}}{\mu_{11}} - \frac{h_{27} \mu_{02} \mu_{21}}{\mu_{01} \mu_{11}} + \frac{h_{28} \mu_{02} \mu_{21}}{\mu_{01} \mu_{11}},\\
h_{13} &= h_{13},\\
h_{14} &= h_{14},\\
h_{15} &= h_{15},\\
h_{16} &= h_{16},\\
h_{17} &= h_{17},\\
h_{18} &= h_{18},\\
h_{21} &= h_{22} - \frac{h_{23} \mu_{12}}{\mu_{11}} + \frac{h_{24} \mu_{12}}{\mu_{11}} - \frac{h_{25} \mu_{02}}{\mu_{01}} + \frac{h_{26} \mu_{02}}{\mu_{01}} - \frac{h_{27} \mu_{02} \mu_{12}}{\mu_{01} \mu_{11}} + \frac{h_{28} \mu_{02} \mu_{12}}{\mu_{01} \mu_{11}},\\
h_{22} &= h_{22},\\
h_{23} &= h_{23},\\
h_{24} &= h_{24},\\
h_{25} &= h_{25},\\
h_{26} &= h_{26},\\
h_{27} &= h_{27},\\
h_{28} &= h_{28}.
\end{aligned}
$$

In payoff-function form:

$$
\begin{aligned}
u_0(a_{01}, a_{11}, a_{21}) &= h_{01},&
u_0(a_{01}, a_{11}, a_{22}) &= h_{02},&
u_0(a_{01}, a_{12}, a_{21}) &= h_{03},\\
u_0(a_{01}, a_{12}, a_{22}) &= h_{04},&
u_0(a_{02}, a_{11}, a_{21}) &= h_{05},&
u_0(a_{02}, a_{11}, a_{22}) &= h_{06},\\
u_0(a_{02}, a_{12}, a_{21}) &= h_{07},&
u_0(a_{02}, a_{12}, a_{22}) &= h_{08},&
u_1(a_{01}, a_{11}, a_{21}) &= h_{11},\\
u_1(a_{01}, a_{11}, a_{22}) &= h_{12},&
u_1(a_{01}, a_{12}, a_{21}) &= h_{13},&
u_1(a_{01}, a_{12}, a_{22}) &= h_{14},\\
u_1(a_{02}, a_{11}, a_{21}) &= h_{15},&
u_1(a_{02}, a_{11}, a_{22}) &= h_{16},&
u_1(a_{02}, a_{12}, a_{21}) &= h_{17},\\
u_1(a_{02}, a_{12}, a_{22}) &= h_{18},&
u_2(a_{01}, a_{11}, a_{21}) &= h_{21},&
u_2(a_{01}, a_{11}, a_{22}) &= h_{22},\\
u_2(a_{01}, a_{12}, a_{21}) &= h_{23},&
u_2(a_{01}, a_{12}, a_{22}) &= h_{24},&
u_2(a_{02}, a_{11}, a_{21}) &= h_{25},\\
u_2(a_{02}, a_{11}, a_{22}) &= h_{26},&
u_2(a_{02}, a_{12}, a_{21}) &= h_{27},&
u_2(a_{02}, a_{12}, a_{22}) &= h_{28}.
\end{aligned}
$$

## Skeleton: 3x3

Pure profiles are ordered as:

$$
\begin{aligned}
1 &\leftrightarrow (a_{01}, a_{11}),\\
2 &\leftrightarrow (a_{01}, a_{12}),\\
3 &\leftrightarrow (a_{01}, a_{13}),\\
4 &\leftrightarrow (a_{02}, a_{11}),\\
5 &\leftrightarrow (a_{02}, a_{12}),\\
6 &\leftrightarrow (a_{02}, a_{13}),\\
7 &\leftrightarrow (a_{03}, a_{11}),\\
8 &\leftrightarrow (a_{03}, a_{12}),\\
9 &\leftrightarrow (a_{03}, a_{13}).
\end{aligned}
$$

Measures:

$$
\mu_{01}, \mu_{02}, \mu_{03}, \mu_{11}, \mu_{12}, \mu_{13} > 0.
$$

Free payoff parameters:

$$
h_{07}, h_{08}, h_{09}, h_{13}, h_{14}, h_{15}, h_{16}, h_{17}, h_{18}, h_{19}.
$$

The symbolic solution dictionary is:

$$
\begin{aligned}
h_{01} &= h_{07} + \frac{h_{14} \left(\mu_{02} \mu_{12} + \mu_{02} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} - \frac{h_{15} \mu_{02} \mu_{12}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} - \frac{h_{16} \mu_{02} \mu_{13}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{17} \left(\mu_{01} \mu_{12} + \mu_{01} \mu_{13} + \mu_{03} \mu_{12} + \mu_{03} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{18} \left(- \mu_{01} \mu_{12} - \mu_{03} \mu_{12}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{19} \left(- \mu_{01} \mu_{13} - \mu_{03} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}},\\
h_{02} &= h_{08} - \frac{h_{14} \mu_{02} \mu_{11}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{15} \left(\mu_{02} \mu_{11} + \mu_{02} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} - \frac{h_{16} \mu_{02} \mu_{13}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{17} \left(- \mu_{01} \mu_{11} - \mu_{03} \mu_{11}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{18} \left(\mu_{01} \mu_{11} + \mu_{01} \mu_{13} + \mu_{03} \mu_{11} + \mu_{03} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{19} \left(- \mu_{01} \mu_{13} - \mu_{03} \mu_{13}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}},\\
h_{03} &= h_{09} - \frac{h_{14} \mu_{02} \mu_{11}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} - \frac{h_{15} \mu_{02} \mu_{12}}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{16} \left(\mu_{02} \mu_{11} + \mu_{02} \mu_{12}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{17} \left(- \mu_{01} \mu_{11} - \mu_{03} \mu_{11}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{18} \left(- \mu_{01} \mu_{12} - \mu_{03} \mu_{12}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}} + \frac{h_{19} \left(\mu_{01} \mu_{11} + \mu_{01} \mu_{12} + \mu_{03} \mu_{11} + \mu_{03} \mu_{12}\right)}{\mu_{01}^{2} + \mu_{01} \mu_{02} + \mu_{01} \mu_{03}},\\
h_{04} &= h_{07} + \frac{h_{14} \left(- \mu_{12} - \mu_{13}\right)}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{15} \mu_{12}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{16} \mu_{13}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{17} \left(\mu_{12} + \mu_{13}\right)}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{18} \mu_{12}}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{19} \mu_{13}}{\mu_{01} + \mu_{02} + \mu_{03}},\\
h_{05} &= h_{08} + \frac{h_{14} \mu_{11}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{15} \left(- \mu_{11} - \mu_{13}\right)}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{16} \mu_{13}}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{17} \mu_{11}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{18} \left(\mu_{11} + \mu_{13}\right)}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{19} \mu_{13}}{\mu_{01} + \mu_{02} + \mu_{03}},\\
h_{06} &= h_{09} + \frac{h_{14} \mu_{11}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{15} \mu_{12}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{16} \left(- \mu_{11} - \mu_{12}\right)}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{17} \mu_{11}}{\mu_{01} + \mu_{02} + \mu_{03}} - \frac{h_{18} \mu_{12}}{\mu_{01} + \mu_{02} + \mu_{03}} + \frac{h_{19} \left(\mu_{11} + \mu_{12}\right)}{\mu_{01} + \mu_{02} + \mu_{03}},\\
h_{07} &= h_{07},\\
h_{08} &= h_{08},\\
h_{09} &= h_{09},\\
h_{11} &= h_{13} - \frac{h_{14} \mu_{02}}{\mu_{01}} + \frac{h_{16} \mu_{02}}{\mu_{01}} - \frac{h_{17} \mu_{03}}{\mu_{01}} + \frac{h_{19} \mu_{03}}{\mu_{01}},\\
h_{12} &= h_{13} - \frac{h_{15} \mu_{02}}{\mu_{01}} + \frac{h_{16} \mu_{02}}{\mu_{01}} - \frac{h_{18} \mu_{03}}{\mu_{01}} + \frac{h_{19} \mu_{03}}{\mu_{01}},\\
h_{13} &= h_{13},\\
h_{14} &= h_{14},\\
h_{15} &= h_{15},\\
h_{16} &= h_{16},\\
h_{17} &= h_{17},\\
h_{18} &= h_{18},\\
h_{19} &= h_{19}.
\end{aligned}
$$

In payoff-function form:

$$
\begin{aligned}
u_0(a_{01}, a_{11}) &= h_{01},&
u_0(a_{01}, a_{12}) &= h_{02},&
u_0(a_{01}, a_{13}) &= h_{03},\\
u_0(a_{02}, a_{11}) &= h_{04},&
u_0(a_{02}, a_{12}) &= h_{05},&
u_0(a_{02}, a_{13}) &= h_{06},\\
u_0(a_{03}, a_{11}) &= h_{07},&
u_0(a_{03}, a_{12}) &= h_{08},&
u_0(a_{03}, a_{13}) &= h_{09},\\
u_1(a_{01}, a_{11}) &= h_{11},&
u_1(a_{01}, a_{12}) &= h_{12},&
u_1(a_{01}, a_{13}) &= h_{13},\\
u_1(a_{02}, a_{11}) &= h_{14},&
u_1(a_{02}, a_{12}) &= h_{15},&
u_1(a_{02}, a_{13}) &= h_{16},\\
u_1(a_{03}, a_{11}) &= h_{17},&
u_1(a_{03}, a_{12}) &= h_{18},&
u_1(a_{03}, a_{13}) &= h_{19}.
\end{aligned}
$$
