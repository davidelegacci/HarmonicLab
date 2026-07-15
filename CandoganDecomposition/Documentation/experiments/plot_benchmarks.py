"""Create publication figures and a compact LaTeX table from benchmark data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "benchmark_results.csv"

COLORS = {
    "requested": "#1b4965",
    "two_player_actions": "#c75b39",
    "binary_players": "#277a6b",
    "ternary_players": "#d49b2a",
    "stress": "#4b5563",
}
LABELS = {
    "requested": "requested small cases",
    "two_player_actions": "two players, growing actions",
    "binary_players": "growing players, two actions",
    "ternary_players": "growing players, three actions",
    "stress": "balanced stress cases",
}


def load_rows() -> list[dict[str, object]]:
    numeric = {
        "n_players": int,
        "profiles": int,
        "edges": int,
        "payoff_entries": int,
        "setup_median_s": float,
        "decompose_median_s": float,
        "stored_numpy_mib": float,
        "reconstruction_max_abs": float,
        "harmonic_codifferential_max_abs": float,
    }
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    for row in rows:
        for key, conversion in numeric.items():
            row[key] = conversion(row[key])
    return rows


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def runtime_by_profiles(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    for family in COLORS:
        subset = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: row["profiles"],
        )
        if not subset:
            continue
        profiles = [row["profiles"] for row in subset]
        axes[0].plot(
            profiles,
            [row["setup_median_s"] for row in subset],
            marker="o",
            linewidth=1.3,
            color=COLORS[family],
            label=LABELS[family],
        )
        axes[1].plot(
            profiles,
            [row["decompose_median_s"] * 1e3 for row in subset],
            marker="o",
            linewidth=1.3,
            color=COLORS[family],
            label=LABELS[family],
        )
    axes[0].set(xscale="log", yscale="log", xlabel="profiles $M$", ylabel="seconds")
    axes[0].set_title("Geometry construction")
    axes[1].set(xscale="log", yscale="log", xlabel="profiles $M$", ylabel="milliseconds")
    axes[1].set_title("One decomposition with cached geometry")
    axes[0].legend(frameon=False, loc="best")
    save(fig, "runtime_by_profiles")


def isolated_scaling(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    action_rows = sorted(
        (row for row in rows if row["family"] == "two_player_actions"),
        key=lambda row: row["profiles"],
    )
    binary_rows = sorted(
        (row for row in rows if row["family"] == "binary_players"),
        key=lambda row: row["n_players"],
    )
    q_values = [round(row["profiles"] ** 0.5) for row in action_rows]
    axes[0].plot(
        q_values,
        [row["setup_median_s"] for row in action_rows],
        color=COLORS["two_player_actions"],
        marker="o",
        label="geometry",
    )
    axes[0].plot(
        q_values,
        [row["decompose_median_s"] for row in action_rows],
        color=COLORS["two_player_actions"],
        marker="s",
        linestyle="--",
        label="cached decomposition",
    )
    axes[0].set(yscale="log", xlabel="actions per player $q$", ylabel="seconds")
    axes[0].set_title("Two-player games $q\\times q$")
    axes[0].legend(frameon=False)

    players = [row["n_players"] for row in binary_rows]
    axes[1].plot(
        players,
        [row["setup_median_s"] for row in binary_rows],
        color=COLORS["binary_players"],
        marker="o",
        label="geometry",
    )
    axes[1].plot(
        players,
        [row["decompose_median_s"] for row in binary_rows],
        color=COLORS["binary_players"],
        marker="s",
        linestyle="--",
        label="cached decomposition",
    )
    axes[1].set(yscale="log", xlabel="players $n$", ylabel="seconds")
    axes[1].set_title("Binary-action games $2^n$")
    axes[1].legend(frameon=False)
    save(fig, "isolated_scaling")


def storage_plot(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    for family in COLORS:
        subset = [row for row in rows if row["family"] == family]
        if not subset:
            continue
        ax.scatter(
            [row["profiles"] for row in subset],
            [row["stored_numpy_mib"] for row in subset],
            color=COLORS[family],
            s=30,
            label=LABELS[family],
        )
    ax.set(xscale="log", yscale="log", xlabel="profiles $M$", ylabel="stored NumPy arrays (MiB)")
    ax.set_title("Measured persistent numerical storage")
    ax.legend(frameon=False, loc="best")
    save(fig, "storage_by_profiles")


def latex_table(rows: list[dict[str, object]]) -> None:
    wanted = {
        "2x2",
        "2x3",
        "2x2x2",
        "4x4",
        "10x10",
        "20x20",
        "32x32",
        "2x2x2x2x2x2",
        "2x2x2x2x2x2x2x2x2x2x2",
        "3x3x3",
        "3x3x3x3x3x3x3",
        "4x4x4x4",
        "5x5x5x5",
        "6x6x6x6",
    }
    selected = [row for row in rows if row["label"] in wanted]
    selected.sort(key=lambda row: (row["profiles"], row["edges"], row["label"]))
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Skeleton & $M$ & $E$ & Setup (ms) & Cached (ms) & Storage (MiB) \\",
        r"\midrule",
    ]
    for row in selected:
        actions = [int(value) for value in str(row["label"]).split("x")]
        if len(actions) >= 3 and len(set(actions)) == 1:
            label = rf"${actions[0]}^{{{len(actions)}}}$"
        else:
            label = str(row["label"]).replace("x", r"$\times$")
        lines.append(
            f"{label} & {row['profiles']} & {row['edges']} & "
            f"{row['setup_median_s'] * 1e3:.3f} & "
            f"{row['decompose_median_s'] * 1e3:.3f} & "
            f"{row['stored_numpy_mib']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (HERE / "benchmark_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure()
    rows = load_rows()
    runtime_by_profiles(rows)
    isolated_scaling(rows)
    storage_plot(rows)
    latex_table(rows)


if __name__ == "__main__":
    main()
