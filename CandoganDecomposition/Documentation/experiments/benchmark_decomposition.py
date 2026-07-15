"""Reproducible timing study for the maintained dense decomposition core."""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from math import comb, prod
from pathlib import Path

# Fix BLAS thread counts before importing NumPy. This reduces timing variance.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CandoganDecomposition import GameGeometry


OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "benchmark_results.csv"
METADATA_PATH = OUTPUT_DIR / "benchmark_metadata.json"
SEED = 20260715
TARGET_BATCH_SECONDS = 0.15
MAX_BATCH_ITERATIONS = 16384
MAX_ESTIMATED_STORED_MIB = 768.0


# The first three cases are the examples requested explicitly. The remaining
# families separate growth in actions from growth in the number of players.
CASES = [
    ("requested", (2, 2)),
    ("requested", (2, 3)),
    ("requested", (2, 2, 2)),
    ("two_player_actions", (3, 3)),
    ("two_player_actions", (4, 4)),
    ("two_player_actions", (6, 6)),
    ("two_player_actions", (8, 8)),
    ("two_player_actions", (10, 10)),
    ("two_player_actions", (12, 12)),
    ("two_player_actions", (16, 16)),
    ("two_player_actions", (20, 20)),
    ("two_player_actions", (24, 24)),
    ("two_player_actions", (28, 28)),
    ("two_player_actions", (32, 32)),
    ("two_player_actions", (36, 36)),
    ("binary_players", (2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ("binary_players", (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)),
    ("ternary_players", (3, 3, 3)),
    ("ternary_players", (3, 3, 3, 3)),
    ("ternary_players", (3, 3, 3, 3, 3)),
    ("ternary_players", (3, 3, 3, 3, 3, 3)),
    ("ternary_players", (3, 3, 3, 3, 3, 3, 3)),
    ("ternary_players", (3, 3, 3, 3, 3, 3, 3, 3)),
    ("stress", (4, 4, 4, 4)),
    ("stress", (5, 5, 5, 5)),
    ("stress", (6, 6, 6, 6)),
    ("stress", (7, 7, 7, 7)),
]


def edge_count(actions: tuple[int, ...]) -> int:
    profiles = prod(actions)
    return sum((profiles // size) * comb(size, 2) for size in actions)


def estimated_stored_mib(actions: tuple[int, ...]) -> float:
    """Leading dense storage: incidence, gradient, and Laplacian pseudoinverse."""

    profiles = prod(actions)
    edges = edge_count(actions)
    return (16 * edges * profiles + 8 * profiles * profiles) / 2**20


def geometry_numpy_bytes(geometry: GameGeometry) -> int:
    arrays = [
        geometry.profile_weights,
        geometry.edge_players,
        geometry.edge_tails,
        geometry.edge_heads,
        geometry.incidence_matrix,
        geometry.gradient_scales,
        geometry.flow_weights,
        geometry.gradient_matrix,
        geometry._weighted_laplacian_pinv,
        geometry.action_masses,
        *geometry.mu,
    ]
    return sum(array.nbytes for array in arrays)


def choose_setup_samples(first_seconds: float) -> int:
    if first_seconds < 0.5:
        return 7
    if first_seconds < 2.0:
        return 5
    if first_seconds < 8.0:
        return 3
    return 1


def time_geometry(
    actions: tuple[int, ...], mu: list[np.ndarray]
) -> tuple[GameGeometry, list[float]]:
    samples = []
    start = time.perf_counter_ns()
    geometry = GameGeometry(actions, mu=mu)
    samples.append((time.perf_counter_ns() - start) * 1e-9)

    for _ in range(choose_setup_samples(samples[0]) - 1):
        gc.collect()
        start = time.perf_counter_ns()
        candidate = GameGeometry(actions, mu=mu)
        samples.append((time.perf_counter_ns() - start) * 1e-9)
        geometry = candidate
    return geometry, samples


def time_cached_decomposition(
    geometry: GameGeometry, payoffs: np.ndarray
) -> tuple[list[float], int]:
    geometry.decompose(payoffs)
    iterations = 1
    while iterations < MAX_BATCH_ITERATIONS:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            geometry.decompose(payoffs)
        elapsed = (time.perf_counter_ns() - start) * 1e-9
        if elapsed >= TARGET_BATCH_SECONDS:
            break
        iterations *= 2

    samples = []
    for _ in range(9):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            geometry.decompose(payoffs)
        samples.append((time.perf_counter_ns() - start) * 1e-9 / iterations)
    return samples, iterations


def benchmark_case(index: int, family: str, actions: tuple[int, ...]) -> dict[str, object]:
    profiles = prod(actions)
    edges = edge_count(actions)
    label = "x".join(str(size) for size in actions)
    estimate = estimated_stored_mib(actions)
    row: dict[str, object] = {
        "case_index": index,
        "family": family,
        "label": label,
        "n_players": len(actions),
        "actions": label,
        "profiles": profiles,
        "edges": edges,
        "payoff_entries": len(actions) * profiles,
        "estimated_leading_storage_mib": estimate,
    }
    if estimate > MAX_ESTIMATED_STORED_MIB:
        row["status"] = "skipped_storage_limit"
        return row

    rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    mu = [rng.uniform(0.5, 2.0, size=size) for size in actions]
    payoffs = rng.standard_normal((len(actions), *actions))

    geometry, setup_samples = time_geometry(actions, mu)
    decompose_samples, batch_iterations = time_cached_decomposition(
        geometry, payoffs
    )
    result = geometry.decompose(payoffs)

    row.update(
        {
            "status": "ok",
            "setup_samples": len(setup_samples),
            "setup_median_s": statistics.median(setup_samples),
            "setup_min_s": min(setup_samples),
            "setup_max_s": max(setup_samples),
            "decompose_samples": len(decompose_samples),
            "decompose_batch_iterations": batch_iterations,
            "decompose_median_s": statistics.median(decompose_samples),
            "decompose_min_s": min(decompose_samples),
            "decompose_max_s": max(decompose_samples),
            "stored_numpy_mib": geometry_numpy_bytes(geometry) / 2**20,
            "reconstruction_max_abs": float(
                np.max(
                    np.abs(
                        payoffs
                        - result.nonstrategic
                        - result.potential
                        - result.harmonic
                    )
                )
            ),
            "harmonic_codifferential_max_abs": float(
                np.max(np.abs(geometry.codifferential(result.harmonic_flow)))
            ),
            "potential_harmonic_inner_product_abs": abs(
                geometry.game_inner_product(result.potential, result.harmonic)
            ),
        }
    )
    return row


def main() -> None:
    rows = []
    for index, (family, actions) in enumerate(CASES):
        print(f"[{index + 1:02d}/{len(CASES):02d}] {family}: {actions}", flush=True)
        row = benchmark_case(index, family, actions)
        rows.append(row)
        if row["status"] == "ok":
            print(
                "  M={profiles}, E={edges}, setup={setup_median_s:.6f}s, "
                "cached={decompose_median_s:.6f}s".format(**row),
                flush=True,
            )
        else:
            print(f"  {row['status']}", flush=True)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine_reported_to_python": platform.machine(),
        "numpy": np.__version__,
        "blas_thread_environment": {
            variable: os.environ.get(variable)
            for variable in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "target_batch_seconds": TARGET_BATCH_SECONDS,
        "maximum_estimated_stored_mib": MAX_ESTIMATED_STORED_MIB,
        "timing_clock": "time.perf_counter_ns",
        "setup_statistic": "median of adaptive independent constructions",
        "decomposition_statistic": "median of nine batches on cached geometry",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
