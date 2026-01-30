# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
import platform
import sys
import random
from datetime import datetime, timezone

import pandas as pd

from src.metrics.structural_metrics import compute_structural_metrics
from src.metrics.structural_distributions import (
    compute_scalar_metric_distributions,
    compute_decade_patterns,
)


DEFAULT_OUTPUT_DIR = Path("reports/monte_carlo/baseline")

BASELINE_PERCENTILES = list(range(5, 100, 5))

@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the Monte Carlo baseline (uniform) simulation."""
    N: int
    seed: int
    output_dir: Path = DEFAULT_OUTPUT_DIR
    export: bool = True
    return_combinations: bool = False


# ------------------------------------------------------------
# 1) UNIFORM GENERATOR
# ------------------------------------------------------------

def generate_uniform_combinations(N: int, seed: int, lo: int = 1, hi: int = 54, k: int = 5) -> List[List[int]]:
    """
    Generate N uniform combinations: choose k distinct numbers from [lo, hi].
    Each k-combination is equally likely (uniform over combinations).
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")
    if (hi - lo + 1) < k:
        raise ValueError("Range [lo, hi] too small for k distinct numbers")

    rng = random.Random(seed)
    population = list(range(lo, hi + 1))

    combos: List[List[int]] = []
    for _ in range(N):
        c = rng.sample(population, k)
        c.sort()  # canonical order (order irrelevant)
        combos.append(c)

    return combos


# ------------------------------------------------------------
# 2) METRICS FOR SIMULATION
# ------------------------------------------------------------

def compute_simulation_metrics_dataframe(combinations: List[List[int]]) -> pd.DataFrame:
    """
    Compute structural metrics for each simulated combination.
    """
    records: List[Dict[str, Any]] = []
    for combo in combinations:
        records.append(compute_structural_metrics(combo))
    return pd.DataFrame(records)


# ------------------------------------------------------------
# 3) AGGREGATION (DISTRIBUTIONS)
# ------------------------------------------------------------

def aggregate_simulation_distributions(
    metrics_df: pd.DataFrame,
    percentiles: List[int],
) -> Dict[str, pd.DataFrame]:
    """
    Build distribution tables (percentiles and decade patterns) from metrics_df.
    Output format matches the historical distributions module.
    """
    scalar = compute_scalar_metric_distributions(
        metrics_df,
        percentiles=percentiles,
    )
    decade = compute_decade_patterns(metrics_df)

    return {
        "scalar_distributions": scalar,
        "decade_patterns": decade,
    }

# ------------------------------------------------------------
# 4) EXPORT (CSV + METADATA)
# ------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def export_baseline_artifacts(
    distributions: Dict[str, pd.DataFrame],
    config: BaselineConfig,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
) -> None:
    """
    Export baseline artifacts:
    - metadata.json
    - range_percentiles.csv / sum_percentiles.csv / std_percentiles.csv
    - decade_patterns.csv
    """
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Percentiles CSVs (same names as historical)
    scalar: Dict[str, pd.DataFrame] = distributions["scalar_distributions"]  # type: ignore[assignment]
    for metric in ("range", "sum", "std"):
        df = scalar[metric].copy()
        # enforce column order and types
        df = df[["percentile", "value"]]
        df.to_csv(out / f"{metric}_percentiles.csv", index=False)

    # Decade patterns CSV
    decade_df = distributions["decade_patterns"].copy()
    decade_df = decade_df[["pattern", "frequency", "relative_frequency"]]
    decade_df.to_csv(out / "decade_patterns.csv", index=False)

    # Metadata JSON
    metadata = {
        "module": "monte_carlo_baseline",
        "type": "uniform",
        "game": "gordo_primitiva",
        "simulation": {
            "N": int(config.N),
            "seed": int(config.seed),
        },
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": float(duration_seconds),
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "platform_detail": platform.platform(),
            "pandas_version": getattr(pd, "__version__", "unknown"),
        },
        "metrics_used": [
            "range",
            "sum",
            "std",
            "decade_distribution",
        ],
        "notes": "Monte Carlo baseline uniforme. Sin historico, sin condicionamiento.",
    }

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ------------------------------------------------------------

def run_monte_carlo_baseline(
    N: int,
    seed: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export: bool = True,
    return_combinations: bool = False,
) -> Dict[str, Any]:
    """
    Run Monte Carlo baseline (uniform) simulation and export artifacts.

    Returns
    -------
    dict with keys:
      - combinations (optional)
      - metrics_df
      - scalar_distributions (dict of DataFrames)
      - decade_patterns (DataFrame)
      - output_dir
    """
    config = BaselineConfig(N=N, seed=seed, output_dir=Path(output_dir), export=export, return_combinations=return_combinations)

    started_at = _iso_utc_now()
    t0 = time.perf_counter()

    combinations = generate_uniform_combinations(config.N, config.seed)
    metrics_df = compute_simulation_metrics_dataframe(combinations)
    distributions = aggregate_simulation_distributions(
         metrics_df,
         percentiles=BASELINE_PERCENTILES,
    )

    t1 = time.perf_counter()
    finished_at = _iso_utc_now()
    duration_seconds = t1 - t0

    if config.export:
        export_baseline_artifacts(
            distributions=distributions,
            config=config,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
        )

    result: Dict[str, Any] = {
        "metrics_df": metrics_df,
        "scalar_distributions": distributions["scalar_distributions"],
        "decade_patterns": distributions["decade_patterns"],
        "output_dir": str(config.output_dir),
        "metadata": {
            "N": config.N,
            "seed": config.seed,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        }
    }

    if config.return_combinations:
        result["combinations"] = combinations

    return result

# ------------------------------------------------------------
# SCRIPT ENTRYPOINT
# ------------------------------------------------------------

def main() -> None:
    result = run_monte_carlo_baseline(
        N=2_000_000,              # ajusta si quieres
        seed=20260126,
        output_dir=Path("reports/monte_carlo/baseline"),
    )

    print("✅ Baseline Monte Carlo generado")
    print(f"→ output_dir: {result['output_dir']}")
    print(f"→ N: {result['metadata']['N']}")
    print(f"→ duration_seconds: {result['metadata']['duration_seconds']:.2f}")


if __name__ == "__main__":
    main()