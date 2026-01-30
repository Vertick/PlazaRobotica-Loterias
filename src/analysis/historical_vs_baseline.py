# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("reports/analysis/historical_vs_baseline")


# ------------------------------------------------------------
# CARGA DE DISTRIBUCIONES
# ------------------------------------------------------------

def load_percentiles(path: Path) -> pd.DataFrame:
    """
    Load a percentiles CSV with columns: percentile, value
    """
    df = pd.read_csv(path)
    return df.sort_values("percentile").reset_index(drop=True)


def load_decade_patterns(path: Path) -> pd.DataFrame:
    """
    Load decade patterns CSV with columns:
    pattern, frequency, relative_frequency
    """
    df = pd.read_csv(path)
    return df.sort_values("relative_frequency", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# COMPARACION DE METRICAS ESCALARES
# ------------------------------------------------------------

def compare_scalar_percentiles(
    historical: pd.DataFrame,
    baseline: pd.DataFrame,
    metric_name: str
) -> pd.DataFrame:
    """
    Compare historical vs baseline percentiles for a scalar metric.
    """
    df = historical.merge(
        baseline,
        on="percentile",
        suffixes=("_historical", "_baseline")
    )

    df["delta_abs"] = df["value_historical"] - df["value_baseline"]
    df["delta_pct"] = df["delta_abs"] / df["value_baseline"] * 100.0
    df["metric"] = metric_name

    return df[
        [
            "metric",
            "percentile",
            "value_historical",
            "value_baseline",
            "delta_abs",
            "delta_pct",
        ]
    ]


# ------------------------------------------------------------
# COMPARACION DE PATRONES DE DECENAS
# ------------------------------------------------------------

def compare_decade_patterns(
    historical: pd.DataFrame,
    baseline: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare top-N decade patterns between historical and baseline.
    """
    hist_top = historical.head(top_n).copy()
    base_top = baseline.head(top_n).copy()

    df = hist_top.merge(
        base_top,
        on="pattern",
        how="outer",
        suffixes=("_historical", "_baseline")
    ).fillna(0.0)

    df["ratio"] = df["relative_frequency_historical"] / df["relative_frequency_baseline"].replace(0, pd.NA)

    return df[
        [
            "pattern",
            "frequency_historical",
            "relative_frequency_historical",
            "frequency_baseline",
            "relative_frequency_baseline",
            "ratio",
        ]
    ].sort_values("relative_frequency_historical", ascending=False)


# ------------------------------------------------------------
# EXPORTACION
# ------------------------------------------------------------

def export_comparison(
    scalar_df: pd.DataFrame,
    decade_df: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> None:
    """
    Export comparison results to CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_df.to_csv(output_dir / "scalar_comparison.csv", index=False)
    decade_df.to_csv(output_dir / "decade_pattern_comparison.csv", index=False)

    summary_path = output_dir / "summary.md"
    if not summary_path.exists():
        summary_path.write_text(
					"# Historico vs Monte Carlo baseline\n\n"
					"Este documento debe completarse manualmente.\n\n"
					"Interpretar:\n"
					"- diferencias de percentiles\n"
					"- estabilidad de patrones de decenas\n"
					"- compatibilidad global con el azar uniforme\n",
					encoding="utf-8"
				)


# ------------------------------------------------------------
# PIPELINE PRINCIPAL
# ------------------------------------------------------------

def run_historical_vs_baseline(
    historical_dir: Path,
    baseline_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Run full comparison between historical and Monte Carlo baseline distributions.
    """
    scalar_results = []

    for metric in ("range", "sum", "std"):
        hist = load_percentiles(historical_dir / f"{metric}_percentiles.csv")
        base = load_percentiles(baseline_dir / f"{metric}_percentiles.csv")

        scalar_results.append(
            compare_scalar_percentiles(hist, base, metric)
        )

    scalar_df = pd.concat(scalar_results, ignore_index=True)

    hist_decades = load_decade_patterns(historical_dir / "decade_patterns.csv")
    base_decades = load_decade_patterns(baseline_dir / "decade_patterns.csv")

    decade_df = compare_decade_patterns(hist_decades, base_decades)

    export_comparison(scalar_df, decade_df, output_dir)

    return {
        "scalar_comparison": scalar_df,
        "decade_pattern_comparison": decade_df,
    }