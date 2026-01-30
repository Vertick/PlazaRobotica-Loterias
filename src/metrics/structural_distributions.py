# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


DEFAULT_REPORTS_DIR = Path("reports/metrics")


# ------------------------------------------------------------
# PERCENTILES PARA METRICAS ESCALARES
# ------------------------------------------------------------

def compute_percentiles(
    series: pd.Series,
    percentiles: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute empirical percentiles for a numeric series.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    values = np.percentile(series.to_numpy(), percentiles)

    return pd.DataFrame({
        "percentile": percentiles,
        "value": values,
    })


def compute_scalar_metric_distributions(
    metrics_df: pd.DataFrame,
    percentiles: list[int] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compute percentile tables for scalar structural metrics.
    """
    distributions: Dict[str, pd.DataFrame] = {}

    for col in ("range", "sum", "std"):
        distributions[col] = compute_percentiles(
            metrics_df[col],
            percentiles=percentiles,
        )

    return distributions


# ------------------------------------------------------------
# DISTRIBUCION DE PATRONES DE DECENAS
# ------------------------------------------------------------

def compute_decade_patterns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frequency of decade distribution patterns.
    """
    decade_cols = ["d1_09", "d10_19", "d20_29", "d30_39", "d40_49", "d50_54"]

    patterns = (
        metrics_df[decade_cols]
        .astype(str)
        .agg(",".join, axis=1)
        .value_counts()
        .reset_index()
    )

    patterns.columns = ["pattern", "frequency"]
    patterns["relative_frequency"] = (
        patterns["frequency"] / patterns["frequency"].sum()
    )

    return patterns