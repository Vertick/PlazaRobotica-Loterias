# -*- coding: utf-8 -*-

from typing import List, Dict
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# UTILIDADES BÁSICAS
# ------------------------------------------------------------

def _validate_combination(numbers: List[int]) -> None:
    if len(numbers) != 5:
        raise ValueError("A Gordo combination must contain exactly 5 numbers")
    if len(set(numbers)) != 5:
        raise ValueError("Combination contains repeated numbers")


# ------------------------------------------------------------
# MÉTRICAS SOBRE UNA COMBINACIÓN
# ------------------------------------------------------------

def range_metrics(numbers: List[int]) -> Dict[str, float]:
    _validate_combination(numbers)
    return {
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
    }


def sum_metrics(numbers: List[int]) -> Dict[str, float]:
    _validate_combination(numbers)
    return {
        "sum": float(sum(numbers)),
        "mean": float(np.mean(numbers)),
    }


def dispersion_metrics(numbers: List[int]) -> Dict[str, float]:
    _validate_combination(numbers)
    return {
        "std": float(np.std(numbers, ddof=0)),
    }


def decade_distribution(numbers: List[int]) -> Dict[str, int]:
    _validate_combination(numbers)

    bins = {
        "d1_09": 0,
        "d10_19": 0,
        "d20_29": 0,
        "d30_39": 0,
        "d40_49": 0,
        "d50_54": 0,
    }

    for n in numbers:
        if 1 <= n <= 9:
            bins["d1_09"] += 1
        elif 10 <= n <= 19:
            bins["d10_19"] += 1
        elif 20 <= n <= 29:
            bins["d20_29"] += 1
        elif 30 <= n <= 39:
            bins["d30_39"] += 1
        elif 40 <= n <= 49:
            bins["d40_49"] += 1
        elif 50 <= n <= 54:
            bins["d50_54"] += 1

    return bins


def compute_structural_metrics(numbers: List[int]) -> Dict[str, float]:
    """
    Compute all structural metrics for a single Gordo combination.
    """
    metrics = {}
    metrics.update(range_metrics(numbers))
    metrics.update(sum_metrics(numbers))
    metrics.update(dispersion_metrics(numbers))
    metrics.update(decade_distribution(numbers))
    return metrics


# ------------------------------------------------------------
# MÉTRICAS SOBRE DATAFRAME DE SORTEOS
# ------------------------------------------------------------

def compute_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structural metrics for each draw in a curated dataframe.
    """
    records = []

    for _, row in df.iterrows():
        numbers = [row[f"n{i}"] for i in range(1, 6)]
        metrics = compute_structural_metrics(numbers)
        records.append(metrics)

    return pd.DataFrame(records)