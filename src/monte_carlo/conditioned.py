# src/monte_carlo/conditioned.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
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

# Nota: este módulo NO usa el histórico directamente.
# Para resolver percentiles (percentile_band) usa un directorio de referencia
# (por defecto, el baseline Monte Carlo del Módulo 4).


DEFAULT_REFERENCE_DIR = Path("reports/monte_carlo/baseline")
DEFAULT_OUTPUT_DIR = Path("reports/monte_carlo/conditioned")

ALLOWED_METRICS = {"range", "sum", "std", "decade_pattern", "decades"}
ALLOWED_RULES = {
    "interval",
    "percentile_band",
    "include_set",
    "exclude_set",
    "distribution_constraint",
}
SCALAR_METRICS = {"range", "sum", "std"}
DISCRETE_METRICS = {"decade_pattern"}


# ---------------------------------------------------------------------
# CONFIG SCHEMA
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """
    Single structural condition, declarative and auditable.

    Fields
    ------
    id: unique identifier
    metric: one of {range, sum, std, decade_pattern}
    rule: one of {interval, percentile_band, include_set, exclude_set}
    params: rule-specific parameters (dict)
    """
    id: str
    metric: str
    rule: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ConditionedConfig:
    """
    Config for conditioned Monte Carlo generation.

    - N_target: number of ACCEPTED combinations desired.
    - max_iterations: hard cap on generated trials to avoid infinite loops.
    - reference_dir: where percentiles/patterns are loaded from (default baseline).
    - output_dir: where artifacts are exported.
    """
    name: str
    description: str
    seed: int
    N_target: int
    max_iterations: int = 500_000
    conditions: List[Condition] = None  # type: ignore[assignment]
    reference_dir: Path = DEFAULT_REFERENCE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    export: bool = True
    return_combinations: bool = False


# ---------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------

def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# ---------------------------------------------------------------------
# LOAD REFERENCE ARTIFACTS (percentiles/patterns)
# ---------------------------------------------------------------------

def _load_percentiles_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require({"percentile", "value"}.issubset(df.columns), f"Percentiles CSV inválido: {path}")
    return df.sort_values("percentile").reset_index(drop=True)


def _load_reference_scalar_percentiles(reference_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Expects files named:
      range_percentiles.csv / sum_percentiles.csv / std_percentiles.csv
    """
    out: Dict[str, pd.DataFrame] = {}
    for m in ("range", "sum", "std"):
        p = reference_dir / f"{m}_percentiles.csv"
        _require(p.exists(), f"No existe percentiles de referencia para '{m}': {p}")
        out[m] = _load_percentiles_csv(p)
    return out


def _load_reference_decade_patterns(reference_dir: Path) -> pd.DataFrame:
    """
    Expects: decade_patterns.csv with columns:
      pattern, frequency, relative_frequency
    """
    p = reference_dir / "decade_patterns.csv"
    _require(p.exists(), f"No existe decade_patterns de referencia: {p}")
    df = pd.read_csv(p)
    _require({"pattern", "frequency", "relative_frequency"}.issubset(df.columns), f"decade_patterns.csv inválido: {p}")
    return df.sort_values("relative_frequency", ascending=False).reset_index(drop=True)


def _value_at_percentile(df: pd.DataFrame, p: int) -> float:
    """
    Resolve a percentile value from a reference percentiles df.

    For now we require exact match of p in df["percentile"] to keep the contract explicit.
    """
    _require(isinstance(p, int), "Percentil debe ser int (p.ej. 10, 25, 50, 75, 90).")
    row = df.loc[df["percentile"] == p]
    _require(len(row) == 1, f"Percentil {p} no está disponible en referencia. Disponibles: {sorted(df['percentile'].unique())}")
    return float(row["value"].iloc[0])


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

def validate_condition(cond: Condition) -> None:
    _require(cond.id and isinstance(cond.id, str), "Condition.id debe ser string no vacío.")
    _require(cond.metric in ALLOWED_METRICS, f"Condition.metric inválida: {cond.metric}")
    _require(cond.rule in ALLOWED_RULES, f"Condition.rule inválida: {cond.rule}")
    _require(isinstance(cond.params, dict), "Condition.params debe ser dict.")

    if cond.rule in {"interval", "percentile_band"}:
        _require(cond.metric in SCALAR_METRICS, f"Regla '{cond.rule}' solo aplica a métricas escalares (range/sum/std).")
    if cond.rule in {"include_set", "exclude_set"}:
        _require(cond.metric in DISCRETE_METRICS, f"Regla '{cond.rule}' solo aplica a métricas discretas (decade_pattern).")

    if cond.rule == "interval":
        _require("min" in cond.params and "max" in cond.params, "interval requiere params: {min, max}")
        _require(isinstance(cond.params["min"], (int, float)), "interval.min debe ser numérico")
        _require(isinstance(cond.params["max"], (int, float)), "interval.max debe ser numérico")
        _require(cond.params["min"] <= cond.params["max"], "interval requiere min <= max")

    if cond.rule == "percentile_band":
        _require("lower" in cond.params and "upper" in cond.params, "percentile_band requiere params: {lower, upper}")
        lo = cond.params["lower"]
        hi = cond.params["upper"]
        _require(isinstance(lo, int) and isinstance(hi, int), "percentile_band.lower/upper deben ser int")
        _require(0 <= lo <= 100 and 0 <= hi <= 100, "percentile_band percentiles deben estar en [0,100]")
        _require(lo <= hi, "percentile_band requiere lower <= upper")

    if cond.rule == "include_set":
        _require("allowed" in cond.params and isinstance(cond.params["allowed"], list), "include_set requiere params: {allowed:[...]}")
        _require(len(cond.params["allowed"]) > 0, "include_set.allowed no puede estar vacío")
        _require(all(isinstance(x, str) for x in cond.params["allowed"]), "include_set.allowed debe ser lista de strings")

    if cond.rule == "exclude_set":
        _require("forbidden" in cond.params and isinstance(cond.params["forbidden"], list), "exclude_set requiere params: {forbidden:[...]}")
        _require(len(cond.params["forbidden"]) > 0, "exclude_set.forbidden no puede estar vacío")
        _require(all(isinstance(x, str) for x in cond.params["forbidden"]), "exclude_set.forbidden debe ser lista de strings")
    if cond.rule == "distribution_constraint":
        _require(
            cond.metric == "decades",
            "distribution_constraint solo aplica a metric='decades'",
        )
        _require(
            any(k in cond.params for k in ("max_bin_count", "min_nonempty_bins")),
            "distribution_constraint requiere al menos uno de: max_bin_count, min_nonempty_bins",
        )


def validate_config(cfg: ConditionedConfig) -> None:
    _require(isinstance(cfg.name, str) and cfg.name.strip(), "Config.name debe ser string no vacío.")
    _require(isinstance(cfg.description, str), "Config.description debe ser string.")
    _require(isinstance(cfg.seed, int), "Config.seed debe ser int.")
    _require(isinstance(cfg.N_target, int) and cfg.N_target > 0, "Config.N_target debe ser int > 0.")
    _require(isinstance(cfg.max_iterations, int) and cfg.max_iterations > 0, "Config.max_iterations debe ser int > 0.")
    _require(cfg.conditions is not None and isinstance(cfg.conditions, list) and len(cfg.conditions) > 0, "Config.conditions debe ser una lista no vacía.")

    # unique ids
    ids = [c.id for c in cfg.conditions]
    _require(len(ids) == len(set(ids)), "Condition.id debe ser único (duplicados detectados).")

    for c in cfg.conditions:
        validate_condition(c)


# ---------------------------------------------------------------------
# COMPILATION OF CONDITIONS (resolve percentile bands -> numeric intervals)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledCondition:
    id: str
    metric: str
    rule: str
    params: Dict[str, Any]
    predicate: Callable[[Dict[str, Any]], bool]  # takes metrics dict


def _compile_conditions(cfg: ConditionedConfig, reference_scalar: Dict[str, pd.DataFrame]) -> List[CompiledCondition]:
    compiled: List[CompiledCondition] = []

    for c in cfg.conditions:
        if c.rule == "interval":
            lo = float(c.params["min"])
            hi = float(c.params["max"])

            def _pred(m: Dict[str, Any], metric=c.metric, lo=lo, hi=hi) -> bool:
                v = float(m[metric])
                return lo <= v <= hi

            compiled.append(CompiledCondition(
                id=c.id,
                metric=c.metric,
                rule=c.rule,
                params={"min": lo, "max": hi},
                predicate=_pred,
            ))

        elif c.rule == "percentile_band":
            lower = int(c.params["lower"])
            upper = int(c.params["upper"])
            df = reference_scalar[c.metric]
            lo = _value_at_percentile(df, lower)
            hi = _value_at_percentile(df, upper)

            def _pred(m: Dict[str, Any], metric=c.metric, lo=lo, hi=hi) -> bool:
                v = float(m[metric])
                return lo <= v <= hi

            compiled.append(CompiledCondition(
                id=c.id,
                metric=c.metric,
                rule=c.rule,
                params={"lower": lower, "upper": upper, "min": lo, "max": hi},
                predicate=_pred,
            ))

        elif c.rule == "include_set":
            allowed = set(c.params["allowed"])

            def _pred(m: Dict[str, Any], allowed=allowed) -> bool:
                return str(m["decade_pattern"]) in allowed

            compiled.append(CompiledCondition(
                id=c.id,
                metric=c.metric,
                rule=c.rule,
                params={"allowed": sorted(list(allowed))},
                predicate=_pred,
            ))

        elif c.rule == "exclude_set":
            forbidden = set(c.params["forbidden"])

            def _pred(m: Dict[str, Any], forbidden=forbidden) -> bool:
                return str(m["decade_pattern"]) not in forbidden

            compiled.append(CompiledCondition(
                id=c.id,
                metric=c.metric,
                rule=c.rule,
                params={"forbidden": sorted(list(forbidden))},
                predicate=_pred,
            ))
        
        elif c.rule == "distribution_constraint" and c.metric == "decades":
             max_bin = c.params.get("max_bin_count")
             min_bins = c.params.get("min_nonempty_bins")
             
             def _pred(m: Dict[str, Any], max_bin=max_bin, min_bins=min_bins) -> bool:
                 # bins vienen de compute_structural_metrics
                 bins = [
                     int(m[k])
                     for k in ["d1_09", "d10_19", "d20_29", "d30_39", "d40_49", "d50_54"]
                 ]
             
                 if max_bin is not None and max(bins) > max_bin:
                     return False
             
                 if min_bins is not None and sum(b > 0 for b in bins) < min_bins:
                     return False
             
                 return True
             
             compiled.append(
                 CompiledCondition(
                     id=c.id,
                     metric="decades",
                     rule="distribution_constraint",
                     params=dict(c.params),
                     predicate=_pred,
                 )
             )

        else:
            raise ValueError(f"Regla no soportada: {c.rule}")

    return compiled


# ---------------------------------------------------------------------
# GENERATION (uniform trials + accept/reject)
# ---------------------------------------------------------------------

def _decade_pattern_from_metrics(metrics: Dict[str, Any]) -> str:
    # structural_metrics returns decade bins as d1_09, d10_19, ... d50_54
    keys = ["d1_09", "d10_19", "d20_29", "d30_39", "d40_49", "d50_54"]
    return ",".join(str(int(metrics[k])) for k in keys)


def _trial_generator(seed: int, lo: int = 1, hi: int = 54, k: int = 5):
    rng = random.Random(seed)
    population = list(range(lo, hi + 1))
    while True:
        combo = rng.sample(population, k)
        combo.sort()
        yield combo


def _accept(metrics: Dict[str, Any], compiled: List[CompiledCondition]) -> bool:
    return all(cc.predicate(metrics) for cc in compiled)


# ---------------------------------------------------------------------
# DISTRIBUTIONS + COMPARISON
# ---------------------------------------------------------------------

def _compute_distributions(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    scalar = compute_scalar_metric_distributions(metrics_df)  # dict metric -> df(percentile,value)
    decade = compute_decade_patterns(metrics_df)              # df(pattern,frequency,relative_frequency)
    return {"scalar_distributions": scalar, "decade_patterns": decade}


def _compare_vs_reference(
    conditioned_scalar: Dict[str, pd.DataFrame],
    reference_scalar: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for metric in ("range", "sum", "std"):
        cdf = conditioned_scalar[metric].copy()
        rdf = reference_scalar[metric].copy()
        merged = rdf.merge(cdf, on="percentile", suffixes=("_reference", "_conditioned"))
        for _, r in merged.iterrows():
            ref = float(r["value_reference"])
            cond = float(r["value_conditioned"])
            delta_abs = cond - ref
            delta_pct = (delta_abs / ref * 100.0) if ref != 0 else None
            rows.append({
                "metric": metric,
                "percentile": int(r["percentile"]),
                "value_reference": ref,
                "value_conditioned": cond,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------

def export_conditioned_artifacts(
    cfg: ConditionedConfig,
    distributions: Dict[str, Any],
    comparison_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    scalar: Dict[str, pd.DataFrame] = distributions["scalar_distributions"]
    for metric in ("range", "sum", "std"):
        df = scalar[metric].copy()[["percentile", "value"]]
        df.to_csv(out / f"conditioned_{metric}_percentiles.csv", index=False)

    decade_df: pd.DataFrame = distributions["decade_patterns"].copy()
    decade_df = decade_df[["pattern", "frequency", "relative_frequency"]]
    decade_df.to_csv(out / "conditioned_decade_patterns.csv", index=False)

    comparison_df.to_csv(out / "comparison_vs_reference.csv", index=False)

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------------------

def run_monte_carlo_conditioned(
    name: str,
    description: str,
    seed: int,
    N_target: int,
    conditions: List[Dict[str, Any]],
    max_iterations: int = 500_000,
    reference_dir: Path = DEFAULT_REFERENCE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export: bool = True,
    return_combinations: bool = False,
) -> Dict[str, Any]:
    """
    Run conditioned Monte Carlo generation (accept/reject based on structural conditions).

    Parameters
    ----------
    conditions : list of dict
        Each dict must match the Condition schema:
          { "id":..., "metric":..., "rule":..., "params": {...} }

    reference_dir : Path
        Directory holding reference percentiles/patterns (default: baseline artifacts)

    Returns
    -------
    dict with:
      - metrics_df
      - scalar_distributions
      - decade_patterns
      - comparison_vs_reference
      - output_dir
      - metadata
      - combinations (optional)
    """
    cond_objs = [Condition(**c) for c in conditions]
    cfg = ConditionedConfig(
        name=name,
        description=description,
        seed=seed,
        N_target=N_target,
        max_iterations=max_iterations,
        conditions=cond_objs,
        reference_dir=Path(reference_dir),
        output_dir=Path(output_dir),
        export=export,
        return_combinations=return_combinations,
    )

    validate_config(cfg)

    started_at = _iso_utc_now()
    t0 = time.perf_counter()

    # Load reference artifacts
    reference_scalar = _load_reference_scalar_percentiles(cfg.reference_dir)

    # Compile conditions (resolve percentile bands)
    compiled = _compile_conditions(cfg, reference_scalar)

    # Generate + accept/reject
    accepted_combos: List[List[int]] = []
    accepted_metrics: List[Dict[str, Any]] = []

    gen = _trial_generator(cfg.seed)

    iterations = 0
    while len(accepted_combos) < cfg.N_target and iterations < cfg.max_iterations:
        iterations += 1
        combo = next(gen)

        m = compute_structural_metrics(combo)
        # add decade_pattern explicitly as a discrete field for filtering
        m["decade_pattern"] = _decade_pattern_from_metrics(m)

        if _accept(m, compiled):
            accepted_combos.append(combo)
            # keep metrics (drop decade_pattern? keep it for debug and/or future)
            accepted_metrics.append(m)

    t1 = time.perf_counter()
    finished_at = _iso_utc_now()
    duration_seconds = t1 - t0

    _require(len(accepted_combos) == cfg.N_target, (
        f"No se alcanzó N_target={cfg.N_target}. "
        f"Aceptadas={len(accepted_combos)} tras iterations={iterations}. "
        f"Relaja condiciones o aumenta max_iterations."
    ))

    metrics_df = pd.DataFrame(accepted_metrics)

    distributions = _compute_distributions(metrics_df)

    comparison_df = _compare_vs_reference(
        conditioned_scalar=distributions["scalar_distributions"],
        reference_scalar=reference_scalar,
    )

    acceptance_rate = float(cfg.N_target) / float(iterations) if iterations > 0 else 0.0

    # Metadata
    metadata = {
        "module": "monte_carlo_conditioned",
        "game": "gordo_primitiva",
        "reference": {
            "type": "baseline_percentiles",
            "reference_dir": str(cfg.reference_dir),
        },
        "config": {
            "name": cfg.name,
            "description": cfg.description,
            "seed": int(cfg.seed),
            "N_target": int(cfg.N_target),
            "max_iterations": int(cfg.max_iterations),
        },
        "conditions": [
            {
                "id": cc.id,
                "metric": cc.metric,
                "rule": cc.rule,
                "params": cc.params,
            }
            for cc in compiled
        ],
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": float(duration_seconds),
            "iterations": int(iterations),
            "accepted": int(cfg.N_target),
            "acceptance_rate": acceptance_rate,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "platform_detail": platform.platform(),
            "pandas_version": getattr(pd, "__version__", "unknown"),
        },
        "notes": "Monte Carlo condicionado por métricas estructurales (aceptación/rechazo). No predictivo.",
    }

    if cfg.export:
        export_conditioned_artifacts(
            cfg=cfg,
            distributions=distributions,
            comparison_df=comparison_df,
            metadata=metadata,
        )

    result: Dict[str, Any] = {
        "metrics_df": metrics_df,
        "scalar_distributions": distributions["scalar_distributions"],
        "decade_patterns": distributions["decade_patterns"],
        "comparison_vs_reference": comparison_df,
        "output_dir": str(cfg.output_dir),
        "metadata": metadata,
    }

    if cfg.return_combinations:
        result["combinations"] = accepted_combos

    return result