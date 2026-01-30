# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import json

from src.metrics.structural_metrics import compute_structural_metrics
from src.monte_carlo.conditioned import (
    Condition,
    ConditionedConfig,
    _accept,
    _compile_conditions,
    _load_reference_scalar_percentiles,
    validate_condition,
)


DECADE_KEYS = ("d1_09", "d10_19", "d20_29", "d30_39", "d40_49", "d50_54")


@dataclass(frozen=True)
class Set8Score:
    coverage_score: float
    concentration_score: float
    frequency_score: float
    total_score: float
    bins: Dict[str, int]


def decade_bins(numbers: Sequence[int]) -> Dict[str, int]:
    bins = {k: 0 for k in DECADE_KEYS}
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


def _coverage_score(bins: Dict[str, int]) -> float:
    nonempty = sum(count > 0 for count in bins.values())
    return nonempty / len(bins)


def _concentration_score(bins: Dict[str, int]) -> float:
    max_bin = max(bins.values()) if bins else 0
    return 1.0 - (max_bin - 1) / 7 if max_bin > 0 else 0.0


def _frequency_score(selected: Iterable[int], freq: Dict[int, int]) -> float:
    selected_list = list(selected)
    if not selected_list:
        return 0.0
    max_freq = max(freq.values()) if freq else 1
    return sum(freq.get(n, 0) / max_freq for n in selected_list) / len(selected_list)


def score_set8(
    set8: Sequence[int],
    freq: Dict[int, int],
    weights: Tuple[float, float, float] = (0.45, 0.35, 0.20),
) -> Set8Score:
    bins = decade_bins(set8)
    coverage = _coverage_score(bins)
    concentration = _concentration_score(bins)
    frequency = _frequency_score(set8, freq)
    total = (
        coverage * weights[0]
        + concentration * weights[1]
        + frequency * weights[2]
    )
    return Set8Score(
        coverage_score=coverage,
        concentration_score=concentration,
        frequency_score=frequency,
        total_score=total,
        bins=bins,
    )


def constraints_ok(bins: Dict[str, int], constraints: Dict[str, int]) -> bool:
    max_bin = constraints.get("max_bin_count")
    min_bins = constraints.get("min_nonempty_bins")
    if max_bin is not None and max(bins.values()) > max_bin:
        return False
    if min_bins is not None and sum(v > 0 for v in bins.values()) < min_bins:
        return False
    return True


def _load_profile(profile_path: Path) -> Dict[str, object]:
    if not profile_path.exists():
        raise FileNotFoundError(f"TODO: profile file not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _conditions_from_profile(profile: Dict[str, object]) -> List[Condition]:
    conditions: List[Condition] = []
    for raw in profile.get("conditions", []):
        condition = Condition(
            id=raw["id"],
            metric=raw["metric"],
            rule=raw["rule"],
            params=raw["params"],
        )
        validate_condition(condition)
        conditions.append(condition)
    return conditions


def _decade_pattern(metrics: Dict[str, int]) -> str:
    return ",".join(str(int(metrics[k])) for k in DECADE_KEYS)


def profile_compatibility(
    set8: Sequence[int],
    *,
    profile_path: Path,
    reference_dir: Path,
) -> Dict[str, object]:
    profile = _load_profile(profile_path)
    conditions = _conditions_from_profile(profile)
    if not conditions:
        raise ValueError("TODO: profile has no conditions to evaluate.")

    reference_scalar = _load_reference_scalar_percentiles(reference_dir)
    cfg = ConditionedConfig(
        name=profile.get("profile_id", "profile") or "profile",
        description=str(profile.get("description", "")),
        seed=0,
        N_target=1,
        conditions=conditions,
        export=False,
    )
    compiled = _compile_conditions(cfg, reference_scalar)

    passed = 0
    total = 0
    for combo in combinations(set8, 5):
        metrics = compute_structural_metrics(list(combo))
        metrics["decade_pattern"] = _decade_pattern(metrics)
        if _accept(metrics, compiled):
            passed += 1
        total += 1

    return {
        "profile_id": profile.get("profile_id", "profile"),
        "passed": passed,
        "total": total,
        "compatibility": passed / total if total else 0.0,
    }
