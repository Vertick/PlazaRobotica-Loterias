# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Sequence
import random


@dataclass(frozen=True)
class BundleMCResult:
    trials: int
    hits: int
    estimated_probability: float
    theoretical_probability: float
    absolute_error: float
    relative_error: float


def _validate_set8(set8: Sequence[int]) -> None:
    if len(set8) != 8:
        raise ValueError("set8 must contain exactly 8 numbers.")
    if len(set(set8)) != 8:
        raise ValueError("set8 must not contain duplicates.")
    if not all(1 <= n <= 54 for n in set8):
        raise ValueError("set8 numbers must be in [1, 54].")


def evaluate_bundle_mc(
    set8: Sequence[int],
    *,
    trials: int,
    seed: int,
) -> BundleMCResult:
    _validate_set8(set8)
    if trials <= 0:
        raise ValueError("trials must be > 0.")

    rng = random.Random(seed)
    population = list(range(1, 55))
    set8_set = set(set8)

    hits = 0
    for _ in range(trials):
        draw = rng.sample(population, 5)
        if set(draw).issubset(set8_set):
            hits += 1

    estimated = hits / trials
    theoretical = comb(8, 5) / comb(54, 5)
    abs_error = abs(estimated - theoretical)
    rel_error = abs_error / theoretical if theoretical > 0 else 0.0

    return BundleMCResult(
        trials=trials,
        hits=hits,
        estimated_probability=estimated,
        theoretical_probability=theoretical,
        absolute_error=abs_error,
        relative_error=rel_error,
    )
