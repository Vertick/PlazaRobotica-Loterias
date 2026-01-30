# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import random

from .set8_scoring import constraints_ok, decade_bins, profile_compatibility, score_set8


@dataclass(frozen=True)
class Set8Selection:
    set8: tuple[int, ...]
    strategy: str
    score: float
    score_components: dict
    constraints_ok: bool
    explain: dict


def _validate_pool5(pool5: Iterable[Sequence[int]]) -> List[Tuple[int, int, int, int, int]]:
    validated: List[Tuple[int, int, int, int, int]] = []
    for combo in pool5:
        if len(combo) != 5:
            raise ValueError("pool5 must contain 5-number combinations.")
        combo_tuple = tuple(int(x) for x in combo)
        if len(set(combo_tuple)) != 5:
            raise ValueError("pool5 contains repeated numbers in a combination.")
        if not all(1 <= n <= 54 for n in combo_tuple):
            raise ValueError("pool5 numbers must be in [1, 54].")
        validated.append(combo_tuple)
    if not validated:
        raise ValueError("pool5 must not be empty.")
    return validated


def _number_frequencies(pool5: Iterable[Sequence[int]]) -> Dict[int, int]:
    freq: Dict[int, int] = {}
    for combo in pool5:
        for n in combo:
            freq[n] = freq.get(n, 0) + 1
    return freq


def _decade_key(n: int) -> str:
    if 1 <= n <= 9:
        return "d1_09"
    if 10 <= n <= 19:
        return "d10_19"
    if 20 <= n <= 29:
        return "d20_29"
    if 30 <= n <= 39:
        return "d30_39"
    if 40 <= n <= 49:
        return "d40_49"
    return "d50_54"


def _ranked_candidates(numbers: Iterable[int], freq: Dict[int, int]) -> List[int]:
    return sorted(numbers, key=lambda n: (-freq.get(n, 0), n))


def _select_cover(
    *,
    pool5: List[Tuple[int, int, int, int, int]],
    constraints: Dict[str, int],
    seed: Optional[int],
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    freq = _number_frequencies(pool5)
    unique_numbers = sorted(freq.keys())
    if len(unique_numbers) < 8:
        raise ValueError("pool5 must contain at least 8 unique numbers.")

    decade_map: Dict[str, List[int]] = {}
    for n in unique_numbers:
        decade_map.setdefault(_decade_key(n), []).append(n)

    selected: List[int] = []
    picked_by_decade: Dict[str, int] = {}
    for decade in ("d1_09", "d10_19", "d20_29", "d30_39", "d40_49", "d50_54"):
        candidates = decade_map.get(decade, [])
        if not candidates or len(selected) >= 8:
            continue
        choice = _ranked_candidates(candidates, freq)[0]
        selected.append(choice)
        picked_by_decade[decade] = choice

    concentration_penalty = float(constraints.get("concentration_penalty", 0.15))

    while len(selected) < 8:
        best = None
        best_score = None
        current_bins = decade_bins(selected)
        for n in unique_numbers:
            if n in selected:
                continue
            decade = _decade_key(n)
            bin_count = current_bins[decade]
            base = freq.get(n, 0) / max(freq.values())
            score = base - concentration_penalty * bin_count
            candidate = (score, -bin_count, -base, n)
            if best_score is None or candidate > best_score:
                best_score = candidate
                best = n
        if best is None:
            break
        selected.append(best)

    selected_tuple = tuple(sorted(selected))
    explain = {
        "picked_by_decade": picked_by_decade,
        "concentration_penalty": concentration_penalty,
        "unique_numbers": len(unique_numbers),
    }
    return selected_tuple, explain


def _select_pv3(
    *,
    pool5: List[Tuple[int, int, int, int, int]],
    constraints: Dict[str, int],
    seed: Optional[int],
) -> Tuple[Tuple[int, ...], Dict[str, object], Dict[str, object]]:
    freq = _number_frequencies(pool5)
    unique_numbers = sorted(freq.keys())
    if len(unique_numbers) < 8:
        raise ValueError("pool5 must contain at least 8 unique numbers.")

    rng = random.Random(seed)
    n_candidates = int(constraints.get("n_candidates", 200))
    max_bin = constraints.get("max_bin_count")
    min_bins = constraints.get("min_nonempty_bins")

    profile_path = Path(constraints.get("profile_path", "configs/profiles/profile_balanced_v3.json"))
    reference_dir = Path(constraints.get("reference_dir", "reports/monte_carlo/baseline"))

    best = None
    best_score = -1.0
    best_profile = {}
    sampled = 0
    while sampled < n_candidates:
        candidate = rng.sample(unique_numbers, 8)
        candidate.sort()
        bins = decade_bins(candidate)
        if max_bin is not None and max(bins.values()) > max_bin:
            sampled += 1
            continue
        if min_bins is not None and sum(v > 0 for v in bins.values()) < min_bins:
            sampled += 1
            continue
        profile_score = profile_compatibility(
            candidate,
            profile_path=profile_path,
            reference_dir=reference_dir,
        )
        sampled += 1
        score = profile_score["compatibility"]
        if score > best_score:
            best_score = score
            best = tuple(candidate)
            best_profile = profile_score

    if best is None:
        raise ValueError("TODO: unable to generate S8_PV3 candidate set.")

    explain = {
        "sampled_candidates": sampled,
        "profile_path": str(profile_path),
        "reference_dir": str(reference_dir),
    }
    return best, explain, best_profile


def select_set8(
    *,
    pool5: list[tuple[int, int, int, int, int]],
    strategy: Literal["S8_COVER", "S8_PV3"],
    profile: Optional[dict],
    constraints: dict,
    seed: Optional[int] = None
) -> Set8Selection:
    validated_pool5 = _validate_pool5(pool5)
    constraints = dict(constraints or {})

    if strategy == "S8_COVER":
        set8, explain = _select_cover(
            pool5=validated_pool5,
            constraints=constraints,
            seed=seed,
        )
        freq = _number_frequencies(validated_pool5)
        score = score_set8(set8, freq)
        bins = score.bins
        return Set8Selection(
            set8=set8,
            strategy=strategy,
            score=score.total_score,
            score_components={
                "coverage": score.coverage_score,
                "concentration": score.concentration_score,
                "frequency": score.frequency_score,
            },
            constraints_ok=constraints_ok(bins, constraints),
            explain={
                **explain,
                "decade_bins": bins,
            },
        )

    if strategy == "S8_PV3":
        set8, explain, profile_score = _select_pv3(
            pool5=validated_pool5,
            constraints=constraints,
            seed=seed,
        )
        freq = _number_frequencies(validated_pool5)
        score = score_set8(set8, freq)
        bins = score.bins
        return Set8Selection(
            set8=set8,
            strategy=strategy,
            score=score.total_score,
            score_components={
                "coverage": score.coverage_score,
                "concentration": score.concentration_score,
                "frequency": score.frequency_score,
                "profile_compatibility": profile_score.get("compatibility", 0.0),
            },
            constraints_ok=constraints_ok(bins, constraints),
            explain={
                **explain,
                "decade_bins": bins,
                "profile_score": profile_score,
            },
        )

    raise ValueError(f"Unknown strategy: {strategy}")
