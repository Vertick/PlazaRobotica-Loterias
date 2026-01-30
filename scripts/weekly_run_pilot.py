# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import csv
import json

from src.evaluation.bundle_mc import evaluate_bundle_mc
from src.selection.set8_selector import select_set8


DEFAULT_POOL5_CSV = Path("reports/final/final_combinations_v3.csv")
DEFAULT_OUTPUT_DIR = Path("reports/weekly/PILOT")


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_pool5_csv(path: Path) -> List[Tuple[int, int, int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"TODO: pool5 CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and all(f"n{i}" in reader.fieldnames for i in range(1, 6)):
            pool: List[Tuple[int, int, int, int, int]] = []
            for row in reader:
                combo = tuple(int(row[f"n{i}"]) for i in range(1, 6))
                pool.append(combo)  # type: ignore[arg-type]
            return pool

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        pool = []
        for row in reader:
            if not row:
                continue
            combo = tuple(int(x) for x in row[:5])
            if len(combo) != 5:
                raise ValueError("pool5 CSV rows must have at least 5 columns.")
            pool.append(combo)  # type: ignore[arg-type]
        return pool


def _build_constraints(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "max_bin_count": args.max_bin_count,
        "min_nonempty_bins": args.min_nonempty_bins,
        "n_candidates": args.n_candidates,
        "profile_path": args.profile_path,
        "reference_dir": args.reference_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Set-8 pilot (02_Gordo_v2).")
    parser.add_argument("--pool5-csv", type=Path, default=DEFAULT_POOL5_CSV)
    parser.add_argument("--strategy", type=str, default="S8_COVER", choices=("S8_COVER", "S8_PV3"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--max-bin-count", type=int, default=2)
    parser.add_argument("--min-nonempty-bins", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=200)
    parser.add_argument("--profile-path", type=str, default="configs/profiles/profile_balanced_v3.json")
    parser.add_argument("--reference-dir", type=str, default="reports/monte_carlo/baseline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pool5 = _load_pool5_csv(args.pool5_csv)
    constraints = _build_constraints(args)

    selection = select_set8(
        pool5=pool5,
        strategy=args.strategy,
        profile=None,
        constraints=constraints,
        seed=args.seed,
    )

    mc_result = evaluate_bundle_mc(
        selection.set8,
        trials=args.trials,
        seed=args.seed,
    )

    output = {
        "run_id": _iso_utc_now(),
        "strategy": selection.strategy,
        "set8": selection.set8,
        "score": selection.score,
        "score_components": selection.score_components,
        "constraints_ok": selection.constraints_ok,
        "constraints": constraints,
        "selection_explain": selection.explain,
        "bundle_mc": {
            "trials": mc_result.trials,
            "hits": mc_result.hits,
            "estimated_probability": mc_result.estimated_probability,
            "theoretical_probability": mc_result.theoretical_probability,
            "absolute_error": mc_result.absolute_error,
            "relative_error": mc_result.relative_error,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"pilot_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Pilot artifact written to: {out_path}")


if __name__ == "__main__":
    main()
