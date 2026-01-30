# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import random


# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def percentile_interp(pctl: dict[int, float], p: float) -> float:
    keys = sorted(pctl.keys())
    if p in pctl:
        return pctl[p]

    for lo, hi in zip(keys[:-1], keys[1:]):
        if lo < p < hi:
            w = (p - lo) / (hi - lo)
            return pctl[lo] * (1 - w) + pctl[hi] * w

    raise ValueError(f"Percentil {p} fuera de rango")


def load_percentiles(path: Path) -> dict[int, float]:
    df = pd.read_csv(path)
    return {int(r.percentile): float(r.value) for r in df.itertuples()}


def compute_score_A(row, pctl):
    score = 0.0
    for m in ("range", "sum", "std"):
        p25, p50, p75 = pctl[m][25], pctl[m][50], pctl[m][75]
        iqr = max(p75 - p25, 1e-9)
        score -= abs((row[m] - p50) / iqr)
    return score


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seleccion final reproducible de combinaciones (A+B+C)"
    )
    parser.add_argument("--metrics", required=True, help="CSV con metrics_df")
    parser.add_argument("--percentiles-dir", required=True)
    parser.add_argument("--n-total", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260126)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.metrics)

    # ------------------------------------------------------------
    # Percentiles del perfil
    # ------------------------------------------------------------
    pctl = {
        "range": load_percentiles(Path(args.percentiles_dir) / "conditioned_range_percentiles.csv"),
        "sum": load_percentiles(Path(args.percentiles_dir) / "conditioned_sum_percentiles.csv"),
        "std": load_percentiles(Path(args.percentiles_dir) / "conditioned_std_percentiles.csv"),
    }

    # ------------------------------------------------------------
    # Score central (A)
    # ------------------------------------------------------------
    df["score_A"] = df.apply(lambda r: compute_score_A(r, pctl), axis=1)
    df = df.sort_values("score_A", ascending=False).reset_index(drop=True)

    n_A = int(args.n_total * 0.5)
    n_B = int(args.n_total * 0.3)
    n_C = args.n_total - n_A - n_B

    selected = []

    # ??? A — Central
    A = df.head(n_A).copy()
    A["bucket"] = "A_central"
    selected.append(A)

    used_patterns = set(A["decade_pattern"])

    # ??? B — Diversificada
    B_rows = []
    for _, r in df.iterrows():
        if len(B_rows) >= n_B:
            break
        if r["decade_pattern"] in used_patterns:
            continue
        B_rows.append(r)
        used_patterns.add(r["decade_pattern"])

    B = pd.DataFrame(B_rows)
    B["bucket"] = "B_diverse"
    selected.append(B)

    # ?? C — Bordes
    def is_edge(r):
        for m in ("range", "sum", "std"):
            low = percentile_interp(pctl[m], 35)
            high = percentile_interp(pctl[m], 65)

            span = high - low
            if span <= 0:
                continue

            # ~15% interior a cada extremo del núcleo
            if (
                low <= r[m] <= low + 0.15 * span
                or high - 0.15 * span <= r[m] <= high
            ):
                return True

        return False

    C = (
        df[~df.index.isin(pd.concat(selected).index)]
        .loc[df.apply(is_edge, axis=1)]
        .copy()
    )

    C = C.sort_values("score_A").head(n_C)
    C["bucket"] = "C_edge"
    selected.append(C)

    # ------------------------------------------------------------
    # Export final (SIN combo, por diseño)
    # ------------------------------------------------------------
    out = pd.concat(selected).reset_index(drop=True)

    out["notes"] = out["bucket"].map({
        "A_central": "Representativo del nucleo del perfil",
        "B_diverse": "Diversificacion estructural interna",
        "C_edge": "Ejemplo de borde permitido del perfil",
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("? Seleccion final generada")
    print(out[["bucket"]].value_counts())


if __name__ == "__main__":
    main()