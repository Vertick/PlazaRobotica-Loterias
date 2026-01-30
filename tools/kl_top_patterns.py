# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


EPSILON = 1e-12


def load_distribution(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pattern", "relative_frequency"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV inválido: {path}, columnas requeridas: {required}")
    return df[["pattern", "relative_frequency"]]


def align_distributions(
    p: pd.DataFrame,
    q: pd.DataFrame,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """
    Alinea soportes P y Q y aplica suavizado.
    """
    df = (
        p.rename(columns={"relative_frequency": "p"})
        .merge(
            q.rename(columns={"relative_frequency": "q"}),
            on="pattern",
            how="outer",
        )
        .fillna(0.0)
    )

    df["p"] = df["p"] + epsilon
    df["q"] = df["q"] + epsilon

    # Renormalización explícita
    df["p"] = df["p"] / df["p"].sum()
    df["q"] = df["q"] / df["q"].sum()

    return df


def kl_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    KL(P‖Q) = sum p * log(p/q)
    """
    df = df.copy()
    df["kl_term"] = df["p"] * np.log(df["p"] / df["q"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descomposición Top-K de KL-divergence por patrón"
    )
    parser.add_argument("--p", required=True, help="CSV del perfil P (destino)")
    parser.add_argument("--q", required=True, help="CSV del perfil Q (referencia)")
    parser.add_argument("--top", type=int, default=10, help="Top-K contribuyentes")
    parser.add_argument("--out", required=True, help="CSV de salida")

    args = parser.parse_args()

    p_path = Path(args.p)
    q_path = Path(args.q)
    out_path = Path(args.out)

    p_df = load_distribution(p_path)
    q_df = load_distribution(q_path)

    aligned = align_distributions(p_df, q_df)
    decomposed = kl_decomposition(aligned)

    kl_total = decomposed["kl_term"].sum()

    top_df = (
        decomposed
        .assign(kl_pct=lambda d: d["kl_term"] / kl_total * 100.0)
        .sort_values("kl_term", ascending=False)
        .head(args.top)
        .reset_index(drop=True)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(out_path, index=False)

    print("✅ KL decomposition completada")
    print(f"→ KL(P‖Q): {kl_total:.6f}")
    print(f"→ Top-{args.top} exportado a: {out_path}")


if __name__ == "__main__":
    main()