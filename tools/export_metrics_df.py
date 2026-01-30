# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sys

# asegurar root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.structural_metrics import compute_structural_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta metrics_df.csv a partir de accepted_combinations.csv"
    )
    parser.add_argument(
        "--combinations",
        required=True,
        help="CSV con combinaciones aceptadas (combo=1-2-3-4-5)",
    )
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(args.combinations)

    if "combo" not in df_in.columns:
        raise ValueError("El CSV de entrada debe tener una columna 'combo'")

    records = []
    for combo_str in df_in["combo"]:
        combo = list(map(int, combo_str.split("-")))
        metrics = compute_structural_metrics(combo)
        metrics["combo"] = combo_str
        records.append(metrics)

    df = pd.DataFrame(records)

    out_path = out_dir / "metrics_df.csv"
    df.to_csv(out_path, index=False)

    print("✅ metrics_df exportado correctamente")
    print(f"→ {out_path}")
    print(f"→ filas: {len(df)}")


if __name__ == "__main__":
    main()