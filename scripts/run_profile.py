# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import csv

# ------------------------------------------------------------
# Ajuste de PYTHONPATH para ejecución directa
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monte_carlo.conditioned import run_monte_carlo_conditioned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run conditioned Monte Carlo profile"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-target", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Cargar perfil
    # ------------------------------------------------------------
    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    profile_id = profile["profile_id"]
    description = profile.get("description", "")
    conditions = profile["conditions"]

    # ------------------------------------------------------------
    # Ejecutar Monte Carlo condicionado
    # ------------------------------------------------------------
    result = run_monte_carlo_conditioned(
        name=profile_id,
        description=description,
        seed=args.seed,
        N_target=args.n_target,
        max_iterations=args.max_iterations,
        reference_dir=Path(args.baseline),
        output_dir=Path(args.out),
        conditions=conditions,
    )

    # ------------------------------------------------------------
    # DEBUG controlado (opcional pero recomendado ahora)
    # ------------------------------------------------------------
    print("DEBUG result keys:", list(result.keys()))

    metadata = result.get("metadata", {})
    if metadata:
        print("DEBUG metadata:")
        print(json.dumps(metadata, indent=2))

        if "execution" in metadata:
            print("DEBUG execution:")
            print(json.dumps(metadata["execution"], indent=2))

    # ------------------------------------------------------------
    # Exportar combinaciones aceptadas (solo si existen)
    # ------------------------------------------------------------
    accepted = (
        result.get("accepted_combinations")
        or result.get("combinations")
        or result.get("samples")
    )

    if accepted:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_dir / "accepted_combinations.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["combo"])
            for combo in accepted:
                writer.writerow(["-".join(map(str, combo))])

        print(f"→ accepted_combinations.csv exportado ({len(accepted)} filas)")
    else:
        print("⚠️ No se encontraron combinaciones aceptadas (normal si el motor no exporta muestras)")

    # ------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------
    exec_md = metadata.get("execution", {})

    print(f"✅ {profile_id} ejecutado")
    print(f"→ output_dir: {args.out}")

    if exec_md:
        print(f"→ accepted: {exec_md.get('accepted')}")
        print(f"→ acceptance_rate: {exec_md.get('acceptance_rate')}")
    else:
        print("→ acceptance_rate: N/A")
        
    # ------------------------------------------------------------
    # Exportar metrics_df (FUENTE CANÓNICA PARA SELECCIÓN FINAL)
    # ------------------------------------------------------------
    if "metrics_df" in result:
        df = result["metrics_df"]
    
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    
        out_csv = out_dir / "metrics_df.csv"
        df.to_csv(out_csv, index=False)
    
        print(f"→ metrics_df.csv exportado ({len(df)} filas)")
    else:
        print("❌ metrics_df no presente en el resultado")


if __name__ == "__main__":
    main()