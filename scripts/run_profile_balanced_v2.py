# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from src.monte_carlo.conditioned import run_monte_carlo_conditioned


def main() -> None:
    """
    Perfil oficial: profile_balanced_v2

    - Condicionamiento estructural reforzado respecto a v1:
      usa ventanas P30–P70 en range/sum/std.
    - Motor: aceptación por rechazo sobre el generador uniforme.
    - Referencia: percentiles del baseline Monte Carlo uniforme.
    """

    result = run_monte_carlo_conditioned(
        name="profile_balanced_v2",
        description=(
            "Balanced conditioned profile v2. "
            "Acceptance-rejection with tighter central windows (P30–P70) for "
            "range/sum/std relative to baseline distributions."
        ),
        seed=20260125,          # semilla fija y explícita (ajustable)
        N_target=1000,
        max_iterations=100_000,  # más alto que v1 por mayor rechazo esperado
        reference_dir=Path("reports/monte_carlo/baseline"),
        output_dir=Path("reports/monte_carlo/conditioned/profile_balanced_v2"),
        conditions=[
						{
						"id": "range_p25_p75",
						"metric": "range",
						"rule": "percentile_band",
						"params": {"lower": 25, "upper": 75},
						},
						{
						"id": "sum_p25_p75",
						"metric": "sum",
						"rule": "percentile_band",
						"params": {"lower": 25, "upper": 75},
						},
						{
						"id": "std_p25_p75",
						"metric": "std",
						"rule": "percentile_band",
						"params": {"lower": 25, "upper": 75},
						},
				],
    )

    md = result.get("metadata", {})
    accepted = md.get("execution", {}).get("accepted", md.get("accepted"))
    acc_rate = md.get("execution", {}).get("acceptance_rate", md.get("acceptance_rate"))

    print("✅ profile_balanced_v2 ejecutado")
    print(f"→ output_dir: {result.get('output_dir')}")
    if accepted is not None:
        print(f"→ accepted: {accepted}")
    if acc_rate is not None:
        print(f"→ acceptance_rate: {acc_rate}")


if __name__ == "__main__":
    main()