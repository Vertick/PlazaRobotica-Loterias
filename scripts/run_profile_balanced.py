from pathlib import Path
from src.monte_carlo.conditioned import run_monte_carlo_conditioned

result = run_monte_carlo_conditioned(
    name="profile_balanced_v1",
    description="Perfil estructuralmente balanceado (P25–P75 en range, sum y std)",
    seed=42,
    N_target=1000,
    max_iterations=100_000,
    conditions=[
        {
            "id": "range_central",
            "metric": "range",
            "rule": "percentile_band",
            "params": {"lower": 25, "upper": 75}
        },
        {
            "id": "sum_central",
            "metric": "sum",
            "rule": "percentile_band",
            "params": {"lower": 25, "upper": 75}
        },
        {
            "id": "std_central",
            "metric": "std",
            "rule": "percentile_band",
            "params": {"lower": 25, "upper": 75}
        }
    ],
    reference_dir=Path("reports/monte_carlo/baseline"),
    output_dir=Path("reports/monte_carlo/conditioned/profile_balanced_v1"),
    return_combinations=True,
)