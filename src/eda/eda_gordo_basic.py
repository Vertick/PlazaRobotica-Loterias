# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = Path("data/curated/curated_gordo_primitiva.parquet")
DEFAULT_REPORTS_DIR = Path("reports/eda")


def load_curated_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load curated dataset for Gordo de la Primitiva.
    """
    if not path.exists():
        raise FileNotFoundError(f"Curated dataset not found: {path}")

    return pd.read_parquet(path)


def compute_number_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frequency of main numbers (1-54).
    """
    numbers = pd.concat(
        [df[f"n{i}"] for i in range(1, 6)],
        ignore_index=True
    )

    freq = numbers.value_counts().sort_index()

    return pd.DataFrame({
        "number": freq.index,
        "frequency": freq.values,
        "relative_frequency": freq.values / freq.values.sum()
    })


def compute_key_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frequency of key number (0-9).
    """
    freq = df["clave"].value_counts().sort_index()

    return pd.DataFrame({
        "clave": freq.index,
        "frequency": freq.values,
        "relative_frequency": freq.values / freq.values.sum()
    })


def compute_draws_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute number of draws per year.
    """
    years = df["draw_date"].dt.year
    freq = years.value_counts().sort_index()

    return pd.DataFrame({
        "year": freq.index,
        "draws": freq.values
    })


def export_reports(
    number_freq: pd.DataFrame,
    key_freq: pd.DataFrame,
    draws_per_year: pd.DataFrame,
    output_dir: Path = DEFAULT_REPORTS_DIR
) -> None:
    """
    Export EDA results to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    number_freq.to_csv(output_dir / "frequency_numbers.csv", index=False)
    key_freq.to_csv(output_dir / "frequency_key.csv", index=False)
    draws_per_year.to_csv(output_dir / "draws_per_year.csv", index=False)


def run_basic_eda(
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_REPORTS_DIR
) -> dict:
    """
    Run basic EDA pipeline.
    """
    df = load_curated_dataset(data_path)

    number_freq = compute_number_frequencies(df)
    key_freq = compute_key_frequencies(df)
    draws_per_year = compute_draws_per_year(df)

    export_reports(number_freq, key_freq, draws_per_year, output_dir)

    return {
        "number_frequencies": number_freq,
        "key_frequencies": key_freq,
        "draws_per_year": draws_per_year
    }