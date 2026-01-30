import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
import json

# ------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------

INPUT_CSV = "Lotoideas.com - Histórico de Resultados - Gordo de la Primitiva - 2005 a 202X.csv"
OUTPUT_DIR = Path("data/curated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = OUTPUT_DIR / "curated_gordo_primitiva.parquet"
OUTPUT_REPORT = OUTPUT_DIR / "curated_gordo_primitiva_report.json"

SOURCE = "lotoideas.com"
GAME = "gordo_primitiva"

# ------------------------------------------------------------------
# CARGA
# ------------------------------------------------------------------

df_raw = pd.read_csv(INPUT_CSV, encoding="latin1")

# ------------------------------------------------------------------
# NORMALIZACIÓN
# ------------------------------------------------------------------

df = pd.DataFrame({
    "draw_date": pd.to_datetime(df_raw["FECHA"], dayfirst=True),
    "n1": df_raw["COMB. GANADORA"],
    "n2": df_raw["Unnamed: 2"],
    "n3": df_raw["Unnamed: 3"],
    "n4": df_raw["Unnamed: 4"],
    "n5": df_raw["Unnamed: 5"],
    "clave": df_raw["CLAVE / R"],
})

df["game"] = GAME
df["source"] = SOURCE
df["ingested_at"] = datetime.now(timezone.utc)

# ------------------------------------------------------------------
# VALIDACIONES
# ------------------------------------------------------------------

errors = []

# Estructura
expected_cols = ["draw_date", "n1", "n2", "n3", "n4", "n5", "clave"]
missing = set(expected_cols) - set(df.columns)
if missing:
    errors.append(f"Missing columns: {missing}")

# Dominio numérico
for col in ["n1", "n2", "n3", "n4", "n5"]:
    if not df[col].between(1, 54).all():
        errors.append(f"Out-of-range values in {col}")

if not df["clave"].between(0, 9).all():
    errors.append("Out-of-range values in clave")

# Repeticiones dentro de sorteo
def has_duplicates(row):
    nums = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
    return len(nums) != len(set(nums))

dup_rows = df[df.apply(has_duplicates, axis=1)]
if not dup_rows.empty:
    errors.append(f"{len(dup_rows)} draws with repeated numbers")

# Nulos
if df.isnull().any().any():
    errors.append("Null values detected")

# ------------------------------------------------------------------
# INFORME
# ------------------------------------------------------------------

report = {
    "dataset": "Gordo de la Primitiva",
    "source": SOURCE,
    "game": GAME,
    "rows": int(len(df)),
    "date_range": {
        "start": df["draw_date"].min().date().isoformat(),
        "end": df["draw_date"].max().date().isoformat()
    },
    "validation": {
        "status": "OK" if not errors else "FAILED",
        "errors": errors
    },
    "generated_at": datetime.now(timezone.utc).isoformat()
}

# ------------------------------------------------------------------
# SALIDA
# ------------------------------------------------------------------

if errors:
    print("❌ VALIDATION FAILED")
    for e in errors:
        print(" -", e)
else:
    df.to_parquet(OUTPUT_PARQUET, index=False)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("✅ Dataset curado generado correctamente")
    print(f"→ {OUTPUT_PARQUET}")
    print(f"→ {OUTPUT_REPORT}")