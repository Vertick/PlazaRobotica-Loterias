Estado del proyecto

Fase actual: Monte Carlo condicionado por métricas estructurales

Última actualización: 2026-01-29

Módulos cerrados
1. Ingesta y curación de datos históricos

Estado: CERRADO

Fuente de datos:

Lotoideas.com (histórico del Gordo de la Primitiva)

Periodo cubierto:

2005-02-06 → 2026-01-18

Descripción:

Carga manual del histórico completo del Gordo de la Primitiva.

Normalización a un esquema canónico estable.

Validaciones estructurales y de dominio:

rangos numéricos

unicidad de números por sorteo

ausencia de nulos

Generación de dataset curado y reporte de validación.

Uso de timestamps UTC con zona horaria explícita.

Esquema canónico:

draw_date

n1, n2, n3, n4, n5

clave

game

source

ingested_at

Artefactos generados:

data/curated/curated_gordo_primitiva.parquet

data/curated/curated_gordo_primitiva_report.json

Resultado:

Dataset validado con 1.084 sorteos.

Validación superada sin errores.

Limitaciones conocidas:

Fuente no oficial (SELAE), aunque ampliamente utilizada.

No incluye información sobre apuestas reales del público.

El histórico no habilita capacidad predictiva causal.

2. Análisis Exploratorio de Datos (EDA) — Gordo de la Primitiva

Estado: CERRADO

Descripción: Se ha realizado un análisis exploratorio básico, objetivo y reproducible del histórico del Gordo de la Primitiva con el fin de caracterizar el espacio estadístico del juego y validar la calidad del dataset curado.

Análisis realizados:

Frecuencia de aparición de los números principales (1–54).

Frecuencia del número clave (0–9).

Evolución temporal del número de sorteos por año.

Chequeos de consistencia estadística como sanity check.

Resultados principales:

Las frecuencias observadas son compatibles con un proceso aleatorio uniforme.

No se detectan anomalías estructurales ni sesgos estadísticamente concluyentes.

El número clave presenta una distribución consistente con uniformidad.

El histórico es temporalmente coherente y estable.

Artefactos generados:

reports/eda/frequency_numbers.csv

reports/eda/frequency_key.csv

reports/eda/draws_per_year.csv

Conclusiones y límites:

El análisis no revela capacidad predictiva real.

Los resultados validan el uso de modelos uniformes como baseline.

Cualquier estrategia posterior debe considerarse heurística y experimental.

3. Definición de métricas estructurales — Gordo de la Primitiva

Estado: CERRADO

Descripción: Se han definido e implementado métricas estructurales empíricas para caracterizar la forma de las combinaciones del Gordo de la Primitiva, sin introducir supuestos predictivos.

Métricas implementadas:

Rango (mínimo, máximo y amplitud).

Suma total y media.

Dispersión interna (desviación estándar).

Distribución por decenas.

Análisis realizados:

Cálculo de distribuciones empíricas mediante percentiles (P10, P25, P50, P75, P90).

Identificación de intervalos estructuralmente típicos.

Análisis de frecuencia de patrones de decenas.

Artefactos generados:

reports/metrics/range_percentiles.csv

reports/metrics/sum_percentiles.csv

reports/metrics/std_percentiles.csv

reports/metrics/decade_patterns.csv

Conclusiones y límites:

Las métricas describen estructura, no probabilidad de acierto.

No se descarta ningún tipo de combinación.

Los resultados sirven como base para simulaciones y generación condicionada.

4. Simulación Monte Carlo — Gordo de la Primitiva

Estado: CERRADO

Descripción: Se ha implementado un módulo completo de simulación Monte Carlo con muestreo uniforme puro, utilizado como baseline de referencia para el análisis estructural del Gordo de la Primitiva.

El objetivo del módulo no es generar predicciones ni mejorar la probabilidad de acierto, sino establecer el comportamiento esperado del sistema bajo un proceso aleatorio ideal.

Resultados clave:

Compatibilidad estructural total entre histórico y baseline.

Ausencia de sesgos explotables.

Base sólida para condicionamiento posterior.

Conclusión: El histórico es estructuralmente indistinguible de un proceso aleatorio uniforme.

Módulo cerrado
5. Monte Carlo condicionado por métricas estructurales — profile_balanced_v3

Estado: CERRADO

Descripción: Perfil condicionado final y estable basado en rechazo Monte Carlo, con ventanas centrales reforzadas y diversidad discreta controlada.

Condiciones estructurales:

range ∈ P35–P65 (baseline)

sum ∈ P35–P65 (baseline)

std ∈ P35–P65 (baseline)

Restricción de décadas: max_bin_count ≤ 2, min_nonempty_bins ≥ 3

Ejecución:

Objetivo: 1.000 combinaciones aceptadas

Iteraciones: 15.543

Tasa de aceptación: ≈ 6.4 %

Reproducibilidad garantizada por semilla

Selección final (A+B+C):

A_central (50 %): Núcleo representativo del perfil

B_diverse (30 %): Diversificación por patrones de decenas

C_edge (20 %): Bordes estructurales permitidos

Nota: C_edge se define mediante semántica OR: basta con que una métrica esté en zona de borde.

Artefactos generados:

reports/monte_carlo/conditioned/profile_balanced_v3/metrics_df.csv

reports/final/final_combinations_v3.csv

reports/monte_carlo/conditioned/profile_balanced_v3/conditioned_*_percentiles.csv

Validación estructural formal

Estado: SUPERADA

La selección final se valida mediante un script independiente que comprueba:

Integridad estructural del CSV

Distribución A/B/C

Cumplimiento de bandas percentiles

Diversidad de patrones de decenas

Reproducibilidad declarativa

Resultado:

{
  "valid": true,
  "checks": {
    "integrity": "OK",
    "buckets": "OK",
    "percentiles": "OK",
    "decades": "OK",
    "reproducibility": "OK"
  },
  "profile": "profile_balanced_v3",
  "n_total": 10,
  "notes": "Full structural validation passed. No explicit combinations."
}
Conclusión final

Este proyecto demuestra que es posible:

Analizar juegos de azar con rigor estadístico

Construir generadores reproducibles sin falacias predictivas

Documentar explícitamente límites y supuestos

No se pretende ganar al azar. Se pretende entenderlo, simularlo y documentarlo con honestidad técnica.

Próximos pasos (opcional)

Nuevos perfiles condicionados con sesgos discretos explícitos

Comparativas multi-perfil

Modelos de Markov de bajo orden (exploratorios)

Cada ampliación se documentará y cerrará formalmente.
