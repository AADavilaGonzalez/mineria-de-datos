"""Pruebas de diferencias entre grupos del Bike Sharing Dataset.

Se comparan los alquileres totales (``cnt``) entre temporadas mediante un
ANOVA de una via y entre anos mediante una prueba t de Welch.
Ejecutar con: python practica4/main.py [ruta/al/dataset.csv]
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import f_oneway, ttest_ind


TARGET_COLUMN = "cnt"
GROUP_COLUMNS = {
    "season": {
        1: "Primavera",
        2: "Verano",
        3: "Otono",
        4: "Invierno",
    },
    "yr": {0: "2011", 1: "2012"},
}
REQUIRED_COLUMNS = set(GROUP_COLUMNS) | {TARGET_COLUMN}
ALPHA = 0.05


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the columns needed for the hypothesis tests."""
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"La columna '{column}' debe ser numerica.")
    return data


def grouped_samples(data: pd.DataFrame, group_column: str) -> list[tuple[int, pd.Series]]:
    """Return non-empty numeric samples ordered by their group label."""
    samples = []
    for group in sorted(data[group_column].dropna().unique()):
        values = data.loc[data[group_column] == group, TARGET_COLUMN].dropna()
        if not values.empty:
            samples.append((int(group), values))
    if len(samples) < 2:
        raise ValueError(f"La columna '{group_column}' debe contener al menos dos grupos.")
    return samples


def print_group_summary(samples: list[tuple[int, pd.Series]], labels: dict[int, str]) -> None:
    for group, values in samples:
        label = labels.get(group, str(group))
        print(f"  {label}: n={len(values)}, media={values.mean():.2f}")


def run_anova(data: pd.DataFrame, group_column: str) -> tuple[float, float]:
    """Run a one-way ANOVA for a categorical variable with two or more groups."""
    samples = grouped_samples(data, group_column)
    statistic, p_value = f_oneway(*(values for _, values in samples))
    labels = GROUP_COLUMNS[group_column]
    print(f"\nANOVA de una via: {TARGET_COLUMN} por {group_column}")
    print_group_summary(samples, labels)
    print(f"  F={statistic:.4f}, p={p_value:.6g} -> {interpret_result(p_value)}")
    return statistic, p_value


def run_t_test(data: pd.DataFrame, group_column: str) -> tuple[float, float]:
    """Run a two-sided Welch t-test for a categorical variable with two groups."""
    samples = grouped_samples(data, group_column)
    if len(samples) != 2:
        raise ValueError(f"La prueba t requiere exactamente dos grupos en '{group_column}'.")
    statistic, p_value = ttest_ind(
        samples[0][1],
        samples[1][1],
        equal_var=False,
        nan_policy="omit",
    )
    labels = GROUP_COLUMNS[group_column]
    print(f"\nPrueba t de Welch: {TARGET_COLUMN} por {group_column}")
    print_group_summary(samples, labels)
    print(f"  t={statistic:.4f}, p={p_value:.6g} -> {interpret_result(p_value)}")
    return statistic, p_value


def interpret_result(p_value: float) -> str:
    return "diferencia significativa" if p_value < ALPHA else "sin diferencia significativa"


def main() -> None:
    default_path = Path(__file__).resolve().parents[1] / "dataset.csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    data = load_dataset(csv_path)

    print("ANALISIS DE DIFERENCIAS ENTRE GRUPOS")
    print(f"Archivo: {csv_path}")
    print(f"Variable analizada: {TARGET_COLUMN} | alfa={ALPHA}")
    run_anova(data, "season")
    run_t_test(data, "yr")


if __name__ == "__main__":
    main()