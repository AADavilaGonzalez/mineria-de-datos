"""
Analysis descriptivo del dataset de Bike Sharing.
Ejecutar con: python practica2/main.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import r_regression

NUMERIC_COLUMNS = [
    "instant",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
]

CATEGORICAL_COLUMNS = ["season", "yr", "mnth", "hr", "weekday", "weathersit"]

LABELS = {
    "season": {1: "spring", 2: "summer", 3: "fall", 4: "winter"},
    "yr": {0: "2011", 1: "2012"},
    "weathersit": {
        1: "clear/partly cloudy",
        2: "mist/cloudy",
        3: "light rain/snow",
        4: "heavy rain/snow",
    },
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the columns documented in dataset-readme.txt."""
    data = pd.read_csv(path)
    missing_columns = set(NUMERIC_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"The CSV is missing documented columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("The CSV is empty.")
    data["dteday"] = pd.to_datetime(data["dteday"])
    return data


def format_number(value: float) -> str:
    return f"{value:.4f}" if value % 1 else str(int(value))


def format_modes(series: pd.Series) -> str:
    modes = series.mode()
    if len(modes) == len(series):
        return "sin moda unica"
    if len(modes) > 5:
        return f"{len(modes)} modas"
    return ", ".join(format_number(value) for value in modes)


def print_numeric_statistics(data: pd.DataFrame) -> None:
    print("\n1. Estadistica descriptiva de variables numericas")
    print(
        f"{'Variable':<12} {'n':>6} {'media':>12} {'mediana':>12} "
        f"{'moda':>16} {'min':>12} {'max':>12} {'desv.std':>12}"
    )
    summary = data[NUMERIC_COLUMNS].describe().T
    for column, row in summary.iterrows():
        print(
            f"{column:<12} {int(row['count']):>6} {format_number(row['mean']):>12} " # pyright: ignore
            f"{format_number(data[column].median()):>12} {format_modes(data[column]):>16} " # pyright: ignore
            f"{format_number(row['min']):>12} {format_number(row['max']):>12} " # pyright: ignore
            f"{format_number(row['std']):>12}" # pyright: ignore
        )


def print_categorical_distributions(data: pd.DataFrame) -> None:
    print("\n2. Distribucion de entidades categoricas")
    for column in CATEGORICAL_COLUMNS:
        counts = data[column].value_counts().sort_index()
        description = ", ".join(
            f"{LABELS.get(column, {}).get(value, value)}={count}" # pyright: ignore
            for value, count in counts.items()
        )
        print(f"- {column}: {description}")


def print_relationships(data: pd.DataFrame) -> None:
    print("\n3. Entidades y relaciones identificadas")
    print("- Registro de alquiler: instant, dteday y hr identifican cada observacion horaria.")
    print("- Calendario: dteday se relaciona con season, yr, mnth, weekday, holiday y workingday.")
    print("- Clima: season y weathersit describen el contexto ambiental junto con temp, atemp, hum y windspeed.")
    print("- Usuarios: casual y registered son tipos de usuario; cnt = casual + registered.")

    user_count_error = (data["cnt"] - data["casual"] - data["registered"]).abs().max()
    print(f"- Integridad de usuarios: maximo error cnt-(casual+registered) = {int(user_count_error)}.")

    correlation_columns = ["temp", "atemp", "hum", "windspeed", "hr", "workingday", "weathersit"]
    correlations = pd.Series(
        r_regression(data[correlation_columns], data["cnt"]),
        index=correlation_columns,
    ).sort_values(key=abs, ascending=False)
    print("- Correlaciones de Pearson mas fuertes con cnt (no implican causalidad):")
    for column, correlation in correlations.head(3).items():
        print(f"  {column} vs cnt: r={correlation:.4f}")


def main() -> None:
    default_path = Path(__file__).resolve().parents[1] / "dataset.csv"
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    data = load_dataset(csv_path)

    print("ANALISIS DEL DATASET BIKE SHARING")
    print(f"Archivo: {csv_path}")
    print(f"Registros: {len(data)} | Variables: {len(data.columns)}")
    print(f"Periodo: {data['dteday'].min().date()} a {data['dteday'].max().date()}")
    print_numeric_statistics(data)
    print_categorical_distributions(data)
    print_relationships(data)


if __name__ == "__main__":
    main()
