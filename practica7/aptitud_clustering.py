"""Evalua si el dataset presenta tendencia a formar clusters.

La prueba de Hopkins compara las distancias entre registros reales y puntos
aleatorios. Valores cercanos a 1 indican tendencia a formar agrupamientos,
mientras que valores cercanos a 0.5 sugieren una distribucion aleatoria.

Ejecutar con:
    python practica7/evaluar_aptitud_clustering.py [ruta/al/dataset.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
REPETITIONS = 10


def load_all_features(path: Path) -> pd.DataFrame:
    """Load every dataset variable, converting the date to a numeric value."""
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    features = data.copy()
    if "dteday" in features:
        parsed_dates = pd.to_datetime(features.pop("dteday"), errors="coerce")
        if parsed_dates.isna().any():
            raise ValueError("La columna dteday contiene fechas invalidas.")
        features["dteday"] = (
            parsed_dates - pd.Timestamp("1970-01-01")
        ).dt.total_seconds() / (24 * 60 * 60)

    non_numeric_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"No se pueden incluir variables no numericas: {non_numeric_columns}"
        )
    features = features.dropna()
    if len(features) < 3:
        raise ValueError("Se necesitan al menos tres registros completos.")
    return features


def hopkins_statistic(features: pd.DataFrame, sample_size: int, seed: int) -> float:
    """Calculate the Hopkins statistic for all standardized variables."""
    values = StandardScaler().fit_transform(features)
    sample_size = min(sample_size, len(values) - 1)
    generator = np.random.default_rng(seed)
    sample_indices = generator.choice(len(values), size=sample_size, replace=False)
    sampled_values = values[sample_indices]
    random_values = generator.uniform(
        low=values.min(axis=0),
        high=values.max(axis=0),
        size=(sample_size, values.shape[1]),
    )

    nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(values)
    real_distances = nearest_neighbors.kneighbors(sampled_values, n_neighbors=2)[0][:, 1]
    random_distances = nearest_neighbors.kneighbors(random_values, n_neighbors=1)[0][:, 0]
    real_distance_sum = real_distances.sum()
    random_distance_sum = random_distances.sum()
    denominator = real_distance_sum + random_distance_sum
    if denominator == 0:
        raise ValueError("No se pudo calcular Hopkins: todas las distancias son cero.")
    return float(random_distance_sum / denominator)


def interpret_hopkins(value: float) -> str:
    """Return a practical interpretation of the Hopkins statistic."""
    if value >= 0.75:
        return "El dataset es adecuado para clustering (tendencia fuerte)."
    if value >= 0.65:
        return "El dataset puede ser adecuado para clustering (tendencia moderada)."
    return "No se observa una tendencia clara a formar clusters."


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repository_root / "dataset.csv"
    data = load_all_features(csv_path)
    sample_size = max(2, min(len(data) // 10, 500))

    values = [
        hopkins_statistic(data, sample_size, RANDOM_STATE + repetition)
        for repetition in range(REPETITIONS)
    ]
    result = np.mean(values)
    deviation = np.std(values, ddof=1)

    print("PRUEBA DE APTITUD PARA CLUSTERING")
    print(f"Archivo: {csv_path}")
    print(f"Registros: {len(data)} | Variables utilizadas: {len(data.columns)}")
    print(f"Variables: {', '.join(data.columns)}")
    print(f"Muestra por repeticion: {sample_size}")
    print(f"Repeticiones: {REPETITIONS}")
    print(f"Estadistico de Hopkins: {result:.4f} +/- {deviation:.4f}")
    print(interpret_hopkins(result)) #pyright: ignore
    print("Referencia: valores cercanos a 0.5 indican aleatoriedad; valores cercanos a 1, agrupamiento.")


if __name__ == "__main__":
    main()
