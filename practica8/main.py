"""Modelo basico de series de tiempo para predecir alquileres.

Ejecutar con: python practica8/main.py [ruta/al/dataset.csv] [directorio/salida]
"""

import sys
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COLUMN = "cnt"
LAG_COLUMNS = [1, 24, 168]
FEATURE_COLUMNS = [
    "hr",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Cargar, ordenar y validar las variables que seran usadas en la serie de tiempo"""
    data = pd.read_csv(path, parse_dates=["dteday"])
    required_columns = {"dteday", *FEATURE_COLUMNS, TARGET_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)
    return data


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Crear calendario, lag y features ciclicas antes de hacer predicciones"""
    features = data[FEATURE_COLUMNS].copy()
    hour_angle = 2 * math.pi * data["hr"] / 24
    features["hour_sin"] = hour_angle.map(math.sin)
    features["hour_cos"] = hour_angle.map(math.cos)

    for lag in LAG_COLUMNS:
        features[f"cnt_lag_{lag}"] = data[TARGET_COLUMN].shift(lag)
    features["cnt_rolling_24"] = data[TARGET_COLUMN].shift(1).rolling(24).mean()
    features[TARGET_COLUMN] = data[TARGET_COLUMN]
    return features.dropna().reset_index(drop=True) # pyright: ignore


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def create_forecast_plot(
    actual: pd.Series,
    predicted: pd.Series,
    output_dir: Path,
) -> None:
    points = min(len(actual), 500)
    plt.figure(figsize=(11, 5))
    plt.plot(actual.iloc[:points].to_numpy(), label="Real", color="#4c78a8")
    plt.plot(predicted.iloc[:points].to_numpy(), label="Predicho", color="#e45756")
    plt.title("Pronostico de alquileres en el conjunto de prueba")
    plt.xlabel("Hora")
    plt.ylabel("Alquileres")
    plt.legend()
    save_figure(output_dir / "01_pronostico_series_tiempo.png")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repository_root / "dataset.csv"
    output_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path(__file__).resolve().parent / "graficas"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(csv_path)
    model_data = create_time_features(data)
    split_index = int(len(model_data) * 0.8)
    train_data = model_data.iloc[:split_index]
    test_data = model_data.iloc[split_index:]
    predictor_columns = [column for column in model_data.columns if column != TARGET_COLUMN]

    model = LinearRegression()
    model.fit(train_data[predictor_columns], train_data[TARGET_COLUMN])
    predictions = pd.Series(
        model.predict(test_data[predictor_columns]),
        index=test_data.index,
    )
    actual = test_data[TARGET_COLUMN]
    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions) ** 0.5
    r2 = r2_score(actual, predictions)

    print("MODELO BASICO DE SERIES DE TIEMPO")
    print(f"Archivo: {csv_path}")
    print(f"Registros utilizables: {len(model_data)}")
    print(f"Entrenamiento: {len(train_data)} | Prueba: {len(test_data)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print("Variables temporales utilizadas:")
    print(f"  Variables contextuales: {', '.join(FEATURE_COLUMNS)}")
    print(f"  Rezagos: {', '.join(str(lag) for lag in LAG_COLUMNS)} horas")
    print("  Promedio movil: 24 horas anteriores")

    create_forecast_plot(actual, predictions, output_dir)
    print(f"Se genero la grafica en: {output_dir}")


if __name__ == "__main__":
    main()
