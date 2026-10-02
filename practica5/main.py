"""Modelo lineal para predecir alquileres del Bike Sharing Dataset.

Ejecutar con: python practica5/main.py [ruta/al/dataset.csv] [directorio/salida]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "cnt"
FEATURE_COLUMNS = [
    "season",
    "yr",
    "hr",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the variables used by the linear model."""
    data = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    model_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    if model_data.empty:
        raise ValueError("No hay registros completos para entrenar el modelo.")
    return model_data


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def create_prediction_plot(actual: pd.Series, predicted: pd.Series, output_dir: Path) -> None:
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.25, s=14, color="#4c78a8")
    plt.plot([minimum, maximum], [minimum, maximum], "--", color="#e45756")
    plt.title("Valores observados y predichos")
    plt.xlabel("Alquileres observados")
    plt.ylabel("Alquileres predichos")
    save_figure(output_dir / "01_valores_observados_predichos.png")


def create_residual_plot(actual: pd.Series, predicted: pd.Series, output_dir: Path) -> None:
    residuals = actual - predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(predicted, residuals, alpha=0.25, s=14, color="#59a14f")
    plt.axhline(0, linestyle="--", color="#e45756")
    plt.title("Residuos del modelo lineal")
    plt.xlabel("Alquileres predichos")
    plt.ylabel("Residuo (observado - predicho)")
    save_figure(output_dir / "02_residuos_modelo.png")


def create_coefficient_plot(model: LinearRegression, output_dir: Path) -> None:
    coefficients = pd.Series(model.coef_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(9, 5))
    coefficients.plot(kind="barh", color="#f28e2b")
    plt.title("Coeficientes del modelo lineal")
    plt.xlabel("Coeficiente")
    plt.ylabel("Variable")
    save_figure(output_dir / "03_coeficientes_modelo.png")


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

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    model = LinearRegression()
    model.fit(train_features, train_target)
    predictions = pd.Series(model.predict(test_features), index=test_target.index)
    r2 = r2_score(test_target, predictions)
    rmse = mean_squared_error(test_target, predictions) ** 0.5

    print("MODELO LINEAL PARA PREDICCION DE ALQUILERES")
    print(f"Archivo: {csv_path}")
    print(f"Registros: {len(data)} | Variables predictoras: {len(FEATURE_COLUMNS)}")
    print(f"R^2 (prueba): {r2:.4f}")
    print(f"RMSE (prueba): {rmse:.4f}")
    print("Coeficientes:")
    for feature, coefficient in zip(FEATURE_COLUMNS, model.coef_):
        print(f"  {feature}: {coefficient:.4f}")

    create_prediction_plot(test_target, predictions, output_dir)
    create_residual_plot(test_target, predictions, output_dir)
    create_coefficient_plot(model, output_dir)
    print(f"Se generaron las graficas en: {output_dir}")


if __name__ == "__main__":
    main()
