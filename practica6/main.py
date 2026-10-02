"""Modelo de clasificacion KNN para el Bike Sharing Dataset.

El modelo predice la situacion meteorologica (``weathersit``) a partir de
variables temporales y ambientales.
Ejecutar con: python practica6/main.py [ruta/al/dataset.csv] [directorio/salida]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "weathersit"
FEATURE_COLUMNS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
WEATHER_LABELS = {
    1: "Despejado",
    2: "Nublado",
    3: "Lluvia/nieve ligera",
    4: "Lluvia intensa",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the variables used by the classifier."""
    data = pd.read_csv(path)
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    model_data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    if model_data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("La variable objetivo debe contener al menos dos clases.")
    return model_data


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def create_confusion_matrix_plot(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[int],
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=labels)
    display_labels = [WEATHER_LABELS.get(label, str(label)) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Cantidad de registros")
    ax.set_xticks(range(len(labels)), display_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), display_labels)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusion del modelo KNN")
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, matrix[row, column], ha="center", va="center")
    save_figure(output_dir / "01_matriz_de_confusion.png")


def create_metrics_plot(metrics: dict[str, float], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#4c78a8", "#59a14f", "#f28e2b"])
    plt.ylim(0, 1)
    plt.title("Metricas de evaluacion del modelo KNN")
    plt.ylabel("Puntuacion")
    for bar, value in zip(bars, metrics.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    save_figure(output_dir / "02_metricas_clasificacion.png")


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
    target = data[TARGET_COLUMN].astype(int)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(scaled_train_features, train_target)
    predictions = pd.Series(model.predict(scaled_test_features), index=test_target.index)

    labels = sorted(target.unique())
    metrics = {
        "Exactitud": accuracy_score(test_target, predictions),
        "Precision": precision_score(test_target, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(test_target, predictions, average="weighted", zero_division=0),
        "F1": f1_score(test_target, predictions, average="weighted", zero_division=0),
    }
    report = classification_report(
        test_target,
        predictions,
        labels=labels,
        target_names=[WEATHER_LABELS.get(label, str(label)) for label in labels],
        zero_division=0,
    )

    print("MODELO DE CLASIFICACION KNN")
    print(f"Archivo: {csv_path}")
    print(f"Registros: {len(data)} | Variables predictoras: {len(FEATURE_COLUMNS)}")
    print(f"Vecinos utilizados: {model.n_neighbors}")
    print("\nMetricas globales:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nReporte de clasificacion:")
    print(report)

    create_confusion_matrix_plot(test_target, predictions, labels, output_dir)
    create_metrics_plot(metrics, output_dir)
    print(f"Se generaron las graficas en: {output_dir}")


if __name__ == "__main__":
    main()