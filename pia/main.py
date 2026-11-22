"""Genera el mockup analitico del PIA.

Uso:
    python pia/main.py

El script crea las graficas y los documentos de presentacion dentro de pia/.
La serie de tiempo estima la cantidad de rentas. HistGradientBoostingClassifier
evalua si el residual del pronostico es subestimado, adecuado o sobreestimado.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
PIA_DIR = Path(__file__).resolve().parent
GRAPH_DIR = PIA_DIR / "graficas"
DATASET = ROOT / "dataset.csv"

TARGET = "cnt"
TIME_CONTEXT = ["hr", "workingday", "weathersit", "temp", "hum", "windspeed"]
CLASSIFIER_FEATURES = [
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
]
LAGS = [1, 24, 168]
CLASS_NAMES = {
    0: "Subestimado",
    1: "Adecuado",
    2: "Sobreestimado",
}
COLORS = {
    "Subestimado": "#e15759",
    "Adecuado": "#59a14f",
    "Sobreestimado": "#4c78a8",
}


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATASET, parse_dates=["dteday"])
    required = set(CLASSIFIER_FEATURES + TIME_CONTEXT + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if data.empty:
        raise ValueError("El dataset esta vacio.")
    data = data.sort_values(["dteday", "hr"]).reset_index(drop=True)
    data["timestamp"] = data["dteday"] + pd.to_timedelta(data["hr"], unit="h")
    return data


def build_time_data(data: pd.DataFrame) -> pd.DataFrame:
    features = data[TIME_CONTEXT].copy()
    hour_angle = 2 * math.pi * data["hr"] / 24
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    for lag in LAGS:
        features[f"cnt_lag_{lag}"] = data[TARGET].shift(lag)
    features["cnt_rolling_24"] = data[TARGET].shift(1).rolling(24).mean()
    features[TARGET] = data[TARGET]
    features["timestamp"] = data["timestamp"]
    return features.dropna().reset_index(drop=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def train_models(data: pd.DataFrame) -> dict:
    time_data = build_time_data(data)
    split = int(len(time_data) * 0.8)
    train_time = time_data.iloc[:split].copy()
    test_time = time_data.iloc[split:].copy()
    time_predictors = [column for column in time_data.columns if column not in {TARGET, "timestamp"}]

    time_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    time_model.fit(train_time[time_predictors], train_time[TARGET])
    time_prediction = pd.Series(
        time_model.predict(test_time[time_predictors]),
        index=test_time.index,
    )
    time_actual = test_time[TARGET]
    time_metrics = {
        "MAE": mean_absolute_error(time_actual, time_prediction),
        "RMSE": mean_squared_error(time_actual, time_prediction) ** 0.5,
        "R2": r2_score(time_actual, time_prediction),
    }

    # Train a residual classifier using only out-of-sample residuals.
    inner_split = int(len(train_time) * 0.8)
    inner_train = train_time.iloc[:inner_split].copy()
    inner_validation = train_time.iloc[inner_split:].copy()

    inner_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    inner_model.fit(
        inner_train[time_predictors],
        inner_train[TARGET],
    )
    inner_prediction = inner_model.predict(inner_validation[time_predictors])
    inner_residual = (
        inner_validation[TARGET].to_numpy() - inner_prediction
    )
    residual_mae = float(np.mean(np.abs(inner_residual)))

    def residual_labels(residual: np.ndarray) -> np.ndarray:
        return np.select(
            [residual > residual_mae, residual < -residual_mae],
            [0, 2],
            default=1,
        ).astype(int)

    train_labels = residual_labels(inner_residual)
    classifier_features = time_predictors + ["forecast"]
    train_classifier = pd.DataFrame(
        inner_validation[time_predictors].to_numpy(),
        index=inner_validation.index,
        columns=time_predictors,
    )
    train_classifier["forecast"] = inner_prediction

    test_classifier = pd.DataFrame(
        test_time[time_predictors].to_numpy(),
        index=test_time.index,
        columns=time_predictors,
    )
    test_classifier["forecast"] = time_prediction.to_numpy()

    test_residual = time_actual.to_numpy() - time_prediction.to_numpy()
    test_labels = residual_labels(test_residual)

    class_counts = np.bincount(train_labels, minlength=3)
    class_weights = {
        label: len(train_labels) / (3 * class_counts[label])
        for label in range(3)
    }
    classifier_sample_weight = np.array([
        class_weights[label]
        for label in train_labels
    ])

    classifier_model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    classifier_model.fit(
        train_classifier[classifier_features],
        train_labels,
        sample_weight=classifier_sample_weight,
    )
    classifier_prediction = pd.Series(
        classifier_model.predict(test_classifier[classifier_features]),
        index=test_classifier.index,
    )
    classifier_probabilities = pd.DataFrame(
        classifier_model.predict_proba(test_classifier[classifier_features]),
        index=test_classifier.index,
        columns=classifier_model.classes_,
    )
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(train_classifier[classifier_features], train_labels)
    dummy_prediction = dummy_model.predict(test_classifier[classifier_features])
    classifier_metrics = {
        "accuracy": accuracy_score(test_labels, classifier_prediction),
        "f1_macro": f1_score(test_labels, classifier_prediction, average="macro"),
        "dummy_accuracy": accuracy_score(test_labels, dummy_prediction),
        "dummy_f1_macro": f1_score(test_labels, dummy_prediction, average="macro"),
    }

    return {
        "data": data,
        "time_data": time_data,
        "train_time": train_time,
        "test_time": test_time,
        "time_prediction": time_prediction,
        "time_actual": time_actual,
        "time_metrics": time_metrics,
        "time_predictors": time_predictors,
        "train_classifier": train_classifier,
        "test_classifier": test_classifier,
        "test_labels": test_labels,
        "test_residual": test_residual,
        "residual_mae": residual_mae,
        "residual_mae": residual_mae,
        "classifier_prediction": classifier_prediction,
        "classifier_probabilities": classifier_probabilities,
        "classifier_model": classifier_model,
        "classifier_metrics": classifier_metrics,
    }


def create_forecast_plot(result: dict) -> None:
    test = result["test_time"]
    actual = result["time_actual"]
    prediction = result["time_prediction"]
    plot_data = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual": actual.to_numpy(),
            "prediction": prediction.to_numpy(),
        }
    )
    plot_data = plot_data[
        (plot_data["timestamp"] >= pd.Timestamp("2012-08-13"))
        & (plot_data["timestamp"] < pd.Timestamp("2012-08-20"))
    ].reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    plt.plot(
        plot_data["timestamp"],
        plot_data["actual"],
        label="Real",
        color="#4c78a8",
    )
    plt.plot(
        plot_data["timestamp"],
        plot_data["prediction"],
        label="Pronosticado",
        color="#e15759",
    )
    plt.title("Pronóstico de rentas con HistGradientBoosting")
    plt.xlabel("Fecha")
    plt.ylabel("Rentas por hora")
    plt.legend()
    plt.xticks(rotation=25)
    save_figure(GRAPH_DIR / "01_pronostico_temporal.png")


def create_threshold_plot(result: dict) -> None:
    residual = result["test_residual"]
    mae = result["residual_mae"]
    plt.figure(figsize=(9, 5))
    plt.hist(residual, bins=40, color="#4c78a8", alpha=0.85)
    plt.axvspan(-mae, mae, color="#59a14f", alpha=0.18, label="Residual adecuado")
    plt.axvline(-mae, color="#4c78a8", linestyle="--", linewidth=1)
    plt.axvline(mae, color="#e15759", linestyle="--", linewidth=1)
    plt.axvline(0, color="#333333", linestyle="-", linewidth=0.8)
    plt.title("Distribución del error de pronóstico")
    plt.xlabel("Residual real - predicción")
    plt.ylabel("Registros")
    plt.legend()
    save_figure(GRAPH_DIR / "02_distribucion_residual.png")


def create_confusion_plot(result: dict) -> None:
    labels = [0, 1, 2]
    matrix = confusion_matrix(result["test_labels"], result["classifier_prediction"], labels=labels)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Registros")
    display = [CLASS_NAMES[label] for label in labels]
    axis.set_xticks(range(3), display)
    axis.set_yticks(range(3), display)
    axis.set_xlabel("Nivel predicho")
    axis.set_ylabel("Nivel real")
    axis.set_title("HistGradientBoosting: matriz de confusión del error")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    save_figure(GRAPH_DIR / "03_matriz_hist_gradient_boosting.png")


def create_importance_plot(result: dict) -> None:
    model = result["classifier_model"]
    test_classifier = result["test_classifier"]
    importance_result = permutation_importance(
        model,
        test_classifier,
        result["test_labels"],
        n_repeats=5,
        random_state=42,
        scoring="f1_macro",
        n_jobs=-1,
    )
    importance = pd.Series(
        importance_result.importances_mean,
        index=test_classifier.columns,
    ).sort_values()
    plt.figure(figsize=(9, 6))
    importance.plot(kind="barh", color="#4c78a8")
    plt.title("Importancia por permutación: HistGradientBoosting")
    plt.xlabel("Disminución promedio del F1 macro")
    save_figure(GRAPH_DIR / "04_importancia_hist_gradient_boosting.png")


def create_action_plot(result: dict) -> None:
    order = ["Subestimado", "Adecuado", "Sobreestimado"]
    counts = pd.Series(result["test_labels"]).map(CLASS_NAMES).value_counts()
    counts = counts.reindex(order, fill_value=0)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=["#e15759", "#59a14f", "#4c78a8"],
    )
    plt.title("Tipos de error observados en el periodo de prueba")
    plt.ylabel("Horas")
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 20, f"{count:,}", ha="center")
    save_figure(GRAPH_DIR / "05_tipos_error_observados.png")


def create_hour_profile_plot(result: dict) -> None:
    data = result["data"]
    profiles = data.groupby("hr")[TARGET].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(profiles.index, profiles.values, marker="o", color="#4c78a8")
    plt.axvspan(7.5, 9, color="#f28e2b", alpha=0.18, label="Pico matutino")
    plt.axvspan(16.5, 18.5, color="#e15759", alpha=0.16, label="Pico vespertino")
    plt.title("Demanda promedio por hora")
    plt.xlabel("Hora del día")
    plt.ylabel("Rentas promedio")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    save_figure(GRAPH_DIR / "06_demanda_por_hora.png")


def create_four_week_forecast_plot(result: dict) -> None:
    test = result["test_time"]
    actual = result["time_actual"]
    prediction = result["time_prediction"]
    plot_data = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual": actual.to_numpy(),
            "prediction": prediction.to_numpy(),
        }
    )
    start = pd.Timestamp("2012-12-03")
    end = pd.Timestamp("2012-12-31")
    plot_data = plot_data[
        (plot_data["timestamp"] >= start)
        & (plot_data["timestamp"] < end)
    ].reset_index(drop=True)

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharey=True)
    week_starts = [start + pd.Timedelta(days=7 * week) for week in range(4)]

    for index, (axis, week_start) in enumerate(zip(axes, week_starts)):
        week_end = week_start + pd.Timedelta(days=7)
        week = plot_data[
            (plot_data["timestamp"] >= week_start)
            & (plot_data["timestamp"] < week_end)
        ]
        axis.plot(
            week["timestamp"],
            week["actual"],
            label="Real",
            color="#4c78a8",
            linewidth=1.2,
        )
        axis.plot(
            week["timestamp"],
            week["prediction"],
            label="Pronosticado",
            color="#e15759",
            linewidth=1.2,
        )
        axis.set_title(
            f"Semana {index + 1}: "
            f"{week_start:%d %b}–{week_end - pd.Timedelta(hours=1):%d %b %Y}"
        )
        axis.set_ylabel("Rentas/h")
        axis.grid(alpha=0.15)
        if index == 0:
            axis.legend(loc="upper right")
        axis.set_xlim(week_start, week_end)
        axis.set_xticks(
            pd.date_range(
                week_start,
                week_end - pd.Timedelta(hours=1),
                freq="24h",
            )
        )
        axis.tick_params(axis="x", rotation=25)

    axes[-1].set_xlabel("Fecha")
    figure.suptitle(
        "Pronóstico semanal: 3 al 30 de diciembre de 2012",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()
    save_figure(GRAPH_DIR / "07_pronostico_4_semanas_diciembre.png")


def create_weekly_test_forecast_plots(result: dict) -> None:
    test = result["test_time"]
    actual = result["time_actual"]
    prediction = result["time_prediction"]
    plot_data = pd.DataFrame(
        {
            "timestamp": test["timestamp"].to_numpy(),
            "actual": actual.to_numpy(),
            "prediction": prediction.to_numpy(),
        }
    )

    first_timestamp = plot_data["timestamp"].min()
    last_timestamp = plot_data["timestamp"].max()
    first_week = first_timestamp.normalize()
    week_count = int(np.ceil((last_timestamp - first_week) / pd.Timedelta(days=7)))
    week_starts = [
        first_week + pd.Timedelta(days=7 * week)
        for week in range(week_count)
    ]
    weekly_dir = GRAPH_DIR / "prueba_por_semana"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    for index, week_start in enumerate(week_starts, start=1):
        week_end = week_start + pd.Timedelta(days=7)
        week = plot_data[
            (plot_data["timestamp"] >= week_start)
            & (plot_data["timestamp"] < week_end)
        ]
        if week.empty:
            continue

        figure, axis = plt.subplots(figsize=(12, 5))
        axis.plot(
            week["timestamp"],
            week["actual"],
            label="Real",
            color="#4c78a8",
            linewidth=1.2,
        )
        axis.plot(
            week["timestamp"],
            week["prediction"],
            label="Pronosticado",
            color="#e15759",
            linewidth=1.2,
        )
        axis.set_title(
            f"Semana {index}: "
            f"{week_start:%d %b %Y}–"
            f"{week_end - pd.Timedelta(hours=1):%d %b %Y}"
        )
        axis.set_xlabel("Fecha")
        axis.set_ylabel("Rentas por hora")
        axis.grid(alpha=0.15)
        axis.legend(loc="upper right")
        axis.set_xlim(week_start, week_end)
        axis.set_xticks(
            pd.date_range(
                week_start,
                week_end - pd.Timedelta(hours=1),
                freq="24h",
            )
        )
        axis.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        save_figure(
            weekly_dir
            / f"{index:02d}_pronostico_semana_{week_start:%Y-%m-%d}.png"
        )


def write_metrics(result: dict) -> None:
    metrics = pd.DataFrame(
        [
            {"modelo": "Serie de tiempo", "metrica": "MAE", "valor": result["time_metrics"]["MAE"]},
            {"modelo": "Serie de tiempo", "metrica": "RMSE", "valor": result["time_metrics"]["RMSE"]},
            {"modelo": "Serie de tiempo", "metrica": "R2", "valor": result["time_metrics"]["R2"]},
            {"modelo": "Clasificador de residuales", "metrica": "MAE usado como umbral", "valor": result["residual_mae"]},
            {"modelo": "HistGradientBoostingClassifier", "metrica": "Accuracy", "valor": result["classifier_metrics"]["accuracy"]},
            {"modelo": "HistGradientBoostingClassifier", "metrica": "F1 macro", "valor": result["classifier_metrics"]["f1_macro"]},
            {"modelo": "Dummy baseline", "metrica": "Accuracy", "valor": result["classifier_metrics"]["dummy_accuracy"]},
        ]
    )
    metrics.to_csv(PIA_DIR / "metricas.csv", index=False)
    report = classification_report(
        result["test_labels"],
        result["classifier_prediction"],
        labels=[0, 1, 2],
        target_names=[CLASS_NAMES[i] for i in [0, 1, 2]],
        zero_division=0,
    )
    (PIA_DIR / "reporte_clasificacion.txt").write_text(report, encoding="utf-8")


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    result = train_models(load_data())
    create_forecast_plot(result)
    create_threshold_plot(result)
    create_confusion_plot(result)
    create_importance_plot(result)
    create_action_plot(result)
    create_hour_profile_plot(result)
    create_four_week_forecast_plot(result)
    create_weekly_test_forecast_plots(result)
    write_metrics(result)

    print("Mockup generado en:", PIA_DIR)
    print(f"Serie de tiempo | MAE={result['time_metrics']['MAE']:.4f} | "
          f"RMSE={result['time_metrics']['RMSE']:.4f} | "
          f"R2={result['time_metrics']['R2']:.4f}")
    print(f"HistGradientBoostingClassifier | accuracy={result['classifier_metrics']['accuracy']:.4f} | "
          f"F1 macro={result['classifier_metrics']['f1_macro']:.4f}")
    print("Graficas:", len(list(GRAPH_DIR.glob("*.png"))))


if __name__ == "__main__":
    main()
