"""Modelo de agrupamiento K-Means para perfiles meteorologicos.

Ejecutar con: python practica7/main.py [ruta/al/dataset.csv] [directorio/salida]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
K_VALUES = range(2, 9)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the variables used by the clustering model."""
    data = pd.read_csv(path)
    missing_columns = set(FEATURE_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")

    model_data = data[FEATURE_COLUMNS].dropna()
    if len(model_data) < max(K_VALUES):
        raise ValueError("El CSV no contiene suficientes registros para evaluar los clusters.")
    return model_data #pyright: ignore


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def evaluate_cluster_counts(features: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    results = []
    for cluster_count in K_VALUES:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10) #pyright: ignore
        clusters = model.fit_predict(scaled_features)
        results.append(
            {
                "k": cluster_count,
                "inercia": model.inertia_,
                "silhouette": silhouette_score(scaled_features, clusters),
            }
        )
    return pd.DataFrame(results)


def create_evaluation_plot(results: pd.DataFrame, output_dir: Path) -> None:
    _, inertia_axis = plt.subplots(figsize=(8, 5))
    silhouette_axis = inertia_axis.twinx()
    inertia_axis.plot(results["k"], results["inercia"], marker="o", color="#4c78a8")
    silhouette_axis.plot(
        results["k"],
        results["silhouette"],
        marker="s",
        color="#e45756",
    )
    inertia_axis.set_xlabel("Numero de clusters (k)")
    inertia_axis.set_ylabel("Inercia", color="#4c78a8")
    silhouette_axis.set_ylabel("Coeficiente silhouette", color="#e45756")
    inertia_axis.set_title("Evaluacion de diferentes cantidades de clusters")
    save_figure(output_dir / "01_evaluacion_clusters.png")


def create_cluster_plot(
    scaled_features,
    clusters,
    cluster_count: int,
    output_dir: Path,
) -> None:
    components = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=clusters,
        cmap="viridis",
        alpha=0.35,
        s=12,
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Clusters K-Means proyectados con PCA (k={cluster_count})")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    save_figure(output_dir / "02_clusters_pca.png")


def create_pairwise_plot(data: pd.DataFrame, clusters, output_dir: Path) -> None:
    figure, axes = plt.subplots(
        len(FEATURE_COLUMNS),
        len(FEATURE_COLUMNS),
        figsize=(12, 12),
        sharex="col",
        sharey="row",
    )
    colors = plt.get_cmap("viridis")(clusters / max(clusters.max(), 1))
    for row, row_column in enumerate(FEATURE_COLUMNS):
        for column, column_name in enumerate(FEATURE_COLUMNS):
            axis = axes[row, column]
            if row == column:
                axis.hist(data[row_column], bins=25, color="#4c78a8", alpha=0.8)
            else:
                axis.scatter(
                    data[column_name],
                    data[row_column],
                    c=colors,
                    alpha=0.15,
                    s=5,
                )
            if row == len(FEATURE_COLUMNS) - 1:
                axis.set_xlabel(column_name)
            if column == 0:
                axis.set_ylabel(row_column)
    figure.suptitle("Relaciones entre variables coloreadas por cluster", y=1.02)
    save_figure(output_dir / "03_matriz_de_dispersion.png")


def create_cluster_profile_plot(
    data: pd.DataFrame,
    clusters,
    output_dir: Path,
) -> None:
    scaled_data = pd.DataFrame(
        StandardScaler().fit_transform(data),
        columns=FEATURE_COLUMNS,
    ).assign(cluster=clusters)
    profiles = scaled_data.groupby("cluster")[FEATURE_COLUMNS].mean()
    figure, axis = plt.subplots(figsize=(9, 4))
    image = axis.imshow(profiles, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    figure.colorbar(image, ax=axis, label="Media estandarizada")
    axis.set_xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=35, ha="right")
    axis.set_yticks(range(len(profiles)), [f"Cluster {cluster}" for cluster in profiles.index])
    axis.set_title("Perfil meteorologico promedio por cluster")
    for row in range(len(profiles)):
        for column in range(len(FEATURE_COLUMNS)):
            axis.text(column, row, f"{profiles.iloc[row, column]:.2f}", ha="center", va="center")
    save_figure(output_dir / "04_perfiles_de_clusters.png")


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

    results = evaluate_cluster_counts(data)
    best_result = results.loc[results["silhouette"].idxmax()]
    cluster_count = int(best_result["k"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10) #pyright: ignore
    clusters = model.fit_predict(scaled_features)
    data_with_clusters = data.assign(cluster=clusters)
    cluster_sizes = data_with_clusters["cluster"].value_counts().sort_index()

    print("MODELO DE AGRUPAMIENTO K-MEANS")
    print(f"Archivo: {csv_path}")
    print(f"Registros: {len(data)} | Variables utilizadas: {len(FEATURE_COLUMNS)}")
    print(f"Numero de clusters seleccionado: {cluster_count}")
    print(f"Inercia: {model.inertia_:.4f}")
    print(f"Coeficiente silhouette: {best_result['silhouette']:.4f}")
    print("Registros por cluster:")
    for cluster, size in cluster_sizes.items():
        print(f"  Cluster {cluster}: {size}")

    create_evaluation_plot(results, output_dir)
    create_cluster_plot(scaled_features, clusters, cluster_count, output_dir)
    create_pairwise_plot(data, clusters, output_dir)
    create_cluster_profile_plot(data, clusters, output_dir)
    print(f"Se generaron las graficas en: {output_dir}")


if __name__ == "__main__":
    main()
