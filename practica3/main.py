"""Generacion automatizada de graficas para el Bike Sharing Dataset.

Ejecutar desde la raiz del repositorio:

    python practica3/main.py

Las imagenes se guardan en practica3/graficas. Tambien se puede indicar otro
CSV como primer argumento y otro directorio de salida como segundo argumento.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SEASON_LABELS = {1: "Primavera", 2: "Verano", 3: "Otono", 4: "Invierno"}


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["dteday"])
    required_columns = {"dteday", "season", "mnth", "hr", "temp", "hum", "cnt"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")
    return data


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def create_pie_chart(data: pd.DataFrame, output_dir: Path) -> None:
    rentals = data.groupby("season")["cnt"].sum().sort_index()
    labels = [SEASON_LABELS.get(season, str(season)) for season in rentals.index]
    plt.figure(figsize=(8, 6))
    plt.pie(rentals, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Proporcion de alquileres por temporada")
    save_figure(output_dir / "01_alquileres_por_temporada_pie.png")


def create_histograms(data: pd.DataFrame, output_dir: Path) -> None:
    # La misma rutina genera histogramas para varias variables.
    histogram_columns = {
        "cnt": "Total de alquileres",
        "temp": "Temperatura normalizada",
        "hum": "Humedad normalizada",
    }
    for column, xlabel in histogram_columns.items():
        plt.figure(figsize=(8, 5))
        plt.hist(data[column], bins=25, color="#4c78a8", edgecolor="white")
        plt.title(f"Distribucion de {column}")
        plt.xlabel(xlabel)
        plt.ylabel("Frecuencia")
        save_figure(output_dir / f"02_histograma_{column}.png")


def create_box_plots(data: pd.DataFrame, output_dir: Path) -> None:
    seasons = sorted(data["season"].unique())
    rentals = [data.loc[data["season"] == season, "cnt"] for season in seasons]
    labels = [SEASON_LABELS.get(season, str(season)) for season in seasons]
    plt.figure(figsize=(8, 6))
    plt.boxplot(rentals, tick_labels=labels, patch_artist=True)
    plt.title("Dispersion de alquileres por temporada")
    plt.xlabel("Temporada")
    plt.ylabel("Total de alquileres")
    save_figure(output_dir / "03_boxplot_alquileres_por_temporada.png")


def create_scatter_plot(data: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(data["temp"], data["cnt"], alpha=0.2, s=12, color="#e45756")
    plt.title("Relacion entre temperatura y alquileres")
    plt.xlabel("Temperatura normalizada")
    plt.ylabel("Total de alquileres")
    save_figure(output_dir / "04_dispersion_temperatura_alquileres.png")


def create_line_chart(data: pd.DataFrame, output_dir: Path) -> None:
    monthly_rentals = data.groupby("mnth")["cnt"].mean()
    plt.figure(figsize=(9, 5))
    plt.plot(monthly_rentals.index, monthly_rentals.values, marker="o", color="#59a14f")
    plt.title("Promedio de alquileres por mes")
    plt.xlabel("Mes")
    plt.ylabel("Promedio de alquileres por hora")
    plt.xticks(range(1, 13))
    plt.grid(alpha=0.25)
    save_figure(output_dir / "05_linea_promedio_mensual.png")


def create_bar_chart(data: pd.DataFrame, output_dir: Path) -> None:
    hourly_rentals = data.groupby("hr")["cnt"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(hourly_rentals.index, hourly_rentals.values, color="#f28e2b")
    plt.title("Promedio de alquileres por hora")
    plt.xlabel("Hora del dia")
    plt.ylabel("Promedio de alquileres")
    plt.xticks(range(24))
    save_figure(output_dir / "06_barras_promedio_por_hora.png")


def create_correlation_heatmap(data: pd.DataFrame, output_dir: Path) -> None:
    columns = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    correlations = data[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for row in range(len(columns)):
        for column in range(len(columns)):
            ax.text(column, row, f"{correlations.iloc[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Correlacion de Pearson")
    ax.set_title("Matriz de correlaciones")
    save_figure(output_dir / "07_mapa_de_calor_correlaciones.png")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repository_root / "dataset.csv"
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "graficas"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(csv_path)
    print(f"Generando graficas para {len(data)} registros...")

    # Cada funcion representa un tipo de grafica y se ejecuta automaticamente.
    chart_generators = [
        create_pie_chart,
        create_histograms,
        create_box_plots,
        create_scatter_plot,
        create_line_chart,
        create_bar_chart,
        create_correlation_heatmap,
    ]
    for chart_generator in chart_generators:
        chart_generator(data, output_dir)
    print(f"Se generaron las graficas en: {output_dir}")


if __name__ == "__main__":
    main()