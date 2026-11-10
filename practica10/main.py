"""Nube de palabras y analisis exploratorio de texto del Bike Sharing Dataset.

Las variables categoricas del dataset se convierten en texto descriptivo para
analizar su frecuencia y su relacion con el total de alquileres.

Ejecutar con: python practica10/main.py [ruta/al/dataset.csv] [directorio/salida]
"""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


SEASON_LABELS = {
    1: "primavera",
    2: "verano",
    3: "otono",
    4: "invierno",
}
WEATHER_LABELS = {
    1: "despejado",
    2: "nublado",
    3: "lluvia_ligera",
    4: "lluvia_intensa",
}
WEEKDAY_LABELS = {
    0: "domingo",
    1: "lunes",
    2: "martes",
    3: "miercoles",
    4: "jueves",
    5: "viernes",
    6: "sabado",
}
REQUIRED_COLUMNS = {"season", "yr", "weekday", "workingday", "weathersit", "cnt"}
STOP_WORDS = {"de", "dia", "fin", "semana"}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the categorical and target columns."""
    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"El CSV no contiene las columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El CSV esta vacio.")
    return data.dropna(subset=REQUIRED_COLUMNS).copy()


def create_text_data(data: pd.DataFrame) -> pd.DataFrame:
    """Translate coded categories into words used for the text analysis."""
    text_data = pd.DataFrame(index=data.index)
    text_data["temporada"] = data["season"].map(SEASON_LABELS).fillna("desconocida")
    text_data["clima"] = data["weathersit"].map(WEATHER_LABELS).fillna("desconocido")
    text_data["ano"] = data["yr"].map({0: "2011", 1: "2012"}).fillna("otro_ano")
    text_data["dia_semana"] = data["weekday"].map(WEEKDAY_LABELS).fillna("otro_dia")
    text_data["tipo_dia"] = data["workingday"].map(
        {0: "fin_de_semana", 1: "dia_laboral"}
    ).fillna("tipo_desconocido")
    text_data["cnt"] = data["cnt"]
    return text_data


def tokenize_text(text_data: pd.DataFrame) -> Counter[str]:
    text = " ".join(
        text_data[["temporada", "clima", "ano", "dia_semana", "tipo_dia"]]
        .astype(str)
        .replace("_", " ", regex=True)
        .to_numpy()
        .ravel()
    )
    tokens = re.findall(r"[a-záéíóúñ0-9]+", text.lower())
    return Counter(token for token in tokens if token not in STOP_WORDS)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Creada: {path}")


def create_word_cloud(frequencies: Counter[str], output_dir: Path) -> None:
    """Create a dependency-free word cloud using text sizes and positions."""
    words = frequencies.most_common(30)
    maximum = words[0][1]
    minimum = words[-1][1]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.axis("off")

    for index, (word, frequency) in enumerate(words):
        size = 14 + 42 * (frequency - minimum) / max(maximum - minimum, 1)
        column = index % 5
        row = index // 5
        x = 0.1 + column * 0.2
        y = 0.9 - row * 0.13
        axis.text(
            x,
            y,
            word,
            fontsize=size,
            color=plt.get_cmap("viridis")(index / max(len(words) - 1, 1)),
            ha="center",
            va="center",
            rotation=0 if index % 4 else 8,
            transform=axis.transAxes,
        )
    axis.set_title("Nube de palabras del dataset", fontsize=18, pad=20)
    save_figure(output_dir / "01_nube_de_palabras.png")


def create_frequency_plot(frequencies: Counter[str], output_dir: Path) -> None:
    top_words = pd.Series(dict(frequencies.most_common(15))).sort_values()
    plt.figure(figsize=(9, 6))
    top_words.plot(kind="barh", color="#4c78a8")
    plt.title("Palabras mas frecuentes")
    plt.xlabel("Frecuencia")
    plt.ylabel("Palabra")
    save_figure(output_dir / "02_frecuencia_de_palabras.png")


def create_category_plot(text_data: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        text_data.groupby("temporada", sort=False)["cnt"]
        .mean()
        .sort_values()
    )
    plt.figure(figsize=(8, 5))
    grouped.plot(kind="barh", color="#59a14f")
    plt.title("Promedio de alquileres por temporada")
    plt.xlabel("Promedio de alquileres")
    plt.ylabel("Temporada")
    save_figure(output_dir / "03_alquileres_por_temporada.png")


def print_analysis(frequencies: Counter[str], text_data: pd.DataFrame) -> None:
    print("ANALISIS EXPLORATORIO DE DATOS TEXTUALES")
    print(f"Registros analizados: {len(text_data)}")
    print(f"Tokens generados: {sum(frequencies.values())}")
    print("\nPalabras mas frecuentes:")
    for word, frequency in frequencies.most_common(10):
        print(f"  {word}: {frequency}")

    print("\nPromedio de alquileres por categoria textual:")
    for column in ["temporada", "clima", "tipo_dia"]:
        print(f"  {column}:")
        averages = text_data.groupby(column)["cnt"].mean().sort_values(ascending=False)
        for category, average in averages.items():
            print(f"    {category}: {average:.2f}")


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
    text_data = create_text_data(data)
    frequencies = tokenize_text(text_data)

    print_analysis(frequencies, text_data)
    create_word_cloud(frequencies, output_dir)
    create_frequency_plot(frequencies, output_dir)
    create_category_plot(text_data, output_dir)
    print(f"\nSe generaron las graficas en: {output_dir}")


if __name__ == "__main__":
    main()
