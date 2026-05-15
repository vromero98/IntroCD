"""Funciones y constantes compartidas entre la notebook y los scripts.

Este módulo centraliza la lógica común utilizada tanto por
``tarea1.ipynb`` como por ``reproducir_informe.py``:

    - Carga (y descarga si es necesario) del dataset.
    - Normalización del texto de los artículos.
    - Constantes visuales: orden canónico de medios, paleta de colores,
      etiquetas, stopwords y parámetros por defecto.
    - Funciones reutilizables para visualizar palabras más frecuentes y
      nubes de palabras por medio.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

# ── Constantes visuales ──────────────────────────────────────────────────────
# Orden canónico de los medios usado en todas las visualizaciones para
# facilitar la comparación entre figuras.
ORDER = ['NY Times', 'Reuters', 'The Hill', 'CNBC', 'People']

# Paleta viridis asignada según el orden canónico. Cada medio mantiene su
# color en todas las figuras del informe.
PALETTE = dict(zip(ORDER, sns.color_palette('viridis', len(ORDER))))

# Mapeo entre etiqueta corta (para gráficos) y nombre completo del medio
# tal como aparece en la columna ``publication`` del dataset.
LABEL_TO_PUB = {
    'NY Times': 'The New York Times',
    'Reuters':  'Reuters',
    'The Hill': 'The Hill',
    'CNBC':     'CNBC',
    'People':   'People',
}
PUB_TO_LABEL = {v: k for k, v in LABEL_TO_PUB.items()}

# Stopwords en inglés utilizadas para filtrar palabras vacías en el análisis
# de frecuencia léxica.
STOPWORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'not', 'no', 'nor', 'so', 'yet',
    'both', 'either', 'neither', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'than', 'too', 'very', 'just', 'this', 'that', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'we', 'you', 'i', 'his', 'her',
    'their', 'our', 'your', 'my', 'who', 'which', 'what', 'when', 'where',
    'how', 'all', 'as', 'if', 'up', 'out', 'about', 'into', 'also', 'after',
    'before', 'over', 'under', 'between', 'through', 'said', 'says', 'say',
    'one', 'two', 'new', 'also', 'now', 'then', 'there', 'here', 'us', 'him',
    'me', 'any', 'been', 's', 't', 're', 've', 'll', 'd', 'm',
])

# Cantidad de palabras a mostrar en los gráficos de "top palabras".
N_TOP_WORDS = 15

# Identificador del dataset en HuggingFace Hub.
HF_DATASET_ID = 'tomas-gr/all-the-news-2-1-Component-one-sampled'


# ── Carga de datos ───────────────────────────────────────────────────────────
def load_dataset(data_dir='data', filename='sampled-data.csv'):
    """Carga el dataset, descargándolo desde HuggingFace si no existe.

    Verifica si el archivo CSV está disponible en ``data_dir``. Si no
    lo está, lo descarga desde HuggingFace Hub y lo guarda como CSV
    antes de cargarlo con pandas.

    Args:
        data_dir (str | Path): Directorio donde buscar/guardar el dataset.
        filename (str): Nombre del archivo CSV.

    Returns:
        pd.DataFrame: Dataset cargado.
    """
    data_dir = Path(data_dir)
    data_path = data_dir / filename
    data_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        print(f'Archivo {data_path} no encontrado. '
              f'Descargando dataset desde HuggingFace...')
        # Import diferido para no requerir la librería ``datasets`` cuando
        # el CSV ya está disponible localmente.
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset(
            HF_DATASET_ID, split='train', cache_dir=str(data_dir)
        )
        ds.to_pandas().to_csv(data_path, index=False)
        print(f'Dataset guardado en {data_path}')

    return pd.read_csv(data_path)


# ── Limpieza de texto ────────────────────────────────────────────────────────
def clean_text(df, column_name):
    """Normaliza el texto de una columna del DataFrame.

    Aplica la siguiente secuencia de transformaciones para preparar el
    texto para análisis basado en bolsa de palabras:

        1. Elimina el contenido previo al primer salto de línea
           (típicamente el dateline editorial).
        2. Convierte a minúsculas.
        3. Normaliza las variantes tipográficas de apóstrofo (curly,
           backtick) al apóstrofe ASCII estándar.
        4. Elimina puntuación y caracteres especiales, preservando los
           apóstrofes (para mantener contracciones como ``it's``, ``don't``).
        5. Elimina secuencias numéricas (años, cantidades, etc.).
        6. Colapsa secuencias de espacios en blanco a un único espacio.
        7. Elimina espacios al inicio y al final.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de texto.
        column_name (str): Nombre de la columna sobre la que aplicar la
            limpieza.

    Returns:
        pd.Series: Serie con el texto normalizado.
    """
    return (
        df[column_name]
        .fillna('')
        .astype(str)
        .str.replace(r'^[^\n]*\n', '', regex=True)
        .str.lower()
        .str.replace(r"[‘’`]", "'", regex=True)
        .str.replace(r"[^\w\s']", ' ', regex=True)
        .str.replace(r'\d+', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )


# ── Visualización ────────────────────────────────────────────────────────────
def plot_top_words(df, labels, axes, text_column='CleanText',
                   n_top=N_TOP_WORDS):
    """Dibuja gráficos horizontales con las palabras más frecuentes por medio.

    Para cada medio en ``labels`` calcula las ``n_top`` palabras más
    frecuentes (excluyendo stopwords y tokens de hasta dos caracteres) y
    las representa como barras horizontales en el eje correspondiente.

    Args:
        df (pd.DataFrame): DataFrame con las columnas ``publication`` y
            ``text_column``.
        labels (list[str]): Lista de etiquetas cortas de medios a graficar.
        axes (list[matplotlib.axes.Axes]): Lista de ejes donde graficar,
            con la misma longitud que ``labels``.
        text_column (str): Nombre de la columna con el texto ya limpio.
        n_top (int): Cantidad de palabras a mostrar por medio.
    """
    for ax, lbl in zip(axes, labels):
        pub = LABEL_TO_PUB[lbl]
        texts = df[df['publication'] == pub][text_column].dropna()
        words = ' '.join(texts).split()
        # Filtra stopwords y tokens muy cortos (típicamente ruido residual).
        words_filtered = [
            w for w in words if w not in STOPWORDS and len(w) > 2
        ]
        word_counts = Counter(words_filtered).most_common(n_top)
        wlist, clist = zip(*word_counts)
        # Invierte el orden para que la palabra más frecuente quede arriba.
        wlist, clist = list(wlist)[::-1], list(clist)[::-1]
        ax.barh(
            wlist, clist,
            color=PALETTE[lbl], edgecolor='white', linewidth=0.5,
        )
        ax.set_title(lbl, fontsize=15, fontweight='bold', pad=8)
        ax.set_xlabel('Frecuencia', fontsize=13)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
        )
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.7, alpha=0.8)
        sns.despine(ax=ax)


def plot_wordclouds_by_source(df, text_column='CleanText',
                              source_column='publication',
                              labels=None, max_words=150,
                              colormap='viridis'):
    """Dibuja una nube de palabras por medio en una grilla compartida.

    Para cada medio listado en ``labels`` genera una nube de palabras
    usando ``WordCloud`` y la agrega a un panel de la figura. El orden
    de los paneles sigue ``labels`` (por defecto, el orden canónico
    definido en ``ORDER``).

    Args:
        df (pd.DataFrame): DataFrame con las columnas ``source_column`` y
            ``text_column``.
        text_column (str): Columna con el texto ya limpio.
        source_column (str): Columna que identifica el medio.
        labels (list[str] | None): Lista de etiquetas cortas de medios a
            graficar; si es ``None`` se usa ``ORDER``.
        max_words (int): Máximo de palabras a mostrar por nube.
        colormap (str): Nombre del colormap de matplotlib para colorear
            las palabras.

    Returns:
        matplotlib.figure.Figure: Figura con la grilla de nubes.
    """
    # Import diferido: ``wordcloud`` solo se requiere si se usa esta función.
    from wordcloud import WordCloud

    if labels is None:
        labels = ORDER
    n = len(labels)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i, lbl in enumerate(labels):
        pub = LABEL_TO_PUB[lbl]
        text = ' '.join(
            df[df[source_column] == pub][text_column].dropna().astype(str)
        )
        wc = WordCloud(
            width=800, height=400, background_color='white',
            colormap=colormap, max_words=max_words,
            collocations=False, stopwords=STOPWORDS,
        ).generate(text)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(lbl, fontsize=14, fontweight='bold')
        axes[i].axis('off')

    # Apaga los ejes sobrantes (si el número de medios es impar).
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    return fig
