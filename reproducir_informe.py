"""Genera y guarda todas las figuras del informe de Tarea 1.

Reproduce de forma no interactiva las visualizaciones presentes en la
notebook ``tarea1.ipynb`` y las exporta como archivos PNG dentro de la
carpeta ``figuras/``. Se utiliza el backend ``Agg`` de matplotlib para
evitar la necesidad de un entorno gráfico.

Uso:
    python reproducir_informe.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI: permite correr el script en servidores.
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    LABEL_TO_PUB,
    N_TOP_WORDS,
    ORDER,
    PALETTE,
    PUB_TO_LABEL,
    clean_text,
    load_dataset,
    plot_top_words,
)

# ── Configuración global ──────────────────────────────────────────────────────
# Tema y parámetros de figura compartidos por todas las visualizaciones del
# informe, para garantizar consistencia visual.
sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

FIG_DIR = Path('figuras')
FIG_DIR.mkdir(exist_ok=True)

# ── Carga y preprocesado ──────────────────────────────────────────────────────
# Se carga el dataset, se restringe a los cinco medios con mayor volumen y se
# generan las columnas auxiliares (texto limpio, conteo de palabras, mes).
print("Cargando datos...")
df = load_dataset()
articles_per_pub = df['publication'].value_counts()
top_5_publications = articles_per_pub.head(5).index.tolist()
df_top_5 = df[df['publication'].isin(top_5_publications)].copy()
df_top_5['publication_label'] = df_top_5['publication'].replace(
    'The New York Times', 'NY Times'
)

df_top_5['CleanText'] = clean_text(df_top_5, 'article')
df_top_5['word_count'] = df_top_5['CleanText'].str.split().str.len()
df_top_5['date_parsed'] = pd.to_datetime(df_top_5['date'], errors='coerce')
df_top_5['year_month'] = df_top_5['date_parsed'].dt.to_period('M')

# ── Figura 1: Distribución de artículos ──────────────────────────────────────
# Gráfico de barras con la cantidad de artículos de cada uno de los cinco
# medios seleccionados, con etiquetas numéricas sobre cada barra.
print("fig_distribucion_pubs.png ...")
counts_ordered = {lbl: articles_per_pub[LABEL_TO_PUB[lbl]] for lbl in ORDER}

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(
    ORDER, [counts_ordered[l] for l in ORDER],
    color=[PALETTE[l] for l in ORDER],
    edgecolor='white', linewidth=0.8, width=0.55,
)
# Etiqueta numérica sobre cada barra.
for bar, lbl in zip(bars, ORDER):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
        f'{counts_ordered[lbl]:,}',
        ha='center', va='bottom', fontsize=13, fontweight='bold',
    )
ax.set_title('Cantidad de artículos por medio de prensa', fontsize=18, pad=15)
ax.set_xlabel('')
ax.set_ylabel('Cantidad de artículos', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
)
ax.set_ylim(0, max(counts_ordered.values()) * 1.15)
ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.8)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_distribucion_pubs.png')
plt.close()

# ── Figura 2: Serie temporal mensual ─────────────────────────────────────────
# Cantidad de artículos publicados por mes y por medio, para visualizar la
# evolución temporal del volumen de cobertura.
print("fig_temporal.png ...")
monthly_counts = (
    df_top_5.groupby(['year_month', 'publication_label'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=ORDER, fill_value=0)
)
# Convierte el índice de Period a Timestamp para que matplotlib lo grafique.
monthly_counts.index = monthly_counts.index.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))
for lbl in ORDER:
    ax.plot(
        monthly_counts.index, monthly_counts[lbl],
        label=lbl, color=PALETTE[lbl], linewidth=1.8, alpha=0.9,
    )
ax.set_title(
    'Artículos publicados por mes — top 5 medios de prensa',
    fontsize=18, pad=15,
)
ax.set_xlabel('')
ax.set_ylabel('Artículos por mes', fontsize=14)
ax.tick_params(axis='both', labelsize=13)
ax.legend(
    title='Publicación', bbox_to_anchor=(1.01, 1), loc='upper left',
    frameon=True, fontsize=12, title_fontsize=12,
)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{int(x):,}')
)
ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.8)
fig.autofmt_xdate()
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_temporal.png')
plt.close()

# ── Figuras 3a y 3b: Palabras más frecuentes (split 3+2) ─────────────────────
# Se separan los cinco medios en dos figuras (3 + 2) por una cuestión de
# espacio en el informe, manteniendo el mismo formato visual.
print("fig_top_words_1.png ...")
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
plot_top_words(df_top_5, ORDER[:3], axes)
plt.suptitle(
    f'Top {N_TOP_WORDS} palabras más frecuentes (sin stopwords)',
    fontsize=16, y=1.01,
)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_top_words_1.png')
plt.close()

print("fig_top_words_2.png ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
plot_top_words(df_top_5, ORDER[3:], axes)
plt.suptitle(
    f'Top {N_TOP_WORDS} palabras más frecuentes (sin stopwords)',
    fontsize=16, y=1.01,
)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_top_words_2.png')
plt.close()

# ── Figura 4: Total de palabras + boxplot ────────────────────────────────────
# Figura compuesta con dos paneles:
#   - Izquierda: total acumulado de palabras por medio.
#   - Derecha: distribución de la longitud de los artículos (sin outliers).
print("fig_word_counts.png ...")
word_stats = (
    df_top_5.groupby('publication_label')['word_count']
    .agg(Media='mean', Mediana='median', Total='sum')
    .reindex(ORDER)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(
    ORDER, word_stats['Total'],
    color=[PALETTE[l] for l in ORDER],
    edgecolor='white', width=0.55,
)
axes[0].set_title('Total de palabras por medio', fontsize=18, pad=15)
axes[0].set_xlabel('')
axes[0].set_ylabel('Total de palabras', fontsize=14)
axes[0].tick_params(axis='both', labelsize=13)
# Formato dinámico: muestra millones cuando supera 1e6, miles en caso contrario.
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x):,}'
    )
)
axes[0].grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.8)
sns.despine(ax=axes[0])

# Boxplot sin outliers para que la escala no quede dominada por artículos
# extremadamente largos.
sns.boxplot(
    data=df_top_5, x='publication_label', y='word_count',
    order=ORDER, hue='publication_label', hue_order=ORDER,
    palette='viridis', legend=False, showfliers=False,
    width=0.5, linewidth=1.5, ax=axes[1],
)
axes[1].set_title(
    'Longitud de artículos por medio de prensa', fontsize=18, pad=15
)
axes[1].set_xlabel('')
axes[1].set_ylabel('Palabras por artículo', fontsize=14)
axes[1].tick_params(axis='both', labelsize=13)
axes[1].grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.8)
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_word_counts.png')
plt.close()

# ── Figura 5: Heatmap de menciones ───────────────────────────────────────────
# Para cada par (medio_origen, medio_destino) se cuenta cuántas veces el
# nombre del medio destino aparece en los artículos del medio origen.
print("fig_heatmap_menciones.png ...")
pub_search_terms = {
    'The New York Times': 'new york times',
    'Reuters':            'reuters',
    'The Hill':           'the hill',
    'CNBC':               'cnbc',
    'People':             'people',
}
mentions_matrix = pd.DataFrame(
    0, index=top_5_publications, columns=top_5_publications
)
for sp in top_5_publications:
    # Concatena todos los artículos del medio para hacer un único conteo.
    ct = ' '.join(
        df_top_5[df_top_5['publication'] == sp]['CleanText'].fillna('')
    )
    for tp in top_5_publications:
        if sp != tp:
            mentions_matrix.loc[sp, tp] = ct.count(pub_search_terms[tp])

# Reindexa la matriz con etiquetas cortas y orden canónico para la visualización.
display_matrix = mentions_matrix.copy()
display_matrix.index = [PUB_TO_LABEL[p] for p in display_matrix.index]
display_matrix.columns = [PUB_TO_LABEL[p] for p in display_matrix.columns]
display_matrix = display_matrix.reindex(index=ORDER, columns=ORDER)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    display_matrix, annot=True, fmt='d', cmap='viridis',
    linewidths=0.5, ax=ax,
    cbar_kws={'label': 'Menciones', 'shrink': 0.8},
    annot_kws={'size': 13, 'weight': 'bold'},
)
ax.set_title(
    'Matriz de menciones entre medios\n(fila = fuente, columna = mencionado)',
    fontsize=16, pad=12,
)
ax.set_xlabel('Medio mencionado', fontsize=14)
ax.set_ylabel('Medio que menciona', fontsize=14)
ax.tick_params(axis='both', labelsize=13)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_heatmap_menciones.png')
plt.close()

# ── Figura 6: Grafo dirigido de menciones ────────────────────────────────────
# Representación en grafo de la matriz de menciones. El color de cada arista
# indica el medio de origen y el grosor es proporcional al logaritmo de la
# cantidad de menciones (para comprimir un rango muy amplio: ~5 a ~4 800).
print("fig_grafo_menciones.png ...")
G = nx.DiGraph()
G.add_nodes_from(ORDER)
for lbl_i in ORDER:
    for lbl_j in ORDER:
        if lbl_i != lbl_j:
            w = int(
                mentions_matrix.loc[LABEL_TO_PUB[lbl_i], LABEL_TO_PUB[lbl_j]]
            )
            if w > 0:
                G.add_edge(lbl_i, lbl_j, weight=w)

all_weights = np.array(
    [G[u][v]['weight'] for u, v in G.edges()], dtype=float
)
log_max = np.log1p(all_weights.max())


def log_width(w, min_w=0.6, max_w=9.0):
    """Mapea una cantidad de menciones a un ancho de arista en escala log.

    El uso de ``log1p`` evita problemas con conteos muy bajos y comprime
    el rango de valores para que las aristas pesadas no opaquen a las livianas.

    Args:
        w (float): Cantidad de menciones a representar.
        min_w (float): Ancho mínimo de arista en píxeles.
        max_w (float): Ancho máximo de arista en píxeles.

    Returns:
        float: Ancho de la arista en píxeles, entre ``min_w`` y ``max_w``.
    """
    return min_w + (max_w - min_w) * np.log1p(w) / log_max


RAD = 0.2  # Curvatura de las aristas (evita superposición entre ida y vuelta).
pos = nx.circular_layout(G, scale=1.3)
fig, ax = plt.subplots(figsize=(12, 10))

nx.draw_networkx_nodes(
    G, pos, ax=ax, node_size=4500,
    node_color=[PALETTE[n] for n in G.nodes()],
    linewidths=1.8, edgecolors='#333333', alpha=0.95,
)
nx.draw_networkx_labels(
    G, pos, ax=ax,
    font_size=13, font_weight='bold', font_color='white',
)

# Se dibujan las aristas agrupadas por nodo fuente para que cada grupo
# herede el color del medio que origina la mención.
for source in ORDER:
    edges_src = [(u, v) for u, v in G.edges() if u == source]
    if not edges_src:
        continue
    widths = [log_width(G[u][v]['weight']) for u, v in edges_src]
    nx.draw_networkx_edges(
        G, pos, ax=ax, edgelist=edges_src, width=widths,
        edge_color=[PALETTE[source]] * len(edges_src),
        arrows=True, arrowsize=20,
        connectionstyle=f'arc3,rad={RAD}',
        min_source_margin=42, min_target_margin=42, alpha=0.85,
    )

# Leyenda manual de la escala logarítmica de grosores.
for val, lbl in zip([10, 100, 1000], ['~10', '~100', '~1 000']):
    ax.plot(
        [], [], color='grey', linewidth=log_width(val),
        label=lbl, solid_capstyle='round',
    )
ax.legend(
    title='Menciones\n(escala log)', loc='lower left',
    frameon=True, fontsize=11, title_fontsize=11, handlelength=2.5,
)

ax.set_title(
    'Grafo de menciones entre medios de prensa\n'
    '(color = origen  |  grosor = log menciones)',
    fontsize=18, pad=15,
)
ax.axis('off')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_grafo_menciones.png')
plt.close()

# ── Resumen ───────────────────────────────────────────────────────────────────
# Lista los archivos generados con su tamaño en KB como verificación final.
print("\n✓ Figuras guardadas en 'figuras/':")
for f in sorted(FIG_DIR.iterdir()):
    print(f"  {f.name:45s} {f.stat().st_size/1024:.1f} KB")
