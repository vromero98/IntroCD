"""Genera y guarda todas las figuras del informe de Tarea 2.

Reproduce de forma no interactiva el pipeline de la notebook
``2026_template_tarea2.ipynb`` (representacion numerica de texto,
PCA y modelos de clasificacion) y exporta las figuras como archivos
PDF dentro de la carpeta ``figuras/``, usando tipografia Times New Roman
y el esquema de colores viridis para mantener consistencia con la Tarea 1.

Ademas imprime por consola todas las metricas numericas que se citan en
el informe.

Uso:
    python reproducir_figuras.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasets import load_dataset

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
)

# ── Configuracion global de figuras ──────────────────────────────────────────
# Tipografia Times New Roman y tamanos legibles. viridis como colormap base.
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})

FIG_DIR = Path("figuras")
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
PCA_MAX_FEATURES = 5000  # Cota de vocabulario para poder densificar en PCA.

# Orden canonico y etiquetas cortas de los tres medios principales.
ORDER = ["Reuters", "NY Times", "CNBC"]
LABEL_TO_PUB = {
    "Reuters": "Reuters",
    "NY Times": "The New York Times",
    "CNBC": "CNBC",
}
PUB_TO_LABEL = {v: k for k, v in LABEL_TO_PUB.items()}
# Paleta viridis fija por medio (se reutiliza en todas las figuras). Se muestrea
# viridis en posiciones [0, 0.45, 0.75] para que el color mas claro no sea el
# amarillo puro del extremo, que se distingue mal sobre fondo blanco.
VIRIDIS_POS = [0.0, 0.45, 0.75]
_cmap = matplotlib.colormaps["viridis"]
PALETTE = dict(zip(ORDER, [_cmap(p) for p in VIRIDIS_POS]))


def clean_text(df, column_name):
    """Normaliza el texto de una columna del DataFrame (version Tarea 1)."""
    return (
        df[column_name]
        .fillna("")
        .astype(str)
        .str.replace(r"^[^\n]*\n", "", regex=True)
        .str.lower()
        .str.replace(r"[‘’`]", "'", regex=True)
        .str.replace(r"[^\w\s']", " ", regex=True)
        .str.replace(r"\d+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def split_top_publications(df, publications, test_size=TEST_SIZE):
    """Filtra el DataFrame a un conjunto de medios y particiona estratificado."""
    d = df[df["publication"].isin(publications)].copy()
    d = d[d["article"].notna()]
    d["CleanText"] = clean_text(d, "article")
    X_train, X_test, y_train, y_test = train_test_split(
        d["CleanText"], d["publication"],
        test_size=test_size, stratify=d["publication"],
        random_state=RANDOM_STATE,
    )
    return d, X_train, X_test, y_train, y_test


def short_labels(y, mapping=PUB_TO_LABEL):
    return pd.Series(y).map(mapping).values


# ── Carga de datos ────────────────────────────────────────────────────────────
print("Cargando datos...")
ds = load_dataset(
    "tomas-gr/all-the-news-2-1-Component-one-sampled",
    split="train", cache_dir="../data",
)
df = ds.to_pandas()

top_3 = df["publication"].value_counts().head(3).index.tolist()
print("Top 3 medios:", top_3)

d3, X_train, X_test, y_train, y_test = split_top_publications(df, top_3)
print(f"Total top 3: {len(d3)} | train: {len(X_train)} | test: {len(X_test)}")
print("\nConteo por medio (total):")
print(d3["publication"].value_counts())


# ── Figura 1: Balance de clases train/test ───────────────────────────────────
print("\nfig_balance.pdf ...")
prop_train = (
    pd.Series(short_labels(y_train)).value_counts(normalize=True).reindex(ORDER)
)
prop_test = (
    pd.Series(short_labels(y_test)).value_counts(normalize=True).reindex(ORDER)
)
print("Proporcion train:\n", (prop_train * 100).round(2))
print("Proporcion test:\n", (prop_test * 100).round(2))

x = np.arange(len(ORDER))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w / 2, prop_train.values * 100, w, label="Train",
            color=PALETTE["Reuters"], edgecolor="white")
b2 = ax.bar(x + w / 2, prop_test.values * 100, w, label="Test",
            color=PALETTE["CNBC"], edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(ORDER)
ax.set_ylabel("Porcentaje de articulos (%)")
ax.set_title("Balance de clases en train y test")
ax.set_ylim(0, max(prop_train.max(), prop_test.max()) * 100 * 1.18)
ax.legend()
ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_balance.pdf")
plt.close()


# ── Parte 1.3: Bag of Words ──────────────────────────────────────────────────
print("\n=== Bag of Words ===")
count_vec = CountVectorizer()
X_train_counts = count_vec.fit_transform(X_train)
n_docs, n_vocab = X_train_counts.shape
nnz = X_train_counts.nnz
sparsity = 100 * (1 - nnz / (n_docs * n_vocab))
print(f"BoW shape: {X_train_counts.shape}")
print(f"nnz: {nnz:,} | sparsity: {sparsity:.2f}%")
print(f"Densa en memoria (float64): {n_docs * n_vocab * 8 / 1e9:.2f} GB")

# Ejemplo: conteos de un documento.
ejemplo_idx = 0
fila = X_train_counts[ejemplo_idx]
vocab = np.array(count_vec.get_feature_names_out())
idxs = fila.indices[np.argsort(fila.data)[::-1][:8]]
print("Ejemplo (doc 0) palabras con mayor conteo:")
for j in idxs:
    print(f"   {vocab[j]:>15s} : {X_train_counts[ejemplo_idx, j]}")


# ── Parte 1.4: TF-IDF ────────────────────────────────────────────────────────
print("\n=== TF-IDF ===")
tfidf_vec = TfidfVectorizer()
X_train_tfidf = tfidf_vec.fit_transform(X_train)
print(f"TF-IDF shape: {X_train_tfidf.shape} | nnz: {X_train_tfidf.nnz:,}")


# ── Parte 1.5: PCA sobre TF-IDF (comparacion de configuraciones) ─────────────
print("\nfig_pca_configs.pdf ...")

configs = [
    ("Base (sin stopwords, sin idf, 1-grama)",
     dict(stop_words=None, use_idf=False, ngram_range=(1, 1))),
    ("+ stop_words='english'",
     dict(stop_words="english", use_idf=False, ngram_range=(1, 1))),
    ("+ use_idf=True",
     dict(stop_words="english", use_idf=True, ngram_range=(1, 1))),
    ("+ ngram_range=(1,2)",
     dict(stop_words="english", use_idf=True, ngram_range=(1, 2))),
]

y_train_lbl = short_labels(y_train)
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()
for ax, (titulo, params) in zip(axes, configs):
    vec = TfidfVectorizer(max_features=PCA_MAX_FEATURES, **params)
    Xv = vec.fit_transform(X_train).toarray()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xv)
    for lbl in ORDER:
        m = y_train_lbl == lbl
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.45,
                   color=PALETTE[lbl], label=lbl, edgecolors="none")
    ev = pca.explained_variance_ratio_ * 100
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    print(f"  {titulo}: var explicada PC1+PC2 = {ev.sum():.2f}%")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13,
           markerscale=2.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("PCA (2 componentes) sobre TF-IDF segun configuracion",
             fontsize=17, y=1.05)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_pca_configs.pdf")
plt.close()


# ── Parte 1.5: Varianza explicada vs numero de componentes ───────────────────
print("\nfig_varianza.pdf ...")
vec = TfidfVectorizer(stop_words="english", use_idf=True,
                      max_features=PCA_MAX_FEATURES)
Xv = vec.fit_transform(X_train).toarray()
pca10 = PCA(n_components=10, random_state=RANDOM_STATE)
pca10.fit(Xv)
evr = pca10.explained_variance_ratio_ * 100
cum = np.cumsum(evr)
print("Var explicada por componente (%):", evr.round(2))
print("Var explicada acumulada (%):", cum.round(2))

comps = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(comps, evr, color=PALETTE["NY Times"], edgecolor="white",
              label="Individual")
ax.plot(comps, cum, "-o", color=PALETTE["Reuters"], linewidth=2,
        markersize=6, label="Acumulada")
for c, v in zip(comps, cum):
    ax.text(c, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Numero de componentes principales")
ax.set_ylabel("Varianza explicada (%)")
ax.set_title("Varianza explicada acumulada por PCA sobre TF-IDF")
ax.set_xticks(comps)
ax.legend()
ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_varianza.pdf")
plt.close()


# ── Helper de matriz de confusion ────────────────────────────────────────────
def plot_confusion(ax, y_true, y_pred, titulo, order=ORDER,
                   label_to_pub=LABEL_TO_PUB):
    # y_true / y_pred vienen con los nombres completos de los medios.
    labels_full = [label_to_pub[l] for l in order]
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    disp = ConfusionMatrixDisplay(cm, display_labels=order)
    disp.plot(ax=ax, cmap="viridis", colorbar=False, values_format="d")
    ax.grid(False)  # Evita que el whitegrid se superponga a las celdas.
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for txt in disp.text_.ravel():
        txt.set_fontsize(13)
    return cm


# ── Parte 2.1: Multinomial Naive Bayes (modelo base) ─────────────────────────
print("\n=== Multinomial Naive Bayes (base) ===")
tfidf_base = TfidfVectorizer(stop_words="english")
Xtr = tfidf_base.fit_transform(X_train)
Xte = tfidf_base.transform(X_test)

nb = MultinomialNB()
nb.fit(Xtr, y_train)
y_pred_nb = nb.predict(Xte)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy NB base: {acc_nb:.4f}")
print(classification_report(y_test, y_pred_nb,
      target_names=[PUB_TO_LABEL[p] for p in nb.classes_]))

print("fig_confusion_nb.pdf ...")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
plot_confusion(ax, y_test, y_pred_nb,
               f"Naive Bayes (accuracy = {acc_nb:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_confusion_nb.pdf")
plt.close()


# ── Parte 2.2: GridSearchCV + violin plot ────────────────────────────────────
print("\n=== GridSearchCV ===")
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__alpha": [0.01, 0.1, 1.0],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy",
                    n_jobs=-1, return_train_score=False)
grid.fit(X_train, y_train)
print("Mejor combinacion:", grid.best_params_)
print(f"Mejor accuracy CV: {grid.best_score_:.4f}")

res = grid.cv_results_
n_splits = 5
rows = []
for i in range(len(res["params"])):
    ng = res["params"][i]["tfidf__ngram_range"]
    al = res["params"][i]["clf__alpha"]
    nombre = f"{'1-2g' if ng == (1, 2) else '1g'}\nalpha={al}"
    for s in range(n_splits):
        rows.append({"modelo": nombre,
                     "accuracy": res[f"split{s}_test_score"][i]})
cv_df = pd.DataFrame(rows)
orden_mod = (cv_df.groupby("modelo")["accuracy"].mean()
             .sort_values().index.tolist())

print("fig_violin_cv.pdf ...")
fig, ax = plt.subplots(figsize=(11, 6))
sns.violinplot(data=cv_df, x="modelo", y="accuracy", order=orden_mod,
               hue="modelo", legend=False, palette="viridis",
               inner="point", cut=0, ax=ax)
ax.set_xlabel("Configuracion (n-gramas / suavizado alpha)")
ax.set_ylabel("Accuracy (5-fold CV)")
ax.set_title("Distribucion de accuracy en validacion cruzada")
ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_violin_cv.pdf")
plt.close()
print(cv_df.groupby("modelo")["accuracy"].agg(["mean", "std"]).round(4))


# ── Parte 2.3 y 2.4: Mejor modelo + modelo alternativo ───────────────────────
print("\n=== Mejor modelo (re-entrenado) y alternativo ===")
best = grid.best_estimator_
best.fit(X_train, y_train)
y_pred_best = best.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy mejor modelo (test): {acc_best:.4f}")
print(classification_report(y_test, y_pred_best,
      target_names=[PUB_TO_LABEL[p] for p in best.classes_]))

# Modelo alternativo: Regresion Logistica sobre las mismas features TF-IDF.
logreg = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=10)),
])
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy LogReg (test): {acc_lr:.4f}")
print(classification_report(y_test, y_pred_lr,
      target_names=[PUB_TO_LABEL[p] for p in logreg.classes_]))

print("fig_confusion_modelos.pdf ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
plot_confusion(axes[0], y_test, y_pred_best,
               f"Naive Bayes optimizado (acc = {acc_best:.3f})")
plot_confusion(axes[1], y_test, y_pred_lr,
               f"Regresion Logistica (acc = {acc_lr:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_confusion_modelos.pdf")
plt.close()


# ── Parte 2.5: Cambio de medio de prensa ─────────────────────────────────────
print("\n=== Cambio de medio: Reuters, NY Times, People ===")
nuevos = ["Reuters", "The New York Times", "People"]
ORDER2 = ["Reuters", "NY Times", "People"]
LABEL_TO_PUB2 = {"Reuters": "Reuters", "NY Times": "The New York Times",
                 "People": "People"}
PUB_TO_LABEL2 = {v: k for k, v in LABEL_TO_PUB2.items()}
PALETTE2 = dict(zip(ORDER2, [_cmap(p) for p in VIRIDIS_POS]))

d2, X2_tr, X2_te, y2_tr, y2_te = split_top_publications(df, nuevos)
print("Conteo nuevo set:\n", d2["publication"].value_counts())

nb2 = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB(alpha=grid.best_params_["clf__alpha"])),
])
nb2.fit(X2_tr, y2_tr)
y2_pred = nb2.predict(X2_te)
acc2 = accuracy_score(y2_te, y2_pred)
print(f"Accuracy nuevo set: {acc2:.4f}")
print(classification_report(y2_te, y2_pred,
      target_names=[PUB_TO_LABEL2[p] for p in nb2.classes_]))

# Figura: balance del nuevo set + matriz de confusion.
print("fig_cambio.pdf ...")
prop2_tr = (pd.Series(pd.Series(y2_tr).map(PUB_TO_LABEL2).values)
            .value_counts(normalize=True).reindex(ORDER2))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
bars = axes[0].bar(ORDER2, prop2_tr.values * 100,
                   color=[PALETTE2[l] for l in ORDER2], edgecolor="white")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=12)
axes[0].set_ylabel("Porcentaje de articulos (%)")
axes[0].set_title("Balance del nuevo conjunto (train)")
axes[0].set_ylim(0, prop2_tr.max() * 100 * 1.2)
axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine(ax=axes[0])

labels_full2 = [LABEL_TO_PUB2[l] for l in ORDER2]
cm2 = confusion_matrix(y2_te, y2_pred, labels=labels_full2)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=ORDER2)
disp2.plot(ax=axes[1], cmap="viridis", colorbar=False, values_format="d")
axes[1].grid(False)
axes[1].set_title(f"Naive Bayes nuevo set (acc = {acc2:.3f})")
axes[1].set_xlabel("Prediccion")
axes[1].set_ylabel("Real")
for txt in disp2.text_.ravel():
    txt.set_fontsize(13)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_cambio.pdf")
plt.close()


# ── Parte 2.7 (opcional): clasificacion a nivel de titulo ────────────────────
print("\n=== Opcional: clasificacion a nivel de titulo (top 3) ===")
dt = df[df["publication"].isin(top_3)].copy()
dt = dt[dt["title"].notna()]
dt["CleanTitle"] = clean_text(dt, "title")
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
    dt["CleanTitle"], dt["publication"], test_size=TEST_SIZE,
    stratify=dt["publication"], random_state=RANDOM_STATE)
nb_t = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB(alpha=grid.best_params_["clf__alpha"])),
])
nb_t.fit(Xt_tr, yt_tr)
yt_pred = nb_t.predict(Xt_te)
acc_t = accuracy_score(yt_te, yt_pred)
print(f"Accuracy a nivel de titulo: {acc_t:.4f}")
print(classification_report(yt_te, yt_pred,
      target_names=[PUB_TO_LABEL[p] for p in nb_t.classes_]))


# ── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN DE METRICAS")
print("=" * 60)
print(f"Accuracy NB base:            {acc_nb:.4f}")
print(f"Accuracy NB optimizado:      {acc_best:.4f}  {grid.best_params_}")
print(f"Accuracy LogReg:             {acc_lr:.4f}")
print(f"Accuracy nuevo set (People): {acc2:.4f}")
print(f"Accuracy a nivel de titulo:  {acc_t:.4f}")
print("\nFiguras guardadas en 'figuras/':")
for f in sorted(FIG_DIR.glob("*.pdf")):
    print(f"  {f.name:32s} {f.stat().st_size/1024:.1f} KB")
