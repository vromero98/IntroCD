"""Genera y guarda todas las figuras del informe de Tarea 2 (version unificada).

Reproduce de forma no interactiva el pipeline de la notebook ``tarea2.ipynb``
(representacion numerica de texto, PCA y modelos de clasificacion) y exporta las
figuras como archivos PDF dentro de ``figuras/``, usando tipografia Times New
Roman y el esquema de colores viridis para mantener consistencia con la Tarea 1.

Refleja las decisiones acordadas al combinar el trabajo de las dos integrantes:
recorte de vocabulario a 20.000 terminos en los modelos, grilla de suavizado
``alpha`` que incluye 0.01, SVM lineal como modelo alternativo, dos cambios de
medio (uno desbalanceado y uno balanceado) y la investigacion de la confusion
entre CNBC y Reuters.

Ademas imprime por consola todas las metricas numericas que se citan en el
informe.

Uso:
    python reproducir_figuras.py
"""

from pathlib import Path
from time import time

import matplotlib
matplotlib.use("Agg")  # Backend sin GUI.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasets import load_dataset

from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

# ── Configuracion global de figuras ──────────────────────────────────────────
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
MAX_FEATURES = 20000   # Recorte de vocabulario usado en modelos y PCA.
BEST_ALPHA = None      # Se fija tras la busqueda de hiperparametros.

# Orden canonico y etiquetas cortas de los tres medios principales.
ORDER = ["Reuters", "NY Times", "CNBC"]
LABEL_TO_PUB = {"Reuters": "Reuters", "NY Times": "The New York Times", "CNBC": "CNBC"}
PUB_TO_LABEL = {v: k for k, v in LABEL_TO_PUB.items()}
# Paleta viridis muestreada en [0, 0.45, 0.75] para que el color mas claro no sea
# amarillo puro (poco legible sobre fondo blanco).
VIRIDIS_POS = [0.0, 0.45, 0.75]
_cmap = matplotlib.colormaps["viridis"]
PALETTE = dict(zip(ORDER, [_cmap(p) for p in VIRIDIS_POS]))


def clean_text(df, column_name):
    """Normaliza el texto de una columna del DataFrame (version Tarea 1)."""
    return (
        df[column_name].fillna("").astype(str)
        .str.replace(r"^[^\n]*\n", "", regex=True)
        .str.lower()
        .str.replace(r"[‘’`]", "'", regex=True)
        .str.replace(r"[^\w\s']", " ", regex=True)
        .str.replace(r"\d+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def split_publications(df, publications, text="article", test_size=TEST_SIZE):
    """Filtra a un conjunto de medios, limpia el texto y particiona estratificado."""
    d = df[df["publication"].isin(publications)].copy()
    d = d[d[text].notna()]
    d["CleanText"] = clean_text(d, text)
    X_tr, X_te, y_tr, y_te = train_test_split(
        d["CleanText"], d["publication"], test_size=test_size,
        stratify=d["publication"], random_state=RANDOM_STATE)
    return d, X_tr, X_te, y_tr, y_te


def short(y):
    return pd.Series(y).map(lambda p: PUB_TO_LABEL.get(p, p)).values


def model_tfidf(**kw):
    """TfidfVectorizer con el recorte de vocabulario y stopwords por defecto."""
    params = dict(max_features=MAX_FEATURES, stop_words="english")
    params.update(kw)
    return TfidfVectorizer(**params)


def plot_confusion(ax, y_true, y_pred, titulo, order, label_to_pub):
    labels_full = [label_to_pub[l] for l in order]
    cm = confusion_matrix(y_true, y_pred, labels=labels_full)
    disp = ConfusionMatrixDisplay(cm, display_labels=order)
    disp.plot(ax=ax, cmap="viridis", colorbar=False, values_format="d")
    ax.grid(False)
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for txt in disp.text_.ravel():
        txt.set_fontsize(13)
    return cm


# ── Carga de datos ────────────────────────────────────────────────────────────
print("Cargando datos...")
ds = load_dataset("tomas-gr/all-the-news-2-1-Component-one-sampled",
                  split="train", cache_dir="../data")
df = ds.to_pandas()

top_3 = df["publication"].value_counts().head(3).index.tolist()
print("Top 3 medios:", top_3)

d3, X_train, X_test, y_train, y_test = split_publications(df, top_3)
print(f"Total top 3: {len(d3)} | train: {len(X_train)} | test: {len(X_test)}")
print(d3["publication"].value_counts())


# ── Discrepancia muestreo aleatorio vs estratificado ─────────────────────────
print("\n=== Aleatorio vs estratificado ===")
yr_tr, yr_te = train_test_split(d3["publication"], test_size=TEST_SIZE,
                                random_state=RANDOM_STATE)  # sin stratify
disc_rand = (yr_tr.value_counts(normalize=True) - yr_te.value_counts(normalize=True)).abs().max()
disc_strat = (pd.Series(y_train).value_counts(normalize=True)
              - pd.Series(y_test).value_counts(normalize=True)).abs().max()
print(f"Discrepancia maxima train/test (aleatorio):     {disc_rand:.4%}")
print(f"Discrepancia maxima train/test (estratificado): {disc_strat:.4%}")


# ── Figura: balance de clases train/test ─────────────────────────────────────
print("\nfig_balance.pdf ...")
prop_train = pd.Series(short(y_train)).value_counts(normalize=True).reindex(ORDER)
prop_test = pd.Series(short(y_test)).value_counts(normalize=True).reindex(ORDER)
print("Proporcion train (%):\n", (prop_train * 100).round(2))

x = np.arange(len(ORDER)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w / 2, prop_train.values * 100, w, label="Train",
            color=PALETTE["Reuters"], edgecolor="white")
b2 = ax.bar(x + w / 2, prop_test.values * 100, w, label="Test",
            color=PALETTE["CNBC"], edgecolor="white")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(ORDER)
ax.set_ylabel("Porcentaje de articulos (%)")
ax.set_title("Balance de clases en train y test")
ax.set_ylim(0, max(prop_train.max(), prop_test.max()) * 100 * 1.18)
ax.legend(); ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine(); plt.tight_layout(); plt.savefig(FIG_DIR / "fig_balance.pdf"); plt.close()


# ── Bag of Words: tamano, sparsity y cobertura del vocabulario ───────────────
print("\n=== Bag of Words ===")
count_vec = CountVectorizer()
X_counts = count_vec.fit_transform(X_train)
n_docs, n_vocab = X_counts.shape
print(f"BoW shape: {X_counts.shape} | nnz: {X_counts.nnz:,} | "
      f"sparsity: {100 * (1 - X_counts.nnz / (n_docs * n_vocab)):.2f}%")
print(f"Densa (float64): {n_docs * n_vocab * 8 / 1e9:.2f} GB")

term_counts = np.asarray(X_counts.sum(axis=0)).ravel()
doc_freq = np.asarray((X_counts > 0).sum(axis=0)).ravel()
order_t = np.argsort(term_counts)[::-1]
cum_cov = np.cumsum(term_counts[order_t]) / term_counts.sum()
print(f"Vocabulario completo train: {n_vocab:,}")
for k in [5000, 10000, 20000, 30000]:
    if k <= n_vocab:
        print(f"  top-{k:,}: cobertura ocurrencias = {cum_cov[k - 1]:.2%}")
print(f"Palabras en 1 solo documento: {(doc_freq <= 1).sum():,} "
      f"({(doc_freq <= 1).mean():.1%} del vocabulario)")


# ── TF-IDF ───────────────────────────────────────────────────────────────────
tfidf_full = TfidfVectorizer()
print(f"TF-IDF (vocab completo) shape: {tfidf_full.fit_transform(X_train).shape}")


# ── PCA sobre TF-IDF: comparacion de configuraciones ─────────────────────────
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
y_lbl = short(y_train)
fig, axes = plt.subplots(2, 2, figsize=(13, 11)); axes = axes.flatten()
for ax, (titulo, params) in zip(axes, configs):
    Xv = TfidfVectorizer(max_features=MAX_FEATURES, **params).fit_transform(X_train).toarray()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xv)
    for lbl in ORDER:
        m = y_lbl == lbl
        ax.scatter(coords[m, 0], coords[m, 1], s=7, alpha=0.45,
                   color=PALETTE[lbl], label=lbl, edgecolors="none")
    ev = pca.explained_variance_ratio_ * 100
    ax.set_title(titulo, fontsize=13)
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    print(f"  {titulo}: var PC1+PC2 = {ev.sum():.2f}%")
    del Xv
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=13,
           markerscale=2.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("PCA (2 componentes) sobre TF-IDF segun configuracion", fontsize=17, y=1.05)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_pca_configs.pdf"); plt.close()


# ── Varianza explicada vs numero de componentes ──────────────────────────────
print("\nfig_varianza.pdf ...")
Xv = model_tfidf(use_idf=True).fit_transform(X_train).toarray()
pca10 = PCA(n_components=10, random_state=RANDOM_STATE).fit(Xv)
evr = pca10.explained_variance_ratio_ * 100
cum = np.cumsum(evr)
print("Var acumulada (%):", cum.round(2))
del Xv

comps = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(comps, evr, color=PALETTE["NY Times"], edgecolor="white", label="Individual")
ax.plot(comps, cum, "-o", color=PALETTE["Reuters"], linewidth=2, markersize=6,
        label="Acumulada")
for c, v in zip(comps, cum):
    ax.text(c, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Numero de componentes principales")
ax.set_ylabel("Varianza explicada (%)")
ax.set_title("Varianza explicada acumulada por PCA sobre TF-IDF")
ax.set_xticks(comps); ax.legend()
ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine(); plt.tight_layout(); plt.savefig(FIG_DIR / "fig_varianza.pdf"); plt.close()


# ── Multinomial Naive Bayes (base) ───────────────────────────────────────────
print("\n=== Naive Bayes base (max_features=20000) ===")
tfidf_base = model_tfidf()
Xtr = tfidf_base.fit_transform(X_train)
Xte = tfidf_base.transform(X_test)
nb = MultinomialNB().fit(Xtr, y_train)
y_pred_nb = nb.predict(Xte)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy NB base: {acc_nb:.4f}")
print(classification_report(y_test, y_pred_nb, target_names=[PUB_TO_LABEL[p] for p in nb.classes_]))

print("fig_confusion_nb.pdf ...")
fig, ax = plt.subplots(figsize=(6.5, 5.5))
plot_confusion(ax, y_test, y_pred_nb, f"Naive Bayes (accuracy = {acc_nb:.3f})",
               ORDER, LABEL_TO_PUB)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_confusion_nb.pdf"); plt.close()


# ── Investigacion: ¿CNBC republica contenido de Reuters? ─────────────────────
print("\n=== Investigacion CNBC <-> Reuters ===")
for pub in ["CNBC", "Reuters", "The New York Times"]:
    art = d3.loc[d3["publication"] == pub, "article"].fillna("").str.lower()
    print(f'  {pub:<20}: {art.str.contains("reuters").mean():.1%} mencionan "reuters"')
common_titles = (set(d3.loc[d3["publication"] == "CNBC", "title"].dropna())
                 & set(d3.loc[d3["publication"] == "Reuters", "title"].dropna()))
print(f"Titulos identicos CNBC y Reuters: {len(common_titles)}")
for t in list(common_titles)[:3]:
    print("   -", str(t)[:70])

import re
def quitar(textos, palabras):
    patron = r"\b(" + "|".join(palabras) + r")\b"
    return textos.str.replace(patron, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

variantes = {
    "baseline": d3["CleanText"],
    'sin "reuters"': quitar(d3["CleanText"], ["reuters"]),
    "sin nombres de medios": quitar(d3["CleanText"],
                                    ["reuters", "cnbc", "nbc", "msnbc", "thomson", "nytimes"]),
}
for etiqueta, txt in variantes.items():
    Xt, Xv2, yt, yv = train_test_split(txt, d3["publication"], test_size=TEST_SIZE,
                                       random_state=RANDOM_STATE, stratify=d3["publication"])
    pipe = Pipeline([("tfidf", model_tfidf(ngram_range=(1, 2))),
                     ("clf", LinearSVC(random_state=RANDOM_STATE))])
    pipe.fit(Xt, yt)
    yp = pipe.predict(Xv2)
    rep = classification_report(yv, yp, output_dict=True)
    print(f"  {etiqueta:24s} accuracy={accuracy_score(yv, yp):.4f}  recall CNBC={rep['CNBC']['recall']:.3f}")


# ── Validacion cruzada + busqueda de hiperparametros ─────────────────────────
print("\n=== GridSearchCV ===")
pipe = Pipeline([("tfidf", model_tfidf()), ("nb", MultinomialNB())])
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "nb__alpha": [0.01, 0.1, 0.5, 1.0],
}
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
BEST_ALPHA = grid.best_params_["nb__alpha"]
BEST_NGRAM = grid.best_params_["tfidf__ngram_range"]
print("Mejor combinacion:", grid.best_params_, "| CV acc:", round(grid.best_score_, 4))

res = grid.cv_results_
rows = []
for i, p in enumerate(res["params"]):
    ng = "1-2g" if p["tfidf__ngram_range"] == (1, 2) else "1g"
    nombre = f"{ng}, alpha={p['nb__alpha']}"
    for s in range(5):
        rows.append({"modelo": nombre, "accuracy": res[f"split{s}_test_score"][i]})
cv_df = pd.DataFrame(rows)
orden = cv_df.groupby("modelo")["accuracy"].mean().sort_values(ascending=False).index.tolist()

print("fig_violin_cv.pdf ...")
fig, ax = plt.subplots(figsize=(10, 6.5))
sns.violinplot(data=cv_df, y="modelo", x="accuracy", order=orden,
               hue="modelo", legend=False, palette="viridis", inner="point", cut=0, ax=ax)
ax.set_ylabel("Configuracion (n-gramas, suavizado alpha)")
ax.set_xlabel("Accuracy (5-fold CV)")
ax.set_title("Distribucion de accuracy en validacion cruzada")
ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.7)
sns.despine(); plt.tight_layout(); plt.savefig(FIG_DIR / "fig_violin_cv.pdf"); plt.close()
print(cv_df.groupby("modelo")["accuracy"].agg(["mean", "std"]).round(4).loc[orden])


# ── Justificacion empirica de k ──────────────────────────────────────────────
print("\n=== Eleccion de k (folds) ===")
pipe_best = Pipeline([("tfidf", model_tfidf(ngram_range=BEST_NGRAM)),
                      ("nb", MultinomialNB(alpha=BEST_ALPHA))])
for k in [2, 3, 5, 10]:
    t0 = time()
    sc = cross_val_score(pipe_best, X_train, y_train, cv=k, scoring="accuracy", n_jobs=-1)
    print(f"  k={k:>2}: acc={sc.mean():.4f} +/- {sc.std():.4f} ({time() - t0:.1f}s)")


# ── Entrenamiento final + modelo alternativo (SVM) ───────────────────────────
print("\n=== Modelo final (NB) + alternativo (SVM) ===")
best = grid.best_estimator_.fit(X_train, y_train)
y_pred_best = best.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
print(f"Accuracy NB optimizado: {acc_best:.4f}")
print(classification_report(y_test, y_pred_best, target_names=[PUB_TO_LABEL[p] for p in best.classes_]))

svm = Pipeline([("tfidf", model_tfidf(ngram_range=(1, 2))),
                ("clf", LinearSVC(random_state=RANDOM_STATE))]).fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy SVM lineal: {acc_svm:.4f}")
print(classification_report(y_test, y_pred_svm, target_names=[PUB_TO_LABEL[p] for p in svm.classes_]))

# Regresion logistica solo para la mencion comparativa de una linea.
logreg = Pipeline([("tfidf", model_tfidf(ngram_range=(1, 2))),
                   ("clf", LogisticRegression(max_iter=1000, C=10))]).fit(X_train, y_train)
acc_lr = accuracy_score(y_test, logreg.predict(X_test))
print(f"Accuracy Regresion Logistica (referencia): {acc_lr:.4f}")

print("fig_confusion_modelos.pdf ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
plot_confusion(axes[0], y_test, y_pred_best, f"Naive Bayes optimizado (acc = {acc_best:.3f})",
               ORDER, LABEL_TO_PUB)
plot_confusion(axes[1], y_test, y_pred_svm, f"SVM lineal (acc = {acc_svm:.3f})",
               ORDER, LABEL_TO_PUB)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_confusion_modelos.pdf"); plt.close()


# ── Cambio de medio 1: desbalanceado (People) ────────────────────────────────
print("\n=== Cambio de medio: Reuters, NY Times, People (desbalanceado) ===")
ORDER_P = ["Reuters", "NY Times", "People"]
L2P = {"Reuters": "Reuters", "NY Times": "The New York Times", "People": "People"}
PAL_P = dict(zip(ORDER_P, [_cmap(p) for p in VIRIDIS_POS]))
dP, XP_tr, XP_te, yP_tr, yP_te = split_publications(df, list(L2P.values()))
print("Conteo:\n", dP["publication"].value_counts())
nbP = Pipeline([("tfidf", model_tfidf(ngram_range=BEST_NGRAM)),
                ("clf", MultinomialNB(alpha=BEST_ALPHA))]).fit(XP_tr, yP_tr)
yP_pred = nbP.predict(XP_te)
accP = accuracy_score(yP_te, yP_pred)
print(f"Accuracy: {accP:.4f} | macro F1: {f1_score(yP_te, yP_pred, average='macro'):.4f}")
print(classification_report(yP_te, yP_pred, target_names=[L2P[short([p])[0]] if False else p for p in nbP.classes_]))

print("fig_cambio.pdf ...")
propP = pd.Series(pd.Series(yP_tr).map({v: k for k, v in L2P.items()}).values).value_counts(normalize=True).reindex(ORDER_P)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
bars = axes[0].bar(ORDER_P, propP.values * 100, color=[PAL_P[l] for l in ORDER_P], edgecolor="white")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=12)
axes[0].set_ylabel("Porcentaje de articulos (%)"); axes[0].set_title("Balance del nuevo conjunto (train)")
axes[0].set_ylim(0, propP.max() * 100 * 1.2)
axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7); sns.despine(ax=axes[0])
plot_confusion(axes[1], yP_te, yP_pred, f"Naive Bayes (acc = {accP:.3f})", ORDER_P, L2P)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cambio.pdf"); plt.close()


# ── Cambio de medio 2: balanceado (Refinery 29, Vice, Mashable) ──────────────
print("\n=== Cambio de medio: trio balanceado (Refinery 29, Vice, Mashable) ===")
trio = ["Refinery 29", "Vice", "Mashable"]
PAL_B = dict(zip(trio, [_cmap(p) for p in VIRIDIS_POS]))
L_B = {t: t for t in trio}
dB, XB_tr, XB_te, yB_tr, yB_te = split_publications(df, trio)
conteos_B = dB["publication"].value_counts()
print("Conteo:\n", conteos_B, f"\nratio max/min = {conteos_B.max() / conteos_B.min():.2f}")
nbB = Pipeline([("tfidf", model_tfidf(ngram_range=BEST_NGRAM)),
                ("clf", MultinomialNB(alpha=BEST_ALPHA))]).fit(XB_tr, yB_tr)
yB_pred = nbB.predict(XB_te)
accB = accuracy_score(yB_te, yB_pred)
f1B = f1_score(yB_te, yB_pred, average="macro")
print(f"Accuracy: {accB:.4f} | macro F1: {f1B:.4f}")
print(classification_report(yB_te, yB_pred))

print("fig_cambio_balanceado.pdf ...")
propB = pd.Series(yB_tr).value_counts(normalize=True).reindex(trio)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
bars = axes[0].bar(trio, propB.values * 100, color=[PAL_B[l] for l in trio], edgecolor="white")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=12)
axes[0].set_ylabel("Porcentaje de articulos (%)"); axes[0].set_title("Balance del trio balanceado (train)")
axes[0].set_ylim(0, propB.max() * 100 * 1.25)
axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.7); sns.despine(ax=axes[0])
plot_confusion(axes[1], yB_te, yB_pred, f"Naive Bayes (acc = {accB:.3f}, macro F1 = {f1B:.3f})", trio, L_B)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig_cambio_balanceado.pdf"); plt.close()


# ── Opcional: clasificacion a nivel de titulo ────────────────────────────────
print("\n=== Opcional: a nivel de titulo ===")
dt, Xt_tr, Xt_te, yt_tr, yt_te = split_publications(df, top_3, text="title")
nbT = Pipeline([("tfidf", model_tfidf(ngram_range=BEST_NGRAM)),
                ("clf", MultinomialNB(alpha=BEST_ALPHA))]).fit(Xt_tr, yt_tr)
yt_pred = nbT.predict(Xt_te)
acc_t = accuracy_score(yt_te, yt_pred)
print(f"Accuracy a nivel de titulo: {acc_t:.4f}")
print(classification_report(yt_te, yt_pred, target_names=[PUB_TO_LABEL[p] for p in nbT.classes_]))


# ── Resumen ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("RESUMEN DE METRICAS")
print("=" * 64)
print(f"NB base:                 {acc_nb:.4f}")
print(f"Mejor combinacion CV:    {grid.best_params_}  (acc {grid.best_score_:.4f})")
print(f"NB optimizado (test):    {acc_best:.4f}")
print(f"SVM lineal (test):       {acc_svm:.4f}")
print(f"Reg. Logistica (test):   {acc_lr:.4f}")
print(f"People (desbalanceado):  {accP:.4f}")
print(f"Trio balanceado:         {accB:.4f}  (macro F1 {f1B:.4f})")
print(f"A nivel de titulo:       {acc_t:.4f}")
print("\nFiguras en 'figuras/':")
for f in sorted(FIG_DIR.glob("*.pdf")):
    print(f"  {f.name:32s} {f.stat().st_size / 1024:.1f} KB")
