# Introducción a la Ciencia de Datos — Tareas 1 y 2

Maestría en Ciencia de Datos y Aprendizaje Automático · Facultad de Ingeniería (UdelaR).

Este repositorio reúne las dos primeras tareas del curso, ambas sobre un subconjunto del dataset *All The News 2.1* de noticias de medios de prensa anglosajones. La Tarea 2 es la continuación de la Tarea 1, por lo que reutiliza los mismos datos y parte de su código.

- **Tarea 1 — Exploración y limpieza de datos:** análisis de valores nulos, visualización temporal, limpieza de texto y conteo de palabras sobre los cinco medios de mayor volumen (Reuters, The New York Times, CNBC, The Hill y People).
- **Tarea 2 — Representación de texto y clasificación de medios:** representación numérica del texto (*bag of words* y TF-IDF), visualización con PCA y entrenamiento de modelos de clasificación (Multinomial Naive Bayes y una SVM lineal) para predecir el medio de prensa a partir del artículo.

## Informes

Cada tarea tiene su informe final en PDF, que es el principal entregable:

- Tarea 1: [`tarea1/informe.pdf`](tarea1/informe.pdf)
- Tarea 2: [`tarea2/informe.pdf`](tarea2/informe.pdf)

## Estructura del repositorio

```
.
├── tarea1/
│   ├── tarea1.ipynb                  # Notebook unificada (versión que produce el informe)
│   ├── informe.pdf                   # Informe final de la Tarea 1
│   ├── reproducir_informe.py         # Reproduce las figuras del informe
│   ├── src/                          # Carga de datos, clean_text y helpers de visualización
│   ├── notebooks/                    # Notebooks individuales de cada integrante (referencia)
│   └── publications_data_profiling_report.html
├── tarea2/
│   ├── tarea2.ipynb                  # Notebook unificada (representación de texto, PCA y modelos)
│   ├── informe.pdf                   # Informe final de la Tarea 2
│   ├── reproducir_figuras.py         # Reproduce las figuras del informe (PDF, viridis)
│   └── notebooks/                    # Notebooks individuales de cada integrante (referencia)
├── data/                             # Cache del dataset (se descarga al correr; no versionado)
├── pyproject.toml                    # Dependencias compartidas (Poetry)
├── poetry.lock
└── README.md
```

## Requisitos

- Python ≥ 3.10, < 3.13
- [Poetry](https://python-poetry.org/) ≥ 2.0

## Instalación

```bash
git clone <url-del-repo>
cd IntroCD
poetry install
```

El archivo `poetry.toml` ya está configurado para crear el entorno virtual dentro del proyecto (`.venv/`). Las dependencias son compartidas por ambas tareas.

## Uso

`poetry run` funciona desde cualquier carpeta del repositorio; cada tarea se corre desde su propio directorio.

### Tarea 1

```bash
cd tarea1
poetry run jupyter notebook tarea1.ipynb     # análisis interactivo
poetry run python reproducir_informe.py      # regenera las figuras (PNG) en figuras/
```

### Tarea 2

```bash
cd tarea2
poetry run jupyter notebook tarea2.ipynb     # representación de texto + modelos
poetry run python reproducir_figuras.py      # regenera las figuras (PDF) en figuras/
```

La primera vez que se ejecute la celda de carga, el dataset se descarga automáticamente desde HuggingFace Hub. El script `reproducir_figuras.py` regenera las figuras del informe en la carpeta `figuras/`.

## Dataset

Muestreo del dataset *All The News 2.1* provisto por la cátedra:

- Original: <https://huggingface.co/datasets/rjac/all-the-news-2-1-Component-one>
- Muestreo: <https://huggingface.co/datasets/tomas-gr/all-the-news-2-1-Component-one-sampled>

Contiene 30.213 artículos publicados entre 2016 y comienzos de 2020 en 26 medios de prensa anglosajones.

## Autoras

- Nicole Castroman
- Valeria Romero

## Nota sobre el uso de asistentes de IA

La redacción de los informes y el código de análisis fueron desarrollados con la asistencia de Claude (Anthropic). Las decisiones metodológicas, la interpretación de los resultados y la revisión final del contenido son responsabilidad de las autoras.
