# Tarea 1 — Exploración y Limpieza de Datos de Noticias

Introducción a la Ciencia de Datos · Maestría en Ciencia de Datos y Aprendizaje Automático · FIng (UdelaR).

Exploración y limpieza de un subconjunto del dataset *All The News 2.1*, con foco en los cinco medios de prensa anglosajones de mayor volumen (Reuters, The New York Times, CNBC, The Hill y People).

## Estructura del proyecto

```
Tarea 1/
├── src/
│   ├── __init__.py
│   └── utils.py                  # Constantes, carga de datos, clean_text, plot_top_words, plot_wordclouds_by_source
├── notebooks/
│   ├── tarea1_ncastroman.ipynb   # Notebook de trabajo individual (Nicole)
│   └── tarea1_vromero.ipynb      # Notebook de trabajo individual (Valeria)
├── data/
│   └── sampled-data.csv          # Descargado automáticamente al correr la notebook (si no existe)
├── figuras/                      # Figuras PNG generadas por el script (no versionadas)
├── tarea1.ipynb                  # Notebook unificada — versión canónica que produce el informe
├── reproducir_informe.py         # Script que reproduce todas las figuras del informe
├── pyproject.toml                # Definición de dependencias (Poetry)
├── poetry.lock                   # Versiones bloqueadas
└── README.md
```

Las notebooks individuales (`notebooks/`) quedan como referencia del trabajo paralelo de cada integrante. La versión canónica que coincide con el informe final es `tarea1.ipynb` en la raíz.

## Requisitos

- Python ≥ 3.10, < 3.13
- [Poetry](https://python-poetry.org/) ≥ 2.0

## Instalación

1. **Instalar Poetry** (si no lo tenés):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clonar el repositorio y entrar al directorio:**

   ```bash
   git clone <url-del-repo>
   cd "Tarea 1"
   ```

3. **Instalar dependencias** con Poetry. El archivo `poetry.toml` ya está configurado para crear el entorno virtual dentro del proyecto (`.venv/`):

   ```bash
   poetry install
   ```

4. **Activar el entorno virtual** (opcional, para correr comandos sin el prefijo `poetry run`):

   ```bash
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows
   ```

## Uso

### Notebook interactiva

```bash
poetry run jupyter notebook tarea1.ipynb
```

La primera vez que se ejecute la celda de carga de datos, el dataset se descargará automáticamente desde HuggingFace Hub y se guardará como `data/sampled-data.csv`. En ejecuciones posteriores se carga directamente del archivo local.

### Reproducir el informe (script)

Para regenerar todas las figuras del informe en formato PNG:

```bash
poetry run python reproducir_informe.py
```

Las imágenes se guardan en la carpeta `figuras/`.

## Dataset

El análisis utiliza un muestreo del dataset *All The News 2.1*, provisto por la cátedra:

- Dataset original: <https://huggingface.co/datasets/rjac/all-the-news-2-1-Component-one>
- Muestreo utilizado: <https://huggingface.co/datasets/tomas-gr/all-the-news-2-1-Component-one-sampled>

Contiene 30.213 artículos publicados entre 2016 y comienzos de 2020 en 26 medios de prensa anglosajones.

## Autoras

- Nicole Castroman
- Valeria Romero

## Nota sobre el uso de asistentes de IA

La redacción del informe y el código de análisis fueron desarrollados con la asistencia de Claude (Anthropic). Las decisiones metodológicas, la interpretación de los resultados y la revisión final del contenido son responsabilidad de las autoras.
