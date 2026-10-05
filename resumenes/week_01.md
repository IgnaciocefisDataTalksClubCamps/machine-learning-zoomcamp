# Semana 1 — Introducción al Machine Learning

> Material original: [01-intro/](../01-intro/)

## 1.1 Introducción al Machine Learning
> [01-what-is-ml.md](../01-intro/01-what-is-ml.md)

ML = **extraer patrones de datos** para predecir algo sobre objetos nuevos. Ejemplo:
sugerir el precio de un coche en una web de compraventa.

- **Features**: todo lo que sabemos del objeto (año, marca, kilómetros…).
- **Target**: lo que queremos predecir (el precio).
- **Entrenar**: dar features + target a un algoritmo → sale un **modelo**, que encapsula los patrones aprendidos.
- **Predecir**: dar al modelo las features de un objeto nuevo (sin target) → sale la predicción.

Las predicciones no son exactas para cada caso, pero son correctas **de media**: igual
que un experto que ha visto muchos coches.

## 1.2 ML vs sistemas basados en reglas
> [02-ml-vs-rules.md](../01-intro/02-ml-vs-rules.md)

Ejemplo: detector de spam.

- **Reglas**: escribes a mano condiciones (`si el remitente es X → spam`, `si contiene "deposit" → spam`). El spam cambia, añades más reglas, aparecen falsos positivos… y el código se vuelve **imposible de mantener**.
- **ML** en 3 pasos:
  1. **Conseguir datos**: el botón "spam" de los usuarios da emails etiquetados.
  2. **Calcular features**: p.ej. binarias (1/0) — ¿título > 10 caracteres?, ¿contiene "deposit"?
  3. **Entrenar y usar** el modelo.

| Software tradicional | ML |
|----------------------|----|
| Datos + código (reglas) → resultado | Datos + resultado → modelo |
| | Datos nuevos + modelo → predicción |

Ideas clave:
- Empezar con reglas **es buena idea**: esas reglas se convierten después en features.
- Un clasificador devuelve una **probabilidad** (0.8 = 80 % spam). Para decidir se usa un **umbral** (p.ej. `>= 0.5` → spam).
- Entrenar también se llama **ajustar** (*fit*) el modelo.

## 1.3 Aprendizaje supervisado
> [03-supervised-ml.md](../01-intro/03-supervised-ml.md)

"Supervisado" porque **enseñamos** al modelo con ejemplos que tienen etiqueta.

- **X — matriz de features**: filas = observaciones, columnas = features.
- **y — vector target**: un valor por cada fila de X.
- **g — el modelo**: función tal que `g(X) ≈ y`. Encontrar g = **entrenar**.

Tipos según el target:

| Tipo | Salida de g | Ejemplo |
|------|-------------|---------|
| **Regresión** | Un número | Precio de un coche o una casa |
| **Clasificación binaria** | Probabilidad entre 0 y 1 | Spam / no spam |
| **Clasificación multiclase** | Una de N categorías | Gato / perro / coche |
| **Ranking** | Un score por item, se ordena | Recomendadores, Google, eBay |

La clasificación binaria es probablemente el tipo más usado en la práctica.

## 1.4 CRISP-DM
> [04-crisp-dm.md](../01-intro/04-crisp-dm.md)

Metodología de los 90 para organizar proyectos de ML (*Cross-Industry Standard Process for Data Mining*). Sigue vigente.

1. **Business understanding**: ¿cuál es el problema y cuánto importa? **¿Hace falta ML?** (a veces basta una regla). El objetivo debe ser **medible** ("reducir el spam un 50 %").
2. **Data understanding**: ¿hay datos?, ¿son fiables? (usuarios que marcan mal), ¿hay suficientes? Puede obligar a volver al paso 1.
3. **Data preparation**: extraer features, limpiar ruido, construir **pipelines**, convertir a formato tabla (X, y).
4. **Modeling**: probar varios modelos y elegir el mejor. A menudo se vuelve al paso 3 a mejorar features.
5. **Evaluation**: ¿se ha cumplido el objetivo de negocio? Si no, retrospectiva: iterar o abandonar.
6. **Deployment**: llevar a producción. Hoy suele ir junto con la evaluación (**online evaluation**: probar con un 5 % de usuarios y luego al resto). Aquí importan monitorización, mantenibilidad y fiabilidad.

**Iterar**: empezar simple, pasar rápido por todos los pasos, aprender del feedback, mejorar.

## 1.5 Selección de modelos
> [05-model-selection.md](../01-intro/05-model-selection.md)

La lección más importante de la semana.

- **Validación** = simular datos futuros: apartar ~20 % de los datos, entrenar con el resto y medir en esa parte (p.ej. *accuracy*).
- **Problema de las comparaciones múltiples**: si comparas muchos modelos en el mismo set de validación, alguno puede ganar **por suerte** (la moneda que acierta el 100 %).
- **Solución**: tres conjuntos sin solapamiento, típicamente **60 / 20 / 20**: train / validation / test.

Receta:
1. Dividir en train, validation y test.
2. Entrenar con **train**.
3. Evaluar en **validation**.
4. Repetir 2–3 con todos los modelos.
5. Elegir el mejor.
6. Aplicarlo al **test** y comprobar que el resultado es parecido al de validación.

**Reutilizar validación**: tras elegir el modelo, reentrenarlo con **train + validation**
(más datos) y comprobarlo en test.

## 1.6 Preparar el entorno
> [06-environment.md](../01-intro/06-environment.md)

- Python 3.11 + `jupyter numpy pandas scikit-learn seaborn` (más adelante: XGBoost y TensorFlow).
- Opción recomendada: **GitHub Codespaces** (VS Code remoto; el puerto 8888 de Jupyter se reenvía automáticamente).
- Alternativas: conda (`conda create -n ml-zoomcamp python=3.11`), Ubuntu/WSL, AWS/GCP, Colab/Kaggle.
- Los notebooks por sí solos no bastan: para los módulos de despliegue hace falta terminal con **Docker**.
- En este repo: `.venv` + [requirements.txt](../requirements.txt).

## 1.7 Introducción a NumPy
> [07-numpy.md](../01-intro/07-numpy.md)

```python
import numpy as np

# Crear arrays
np.zeros(10); np.ones(10); np.full(10, 2.5)
np.array([1, 2, 3])
np.arange(3, 10)            # [3..9], el final es exclusivo
np.linspace(0, 100, 11)     # 11 valores equiespaciados entre 0 y 100
np.zeros((5, 2))            # 5 filas x 2 columnas

# Indexado 2D
n[0, 1]      # fila 0, columna 1
n[2]         # fila 2 entera
n[:, 1]      # columna 1 entera

# Aleatorios: fijar la semilla para que sea reproducible
np.random.seed(2)
np.random.rand(5, 2)                            # uniforme [0, 1)
np.random.randn(5, 2)                           # normal estándar
np.random.randint(low=0, high=100, size=(5, 2))

# Elemento a elemento (sin bucles)
a + 1; a * 2; a + b; (10 + a * 2) ** 2 / 100

# Comparaciones → arrays booleanos, que sirven para filtrar
a >= 2
a[a > b]

# Resumen → un solo número
a.min(); a.max(); a.sum(); a.mean(); a.std()
```

## 1.8 Repaso de álgebra lineal
> [08-linear-algebra.md](../01-intro/08-linear-algebra.md)

Todo esto se usa en la semana 2 para deducir la regresión lineal.

| Operación | NumPy | Resultado |
|-----------|-------|-----------|
| Escalar · vector, vector + vector | `2 * u`, `u + v` | Elemento a elemento |
| **Producto escalar** (vector · vector) | `u.dot(v)` | Un número: `Σ uᵢ·vᵢ`. Mismo tamaño obligatorio |
| Matriz · vector | `U.dot(v)` | Vector: producto escalar de cada fila de U con v. `U.shape[1] == len(v)` |
| Matriz · matriz | `U.dot(V)` | Matriz: U por cada columna de V. `U.shape[1] == V.shape[0]` |
| Transpuesta | `X.T` | Filas ↔ columnas |
| **Identidad** | `np.eye(3)` | 1 en la diagonal; `A.dot(I) == A` (el "1" de las matrices) |
| **Inversa** | `np.linalg.inv(A)` | `A⁻¹·A = I`. Solo para matrices **cuadradas** (y no singulares) |

⚠️ En NumPy `u * v` **no** es el producto escalar: es multiplicación elemento a elemento.

## 1.9 Introducción a Pandas
> [09-pandas.md](../01-intro/09-pandas.md)

- **DataFrame** = tabla; **Series** = una columna.

```python
import pandas as pd

df = pd.read_csv("archivo.csv")       # o pd.DataFrame(data, columns=columns)
df.head(n=2); df.shape; len(df); df.dtypes

# Columnas
df.Make; df["Engine HP"]              # → Series
df[["Make", "Model", "MSRP"]]         # → DataFrame
df["id"] = [1, 2, 3, 4, 5]; del df["id"]

# Índice
df.loc[1]                             # por etiqueta
df.iloc[[1, 2, 4]]                    # por posición
df = df.reset_index(drop=True)

# Elemento a elemento y filtrado
df["Engine HP"] * 2
df[(df.Make == "Nissan") & (df.Year >= 2015)]

# Strings (vía .str)
df["Vehicle_Style"].str.replace(" ", "_").str.lower()

# Resúmenes
df.MSRP.mean(); df.MSRP.describe(); df.describe().round(2)
df.Make.nunique(); df.Make.value_counts()

# Missing values
df.isnull().sum()

# Agrupar (GROUP BY de SQL)
df.groupby("Transmission Type").MSRP.max()

# Salir de Pandas
df.MSRP.values                        # → array de NumPy
df.to_dict(orient="records")          # → lista de dicts
```

## 1.10 Resumen de la semana
> [10-summary.md](../01-intro/10-summary.md)

- ML aprende patrones de **features → target** y produce un **modelo**.
- Sustituye a las reglas cuando estas se vuelven inmanejables.
- Supervisado: `g(X) ≈ y`; regresión, clasificación (binaria/multiclase) y ranking.
- CRISP-DM organiza el proyecto; se itera empezando simple.
- Se eligen modelos con **train / validation / test**.
- Herramientas: NumPy, álgebra lineal y Pandas — la base de la semana 2.

---

## Para el proyecto

- **Problem description** (rúbrica): explica el problema como en el paso 1 de CRISP-DM — qué se predice, para quién y cómo se usaría el modelo.
- Elige el tipo de problema (regresión / clasificación) según el target.
- **Siempre** monta el esquema train/val/test y no toques test hasta el final.
- Fija la semilla (`np.random.seed`, `random_state=`) para que sea **reproducible** (también es criterio de la rúbrica).
