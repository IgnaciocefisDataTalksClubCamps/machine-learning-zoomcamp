# Semana 3 — Machine Learning para Clasificación

> Material original: [03-classification/](../03-classification/) · Notebooks: [notebook.ipynb](../03-classification/notebook.ipynb), [notebook-scaling-ohe.ipynb](../03-classification/notebook-scaling-ohe.ipynb)

## 3.1 Proyecto: predicción de churn
> [01-churn-project.md](../03-classification/01-churn-project.md)

- **Churn** = un cliente deja de usar el servicio. Una telco quiere detectar a los que se van a ir **antes** de que se vayan y retenerlos (p.ej. con un email con descuento).
- Es **clasificación binaria**: `g(xᵢ) ≈ yᵢ`, con `y ∈ {0, 1}`.
  - **1 = clase positiva** (el cliente se fue).
  - **0 = clase negativa** (se quedó).
- El modelo devuelve un **score entre 0 y 1**: la probabilidad de churn.
- Se entrena con datos históricos (clientes del mes pasado, de los que ya sabemos si se fueron) y se aplica a los clientes actuales.
- Dataset: **Telco Customer Churn** (Kaggle). ~7 000 clientes, con datos demográficos, servicios contratados, tipo de contrato, método de pago, cargos y `tenure` (meses de permanencia).

## 3.2 Preparación de datos
> [02-data-preparation.md](../03-classification/02-data-preparation.md)

```python
df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical_columns = list(df.dtypes[df.dtypes == "object"].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(" ", "_")

# totalcharges se leyó como texto: los NaN venían como " " (ahora "_")
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")   # lo no numérico → NaN
df.totalcharges = df.totalcharges.fillna(0)    # clientes recién llegados: aún no han pagado

# target a 0/1
df.churn = (df.churn == "yes").astype(int)
```

- **Revisa siempre `df.dtypes`**: una columna numérica leída como `object` indica valores raros.
- `errors="coerce"` convierte en NaN lo que no se puede parsear.
- `df.head().T` (transpuesta) ayuda a ver todas las columnas cuando son muchas.

## 3.3 Framework de validación con Scikit-Learn
> [03-validation.md](../03-classification/03-validation.md)

```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
# 0.25 porque el 25 % del 80 % = 20 % del total → 60/20/20

df_train = df_train.reset_index(drop=True)    # (igual con val y test)

y_train = df_train.churn.values               # (igual con val y test)
del df_train["churn"]                         # que el target no se cuele como feature
```

- `train_test_split` sustituye al split manual de la semana 2.
- `random_state` fija la semilla: el split es reproducible.
- `df_full_train` **conserva** `churn` porque se usa en el EDA.

## 3.4 EDA
> [04-eda.md](../03-classification/04-eda.md)

```python
df_full_train.isnull().sum()                        # missing values
df_full_train.churn.value_counts(normalize=True)    # 73 % se quedan, 27 % se van
df_full_train.churn.mean()                          # churn rate global ≈ 0.27
df_full_train[categorical].nunique()                # nº de categorías por variable
```

- **Churn rate** = proporción de unos. Truco: **la media de una columna binaria es la fracción de unos**.
- Separar las variables:
  - **Numéricas**: `tenure`, `monthlycharges`, `totalcharges`.
  - **Categóricas**: las otras 16. `seniorcitizen` es 0/1 pero es categórica.
  - `customerid` no sirve como feature: es un identificador.
- El nº de categorías determina cuántas columnas salen tras el one-hot.

## 3.5 Importancia de features: churn rate y risk ratio
> [05-risk.md](../03-classification/05-risk.md)

Para cada grupo de una variable categórica se compara su churn rate con el global:

- **Diferencia** = `media_grupo - global`. Si es > 0, el grupo se va más que la media.
- **Risk ratio** = `media_grupo / global`.
  - `= 1`: riesgo igual a la media.
  - `> 1`: más riesgo (1.22 = un 22 % más).
  - `< 1`: menos riesgo.

```python
global_churn = df_full_train.churn.mean()

for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(["mean", "count"])
    df_group["diff"] = df_group["mean"] - global_churn
    df_group["risk"] = df_group["mean"] / global_churn
    display(df_group)                 # from IPython.display import display
```

Hallazgos:
- Las variables más importantes son `contract`, `internetservice` y `paymentmethod`. Los clientes con contrato mes a mes tienen riesgo ×1.6, los de fibra ×1.57 y los que pagan con cheque electrónico ×1.69.
- `gender` y `phoneservice` apenas influyen.

Limitación: no permite comparar **variables enteras** entre sí → información mutua.

## 3.6 Importancia de features: información mutua
> [06-mutual-info.md](../03-classification/06-mutual-info.md)

- Mide **cuánto aprendemos de una variable conociendo otra**. Es 0 si no aporta nada y más alta cuanto más informa.
- Da **un número por variable**, así que permite ordenarlas. Se usa con **categóricas**.
- Es simétrica: el orden de los argumentos da igual.

```python
from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
```

Ranking: `contract` (0.098) > `onlinesecurity` > `techsupport` > … > `gender` (≈ 0).

## 3.7 Importancia de features: correlación
> [07-correlation.md](../03-classification/07-correlation.md)

Para **numéricas** se usa el **coeficiente de correlación de Pearson** (r ∈ [-1, 1]):

| \|r\| | Fuerza |
|-------|--------|
| 0 – 0.2 | Baja |
| 0.2 – 0.5 | Media |
| 0.5 – 1 | Fuerte |

- **Signo** = dirección. Si r > 0, al subir la feature sube el churn.
- **Valor absoluto** = importancia.

```python
df_full_train[numerical].corrwith(df_full_train.churn)
# tenure -0.35 | monthlycharges 0.20 | totalcharges -0.20
```

- `tenure` es la numérica más importante: cuanto más tiempo lleva el cliente, menos se va. Los de ≤ 2 meses se van al 60 % y los de > 12 meses al 18 %.
- `monthlycharges`: cuanto más paga, más se va.
- Se puede comprobar agrupando en rangos y calculando el churn rate de cada uno.

## 3.8 One-hot encoding
> [08-ohe.md](../03-classification/08-ohe.md)

Cada variable categórica se convierte en **una columna binaria por categoría**. Solo una vale 1 (está "hot").

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient="records")   # lista de dicts
X_train = dv.fit_transform(train_dict)       # fit: aprende las categorías; transform: matriz

val_dict = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dict)               # ⚠️ solo transform en val/test

dv.get_feature_names_out()                   # 45 columnas: "contract=two_year", ..., "tenure"
```

- `DictVectorizer` codifica las categóricas y **deja pasar las numéricas** tal cual.
- **`fit` solo con train.** Si lo haces también con val, las columnas pueden salir en otro orden o aparecer columnas nuevas.
- `sparse=False` devuelve un array normal. Por defecto devuelve una matriz *sparse*, que solo guarda los valores distintos de cero.
- Alternativa: `OneHotEncoder` (trabaja con DataFrames), junto con `StandardScaler` para escalar las numéricas. Ver [notebook-scaling-ohe.ipynb](../03-classification/notebook-scaling-ohe.ipynb).

## 3.9 Regresión logística
> [09-logistic-regression.md](../03-classification/09-logistic-regression.md)

Es la **regresión lineal + una sigmoide** al final, que aplasta el score a (0, 1) para que sea una probabilidad:

$$g(x_i) = \sigma(w_0 + w^T x_i) \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

- σ(0) = 0.5; z ≫ 0 → ~1; z ≪ 0 → ~0. Tiene forma de S.
- Las dos son **modelos lineales**: dentro hay una combinación lineal de las features.

## 3.10 Entrenar la regresión logística con Scikit-Learn
> [10-training-log-reg.md](../03-classification/10-training-log-reg.md)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver="lbfgs")
model.fit(X_train, y_train)

model.intercept_[0]       # w0 (bias)
model.coef_[0]            # pesos, uno por columna del DictVectorizer

y_pred = model.predict_proba(X_val)[:, 1]   # probabilidad de la clase 1
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()            # accuracy ≈ 0.80
```

- `predict` da predicciones **duras** (0/1). `predict_proba` da predicciones **blandas** (probabilidades). Usa la segunda para poder elegir tu propio umbral.
- `predict_proba` devuelve 2 columnas, P(0) y P(1), que suman 1 → `[:, 1]`.
- **Accuracy** = fracción de aciertos.

## 3.11 Interpretación del modelo
> [11-log-reg-interpretation.md](../03-classification/11-log-reg-interpretation.md)

```python
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))
```

- Un **peso positivo** sube la probabilidad de churn y uno negativo la baja. Ejemplos: `contract=month-to-month` +0.97, `contract=two_year` -0.95, `tenure` -0.036 por mes.
- Con one-hot, de todas las columnas de una variable **solo se usa el peso de la categoría del cliente**, porque las demás valen 0.
- Si hay muchas features correlacionadas, los pesos individuales son poco fiables. Un modelo pequeño (`contract`, `tenure`, `monthlycharges`) se interpreta mejor.
- Cálculo a mano: `score = w0 + Σ wⱼ·xⱼ` y `prob = sigmoid(score)`. Score > 0 → prob > 50 %.

## 3.12 Usar el modelo
> [12-using-log-reg.md](../03-classification/12-using-log-reg.md)

```python
# Modelo final: train + val, con un DictVectorizer nuevo
dicts_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
model = LogisticRegression(solver="lbfgs").fit(X_full_train, df_full_train.churn.values)

# Test
X_test = dv.transform(df_test[categorical + numerical].to_dict(orient="records"))
((model.predict_proba(X_test)[:, 1] >= 0.5) == y_test).mean()   # ≈ 0.815 (val: 0.803)

# Un cliente (llega como dict, como en una API)
model.predict_proba(dv.transform([customer]))[0, 1]             # ≈ 0.60 → churn
```

- Si las métricas de test y validación se parecen, el modelo generaliza bien. Una diferencia grande indica **overfitting**.
- **El modelo = DictVectorizer + LogisticRegression**: van siempre juntos y hay que guardarlos ambos (semana 5).

## 3.13 Resumen
> [13-summary.md](../03-classification/13-summary.md)

- Importancia de features: **risk ratio**, **información mutua** (categóricas) y **correlación** (numéricas).
- One-hot encoding con **`DictVectorizer`**.
- **Regresión logística**: un modelo lineal cuya salida es una probabilidad. Sus pesos se interpretan como en la regresión lineal.
- El mismo esquema vale para cualquier problema sí/no: spam, impagos, *lead scoring*…

## 3.14 Para explorar más
> [14-explore-more.md](../03-classification/14-explore-more.md)

- Quitar las features menos útiles y comprobar si el modelo mejora.
- Rehacer el proyecto de la semana 2 con `train_test_split`, `LinearRegression` y `Ridge`, buscando el mejor `alpha`.
- Probar `OneHotEncoder` y `StandardScaler`: escalar ayuda a solvers iterativos como `lbfgs`.
- Datasets: Lead scoring (Kaggle), Default of credit card clients (UCI).

---

## Para el proyecto

- **EDA extensa (2 puntos)**: análisis del target (¿está desbalanceado?) e **importancia de features**. Usa risk ratio o información mutua para las categóricas y correlación para las numéricas.
- `DictVectorizer` + `to_dict(orient="records")` es la forma más cómoda de preparar features. Además encaja con `predict.py`, que recibirá un JSON/dict.
- Ten siempre listas dos funciones, **`train(df, y)` → `(dv, model)`** y **`predict(df, dv, model)`**, para reutilizarlas en `train.py`.
- La regresión logística es el **modelo lineal** que pide la rúbrica para clasificación ("linear and tree-based").
- El dataset de churn **no** se puede usar para el proyecto.
