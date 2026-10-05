# Semana 2 — Machine Learning para Regresión

> Material original: [02-regression/](../02-regression/) · Notebook de clase: [notebook.ipynb](../02-regression/notebook.ipynb)

## 2.1 Proyecto: predicción del precio de coches
> [01-car-price-intro.md](../02-regression/01-car-price-intro.md)

Toda la semana es un único proyecto: sugerir el precio de un coche a quien lo vende en
una web. Dataset de Kaggle: cada fila es un coche, cada columna una característica.
El target es **`msrp`** (*Manufacturer Suggested Retail Price*).

Plan: EDA → preparar datos → regresión lineal **implementada a mano** → evaluar con
RMSE → feature engineering → regularización → usar el modelo.

## 2.2 Preparación de datos
> [02-data-preparation.md](../02-regression/02-data-preparation.md)

Hacer los datos **consistentes**: minúsculas y `_` en lugar de espacios, tanto en los
nombres de columna como en los valores de texto.

```python
df.columns = df.columns.str.lower().str.replace(" ", "_")

strings = list(df.dtypes[df.dtypes == "object"].index)   # columnas de texto
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ", "_")
```

## 2.3 Análisis exploratorio (EDA)
> [03-eda.md](../02-regression/03-eda.md)

```python
for col in df.columns:
    print(col, df[col].unique()[:5], df[col].nunique())

sns.histplot(df.msrp, bins=50)    # distribución del target
df.isnull().sum()                 # missing values
```

- El precio tiene una **cola larga**: muchos coches baratos y unos pocos carísimos. Esto confunde al modelo.
- Solución: **transformación logarítmica** con `log1p` = `log(1 + x)` (evita `log(0) = -inf`). La distribución resultante se parece a una normal, ideal para regresión lineal.

```python
price_logs = np.log1p(df.msrp)   # para entrenar
np.expm1(pred)                   # para volver a dólares
```

## 2.4 Framework de validación
> [04-validation-framework.md](../02-regression/04-validation-framework.md)

```python
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test        # el resto, para no perder filas por redondeo

np.random.seed(2)
idx = np.arange(n)
np.random.shuffle(idx)              # ¡barajar! los datos pueden venir ordenados

df_train = df.iloc[idx[:n_train]]
df_val   = df.iloc[idx[n_train:n_train + n_val]]
df_test  = df.iloc[idx[n_train + n_val:]]

df_train = df_train.reset_index(drop=True)   # (igual con val y test)

y_train = np.log1p(df_train.msrp.values)     # (igual con val y test)
del df_train["msrp"]                         # que el target no se use como feature
```

- `iloc` selecciona por posición; barajando los índices obtenemos un split aleatorio.
- El orden `seed → arange → shuffle` determina el resultado: cámbialo y cambia el split.

## 2.5 Regresión lineal
> [05-linear-regression-simple.md](../02-regression/05-linear-regression-simple.md)

Para un coche `xᵢ` con n features:

$$g(x_i) = w_0 + \sum_{j=1}^{n} w_j \cdot x_{ij}$$

- **`w0` (bias)**: la predicción si no supiéramos nada del coche.
- **`wj` (pesos)**: cuánto aporta cada feature. Si `w` de los caballos es 0.01, cada caballo extra suma 0.01 a la predicción.
- Como el target está en escala log, la predicción se pasa a dólares con `np.expm1`.

## 2.6 Regresión lineal en forma vectorial
> [06-linear-regression-vector.md](../02-regression/06-linear-regression-vector.md)

- La suma es un **producto escalar**: `g(xᵢ) = w0 + xᵢ · w`.
- Truco de la **feature ficticia**: añadir un `1` al principio de cada `xᵢ` y meter `w0` al principio de `w` → `g(xᵢ) = xᵢ · w`.
- Para todos los coches a la vez (X con una columna de unos): **`g(X) = X · w`**, un producto matriz·vector.

## 2.7 Entrenar la regresión lineal: ecuación normal
> [07-linear-regression-training.md](../02-regression/07-linear-regression-training.md)

Queremos `Xw ≈ y`. X no es cuadrada → no tiene inversa. Multiplicamos por Xᵀ para
obtener la **matriz de Gram** `XᵀX` (cuadrada, invertible) y despejamos:

$$w = (X^T X)^{-1} X^T y$$

```python
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])     # columna de unos → bias

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]       # w0, w
```

Es la solución **aproximada** que más se acerca a y. Predicción: `w0 + X.dot(w)`.

## 2.8 Modelo baseline
> [08-baseline-model.md](../02-regression/08-baseline-model.md)

- Empezar con un modelo sencillo con pocas features numéricas: `engine_hp`, `engine_cylinders`, `highway_mpg`, `city_mpg`, `popularity`.
- La regresión lineal **no admite NaN** → `fillna(0)`. Equivale a ignorar esa feature para esa fila (`0·w = 0`), aunque no siempre tiene sentido físico (0 caballos).
- Comparar visualmente predicciones y valores reales con dos histogramas superpuestos: el baseline subestima los precios altos.

## 2.9 RMSE
> [09-rmse.md](../02-regression/09-rmse.md)

Métrica objetiva para regresión: error medio del modelo, en las unidades del target. **Menor = mejor.**

$$RMSE = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(g(x_i) - y_i)^2}$$

```python
def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)
```

Pasos: diferencia → cuadrado → media → raíz.

## 2.10 RMSE en validación
> [10-car-price-validation.md](../02-regression/10-car-price-validation.md)

- Medir en train no dice cómo funciona con datos nuevos → medir en **validation**.
- Toda la preparación va en **una función `prepare_X`**, que se aplica igual a train, val, test y a datos nuevos.

```python
def prepare_X(df):
    df_num = df[base].fillna(0)
    return df_num.values

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)        # ≈ 0.76
```

Este patrón (preparar → entrenar en train → predecir en val → métrica) se repite en todo el curso.

## 2.11 Feature engineering
> [11-feature-engineering.md](../02-regression/11-feature-engineering.md)

Crear features nuevas a partir de las existentes. Ejemplo: el **año** se convierte en **antigüedad** (`age = 2017 - year`, el año del dataset).

```python
def prepare_X(df):
    df = df.copy()              # ¡no modificar el DataFrame original!
    df["age"] = 2017 - df.year
    features = base + ["age"]
    return df[features].fillna(0).values
```

Una sola feature bien pensada baja el RMSE de **0.76 → 0.517**.

## 2.12 Variables categóricas
> [12-categorical-variables.md](../02-regression/12-categorical-variables.md)

- Las categorías (marca, tipo de combustible…) no se pueden meter tal cual: hay que convertirlas en números.
- **One-hot encoding** manual: una columna binaria 0/1 por cada valor (solo los **top-5 más frecuentes** de cada variable).
- `number_of_doors` es numérica pero en realidad es **categórica** (2, 3 o 4).

```python
categories = {}
for c in categorical_variables:
    categories[c] = list(df_train[c].value_counts().head().index)

# dentro de prepare_X
for c, values in categories.items():
    for v in values:
        df["%s_%s" % (c, v)] = (df[c] == v).astype("int")
        features.append("%s_%s" % (c, v))
```

⚠️ Al añadir todas las categóricas el RMSE **explota a 41** y los pesos llegan a ~10¹⁵. Se explica en la siguiente lección.

## 2.13 Regularización
> [13-regularization.md](../02-regression/13-regularization.md)

- **Causa**: con tantas columnas binarias, algunas son (casi) **combinación lineal** de otras → `XᵀX` es (casi) **singular** → su inversa tiene valores enormes → pesos enormes.
- Con datos reales con ruido, la inversa "existe" pero es numéricamente inestable.
- **Solución**: sumar un número pequeño `r` a la diagonal de `XᵀX`. Equivale a **Ridge regression** y mantiene los pesos controlados.

$$w = (X^T X + r \cdot I)^{-1} X^T y$$

```python
def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]
```

Con `r = 0.01` el RMSE baja a **0.461**. Cuanto mayor `r`, más pequeños los pesos.

## 2.14 Ajuste del modelo (tuning)
> [14-tuning-model.md](../02-regression/14-tuning-model.md)

`r` es un **hiperparámetro**: no se aprende de los datos, se elige probando valores y
comparando el RMSE en **validación**.

```python
for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    w0, w = train_linear_regression_reg(prepare_X(df_train), y_train, r=r)
    y_pred = w0 + prepare_X(df_val).dot(w)
    print(r, w0, rmse(y_val, y_pred))
```

- Con `r = 0` el `w0` es gigantesco (inestable); con `r` grande el modelo empeora.
- Se elige `r = 0.001`: buen RMSE y pesos estables.

## 2.15 Usar el modelo
> [15-using-model.md](../02-regression/15-using-model.md)

1. **Reentrenar con train + validation** (más datos) con el `r` elegido.
2. Evaluar **una vez** en test: si el RMSE se parece al de validación, el modelo generaliza.
3. Predecir un coche nuevo: llega como **dict** (como llegará por una API en la semana 5).

```python
df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)
X_full_train = prepare_X(df_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

rmse(y_test, w0 + prepare_X(df_test).dot(w))   # ≈ 0.460

car = df_test.iloc[20].to_dict()
X_small = prepare_X(pd.DataFrame([car]))
np.expm1(w0 + X_small.dot(w)[0])               # ≈ 41 459 $ (real: 35 000 $)
```

## 2.16 Resumen del proyecto
> [16-summary.md](../02-regression/16-summary.md)

| Paso | RMSE validación |
|------|-----------------|
| Baseline (5 numéricas) | 0.76 |
| + `age` | 0.517 |
| + puertas | 0.516 |
| + todas las categóricas | 41.45 💥 |
| + regularización | 0.461 |
| Test (modelo final) | 0.460 |

Lo aprendido: limpiar datos, EDA y `log1p`, split train/val/test, regresión lineal
desde cero (ecuación normal), RMSE, feature engineering, one-hot, regularización,
tuning y uso del modelo. La semana 3 pasa a **clasificación** con Scikit-Learn.

## 2.17 Para explorar más
> [17-explore-more.md](../02-regression/17-explore-more.md)

- ¿Qué pasa si usas el top-10 de valores en lugar del top-5 en las categóricas?
- Datasets para practicar regresión: California housing, Student Performance, repositorio UCI.

---

## Para el proyecto

- **EDA (2 puntos)**: rangos de valores, missing values, **análisis del target** (¿cola larga? → `log1p`) e importancia de features.
- Guarda toda la transformación en una función tipo `prepare_X` (en semanas posteriores será `DictVectorizer`/pipeline) → la reutilizarás en `train.py` y `predict.py`.
- Prueba al menos un modelo lineal (regresión lineal / Ridge) como **baseline** antes de los de árboles (rúbrica: "multiple models, linear and tree-based").
- El **tuning** de hiperparámetros (como `r`) se hace siempre en validación, nunca en test.
- Recuerda: el dataset de precio de coches **no** se puede usar para el proyecto.
