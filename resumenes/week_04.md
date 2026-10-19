# Semana 4 — Métricas de evaluación para clasificación

> Material original: [04-evaluation/](../04-evaluation/) · Notebook: [notebook.ipynb](../04-evaluation/notebook.ipynb)

## 4.1 Visión general
> [01-overview.md](../04-evaluation/01-overview.md)

- **Métrica** = función que compara las predicciones con los valores reales y devuelve **un número** que dice lo buenas que son.
- Se sigue con el modelo de churn de la semana 3 (regresión logística, accuracy 0.80 en validación).
- La pregunta de la semana: **¿es bueno un 80 %?**
- Temario: accuracy y modelo dummy, tabla de confusión, precision y recall, curvas ROC, AUC y validación cruzada.

## 4.2 Accuracy y modelo dummy
> [02-accuracy.md](../04-evaluation/02-accuracy.md)

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_val, y_pred >= 0.5)     # 0.803

# Probar varios umbrales
thresholds = np.linspace(0, 1, 21)
for t in thresholds:
    print("%.2f %.3f" % (t, accuracy_score(y_val, y_pred >= t)))
```

- **Accuracy** = aciertos / total.
- El umbral 0.5 no tiene por qué ser el mejor: **pruébalos todos**. Aquí 0.5 resulta ser el mejor, pero en otros problemas no.
- **Modelo dummy**: predecir "nadie se va" (umbral = 1.0) da **73 %** de accuracy sin mirar los datos (`1 - y_val.mean()`).
- Por tanto el modelo solo mejora 7 puntos sobre el dummy.
- **Desbalanceo de clases**: con 73 % / 27 %, predecir siempre la clase mayoritaria ya da una accuracy alta. **La accuracy engaña con clases desbalanceadas.**

## 4.3 Tabla de confusión
> [03-confusion-table.md](../04-evaluation/03-confusion-table.md)

|                    | Predicho negativo | Predicho positivo |
|--------------------|-------------------|-------------------|
| **Real negativo**  | TN = 922          | FP = 101          |
| **Real positivo**  | FN = 176          | TP = 210          |

- **Positivo/negativo** = lo que predijimos. **True/False** = si acertamos.
- **FP**: enviamos un descuento a alguien que no se iba a ir (dinero perdido).
- **FN**: el cliente se fue y no lo detectamos (suele ser lo más caro).

```python
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

confusion_matrix = np.array([[tn, fp], [fn, tp]])
(confusion_matrix / confusion_matrix.sum()).round(2)    # en proporciones
```

- Accuracy = `(TP + TN) / total`. La tabla además **separa el 20 % de error** en FP (7 %) y FN (12 %), que tienen costes de negocio muy distintos.
- En sklearn: `sklearn.metrics.confusion_matrix(y_val, y_pred >= 0.5)`.

## 4.4 Precision y recall
> [04-precision-recall.md](../04-evaluation/04-precision-recall.md)

Ambas miran la **clase positiva**, la que nos interesa.

| Métrica | Fórmula | Pregunta | Churn |
|---------|---------|----------|-------|
| **Precision** | `TP / (TP + FP)` | De los que marqué como positivos, ¿cuántos lo eran? | 0.67: 1 de cada 3 emails va a quien no se iba |
| **Recall** | `TP / (TP + FN)` | De los positivos reales, ¿cuántos encontré? | 0.54: se nos escapa casi la mitad de los que se van |

- Muestran lo que la accuracy esconde: con un 80 % de accuracy, el modelo pierde el 46 % de los churners.
- Úsalas cuando hay **desbalanceo**: churn, fraude, diagnóstico médico…
- Hay un **trade-off**: subir el umbral aumenta la precision y baja el recall.
- **F1** = `2·P·R / (P + R)`: combina las dos en un número (ver 4.9).
- En sklearn: `precision_score`, `recall_score` y `f1_score` de `sklearn.metrics`.

## 4.5 Curvas ROC
> [05-roc.md](../04-evaluation/05-roc.md)

Evalúan el modelo **en todos los umbrales a la vez**.

| Tasa | Fórmula | Fila de la tabla | Objetivo |
|------|---------|------------------|----------|
| **TPR** (= recall) | `TP / (TP + FN)` | Positivos reales | Maximizar |
| **FPR** | `FP / (FP + TN)` | Negativos reales | Minimizar |

Modelos de referencia:
- **Aleatorio** (`np.random.uniform`): TPR = FPR en todos los umbrales. En la curva ROC es la **diagonal**.
- **Ideal**: ordena perfectamente a todos los positivos por encima de los negativos. Llega a TPR = 1 con FPR = 0, la **esquina superior izquierda**.

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="Model")
plt.plot([0, 1], [0, 1], label="Random", linestyle="--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
```

- La curva ROC pinta **TPR frente a FPR**. Cada punto es un umbral.
  - Abajo a la izquierda: umbral alto (casi nadie marcado).
  - Arriba a la derecha: umbral bajo (todos marcados).
- **Cuanto más se acerca la curva a la esquina superior izquierda, mejor el modelo.**
- Funciona bien con clases desbalanceadas.

## 4.6 ROC AUC
> [06-auc.md](../04-evaluation/06-auc.md)

**AUC** = área bajo la curva ROC. Resume la curva en un número.

| AUC | Significado |
|-----|-------------|
| 0.5 | Aleatorio |
| 0.6 | Malo |
| 0.8 | Bueno |
| 0.9 | Muy bueno |
| 1.0 | Ideal |

```python
from sklearn.metrics import roc_auc_score, auc

roc_auc_score(y_val, y_pred)     # 0.844 — atajo directo
auc(fpr, tpr)                    # mismo valor; auc() vale para cualquier curva
```

**Interpretación probabilística**: el AUC es la **probabilidad de que un positivo
elegido al azar tenga mayor score que un negativo elegido al azar**. Mide lo bien que
el modelo **ordena/separa** las clases.

```python
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]
pos_ind = np.random.randint(0, len(pos), size=50000)
neg_ind = np.random.randint(0, len(neg), size=50000)
(pos[pos_ind] > neg[neg_ind]).mean()     # ≈ AUC
```

No depende del umbral ni del desbalanceo, y por eso es la métrica más usada en clasificación binaria.

## 4.7 Validación cruzada (K-Fold)
> [07-cross-validation.md](../04-evaluation/07-cross-validation.md)

Evaluar el mismo modelo en **varios splits** para obtener una **media y una desviación**
en lugar de un único número. Se usa para **ajustar hiperparámetros**.

1. Dividir `df_full_train` (train + val) en k partes (*folds*).
2. Entrenar con k-1 folds y evaluar en el que queda.
3. Repetir k veces, de modo que cada fold sea validación una vez.
4. Calcular la media ± std.

```python
from sklearn.model_selection import KFold

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient="records")
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]

for C in [0.001, 0.01, 0.1, 0.5, 1, 5, 10]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in kfold.split(df_full_train):
        df_tr = df_full_train.iloc[train_idx]
        df_va = df_full_train.iloc[val_idx]
        dv, model = train(df_tr, df_tr.churn.values, C=C)
        y_pred = predict(df_va, dv, model)
        scores.append(roc_auc_score(df_va.churn.values, y_pred))
    print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))
```

- **`C`** es la regularización de `LogisticRegression` y es la **inversa** de la `r` de la semana 2: **C pequeño = más regularización**.
- La **media** indica el rendimiento y la **std** la estabilidad.
- Una vez elegido el parámetro, se entrena con **todo `df_full_train`** y se evalúa **una vez** en test (AUC 0.857).
- ¿Cuándo usar cada método?
  - **Dataset grande**: basta con un split de validación, que es más rápido.
  - **Dataset pequeño**, o si quieres saber lo estable que es el modelo: K-Fold.
- `tqdm` muestra una barra de progreso en los bucles largos.

## 4.8 Resumen
> [08-summary.md](../04-evaluation/08-summary.md)

| Métrica | Cuándo |
|---------|--------|
| Accuracy | Fácil de entender, pero **engaña con desbalanceo** (el dummy ya da 73 %) |
| Precision / Recall | Desbalanceo; cuando importa el tipo de error (FP vs FN) |
| Curva ROC | Ver el modelo en todos los umbrales |
| ROC AUC | Un número independiente del umbral; 0.5 = aleatorio, 1 = ideal |
| K-Fold CV | Estimación fiable (media ± std) y tuning de parámetros |

En la semana 5 este modelo de churn se **despliega** como servicio web.

## 4.9 Para explorar más
> [09-explore-more.md](../04-evaluation/09-explore-more.md)

- Calcular precision y recall del modelo dummy (siempre `False`). La precision no está definida, porque no hay positivos predichos.
- **F1 score** = `2PR / (P + R)`.
- **Curva precision-recall**: P frente a R en todos los umbrales. El área bajo ella (AUC-PR) también es útil con mucho desbalanceo.
- Calcular estas métricas con los datasets de la semana anterior.

---

## Para el proyecto

- **Elige la métrica según el problema** y justifícala en el README. Con clases desbalanceadas no uses solo accuracy: usa **ROC AUC**, F1 o precision/recall.
- Compara siempre con un **baseline dummy** (clase mayoritaria) para demostrar que el modelo aporta.
- **"Trained multiple models and tuned their parameters" (3 puntos)**: ajusta hiperparámetros (`C`, y más adelante `max_depth`, etc.) con **validación** o **K-Fold**, nunca con test.
- Estructura `train()` / `predict()` como en 4.7: pasa casi tal cual a `train.py`.
- Si el umbral importa para el negocio (coste de FP frente a FN), explica cuál eliges y por qué.
