import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# leer archivo csv con los datos del banco
df = pd.read_csv("bank.csv")

# buscar columna objetivo
posibles_objetivos = ["y", "deposit", "subscribed"]
objetivo = None
for c in posibles_objetivos:
    if c in df.columns:
        objetivo = c
        break
if objetivo is None:
    raise ValueError(f"No se encontro la columna objetivo (busque {posibles_objetivos}). Columnas: {list(df.columns)}")

# convertir la columna objetivo a valores numericos
df[objetivo] = (
    df[objetivo].astype(str).str.strip().str.lower()
      .replace({"yes": 1, "no": 0, "y": 1, "n": 0})
)
try:
    df[objetivo] = df[objetivo].astype(int)
except Exception:
    pass

# convertir variables categoricas en dummies
cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != objetivo]
if len(cat_cols) > 0:
    df = pd.concat([df.drop(columns=cat_cols),
                    pd.get_dummies(df[cat_cols], dtype=int, drop_first=True)], axis=1)

# rellenar valores nulos en numericas con la media
for c in df.select_dtypes(include="number").columns:
    if df[c].isna().any():
        df[c].fillna(df[c].mean(), inplace=True)

# elegir columnas para kmeans
preferidas = [c for c in ["age","balance","duration","campaign","pdays","previous"] if c in df.columns]
if len(preferidas) >= 2:
    cols_kmeans = preferidas
else:
    cols_kmeans = [c for c in df.select_dtypes(include="number").columns if c != objetivo][:3]

# aplicar kmeans para segmentar clientes
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
df["segmento_cliente"] = kmeans.fit_predict(df[cols_kmeans])

# grafico de dispersion por segmentos
if len(cols_kmeans) >= 2:
    sb.scatterplot(data=df, x=cols_kmeans[0], y=cols_kmeans[1], hue="segmento_cliente", palette="bright")
    plt.title("K-Means - Dispersion por segmentos")
    plt.tight_layout(); plt.savefig("dispersion_kmeans.png", dpi=150); plt.close()

# grafico de conteo por segmento
sb.countplot(data=df, x="segmento_cliente")
plt.title("Cantidad de clientes por segmento")
plt.tight_layout(); plt.savefig("conteo_segmentos.png", dpi=150); plt.close()

# tasa de suscripcion promedio por segmento
df.groupby("segmento_cliente")[objetivo].mean().plot(kind="bar")
plt.title(f"Tasa de suscripcion por segmento ({objetivo})")
plt.xlabel("segmento_cliente"); plt.ylabel("promedio")
plt.tight_layout(); plt.savefig("tasa_suscripcion_por_segmento.png", dpi=150); plt.close()

# modelo de arbol de decision
X = df.drop(columns=[objetivo, "segmento_cliente"])
y = df[objetivo].astype(int)

# dividir datos en entrenamiento y prueba
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=42)

# entrenar arbol con profundidad 3
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo = clf.fit(X_ent, y_ent)

# evaluar el modelo
pred = modelo.predict(X_pru)
matriz = confusion_matrix(y_pru, pred)
exactitud = accuracy_score(y_pru, pred)
print("Matriz de confusion:\n", matriz)
print("Exactitud:", exactitud)

# probar distintas profundidades
for p in range(1, 11):
    m = DecisionTreeClassifier(max_depth=p, random_state=42)
    m.fit(X_ent, y_ent)
    pr = m.predict(X_pru)
    print(f"max_depth={p} -> exactitud={accuracy_score(y_pru, pr):.4f}")

# grafico del arbol
plt.figure(figsize=(24, 16))
plot_tree(modelo, feature_names=X_ent.columns, class_names=["No","Si"], filled=True, label="none")
plt.title("Arbol de Decision (max_depth=3)")
plt.tight_layout(); plt.savefig("arbol_decision.png", dpi=150); plt.close()

# guardar dataset con segmentos
df.to_csv("salida_banco_segmentos.csv", index=False)
print("Listo. Archivos generados: dispersion_kmeans.png, conteo_segmentos.png, tasa_suscripcion_por_segmento.png, arbol_decision.png, salida_banco_segmentos.csv")
